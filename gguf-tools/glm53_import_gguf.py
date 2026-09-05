#!/usr/bin/env python3
"""Import published GLM Flash quantized matrices into DS4's source conventions.

This is a source-aligned import, not a reproduction of another engine's graph.
Quantized payloads are copied unchanged. Native A_log is restored from the
checkpoint instead of inverting the publisher's rounded -exp(A_log). Indexer
APE may narrow from F32 to BF16 only when every value is exactly representable.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from glm53_quantize import (
    SourceDB, align, build_plan, fail, kv_bool, kv_f32, kv_string,
    load_tokenizer_records, model_metadata, read_exact,
    read_gguf_string, read_u32, read_u64, tensor_header,
    validate_tokenizer_template,
)

LAYOUTS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84),
           12: (256, 144), 13: (256, 176), 14: (256, 210),
           16: (256, 66), 30: (1, 2)}
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
KDA_NAMES = {
    'kda_q': 'attn_q', 'kda_k': 'attn_k', 'kda_v': 'attn_v',
    'kda_output': 'attn_output', 'kda_q_conv': 'ssm_conv1d_q',
    'kda_k_conv': 'ssm_conv1d_k', 'kda_v_conv': 'ssm_conv1d_v',
    'kda_f_a': 'ssm_f_a', 'kda_f_b': 'ssm_f_b', 'kda_g_a': 'ssm_g_a',
    'kda_g_b': 'ssm_g_b', 'kda_beta': 'ssm_beta', 'kda_o_norm': 'ssm_norm',
}


def file_version(stat):
    return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_nlink,
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class OpenShard:
    def __init__(self, path):
        self.requested_path = Path(path).absolute()
        self.fp = self.requested_path.open('rb')
        self.verified_sha256 = None
        try:
            opened = os.fstat(self.fp.fileno())
            self.path = self.requested_path.resolve(strict=True)
            named = os.stat(self.requested_path)
            resolved = os.stat(self.path)
            identity = (opened.st_dev, opened.st_ino)
            if (identity != (named.st_dev, named.st_ino) or
                    identity != (resolved.st_dev, resolved.st_ino)):
                fail(f'{self.requested_path}: input path changed while opening')
            self._version = file_version(opened)
            self.metadata, self.tensors = read_shard(self)
            self.assert_unchanged()
        except BaseException:
            self.fp.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fp.close()

    def assert_unchanged(self):
        try:
            opened = os.fstat(self.fp.fileno())
            named = os.stat(self.requested_path)
        except OSError:
            fail(f'{self.requested_path}: input changed during import')
        if (file_version(opened) != self._version or
                (named.st_dev, named.st_ino) != self._version[:2]):
            fail(f'{self.requested_path}: input changed during import')

    def read_at(self, offset, length, label):
        self.assert_unchanged()
        self.fp.seek(offset)
        data = read_exact(self.fp, length, label)
        self.assert_unchanged()
        return data

    def copy_range(self, out, offset, length, label, digest):
        self.assert_unchanged()
        self.fp.seek(offset)
        remaining = length
        while remaining:
            block = read_exact(self.fp, min(remaining, 8 << 20), label)
            out.write(block)
            digest.update(block)
            remaining -= len(block)
        self.assert_unchanged()

    def sha256(self):
        self.assert_unchanged()
        digest = hashlib.sha256()
        self.fp.seek(0)
        while block := self.fp.read(8 << 20):
            digest.update(block)
        self.assert_unchanged()
        return digest.hexdigest()

    def sha256_range(self, offset, length, label):
        digest = hashlib.sha256()
        self.assert_unchanged()
        self.fp.seek(offset)
        remaining = length
        while remaining:
            block = read_exact(self.fp, min(remaining, 8 << 20), label)
            digest.update(block)
            remaining -= len(block)
        self.assert_unchanged()
        return digest.hexdigest()

    def revalidate(self):
        if self.verified_sha256 is None:
            fail(f'{self.path}: input was not checksum verified')
        if self.sha256() != self.verified_sha256:
            fail(f'{self.path}: input changed during import')


def tensor_bytes(qtype, shape):
    if qtype not in LAYOUTS or not shape or any(d <= 0 for d in shape):
        fail(f'unsupported tensor layout: type={qtype}, shape={shape}')
    block, size = LAYOUTS[qtype]
    count = math.prod(shape)
    if shape[0] % block or count > (1 << 63) - 1:
        fail(f'invalid tensor block layout: {shape}')
    return count // block * size


def metadata_value(fp, kind, depth=0):
    if depth > 2:
        fail('nested metadata arrays are unsupported')
    if kind == 8:
        return read_gguf_string(fp, 'metadata value')
    if kind == 9:
        subtype, count = read_u32(fp, 'array type'), read_u64(fp, 'array count')
        if count > 1 << 22:
            fail('metadata array is too large')
        return [metadata_value(fp, subtype, depth + 1) for _ in range(count)]
    if kind not in SCALARS:
        fail(f'unsupported metadata type {kind}')
    fmt = '<' + SCALARS[kind]
    return struct.unpack(fmt, read_exact(fp, struct.calcsize(fmt), 'metadata value'))[0]


def read_shard(shard):
    path, fp = shard.path, shard.fp
    size = shard._version[4]
    fp.seek(0)
    if read_exact(fp, 4, 'magic') != b'GGUF' or read_u32(fp, 'version') != 3:
        fail(f'{path}: expected GGUF v3')
    nt, nk = read_u64(fp, 'tensor count'), read_u64(fp, 'metadata count')
    if nt > min(size // 32, 100000) or nk > min(size // 13, 100000):
        fail(f'{path}: implausible header counts')
    metadata = {}
    for _ in range(nk):
        key = read_gguf_string(fp, 'metadata key')
        if key in metadata:
            fail(f'{path}: duplicate metadata key {key}')
        metadata[key] = metadata_value(fp, read_u32(fp, 'metadata type'))
        if fp.tell() > 64 << 20:
            fail(f'{path}: header is too large')
    alignment = metadata.get('general.alignment', 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment > 65536 or alignment & (alignment - 1):
        fail(f'{path}: invalid alignment')
    tensors = []
    names = set()
    for _ in range(nt):
        name = read_gguf_string(fp, 'tensor name')
        rank = read_u32(fp, 'tensor rank')
        if not 1 <= rank <= 4 or name in names:
            fail(f'{path}: invalid or duplicate tensor {name}')
        names.add(name)
        shape = tuple(read_u64(fp, 'dimension') for _ in range(rank))
        qtype, offset = read_u32(fp, 'tensor type'), read_u64(fp, 'tensor offset')
        nbytes = tensor_bytes(qtype, shape)
        if offset % alignment:
            fail(f'{path}: unaligned tensor {name}')
        tensors.append(dict(name=name, shape=shape, qtype=qtype, offset=offset, nbytes=nbytes))
    data_offset = align(fp.tell(), alignment)
    spans = []
    for tensor in tensors:
        tensor.update(shard=shard, abs_offset=data_offset + tensor['offset'])
        end = tensor['abs_offset'] + tensor['nbytes']
        if end > size:
            fail(f'{path}: truncated tensor {tensor["name"]}')
        spans.append((tensor['abs_offset'], end))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        fail(f'{path}: overlapping tensors')
    return metadata, tensors


def published_name(name):
    for old, new in KDA_NAMES.items():
        name = name.replace('.' + old + '.', '.' + new + '.')
    return (name.replace('.kda_a_log.weight', '.ssm_a')
            .replace('.kda_dt_bias.weight', '.ssm_dt.bias')
            .replace('.indexer.pool_ape.weight', '.indexer_compressor_ape.weight')
            .replace('.indexer.pool_gate.weight', '.indexer_compressor_gate.weight'))


def payload(tensor):
    return tensor['shard'].read_at(tensor['abs_offset'], tensor['nbytes'], tensor['name'])


def exact_bf16(raw):
    import numpy as np
    bits = np.frombuffer(raw, dtype='<u4')
    values = bits.view('<f4')
    if not np.isfinite(values).all() or np.any(bits & 0xffff):
        fail('F32 indexer tensor is not exactly representable as BF16')
    return (bits >> 16).astype('<u2').tobytes()


def native_decay(raw, source):
    import numpy as np
    published = np.frombuffer(raw, dtype='<f4')
    native = np.frombuffer(source, dtype='<f4')
    # A consistency check against the publisher's preprocessing, not an
    # assertion of identical exp implementations or an inverse conversion.
    with np.errstate(over='ignore', invalid='ignore'):
        expected = -np.exp(native.astype(np.float64))
    if (published.shape != native.shape or not np.isfinite(native).all() or
            not np.isfinite(published).all() or not np.all(published < 0) or
            not np.allclose(published, expected, rtol=5e-7, atol=0)):
        fail('published decay does not match this checkpoint A_log')
    return source


def source_f32(raw, dtype):
    import numpy as np
    if dtype == 'BF16':
        raw = (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).tobytes()
    elif dtype != 'F32':
        fail('unquantized controls must originate as BF16 or F32')
    if not np.isfinite(np.frombuffer(raw, dtype='<f4')).all():
        fail('nonfinite source control parameter')
    return raw


def verify_file(shard, expected):
    path = shard.path
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        fail(f'{path}: invalid expected SHA256')
    shard.verified_sha256 = None
    if shard.sha256() != expected:
        fail(f'{path}: checksum mismatch')
    shard.verified_sha256 = expected
    print(f'verified input {Path(path).name}', flush=True)


def validate_metadata(meta):
    arch = meta.get('general.architecture')
    if arch not in ('glm5next', 'glm5-next'):
        fail('input is not a published GLM Flash GGUF')
    # These are graph semantics, not just dimensions inferable from tensors.
    expected = {
        'block_count': 46, 'context_length': 1048576, 'embedding_length': 4096,
        'vocab_size': 154880, 'expert_count': 288, 'expert_used_count': 8,
        'expert_group_count': 1, 'expert_group_used_count': 1, 'expert_gating_func': 2,
        'expert_weights_scale': 2.5, 'expert_weights_norm': True,
        'leading_dense_block_count': 3, 'nextn_predict_layers': 1,
        'attention.head_count': 64, 'attention.q_lora_rank': 1536,
        'attention.kv_lora_rank': 512, 'rope.dimension_count': 0,
        'attention.key_length_mla': 256, 'attention.value_length_mla': 256,
        'attention.layer_norm_rms_epsilon': 1e-5, 'attention.layer_norm_epsilon': 1e-6,
        'ssm.conv_kernel': 4, 'kda.head_dim': 128, 'kda.gate_lower_bound': -5.0,
        'attention.indexer.head_count': 32, 'attention.indexer.key_length': 128,
        'attention.indexer.top_k': 2048, 'attention.indexer.kpool': 4,
        'hyper_connection.count': 4, 'hyper_connection.sinkhorn_iterations': 20,
        'hyper_connection.epsilon': 1e-6,
        'attention.head_count_kv': [int(i % 4 == 3 or i == 45) for i in range(46)],
        'swiglu_clamp_exp': [10.0] * 46, 'swiglu_clamp_shexp': [10.0] * 46,
    }
    for key, want in expected.items():
        got = meta.get(arch + '.' + key)
        if isinstance(want, float):
            want = struct.unpack('<f', struct.pack('<f', want))[0]
        if got != want:
            fail(f'unsupported metadata {key}: {got!r}, expected {want!r}')


def write_import(output, entries, records, provenance, input_shards):
    output = Path(output)
    report_path = Path(str(output) + '.import.json')
    if output.exists() or report_path.exists():
        fail('output or import report already exists')
    input_shards = tuple(input_shards)
    for shard in input_shards:
        if shard.verified_sha256 is None:
            fail(f'{shard.path}: input was not checksum verified')
        shard.assert_unchanged()
    if any(entry['input']['shard'] not in input_shards for entry in entries):
        fail('import entry does not belong to a verified input shard')
    offset = 0
    for entry in entries:
        entry['plan'].offset = offset
        offset += align(entry['plan'].nbytes)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(entries), len(records))
    header += b''.join(records) + b''.join(tensor_header(e['plan']) for e in entries)
    header += bytes(align(len(header)) - len(header))
    if shutil.disk_usage(output.parent).free < len(header) + offset + (32 << 30):
        fail('insufficient free space for output plus 32 GiB reserve')
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.partial', dir=output.parent)
    written = []
    report_created = False
    published = False
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(header)
            for i, entry in enumerate(entries, 1):
                plan, src = entry['plan'], entry['input']
                digest = hashlib.sha256()
                if entry.get('replacement') is not None:
                    data = entry['replacement']
                    if len(data) != plan.nbytes:
                        fail(f'{plan.name}: replacement size mismatch')
                    out.write(data)
                    digest.update(data)
                else:
                    src['shard'].copy_range(out, src['abs_offset'], plan.nbytes,
                                            plan.name, digest)
                out.write(bytes(align(plan.nbytes) - plan.nbytes))
                written.append(dict(name=plan.name, source_name=src['name'], source_file=str(src['shard'].path),
                                    checkpoint_tensor=plan.source,
                                    source_offset=src['abs_offset'], type=plan.qtype,
                                    shape=plan.shape, nbytes=plan.nbytes, sha256=digest.hexdigest(),
                                    action=entry.get('action', 'byte-copy')))
                if i % 50 == 0 or i == len(entries):
                    print(f'imported {i}/{len(entries)} tensors ({out.tell()} bytes)', flush=True)
            out.flush()
            os.fsync(out.fileno())
        for shard in input_shards:
            shard.revalidate()
        report = dict(provenance, tensors=written, output_bytes=os.stat(temporary).st_size,
                      header_bytes=len(header), header_sha256=hashlib.sha256(header).hexdigest())
        for shard in input_shards:
            shard.assert_unchanged()
        with report_path.open('x') as fp:
            report_created = True
            json.dump(report, fp, indent=2)
            fp.write('\n')
            fp.flush()
            os.fsync(fp.fileno())
        for shard in input_shards:
            shard.assert_unchanged()
        os.link(temporary, output)
        published = True
        os.unlink(temporary)
    except BaseException:
        if os.path.exists(temporary): os.unlink(temporary)
        if report_created and not published: report_path.unlink()
        raise
    return report


def verify_import(output):
    report = json.loads(Path(str(output) + '.import.json').read_text())
    with OpenShard(output) as shard:
        tensors = shard.tensors
        expected = {t['name']: t for t in report['tensors']}
        if (len(expected) != len(tensors) or len(expected) != len(report['tensors']) or
                shard._version[4] != report['output_bytes']):
            fail('output does not match import manifest')
        if hashlib.sha256(shard.read_at(0, report['header_bytes'], 'header')).hexdigest() != report['header_sha256']:
            fail('header differs from import manifest')
        for t in tensors:
            recorded = expected.get(t['name'])
            if recorded is None or (t['qtype'], list(t['shape']), t['nbytes']) != (
                    recorded['type'], recorded['shape'], recorded['nbytes']):
                fail(f'{t["name"]}: layout differs from import manifest')
            if shard.sha256_range(t['abs_offset'], t['nbytes'], t['name']) != recorded['sha256']:
                fail(f'{t["name"]}: payload differs from import manifest')
        shard.assert_unchanged()
    print(f'verified {len(tensors)} imported tensor payloads', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--hf', type=Path)
    parser.add_argument('--tokenizer-template', type=Path)
    parser.add_argument('--source-revision')
    parser.add_argument('--quant-revision')
    parser.add_argument('--checksums', type=Path, help='JSON mapping input filenames to pinned Hub LFS SHA256 hashes')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify', type=Path, help='verify a completed import against its payload manifest')
    args = parser.parse_args()
    if args.verify:
        verify_import(args.verify)
        return
    if any(v is None for v in [args.input_dir, args.hf, args.tokenizer_template,
                               args.source_revision, args.quant_revision, args.out, args.checksums]):
        parser.error('input-dir, hf, tokenizer-template, both revisions, checksums and out are required')
    for revision in [args.source_revision, args.quant_revision]:
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            fail('revisions must be full Git commit hashes')
    if not args.dry_run and (args.out.exists() or Path(str(args.out) + '.import.json').exists()):
        fail('output or import report already exists')
    paths = sorted(args.input_dir.glob('*.gguf'))
    if not paths:
        fail('input directory has no GGUF shards')
    with contextlib.ExitStack() as stack:
        shards = [stack.enter_context(OpenShard(path)) for path in paths]
        first = shards[0].metadata
        validate_metadata(first)
        if first.get('split.count') != len(paths):
            fail('incomplete split GGUF')
        inputs = {}
        for i, shard in enumerate(shards):
            meta, tensors = shard.metadata, shard.tensors
            if meta.get('split.no') != i or meta.get('split.count') != len(paths):
                fail('GGUF shard sequence/count mismatch')
            for tensor in tensors:
                if tensor['name'] in inputs:
                    fail(f'duplicate tensor across shards: {tensor["name"]}')
                inputs[tensor['name']] = tensor
        if first.get('split.tensors.count') != len(inputs):
            fail('split tensor count mismatch')
        checksums = json.loads(args.checksums.read_text())
        for path in paths:
            if path.name not in checksums:
                fail(f'{path}: missing pinned input checksum')
        tokenizer, tokens = load_tokenizer_records(args.tokenizer_template)
        validate_tokenizer_template(args.hf, tokens, 154880)
        if first.get('tokenizer.ggml.tokens') != tokens:
            fail('published GGUF token IDs differ from the template')
        db = SourceDB(args.hf)
        entries = []
        try:
            plans = build_plan(db, 'q4')
            for plan in plans:
                name = published_name(plan.name)
                src = inputs.pop(name, None)
                if src is None or tuple(src['shape']) != tuple(plan.shape):
                    fail(f'{plan.name}: missing source tensor or incompatible shape')
                entry = dict(input=src)
                qtype = src['qtype']
                if plan.name.endswith('.kda_a_log.weight'):
                    if qtype != 0 or db.info(plan.source)['dtype'] != 'F32':
                        fail('KDA decay parameters must be F32')
                    entry.update(replacement=native_decay(payload(src), db.read(plan.source)),
                                 action='restore-source-A_log')
                elif plan.qtype == 0:
                    if qtype != 0 or plan.transform is not None or plan.row_count is not None:
                        fail(f'{plan.name}: unsupported control parameter encoding')
                    expected = source_f32(db.read(plan.source), db.info(plan.source)['dtype'])
                    if payload(src) != expected:
                        fail(f'{plan.name}: published control differs from the checkpoint')
                    entry['action'] = 'byte-copy-verified-source'
                elif plan.name.endswith(('.indexer.pool_ape.weight', '.indexer.proj.weight')) and qtype == 0:
                    replacement = exact_bf16(payload(src))
                    if db.info(plan.source)['dtype'] != 'BF16' or replacement != db.read(plan.source):
                        fail('published indexer tensor differs from the checkpoint')
                    entry.update(replacement=replacement, action='exact-F32-to-BF16')
                    qtype = 30
                if plan.is_expert:
                    allowed = (8, 10, 12, 13, 14, 16)
                elif plan.qtype == 0:
                    allowed = (0,)
                elif plan.name.endswith(('.hc_attn_fn.weight', '.hc_ffn_fn.weight', '.indexer.pool_gate.weight')):
                    allowed = (8, 30)
                elif plan.name.endswith('.indexer.pool_ape.weight'):
                    allowed = (30,)
                else:
                    allowed = (8, 12, 30)
                if qtype not in allowed:
                    fail(f'{plan.name}: type {qtype} is unsupported for this operation')
                entry['plan'] = dataclasses.replace(plan, qtype=qtype, nbytes=tensor_bytes(qtype, plan.shape))
                entries.append(entry)
            if inputs:
                fail(f'unmapped tensors: {sorted(inputs)[:5]}')
            architecture = first['general.architecture']
            source_eps = first.get(architecture + '.hyper_connection.epsilon')
            config = json.loads((args.hf / 'config.json').read_text())['text_config']
            if config.get('hc_eps') != 1e-6:
                fail('this importer targets DS4 source HC epsilon 1e-6')
            records = model_metadata(args.hf, args.source_revision) + tokenizer + [
                kv_string('ds4.import.quantization_revision', args.quant_revision),
                kv_string('ds4.import.quantization_source', 'unsloth/GLM-5.3-Flash-GGUF'),
                kv_bool('ds4.import.source_aligned', True),
                kv_f32('ds4.import.published_hc_epsilon', source_eps),
            ]
            provenance = dict(mode='source-aligned', source_revision=args.source_revision,
                              quantization_revision=args.quant_revision, published_hc_epsilon=source_eps,
                              engine_hc_epsilon=1e-6,
                              input_files=[dict(path=str(shard.path), sha256=checksums[path.name])
                                           for path, shard in zip(paths, shards)],
                              note='Quantized matrices are byte-copied; native A_log and DS4 model conventions are restored.')
            print(json.dumps(dict(provenance, tensor_count=len(entries),
                                  tensor_bytes=sum(e['plan'].nbytes for e in entries)), indent=2), flush=True)
            if not args.dry_run:
                for path, shard in zip(paths, shards):
                    verify_file(shard, checksums[path.name])
                write_import(args.out, entries, records, provenance, shards)
        finally:
            db.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f'glm53-import: {error}', file=sys.stderr)
        sys.exit(1)
