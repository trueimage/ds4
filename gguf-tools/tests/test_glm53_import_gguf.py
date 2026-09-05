#!/usr/bin/env python3
"""Exercise the imported GGUF byte contract and parameter conversions."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm53_import_gguf as importer
from glm53_quantize import TensorPlan, align, kv_string, kv_u32, tensor_header


def make_gguf(path, tensors, records=()):
    plans = []
    data = bytearray()
    for name, shape, qtype, raw in tensors:
        plans.append(TensorPlan(name, shape, qtype, 'test', offset=len(data), nbytes=len(raw)))
        data.extend(raw)
        data.extend(bytes(align(len(raw)) - len(raw)))
    header = b'GGUF' + struct.pack('<IQQ', 3, len(plans), len(records))
    header += b''.join(records) + b''.join(tensor_header(p) for p in plans)
    path.write_bytes(header + bytes(align(len(header)) - len(header)) + data)


class ImportTests(unittest.TestCase):
    def setUp(self):
        # Tiny fixtures exercise the writer without requiring production-model
        # disk capacity on CI hosts.
        disk = patch.object(importer.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 40))
        disk.start()
        self.addCleanup(disk.stop)

    def test_quantized_bytes_and_header_survive_import_and_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input.gguf', root / 'output.gguf'
            specs = []
            for qtype in (8, 12, 13, 14):
                n = importer.tensor_bytes(qtype, (256, 2))
                specs.append((f'published.{qtype}', (256, 2), qtype, bytes(i % 256 for i in range(n))))
            make_gguf(source, specs)
            source_before = source.read_bytes()
            with importer.OpenShard(source) as shard:
                importer.verify_file(shard, hashlib.sha256(source_before).hexdigest())
                entries = [dict(input=t, plan=TensorPlan(f'ds4.{t["qtype"]}', t['shape'], t['qtype'],
                                                        'test', nbytes=t['nbytes']))
                           for t in shard.tensors]
                importer.write_import(output, entries, [kv_string('general.name', 'test')], {}, [shard])
                existing_output = output.read_bytes()
                report_path = Path(str(output) + '.import.json')
                existing_report = report_path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    importer.write_import(output, entries, [], {}, [shard])
                self.assertEqual(output.read_bytes(), existing_output)
                self.assertEqual(report_path.read_bytes(), existing_report)
            importer.verify_import(output)
            with importer.OpenShard(output) as imported_shard:
                self.assertEqual(imported_shard.metadata['general.name'], 'test')
                self.assertEqual([importer.payload(t) for t in imported_shard.tensors],
                                 [s[3] for s in specs])
                imported = imported_shard.tensors
            self.assertEqual(source.read_bytes(), source_before)
            original_output = output.read_bytes()
            output.write_bytes(original_output.replace(b'test', b'best', 1))
            with self.assertRaisesRegex(ValueError, 'header differs'):
                importer.verify_import(output)
            output.write_bytes(original_output)
            with output.open('r+b') as fp:
                fp.seek(imported[2]['abs_offset'] + 5)
                original = fp.read(1)
                fp.seek(-1, 1)
                fp.write(bytes([original[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'payload differs'):
                importer.verify_import(output)

    def test_published_metadata_semantics(self):
        # Captured from the pinned publisher's metadata-only Q4 shard.
        meta = json.loads((Path(__file__).parent / 'data/glm53-unsloth-q4-metadata.json').read_text())
        importer.validate_metadata(meta)
        for key, incompatible in [('hyper_connection.epsilon', 1e-7),
                                  ('expert_weights_norm', False),
                                  ('kda.gate_lower_bound', -4.0),
                                  ('attention.indexer.kpool', 8)]:
            changed = dict(meta)
            changed['glm5next.' + key] = incompatible
            with self.assertRaisesRegex(ValueError, 'unsupported metadata'):
                importer.validate_metadata(changed)

    def test_exact_bf16_preserves_signed_zero_and_rejects_rounding(self):
        words = np.array([0, 0x8000, 0x3f80, 0xbf80, 1, 0x7f7f], dtype='<u2')
        raw = (words.astype('<u4') << 16).tobytes()
        self.assertEqual(importer.exact_bf16(raw), words.tobytes())
        self.assertEqual(importer.source_f32(words.tobytes(), 'BF16'), raw)
        for bits in (0x3f800001, 0x7f800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                importer.exact_bf16(struct.pack('<I', bits))

    def test_restores_source_decay_without_inverting_rounded_exp(self):
        original = np.array([1.562482595, -2.1234567, 0.0], dtype='<f4')
        published = (-np.exp(original.astype(np.float64))).astype('<f4').tobytes()
        self.assertEqual(importer.native_decay(published, original.tobytes()), original.tobytes())
        for invalid in (b'', np.ones(3, dtype='<f4').tobytes(),
                        np.array([-1, -1, -1], dtype='<f4').tobytes(),
                        np.full(3, np.nan, dtype='<f4').tobytes()):
            with self.assertRaises(ValueError):
                importer.native_decay(invalid, original.tobytes())

    def test_rejects_truncation_overlap_duplicates_and_bad_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.gguf'
            make_gguf(path, [('x', (32,), 8, bytes(34))])
            path.write_bytes(path.read_bytes()[:-40])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                importer.OpenShard(path)
            make_gguf(path, [('x', (32,), 8, bytes(34)), ('x', (32,), 8, bytes(34))])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                importer.OpenShard(path)
            plans = [TensorPlan(n, (32,), 8, 'test', offset=0, nbytes=34) for n in ('x', 'y')]
            header = b'GGUF' + struct.pack('<IQQ', 3, 2, 0) + b''.join(tensor_header(p) for p in plans)
            path.write_bytes(header + bytes(align(len(header)) - len(header)) + bytes(128))
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                importer.OpenShard(path)
            make_gguf(path, [], [kv_u32('general.alignment', 3)])
            with self.assertRaisesRegex(ValueError, 'alignment'):
                importer.OpenShard(path)
        with self.assertRaisesRegex(ValueError, 'block layout'):
            importer.tensor_bytes(13, (255, 2))

    def test_checksum_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.gguf'
            make_gguf(path, [('x', (32,), 8, bytes(34))])
            with importer.OpenShard(path) as shard:
                importer.verify_file(shard, hashlib.sha256(path.read_bytes()).hexdigest())
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    importer.verify_file(shard, '0' * 64)

    def test_symlink_replacement_invalidates_open_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, replacement = root / 'original.gguf', root / 'replacement.gguf'
            link, output = root / 'input.gguf', root / 'out.gguf'
            make_gguf(original, [('x', (32,), 8, bytes(34))])
            make_gguf(replacement, [('x', (32,), 8, bytes([1]) * 34)])
            link.symlink_to(original.name)
            foreign_partial = root / 'out.gguf.foreign.partial'
            foreign_partial.write_bytes(b'preserve')
            with importer.OpenShard(link) as shard:
                importer.verify_file(shard, hashlib.sha256(original.read_bytes()).hexdigest())
                entry = dict(input=shard.tensors[0],
                             plan=TensorPlan('x', (32,), 8, 'test', nbytes=34))
                link.unlink()
                link.symlink_to(replacement.name)
                with self.assertRaisesRegex(ValueError, 'input changed during import'):
                    importer.write_import(output, [entry], [], {}, [shard])
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + '.import.json').exists())
            self.assertEqual(foreign_partial.read_bytes(), b'preserve')

    def test_modification_during_copy_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / 'input.gguf', root / 'out.gguf'
            make_gguf(path, [('x', (32,), 8, bytes(34))])
            foreign_partial = root / 'out.gguf.foreign.partial'
            foreign_partial.write_bytes(b'preserve')
            with importer.OpenShard(path) as shard:
                importer.verify_file(shard, hashlib.sha256(path.read_bytes()).hexdigest())
                tensor = shard.tensors[0]
                entry = dict(input=tensor, plan=TensorPlan('x', (32,), 8, 'test', nbytes=34))
                original_copy = shard.copy_range

                def copy_then_modify(*args):
                    original_copy(*args)
                    with path.open('r+b') as fp:
                        fp.seek(tensor['abs_offset'])
                        fp.write(b'\x01')

                with patch.object(shard, 'copy_range', side_effect=copy_then_modify):
                    with self.assertRaisesRegex(ValueError, 'input changed during import'):
                        importer.write_import(output, [entry], [], {}, [shard])
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + '.import.json').exists())
            self.assertEqual(foreign_partial.read_bytes(), b'preserve')
            self.assertEqual(list(root.glob('out.gguf.*.partial')), [foreign_partial])

    def test_insufficient_space_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out.gguf'
            with patch.object(importer.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, 'insufficient free space'):
                    importer.write_import(output, [], [], {}, [])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
