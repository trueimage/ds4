#!/usr/bin/env python3

import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepseek41_validate_gguf as audit
from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import dspark_metadata
from glm53_quantize import (GGUF_ARRAY, GGUF_UINT64, QTYPE_F32, TensorPlan, align, kv_u32,
                            kv_u32_array, kv_u64, pack_string, tensor_header)


class FakeSourceDB:
    def __init__(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


class FakeQuantizer:
    np = None

    def __init__(self, _library):
        pass

    def to_f32(self, _db, _source):
        return 0.0

    def encode(self, _value, _qtype):
        return struct.pack("<f", 0.0)


class DsparkAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.revision = "a" * 40
        self.config = {"text_config": {
            "dspark_block_size": 4,
            "dspark_markov_rank": 2,
            "dspark_noise_token_id": 5,
            "dspark_target_layer_ids": [1, 3],
            "num_nextn_predict_layers": 1,
            "num_hidden_layers": 5,
            "dspark_n_routed_experts": 2,
            "dspark_num_experts_per_tok": 1,
        }}
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.plan = TensorPlan("mtp.0.norm.weight", (1,), QTYPE_F32, "norm", source="mtp.0.norm.weight",
                               nbytes=4)
        self.args = SimpleNamespace(hf=self.root, gguf=self.root / "support.gguf",
                                    source_revision=self.revision, dspark_support=True,
                                    payload=True, quants_library="unused", imatrix=None)

    def records(self):
        return dspark_metadata(self.config, self.revision)

    def write_gguf(self, records):
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(records))
        header += b"".join(records) + tensor_header(self.plan)
        header += bytes(align(len(header), GGUF_ALIGNMENT) - len(header))
        (self.root / "support.gguf").write_bytes(
            header + struct.pack("<f", 0.0) + bytes(align(4, GGUF_ALIGNMENT) - 4))

    def audit(self):
        with patch.object(audit, "SourceDB", FakeSourceDB), \
             patch.object(audit, "build_dspark_plan", return_value=[self.plan]), \
             patch.object(audit, "NativeQuantizer", FakeQuantizer), \
             patch.object(audit, "Imatrix", return_value=SimpleNamespace()):
            audit.validate(self.args)

    def test_valid_support_header_and_payload(self):
        self.write_gguf(self.records())
        self.audit()

    def test_each_runtime_field_must_match_source(self):
        replacements = {
            "dspark.block_size": kv_u32("dspark.block_size", 3),
            "dspark.markov_rank": kv_u32("dspark.markov_rank", 3),
            "dspark.noise_token_id": kv_u32("dspark.noise_token_id", 6),
            "dspark.target_layer_ids": kv_u32_array("dspark.target_layer_ids", [1, 4]),
            "dspark.stage_count": kv_u32("dspark.stage_count", 2),
            "dspark.n_layers": kv_u32("dspark.n_layers", 6),
            "dspark.expert_count": kv_u32("dspark.expert_count", 3),
            "dspark.expert_used_count": kv_u32("dspark.expert_used_count", 2),
        }
        for key, replacement in replacements.items():
            with self.subTest(key=key):
                records = [replacement if record.startswith(pack_string(key)) else record
                           for record in self.records()]
                self.write_gguf(records)
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    self.audit()

    def test_runtime_fields_require_their_gguf_types(self):
        for key, replacement in (
            ("dspark.noise_token_id", kv_u64("dspark.noise_token_id", 5)),
            ("dspark.target_layer_ids", kv_u32("dspark.target_layer_ids", 1)),
            ("dspark.target_layer_ids", pack_string("dspark.target_layer_ids")
             + struct.pack("<IIQQQ", GGUF_ARRAY, GGUF_UINT64, 2, 1, 3)),
        ):
            with self.subTest(key=key):
                records = [replacement if record.startswith(pack_string(key)) else record
                           for record in self.records()]
                self.write_gguf(records)
                with self.assertRaisesRegex(ValueError, "expected uint32"):
                    self.audit()

    def test_missing_and_duplicate_runtime_fields_fail(self):
        key = "dspark.noise_token_id"
        records = self.records()
        self.write_gguf([record for record in records if not record.startswith(pack_string(key))])
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.audit()
        self.write_gguf(records + [kv_u32(key, 5)])
        with self.assertRaisesRegex(ValueError, "duplicate metadata"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
