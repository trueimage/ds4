#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek41_quantize import NativeQuantizer, calibration
from deepseek41_validate_gguf import mxfp4_to_f32
from glm53_quantize import QTYPE_MXFP4, qtype_nbytes


class FakeSourceDB:
    def __init__(self, codes, scales):
        self.tensors = {"e.weight": ("I8", codes), "e.scale": ("F8_E8M0", scales)}

    def info(self, name):
        dtype, values = self.tensors[name]
        return {"dtype": dtype, "shape": list(values.shape)}

    def read(self, name):
        return self.tensors[name][1].tobytes()


def quantizer():
    # The repack and the source decoder need NumPy only, not the C library.
    q = NativeQuantizer.__new__(NativeQuantizer)
    q.np = np
    return q


def reference_repack(codes, scales):
    """Scalar mirror of repack_fp4_weight_mxfp4 in deepseek4-quantize.c."""
    out = bytearray()
    for r in range(codes.shape[0]):
        for b in range(scales.shape[1]):
            src = codes[r, b * 16:(b + 1) * 16]
            out.append(int(scales[r, b]))
            for i in range(16):
                lo = src[i // 2] >> 4 if i & 1 else src[i // 2] & 15
                hi = src[(16 + i) // 2] >> 4 if i & 1 else src[(16 + i) // 2] & 15
                out.append(int(lo) | int(hi) << 4)
    return bytes(out)


class NativeMxfp4Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        self.codes = rng.integers(0, 256, size=(6, 48), dtype=np.uint8)
        self.scales = rng.integers(1, 251, size=(6, 3), dtype=np.uint8)
        self.db = FakeSourceDB(self.codes, self.scales)

    def test_repack_matches_the_c_layout(self):
        self.scales[0, 0], self.scales[0, 1] = 0, 254
        data = quantizer().repack_mxfp4(self.db, "e.weight")
        self.assertEqual(len(data), qtype_nbytes(QTYPE_MXFP4, (96, 6)))
        self.assertEqual(data, reference_repack(self.codes, self.scales))

    def test_runtime_decode_equals_the_source_weights(self):
        q = quantizer()
        decoded = mxfp4_to_f32(np, q.repack_mxfp4(self.db, "e.weight"), 96)
        source = q.to_f32(self.db, "e.weight")
        self.assertEqual(decoded.tobytes(), source.tobytes())

    def test_every_code_and_scale_survives(self):
        codes = np.arange(256, dtype=np.uint8).reshape(1, 256)
        scales = np.arange(16, dtype=np.uint8).reshape(1, 16) * 16
        data = np.frombuffer(quantizer().repack_mxfp4(FakeSourceDB(codes, scales), "e.weight"),
                             dtype=np.uint8).reshape(16, 17)
        self.assertEqual(data[:, 0].tolist(), scales[0].tolist())
        values = np.concatenate([data[:, 1:] & 15, data[:, 1:] >> 4], axis=1).reshape(-1)
        expected = np.stack([codes[0] & 15, codes[0] >> 4], axis=1).reshape(-1)
        self.assertEqual(values.tolist(), expected.tolist())

    def test_rejects_what_is_not_native_fp4(self):
        q = quantizer()
        self.scales[2, 1] = 255
        with self.assertRaisesRegex(ValueError, "nonfinite scale"):
            q.repack_mxfp4(self.db, "e.weight")
        with self.assertRaisesRegex(ValueError, "not a native FP4 expert"):
            q.repack_mxfp4(FakeSourceDB(self.codes, self.scales[:, :2]), "e.weight")
        with self.assertRaisesRegex(ValueError, "omit --imatrix"):
            calibration("mxfp4", "calibration.dat")
        self.assertEqual(calibration("mxfp4", None), "native")


if __name__ == "__main__":
    unittest.main()
