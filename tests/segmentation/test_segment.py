import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.segmentation.segment import segment_image
from src.segmentation.baseline import segment_image_baseline

MODEL = {
    "schema_version": 1,
    "color_space": "OpenCV-YCrCb-uint8-full-range",
    "feature_order": ["Cb", "Cr"],
    "mean": [113.86623333333333, 143.3184875],
    "covariance": [
        [13.11251944771992, -7.345890096625399],
        [-7.345890096625399, 16.315821194932038],
    ],
    "regularization_lambda": 1.0,
    "model_revision": "test-rev",
}

CONFIG = {
    "distance_threshold": 20.0,
    "model_revision": "test-rev",
    "postprocess": {
        "open_kernel": 3,
        "close_kernel": 5,
        "min_area_ratio": 0.001,
        "max_area_ratio": 0.5,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0,
        "min_fill_ratio": 0.4,
        "min_ellipse_similarity": 0.5,
        "hole_fill_max_area_ratio": 0.3,
    },
}


def _bgr(y, cr, cb):
    arr = np.array([[[y, cr, cb]]], dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_YCrCb2BGR)[0, 0].tolist()


SKIN_BGR = _bgr(128, 143, 114)
BG_BGR = _bgr(128, 90, 120)


def _make_image(h, w, rects):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = BG_BGR
    for x, y, rw, rh in rects:
        img[y:y + rh, x:x + rw] = SKIN_BGR
    return img


class TestSegment(unittest.TestCase):

    def test_invalid_image(self):
        with self.assertRaises(ValueError):
            segment_image(np.zeros((10, 10), dtype=np.uint8), MODEL, CONFIG)
        with self.assertRaises(ValueError):
            segment_image(np.zeros((10, 10, 3), dtype=np.float32), MODEL, CONFIG)

    def test_invalid_model(self):
        bad = dict(MODEL)
        bad["schema_version"] = 2
        with self.assertRaises(ValueError):
            segment_image(_make_image(20, 20, []), bad, CONFIG)

    def test_revision_mismatch(self):
        cfg = dict(CONFIG)
        cfg["model_revision"] = "other"
        with self.assertRaises(ValueError):
            segment_image(_make_image(20, 20, []), MODEL, cfg)

    def test_no_candidate(self):
        img = _make_image(50, 50, [])
        skin, face, cands, t = segment_image(img, MODEL, CONFIG)
        self.assertEqual(skin.shape, (50, 50))
        self.assertEqual(face.shape, (50, 50))
        self.assertEqual(skin.dtype, np.uint8)
        self.assertEqual(face.dtype, np.uint8)
        self.assertTrue(set(np.unique(skin).tolist()).issubset({0, 1}))
        self.assertTrue(set(np.unique(face).tolist()).issubset({0, 1}))
        self.assertEqual(cands, [])
        self.assertEqual(int(face.sum()), 0)
        self.assertGreaterEqual(t, 0.0)

    def test_multiple_candidates(self):
        img = _make_image(100, 100, [(10, 10, 20, 20), (70, 70, 20, 20)])
        skin, face, cands, t = segment_image(img, MODEL, CONFIG)
        self.assertEqual(len(cands), 2)
        for c in cands:
            self.assertIn("bbox", c)
            self.assertIn("area", c)
            for k in ("x", "y", "width", "height"):
                self.assertIn(k, c["bbox"])

    def test_small_image(self):
        img = _make_image(10, 10, [])
        skin, face, cands, t = segment_image(img, MODEL, CONFIG)
        self.assertEqual(skin.shape, (10, 10))
        self.assertEqual(face.shape, (10, 10))

    def test_baseline_runs(self):
        img = _make_image(100, 100, [(10, 10, 20, 20)])
        skin, face, cands, t = segment_image_baseline(img, MODEL, CONFIG)
        self.assertEqual(skin.shape, (100, 100))
        self.assertEqual(face.shape, (100, 100))
        self.assertEqual(skin.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()