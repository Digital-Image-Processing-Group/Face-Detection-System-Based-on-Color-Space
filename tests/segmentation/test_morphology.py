import unittest
import cv2
import numpy as np
from src.segmentation.postprocess import _fill_holes_limited, _split_bridges, postprocess
from src.segmentation.segment import _validate_config
from tests.segmentation.test_segment import CONFIG
from copy import deepcopy


class MorphologyTests(unittest.TestCase):
    def test_hole_budget_and_border_background(self):
        mask = np.ones((30, 30), np.uint8)
        mask[5:8, 5:8] = 0
        mask[15:19, 15:19] = 0
        mask[:4, 22:25] = 0
        filled = _fill_holes_limited(mask, 20, total_budget=15)
        self.assertEqual(int(filled.sum() - mask.sum()), 9)
        self.assertFalse(filled[:4, 22:25].any())

    def test_bridge_partition_preserves_original_and_separates_cores(self):
        mask = np.zeros((80, 160), np.uint8)
        cv2.circle(mask, (35, 40), 25, 1, -1)
        cv2.circle(mask, (125, 40), 25, 1, -1)
        mask[39:42, 35:126] = 1
        parts = _split_bridges(mask, .12)
        self.assertEqual(len(parts), 2)
        np.testing.assert_array_equal(np.sum(parts, axis=0), mask)
        self.assertFalse((parts[0] & parts[1]).any())

    def test_adaptive_keeps_multiple_small_faces_and_input(self):
        pp = dict(CONFIG['postprocess'], method='adaptive', local_kernel_ratio=.02,
                  noise_area_ratio=.00002, bridge_open_ratio=0, hole_fill_total_area_ratio=.3)
        mask = np.zeros((250, 300), np.uint8)
        cv2.ellipse(mask, (45, 80), (14, 22), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (210, 150), (30, 40), 0, 0, 360, 1, -1)
        mask[240, 5] = 1
        before = mask.copy()
        face, candidates = postprocess(mask, pp)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(face[240, 5], 0)
        np.testing.assert_array_equal(mask, before)

    def test_invalid_adaptive_parameters(self):
        cfg = deepcopy(CONFIG)
        cfg['postprocess'].update(method='adaptive', local_kernel_ratio=.02,
            noise_area_ratio=0, bridge_open_ratio=0, hole_fill_total_area_ratio=float('nan'))
        with self.assertRaises(ValueError):
            _validate_config(cfg)

    def test_submission_and_engineering_results_match(self):
        import importlib.util
        from pathlib import Path
        from src.segmentation import segment_image, segment_image_baseline
        import json
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('submission_morphology_test', root/'submission/main.py')
        bundle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bundle)
        model = json.loads((root/'models/skin_model.json').read_text(encoding='utf-8'))
        image = np.zeros((160, 260, 3), np.uint8)
        image[:] = (128, 90, 120)
        cv2.ellipse(image, (50, 70), (20, 32), 0, 0, 360, (128, 143, 114), -1)
        cv2.ellipse(image, (190, 100), (14, 20), 0, 0, 360, (128, 143, 114), -1)
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        for diagonal, name, function in [(False, 'pipeline.json', segment_image),
                                         (True, 'baseline.json', segment_image_baseline)]:
            config = json.loads((root/'config'/name).read_text(encoding='utf-8'))
            actual = function(image, model, config)
            bundled = bundle.segment_image(image, diagonal=diagonal)
            for key in ('skin_mask', 'face_mask'):
                np.testing.assert_array_equal(actual[key], bundled[key])
            self.assertEqual(actual['candidates'], bundled['candidates'])


if __name__ == '__main__':
    unittest.main()
