import unittest

import cv2
import numpy as np

from src.segmentation.morphology_refinement import filter_stage, morphology_stage, postprocess_refined
from src.segmentation.postprocess import postprocess


PP = dict(open_kernel=5, close_kernel=9, kernel_reference_side=1000,
          min_area_ratio=.001, max_area_ratio=.85, min_aspect_ratio=.3,
          max_aspect_ratio=2., min_fill_ratio=.3, min_ellipse_similarity=.3,
          hole_fill_max_area_ratio=.3)


class TestMorphologyRefinement(unittest.TestCase):
    def test_original_profile_preserves_mask_and_candidates(self):
        for side in (65, 256, 1000):
            with self.subTest(side=side):
                mask = np.zeros((side, side), np.uint8)
                cv2.ellipse(mask, (side//2, side//2), (side//5, side//3), 20, 0, 360, 1, -1)
                cv2.circle(mask, (side//2, side//2), max(1, side//20), 0, -1)
                expected, candidates = postprocess(mask, PP)
                actual, actual_candidates = postprocess_refined(mask, PP, dict(close_kernel=9, early_fill=False, candidate_fraction=0.))
                np.testing.assert_array_equal(expected, actual)
                self.assertEqual(candidates, actual_candidates)

    def test_early_fill_can_rescue_low_occupancy_face(self):
        mask = np.zeros((220, 220), np.uint8)
        cv2.ellipse(mask, (110, 110), (40, 60), 0, 0, 360, 1, -1)
        cv2.circle(mask, (110, 110), 16, 0, -1)
        pp = dict(PP, min_fill_ratio=.73)
        late, _ = filter_stage(mask, pp, False)
        early, _ = filter_stage(mask, pp, True)
        self.assertFalse(late.any())
        self.assertTrue(early[110, 110])

    def test_weak_closing_avoids_joining_nearby_regions(self):
        mask = np.zeros((1000, 1000), np.uint8)
        mask[400:600, 300:450] = 1
        mask[400:600, 454:604] = 1
        weak = morphology_stage(mask, PP, dict(close_kernel=1, candidate_fraction=0.))
        strong = morphology_stage(mask, PP, dict(close_kernel=9, candidate_fraction=0.))
        self.assertEqual(cv2.connectedComponents(weak, connectivity=8)[0]-1, 2)
        self.assertEqual(cv2.connectedComponents(strong, connectivity=8)[0]-1, 1)

    def test_candidate_scaling_does_not_depend_on_canvas_size(self):
        small = np.zeros((300, 300), np.uint8)
        cv2.ellipse(small, (150, 150), (40, 65), 0, 0, 360, 1, -1)
        large = np.pad(small, ((0, 300), (0, 300)))
        profile = dict(close_kernel=5, candidate_fraction=.025)
        a = morphology_stage(small, PP, profile)
        b = morphology_stage(large, PP, profile)
        np.testing.assert_array_equal(a, b[:300, :300])


if __name__ == '__main__':
    unittest.main()
