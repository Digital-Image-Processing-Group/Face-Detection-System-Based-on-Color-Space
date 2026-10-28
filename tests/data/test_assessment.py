"""Hold-out isolation, stale-model rejection and binary face-baseline contracts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.training.assessment import assess, confusion, face_proposal, metrics
from src.training.dataset import Record, write_manifest
from src.training.model import fit_model


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        for i, split in enumerate(["train", "train", "validation", "test"]):
            name = f"person_{i}"
            photo = np.random.default_rng(i).integers(20, 240, (24, 24, 3), dtype=np.uint8)
            mask = np.zeros((24, 24), dtype=np.uint8)
            mask[4:20, 4:20] = 255
            Image.fromarray(photo).save(self.root / f"{name}.png")
            for kind in ("skin", "face"):
                Image.fromarray(mask).save(self.root / f"{name}_{kind}.png")
            self.rows.append(Record(name, f"{name}.png", name, name, split, f"{name}_skin.png",
                                    f"{name}_face.png", "fixture", "synthetic:fixture", "fixture",
                                    "test-only", "verified", "complete", "complete"))
        write_manifest(self.manifest, self.rows)
        self.model, _, _ = fit_model(self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def run_assessment(self):
        return assess(self.manifest, self.model, self.root / "result.json",
                      self.root / "point.json", self.root / "previews")

    def test_test_labels_do_not_select_threshold(self):
        before = self.run_assessment()
        Image.fromarray(np.zeros((24, 24), dtype=np.uint8)).save(self.root / self.rows[-1].skin_mask_path)
        after = self.run_assessment()
        self.assertEqual(before["operating_point"], after["operating_point"])
        self.assertNotEqual(before["summary"]["test"]["skin"], after["summary"]["test"]["skin"])
        exported = np.array(Image.open(self.root / "previews/person_3_skin.png"))
        self.assertTrue(set(np.unique(exported)).issubset({0, 255}))

    def test_changed_training_mask_rejects_stale_model(self):
        Image.fromarray(np.full((24, 24), 255, dtype=np.uint8)).save(self.root / self.rows[0].skin_mask_path)
        with self.assertRaisesRegex(ValueError, "Training files changed"):
            self.run_assessment()

    def test_no_face_images_have_separate_image_false_positive_accounting(self):
        for kind in ('skin','face'):
            Image.fromarray(np.zeros((24,24),dtype=np.uint8)).save(self.root / getattr(self.rows[-1],f'{kind}_mask_path'))
        report=self.run_assessment()
        result=report['no_face']['test']
        self.assertEqual(result['images'],1)
        predicted=np.array(Image.open(self.root/'previews/person_3_face_baseline.png'))
        self.assertEqual(result['nonempty_predictions'],int(predicted.any()))
        self.assertEqual(result['false_positive_pixels'],int(np.count_nonzero(predicted)))
        self.assertIsNone(report['summary']['test']['face_baseline']['mean_positive_image_iou'])

    def test_training_margin_uses_only_interior(self):
        model, _, _ = fit_model(self.manifest, samples_per_image=1000, mask_margin=2)
        self.assertEqual(model["training_info"]["sampled_pixel_count"], 2 * 12 * 12)

    def test_face_baseline_fills_holes_and_removes_small_component(self):
        skin = np.zeros((100, 100), dtype=np.uint8)
        skin[10:70, 10:70] = 1
        skin[30:40, 30:40] = 0
        skin[80:85, 80:85] = 1
        face = face_proposal(skin)
        self.assertEqual(face[35, 35], 1)
        self.assertEqual(face[82, 82], 0)
        self.assertFalse(face_proposal(np.zeros_like(skin)).any())

    def test_confusion_and_empty_metric_conventions(self):
        self.assertEqual(confusion(np.array([1, 1, 0]), np.array([1, 0, 1])), (1, 1, 1))
        self.assertEqual(metrics([1, 1, 1])["dice"], 0.5)
        self.assertIsNone(metrics([0, 0, 0])["iou"])
        self.assertEqual(metrics([0, 0, 5])["dice"], 0)


if __name__ == "__main__":
    unittest.main()
