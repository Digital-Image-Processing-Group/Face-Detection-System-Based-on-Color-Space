"""Critical contracts: orientation, annotation semantics, leakage and fitting."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.training.annotations import polygon_mask, rasterize
from src.training.cli import main
from src.training.dataset import (Record, assign_splits, connected_groups, read_bgr,
                                  read_manifest, read_mask, validate_dataset,
                                  write_json, write_manifest)
from src.training.model import check_model, fit_model


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def make_record(self, number, split="train", *, skin_status="complete", identity_status="verified"):
        image_id = f"image_{number}"
        image = np.random.default_rng(number).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        Image.fromarray(image).save(self.root / f"{image_id}.png")
        Image.fromarray(np.full((12, 16), 255, dtype=np.uint8)).save(self.root / f"{image_id}_skin.png")
        Image.fromarray(np.full((12, 16), 255, dtype=np.uint8)).save(self.root / f"{image_id}_face.png")
        return Record(image_id, f"{image_id}.png", f"subject_{number}", f"scene_{number}", split,
                      f"{image_id}_skin.png", f"{image_id}_face.png", "synthetic_test",
                      "synthetic:test-fixture", "synthetic", "test-fixture-only", identity_status,
                      skin_status, "complete")

    def test_manifest_roundtrip_and_duplicate_id_rejected(self):
        row = self.make_record(1)
        write_manifest(self.manifest, [row])
        self.assertEqual(read_manifest(self.manifest), [row])
        write_manifest(self.manifest, [row, row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_manifest(self.manifest)

    def test_exif_direction_and_bgr_order(self):
        image = Image.new("RGB", (5, 3), (220, 30, 10))
        exif = image.getexif()
        exif[274] = 6
        path = self.root / "方向.jpg"
        image.save(path, exif=exif)
        bgr = read_bgr(path)
        self.assertEqual(bgr.shape, (5, 3, 3))
        self.assertGreater(bgr[0, 0, 2], bgr[0, 0, 0])

    def test_transparency_and_grayscale_rejected(self):
        for mode, color in (("RGBA", (0, 0, 0, 0)), ("L", 10)):
            path = self.root / f"{mode}.png"
            Image.new(mode, (4, 4), color).save(path)
            with self.assertRaises(ValueError):
                read_bgr(path)

    def test_mask_encodings_and_shape(self):
        path = self.root / "mask.png"
        Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(path)
        np.testing.assert_array_equal(read_mask(path, (1, 2)), [[0, 1]])
        with self.assertRaisesRegex(ValueError, "shape"):
            read_mask(path, (2, 1))
        Image.fromarray(np.array([[0, 1, 255]], dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, "mix"):
            read_mask(path, (1, 3))
        path = self.root / "mask.npy"
        np.save(path, np.array([[0, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(read_mask(path, (1, 2)), [[0, 1]])

    def test_fit_rejects_unverified_or_incomplete_annotation(self):
        rows = [self.make_record(1, skin_status="sampled", identity_status="unverified"),
                self.make_record(2, skin_status="sampled", identity_status="unverified")]
        write_manifest(self.manifest, rows)
        self.assertFalse(validate_dataset(self.manifest, "train")["valid"])
        self.assertFalse(validate_dataset(self.manifest, "evaluation")["valid"])
        with self.assertRaisesRegex(ValueError, "Dataset validation failed"):
            fit_model(self.manifest)

    def test_subject_leakage_and_transitive_groups(self):
        rows = [self.make_record(1), self.make_record(2, "validation"), self.make_record(3, "test")]
        rows[0] = replace(rows[0], subject_ids="one;two")
        rows[1] = replace(rows[1], subject_ids="two;three")
        rows[2] = replace(rows[2], subject_ids="three")
        self.assertEqual(len(connected_groups(rows)), 1)
        write_manifest(self.manifest, rows)
        report = validate_dataset(self.manifest)
        self.assertTrue(any("crosses splits" in error for error in report["errors"]))

    def test_identical_images_across_splits_rejected(self):
        rows = [self.make_record(1), self.make_record(2, "test")]
        (self.root / rows[1].image_path).write_bytes((self.root / rows[0].image_path).read_bytes())
        write_manifest(self.manifest, rows)
        self.assertTrue(any("Identical" in error for error in validate_dataset(self.manifest)["errors"]))

    def test_group_split_is_reproducible_and_keeps_shared_subjects(self):
        rows = [replace(self.make_record(i), split="") for i in range(10)]
        rows[1] = replace(rows[1], subject_ids=rows[0].subject_ids)
        first = assign_splits(rows, seed=7)
        self.assertEqual(first, assign_splits(rows, seed=7))
        self.assertEqual(first[0].split, first[1].split)
        self.assertEqual({row.split for row in first}, {"train", "validation", "test"})
        self.assertEqual(self.manifest.exists(), False)

    def test_split_refuses_unverified_and_existing_splits(self):
        rows = [self.make_record(i) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "existing"):
            assign_splits(rows)
        rows = [replace(row, split="", identity_status="unverified") for row in rows]
        with self.assertRaisesRegex(ValueError, "unverified"):
            assign_splits(rows)

    def test_fitting_uses_cbcr_order_and_sample_covariance(self):
        rows = [self.make_record(1), self.make_record(2)]
        write_manifest(self.manifest, rows)
        model, pixels, _ = fit_model(self.manifest, samples_per_image=1000)
        expected = []
        for row in rows:
            image = read_bgr(self.root / row.image_path)
            color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            expected.append(color[:, :, [2, 1]].reshape(-1, 2))
        expected = np.concatenate(expected).astype(float)
        np.testing.assert_allclose(model["mean"], expected.mean(axis=0))
        np.testing.assert_allclose(model["covariance"], np.cov(expected, rowvar=False, ddof=1))
        self.assertEqual(len(pixels), 384)
        check_model(model)

    def test_balanced_sampling_and_determinism(self):
        rows = [self.make_record(1), self.make_record(2)]
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[:3, :3] = 255
        Image.fromarray(mask).save(self.root / rows[0].skin_mask_path)
        write_manifest(self.manifest, rows)
        model, pixels, _ = fit_model(self.manifest, samples_per_image=20, seed=99)
        again, second_pixels, _ = fit_model(self.manifest, samples_per_image=20, seed=99)
        self.assertEqual(model, again)
        np.testing.assert_array_equal(pixels, second_pixels)
        self.assertEqual(model["training_info"]["actual_samples_per_image"], 9)
        self.assertEqual(len(pixels), 18)

    def test_held_out_pixels_do_not_enter_fit(self):
        rows = [self.make_record(1), self.make_record(2), self.make_record(3, "test")]
        write_manifest(self.manifest, rows)
        before, _, _ = fit_model(self.manifest, samples_per_image=30)
        changed = np.random.default_rng(100).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        Image.fromarray(changed).save(self.root / rows[2].image_path)
        after, _, _ = fit_model(self.manifest, samples_per_image=30)
        self.assertEqual(before["mean"], after["mean"])
        self.assertEqual(before["covariance"], after["covariance"])
        self.assertEqual(before["model_revision"], after["model_revision"])

    def test_bad_covariance_rejected(self):
        rows = [self.make_record(1), self.make_record(2)]
        write_manifest(self.manifest, rows)
        model, _, _ = fit_model(self.manifest)
        bad = dict(model, covariance=[[1, 4], [4, 1]])
        with self.assertRaisesRegex(ValueError, "semidefinite"):
            check_model(bad)

    def test_polygon_holes_and_coordinate_validation(self):
        spec = {"width": 10, "height": 8, "skin": {
            "include": [[[1, 1], [8, 1], [8, 6], [1, 6]]],
            "exclude": [[[3, 3], [5, 3], [5, 5], [3, 5]]],
        }}
        mask = polygon_mask(spec, (8, 10), "skin")
        self.assertEqual(mask[2, 2], 1)
        self.assertEqual(mask[4, 4], 0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            polygon_mask(spec, (10, 8), "skin")

    def test_rasterize_retains_annotation_semantics(self):
        row = replace(self.make_record(1), skin_status="missing", skin_mask_path="",
                      face_status="missing", face_mask_path="")
        write_manifest(self.manifest, [row])
        polygons = self.root / "polygons"
        write_json(polygons / f"{row.image_id}.json", {
            "image_id": row.image_id, "annotator": "unit-test", "width": 16, "height": 12,
            "skin": {"status": "sampled", "include": [[[1, 1], [3, 1], [3, 3]]], "exclude": []},
            "face": {"status": "draft", "include": [[[0, 0], [10, 0], [10, 10]]], "exclude": []},
        })
        self.assertEqual(rasterize(self.manifest, polygons), 1)
        updated = read_manifest(self.manifest)[0]
        self.assertEqual(updated.skin_status, "sampled")
        self.assertEqual(updated.face_status, "draft")
        self.assertTrue(read_mask(self.root / updated.skin_mask_path, (12, 16)).any())

    def test_cli_split_checks_real_files_before_export(self):
        rows = [replace(self.make_record(i), split="") for i in range(6)]
        write_manifest(self.manifest, rows)
        output = self.root / "split.csv"
        self.assertEqual(main(["split", "--manifest", str(self.manifest), "--output", str(output)]), 0)
        self.assertTrue(validate_dataset(output, "evaluation")["valid"])

    def test_cli_train_exports_valid_model_and_diagnostics(self):
        rows = [self.make_record(1), self.make_record(2)]
        write_manifest(self.manifest, rows)
        output = self.root / "model.json"
        diagnostics = self.root / "diagnostics"
        self.assertEqual(main(["train", "--manifest", str(self.manifest), "--output", str(output),
                               "--diagnostics", str(diagnostics), "--samples-per-image", "10", "--mask-margin", "0"]), 0)
        import json
        check_model(json.loads(output.read_text(encoding="utf-8")))
        self.assertTrue((diagnostics / "cbcr_distribution.png").exists())
        self.assertTrue((diagnostics / "training_summary.json").exists())

    def test_batch_annotations_and_duplicate_ids(self):
        row = replace(self.make_record(1), skin_status="missing", skin_mask_path="")
        write_manifest(self.manifest, [row])
        polygons = self.root / "polygons"
        spec = {"image_id": row.image_id, "annotator": "test", "width": 16, "height": 12,
                "skin": {"status": "sampled", "include": [[[1, 1], [4, 1], [4, 4]]], "exclude": []}}
        write_json(polygons / "batches" / "one.json", {"annotations": [spec]})
        self.assertEqual(rasterize(self.manifest, polygons), 1)
        write_json(polygons / "duplicate.json", spec)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rasterize(self.manifest, polygons)

    def test_colorless_and_review_excluded_samples_do_not_enter_model(self):
        rows = [self.make_record(i) for i in range(1, 5)]
        Image.new("RGB", (16, 12), (100, 100, 100)).save(self.root / rows[2].image_path)
        rows[3] = replace(rows[3], scene_tags="exclude_color_training")
        write_manifest(self.manifest, rows)
        model, _, _ = fit_model(self.manifest)
        self.assertEqual(model["training_info"]["training_image_count"], 2)
        skipped = {item["image_id"] for item in model["training_info"]["skipped"]}
        self.assertEqual(skipped, {rows[2].image_id, rows[3].image_id})


if __name__ == "__main__":
    unittest.main()
