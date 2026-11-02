import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from src.app.service import ROOT, detect, export_result, load_photo, load_pipeline, validate_result
from src.evaluation.metrics import counts, metrics, summarize
from src.segmentation import segment_image
from src.segmentation.model_io import validate_model
from src.segmentation.postprocess import _fill_holes_limited, postprocess
from src.segmentation.segment import _validate_config


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = json.loads((ROOT/"models/skin_model.json").read_text(encoding="utf-8"))
        self.config = {"distance_squared_threshold": 4., "model_revision": self.model["model_revision"],
            "postprocess": dict(open_kernel=3, close_kernel=5, min_area_ratio=.001, max_area_ratio=.8,
                min_aspect_ratio=.3, max_aspect_ratio=2., min_fill_ratio=.3,
                min_ellipse_similarity=.3, hole_fill_max_area_ratio=.2)}

    def tearDown(self):
        self.temp.cleanup()

    def photo(self, name="输入照片.png", multiple=True):
        ycrcb = np.zeros((120, 180, 3), dtype=np.uint8)
        ycrcb[:] = [128, 90, 120]
        if multiple:
            cv2.ellipse(ycrcb, (40, 55), (18, 26), 0, 0, 360, (128, 143, 114), -1)
            cv2.ellipse(ycrcb, (130, 55), (18, 26), 0, 0, 360, (128, 143, 114), -1)
        bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        path = self.root/name
        Image.fromarray(bgr[:, :, ::-1]).save(path)
        return path

    def test_real_model_multicandidate_and_exact_export(self):
        image, result, meta = detect(self.photo(), self.model, self.config)
        self.assertEqual(len(result["candidates"]), 2)
        folder = export_result(self.root/"输出结果", image, result, meta)
        self.assertEqual(len(list(folder.iterdir())), 6)
        for kind in ("skin", "face"):
            saved = np.load(folder/(kind+"_mask.npy"), allow_pickle=False)
            np.testing.assert_array_equal(saved, result[kind+"_mask"])
            np.testing.assert_array_equal(np.asarray(Image.open(folder/(kind+"_mask.png"))), saved*255)
        payload = json.loads((folder/"result.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["image_sha256"], meta["image_sha256"])
        self.assertEqual(len(payload["export_sha256"]), 5)
        with self.assertRaises(FileExistsError):
            export_result(folder, image, result, meta)

    def test_export_failure_leaves_no_partial_directory(self):
        image, result, meta = detect(self.photo(), self.model, self.config)
        with patch("src.app.service.np.save", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                export_result(self.root/"failed", image, result, meta)
        self.assertFalse((self.root/"failed").exists())
        self.assertFalse(list(self.root.glob(".export-*")))

    def test_no_candidate_is_successful_empty_export(self):
        image, result, meta = detect(self.photo(multiple=False), self.model, self.config)
        self.assertFalse(result["face_mask"].any())
        self.assertEqual(result["candidates"], [])
        export_result(self.root/"empty", image, result, meta)

    def test_exif_rotation_and_rgb_bgr(self):
        rgb = np.zeros((12, 20, 3), dtype=np.uint8)
        rgb[:6] = [255, 0, 0]
        rgb[6:] = [0, 0, 255]
        exif = Image.Exif()
        exif[274] = 6
        path = self.root/"rotated.png"
        Image.fromarray(rgb).save(path, exif=exif)
        actual = load_photo(path)
        expected_rgb = np.rot90(rgb, -1)
        self.assertEqual(actual.shape, (20, 12, 3))
        np.testing.assert_array_equal(actual[:, :, ::-1], expected_rgb)

    def test_transparency_gray_corrupt_and_wrong_extension_rejected(self):
        for mode, color, name in (("RGBA", (1, 2, 3, 10), "alpha.png"), ("L", 0, "gray.png")):
            path = self.root/name
            Image.new(mode, (10, 10), color).save(path)
            with self.assertRaises(ValueError):
                load_photo(path)
        bad = self.root/"corrupt.jpg"
        bad.write_bytes(b"not a photograph")
        with self.assertRaises(OSError):
            load_photo(bad)
        with self.assertRaises(ValueError):
            load_photo(self.root/"any.gif")

    def test_invalid_covariance_rejected_before_inference(self):
        for cov, lam in (([[1, 2], [0, 1]], 1), ([[1, 2], [2, 1]], 5), ([[0, 0], [0, 0]], 0)):
            bad = dict(self.model, covariance=cov, regularization_lambda=lam)
            with self.assertRaises(ValueError):
                validate_model(bad)

    def test_invalid_config_and_conflicting_thresholds(self):
        for key, value in (("open_kernel", 2), ("min_fill_ratio", float("nan")), ("min_area_ratio", 1.1), ("kernel_reference_side", 0)):
            config = copy.deepcopy(self.config)
            config["postprocess"][key] = value
            with self.assertRaises(ValueError):
                _validate_config(config)
        with self.assertRaises(ValueError):
            _validate_config(dict(self.config, distance_threshold=9))
        with self.assertRaises(ValueError):
            _validate_config(dict(self.config, distance_squared_threshold=-1))

    def test_model_revision_and_hash_binding(self):
        model_path, config_path = self.root/"model.json", self.root/"config.json"
        model_path.write_text(json.dumps(self.model), encoding="utf-8")
        for config in (dict(self.config, model_revision="other"), dict(self.config, model_sha256="bad")):
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pipeline(model_path, config_path)

    def test_bounded_holes_and_candidate_area(self):
        region = np.ones((30, 30), dtype=np.uint8)
        region[5:7, 5:7] = 0
        region[10:20, 10:20] = 0
        filled = _fill_holes_limited(region, 10)
        self.assertEqual(filled[5, 5], 1)
        self.assertEqual(filled[15, 15], 0)
        skin = np.zeros((120, 180), dtype=np.uint8)
        cv2.ellipse(skin, (60, 60), (30, 40), 0, 0, 360, 1, -1)
        skin[50:55, 50:55] = 0
        face, candidates = postprocess(skin, self.config["postprocess"])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["area"], int(face.sum()))

    def test_metrics_known_counts_and_empty_conventions(self):
        pred, ref = np.array([[1, 1, 0, 0]], dtype=np.uint8), np.array([[1, 0, 1, 0]], dtype=np.uint8)
        self.assertEqual(counts(pred, ref), [1, 1, 1])
        self.assertAlmostEqual(metrics([1, 1, 1])["iou"], 1/3)
        self.assertEqual(metrics([1, 1, 1])["dice"], .5)
        self.assertIsNone(metrics([0, 0, 0])["iou"])
        self.assertEqual(metrics([0, 0, 3])["dice"], 0)
        self.assertIsNone(metrics([0, 0, 3])["precision"])
        rows = [{"skin": metrics([1, 1, 1]), "face": metrics([0, 0, 3]), "timing_ms": 1},
                {"skin": metrics([0, 0, 0]), "face": metrics([0, 2, 0]), "timing_ms": 2}]
        s = summarize(rows)
        self.assertEqual(s["face"]["mean_positive_image_iou"], 0)
        self.assertEqual(s["face"]["fp"], 2)
        self.assertEqual(s["no_face"]["image_false_positive_rate"], 1)

    def test_contract_rejects_mask_encoding_or_coordinates(self):
        image = load_photo(self.photo())
        result = segment_image(image, self.model, self.config)
        bad = dict(result, skin_mask=result["skin_mask"]*255)
        with self.assertRaises(ValueError):
            validate_result(bad, image.shape[:2])
        result["candidates"][0]["bbox"]["x"] = 99999
        with self.assertRaises(ValueError):
            validate_result(result, image.shape[:2])

    def test_calibration_does_not_read_test_pixels_or_labels(self):
        from src.evaluation.calibrate import calibrate
        from src.training.dataset import Record, write_manifest, fingerprint
        photo = self.photo()
        bgr = load_photo(photo)
        mask = segment_image(bgr, self.model, self.config)["skin_mask"]*255
        source_dir = self.root/"sources"
        source_dir.mkdir()
        manifest = self.root/"manifest.csv"
        rows, sources = [], []
        for split in ("validation", "test"):
            photo_path = self.root/(split+".png")
            photo_path.write_bytes(photo.read_bytes())
            ref = self.root/(split+"_ref.png")
            Image.fromarray(mask).save(ref)
            rows.append(Record(split, photo_path.name, split, split, split, ref.name, ref.name,
                               "fixture", "synthetic:fixture", "fixture", "test-only", "verified", "complete", "complete"))
            sources.append({"image_id": split, "sha256": fingerprint(photo_path)})
        write_manifest(manifest, rows)
        (source_dir/"hawaii_downloads.json").write_text(json.dumps(sources), encoding="utf-8")
        (source_dir/"bsds_downloads.json").write_text("[]", encoding="utf-8")
        model = copy.deepcopy(self.model)
        model["training_info"]["manifest_sha256"] = fingerprint(manifest)
        model_path = self.root/"model.json"
        model_path.write_text(json.dumps(model), encoding="utf-8")
        before = calibrate(manifest, model_path, self.root/"config", self.root/"cal.json")
        # Corrupt the test files. Selection still succeeds and remains identical.
        (self.root/"test.png").write_bytes(b"unreadable test image")
        (self.root/"test_ref.png").write_bytes(b"unreadable test labels")
        after = calibrate(manifest, model_path, self.root/"config", self.root/"cal.json")
        self.assertEqual(before["selected"], after["selected"])
        self.assertEqual(set(after["input_integrity"]), {"validation"})


if __name__ == "__main__":
    unittest.main()
