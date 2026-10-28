import io
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.training.cli import main
from src.training.dataset import Record, read_manifest, validate_dataset, write_manifest
from src.training.radiate import extract_entry, parse_directory
from src.training.skin_benchmark import decode_ground_truth, import_skin_benchmark


class PublicSourceTests(unittest.TestCase):
    def test_radiate_directory_and_crc_checked_extraction(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("AF01_NC.jpg", b"original-photo" * 100)
        raw = stream.getvalue()
        central_start = raw.find(b"PK\x01\x02")
        central_end = raw.rfind(b"PK\x05\x06")
        entry = parse_directory(raw[central_start:central_end])[0]
        with patch("src.training.radiate.byte_range", lambda start, end, cache: raw[start:end + 1]):
            self.assertEqual(extract_entry(entry, Path("unused")), b"original-photo" * 100)
            with self.assertRaisesRegex(ValueError, "CRC"):
                extract_entry(dict(entry, crc32=0), Path("unused"))

    def test_benchmark_masks_follow_published_first_channel_rule(self):
        values = np.zeros((20, 20, 4), dtype=np.uint8)
        values[:, :, 3] = 255
        values[1:5, 1:5, :3] = 255
        values[10, 10, :3] = [3, 2, 1]
        content = io.BytesIO()
        Image.fromarray(values).save(content, format="PNG")
        mask = decode_ground_truth(content.getvalue(), (20, 20))
        self.assertEqual(np.count_nonzero(mask), 16)
        self.assertEqual(mask[10, 10], 0)
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_ground_truth(content.getvalue(), (21, 20))

    def test_benchmark_import_pairs_and_preserves_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo, mask = io.BytesIO(), io.BytesIO()
            Image.new("RGB", (20, 20), (170, 110, 80)).save(photo, format="JPEG")
            Image.new("L", (20, 20), 255).save(mask, format="PNG")
            source = root / "source.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("FacePhoto/example.jpg", photo.getvalue())
                archive.writestr("GroundT_FacePhoto/example.png", mask.getvalue())
            report = import_skin_benchmark(source, root / "output")
            self.assertEqual(report["paired_images"], 1)
            self.assertIn("Asian-only population", report["not_certified"])
            self.assertTrue((root / "output" / "manifest.csv").exists())

    def test_prepare_split_does_not_certify_missing_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(6):
                image_id = f"source_{index}"
                values = np.random.default_rng(index).integers(0, 255, (12, 16, 3), dtype=np.uint8)
                Image.fromarray(values).save(root / f"{image_id}.png")
                rows.append(Record(image_id, f"{image_id}.png", image_id, image_id, "", "", "", "fixture",
                                   "synthetic:fixture", "author-label-fixture", "test-only", "verified", "missing", "missing"))
            manifest = root / "unsplit.csv"
            output = root / "split.csv"
            write_manifest(manifest, rows)
            self.assertEqual(main(["split", "--manifest", str(manifest), "--output", str(output), "--prepare-only"]), 0)
            self.assertEqual({row.split for row in read_manifest(output)}, {"train", "validation", "test"})
            self.assertFalse(validate_dataset(output, "evaluation")["valid"])


if __name__ == "__main__":
    unittest.main()
