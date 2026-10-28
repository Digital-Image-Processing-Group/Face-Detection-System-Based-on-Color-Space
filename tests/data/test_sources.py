import io
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.training.cli import main
from src.training.dataset import Record, read_manifest, validate_dataset, write_json, write_manifest
from src.training.radiate import collect_radiate, extract_entry, parse_directory


class PublicSourceTests(unittest.TestCase):
    def test_complete_local_sources_resume_offline_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, sources = [], []
            for i in range(44):
                name = f"radiate_{i}"
                image_path = root / f"{name}.png"
                Image.new("RGB", (4, 4), (i, 100, 80)).save(image_path)
                rows.append(Record(name, image_path.name, f"p{i//2}", f"p{i//2}", "train",
                                   "skin.png", "face.png", "fixture", "synthetic:fixture", "fixture",
                                   "test-only", "verified", "complete", "complete"))
                sources.append({"image_id": name, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            write_json(root / "sources/radiate_downloads.json", sources)
            before = manifest.read_bytes()
            with patch("src.training.radiate.byte_range", side_effect=AssertionError("Unexpected network access")):
                report = collect_radiate(manifest)
                self.assertEqual(report["author_participant_groups"], 22)
                self.assertEqual(manifest.read_bytes(), before)
                (root / rows[0].image_path).write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    collect_radiate(manifest)

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
