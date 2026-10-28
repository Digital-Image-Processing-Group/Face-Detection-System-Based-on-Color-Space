import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.training.cli import main
from src.training.dataset import Record, read_manifest, validate_dataset, write_json, write_manifest


class PublicSourceTests(unittest.TestCase):
    def test_restore_preserves_frozen_annotations_and_refuses_corrupt_images(self):
        from src.training.public_sources import restore
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/'original.png'
            Image.new('RGB',(4,4),(170,110,90)).save(path)
            row=Record('hfd_fixture',path.name,'participant_01','participant_01','test',
                       'skin.png','face.png','fixture','synthetic:fixture','Asian fixture',
                       'test-only','verified','complete','complete')
            manifest=root/'manifest.csv';write_manifest(manifest,[row])
            source={'image_id':row.image_id,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            write_json(root/'sources/hawaii_downloads.json',[source])
            write_json(root/'sources/bsds_downloads.json',[])
            before=manifest.read_bytes()
            with patch('src.training.public_sources.fetch',side_effect=AssertionError('Unexpected network')):
                self.assertFalse(restore(manifest)['network_used'])
                self.assertEqual(manifest.read_bytes(),before)
                path.write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    restore(manifest)

    def test_restore_checks_author_hash_before_installing_a_download(self):
        from src.training.public_sources import restore
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=root/'manifest.csv'
            row=Record('hfd_fixture','images/photo.jpg','p','p','train','','','fixture',
                       'synthetic:fixture','Asian fixture','test-only','verified','missing','missing')
            write_manifest(manifest,[row])
            source={'image_id':row.image_id,'sha256':'0'*64,'author_sha256':'0'*64,'download_url':'https://invalid.example/not-called'}
            write_json(root/'sources/hawaii_downloads.json',[source]);write_json(root/'sources/bsds_downloads.json',[])
            with patch('src.training.public_sources.fetch',return_value=b'wrong bytes'):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    restore(manifest)
            self.assertFalse((root/row.image_path).exists())

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
