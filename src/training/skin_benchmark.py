"""Import the university-hosted SkinColor/Pratheepan image/ground-truth pairs.

This is an external mixed-population skin benchmark, not an Asian-only or
identity-disjoint face-region evaluation set.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .dataset import read_bgr, write_json


MIRROR = "https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/_downloads/SkinColor.zip"
SOURCE = "https://cs-chan.com/downloads_skin_dataset.html"
COURSE = "https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/20172018/LabExercises/SkinColorDetection.html"


def decode_ground_truth(content: bytes, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        values = np.asarray(image.convert("RGB"))
    if values.shape[:2] != shape:
        raise ValueError("Ground-truth shape does not match the original photo")
    mask = values[:, :, 0]
    black_or_white = np.all(values == 0, axis=2) | np.all(values == 255, axis=2)
    if np.mean(~black_or_white) > 0.01:
        raise ValueError("More than 1% of ground truth differs from black/white; review the source")
    # Follow the university's published reader exactly: mask[..., 0] == 255.
    # A few legacy PNGs have sparse nonbinary editing noise; do not invent labels
    # by treating every nonzero value as skin.
    return (mask == 255).astype(np.uint8) * 255


def import_skin_benchmark(archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        for name in sorted(names):
            member = PurePosixPath(name)
            if member.parent.name not in ("FacePhoto", "FamilyPhoto") or member.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            target_name = f"GroundT_{member.parent.name}/{member.stem}.png"
            if target_name not in names:
                raise ValueError(f"Missing ground truth for {name}")
            if package.getinfo(name).file_size > 20 * 1024 * 1024 or package.getinfo(target_name).file_size > 20 * 1024 * 1024:
                raise ValueError("Unexpectedly large benchmark member")
            image_id = "prat_" + hashlib.sha256(name.encode()).hexdigest()[:16]
            image_content = package.read(name)
            image_path = output / "images" / f"{image_id}{member.suffix.lower()}"
            mask_path = output / "masks" / f"{image_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_content)
            image = read_bgr(image_path)
            mask = decode_ground_truth(package.read(target_name), image.shape[:2])
            Image.fromarray(mask).save(mask_path)
            rows.append({"image_id": image_id, "image_path": image_path.relative_to(output).as_posix(),
                         "skin_mask_path": mask_path.relative_to(output).as_posix(), "collection": member.parent.name,
                         "original_member": name, "ground_truth_member": target_name, "skin_status": "complete_from_source",
                         "face_status": "not_provided", "population": "mixed_not_asian_only",
                         "identity_grouping": "not_provided", "source_url": SOURCE, "mirror_url": MIRROR,
                         "mask_conversion": "first RGB channel ==255 per university reader; alpha ignored; <=1% sparse non-black/white artifacts allowed",
                         "license": "noncommercial_research_per_author", "image_sha256": hashlib.sha256(image_content).hexdigest(),
                         "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()})
    if not rows:
        raise ValueError("No image/mask pairs found")
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"paired_images": len(rows), "source_url": SOURCE, "mirror_url": MIRROR, "course_url": COURSE,
               "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "purpose": "external_mixed_population_skin_segmentation_only",
               "not_certified": ["Asian-only population", "identity-disjoint test", "full face-region ground truth"]}
    write_json(output / "provenance.json", summary)
    return summary
