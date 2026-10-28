"""Manifest contract, orientation-aware IO, annotation checks and group splits."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SPLITS = ("train", "validation", "test")
SKIN_STATUSES = ("missing", "sampled", "complete")
FACE_STATUSES = ("missing", "draft", "complete")


@dataclass(frozen=True)
class Record:
    image_id: str
    image_path: str
    subject_ids: str
    group_id: str
    split: str
    skin_mask_path: str
    face_mask_path: str
    scene_tags: str
    source_url: str
    source_label: str
    license: str
    identity_status: str
    skin_status: str
    face_status: str

    @property
    def subjects(self) -> set[str]:
        return {part.strip() for part in self.subject_ids.split(";") if part.strip()}


FIELDS = tuple(Record.__dataclass_fields__)


def read_manifest(path: Path) -> list[Record]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Manifest header must be: {','.join(FIELDS)}")
        result = []
        seen = set()
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV at line {line}")
            row = {key: value.strip() for key, value in row.items()}
            if not row["image_id"] or row["image_id"] in seen:
                raise ValueError(f"Empty or duplicate image_id at line {line}")
            if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in row["image_id"]):
                raise ValueError(f"Unsafe image_id at line {line}")
            seen.add(row["image_id"])
            result.append(Record(**row))
    if not result:
        raise ValueError("Manifest contains no records")
    return result


def write_manifest(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)


def data_path(manifest: Path, relative: str) -> Path:
    if not relative:
        raise ValueError("Missing file path")
    root = manifest.resolve().parent
    result = (root / relative).resolve()
    if Path(relative).is_absolute() or not result.is_relative_to(root):
        raise ValueError(f"Manifest paths must stay relative to {root}: {relative} (resolved as {result})")
    return result


def read_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "RGBA", "P", "CMYK"):
            raise ValueError(f"Expected a color image, got mode {image.mode}: {path}")
        if image.mode == "RGBA" or "transparency" in image.info:
            if image.convert("RGBA").getchannel("A").getextrema()[0] < 255:
                raise ValueError(f"Transparent images are unsupported: {path}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
        allowed = (0, 1)
    elif path.suffix.lower() == ".png":
        with Image.open(path) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"Masks must already have upright coordinates: {path}")
            mask = np.asarray(image)
        allowed = (0, 1, 255)
    else:
        raise ValueError(f"Masks must be PNG or NPY: {path}")
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} != image shape {shape}: {path}")
    values = set(np.unique(mask).tolist())
    if mask.dtype.kind not in "bui" or not values.issubset(allowed):
        raise ValueError(f"Mask must be a binary integer array: {path}")
    if 1 in values and 255 in values:
        raise ValueError(f"Do not mix 0/1 and 0/255 encodings: {path}")
    return (mask != 0).astype(np.uint8)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_digest(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def connected_groups(records: list[Record]) -> list[list[int]]:
    """Transitive closure over subject IDs and acquisition/scene group IDs."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners: dict[str, int] = {}
    for index, row in enumerate(records):
        tokens = [f"subject:{value}" for value in sorted(row.subjects)]
        if row.group_id:
            tokens.append(f"group:{row.group_id}")
        for token in tokens:
            if token in owners:
                parent[find(index)] = find(owners[token])
            else:
                owners[token] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=lambda group: records[group[0]].image_id)


def assign_splits(records: list[Record], seed: int = 42,
                  ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> list[Record]:
    if len(ratios) != 3 or not np.isfinite(ratios).all() or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Split ratios must be three positive numbers summing to 1")
    if any(row.split for row in records):
        raise ValueError("Refusing to reassign existing splits; create a separate unsplit manifest explicitly")
    if any(row.identity_status != "verified" for row in records):
        raise ValueError("Identity grouping is unverified; row IDs are not verified subject IDs")
    if any(not row.group_id for row in records):
        raise ValueError("Every record needs an acquisition/scene group_id")
    if any(not row.subjects and row.source_label for row in records):
        raise ValueError("Verified subject IDs are required for face sources before splitting")
    groups = connected_groups(records)
    if len(groups) < 3:
        raise ValueError("At least three independent groups are needed for train/validation/test")
    generator = np.random.default_rng(seed)
    generator.shuffle(groups)
    groups.sort(key=len, reverse=True)
    counts = np.zeros(3, dtype=int)
    targets = np.array(ratios) * len(records)
    assigned = {}
    for number, group in enumerate(groups):
        empty = np.flatnonzero(counts == 0)
        if len(groups) - number == len(empty):
            chosen = int(empty[0])
        else:
            chosen = int(np.argmax(targets - counts))
        for index in group:
            assigned[index] = SPLITS[chosen]
        counts[chosen] += len(group)
    return [Record(**(asdict(row) | {"split": assigned[index]})) for index, row in enumerate(records)]


def validate_dataset(manifest: Path, purpose: str = "train") -> dict:
    records = read_manifest(manifest)
    errors: list[str] = []
    warnings: list[str] = []
    hashes: dict[str, Record] = {}
    perceptual: list[tuple[Record, np.ndarray, np.ndarray]] = []
    samples = 0
    for row in records:
        try:
            if row.split not in SPLITS:
                raise ValueError("split must be train, validation or test")
            if not row.group_id or row.identity_status not in ("verified", "unverified"):
                raise ValueError("group_id and valid identity_status are required")
            if row.skin_status not in SKIN_STATUSES or row.face_status not in FACE_STATUSES:
                raise ValueError("Unknown annotation status")
            if not row.source_url or not row.license:
                raise ValueError("source_url and license are required")
            image = read_bgr(data_path(manifest, row.image_path))
            shape = image.shape[:2]
            digest = image_digest(image)
            if digest in hashes:
                other = hashes[digest]
                if row.split != other.split:
                    raise ValueError(f"Identical decoded image crosses splits with {other.image_id}")
                warnings.append(f"{row.image_id}: duplicate decoded image of {other.image_id}")
            hashes[digest] = row
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            dhash = (small[:, 1:] > small[:, :-1]).ravel()
            thumbnail = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            for other, other_hash, other_thumbnail in perceptual:
                if other.split != row.split and np.count_nonzero(dhash != other_hash) <= 4:
                    difference = np.abs(thumbnail - other_thumbnail)
                    if difference.mean() <= 1.5 and np.percentile(difference, 90) <= 4:
                        raise ValueError(f"Potential near-duplicate across splits with {other.image_id}; review/group them")
                    warnings.append(f"{row.image_id}: coarse hash resembles {other.image_id}, but RGB pixels differ; hash alone is not evidence of duplicate subjects")
            perceptual.append((row, dhash, thumbnail))
            skin = None
            face = None
            if row.skin_status != "missing":
                skin = read_mask(data_path(manifest, row.skin_mask_path), shape)
            elif row.skin_mask_path:
                raise ValueError("skin_status is missing but a skin mask path is set")
            if row.face_status != "missing":
                face = read_mask(data_path(manifest, row.face_mask_path), shape)
                if face.any() and not row.subjects and row.identity_status == "verified":
                    raise ValueError("Face-positive record needs subject_ids for verified grouping")
            elif row.face_mask_path:
                raise ValueError("face_status is missing but a face mask path is set")
            if purpose == "evaluation":
                if row.identity_status != "verified":
                    raise ValueError("Formal evaluation requires verified identity grouping")
                if row.skin_status != "complete" or row.face_status != "complete":
                    raise ValueError("Formal evaluation requires complete skin and face annotations")
            if row.identity_status == "unverified":
                warnings.append(f"{row.image_id}: identity grouping unverified; bootstrap use only")
                if row.split != "train":
                    raise ValueError("Unverified identities must stay in bootstrap training, not validation/test")
            if row.split == "train" and skin is not None:
                samples += int(skin.sum())
            if row.split == "train" and row.skin_status == "missing":
                warnings.append(f"{row.image_id}: no skin annotation; trainer will skip")
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{row.image_id}: {error}")
    for group in connected_groups(records):
        splits = {records[index].split for index in group}
        if len(splits) > 1:
            errors.append(f"Subject/acquisition group crosses splits: {[records[index].image_id for index in group]}")
    return {
        "valid": not errors, "purpose": purpose, "records": len(records),
        "splits": dict(Counter(row.split for row in records)),
        "skin_status": dict(Counter(row.skin_status for row in records)),
        "face_status": dict(Counter(row.face_status for row in records)),
        "identity_status": dict(Counter(row.identity_status for row in records)),
        "training_skin_pixels": samples, "errors": errors, "warnings": warnings,
        "manifest_sha256": fingerprint(manifest),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
