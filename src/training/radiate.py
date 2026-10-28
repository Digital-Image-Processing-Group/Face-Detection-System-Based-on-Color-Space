"""Acquire author-labeled RADIATE color photos using bounded ZIP byte ranges."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from .dataset import Record, data_path, read_bgr, read_manifest, write_json, write_manifest

SOURCE = "https://abcdstudy.org/scientists/abcd-fmri-tasks-and-tools/"
ARCHIVE = "https://abcdstudy.org/newsletter/fMRI_files/RADIATE_JPEGS1.zip"
ARCHIVE_SIZE = 1473485405
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6446554/"


def byte_range(start: int, end: int, cache: Path) -> bytes:
    if start < 0 or end < start or end - start > 8 * 1024 * 1024:
        raise ValueError("Invalid or oversized archive byte range")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{start}-{end}.bin"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return path.read_bytes()
    request = Request(ARCHIVE, headers={"Range": f"bytes={start}-{end}", "User-Agent": "ColorSpaceTraining/1.0"})
    with urlopen(request, timeout=20) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
            raise ValueError("RADIATE archive changed or server ignored the bounded range request")
        content = response.read(expected + 1)
    if len(content) != expected:
        raise ValueError("Archive response length mismatch")
    path.write_bytes(content)
    return content


def parse_directory(content: bytes) -> list[dict]:
    entries, position = [], 0
    while position < len(content):
        fields = struct.unpack_from("<4s6H3L5H2L", content, position)
        if fields[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central directory")
        name_size, extra_size, comment_size = fields[10:13]
        name = content[position + 46:position + 46 + name_size].decode("utf-8")
        entries.append({"name": name, "compression": fields[4], "crc32": fields[7],
                        "compressed_size": fields[8], "size": fields[9], "offset": fields[16]})
        position += 46 + name_size + extra_size + comment_size
    return entries


def extract_entry(entry: dict, cache: Path) -> bytes:
    if not 0 < entry["size"] <= 20 * 1024 * 1024 or not 0 < entry["compressed_size"] <= 8 * 1024 * 1024:
        raise ValueError("Unexpected ZIP image size")
    offset = entry["offset"]
    header = byte_range(offset, offset + 29, cache)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local file header")
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    start = offset + 30 + name_size + extra_size
    packed = byte_range(start, start + entry["compressed_size"] - 1, cache)
    if entry["compression"] == 0:
        content = packed
    elif entry["compression"] == 8:
        decoder = zlib.decompressobj(-15)
        content = decoder.decompress(packed, entry["size"] + 1)
        if not decoder.eof:
            raise ValueError("Incomplete or oversized ZIP member")
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(content) != entry["size"] or zlib.crc32(content) != entry["crc32"]:
        raise ValueError("Extracted image failed ZIP size/CRC validation")
    return content


def collect_radiate(manifest: Path) -> dict:
    manifest = manifest.resolve()
    root = manifest.parent
    provenance_path = root / "sources" / "radiate_downloads.json"
    existing = {row.image_id: row for row in read_manifest(manifest)} if manifest.exists() else {}
    provenance = {item["image_id"]: item for item in json.loads(provenance_path.read_text(encoding="utf-8"))} if provenance_path.exists() else {}
    if existing and any(not name.startswith("radiate_") for name in existing):
        raise ValueError("Only the RADIATE dataset is supported in this manifest")
    # Reuse a complete local delivery offline, preserving split and annotation fields.
    if len(existing) == 44 and existing.keys() == provenance.keys():
        complete = True
        for image_id, row in existing.items():
            target = data_path(manifest, row.image_path)
            if not target.exists():
                complete = False
                continue
            if hashlib.sha256(target.read_bytes()).hexdigest() != provenance[image_id]["sha256"]:
                raise ValueError(f"Local source image hash mismatch: {image_id}")
        if complete:
            return {"photos": 44, "author_participant_groups": len({row.group_id for row in existing.values()}),
                    "manifest": str(manifest), "status": "verified_local_files_reused"}
    cache = root / "cache" / "radiate_ranges"
    tail = byte_range(ARCHIVE_SIZE - 65536, ARCHIVE_SIZE - 1, cache)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0:
        raise ValueError("ZIP end record not found")
    end = struct.unpack_from("<4s4H2LH", tail, marker)
    directory = parse_directory(byte_range(end[6], end[6] + end[5] - 1, cache))
    pattern = re.compile(r"^RADIATE_JPEGS1/A[FM]_Color_JPEG/(A[FM]\d+)_Color/\1_(NC|HC)\.jpg$")
    selected = [entry for entry in directory if pattern.fullmatch(entry["name"])]
    subjects = {Path(entry["name"]).stem.split("_")[0] for entry in selected}
    if len(subjects) != 22 or len(selected) != 44:
        raise ValueError(f"Expected 22 author-labeled Asian participant codes and 44 photos, got {len(subjects)}, {len(selected)}")
    (root / "images" / "radiate").mkdir(parents=True, exist_ok=True)
    targets = {entry["name"]: data_path(manifest, f"images/radiate/{Path(entry['name']).stem}.jpg") for entry in selected}

    def download(entry):
        stem = Path(entry["name"]).stem
        subject = stem.split("_")[0]
        image_id = "radiate_" + stem
        target = targets[entry["name"]]
        if not target.exists():
            target.write_bytes(extract_entry(entry, cache))
        if image_id in provenance and hashlib.sha256(target.read_bytes()).hexdigest() != provenance[image_id]["sha256"]:
            raise ValueError(f"Source image hash mismatch: {image_id}")
        image = read_bgr(target)
        row = Record(image_id, f"images/radiate/{stem}.jpg", "radiate_" + subject, "radiate_" + subject,
                     "", "", "", "controlled_portrait;author_color_version", SOURCE,
                     "Asian (author AF/AM group)", "RADIATE-research-and-education-only", "verified", "missing", "missing")
        return existing.get(image_id, row), {
            "image_id": image_id, "source_url": SOURCE, "paper_url": PAPER, "archive_url": ARCHIVE,
            "archive_size": ARCHIVE_SIZE, "archive_member": entry["name"], "author_participant_code": subject,
            "grouping_basis": "anonymous_author_participant_code_not_inferred_from_image",
            "source_group": "Asian (author AF/AM folders)", "variant": "Color", "expression": stem.split("_")[1],
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "width": image.shape[1], "height": image.shape[0],
            "terms": "Research/education only; do not identify depicted individuals. Images are kept local, not redistributed.",
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        for number, (row, source) in enumerate(executor.map(download, selected), 1):
            existing[row.image_id] = row
            provenance[row.image_id] = source
            write_manifest(manifest, sorted(existing.values(), key=lambda item: item.image_id))
            write_json(provenance_path, sorted(provenance.values(), key=lambda item: item["image_id"]))
            print(f"RADIATE color photos: {number}/{len(selected)}", flush=True)
    return {"photos": len(selected), "author_participant_groups": len(subjects), "manifest": str(manifest),
            "annotations": "Source supplies no masks; existing project annotations are preserved", "status": "downloaded_with_verified_author_grouping"}
