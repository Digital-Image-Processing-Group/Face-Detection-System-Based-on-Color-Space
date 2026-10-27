"""Download a bounded FairFace bootstrap subset with source-label provenance."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .dataset import Record, data_path, read_bgr, read_manifest, write_json, write_manifest


DATASET = "HuggingFaceM4/FairFace"
AUTHOR_URL = "https://github.com/joojs/fairface"
LABELS = {0: "East Asian", 6: "Southeast Asian"}


def fetch_bytes(url: str, timeout: int = 45) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only HTTPS download URLs are supported")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "ColorSpaceFaceTraining/1.0"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def collect_fairface(manifest: Path, count: int = 200, pages: int = 16) -> dict:
    if not 1 <= count <= 1000 or not 1 <= pages <= 100:
        raise ValueError("count must be 1..1000 and pages must be 1..100")
    manifest = manifest.resolve()
    root = manifest.parent
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    existing = {row.image_id: row for row in read_manifest(manifest)} if manifest.exists() else {}
    selected = []
    for page in range(pages):
        path = cache / f"fairface_rows_{page * 100}.json"
        # The first exploratory request may have a UTF-8 BOM from PowerShell.
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            query = urlencode({"dataset": DATASET, "config": "0.25", "split": "train", "offset": page * 100, "length": 100})
            payload = json.loads(fetch_bytes("https://datasets-server.huggingface.co/rows?" + query))
            write_json(path, payload)
        names = {item["name"]: item["type"] for item in payload["features"]}
        if any(names["race"]["names"][index] != label for index, label in LABELS.items()):
            raise ValueError("FairFace upstream class mapping changed; review the source before downloading")
        for item in payload["rows"]:
            row = item["row"]
            if row["race"] in LABELS and row["age"] >= 3:
                selected.append((item["row_idx"], row))
                if len(selected) == count:
                    break
        print(f"Metadata page {page + 1}: selected {len(selected)}/{count}", flush=True)
        if len(selected) >= count or len(payload["rows"]) < 100:
            break
    if not selected:
        raise ValueError("No matching adult source-labeled samples found")

    provenance = []
    image_directory = root / "images" / "fairface"
    image_directory.mkdir(parents=True, exist_ok=True)
    targets = {index: data_path(manifest, f"images/fairface/fairface_train_{index:06d}.jpg")
               for index, _ in selected}

    def download(item: tuple[int, dict]) -> tuple[Record, dict]:
        index, source = item
        image_id = f"fairface_train_{index:06d}"
        relative = f"images/fairface/{image_id}.jpg"
        target = targets[index]
        if not target.exists():
            content = fetch_bytes(source["image"]["src"])
            temporary = target.with_suffix(".download")
            temporary.write_bytes(content)
            read_bgr(temporary)  # Decode before accepting the download.
            temporary.replace(target)
        image = read_bgr(target)
        source_url = f"https://huggingface.co/datasets/{DATASET}/viewer/0.25/train?row={index}"
        row = Record(image_id=image_id, image_path=relative, subject_ids="",
                     group_id="fairface_identity_unverified", split="train", skin_mask_path="", face_mask_path="",
                     scene_tags="cropped_face;bootstrap", source_url=source_url,
                     source_label=LABELS[source["race"]], license="CC-BY-4.0",
                     identity_status="unverified", skin_status="missing", face_status="missing")
        return existing.get(image_id, row), {
            "image_id": image_id, "dataset": DATASET, "config": "0.25", "upstream_split": "train",
            "upstream_row_index": index, "source_url": source_url, "author_url": AUTHOR_URL,
            "source_label": LABELS[source["race"]], "label_origin": "upstream_dataset_annotation",
            "license": "CC-BY-4.0", "download_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "width": image.shape[1], "height": image.shape[0], "identity_status": "unverified",
        }

    # Preserve each successful download in the manifest even if a later one fails.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for row, source in executor.map(download, selected):
            existing[row.image_id] = row
            provenance.append(source)
            write_manifest(manifest, sorted(existing.values(), key=lambda record: (not record.image_id.startswith("fairface"), record.image_id)))
    provenance_path = root / "sources" / "fairface_downloads.json"
    old_sources = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else []
    combined = {item["image_id"]: item for item in old_sources + provenance}
    write_json(provenance_path, sorted(combined.values(), key=lambda item: item["image_id"]))
    return {"downloaded_or_reused": len(selected), "requested": count,
            "usage": "bootstrap_only", "manifest": str(manifest),
            "note": "No upstream subject IDs: all samples stay in one training-only group. Not a subject-disjoint benchmark."}


def collect_beans(manifest: Path, count: int = 20) -> dict:
    """Collect candidate negatives; zero masks require subsequent visual review."""
    if not 1 <= count <= 100:
        raise ValueError("count must be 1..100")
    manifest = manifest.resolve()
    root = manifest.parent
    cache = root / "cache" / "beans_rows.json"
    dataset = "AI-Lab-Makerere/beans"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        query = urlencode({"dataset": dataset, "config": "default", "split": "train", "offset": 0, "length": 100})
        payload = json.loads(fetch_bytes("https://datasets-server.huggingface.co/rows?" + query))
        write_json(cache, payload)
    # Spread selection across the page; do not create crops to inflate sample count.
    import numpy as np
    selected = [payload["rows"][index] for index in np.linspace(0, len(payload["rows"]) - 1, count, dtype=int)]
    existing = {row.image_id: row for row in read_manifest(manifest)} if manifest.exists() else {}
    image_directory = root / "images" / "beans"
    image_directory.mkdir(parents=True, exist_ok=True)
    targets = {item["row_idx"]: data_path(manifest, f"images/beans/beans_train_{item['row_idx']:06d}.jpg")
               for item in selected}

    def download(item: dict) -> tuple[Record, dict]:
        index, source = item["row_idx"], item["row"]
        image_id = f"beans_train_{index:06d}"
        relative = f"images/beans/{image_id}.jpg"
        target = targets[index]
        if not target.exists():
            temporary = target.with_suffix(".download")
            temporary.write_bytes(fetch_bytes(source["image"]["src"]))
            read_bgr(temporary)
            temporary.replace(target)
        image = read_bgr(target)
        source_url = f"https://huggingface.co/datasets/{dataset}/viewer/default/train?row={index}"
        row = Record(image_id, relative, "", "beans_acquisition_unverified", "train", "", "",
                     "leaf;candidate_negative;bootstrap", source_url, "", "MIT", "unverified", "missing", "missing")
        return existing.get(image_id, row), {
            "image_id": image_id, "dataset": dataset, "upstream_split": "train", "upstream_row_index": index,
            "source_url": source_url, "author_url": "https://github.com/AI-Lab-Makerere/ibean",
            "license": "MIT", "download_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "width": image.shape[1], "height": image.shape[0],
            "note": "Candidate negative pending image review; no automatic ground-truth masks.",
        }
    provenance = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for row, source in executor.map(download, selected):
            existing[row.image_id] = row
            provenance.append(source)
            # Keep the face subset first so its annotation sheets remain stable.
            write_manifest(manifest, sorted(existing.values(), key=lambda row: (not row.image_id.startswith("fairface"), row.image_id)))
    write_json(root / "sources" / "beans_downloads.json", provenance)
    return {"downloaded_or_reused": len(selected), "usage": "candidate_negatives_pending_review"}
