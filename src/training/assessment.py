"""Validate the statistical model without using held-out people in fitting.

The filled largest-component face proposal is an explicitly limited baseline
for these single-portrait images, rather than a multi-face product detector.
"""

from __future__ import annotations

import hashlib
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .dataset import (data_path, fingerprint, image_digest, read_bgr, read_manifest,
                      read_mask, validate_dataset, write_json)
from .model import check_model


def distance_map(image: np.ndarray, model: dict) -> np.ndarray:
    color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    features = color[:, :, [2, 1]].astype(np.float32)
    features -= np.asarray(model["mean"], dtype=np.float32)
    covariance = np.asarray(model["covariance"], dtype=np.float64)
    inverse = np.linalg.inv(covariance + model["regularization_lambda"] * np.eye(2))
    return np.einsum("...i,ij,...j->...", features, inverse, features, optimize=True).astype(np.float32)


def confusion(prediction: np.ndarray, truth: np.ndarray) -> tuple[int, int, int]:
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    return (int(np.count_nonzero(prediction & truth)), int(np.count_nonzero(prediction & ~truth)),
            int(np.count_nonzero(~prediction & truth)))


def metrics(counts: tuple[int, int, int] | list[int]) -> dict:
    tp, fp, fn = map(int, counts)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def face_proposal(skin: np.ndarray) -> np.ndarray:
    size = max(3, int(round(min(skin.shape) * 0.004)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    cleaned = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if count <= 1:
        return np.zeros_like(cleaned)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == best).astype(np.uint8)
    padded = np.pad(component, 1)
    exterior = padded.copy()
    cv2.floodFill(exterior, None, (0, 0), 2)
    return ((padded == 1) | (exterior == 0))[1:-1, 1:-1].astype(np.uint8)


def assess(manifest: Path, model: dict, output: Path, operating_point: Path, previews: Path) -> dict:
    check_model(model)
    validation = validate_dataset(manifest, "evaluation")
    if not validation["valid"]:
        raise ValueError("Complete verified annotations are required for assessment")
    if model["training_info"]["manifest_sha256"] != validation["manifest_sha256"]:
        raise ValueError("Model and dataset manifest revisions do not match")
    records = read_manifest(manifest)
    if {row.split for row in records} != {"train", "validation", "test"}:
        raise ValueError("Assessment requires nonempty train, validation and test splits")
    indexed = {row.image_id: row for row in records}
    training = model["training_info"]
    fitted_ids = {item["image_id"] for item in training["sources"]}
    skipped_ids = {item["image_id"] for item in training["skipped"]}
    if fitted_ids | skipped_ids != {row.image_id for row in records if row.split == "train"}:
        raise ValueError("Training image membership does not match the fitted model")
    for source in training["sources"]:
        row = indexed[source["image_id"]]
        if (image_digest(read_bgr(data_path(manifest, row.image_path))) != source["image_sha256"]
                or fingerprint(data_path(manifest, row.skin_mask_path)) != source["skin_mask_sha256"]):
            raise ValueError(f"Training files changed since fitting: {row.image_id}")
    # Fixed candidate grid; selection uses validation skin Dice only.
    candidates = np.arange(0.5, 20.01, 0.25)
    positive = np.zeros(len(candidates) + 1, dtype=np.int64)
    negative = np.zeros_like(positive)
    # Upper histogram bounds must include equality, like the <= prediction rule.
    bins = np.r_[0, np.nextafter(candidates, np.inf), np.inf]
    for row in records:
        if row.split != "validation":
            continue
        image = read_bgr(data_path(manifest, row.image_path))
        distance = distance_map(image, model)
        truth = read_mask(data_path(manifest, row.skin_mask_path), image.shape[:2]).astype(bool)
        positive += np.histogram(distance[truth], bins=bins)[0]
        negative += np.histogram(distance[~truth], bins=bins)[0]
    tp = np.cumsum(positive)[:-1]
    fp = np.cumsum(negative)[:-1]
    fn = positive.sum() - tp
    dice = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    selected = int(np.argmax(dice))  # Smaller threshold wins ties.
    threshold = float(candidates[selected])
    point = {"schema_version": 1, "model_revision": model["model_revision"],
             "manifest_sha256": validation["manifest_sha256"], "distance_squared_threshold": threshold,
             "selection": "validation_skin_micro_dice", "candidate_grid": {"min": 0.5, "max": 20.0, "step": 0.25},
             "face_baseline": "fixed scale-aware opening/closing, largest component and hole fill for single portraits"}
    # Save the operating point before reading any test image for performance evaluation.
    write_json(operating_point, point)
    results = []
    totals = {split: {kind: [0, 0, 0] for kind in ("skin", "face_baseline")} for split in ("validation", "test")}
    previews.mkdir(parents=True, exist_ok=True)
    for row in records:
        if row.split == "train":
            continue
        image = read_bgr(data_path(manifest, row.image_path))
        started = time.perf_counter()
        skin = (distance_map(image, model) <= threshold).astype(np.uint8)
        face = face_proposal(skin)
        elapsed = (time.perf_counter() - started) * 1000
        item = {"image_id": row.image_id, "subject_ids": sorted(row.subjects), "split": row.split,
                "shape": list(skin.shape), "processing_ms": elapsed,
                "scene_tags": row.scene_tags.split(';'), "source_population_label":row.source_label}
        for kind, prediction, field in (("skin", skin, "skin_mask_path"), ("face_baseline", face, "face_mask_path")):
            truth = read_mask(data_path(manifest, getattr(row, field)), image.shape[:2])
            counts = confusion(prediction, truth)
            item[kind] = metrics(counts)
            totals[row.split][kind] = [a + b for a, b in zip(totals[row.split][kind], counts)]
            Image.fromarray(prediction * 255).save(previews / f"{row.image_id}_{kind}.png")
        results.append(item)
    summary = {split: {kind: metrics(value) for kind, value in kinds.items()} for split, kinds in totals.items()}
    for split in summary:
        for kind in summary[split]:
            values = [item[kind]["iou"] for item in results if item["split"] == split and item[kind]["iou"] is not None]
            summary[split][kind]["mean_image_iou"] = float(np.mean(values)) if values else None
            positive_values=[item[kind]['iou'] for item in results if item['split']==split
                             and item[kind]['tp']+item[kind]['fn']>0]
            summary[split][kind]['mean_positive_image_iou']=float(np.mean(positive_values)) if positive_values else None
    no_face={}
    scene_summary={}
    for split in ('validation','test'):
        absent=[item for item in results if item['split']==split and item['face_baseline']['tp']+item['face_baseline']['fn']==0]
        false_images=sum(item['face_baseline']['fp']>0 for item in absent)
        no_face[split]={'images':len(absent),'nonempty_predictions':false_images,
                        'image_false_positive_rate':false_images/len(absent) if absent else None,
                        'false_positive_pixels':sum(item['face_baseline']['fp'] for item in absent)}
        scene_summary[split]={}
        for scene in ('frontal','profile','natural_background'):
            subset=[item for item in results if item['split']==split and scene in item['scene_tags']]
            if not subset:
                continue
            scene_summary[split][scene]={'images':len(subset)}
            for kind in ('skin','face_baseline'):
                counts=[sum(item[kind][key] for item in subset) for key in ('tp','fp','fn')]
                value=metrics(counts)
                positives=[item[kind]['iou'] for item in subset if item[kind]['tp']+item[kind]['fn']>0]
                value['mean_positive_image_iou']=float(np.mean(positives)) if positives else None
                scene_summary[split][scene][kind]=value
    report = {"model_revision": model["model_revision"], "operating_point": point,
              "dataset": {"manifest_sha256": validation["manifest_sha256"], "images": len(records),
                          "group_count": len({row.group_id for row in records}),
                          "participant_groups": len({s for row in records for s in row.subjects}),
                          "background_scene_groups": len({row.group_id for row in records if not row.subjects}),
                          "splits": validation["splits"], "skin_status": validation["skin_status"],
                          "face_status": validation["face_status"], "identity_status": validation["identity_status"],
                          "validation_errors": validation["errors"], "validation_warnings": validation["warnings"]},
              "annotation_reference": "Project silver-standard references: independent part-parsing initialization, visual review and recorded corrections. Background negatives visually checked. Not source-provided or independently expert-reviewed pixel truth; scores measure agreement with these references.",
              "scope": "Author-labeled Asian single-person portraits with frontal expressions and profiles; person-free natural background negatives. No real multi-person or unconstrained positive-scene performance claim.",
              "summary": summary, "no_face":no_face,"scene_summary":scene_summary,"per_image": results,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "opencv": cv2.__version__, "platform": platform.platform(),
                              "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())},
              "timing": {"scope": "distance classification and face baseline only; excludes decoding, truth loading and export",
                         "samples": len(results), "median_ms": float(np.median([item["processing_ms"] for item in results])),
                         "p95_ms": float(np.percentile([item["processing_ms"] for item in results], 95))},
              "annotation_integrity": {row.image_id: {kind: fingerprint(data_path(manifest, getattr(row, f"{kind}_mask_path")))
                                                      for kind in ("skin", "face")} for row in records},
              "test_file_integrity": {row.image_id: hashlib.sha256(data_path(manifest, row.image_path).read_bytes()).hexdigest()
                                      for row in records if row.split == "test"}}
    write_json(output, report)
    return report
