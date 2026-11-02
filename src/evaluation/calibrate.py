"""Validation-only finite search; frozen configs are written before evaluation."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json

import cv2
import numpy as np

from src.app.service import digest_json
from src.training.dataset import fingerprint, write_json
from src.segmentation.model_io import validate_model, get_cb_cr
from src.segmentation.postprocess import postprocess
from .data import main_samples
from .metrics import counts, metrics

THRESHOLDS = [2., 4., 6., 9., 12., 20.]


def profiles():
    base = dict(open_kernel=3, close_kernel=5, min_area_ratio=.001, max_area_ratio=.5,
                min_aspect_ratio=.5, max_aspect_ratio=2., min_fill_ratio=.4,
                min_ellipse_similarity=.5, hole_fill_max_area_ratio=.3)
    return [base, dict(base, kernel_reference_side=1000, open_kernel=5, close_kernel=5, max_area_ratio=.85,
                       min_aspect_ratio=.3, min_fill_ratio=.3, min_ellipse_similarity=.3, hole_fill_max_area_ratio=.15),
            dict(base, kernel_reference_side=1000, open_kernel=5, close_kernel=9, max_area_ratio=.85,
                 min_aspect_ratio=.3, min_fill_ratio=.3, min_ellipse_similarity=.3, hole_fill_max_area_ratio=.3)]


def calibrate(manifest, model_path, output_dir, report_path):
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    validated = validate_model(model)
    if model["training_info"]["manifest_sha256"] != fingerprint(Path(manifest)):
        raise ValueError("模型与数据清单版本不匹配")
    samples = main_samples(manifest, "validation")
    if not samples:
        raise ValueError("没有验证样本")
    trials = {mode: [{"distance_squared_threshold": t, "postprocess": pp, "skin_counts": [0, 0, 0], "face_counts": [0, 0, 0]}
                     for t in THRESHOLDS for pp in profiles()] for mode in ("full", "diagonal")}
    integrity = {}
    for index, sample in enumerate(samples, 1):
        image, skin_ref, face_ref = sample.load()
        integrity[sample.image_id] = {"photo": fingerprint(sample.photo), "skin": fingerprint(sample.skin_path), "face": fingerprint(sample.face_path)}
        cb, cr = get_cb_cr(image)
        delta = np.stack([cb-validated["mean"][0], cr-validated["mean"][1]], axis=-1)
        for mode in trials:
            cov = validated["covariance"]
            if mode == "diagonal":
                cov = np.diag(np.diag(cov))
            inv = np.linalg.inv(cov+validated["regularization_lambda"]*np.eye(2))
            d2 = np.einsum("...i,ij,...j->...", delta, inv, delta)
            for t in THRESHOLDS:
                skin = (d2 <= t).astype(np.uint8)
                skin_counts = counts(skin, skin_ref)
                for trial in (trial for trial in trials[mode] if trial["distance_squared_threshold"] == t):
                    face, _ = postprocess(skin, trial["postprocess"])
                    trial["skin_counts"] = [a+b for a, b in zip(trial["skin_counts"], skin_counts)]
                    trial["face_counts"] = [a+b for a, b in zip(trial["face_counts"], counts(face, face_ref))]
        if index % 5 == 0 or index == len(samples):
            print(f"Calibration validation only: {index}/{len(samples)}", flush=True)
    configs = {}
    for mode, items in trials.items():
        for item in items:
            item["skin"] = metrics(item.pop("skin_counts"))
            item["face"] = metrics(item.pop("face_counts"))
        # Stable enumeration: smaller threshold and earlier profile win exact ties.
        best = max(items, key=lambda item: item["face"]["iou"] or 0)
        config = {"schema_version": 1, "algorithm_revision": "integrated-v1",
                  "model_revision": model["model_revision"], "model_sha256": fingerprint(Path(model_path)),
                  "manifest_sha256": fingerprint(Path(manifest)), "covariance_mode": mode,
                  "distance_squared_threshold": best["distance_squared_threshold"], "postprocess": deepcopy(best["postprocess"]),
                  "selection": "validation_face_micro_iou_including_background_false_positives",
                  "calibration_inputs_sha256": digest_json(integrity)}
        config["config_revision"] = digest_json(config)[:16]
        configs[mode] = config
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "selection_split": "validation",
              "test_references_used_for_selection": False, "validation_images": len(samples),
              "objective": "maximize validation accumulated face IoU; no-face FP pixels included; first enumerated choice wins ties",
              "thresholds": THRESHOLDS, "profiles": profiles(), "trials": trials, "selected": configs,
              "input_integrity": integrity,
              "target_subset_predeclared": {"split": "test", "tag": "frontal", "target_mean_face_iou": .65,
                  "limitation": "Controlled frontal subset only; no verified small-angle subset exists. Previously reported A test references: exploratory integration evaluation."}}
    output_dir = Path(output_dir)
    write_json(output_dir/"pipeline.json", configs["full"])
    write_json(output_dir/"baseline.json", configs["diagonal"])
    write_json(Path(report_path), report)
    return report
