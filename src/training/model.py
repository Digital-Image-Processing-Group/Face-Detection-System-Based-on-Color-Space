"""Reproducible, per-image balanced Cb/Cr Gaussian fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import (data_path, fingerprint, image_digest, read_bgr, read_manifest,
                      read_mask, validate_dataset, write_json)


COLOR_SPACE = "OpenCV-YCrCb-uint8-full-range"


def check_model(model: dict) -> None:
    if model.get("schema_version") != 1 or model.get("color_space") != COLOR_SPACE:
        raise ValueError("Unsupported model schema or color conversion")
    if model.get("feature_order") != ["Cb", "Cr"]:
        raise ValueError("Model feature_order must be [Cb, Cr]")
    mean = np.asarray(model.get("mean"), dtype=np.float64)
    covariance = np.asarray(model.get("covariance"), dtype=np.float64)
    regularization = model.get("regularization_lambda")
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("Expected a 2-vector mean and 2x2 covariance")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Model parameters must be finite")
    if np.any(mean < 0) or np.any(mean > 255):
        raise ValueError("Mean is outside the uint8 feature range")
    if not isinstance(regularization, (int, float)) or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization_lambda must be finite and nonnegative")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0):
        raise ValueError("Covariance must be symmetric")
    if np.linalg.eigvalsh(covariance).min() < -1e-9:
        raise ValueError("Covariance must be positive semidefinite")
    if np.linalg.eigvalsh(covariance + regularization * np.eye(2)).min() <= 0:
        raise ValueError("Regularized covariance must be positive definite")
    if not isinstance(model.get("model_revision"), str) or not model["model_revision"]:
        raise ValueError("A model_revision is required")


def fit_model(manifest: Path, samples_per_image: int = 2000, seed: int = 42,
              regularization: float = 1.0) -> tuple[dict, np.ndarray, dict]:
    if samples_per_image < 3 or seed < 0:
        raise ValueError("samples_per_image must be >= 3 and seed must be nonnegative")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    report = validate_dataset(manifest, "train")
    if not report["valid"]:
        raise ValueError("Dataset validation failed:\n" + "\n".join(report["errors"]))
    records = sorted(read_manifest(manifest), key=lambda row: row.image_id)
    available = []
    seen = set()
    skipped = []
    for row in records:
        if row.split == "train" and "exclude_color_training" in row.scene_tags.split(";"):
            skipped.append({"image_id": row.image_id, "reason": "explicit visual-review exclusion; see annotation provenance"})
            continue
        if row.split != "train" or row.skin_status == "missing":
            continue
        image = read_bgr(data_path(manifest, row.image_path))
        digest = image_digest(image)
        if digest in seen:
            skipped.append({"image_id": row.image_id, "reason": "duplicate decoded image"})
            continue
        seen.add(digest)
        mask_path = data_path(manifest, row.skin_mask_path)
        mask = read_mask(mask_path, image.shape[:2])
        if mask.any() and np.mean(np.ptp(image[mask.astype(bool)], axis=1) <= 2) >= 0.98:
            skipped.append({"image_id": row.image_id, "reason": "achromatic skin samples carry no reliable color information"})
            continue
        # OpenCV returns Y, Cr, Cb, whereas the model contract is Cb, Cr.
        color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        features = color[:, :, [2, 1]][mask.astype(bool)].astype(np.float64)
        if len(features) == 0:
            skipped.append({"image_id": row.image_id, "reason": "empty skin mask"})
            continue
        if len(features) < 3:
            raise ValueError(f"Too few skin pixels in {row.image_id}: {len(features)}")
        available.append((row, features, digest, fingerprint(mask_path)))
    if len(available) < 2:
        raise ValueError("At least two distinct annotated training images are required")
    # One common count per image guarantees equal contribution without replacement.
    count = min(samples_per_image, min(len(item[1]) for item in available))
    sampled = []
    sources = []
    for row, features, digest, mask_digest in available:
        per_image_seed = int.from_bytes(hashlib.sha256(f"{seed}:{row.image_id}".encode()).digest()[:8], "big")
        generator = np.random.default_rng(per_image_seed)
        sampled.append(features[generator.choice(len(features), count, replace=False)])
        sources.append({"image_id": row.image_id, "image_sha256": digest,
                        "skin_mask_sha256": mask_digest, "available_pixels": len(features),
                        "sampled_pixels": count, "skin_status": row.skin_status,
                        "identity_status": row.identity_status})
    pixels = np.concatenate(sampled)
    mean = pixels.mean(axis=0)
    covariance = np.cov(pixels, rowvar=False, ddof=1)
    bootstrap = any(item[0].identity_status != "verified" for item in available)
    training_info = {
        "usage": "bootstrap_only" if bootstrap else "identity_grouping_verified",
        "skin_annotation_scope": "sampled" if any(item[0].skin_status == "sampled" for item in available) else "complete",
        "training_image_count": len(available), "sampled_pixel_count": len(pixels),
        "requested_samples_per_image": samples_per_image, "actual_samples_per_image": count,
        "random_seed": seed, "manifest_sha256": report["manifest_sha256"],
        "data_revision": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "sources": sources, "skipped": skipped,
    }
    model = {"schema_version": 1, "color_space": COLOR_SPACE, "feature_order": ["Cb", "Cr"],
             "mean": mean.tolist(), "covariance": covariance.tolist(),
             "regularization_lambda": float(regularization), "training_info": training_info}
    canonical = json.dumps(model, sort_keys=True, allow_nan=False).encode()
    model["model_revision"] = "skin-" + hashlib.sha256(canonical).hexdigest()[:16]
    check_model(model)
    return model, pixels, report


def save_diagnostics(model: dict, pixels: np.ndarray, directory: Path) -> None:
    # Headless rendering works on CI and Windows without a GUI session.
    import os
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse

    mean = np.asarray(model["mean"])
    covariance = np.asarray(model["covariance"])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    display = pixels[::max(1, len(pixels) // 15000)]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(display[:, 0], display[:, 1], s=2, alpha=0.12, label="Labeled training skin")
    axis.scatter(*mean, marker="x", color="red", label="Mean")
    axis.add_patch(Ellipse(mean, width=4 * np.sqrt(eigenvalues[0]), height=4 * np.sqrt(eigenvalues[1]),
                           angle=angle, fill=False, color="red", label="2-sigma contour (not a tuned threshold)"))
    axis.set(xlabel="Cb (uint8)", ylabel="Cr (uint8)", xlim=(0, 255), ylim=(0, 255),
             title=f"Cb-Cr training distribution: {model['training_info']['usage']}")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "cbcr_distribution.png", dpi=160)
    plt.close(figure)
    write_json(directory / "training_summary.json", {
        "model_revision": model["model_revision"], "mean": model["mean"],
        "covariance": model["covariance"], "variance": np.diag(covariance).tolist(),
        "standard_deviation": np.sqrt(np.diag(covariance)).tolist(),
        "regularization_lambda": model["regularization_lambda"],
        "training_info": model["training_info"],
    })
