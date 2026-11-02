"""Shared GUI/CLI boundary: orientation, contracts, provenance and atomic exports."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.training.dataset import read_bgr, fingerprint
from src.segmentation import segment_image, segment_image_baseline
from src.segmentation.model_io import validate_model
from src.segmentation.segment import _validate_config

ROOT = Path(__file__).resolve().parents[2]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_pipeline(model_path=ROOT / "models/skin_model.json", config_path=ROOT / "config/pipeline.json"):
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_model(model)
    _validate_config(config)
    if model["model_revision"] != config["model_revision"]:
        raise ValueError("模型与配置版本不匹配，请选择配套文件")
    if "model_sha256" in config and config["model_sha256"] != fingerprint(Path(model_path)):
        raise ValueError("模型文件哈希与冻结配置不匹配")
    return model, config


def load_photo(path):
    path = Path(path)
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("请选择 JPG、JPEG 或 PNG 彩色照片")
    return read_bgr(path)


def validate_result(result, shape):
    for key in ("skin_mask", "face_mask"):
        mask = result[key]
        if mask.dtype != np.uint8 or mask.shape != shape or not np.isin(mask, (0, 1)).all():
            raise ValueError(key + " 必须是原尺寸 uint8 0/1 掩膜")
    if not np.isfinite(result["timing_ms"]) or result["timing_ms"] < 0:
        raise ValueError("分割耗时不合法")
    for item in result["candidates"]:
        b = item["bbox"]
        x, y, w, h = (b[k] for k in ("x", "y", "width", "height"))
        if not all(type(v) is int for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > shape[1] or y+h > shape[0]:
            raise ValueError("候选坐标超出原图范围")
        if not 0 < item["area"] <= w*h:
            raise ValueError("候选面积不合法")
    if bool(result["face_mask"].any()) != bool(result["candidates"]):
        raise ValueError("人脸掩膜与候选列表不一致")


def overlay_image(bgr, result):
    overlay = bgr.copy()
    face = result["face_mask"].astype(bool)
    overlay[face] = (overlay[face] * .65 + np.array([70, 200, 70]) * .35).astype(np.uint8)
    contours, _ = cv2.findContours(result["face_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 2)
    for index, item in enumerate(result["candidates"], 1):
        b = item["bbox"]
        x, y, w, h = (b[k] for k in ("x", "y", "width", "height"))
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (255, 160, 20), 2)
        cv2.putText(overlay, str(index), (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 160, 20), 2)
    return overlay


def detect(path, model, config, diagonal=False):
    start = time.perf_counter()
    expected_mode = "diagonal" if diagonal else "full"
    if config.get("covariance_mode", expected_mode) != expected_mode:
        raise ValueError("算法与配置不匹配：独立通道需 --diagonal 及 config/baseline.json")
    # One read binds decoding and the recorded source hash to the same bytes.
    path = Path(path)
    before = fingerprint(path)
    bgr = load_photo(path)
    if before != fingerprint(path):
        raise ValueError("照片在读取期间发生改变，请重试")
    function = segment_image_baseline if diagonal else segment_image
    result = function(bgr, model, config)
    validate_result(result, bgr.shape[:2])
    metadata = {
        "schema_version": 1, "image_name": path.name, "image_sha256": before,
        "width": bgr.shape[1], "height": bgr.shape[0], "orientation": "EXIF-transposed",
        "model_revision": model["model_revision"], "model_content_sha256": digest_json(model),
        "config": config, "config_content_sha256": digest_json(config),
        "algorithm": "diagonal" if diagonal else "full_covariance",
        "algorithm_revision": config.get("algorithm_revision", "integrated-v1"),
        "candidates": result["candidates"], "timing_ms": result["timing_ms"],
        "load_and_segment_ms": (time.perf_counter()-start)*1000,
        "timing_scope": "timing_ms: segmentation and postprocess only; excludes read, drawing, export",
        "mask_encoding": {"npy": "uint8 0/1", "png": "uint8 0/255"},
    }
    return bgr, result, metadata


def export_result(destination, bgr, result, metadata):
    """Publish a complete directory or leave no partial result. Never overwrite."""
    validate_result(result, bgr.shape[:2])
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("结果目录已存在，请使用一个新名称：" + str(destination))
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        for kind in ("skin", "face"):
            mask = result[kind+"_mask"]
            np.save(staging / (kind+"_mask.npy"), mask, allow_pickle=False)
            Image.fromarray(mask*255).save(staging / (kind+"_mask.png"))
        Image.fromarray(overlay_image(bgr, result)[:, :, ::-1]).save(staging / "overlay.png")
        payload = dict(metadata)
        payload["export_sha256"] = {p.name: fingerprint(p) for p in sorted(staging.iterdir())}
        (staging/"result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
