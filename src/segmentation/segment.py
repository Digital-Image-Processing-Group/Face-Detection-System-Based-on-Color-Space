import time
import numpy as np
from .model_io import validate_model
from .skin_mask import skin_mask_from_model
from .postprocess import postprocess

_REQUIRED_CONFIG = ("postprocess", "model_revision")
_REQUIRED_PP = (
    "open_kernel", "close_kernel", "min_area_ratio", "max_area_ratio",
    "min_aspect_ratio", "max_aspect_ratio", "min_fill_ratio",
    "min_ellipse_similarity", "hole_fill_max_area_ratio",
)


def _validate_input(bgr_image):
    if (not isinstance(bgr_image, np.ndarray) or bgr_image.dtype != np.uint8
            or bgr_image.ndim != 3 or bgr_image.shape[2] != 3
            or bgr_image.shape[0] == 0 or bgr_image.shape[1] == 0):
        raise ValueError("输入必须是 HxWx3 的 uint8 BGR 数组")


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("config 必须是 dict")
    for k in _REQUIRED_CONFIG:
        if k not in config:
            raise ValueError("config 缺少 %s" % k)
    pp = config["postprocess"]
    if not isinstance(pp, dict):
        raise ValueError("postprocess 必须是 dict")
    for k in _REQUIRED_PP:
        if k not in pp:
            raise ValueError("postprocess 缺少 %s" % k)
    threshold = squared_threshold(config)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("马氏距离平方阈值必须是正有限数")
    for key in ("open_kernel", "close_kernel"):
        value = pp[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value % 2 != 1 or value > 101:
            raise ValueError(key + " 必须为 1 至 101 的奇数")
    for key in _REQUIRED_PP[2:]:
        value = pp[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(key + " 必须为非负有限数")
        if "aspect" not in key and value > 1:
            raise ValueError(key + " 必须在 0 至 1 之间")
    for small, large in (("min_area_ratio", "max_area_ratio"), ("min_aspect_ratio", "max_aspect_ratio")):
        if pp[small] > pp[large] or pp[large] <= 0:
            raise ValueError(small + " 与 " + large + " 范围不合法")
    reference = pp.get("kernel_reference_side", 1000)
    if isinstance(reference, bool) or not isinstance(reference, (int, float)) or not np.isfinite(reference) or reference <= 0:
        raise ValueError("kernel_reference_side 必须为正有限数")


def squared_threshold(config):
    """Read canonical field; accept the original B field for old callers."""
    if "distance_squared_threshold" in config and "distance_threshold" in config:
        if config["distance_squared_threshold"] != config["distance_threshold"]:
            raise ValueError("两个距离平方阈值字段不一致")
    value = config.get("distance_squared_threshold", config.get("distance_threshold"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("缺少有效的 distance_squared_threshold")
    return float(value)


def segment_image(bgr_image, model, config):
    _validate_input(bgr_image)
    validated = validate_model(model)
    _validate_config(config)
    if config["model_revision"] != validated["model_revision"]:
        raise ValueError("配置与模型 model_revision 不匹配")

    t0 = time.perf_counter()
    skin = skin_mask_from_model(bgr_image, validated, squared_threshold(config))
    face, candidates = postprocess(skin, config["postprocess"])
    timing_ms = (time.perf_counter() - t0) * 1000.0
    return {"skin_mask": skin, "face_mask": face, "candidates": candidates, "timing_ms": timing_ms}
