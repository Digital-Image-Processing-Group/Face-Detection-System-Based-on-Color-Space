import time
import numpy as np
from .model_io import validate_model
from .skin_mask import skin_mask_from_model
from .postprocess import postprocess

_REQUIRED_CONFIG = ("distance_threshold", "postprocess", "model_revision")
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


def segment_image(bgr_image, model, config):
    _validate_input(bgr_image)
    validated = validate_model(model)
    _validate_config(config)
    if config["model_revision"] != validated["model_revision"]:
        raise ValueError("配置与模型 model_revision 不匹配")

    t0 = time.perf_counter()
    skin = skin_mask_from_model(bgr_image, validated, float(config["distance_threshold"]))
    face, candidates = postprocess(skin, config["postprocess"])
    timing_ms = (time.perf_counter() - t0) * 1000.0
    return skin, face, candidates, timing_ms