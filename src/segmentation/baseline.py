import time
from .model_io import validate_model
from .skin_mask import skin_mask_baseline
from .postprocess import postprocess
from .segment import _validate_input, _validate_config, squared_threshold


def segment_image_baseline(bgr_image, model, config):
    _validate_input(bgr_image)
    validated = validate_model(model)
    _validate_config(config)
    if config["model_revision"] != validated["model_revision"]:
        raise ValueError("配置与模型 model_revision 不匹配")

    t0 = time.perf_counter()
    skin = skin_mask_baseline(bgr_image, validated, squared_threshold(config))
    face, candidates = postprocess(skin, config["postprocess"])
    timing_ms = (time.perf_counter() - t0) * 1000.0
    return {"skin_mask": skin, "face_mask": face, "candidates": candidates, "timing_ms": timing_ms}
