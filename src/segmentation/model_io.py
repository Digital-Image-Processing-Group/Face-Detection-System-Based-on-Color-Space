import cv2
import numpy as np

COLOR_SPACE = "OpenCV-YCrCb-uint8-full-range"
FEATURE_ORDER = ["Cb", "Cr"]


def validate_model(model):
    if not isinstance(model, dict):
        raise ValueError("model 必须是 dict")
    if model.get("schema_version") != 1:
        raise ValueError("schema_version 必须为 1")
    if model.get("color_space") != COLOR_SPACE:
        raise ValueError("color_space 不匹配")
    if model.get("feature_order") != FEATURE_ORDER:
        raise ValueError("feature_order 必须为 ['Cb','Cr']")

    mean = model.get("mean")
    if not isinstance(mean, (list, tuple)) or len(mean) != 2:
        raise ValueError("mean 长度必须为 2")
    cov = model.get("covariance")
    if (not isinstance(cov, (list, tuple)) or len(cov) != 2
            or any(not isinstance(r, (list, tuple)) or len(r) != 2 for r in cov)):
        raise ValueError("covariance 必须为 2x2")

    lam = model.get("regularization_lambda")
    if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not np.isfinite(lam) or lam < 0:
        raise ValueError("regularization_lambda 必须为非负有限数")

    mean_arr = np.asarray(mean, dtype=np.float64)
    cov_arr = np.asarray(cov, dtype=np.float64)
    if not np.all(np.isfinite(mean_arr)) or not np.all(np.isfinite(cov_arr)):
        raise ValueError("模型参数含非有限值")
    if np.any(mean_arr < 0) or np.any(mean_arr > 255):
        raise ValueError("肤色均值必须在 0 至 255 之间")
    if not np.allclose(cov_arr, cov_arr.T, rtol=0, atol=1e-10):
        raise ValueError("协方差矩阵必须对称")
    if np.linalg.eigvalsh(cov_arr).min() < -1e-10:
        raise ValueError("原始协方差矩阵必须半正定")
    if np.linalg.eigvalsh(cov_arr + lam * np.eye(2)).min() <= 0:
        raise ValueError("正则化后的协方差矩阵必须正定")
    info = model.get("training_info")
    if info is not None and (not isinstance(info, dict) or not isinstance(info.get("sampled_pixel_count"), int) or info["sampled_pixel_count"] < 3):
        raise ValueError("模型至少需要三个训练像素")

    rev = model.get("model_revision")
    if not isinstance(rev, str) or not rev:
        raise ValueError("model_revision 必须为非空字符串")

    return {
        "mean": mean_arr,
        "covariance": cov_arr,
        "regularization_lambda": float(lam),
        "model_revision": rev,
    }


def get_cb_cr(bgr_image):
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    # OpenCV 返回 Y, Cr, Cb
    return ycrcb[:, :, 2].astype(np.float64), ycrcb[:, :, 1].astype(np.float64)
