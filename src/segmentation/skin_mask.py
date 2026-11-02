import numpy as np
from .model_io import get_cb_cr


def _mahalanobis_mask(bgr_image, mean, cov, lam, threshold):
    cb, cr = get_cb_cr(bgr_image)
    try:
        inv = np.linalg.inv(cov + lam * np.eye(2))
    except np.linalg.LinAlgError as e:
        raise ValueError("协方差矩阵不可逆") from e
    diff = np.stack([cb - mean[0], cr - mean[1]], axis=-1)
    d2 = np.einsum("...i,ij,...j->...", diff, inv, diff)
    return (d2 <= threshold).astype(np.uint8)


def skin_mask_from_model(bgr_image, model, threshold):
    return _mahalanobis_mask(
        bgr_image, model["mean"], model["covariance"],
        model["regularization_lambda"], threshold,
    )


def skin_mask_baseline(bgr_image, model, threshold):
    cov = np.diag(np.diag(model["covariance"]))
    return _mahalanobis_mask(
        bgr_image, model["mean"], cov,
        model["regularization_lambda"], threshold,
    )