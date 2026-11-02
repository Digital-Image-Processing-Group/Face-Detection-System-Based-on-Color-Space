"""Pixel metrics with explicit undefined denominators and no-face accounting."""
import numpy as np


def counts(prediction, truth):
    if prediction.shape != truth.shape or not np.isin(prediction, (0, 1)).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("指标输入必须是同尺寸 0/1 掩膜")
    pred, ref = prediction.astype(bool), truth.astype(bool)
    return [int(np.count_nonzero(pred & ref)), int(np.count_nonzero(pred & ~ref)), int(np.count_nonzero(~pred & ref))]


def metrics(value):
    tp, fp, fn = map(int, value)
    def ratio(n, d):
        return n/d if d else None
    return dict(tp=tp, fp=fp, fn=fn, precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn))


def summarize(rows):
    summary = {"images": len(rows)}
    for kind in ("skin", "face"):
        summary[kind] = metrics([sum(r[kind][k] for r in rows) for k in ("tp", "fp", "fn")])
        positive = [r[kind]["iou"] for r in rows if r[kind]["tp"]+r[kind]["fn"] > 0]
        summary[kind]["positive_images"] = len(positive)
        summary[kind]["mean_positive_image_iou"] = float(np.mean(positive)) if positive else None
    absent = [r for r in rows if r["face"]["tp"]+r["face"]["fn"] == 0]
    false = sum(r["face"]["fp"] > 0 for r in absent)
    summary["no_face"] = {"images": len(absent), "nonempty_predictions": false,
                          "image_false_positive_rate": false/len(absent) if absent else None,
                          "false_positive_pixels": sum(r["face"]["fp"] for r in absent)}
    times = [r["timing_ms"] for r in rows]
    summary["timing"] = {"samples": len(times), "median_ms": float(np.median(times)) if times else None,
                         "p95_ms": float(np.percentile(times, 95)) if times else None,
                         "max_ms": max(times) if times else None,
                         "scope": "B segmentation and postprocess; excludes reading, reference loading, display and export"}
    return summary
