import cv2
import numpy as np


def _fill_holes_limited(region_mask, max_hole_area):
    h, w = region_mask.shape
    inv = (1 - region_mask).astype(np.uint8)
    num, labels = cv2.connectedComponents(inv, connectivity=8)
    border = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    out = region_mask.copy()
    for lab in range(1, num):
        if lab in border:
            continue
        hole = labels == lab
        if int(hole.sum()) <= max_hole_area:
            out[hole] = 1
    return out


def postprocess(skin_mask, pp):
    h, w = skin_mask.shape
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pp["open_kernel"], pp["open_kernel"]))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pp["close_kernel"], pp["close_kernel"]))
    m = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, ko)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    face = np.zeros((h, w), dtype=np.uint8)
    candidates = []
    img_area = float(h * w)
    max_hole = int(pp["hole_fill_max_area_ratio"] * img_area)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area == 0:
            continue
        rel = area / img_area
        if rel < pp["min_area_ratio"] or rel > pp["max_area_ratio"]:
            continue
        ar = bw / bh if bh > 0 else 0.0
        if ar < pp["min_aspect_ratio"] or ar > pp["max_aspect_ratio"]:
            continue
        fill = area / (bw * bh) if bw * bh > 0 else 0.0
        if fill < pp["min_fill_ratio"]:
            continue

        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            continue
        _, (ma, mb), _ = cv2.fitEllipse(cnt)
        ell_area = np.pi * (ma / 2.0) * (mb / 2.0)
        if ell_area <= 0:
            continue
        sim = min(area, ell_area) / max(area, ell_area)
        if sim < pp["min_ellipse_similarity"]:
            continue

        roi = comp[y:y + bh, x:x + bw]
        roi_filled = _fill_holes_limited(roi, max_hole)
        face[y:y + bh, x:x + bw] = np.maximum(face[y:y + bh, x:x + bw], roi_filled)
        candidates.append({
            "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
            "area": int(area),
        })

    return face, candidates