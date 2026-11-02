import cv2
import numpy as np


def _fill_holes_limited(region_mask, max_hole_area):
    h, w = region_mask.shape
    inv = (1 - region_mask).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    border = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    out = region_mask.copy()
    for lab in range(1, num):
        if lab in border:
            continue
        if int(stats[lab, cv2.CC_STAT_AREA]) <= max_hole_area:
            out[labels == lab] = 1
    return out


def postprocess(skin_mask, pp):
    h, w = skin_mask.shape
    scale = min(h, w) / pp.get("kernel_reference_side", min(h, w))
    def size(key):
        return min(101, max(1, round(pp[key] * scale)) | 1)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size("open_kernel"), size("open_kernel")))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size("close_kernel"), size("close_kernel")))
    m = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, ko)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    face = np.zeros((h, w), dtype=np.uint8)
    candidates = []
    img_area = float(h * w)

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

        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
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

        # Limit holes relative to this component, not the whole photograph.
        max_hole = int(pp["hole_fill_max_area_ratio"] * area)
        roi_filled = _fill_holes_limited(comp, max_hole)
        face[y:y + bh, x:x + bw] = np.maximum(face[y:y + bh, x:x + bw], roi_filled)
        candidates.append({
            "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
            "area": int(roi_filled.sum()),
        })

    return face, candidates
