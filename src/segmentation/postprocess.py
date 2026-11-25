import cv2
import numpy as np


def _fill_holes_limited(region_mask, max_hole_area, total_budget=None):
    h, w = region_mask.shape
    inv = (1 - region_mask).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    border = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    out = region_mask.copy()
    budget = region_mask.size if total_budget is None else total_budget
    for lab in sorted(range(1, num), key=lambda k: int(stats[k, cv2.CC_STAT_AREA])):
        if lab in border:
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= max_hole_area and area <= budget:
            out[labels == lab] = 1
            budget -= area
    return out


def _global_cleanup(skin_mask, pp):
    h, w = skin_mask.shape
    scale = min(h, w) / pp.get("kernel_reference_side", min(h, w))
    def size(key):
        return min(101, max(1, round(pp[key] * scale)) | 1)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size("open_kernel"), size("open_kernel")))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size("close_kernel"), size("close_kernel")))
    m = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, ko)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    return m


def _postprocess_legacy(skin_mask, pp):
    h, w = skin_mask.shape
    m = _global_cleanup(skin_mask, pp)

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


def _components(mask):
    """Yield tight local masks with offsets; never join separate objects."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        yield int(x), int(y), (labels[y:y+h, x:x+w] == label).astype(np.uint8)


def _local_morph(mask, kernel_size, operation):
    # Explicit zero padding prevents ROI edges acting as artificial foreground.
    pad = kernel_size + 1
    padded = np.pad(mask, pad)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.morphologyEx(padded, operation, kernel)
    return result[pad:-pad, pad:-pad]


def _split_bridges(mask, ratio):
    """Open thin bridges, then partition original pixels by nearest retained core.

    This is a binary morphology partition, not a semantic face/hand classifier.
    Recovery is label constrained so different cores cannot reconnect.
    """
    if not ratio:
        return [mask]
    size = min(101, max(3, round(min(mask.shape) * ratio)) | 1)
    opened = _local_morph(mask, size, cv2.MORPH_OPEN)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cores = np.zeros_like(mask)
    retained = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= max(9, .1 * mask.sum()):
            cores[labels == label] = 1
            retained += 1
    if retained < 2 or retained > 6:
        return [mask]
    _, nearest = cv2.distanceTransformWithLabels(1 - cores, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_CCOMP)
    return [((nearest == label) & (mask != 0)).astype(np.uint8)
            for label in range(1, retained + 1)]


def _adaptive_postprocess(skin_mask, pp):
    h, w = skin_mask.shape
    image_area = h * w
    face = np.zeros_like(skin_mask)
    candidates = []
    initial = _global_cleanup(skin_mask, pp) if pp.get("preclean", False) else skin_mask
    for x, y, raw in _components(initial):
        if raw.sum() < max(2, image_area * pp.get("noise_area_ratio", .00002)):
            continue
        side = min(raw.shape)
        size = min(31, max(1, round(side * pp["local_kernel_ratio"])) | 1)
        clean = _local_morph(raw, size, cv2.MORPH_OPEN)
        # Close within each surviving component: different faces stay separate.
        for dx, dy, component in _components(clean):
            close_size = min(31, max(1, round(min(component.shape) * pp["local_kernel_ratio"] * 2)) | 1)
            component = _local_morph(component, close_size, cv2.MORPH_CLOSE)
            for part in _split_bridges(component, pp.get("bridge_open_ratio", 0)):
                for px, py, tight in _components(part):
                    ph, pw = tight.shape
                    area = int(tight.sum())
                    if not pp["min_area_ratio"] <= area / image_area <= pp["max_area_ratio"]:
                        continue
                    if not pp["min_aspect_ratio"] <= pw / ph <= pp["max_aspect_ratio"]:
                        continue
                    if area / (pw * ph) < pp["min_fill_ratio"]:
                        continue
                    contours, _ = cv2.findContours(tight, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = max(contours, key=cv2.contourArea)
                    if len(contour) < 5:
                        continue
                    _, axes, _ = cv2.fitEllipse(contour)
                    ellipse_area = np.pi * axes[0] * axes[1] / 4
                    if ellipse_area <= 0 or min(area, ellipse_area) / max(area, ellipse_area) < pp["min_ellipse_similarity"]:
                        continue
                    filled = _fill_holes_limited(tight,
                        int(area * pp["hole_fill_max_area_ratio"]),
                        int(area * pp["hole_fill_total_area_ratio"]))
                    left, top = x + dx + px, y + dy + py
                    face[top:top+ph, left:left+pw] |= filled
                    candidates.append({"bbox": {"x": left, "y": top, "width": pw, "height": ph},
                                       "area": int(filled.sum())})
    return face, candidates


def postprocess(skin_mask, pp):
    if pp.get("method", "legacy") == "adaptive":
        return _adaptive_postprocess(skin_mask, pp)
    return _postprocess_legacy(skin_mask, pp)
