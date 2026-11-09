"""较弱闭运算、提前填洞和候选尺度结构元素。"""
import time

import cv2
import numpy as np

from .postprocess import _fill_holes_limited
from .segment import _validate_config, _validate_input, squared_threshold
from .model_io import validate_model
from .skin_mask import skin_mask_from_model


def odd_size(value):
    return min(101, max(1, round(value)) | 1)


def _morph(mask, opening, closing):
    for operation, size in ((cv2.MORPH_OPEN, opening), (cv2.MORPH_CLOSE, closing)):
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.morphologyEx(mask, operation, kernel)
    return mask


def morphology_stage(skin, pp, profile):
    h, w = skin.shape
    fraction = profile['candidate_fraction']
    if not fraction:
        scale = min(h, w) / pp.get('kernel_reference_side', min(h, w))
        return _morph(skin, odd_size(pp['open_kernel']*scale),
                      odd_size(profile['close_kernel']*scale))
    number, labels, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
    output = np.zeros_like(skin)
    for label in range(1, number):
        x, y, bw, bh, area = map(int, stats[label])
        # 留出闭运算填充小间隙可能增加的面积。
        if area < h*w*pp['min_area_ratio']/4:
            continue
        side = min(bw, bh)
        opening = odd_size(side*fraction)
        closing = odd_size(side*fraction*profile['close_kernel']/pp['open_kernel'])
        pad = max(opening, closing)
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(w, x+bw+pad), min(h, y+bh+pad)
        component = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        output[y0:y1, x0:x1] |= _morph(component, opening, closing)
    return output


def filter_stage(mask, pp, early_fill):
    h, w = mask.shape
    number, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    candidates = []
    for label in range(1, number):
        x, y, bw, bh, raw_area = map(int, stats[label])
        if bw*bh < h*w*pp['min_area_ratio'] or raw_area > h*w*pp['max_area_ratio']:
            continue
        if not pp['min_aspect_ratio'] <= bw/bh <= pp['max_aspect_ratio']:
            continue
        component = (labels[y:y+bh, x:x+bw] == label).astype(np.uint8)
        if early_fill:
            component = _fill_holes_limited(component, int(raw_area*pp['hole_fill_max_area_ratio']))
        area = int(component.sum())
        if not pp['min_area_ratio'] <= area/(h*w) <= pp['max_area_ratio']:
            continue
        if area/(bw*bh) < pp['min_fill_ratio']:
            continue
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            continue
        _, (a, b), _ = cv2.fitEllipse(contour)
        ellipse_area = np.pi*a*b/4
        if ellipse_area <= 0 or min(area, ellipse_area)/max(area, ellipse_area) < pp['min_ellipse_similarity']:
            continue
        if not early_fill:
            component = _fill_holes_limited(component, int(raw_area*pp['hole_fill_max_area_ratio']))
        output[y:y+bh, x:x+bw] |= component
        candidates.append({'bbox': dict(x=x, y=y, width=bw, height=bh), 'area': int(component.sum())})
    return output, candidates


def postprocess_refined(skin, pp, profile):
    return filter_stage(morphology_stage(skin, pp, profile), pp, profile['early_fill'])


def segment_refined(image, model, config, profile):
    _validate_input(image)
    _validate_config(config)
    checked = validate_model(model)
    if config['model_revision'] != checked['model_revision']:
        raise ValueError('配置与模型版本不匹配')
    start = time.perf_counter()
    skin = skin_mask_from_model(image, checked, squared_threshold(config))
    face, candidates = postprocess_refined(skin, config['postprocess'], profile)
    return dict(skin_mask=skin, face_mask=face, candidates=candidates,
                timing_ms=(time.perf_counter()-start)*1000)
