"""Rasterize explicit reviewed polygons; never treat model predictions as GT."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset import Record, data_path, read_bgr, read_manifest, write_manifest


def polygon_mask(spec: dict, shape: tuple[int, int], kind: str) -> np.ndarray:
    height, width = shape
    if spec.get("width") != width or spec.get("height") != height:
        raise ValueError("Polygon coordinates must use the upright original image dimensions")
    layer = spec.get(kind)
    if not isinstance(layer, dict):
        raise ValueError(f"Missing {kind} layer")
    if not isinstance(layer.get("include"), list) or not isinstance(layer.get("exclude"), list):
        raise ValueError("Each layer needs include and exclude polygon lists")
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for operation, value in (("include", 1), ("exclude", 0)):
        for points in layer[operation]:
            array = np.asarray(points, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3 or not np.isfinite(array).all():
                raise ValueError("A polygon needs at least three finite x/y points")
            if np.any(array[:, 0] < 0) or np.any(array[:, 0] >= width) or np.any(array[:, 1] < 0) or np.any(array[:, 1] >= height):
                raise ValueError("Polygon coordinates are outside the image")
            draw.polygon([tuple(point) for point in array], fill=value)
    return np.asarray(image, dtype=np.uint8)


def rasterize(manifest: Path, polygons: Path) -> int:
    import json

    records = read_manifest(manifest)
    specifications = {}
    paths = sorted(polygons.rglob("*.json")) if polygons.is_dir() else [polygons]
    for path in paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        items = content.get("annotations", [content])
        if not isinstance(items, list):
            raise ValueError(f"annotations must be a list: {path}")
        for spec in items:
            image_id = spec.get("image_id")
            if not image_id or image_id in specifications:
                raise ValueError(f"Missing or duplicate annotation image_id: {path}")
            specifications[image_id] = spec
    unknown = set(specifications) - {row.image_id for row in records}
    if unknown:
        raise ValueError(f"Annotations reference unknown images: {sorted(unknown)}")
    updated = []
    count = 0
    for row in records:
        if row.image_id not in specifications:
            updated.append(row)
            continue
        spec = specifications[row.image_id]
        if spec.get("image_id") != row.image_id or not spec.get("annotator"):
            raise ValueError(f"Annotation needs matching image_id and annotator: {row.image_id}")
        image = read_bgr(data_path(manifest, row.image_path))
        fields = asdict(row)
        if spec.get("training_exclusion_reason"):
            fields["scene_tags"] = ";".join(dict.fromkeys(row.scene_tags.split(";") + ["exclude_color_training"]))
        for kind, allowed in (("skin", ("sampled", "complete")), ("face", ("draft", "complete"))):
            if kind not in spec:
                continue
            status = spec[kind].get("status")
            if status not in allowed:
                raise ValueError(f"Invalid {kind} status: {row.image_id}")
            mask = polygon_mask(spec, image.shape[:2], kind)
            if kind == "skin" and status == "sampled" and not mask.any():
                raise ValueError(f"Sampled skin must contain positive pixels: {row.image_id}")
            relative = f"annotations/{kind}/{row.image_id}.png"
            target = data_path(manifest, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask * 255).save(target)
            fields[f"{kind}_status"] = status
            fields[f"{kind}_mask_path"] = relative
        updated.append(Record(**fields))
        count += 1
    write_manifest(manifest, updated)
    return count


def contact_sheet(manifest: Path, output: Path, offset: int = 0, limit: int = 20,
                  overlays: bool = False) -> None:
    from PIL import ImageFont
    from .dataset import read_mask

    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
    records = read_manifest(manifest)[offset:offset + limit]
    if not records:
        raise ValueError("No records at this offset")
    tile, label_height, columns = 224, 32, 5
    sheet = Image.new("RGB", (columns * tile, ((len(records) + columns - 1) // columns) * (tile + label_height)), "white")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(records):
        bgr = read_bgr(data_path(manifest, row.image_path))
        rgb = bgr[:, :, ::-1].copy()
        if overlays:
            for kind, tint in (("face", np.array([40, 100, 255])), ("skin", np.array([0, 255, 0]))):
                relative = getattr(row, f"{kind}_mask_path")
                if relative:
                    mask = read_mask(data_path(manifest, relative), rgb.shape[:2]).astype(bool)
                    rgb[mask] = (rgb[mask] * 0.55 + tint * 0.45).astype(np.uint8)
        image = Image.fromarray(rgb)
        image.thumbnail((tile, tile))
        x, y = index % columns * tile, index // columns * (tile + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + tile), row.image_id, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
