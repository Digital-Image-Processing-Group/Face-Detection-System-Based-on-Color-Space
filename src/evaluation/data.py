"""Hash-bound evaluation inputs; calibration loads validation references only."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.training.dataset import data_path, fingerprint, read_bgr, read_manifest, read_mask


@dataclass
class Sample:
    image_id: str
    photo: Path
    tags: list[str]
    skin_path: Path | None = None
    face_path: Path | None = None
    packed_path: Path | None = None
    expected_photo: str | None = None
    expected_reference: str | None = None
    crop: list[int] | None = None
    parent: str | None = None
    source_reference: dict | None = None
    manifest: Path | None = None

    def load(self):
        if self.expected_photo and fingerprint(self.photo) != self.expected_photo:
            raise ValueError("原图 SHA256 不匹配："+self.image_id)
        image = read_bgr(self.photo)
        if self.packed_path:
            if fingerprint(self.packed_path) != self.expected_reference:
                raise ValueError("参考 SHA256 不匹配："+self.image_id)
            with Image.open(self.packed_path) as source:
                if source.mode != "L" or source.getexif().get(274, 1) != 1:
                    raise ValueError("位编码参考必须是转正后的灰度 PNG")
                packed = np.asarray(source)
            if packed.shape != image.shape[:2] or not np.isin(packed, (0, 1, 2, 3)).all():
                raise ValueError("位编码参考尺寸或值域不合法")
            skin, face = ((packed & bit) != 0 for bit in (1, 2))
        else:
            skin = read_mask(self.skin_path, image.shape[:2])
            face = read_mask(self.face_path, image.shape[:2])
            if self.source_reference:
                from src.training.annotations import reference_mask, polygon_mask
                spec = self.source_reference
                for kind, actual in (("skin", skin), ("face", face)):
                    expected = (reference_mask(self.manifest, spec, image.shape[:2], kind)
                                if "reference_path" in spec else polygon_mask(spec, image.shape[:2], kind))
                    if not np.array_equal(actual, expected):
                        raise ValueError("标注像素与冻结参考不一致："+self.image_id+"/"+kind)
        if self.crop:
            x0, y0, x1, y1 = self.crop
            image, skin, face = (a[y0:y1, x0:x1] for a in (image, skin, face))
            if face.any():
                raise ValueError("手部负样本裁剪包含人脸参考")
        return image, skin.astype(np.uint8), face.astype(np.uint8)


def main_samples(manifest, split):
    manifest = Path(manifest)
    sources = {}
    for name in ("hawaii_downloads.json", "bsds_downloads.json"):
        for source in json.loads((manifest.parent/"sources"/name).read_text(encoding="utf-8")):
            sources[source["image_id"]] = source["sha256"]
    specs = {}
    directory = manifest.parent/"annotations/records"
    for path in sorted(directory.glob("*.json")):
        for spec in json.loads(path.read_text(encoding="utf-8"))["annotations"]:
            if spec["image_id"] in specs:
                raise ValueError("重复参考记录")
            specs[spec["image_id"]] = spec
    return [Sample(r.image_id, data_path(manifest, r.image_path), r.scene_tags.split(";"),
                   skin_path=data_path(manifest, r.skin_mask_path), face_path=data_path(manifest, r.face_mask_path),
                   expected_photo=sources[r.image_id], source_reference=specs[r.image_id] if directory.exists() else None,
                   manifest=manifest) for r in read_manifest(manifest) if r.split == split]


def scene_samples(manifest):
    root = Path(manifest).parent
    records = json.loads((root/"sources/wider_fair_downloads.json").read_text(encoding="utf-8"))["records"]
    result = []
    for r in records:
        if r["review_status"] != "visually_reviewed_project_silver_reference":
            raise ValueError("场景参考尚未复核")
        result.append(Sample(r["image_id"], data_path(Path(manifest), r["image_path"]), r["scene_tags"],
                             packed_path=data_path(Path(manifest), r["reference_path"]),
                             expected_photo=r["sha256"], expected_reference=r["reference_sha256"]))
    return result


def hand_samples(manifest, scenes, a_report):
    by_id = {s.image_id: s for s in scenes}
    records = json.loads(Path(a_report).read_text(encoding="utf-8"))["derived_hand_only"]["records"]
    return [Sample(r["parent_image_id"]+"_hand", by_id[r["parent_image_id"]].photo, ["hand_only_crop"],
                   packed_path=by_id[r["parent_image_id"]].packed_path,
                   expected_photo=by_id[r["parent_image_id"]].expected_photo,
                   expected_reference=by_id[r["parent_image_id"]].expected_reference,
                   crop=r["crop_xyxy"], parent=r["parent_image_id"]) for r in records]
