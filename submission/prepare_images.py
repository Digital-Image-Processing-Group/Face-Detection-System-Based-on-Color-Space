"""从已恢复的数据生成课程提交版验证输入。"""
import csv
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = ROOT / 'data'


def main():
    with (DATA / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        selected = [(row['image_id'], DATA / row['image_path']) for row in csv.DictReader(stream)
                    if row['split'] == 'test' or row['image_id'].startswith('bsds_')]
    scenes = json.loads((DATA / 'sources/wider_fair_downloads.json').read_text(encoding='utf-8'))['records']
    selected.extend((row['image_id'], DATA / row['image_path']) for row in scenes)
    missing = [str(path) for _, path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError('请先按README恢复数据。缺少图片：' + missing[0])
    if len(selected) != 77:
        raise ValueError('验证输入清单应包含77张原图')
    sources = {}
    for filename in ('hawaii_downloads.json', 'bsds_downloads.json'):
        for row in json.loads((DATA / 'sources' / filename).read_text(encoding='utf-8')):
            sources[row['image_id']] = row['sha256']
    sources.update({row['image_id']: row['sha256'] for row in scenes})
    for image_id, path in selected:
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources[image_id]:
            raise ValueError('图片校验失败：' + image_id)

    target = BASE / 'images'
    target.mkdir(exist_ok=True)
    by_id = {}
    for image_id, path in selected:
        destination = target / (image_id + path.suffix.lower())
        if destination.exists() and destination.read_bytes() != path.read_bytes():
            raise FileExistsError('已有不同内容的图片：' + str(destination))
        if not destination.exists():
            shutil.copy2(path, destination)
        by_id[image_id] = path
    crops = json.loads((ROOT / 'models/stress_evaluation.json').read_text(encoding='utf-8'))['derived_hand_only']['records']
    for row in crops:
        parent = row['parent_image_id']
        with Image.open(by_id[parent]) as source:
            image = ImageOps.exif_transpose(source).convert('RGB').crop(row['crop_xyxy'])
        destination = target / (parent + '_hand.png')
        if destination.exists():
            with Image.open(destination) as existing:
                if existing.size != image.size or existing.convert('RGB').tobytes() != image.tobytes():
                    raise FileExistsError('已有不同内容的裁剪：' + str(destination))
        else:
            image.save(destination)
    print(f'Prepared {len(selected) + len(crops)} inputs in {target}')


if __name__ == '__main__':
    main()
