"""验证集选择形态学配置，再比较原图尺寸的分割效果。"""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.app.service import load_pipeline
from src.evaluation.data import main_samples, scene_samples, hand_samples
from src.evaluation.metrics import counts, metrics
from src.segmentation import segment_image
from src.segmentation.morphology_refinement import morphology_stage, filter_stage, segment_refined
from src.segmentation.skin_mask import skin_mask_from_model
from src.segmentation.model_io import validate_model
from src.training.dataset import fingerprint, read_bgr


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def profiles():
    return [dict(close_kernel=close, early_fill=early, candidate_fraction=fraction)
            for fraction in (0., .01, .025) for close in (1, 3, 5, 9) for early in (False, True)]


def aggregate(rows):
    result = metrics([sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn')])
    positive = [row['iou'] for row in rows if row['tp']+row['fn']]
    absent = [row for row in rows if not row['tp']+row['fn']]
    result.update(images=len(rows), mean_positive_image_iou=float(np.mean(positive)) if positive else None,
                  no_face_images=len(absent), no_face_nonempty=sum(row['fp'] > 0 for row in absent))
    return result


def validate(folder):
    model, config = load_pipeline()
    samples = main_samples(ROOT/'data/manifest.csv', 'validation')
    configs = profiles()
    write_json(folder/'validation_plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
               split='validation', objective='maximum accumulated face IoU including background false positives',
               profiles=configs, resolution='original', color_model_sha256=fingerprint(ROOT/'models/skin_model.json'),
               algorithm_sha256=fingerprint(ROOT/'src/segmentation/morphology_refinement.py')))
    results = [[] for _ in configs]
    baseline = []
    checked = validate_model(model)
    frozen = json.loads((ROOT/'reports/evaluation.json').read_text(encoding='utf-8'))['results']['full']['validation']['per_image']
    by_id = {row['image_id']: row for row in frozen}
    integrity = {}
    for index, sample in enumerate(samples, 1):
        image, _, truth = sample.load()
        skin = skin_mask_from_model(image, checked, config['distance_squared_threshold'])
        cache = {}
        for position, profile in enumerate(configs):
            key = (profile['close_kernel'], profile['candidate_fraction'])
            if key not in cache:
                cache[key] = morphology_stage(skin, config['postprocess'], profile)
            prediction, _ = filter_stage(cache[key], config['postprocess'], profile['early_fill'])
            value = metrics(counts(prediction, truth))
            results[position].append(dict(image_id=sample.image_id, **value))
            if profile == dict(close_kernel=9, early_fill=False, candidate_fraction=0.):
                assert all(value[k] == by_id[sample.image_id]['face'][k] for k in ('tp', 'fp', 'fn')), sample.image_id
                baseline.append(value)
        integrity[sample.image_id] = dict(photo=fingerprint(sample.photo), face=fingerprint(sample.face_path))
        if index % 5 == 0 or index == len(samples):
            print(f'Validation {index}/{len(samples)}', flush=True)
    trials = [dict(profile=p, summary=aggregate(rows), per_image=rows) for p, rows in zip(configs, results)]
    best = max(range(len(trials)), key=lambda k: trials[k]['summary']['iou'] or 0.)
    groups = {
        'weak_close': [k for k, p in enumerate(configs) if p['candidate_fraction'] == 0 and not p['early_fill'] and p['close_kernel'] < 9],
        'early_fill': [k for k, p in enumerate(configs) if p['candidate_fraction'] == 0 and p['early_fill'] and p['close_kernel'] == 9],
        'weak_close_early_fill': [k for k, p in enumerate(configs) if p['candidate_fraction'] == 0 and p['early_fill'] and p['close_kernel'] < 9],
        'adaptive': [k for k, p in enumerate(configs) if p['candidate_fraction'] > 0],
    }
    selected = dict(profile=configs[best], selection_split='validation', objective='accumulated face IoU',
                    summary=trials[best]['summary'], baseline_summary=aggregate(baseline),
                    ablations={name: configs[max(indices, key=lambda k: trials[k]['summary']['iou'] or 0.)] for name, indices in groups.items()},
                    algorithm_sha256=fingerprint(ROOT/'src/segmentation/morphology_refinement.py'),
                    model_sha256=fingerprint(ROOT/'models/skin_model.json'),
                    pipeline_sha256=fingerprint(ROOT/'config/pipeline.json'),
                    validation_inputs=integrity, frozen_utc=datetime.now(timezone.utc).isoformat())
    write_json(folder/'validation.json', dict(baseline=aggregate(baseline), trials=trials, selected_index=best))
    write_json(folder/'selected.json', selected)
    print(json.dumps({k:v for k,v in selected.items() if k != 'validation_inputs'}, ensure_ascii=False), flush=True)


def check_frozen(selected):
    for key, path in [('algorithm_sha256', 'src/segmentation/morphology_refinement.py'),
                      ('model_sha256', 'models/skin_model.json'), ('pipeline_sha256', 'config/pipeline.json')]:
        if fingerprint(ROOT/path) != selected[key]:
            raise ValueError('冻结文件发生变化：'+path)


def evaluate(folder):
    selected = json.loads((folder/'selected.json').read_text(encoding='utf-8'))
    check_frozen(selected)
    model, config = load_pipeline()
    manifest = ROOT/'data/manifest.csv'
    scenes = scene_samples(manifest)
    collections = dict(test=main_samples(manifest, 'test'), external_scenes=scenes,
                       background_validation=[s for s in main_samples(manifest, 'validation') if 'no_face' in s.tags],
                       derived_hands=hand_samples(manifest, scenes, ROOT/'models/stress_evaluation.json'))
    original = json.loads((ROOT/'reports/evaluation.json').read_text(encoding='utf-8'))['results']['full']
    original_rows = {r['image_id']: r for group in ('test', 'external_scenes', 'validation', 'derived_hands') for r in original[group]['per_image']}
    methods = dict(refined=selected['profile'], **selected['ablations'])
    write_json(folder/'evaluation_freeze.json', dict(selected_sha256=fingerprint(folder/'selected.json'), methods=methods,
               runner_sha256=fingerprint(Path(__file__)), resolution='original', created_utc=datetime.now(timezone.utc).isoformat()))
    rows = []
    verified_masks = 0
    for collection, samples in collections.items():
        for index, sample in enumerate(samples, 1):
            image, _, truth = sample.load()
            old = segment_image(image, model, config)
            old_metric = metrics(counts(old['face_mask'], truth))
            assert all(old_metric[k] == original_rows[sample.image_id]['face'][k] for k in ('tp', 'fp', 'fn')), sample.image_id
            masks = dict(baseline=old['face_mask'])
            timings = dict(baseline=old['timing_ms'])
            # 逐方法独立计时，均包括同一颜色分割步骤。
            for name, profile in methods.items():
                new = segment_refined(image, model, config, profile)
                assert np.array_equal(new['skin_mask'], old['skin_mask'])
                masks[name], timings[name] = new['face_mask'], new['timing_ms']
            row = dict(image_id=sample.image_id, collection=collection, tags=sample.tags,
                       image_sha256=fingerprint(sample.photo), shape=list(truth.shape), timing_ms=timings,
                       **{name: metrics(counts(mask, truth)) for name, mask in masks.items()})
            row['delta_iou'] = row['refined']['iou']-row['baseline']['iou'] if truth.any() else None
            row['evidence'] = 'comparisons/'+sample.image_id+'.jpg'
            for name in ('baseline', 'refined'):
                path = folder/'masks'/name/(sample.image_id+'.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(masks[name]*255).save(path)
                with Image.open(path) as saved:
                    assert np.array_equal(np.asarray(saved)//255, masks[name])
                verified_masks += 1
            comparison_sheet(folder/row['evidence'], image, truth, masks['baseline'], masks['refined'], row)
            rows.append(row)
            if index % 5 == 0 or index == len(samples):
                print(f'Evaluate {collection} {index}/{len(samples)}', flush=True)
    all_methods = ['baseline', *methods]
    summary = {}
    for group in [*collections, 'test_frontal', 'test_profile']:
        chosen = [r for r in rows if (r['collection'] == 'test' and group[5:] in r['tags'])] if group.startswith('test_') else [r for r in rows if r['collection'] == group]
        summary[group] = {name: aggregate([r[name] for r in chosen]) for name in all_methods}
        for name in all_methods:
            summary[group][name]['median_ms'] = float(np.median([r['timing_ms'][name] for r in chosen]))
    write_json(folder/'comparison.json', dict(selection=selected, summary=summary, per_image=rows))
    write_json(folder/'verification.json', dict(inputs=len(rows), mask_roundtrips=verified_masks,
               frozen_baseline_counts_match=True, color_masks_unchanged=True, original_resolution=True,
               comparison_sheets=len(rows)))
    report(folder, rows, summary, selected)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def report(folder, rows, summary, selected):
    names = dict(test='测试人像24张', external_scenes='复杂人脸场景33张', background_validation='无脸背景20张',
                 derived_hands='手部裁剪2张', test_frontal='其中正脸8张', test_profile='其中侧脸16张')
    def fmt(v): return '—' if v is None else f'{v:.4f}'
    lines = ['# 形态学调整对比实验', '',
             '颜色模型、距离平方阈值4和原参考标注保持一致；24组形态学配置只用100张验证图选择，所有处理和评估均使用原图尺寸。',
             '候选尺度方法先分离肤色连通域，再按候选短边的1%或2.5%决定开运算核；闭运算核乘以配置中闭核与原开核的比值。小于最低候选面积四分之一的碎片先移除。',
             '提前填洞使用原候选前景面积30%的单洞上限，再计算填充率和椭圆相似度；其余几何约束不变。', '',
             '选中配置：`'+json.dumps(selected['profile'], ensure_ascii=False)+'`。',
             f'验证集累计IoU：{selected["baseline_summary"]["iou"]:.4f} → {selected["summary"]["iou"]:.4f}。', '',
             '| 集合 | 原平均IoU | 新平均IoU | 平均变化/百分点 | 原累计IoU | 新累计IoU |', '|---|---:|---:|---:|---:|---:|']
    for group in ('test', 'test_frontal', 'test_profile', 'external_scenes'):
        old, new = summary[group]['baseline'], summary[group]['refined']
        lines.append(f'| {names[group]} | {fmt(old["mean_positive_image_iou"])} | {fmt(new["mean_positive_image_iou"])} | '
                     f'{100*(new["mean_positive_image_iou"]-old["mean_positive_image_iou"]):+.2f} | {fmt(old["iou"])} | {fmt(new["iou"])} |')
    lines.extend(['', '| 无脸集合 | 原误检张数 | 新误检张数 | 原误检像素 | 新误检像素 |', '|---|---:|---:|---:|---:|'])
    for group in ('background_validation', 'derived_hands'):
        old, new = summary[group]['baseline'], summary[group]['refined']
        lines.append(f'| {names[group]} | {old["no_face_nonempty"]} | {new["no_face_nonempty"]} | {old["fp"]} | {new["fp"]} |')
    lines.extend(['', '## 各项改动的对照', '', '各方法分别在验证集内选择；下表为人脸图逐张平均IoU。', '',
                  '| 集合 | 原方案 | 减弱闭运算 | 仅提前填洞 | 减弱闭运算＋提前填洞 | 候选尺度自适应 | 最终选中 |', '|---|---:|---:|---:|---:|---:|---:|'])
    for group in ('test', 'external_scenes'):
        lines.append('| '+names[group]+' | '+' | '.join(fmt(summary[group][name]['mean_positive_image_iou']) for name in ('baseline', 'weak_close', 'early_fill', 'weak_close_early_fill', 'adaptive', 'refined'))+' |')
    lines.extend(['', '各方法配置：', '', '```json', json.dumps(selected['ablations'], ensure_ascii=False, indent=2), '```'])
    featured = []
    for group in ('test', 'external_scenes'):
        group_rows = sorted([r for r in rows if r['collection'] == group], key=lambda r: r['delta_iou'], reverse=True)
        featured.extend([r for r in group_rows if r['delta_iou'] > .005][:3])
        featured.extend([r for r in reversed(group_rows) if r['delta_iou'] < -.005][:2])
    write_json(folder/'featured.json', [dict(image_id=r['image_id'], before=r['baseline']['iou'], after=r['refined']['iou'],
               delta=r['delta_iou'], evidence=r['evidence']) for r in featured])
    lines.extend(['', '## 变化明显的人脸样例', '', '| 图片 | 原IoU | 新IoU | 变化/百分点 | 对照图 |', '|---|---:|---:|---:|---|'])
    for r in featured:
        lines.append(f'| {r["image_id"]} | {r["baseline"]["iou"]:.4f} | {r["refined"]["iou"]:.4f} | {r["delta_iou"]*100:+.2f} | [查看]({r["evidence"]}) |')
    lines.extend(['', '绿色为正确保留，红色为误保留，蓝色为漏检。index.html默认只显示人脸图，动物和其他无脸图通过单独按钮查看。',
                  '复现：先运行 `python scripts/compare_morphology.py validate`，完成后运行 `python scripts/compare_morphology.py evaluate`。',
                  '使用已冻结配置处理单张图片：`python scripts/compare_morphology.py predict --image 图片路径`。',
                  '无脸背景20张属于验证集；手部裁剪来自既有场景。'])
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    columns = ['image_id', 'collection']+[name+'_'+metric for name in ('baseline', 'refined', *selected['ablations']) for metric in ('tp','fp','fn','iou','precision','recall')]+['delta_iou','evidence']
    with (folder/'per_image.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({key: r[key] for key in ('image_id','collection','delta_iou','evidence')} | {name+'_'+key: r[name][key] for name in ('baseline','refined',*selected['ablations']) for key in ('tp','fp','fn','iou','precision','recall')})
    cards = []
    for r in sorted(rows, key=lambda r: r['delta_iou'] if r['delta_iou'] is not None else -2., reverse=True):
        delta = f'{r["delta_iou"]*100:+.2f}个百分点' if r['delta_iou'] is not None else '无脸对照'
        cards.append(f'<article data-kind="{r["collection"]}"><h2>{html.escape(r["image_id"])}　{delta}</h2><a href="{r["evidence"]}"><img loading="lazy" src="{r["evidence"]}"></a></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>形态学调整对比</title><style>body{font-family:system-ui;max-width:1300px;margin:24px auto;background:#f5f6f8}article{background:white;margin:20px 0;padding:10px}img{width:100%}h2{font-size:18px;overflow-wrap:anywhere}button{padding:10px;margin:4px}nav{position:sticky;top:0;background:#f5f6f8}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>原方案与形态学调整</h1><p>默认展示亚洲人脸；按IoU提升排序。绿色正确保留，红色误检，蓝色漏检。</p>'
    page += '<table><tr><th>集合</th><th>原平均IoU</th><th>新平均IoU</th></tr>'
    for group in ('test','external_scenes'):
        page += f'<tr><td>{names[group]}</td><td>{summary[group]["baseline"]["mean_positive_image_iou"]:.4f}</td><td>{summary[group]["refined"]["mean_positive_image_iou"]:.4f}</td></tr>'
    page += '</table><nav>' + ''.join(f'<button onclick="filter(\'{kind}\')">{label}</button>' for kind,label in [('faces','人脸图'),('test','测试人像'),('external_scenes','复杂人脸场景'),('background_validation','无脸背景'),('derived_hands','手部'),('all','全部')])+'</nav>'
    page += '\n'.join(cards)+"<script>function filter(k){document.querySelectorAll('article').forEach(e=>{e.hidden=k==='faces'?!['test','external_scenes'].includes(e.dataset.kind):k!=='all'&&e.dataset.kind!==k})}filter('faces')</script></html>"
    (folder/'index.html').write_text(page, encoding='utf-8')


def comparison_sheet(path, image, truth, old, new, row):
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    title_font = ImageFont.truetype(font_path, 25)
    label_font = ImageFont.truetype(font_path, 21)
    small_font = ImageFont.truetype(font_path, 18)
    canvas = Image.new('RGB', (1500, 920), '#f5f6f8')
    draw = ImageDraw.Draw(canvas)
    name = row['image_id']
    draw.text((18, 8), name[:90], font=title_font, fill='#182337')
    before, after = row['baseline']['iou'], row['refined']['iou']
    def value(v): return '不适用' if v is None else f'{v:.4f}'
    delta = row['delta_iou']
    change = f'IoU变化 {delta:+.4f}（{delta*100:+.2f}个百分点）' if delta is not None else '无脸背景：比较误检像素'
    draw.text((18, 45), f'原方法 {value(before)}   新方法 {value(after)}   {change}',
              font=label_font, fill='#147d4d' if (delta or 0) > 0 else '#a13b32')

    def overlay(mask):
        rgb = image[:, :, ::-1].copy()
        for region, color in ((mask.astype(bool) & truth.astype(bool), [40, 210, 90]),
                              (mask.astype(bool) & ~truth.astype(bool), [245, 55, 55]),
                              (~mask.astype(bool) & truth.astype(bool), [50, 120, 245])):
            rgb[region] = (rgb[region]*.45 + np.array(color)*.55).astype(np.uint8)
        return rgb
    arrays = [image[:, :, ::-1], overlay(old), overlay(new), truth*255, old*255, new*255]
    titles = ['原图', '原方法：预测与参考对照', '新方法：预测与参考对照',
              '人脸参考', '原方法人脸掩膜', '新方法人脸掩膜']
    for i, (array, title) in enumerate(zip(arrays, titles)):
        x, y = (i % 3)*500, 88+(i//3)*390
        draw.text((x+12, y), title, font=label_font, fill='#182337')
        tile = Image.fromarray(array).convert('RGB')
        tile.thumbnail((480, 345))
        canvas.paste(tile, (x+(500-tile.width)//2, y+35+(345-tile.height)//2))
    draw.text((18, 878), '绿色：正确保留    红色：误保留    蓝色：漏检    指标按原图尺寸逐像素计算', font=small_font, fill='#34435a')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('validate','evaluate','predict'))
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/morphology-comparison')
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    if args.phase == 'validate':
        validate(args.output)
    elif args.phase == 'evaluate':
        evaluate(args.output)
    else:
        if args.image is None:
            parser.error('predict需要--image')
        selected_path = args.output/'selected.json'
        if not selected_path.exists():
            selected_path = ROOT/'experiments/morphology_refinement/selected.json'
        selected = json.loads(selected_path.read_text(encoding='utf-8'))
        check_frozen(selected)
        model, config = load_pipeline()
        image = read_bgr(args.image)
        result = segment_refined(image, model, config, selected['profile'])
        args.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result['face_mask']*255).save(args.output/(args.image.stem+'_face.png'))
        overlay = image[:, :, ::-1].copy()
        foreground = result['face_mask'].astype(bool)
        overlay[foreground] = (.5*overlay[foreground]+np.array([30, 220, 80])*.5).astype(np.uint8)
        Image.fromarray(overlay).save(args.output/(args.image.stem+'_overlay.jpg'))
        print('人脸掩膜已保存：', args.output/(args.image.stem+'_face.png'))


if __name__ == '__main__':
    main()
