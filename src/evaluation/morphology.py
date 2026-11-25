"""Validation-only morphology selection followed by frozen exploratory evaluation."""
from copy import deepcopy
import json
from pathlib import Path

import cv2

from src.app.service import ROOT, digest_json, validate_result
from src.segmentation import segment_image, segment_image_baseline
from src.segmentation.postprocess import postprocess
from src.segmentation.skin_mask import skin_mask_from_model, skin_mask_baseline
from src.segmentation.model_io import validate_model
from src.training.dataset import fingerprint, write_json
from .data import main_samples, scene_samples, hand_samples
from .metrics import counts, metrics, summarize


def optimize(root=ROOT):
    root = Path(root)
    cv2.setNumThreads(4)
    manifest = root / 'data/manifest.csv'
    model_path = root / 'models/skin_model.json'
    model = json.loads(model_path.read_text(encoding='utf-8'))
    validated = validate_model(model)
    # Immutable original configurations, including after a repeated optimization run.
    previous = root / 'reports/morphology.json'
    prior_report = json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else None
    original = (prior_report['original_configs'] if prior_report
                else json.loads((root / 'reports/freeze.json').read_text(encoding='utf-8'))['configs'])
    if any(c['model_sha256'] != fingerprint(model_path) or c['manifest_sha256'] != fingerprint(manifest)
           for c in original.values()):
        raise ValueError('Original model/manifest changed; cannot compare frozen baseline')
    trials = {}
    for mode, config in original.items():
        variants = [('legacy', deepcopy(config['postprocess']))]
        for minimum_area in (.0015, .002, .003, .004):
            pp = dict(config['postprocess'], method='adaptive', local_kernel_ratio=.01,
                      bridge_open_ratio=0, noise_area_ratio=.00002,
                      hole_fill_total_area_ratio=.3, min_area_ratio=minimum_area)
            variants.append((f'adaptive-area-{minimum_area}', pp))
        trials[mode] = [dict(name=name, postprocess=pp, counts=[0, 0, 0], positive_ious=[],
                             background_images=0, background_nonempty=0) for name, pp in variants]
    samples = main_samples(manifest, 'validation')
    integrity = {}
    for index, sample in enumerate(samples, 1):
        image, _, truth = sample.load()
        integrity[sample.image_id] = dict(photo=fingerprint(sample.photo), skin=fingerprint(sample.skin_path),
                                         face=fingerprint(sample.face_path))
        for mode, items in trials.items():
            function = skin_mask_from_model if mode == 'full' else skin_mask_baseline
            skin = function(image, validated, original[mode]['distance_squared_threshold'])
            for trial in items:
                face, _ = postprocess(skin, trial['postprocess'])
                value = counts(face, truth)
                trial['counts'] = [a+b for a, b in zip(trial['counts'], value)]
                if truth.any():
                    trial['positive_ious'].append(metrics(value)['iou'])
                else:
                    trial['background_images'] += 1
                    trial['background_nonempty'] += int(face.any())
        if index % 10 == 0:
            print(f'Morphology validation: {index}/{len(samples)}', flush=True)
    selected = {}
    for mode, items in trials.items():
        for trial in items:
            trial['face'] = metrics(trial.pop('counts'))
            values = trial.pop('positive_ious')
            trial['mean_face_iou'] = sum(values) / len(values) if values else None
        # Keep original on exact ties. Guard positive-image average and background count.
        baseline = items[0]
        eligible = [t for t in items if t['mean_face_iou'] >= baseline['mean_face_iou']
                    and t['background_nonempty'] <= baseline['background_nonempty']]
        best = max(eligible, key=lambda t: t['face']['iou'])
        config = deepcopy(original[mode])
        config.update(algorithm_revision='morphology-v2', postprocess=best['postprocess'],
                      selection='validation_face_micro_iou_with_mean_iou_and_background_guards',
                      calibration_inputs_sha256=digest_json(integrity))
        config.pop('config_revision', None)
        config['config_revision'] = digest_json(config)[:16]
        selected[mode] = config
        write_json(root / 'config' / ('pipeline.json' if mode == 'full' else 'baseline.json'), config)
    report = dict(selection_split='validation', validation_images=len(samples),
                  thresholds_changed=False, trials=trials, original_configs=original, selected=selected,
                  source_sha256=fingerprint(root / 'src/segmentation/postprocess.py'),
                  evaluation_scope='Exploratory on previously published datasets; not an unseen test.')
    if prior_report:
        report['earlier_validation_trials'] = prior_report.get('earlier_validation_trials', prior_report['trials'])
    destination = root / 'reports/morphology.json'
    write_json(destination, report)  # Freeze before loading any test or scene references.
    scenes = scene_samples(manifest)
    collections = dict(test=main_samples(manifest, 'test'), external_scenes=scenes,
                       derived_hands=hand_samples(manifest, scenes, root / 'models/stress_evaluation.json'))
    evaluation = {}
    for mode in original:
        function = segment_image if mode == 'full' else segment_image_baseline
        evaluation[mode] = {}
        for name, collection in collections.items():
            by_variant = {}
            for variant, config in [('original', original[mode]), ('optimized', selected[mode])]:
                rows = []
                for sample in collection:
                    image, skin_ref, face_ref = sample.load()
                    result = function(image, model, config)
                    validate_result(result, image.shape[:2])
                    rows.append(dict(image_id=sample.image_id, scene_tags=sample.tags,
                        skin=metrics(counts(result['skin_mask'], skin_ref)),
                        face=metrics(counts(result['face_mask'], face_ref)),
                        candidate_count=len(result['candidates']),
                        timing_ms=result['timing_ms']))
                by_variant[variant] = dict(summary=summarize(rows), per_image=rows,
                    by_scene={tag: summarize([r for r in rows if tag in r['scene_tags']])
                              for tag in sorted({tag for r in rows for tag in r['scene_tags']})})
            evaluation[mode][name] = by_variant
            print(f'Evaluated {mode}/{name}', flush=True)
    report['evaluation'] = evaluation
    write_json(destination, report)
    return report


if __name__ == '__main__':
    optimize()
