"""Check frozen source/annotation integrity and reproduce the delivered fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .annotations import polygon_mask, rasterize, reference_mask
from .dataset import data_path, fingerprint, read_bgr, read_manifest, read_mask, write_json
from .model import fit_model
from .public_sources import checked


def audit(manifest: Path, model_path: Path, output: Path) -> dict:
    root=manifest.resolve().parent
    rows=read_manifest(manifest)
    sources={s['image_id']:s for name in ('hawaii_downloads.json','bsds_downloads.json')
             for s in json.loads((root/'sources'/name).read_text(encoding='utf-8'))}
    specs={s['image_id']:s for path in (root/'annotations/records').glob('*.json')
           for s in json.loads(path.read_text(encoding='utf-8'))['annotations']}
    people={p['participant_code']:p for p in json.loads(
        (root/'sources/hawaii_participants.json').read_text(encoding='utf-8'))['participants']}
    if set(sources)!=set(specs) or set(sources)!={r.image_id for r in rows}:
        raise ValueError('Manifest, source and annotation membership differ')
    before={r.image_id:{k:fingerprint(data_path(manifest,getattr(r,f'{k}_mask_path')))
                        for k in ('skin','face')} for r in rows}
    for row in rows:
        source=sources[row.image_id]
        checked(data_path(manifest,row.image_path).read_bytes(),source['sha256'])
        if row.subjects:
            checked(data_path(manifest,row.image_path).read_bytes(),source['author_sha256'])
            code=source['author_participant_code']
            if (row.subjects!={'hfd_'+code} or row.group_id!='hfd_'+code
                    or people[code]['researcher_categorized_race'] not in
                       ('Asian','East Asian','Southeast Asian')
                    or source['source_group']!=row.source_label):
                raise ValueError(f'Author participant/population mismatch: {row.image_id}')
        shape=read_bgr(data_path(manifest,row.image_path)).shape[:2]
        spec=specs[row.image_id]
        if row.subjects and spec['initialization']['source_image_sha256']!=source['sha256']:
            raise ValueError('Annotation source changed')
        for kind in ('skin','face'):
            expected=(reference_mask(manifest,spec,shape,kind) if 'reference_path' in spec
                      else polygon_mask(spec,shape,kind))
            if not np.array_equal(expected,read_mask(data_path(manifest,getattr(row,f'{kind}_mask_path')),shape)):
                raise ValueError(f'Final reference mismatch: {row.image_id}/{kind}')
    saved=json.loads(model_path.read_text(encoding='utf-8'))
    training=saved['training_info']
    reproduced,_,validation=fit_model(manifest,training['requested_samples_per_image'],
        training['random_seed'],saved['regularization_lambda'],training['training_mask_margin_pixels'])
    if reproduced!=saved:
        raise ValueError('Fit is not identical in this environment')
    original_manifest=fingerprint(manifest)
    rasterize(manifest,root/'annotations/records')
    if fingerprint(manifest)!=original_manifest:
        raise ValueError('Reference reconstruction changed the frozen manifest')
    after={r.image_id:{k:fingerprint(data_path(manifest,getattr(r,f'{k}_mask_path')))
                       for k in ('skin','face')} for r in rows}
    if before!=after:
        raise ValueError('Reference reconstruction changed mask bytes')
    report={'model_revision':saved['model_revision'],'manifest_sha256':original_manifest,
        'original_photos':len(rows),'author_participant_groups':len(people),
        'original_file_hashes_verified':len(rows),'masks_reconstructed_identically':len(rows)*2,
        'author_population_and_identity_verified':True,'fit_reproduced_exactly':True,
        'dataset_valid':validation['valid'],'duplicate_check_warning_count':len(validation['warnings']),
        'split_counts':validation['splits'],'limitations':'Project silver-standard references; no independent expert pixel-truth review. New real scenes have author-perceived population labels but no verified subject identities; evaluated separately as external stress data.'}
    write_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('data/manifest.csv'))
    parser.add_argument('--model',type=Path,default=Path('models/skin_model.json'))
    parser.add_argument('--output',type=Path,default=Path('models/delivery_checks.json'))
    args=parser.parse_args()
    print(json.dumps(audit(args.manifest,args.model,args.output),indent=2))
