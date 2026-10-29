"""Frozen-model external scene diagnostics; no identity-independence claim."""
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from PIL import Image
from .dataset import fingerprint,read_bgr,write_json
from .assessment import distance_map,face_proposal,confusion,metrics


def assess_scenes():
    root=Path('data');model=json.loads(Path('models/skin_model.json').read_text(encoding='utf-8'))
    point=json.loads(Path('models/operating_point.json').read_text(encoding='utf-8'))
    if model['model_revision']!=point['model_revision']:raise ValueError('Model and threshold mismatch')
    sources=json.loads((root/'sources/wider_fair_downloads.json').read_text(encoding='utf-8'))['records']
    results=[];totals={k:[0,0,0] for k in ('skin','face_baseline')};crops=[]
    for s in sources:
        if not all(b['Ethnicity']=='Asian' and b['Valid']=='True' for b in s['author_annotations']):
            raise ValueError('Scene contains an ineligible author population annotation')
        if s['review_status']!='visually_reviewed_project_silver_reference':raise ValueError('Unreviewed scene')
        photo=root/s['image_path'];reference=root/s['reference_path']
        if fingerprint(photo)!=s['sha256'] or fingerprint(reference)!=s['reference_sha256']:raise ValueError('Scene hash mismatch')
        bgr=read_bgr(photo);a=np.asarray(Image.open(reference))
        if a.shape!=bgr.shape[:2] or not set(np.unique(a)).issubset({0,1,2,3}):raise ValueError('Invalid scene reference')
        skin=(distance_map(bgr,model)<=point['distance_squared_threshold']).astype('uint8');face=face_proposal(skin)
        item={'image_id':s['image_id'],'scene_tags':s['scene_tags'],'author_face_count':len(s['author_annotations'])}
        for kind,pred,bit in [('skin',skin,1),('face_baseline',face,2)]:
            counts=confusion(pred,(a&bit)>0);item[kind]=metrics(counts);totals[kind]=[x+y for x,y in zip(totals[kind],counts)]
        results.append(item)
        if s['candidate_index'] in (4,5):
            h,w=a.shape;x0=int(.15*w);x1=int(.85*w);y0=int(.55*h);y1=int(.98*h)
            ref=a[y0:y1,x0:x1]
            if (ref&2).any():raise ValueError('Hand-only crop intersects a face reference')
            cut=bgr[y0:y1,x0:x1];pred=face_proposal((distance_map(cut,model)<=point['distance_squared_threshold']).astype('uint8'))
            folder=root/'derived/hand_rois';folder.mkdir(parents=True,exist_ok=True)
            target=folder/f'{s["image_id"]}.png';Image.fromarray(cut[:,:,::-1]).save(target)
            truth=folder/f'{s["image_id"]}_reference.png';Image.fromarray(ref).save(truth)
            crops.append({'parent_image_id':s['image_id'],'crop_xyxy':[x0,y0,x1,y1],
                'image_path':target.relative_to(root).as_posix(),'sha256':fingerprint(target),
                'reference_path':truth.relative_to(root).as_posix(),'reference_sha256':fingerprint(truth),
                'original_independent_scene':False,'face_false_positive_pixels':int(pred.sum()),'nonempty_prediction':bool(pred.any())})
    scenes={}
    for tag in sorted({t for s in sources for t in s['scene_tags']}):
        subset=[r for r in results if tag in r['scene_tags']]
        scenes[tag]={'images':len(subset),**{kind:metrics([sum(r[kind][key] for r in subset) for key in ('tp','fp','fn')]) for kind in totals}}
    report={'model_revision':model['model_revision'],'threshold':point['distance_squared_threshold'],
        'original_photos':len(sources),'original_file_hashes_verified':len(sources),
        'scope':'External author-perceived-Asian real scenes, never used for fitting or calibration; anonymous scene IDs are not person IDs. Identity overlap with other sources cannot be proven absent; not pooled with subject-independent primary metrics.',
        'reference':'Project assisted silver references with visual review and explicit corrections; not official pixel truth.',
        'summary':{k:metrics(v) for k,v in totals.items()},'scene_summary':scenes,'per_image':results,
        'derived_hand_only':{'images':len(crops),'nonempty_predictions':sum(c['nonempty_prediction'] for c in crops),
            'scope':'Crops of two reviewed real parent photos, not new independent originals or participants','records':crops}}
    write_json(Path('models/stress_evaluation.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k in ('model_revision','original_photos','summary','scene_summary')},indent=2))


if __name__=='__main__':assess_scenes()
