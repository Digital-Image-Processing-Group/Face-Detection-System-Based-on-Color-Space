"""Prepare independent annotation drafts; visual review is required to finalize.

Optional annotation tooling only. The delivered detector/trainer never imports
these neural models. Resulting references are explicitly silver-standard,
project-authored annotations, not author-provided or expert ground truth.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .dataset import data_path, fingerprint, read_bgr, read_manifest, read_mask, write_json, write_manifest


def prepare(manifest: Path, models: Path, offset: int = 0, limit: int = 500) -> None:
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    cv2.setNumThreads(4)
    parser_path = models / 'resnet18.onnx'
    body_path = models / 'selfie_multiclass.tflite'
    model_hashes = {p.name: fingerprint(p) for p in (parser_path, body_path)}
    network = cv2.dnn.readNetFromONNX(str(parser_path))
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    options = vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(body_path)),
        running_mode=vision.RunningMode.IMAGE, output_category_mask=True,
        output_confidence_masks=False)
    rows = read_manifest(manifest)
    metadata_path = manifest.parent / 'annotations/proposals/provenance.json'
    provenance = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    updated = {r.image_id:r for r in rows}
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        for row in rows[offset:offset+limit]:
            paths={kind:f'annotations/proposals/{kind}/{row.image_id}.png' for kind in ('skin','face')}
            prior=provenance.get(row.image_id,{})
            if (prior.get('source_image_sha256')==fingerprint(data_path(manifest,row.image_path))
                    and prior.get('model_sha256')==model_hashes
                    and prior.get('method_revision')==3
                    and all(data_path(manifest,p).exists() for p in paths.values())):
                updated[row.image_id]=replace(row,skin_mask_path=paths['skin'],face_mask_path=paths['face'],
                                             skin_status='sampled',face_status='draft')
                continue
            bgr=read_bgr(data_path(manifest,row.image_path)); h,w=bgr.shape[:2]
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            result=segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            body=result.category_mask.numpy_view().copy()
            if body.shape != (h,w):
                body=cv2.resize(body,(w,h),interpolation=cv2.INTER_NEAREST)
            regions=(body==3).astype(np.uint8)
            count, labels, stats,_=cv2.connectedComponentsWithStats(regions)
            if count<2:
                raise ValueError(f'Face initialization missing; manual annotation required: {row.image_id}')
            x,y,width,height,_=stats[1+np.argmax(stats[1:,cv2.CC_STAT_AREA])]
            x0=max(0,int(x-.45*width)); x1=min(w,int(x+1.45*width))
            y0=max(0,int(y-.35*height)); y1=min(h,int(y+1.5*height))
            patch=rgb[y0:y1,x0:x1]
            tensor=cv2.resize(patch,(512,512),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
            tensor=(tensor-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
            network.setInput(tensor.transpose(2,0,1)[None])
            logits=network.forward()
            if logits.shape != (1,19,512,512):
                raise ValueError(f'Unexpected annotation-model output: {logits.shape}')
            parsed=cv2.resize(logits[0].argmax(0).astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
            skin=(body==2).astype(np.uint8)
            face=np.zeros((h,w),np.uint8)
            face_local=np.isin(parsed,[1,2,3,4,5,10,11,12,13]) | (body[y0:y1,x0:x1]==3)
            face_local[np.isin(parsed,[6,7,8,9,14,15,16,17,18])]=False
            face[y0:y1,x0:x1]=face_local.astype(np.uint8)
            # Body skin supplies exposed neck/chest outside the detailed head crop.
            local_skin=np.isin(parsed,[1,7,8,10,14]).astype(np.uint8)
            local_skin |= face_local.astype(np.uint8)
            local_skin |= skin[y0:y1,x0:x1]
            local_skin[np.isin(parsed,[2,3,4,5,6,9,11,12,13,15,16,17,18])]=0
            skin[y0:y1,x0:x1] = local_skin
            paths={}
            for kind,mask in [('skin',skin),('face',face)]:
                relative=f'annotations/proposals/{kind}/{row.image_id}.png'
                target=data_path(manifest,relative); target.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(mask*255).save(target)
                paths[kind]=relative
            updated[row.image_id]=replace(row,skin_mask_path=paths['skin'],face_mask_path=paths['face'],
                                         skin_status='sampled',face_status='draft')
            provenance[row.image_id]={'source_image_sha256':fingerprint(data_path(manifest,row.image_path)),
                                      'model_sha256':model_hashes,'method_revision':3,'head_crop_xyxy':[x0,y0,x1,y1],
                                      'face_pixels':int(face.sum()),'skin_pixels':int(skin.sum()),
                                      'review_status':'pending','method':'Independent neural initialization, not evaluated YCrCb predictions'}
            write_json(metadata_path,provenance)
            write_manifest(manifest.with_name('hawaii.review.csv'),[updated[r.image_id] for r in rows])
            print(f'Annotation draft: {row.image_id}',flush=True)


def finalize(manifest: Path, review: Path, output: Path, record_name: str = 'hawaii.json') -> dict:
    """Materialize only an explicitly reviewed set; never promote drafts implicitly."""
    from .annotations import rasterize
    if output.resolve().parent != manifest.resolve().parent:
        raise ValueError('Output manifest must remain beside source manifest')
    if Path(record_name).name != record_name or not record_name.endswith('.json'):
        raise ValueError('Record name must be a JSON filename')
    rows=read_manifest(manifest)
    ledger=json.loads(review.read_text(encoding='utf-8'))
    reviewed=ledger.get('reviewed_image_ids',[])
    if len(reviewed)!=len(set(reviewed)) or set(reviewed)!={r.image_id for r in rows}:
        raise ValueError('Review ledger must explicitly cover every image exactly once')
    if ledger.get('reviewer')!='codex_visual_review' or not ledger.get('reference_limitations'):
        raise ValueError('Explicit reviewer and reference limitations are required')
    draft_root=manifest.parent/'annotations/proposals'
    provenance=json.loads((draft_root/'provenance.json').read_text())
    corrections=ledger.get('corrections',{})
    specs=[]
    for row in rows:
        metadata=provenance[row.image_id]
        if metadata['source_image_sha256']!=fingerprint(data_path(manifest,row.image_path)):
            raise ValueError(f'Source image changed during annotation: {row.image_id}')
        image=read_bgr(data_path(manifest,row.image_path)); h,w=image.shape[:2]
        layers={kind:read_mask(draft_root/kind/f'{row.image_id}.png',(h,w)).copy() for kind in ('skin','face')}
        expected=ledger.get('draft_sha256',{}).get(row.image_id,{})
        for kind in layers:
            if expected.get(kind)!=fingerprint(draft_root/kind/f'{row.image_id}.png'):
                raise ValueError(f'Reviewed draft hash missing or changed: {row.image_id}/{kind}')
        for change in corrections.get(row.image_id,[]):
            if change.get('value') not in (0,1) or not set(change.get('layers',[])).issubset({'skin','face'}):
                raise ValueError('Correction needs binary value and valid layers')
            scale=w/float(change.get('canvas_width',600))
            polygon=np.rint(np.asarray(change['polygon'],dtype=float)*scale).astype(np.int32)
            if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3:
                raise ValueError('Correction needs at least three xy points')
            if np.any(polygon<0) or np.any(polygon[:,0]>=w) or np.any(polygon[:,1]>=h):
                raise ValueError('Correction coordinates exceed source image')
            for kind in change['layers']:
                cv2.fillPoly(layers[kind],[polygon],int(change['value']))
        if not layers['face'].any() or not layers['skin'].any():
            raise ValueError(f'Face-positive annotation is empty: {row.image_id}')
        packed=layers['skin'] | (layers['face']<<1)
        relative=f'annotations/reference/{row.image_id}.png'
        target=data_path(manifest,relative); target.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(packed).save(target)
        specs.append({'image_id':row.image_id,'width':w,'height':h,'annotator':'codex_assisted_visual_review',
                      'review_status':'visually_reviewed','reference_path':relative,'reference_sha256':fingerprint(target),
                      'method':'Independent BiSeNet part parsing and multiclass skin initialization, followed by project visual review and explicit corrections; no evaluated YCrCb predictions used as references',
                      'reference_limitations':ledger['reference_limitations'],
                      'initialization':{k:v for k,v in metadata.items() if k!='review_status'},
                      'review_corrections':corrections.get(row.image_id,[]),
                      'skin':{'status':'complete'},'face':{'status':'complete'}})
    write_manifest(output,rows)
    spec_path=manifest.parent/'annotations/records'/record_name
    write_json(spec_path,{'annotations':specs})
    rasterize(output,spec_path)
    return {'reviewed_images':len(specs),'reference_type':'project-assisted silver-standard, not official pixel ground truth'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--models',type=Path,default=Path('data/cache/annotation_models'))
    p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=500)
    p.add_argument('--finalize-review',type=Path)
    p.add_argument('--record-name',default='hawaii.json')
    p.add_argument('--output',type=Path,default=Path('data/hawaii.annotated.csv'))
    args=p.parse_args()
    if args.finalize_review:
        print(json.dumps(finalize(args.manifest,args.finalize_review,args.output,args.record_name),indent=2))
    else:
        prepare(args.manifest,args.models,args.offset,args.limit)
