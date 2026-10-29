"""Independent annotation drafts for source-reviewed scenes, never auto-certified."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from .dataset import write_json, fingerprint


def prepare():
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    cv2.setNumThreads(4)
    root=Path('data/cache/annotation_models')
    net=cv2.dnn.readNetFromONNX(str(root/'resnet18.onnx'))
    options=vision.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(root/'selfie_multiclass.tflite')),
        running_mode=vision.RunningMode.IMAGE,output_category_mask=True,output_confidence_masks=False)
    sources=json.loads(Path('data/sources/wider_fair_downloads.json').read_text(encoding='utf-8'))['records']
    records=[]
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        for i,s in enumerate(sources):
            image=np.asarray(Image.open(Path('data')/s['image_path']).convert('RGB'));h,w=image.shape[:2]
            body=segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=image)).category_mask.numpy_view().copy()
            if body.shape!=(h,w):body=cv2.resize(body,(w,h),interpolation=cv2.INTER_NEAREST)
            skin=(body==2).astype('uint8');face=np.zeros((h,w),'uint8')
            for b in s['author_annotations']:
                x,y,x2,y2=[float(b[k]) for k in ('x1','y1','x2','y2')];bw=x2-x;bh=y2-y
                x0=max(0,int(x-.25*bw));y0=max(0,int(y-.2*bh));x1=min(w,int(x2+.25*bw));y1=min(h,int(y2+.2*bh))
                a=cv2.resize(image[y0:y1,x0:x1],(512,512)).astype('float32')/255
                a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                net.setInput(a.transpose(2,0,1)[None]);parsed=cv2.resize(net.forward()[0].argmax(0).astype('uint8'),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
                f=np.isin(parsed,[1,2,3,4,5,10,11,12,13]);sk=np.isin(parsed,[1,7,8,10,14])
                face[y0:y1,x0:x1]|=f.astype('uint8');skin[y0:y1,x0:x1]|=sk.astype('uint8')
                skin[y0:y1,x0:x1][np.isin(parsed,[2,3,4,5,6,9,11,12,13,15,16,17,18])]=0
            path=Path(f'data/annotations/scene_drafts/{s["image_id"]}.png');path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(skin|(face<<1)).save(path)
            records.append({'image_id':s['image_id'],'reference_path':path.relative_to('data').as_posix(),
                'sha256':fingerprint(path),'source_sha256':s['sha256'],'width':w,'height':h,'status':'draft',
                'initialization':'BiSeNet face parts per author face box; MediaPipe body skin; no evaluated color predictions',
                'helper_sha256':{p.name:fingerprint(p) for p in root.glob('*') if p.suffix in ('.onnx','.tflite')}})
            print(f'Scene drafts {i+1}/{len(sources)}',flush=True)
    write_json(Path('data/annotations/scene_drafts/records.json'),{'records':records})


if __name__=='__main__':prepare()
