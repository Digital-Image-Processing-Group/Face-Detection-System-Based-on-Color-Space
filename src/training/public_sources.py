"""Restore the frozen, provenance-checked research delivery without changing splits."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import ssl
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
import truststore
from .dataset import data_path, read_bgr, read_manifest

ASSETS = {
    "resnet18.onnx": ("https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx", "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"),
    "selfie_multiclass.tflite": ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite", "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
}

def fetch(url: str, target: Path, limit: int = 8_000_000) -> bytes:
    if target.exists():
        content=target.read_bytes()
        if len(content)>limit:
            raise ValueError("Cached source exceeds size bound")
        return content
    request=Request(url,headers={"User-Agent":"ColorSpaceResearch/1.0"})
    with urlopen(request,timeout=25,context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as response:
        content=response.read(limit+1)
    if len(content)>limit:
        raise ValueError("Source exceeds download size bound")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(content)
    return content

def checked(content: bytes, expected: str) -> None:
    if hashlib.sha256(content).hexdigest()!=expected:
        raise ValueError("Source image hash mismatch")

def restore(manifest: Path) -> dict:
    manifest=manifest.resolve(); root=manifest.parent
    rows=read_manifest(manifest)
    sources={}
    for name in ("hawaii_downloads.json","bsds_downloads.json"):
        for source in json.loads((root/"sources"/name).read_text(encoding="utf-8")):
            if source["image_id"] in sources:
                raise ValueError("Duplicate source image ID")
            sources[source["image_id"]]=source
    if set(sources)!={r.image_id for r in rows}:
        raise ValueError("Frozen manifest and source records do not match")
    missing=[]
    for row in rows:
        target=data_path(manifest,row.image_path)
        if target.exists():
            checked(target.read_bytes(),sources[row.image_id]["sha256"])
        else:
            missing.append(row)
    direct=[r for r in missing if "download_url" in sources[r.image_id]]
    def download(row):
        source=sources[row.image_id]; target=data_path(manifest,row.image_path)
        temporary=target.with_suffix(target.suffix+".download")
        content=fetch(source["download_url"],temporary)
        checked(content,source["author_sha256"])
        temporary.replace(target)
        read_bgr(target)
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index,_ in enumerate(executor.map(download,direct),1):
            if index%10==0 or index==len(direct):
                print(f"Restored author-verified color photos: {index}/{len(direct)}",flush=True)
    archived=[r for r in missing if "archive_member" in sources[r.image_id]]
    if archived:
        source=sources[archived[0].image_id]
        content=fetch(source["archive_url"],root/"cache/bsds/BSR_bsds500.tgz",120_000_000)
        checked(content,source["archive_sha256"])
        with tarfile.open(fileobj=io.BytesIO(content),mode="r:gz") as archive:
            for row in archived:
                source=sources[row.image_id]
                member=archive.getmember(source["archive_member"])
                if not member.isfile() or member.size>4_000_000:
                    raise ValueError("Unexpected source archive member")
                payload=archive.extractfile(member).read()
                checked(payload,source["sha256"])
                target=data_path(manifest,row.image_path)
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
                read_bgr(target)
    return {"photos":len(rows),"restored":len(missing),"reused":len(rows)-len(missing),
            "participant_groups":len({s for row in rows for s in row.subjects}),
            "manifest_preserved":True,"network_used":bool(missing)}

def annotation_assets(directory: Path) -> None:
    for name,(url,sha256) in ASSETS.items():
        content=fetch(url,directory/name,100_000_000)
        checked(content,sha256)
        print(f"Annotation helper asset verified: {name}",flush=True)

def restore_stress(root: Path) -> dict:
    sources=json.loads((root/"sources/shape_examples.json").read_text())
    for source in sources:
        target=data_path(root/"manifest.csv",source["file"])
        content=fetch(source["source_url"],target,4_000_000)
        checked(content,source["sha256"])
    return {"public_examples":len(sources),"role":"qualitative hand illustrations only; excluded from original-photo counts and independent metrics"}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("restore","annotation-assets","restore-stress"))
    p.add_argument("--manifest",type=Path,default=Path("data/manifest.csv"))
    p.add_argument("--directory",type=Path,default=Path("data/cache/annotation_models"))
    args=p.parse_args()
    try:
        if args.command=="restore": print(json.dumps(restore(args.manifest),indent=2))
        elif args.command=="restore-stress": print(json.dumps(restore_stress(args.manifest.resolve().parent),indent=2))
        else: annotation_assets(args.directory)
    except (OSError,ValueError,KeyError) as error:
        p.exit(1,f"Error: {error}\n")
