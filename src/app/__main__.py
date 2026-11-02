import argparse
import json
import sys
from pathlib import Path

from .service import ROOT, detect, export_result, load_pipeline


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="YCbCr 人脸区域检测：无参数打开桌面界面")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, help="新的结果目录，禁止覆盖已有目录")
    parser.add_argument("--model", type=Path, default=ROOT/"models/skin_model.json")
    parser.add_argument("--config", type=Path, default=ROOT/"config/pipeline.json")
    parser.add_argument("--diagonal", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.image:
            if not args.output:
                parser.error("--image 需要 --output")
            model, config = load_pipeline(args.model, args.config)
            image, result, metadata = detect(args.image, model, config, args.diagonal)
            folder = export_result(args.output, image, result, metadata)
            print(json.dumps({"output": str(folder), "candidates": len(result["candidates"]), "timing_ms": result["timing_ms"]}, ensure_ascii=False))
        else:
            from .gui import launch
            launch(args.model, args.config)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("运行失败："+str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
