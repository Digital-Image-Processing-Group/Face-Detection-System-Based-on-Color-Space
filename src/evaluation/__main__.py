import argparse
import json
import sys
from pathlib import Path

from src.app.service import ROOT


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="验证集校准与完整系统评估（不覆盖 A 报告）")
    parser.add_argument("command", choices=("calibrate", "run"))
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/manifest.csv")
    parser.add_argument("--model", type=Path, default=ROOT/"models/skin_model.json")
    parser.add_argument("--config-dir", type=Path, default=ROOT/"config")
    parser.add_argument("--output", type=Path, default=ROOT/"reports")
    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            from .calibrate import calibrate
            report = calibrate(args.manifest, args.model, args.config_dir, args.output/"calibration.json")
            print(json.dumps(report["selected"], indent=2))
        else:
            from .runner import evaluate
            report = evaluate(args.manifest, args.model, args.config_dir/"pipeline.json", args.config_dir/"baseline.json", args.output)
            print(json.dumps({mode: r["target_check"] for mode, r in report["results"].items()}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, "评估失败："+str(error)+"\n")


if __name__ == "__main__":
    main()
