"""CLI for A-owned work only; no detector or product GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .annotations import contact_sheet, rasterize
from .dataset import (assign_splits, read_manifest, validate_dataset,
                      write_json, write_manifest)
from .model import fit_model, save_diagnostics
from .assessment import assess


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="A: dataset preparation and YCrCb skin-model training")
    commands = root.add_subparsers(dest="command", required=True)
    radiate = commands.add_parser("collect-radiate", help="Download official RADIATE color photos with anonymous author participant codes")
    radiate.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    validate = commands.add_parser("validate", help="Check files, masks, provenance and split leakage")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--purpose", choices=("train", "evaluation"), default="train")
    validate.add_argument("--report", type=Path)
    split = commands.add_parser("split", help="Group-aware split of a verified, explicitly unsplit manifest")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--ratios", nargs=3, type=float, default=(0.6, 0.2, 0.2))
    split.add_argument("--prepare-only", action="store_true",
                       help="Prepare verified subject groups before masks are complete; does not certify evaluation readiness")
    annotate = commands.add_parser("rasterize", help="Render reviewed polygon annotations and update their manifest")
    annotate.add_argument("--manifest", type=Path, required=True)
    annotate.add_argument("--polygons", type=Path, default=Path("data/annotations/polygons"))
    sheet = commands.add_parser("contact-sheet", help="Save a labeled image sheet for visual review")
    sheet.add_argument("--manifest", type=Path, required=True)
    sheet.add_argument("--output", type=Path, required=True)
    sheet.add_argument("--offset", type=int, default=0)
    sheet.add_argument("--limit", type=int, default=20)
    sheet.add_argument("--overlays", action="store_true")
    sheet.add_argument("--tile-size", type=int, default=600)
    sheet.add_argument("--columns", type=int, default=2)
    train = commands.add_parser("train", help="Fit a Gaussian using annotated training skin only")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("models/skin_model.json"))
    train.add_argument("--diagnostics", type=Path, default=Path("artifacts/radiate_model"))
    train.add_argument("--samples-per-image", type=int, default=2000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--regularization", type=float, default=1.0)
    train.add_argument("--mask-margin", type=int, default=15,
                       help="Erode training-only skin masks by this many original-image pixels")
    evaluation = commands.add_parser("assess", help="Select threshold on validation people, then report held-out test metrics")
    evaluation.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    evaluation.add_argument("--model", type=Path, default=Path("models/skin_model.json"))
    evaluation.add_argument("--output", type=Path, default=Path("models/evaluation.json"))
    evaluation.add_argument("--operating-point", type=Path, default=Path("models/operating_point.json"))
    evaluation.add_argument("--previews", type=Path, default=Path("artifacts/radiate_predictions"))
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "collect-radiate":
            from .radiate import collect_radiate
            print(json.dumps(collect_radiate(args.manifest), indent=2))
        elif args.command == "validate":
            report = validate_dataset(args.manifest, args.purpose)
            if args.report:
                write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["valid"] else 1
        elif args.command == "split":
            if args.output.resolve().parent != args.manifest.resolve().parent:
                raise ValueError("Output manifest must stay in the same directory so relative file paths remain valid")
            if args.output.exists():
                raise ValueError("Split output already exists; choose a new output filename")
            rows = assign_splits(read_manifest(args.manifest), args.seed, tuple(args.ratios))
            # Validate before writing. Existing annotations and decoded images are checked.
            import tempfile
            with tempfile.TemporaryDirectory(dir=args.manifest.parent) as temporary:
                # Paths are relative to the manifest, so keep the temporary manifest beside it.
                trial = args.manifest.parent / (Path(temporary).name + ".csv")
                try:
                    write_manifest(trial, rows)
                    report = validate_dataset(trial, "train" if args.prepare_only else "evaluation")
                    if not report["valid"]:
                        raise ValueError("Split validation failed:\n" + "\n".join(report["errors"]))
                finally:
                    trial.unlink(missing_ok=True)
            write_manifest(args.output, rows)
            write_json(args.output.with_suffix(".split.json"), {
                "seed": args.seed, "ratios": args.ratios, "counts": report["splits"],
                "input_manifest": str(args.manifest), "strategy": "transitive subject and scene groups",
                "annotation_check": "preparation_only" if args.prepare_only else "complete_evaluation_annotations_required",
                "evaluation_ready": False if args.prepare_only else True,
            })
            print(f"Saved split manifest: {args.output}")
        elif args.command == "rasterize":
            count = rasterize(args.manifest, args.polygons)
            print(f"Rasterized annotations for {count} images")
        elif args.command == "contact-sheet":
            contact_sheet(args.manifest, args.output, args.offset, args.limit, args.overlays, args.tile_size, args.columns)
            print(f"Saved contact sheet: {args.output}")
        elif args.command == "train":
            model, pixels, report = fit_model(args.manifest, args.samples_per_image, args.seed, args.regularization, args.mask_margin)
            save_diagnostics(model, pixels, args.diagnostics)
            write_json(args.diagnostics / "dataset_validation.json", report)
            write_json(args.output, model)
            print(json.dumps({"model": str(args.output), "model_revision": model["model_revision"],
                              "usage": model["training_info"]["usage"],
                              "images": model["training_info"]["training_image_count"],
                              "sampled_pixels": model["training_info"]["sampled_pixel_count"]}, indent=2))
        elif args.command == "assess":
            model = json.loads(args.model.read_text(encoding="utf-8"))
            report = assess(args.manifest, model, args.output, args.operating_point, args.previews)
            print(json.dumps(report["summary"], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
