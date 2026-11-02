"""Run frozen full/diagonal pipelines and save transparent per-image evidence."""
from datetime import datetime, timezone
import csv
import json
import os
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.app.service import digest_json, load_pipeline, validate_result, overlay_image
from src.segmentation import segment_image, segment_image_baseline
from src.training.dataset import fingerprint, write_json
from .data import main_samples, scene_samples, hand_samples
from .metrics import counts, metrics, summarize


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "platform": platform.platform(), "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
            "logical_cpus": os.cpu_count(), "opencv_threads": cv2.getNumThreads()}


def evidence_sheet(path, image, result, skin_ref, face_ref, caption):
    # English labels keep the evidence readable with any installed font.
    tiles = [image[:, :, ::-1], result["skin_mask"]*255, result["face_mask"]*255,
             overlay_image(image, result)[:, :, ::-1], skin_ref*255, face_ref*255]
    titles = ["Original", "Predicted skin", "Predicted face", "Candidates overlay", "Skin reference", "Face reference"]
    sheet = Image.new("RGB", (1200, 690), "#eef2f5")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (array, title) in enumerate(zip(tiles, titles)):
        tile = Image.fromarray(array).convert("RGB")
        tile.thumbnail((390, 280))
        x, y = (i%3)*400, (i//3)*330
        draw.text((x+10, y+8), title, fill="#1f2937", font=font)
        sheet.paste(tile, (x+(400-tile.width)//2, y+35+(280-tile.height)//2))
    draw.text((10, 665), caption[:140], fill="#1f2937", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def evaluate(manifest, model_path, full_path, diagonal_path, output_dir):
    manifest, output_dir = Path(manifest), Path(output_dir)
    full_model, full_config = load_pipeline(model_path, full_path)
    _, diagonal_config = load_pipeline(model_path, diagonal_path)
    configs = {"full": full_config, "diagonal": diagonal_config}
    for config in configs.values():
        if config.get("manifest_sha256") != fingerprint(manifest):
            raise ValueError("配置与清单哈希不匹配，请先完成验证集校准")
    scenes = scene_samples(manifest)
    collections = {"validation": main_samples(manifest, "validation"), "test": main_samples(manifest, "test"),
                   "external_scenes": scenes,
                   "derived_hands": hand_samples(manifest, scenes, Path(model_path).parent/"stress_evaluation.json")}
    validation_integrity = {s.image_id: {"photo": fingerprint(s.photo), "skin": fingerprint(s.skin_path),
                                        "face": fingerprint(s.face_path)} for s in collections["validation"]}
    for config in configs.values():
        if config.get("calibration_inputs_sha256") != digest_json(validation_integrity):
            raise ValueError("验证集文件在校准后发生改变，拒绝使用旧配置")
    # Persist the freeze before any performance evaluation on the test set.
    freeze = {"created_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": fingerprint(Path(model_path)),
              "algorithm_files_sha256": {p.name: fingerprint(p) for p in sorted((Path(__file__).resolve().parents[1]/"segmentation").glob("*.py"))},
              "reference_record_sha256": {p.name: fingerprint(p) for p in sorted((manifest.parent/"annotations/records").glob("*.json"))},
              "manifest_sha256": fingerprint(manifest), "configs": configs,
              "config_file_sha256": {mode: fingerprint(Path(path)) for mode, path in (("full", full_path), ("diagonal", diagonal_path))},
              "test_status": "Exploratory integration evaluation: A test results were previously published. No new blind-test claim.",
              "target_subset": {"tag": "frontal", "split": "test", "target": .65,
                  "scope": "controlled frontal portraits, not a verified small-angle or arbitrary normal-light subset"}}
    write_json(output_dir/"freeze.json", freeze)
    report = {"freeze": freeze, "environment": environment(), "results": {},
              "reference_limitation": "Project assisted silver references; not official pixel truth and not independently expert-reviewed.",
              "metric_conventions": "Undefined denominator -> null. Positive reference with empty prediction -> IoU/Dice 0. Empty/empty excluded from positive-image mean. Background FP contributes to accumulated counts.",
              "scene_scope": "External scenes have unknown identities; no proof of identity independence. Tags overlap. Hand crops are derivatives, not independent originals."}
    csv_rows = []
    for mode, config in configs.items():
        function = segment_image if mode == "full" else segment_image_baseline
        mode_report = {}
        for collection, samples in collections.items():
            rows = []
            for index, sample in enumerate(samples, 1):
                image, skin_ref, face_ref = sample.load()
                result = function(image, full_model, config)
                validate_result(result, image.shape[:2])
                row = {"image_id": sample.image_id, "scene_tags": sample.tags, "shape": list(image.shape[:2]),
                       "timing_ms": result["timing_ms"], "candidate_count": len(result["candidates"]),
                       "candidates": result["candidates"], "image_sha256": fingerprint(sample.photo),
                       "skin_reference_sha256": fingerprint(sample.packed_path or sample.skin_path),
                       "face_reference_sha256": fingerprint(sample.packed_path or sample.face_path),
                       "parent_image_id": sample.parent, "crop_xyxy": sample.crop,
                       "skin": metrics(counts(result["skin_mask"], skin_ref)),
                       "face": metrics(counts(result["face_mask"], face_ref))}
                rows.append(row)
                csv_rows.append({"mode": mode, "collection": collection, "image_id": sample.image_id,
                                 "height": image.shape[0], "width": image.shape[1], "timing_ms": result["timing_ms"],
                                 "candidate_count": row["candidate_count"],
                                 **{kind+"_"+key: value for kind in ("skin", "face") for key, value in row[kind].items()}})
                # Compact derived evidence only; no redistribution of original photographs.
                if mode == "full" and collection in ("test", "external_scenes", "derived_hands"):
                    path = output_dir/"evidence"/collection/(sample.image_id+".jpg")
                    evidence_sheet(path, image, result, skin_ref, face_ref,
                                   f"{sample.image_id} | face IoU: {row['face']['iou']} | candidates: {row['candidate_count']}")
                    row["evidence"] = path.relative_to(output_dir).as_posix()
                if index % 10 == 0 or index == len(samples):
                    print(f"Evaluate {mode} {collection}: {index}/{len(samples)}", flush=True)
            mode_report[collection] = {"summary": summarize(rows),
                "by_scene": {tag: summarize([r for r in rows if tag in r["scene_tags"]]) for tag in sorted({t for r in rows for t in r["scene_tags"]})},
                "per_image": rows}
        subset = mode_report["test"]["by_scene"].get("frontal", {"images": 0, "face": {"mean_positive_image_iou": None}})
        score = subset["face"]["mean_positive_image_iou"]
        mode_report["target_check"] = {"images": subset["images"], "mean_face_iou": score, "target": .65,
                                       "observed_target_met": score >= .65 if score is not None else None,
                                       "qualification": "Exploratory on previously reported reference set; no new blind-test certification"}
        report["results"][mode] = mode_report
    write_json(output_dir/"evaluation.json", report)
    with (output_dir/"per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_report(report, output_dir)
    return report


def write_report(report, folder):
    def val(value):
        return "不适用" if value is None else f"{value:.4f}"
    lines = ["# 最终集成实验报告", "", "## 设计思路", "",
             "采用 A 的真实 Cb–Cr 二维高斯模型，接入 B 的马氏距离分割、形态学、连通域几何筛选与受限填洞；C 完成 EXIF 读取、Tkinter 展示、可追溯导出、验证集校准和评估。完整协方差与独立通道模型分别在同一验证集选择配置。", "",
             "输入为 uint8 BGR，OpenCV 返回 Y/Cr/Cb，模型按 Cb/Cr 取特征。仅在推理时加一次 λI；d²≤T 为肤色。保留所有通过筛选的连通域，不强行保留最大区域。填洞按候选区域面积限制；最终掩膜为原尺寸 uint8 0/1。", "",
             "## 实验过程", "",
             "主清单 244 张：训练 120、验证 100、测试 24；48 位作者编号人物按 24/16/8 隔离，20 张无脸背景全部在验证集。外部场景 33 张另列，两张手部裁剪不增加独立原图数。主数据与场景分别核对原图哈希；场景位编码参考先核对哈希、尺寸和取值。", "",
             "模型、数据清单、配置及逐图原图/参考哈希保存在 freeze.json、evaluation.json 与 calibration.json。校准只读 validation 参考：6 个距离平方阈值 × 3 套后处理，共每模型 18 组；以包括无脸 FP 的验证集累计人脸 IoU 最大为选择依据，精确并列时先枚举者优先。随后冻结配置，再评估现有测试和外部场景。", "",
             "**评估性质：探索性集成评估。** A 已公开现有测试与场景结果，本轮没有新的盲测集。参考为辅助标注并经项目复核，未经独立专家逐像素精修；结果表示与该参考的一致程度。", "",
             "## 实际结果", "", "| 模型 / 数据 | 数量 | 肤色 P / R / IoU / Dice | 人脸 P / R / IoU / Dice | 有脸逐图平均 IoU |",
             "| --- | ---: | --- | --- | ---: |"]
    names = {"validation": "验证集", "test": "现有测试", "external_scenes": "外部场景", "derived_hands": "手部派生裁剪"}
    for mode, result in report["results"].items():
        for collection, label in names.items():
            s = result[collection]["summary"]
            lines.append(f"| {mode} / {label} | {s['images']} | " + " / ".join(val(s['skin'][k]) for k in ('precision','recall','iou','dice'))+" | "+" / ".join(val(s['face'][k]) for k in ('precision','recall','iou','dice'))+f" | {val(s['face']['mean_positive_image_iou'])} |")
    lines += ["", "主结果以累计 TP/FP/FN 计算；各图权重不均。逐图平均只包含真值非空图片。真值、预测均为空的 IoU/Dice 为不适用；正真值、空预测的 IoU/Dice 为 0。空分母不记为满分。", "", "### 无脸结果与目标子集", ""]
    full_test = report["results"]["full"]["test"]["summary"]
    diag_test = report["results"]["diagonal"]["test"]["summary"]
    lines.insert(lines.index("### 无脸结果与目标子集"),
        f"对照解释：完整协方差 / 独立通道在现有测试上的肤色累计 IoU 为 {val(full_test['skin']['iou'])} / {val(diag_test['skin']['iou'])}，人脸累计 IoU 为 {val(full_test['face']['iou'])} / {val(diag_test['face']['iou'])}；人脸逐图平均为 {val(full_test['face']['mean_positive_image_iou'])} / {val(diag_test['face']['mean_positive_image_iou'])}。累计指标与逐图指标反映不同权重，结果不支持宣称完整协方差在每一种口径上均优于独立通道。\n")
    for mode, result in report["results"].items():
        config = report["freeze"]["configs"][mode]
        lines.append(f"- {mode}：T={config['distance_squared_threshold']}；配置版本 `{config['config_revision']}`。")
        for collection in ("validation", "test", "derived_hands"):
            nf = result[collection]["summary"]["no_face"]
            lines.append(f"- {mode} {names[collection]}：无脸 {nf['images']} 张，非空预测 {nf['nonempty_predictions']} 张，图片误检率 {val(nf['image_false_positive_rate'])}；误检像素 {nf['false_positive_pixels']}。")
        target = result["target_check"]
        lines.append(f"- {mode} 预先固定 frontal 标签子集：{target['images']} 张，人脸逐图平均 IoU={val(target['mean_face_iou'])}，目标 0.65；观测{'达到' if target['observed_target_met'] else '未达到'}。该子集仅代表受控正脸，不覆盖未标定的小角度或任意普通光照条件；不构成新盲测验收。")
    lines += ["", "### 场景和耗时", "", "| 模型 / 数据 | 算法中位 ms | P95 ms | 最慢 ms |", "| --- | ---: | ---: | ---: |"]
    for mode, result in report["results"].items():
        for collection in names:
            t = result[collection]["summary"]["timing"]
            lines.append(f"| {mode} / {names[collection]} | {val(t['median_ms'])} | {val(t['p95_ms'])} | {val(t['max_ms'])} |")
    lines += ["", "计时只含分割与后处理，不含读取、标注加载、界面、导出；所有图片按转正原尺寸处理。逐图分辨率、耗时和候选保存在 per_image.csv / evaluation.json；单次运行计时受缓存和系统负载影响。", "", "```json", json.dumps(report["environment"], ensure_ascii=False, indent=2), "```", "", "| 外部场景标签（可重叠） | 样本数 | full 人脸累计 IoU | full 逐图平均 IoU |", "| --- | ---: | ---: | ---: |"]
    for tag, s in report["results"]["full"]["external_scenes"]["by_scene"].items():
        lines.append(f"| {tag} | {s['images']} | {val(s['face']['iou'])} | {val(s['face']['mean_positive_image_iou'])} |")
    lines += ["", "### 失败案例与演示素材", "", "以下根据冻结结果挑选低 IoU 样本用于解释，不据此继续改参数。完整测试与场景预览在 evidence/，包含原图、两层预测、叠加及两层参考。这里只在本地作课程/研究示例，原图与整批预览不默认发布到公开仓库。", ""]
    for collection in ("test", "external_scenes", "derived_hands"):
        rows = report["results"]["full"][collection]["per_image"]
        selected = sorted(rows, key=lambda r: r['face']['iou'] if r['face']['iou'] is not None else -1)[:2]
        for r in selected:
            lines += [f"- {names[collection]} `{r['image_id']}`：人脸 IoU={val(r['face']['iou'])}，候选 {r['candidate_count']}；[六联预览]({r['evidence']})。"]
    lines += ["", "视觉复核：侧脸 hfd_5_AFA8 中肤色沿脸颈连通，人脸后处理没有保留该区域，造成完全漏检；场景 wf_1_Handshaking_Handshaking_1_380 中面部肤色分类已大量漏失，最终仅保留背景小区域。多人界面示例同时保留多个候选，也显示了手臂和衣物误检。因此‘保留多候选’不能等同于正确检出每张人脸。", "",
              "### 运行验证与模型复现", "",
              "自动检查、真实 Tk 窗口、导出回读及截图复核见 [VERIFICATION.md](VERIFICATION.md)。训练复跑没有覆盖原模型；[model_reproduction.json](model_reproduction.json) 分别记录数值、原始文件及历史解码摘要的差异。均值一致、协方差误差约 1e-14；全部 488 张标注与冻结参考像素一致。生成 PNG 的压缩字节不同，另有 5 张训练图的历史解码摘要不同，故不宣称对 A 的字节级完全复现。", ""]
    lines += ["", "## 实验总结", "",
             "系统已打通真实模型、分割、界面、导出和评估。颜色模型不能判断区域究竟是脸、手还是背景；几何规则也不能可靠分离脸颈。多人保留属于接口与连通域功能保证，复杂场景的实际召回率仍以本报告为准。正脸/侧脸分开统计，不能将单个受控子集成绩推广到所有场景。", "",
             "未知身份场景无法证明人物完全独立；主测试缺少未使用的无脸负样本，故测试误检率不适用。偏色仅少量样本，脸颈相连没有独立冻结标签，不能给出这类总体准确率。改进方向为新增专家精修参考、更多独立场景及无脸留出、面部几何约束与脸颈分离；如依据本轮结果继续调参，必须保留探索性标签，或另建未使用的盲测集。", ""]
    (folder/"FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
