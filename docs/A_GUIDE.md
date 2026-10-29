# A 分支使用说明

唯一正式清单为 `data/manifest.csv`，唯一统计模型为 `models/skin_model.json`。结果与限制见 [状态](A_STATUS.md)，B、C 接口见 [交接](A_HANDOFF.md)。

## 安装与复现

在仓库根目录 PowerShell 执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-A.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests/data -v
.\.venv\Scripts\python.exe -m src.training collect
.\.venv\Scripts\python.exe -m src.training rasterize --manifest data/manifest.csv
.\.venv\Scripts\python.exe -m src.training validate --manifest data/manifest.csv --purpose evaluation --report artifacts/model/dataset_validation.json
.\.venv\Scripts\python.exe -m src.training train --manifest data/manifest.csv --samples-per-image 2000 --seed 42 --regularization 1 --mask-margin 15
.\.venv\Scripts\python.exe -m src.training assess --manifest data/manifest.csv
.\.venv\Scripts\python.exe -m src.training.audit
```

`collect` 依据固定来源恢复缺失原图并检查 SHA256，不修改划分和标注字段；齐全时不联网，损坏时报错。需访问作者 OSF 与 Berkeley 站点，遵守 [使用条件](DATA_SOURCES.md)。

`rasterize` 从 Git 内的参考及记录重建 440 张 skin/face PNG，不需要辅助模型和 MediaPipe。压缩参考为位编码 0/1/2/3，不可直接当二值图；生成 PNG 为 0/255，`read_mask` 读取后为 0/1。

`assess` 用验证集定阈值，先保存 operating_point 再报告测试。重复执行仅适合固定配置复现，不要利用已报告测试结果调参。实际依赖版本保存在模型和评估报告中。

## 数据与查看

224 张 HFD 对应 48 位作者编码人物：原 40 人各五张，新增 8 人各三张。原测试已移入验证；新人物在看预测前固定为最终留出。主清单训练/验证/测试 120/100/24 张，人物 24/16/8 人；20 张背景全部在验证集。分组与完整性见 data/split.json。

另有 33 张真实 WIDER-FAIR 场景，作者逐脸标为 Asian，但无人物编号，所以只作外部压力测试。恢复与运行：

```powershell
.\.venv\Scripts\python.exe -m src.training.public_sources restore-scenes
.\.venv\Scripts\python.exe -m src.training.stress_assessment
```

压力测试不选阈值，必须使用同版本 skin_model 与 operating_point；结果在 models/stress_evaluation.json。两张纯手部图由真实原图裁剪得到，坐标与父图来源在报告中；不计为独立原图。

主清单所有 244 张都有整图双层参考。人像由独立模型初始化后 Codex 逐图复核，尚无独立人工/专家审核；不是官方像素真值。定义见 [标注规范](annotation_guide.md)。检查拒绝人物跨集合、完全重复或近重复 RGB 图跨集合、尺寸与编码错误。受控人像易发生粗感知哈希碰撞，保留相关警告，不凭碰撞判定身份；人物分组依据作者编码。

```powershell
.\.venv\Scripts\python.exe -m src.training contact-sheet --manifest data/manifest.csv --output artifacts/review/overlay_000.png --offset 0 --limit 5 --overlays --tile-size 600 --columns 3
```

offset=0、5……195 查看全部人像；200、210 查看背景。蓝色为 face、绿色为 skin，交集颜色叠加。训练分布图为 `artifacts/model/cbcr_distribution.png`，统计为 `artifacts/model/training_summary.json`；验证/测试预测为 `artifacts/predictions/`。这些可重建产物不纳入 Git。

每张训练图的 skin 向内腐蚀 15 原始像素，均衡无放回取 2000 像素，共 240000 像素；特征 `[Cb, Cr]`，样本协方差 ddof=1。仅在距离计算时加 λI，不重复加到导出协方差。不对输入整图重新拟合肤色。

## 可选标注工具

只在新增标注时安装 `requirements-annotation-tools.txt`，再运行 `python -m src.training.public_sources annotation-assets` 获取固定权重。`assisted_annotations --manifest <仅含待标注人像的清单>` 产生 sampled/draft；逐图查看，记录哈希与修正后才可用 `--finalize-review <记录> --output <同目录输出清单>` 定稿。复核记录应如实写明审核者及限制。

草稿不能直接评估；不要对含背景的总清单运行人像初始化。
