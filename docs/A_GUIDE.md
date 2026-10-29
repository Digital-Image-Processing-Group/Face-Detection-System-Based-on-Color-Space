# A 交付与使用说明

主实验清单为 `data/manifest.csv`，统计模型为 `models/skin_model.json`，33 张外部场景另有来源清单。A 的数据、标注参考、统计模型和基线报告已交付；B/C 完整系统尚未验收。具体工作见 [B 交接](HANDOFF_B.md)、[C 交接](HANDOFF_C.md)，来源与使用条件见 [数据来源](DATA_SOURCES.md)。以下路径均相对仓库根目录。

## 交付文件与结果

| 文件 | 用途 |
| --- | --- |
| data/manifest.csv、data/split.json | 244 张主数据及冻结划分：训练120、验证100、测试24 |
| data/sources/ | 作者标签、下载地址、原图哈希及33张外部场景 records |
| data/annotations/records/、reference/、scenes/及标注记录 | 双层参考、重建定义、复核与修正来源；子目录均在data/annotations/下 |
| models/skin_model.json | schema_version=1、model_revision=skin-d239a8aaf4ae7f05、feature_order=[Cb,Cr]、λ=1 |
| models/operating_point.json | distance_squared_threshold=4.0，与模型版本及清单哈希绑定 |
| models/cbcr_distribution.png | 已交付模型的训练肤色分布图 |
| models/evaluation.json、models/stress_evaluation.json | A诊断基线的主评估与外部场景结果，不代表B完整算法 |
| models/delivery_checks.json | 主数据哈希、488张掩膜重建及拟合复现结果 |
| src/training/、tests/data/、requirements-A.txt | A工具、数据/模型检查和运行依赖 |

高斯均值、原始协方差和训练统计直接读取模型的 mean、covariance、training_info，不维护重复参数文件。颜色表示为 OpenCV-YCrCb-uint8-full-range，通道特征顺序固定[Cb,Cr]，原始Σ仅在推理时加一次λI。数据/参考、模型或配置改变时同步更新配套哈希、版本及报告。

| 新人物测试24张 | 肤色 | 简易人脸基线 |
| --- | --- | --- |
| Precision | 0.9356 | 0.6758 |
| Recall | 0.8378 | 0.8708 |
| 累计像素IoU | 0.7921 | 0.6142 |
| Dice | 0.8840 | 0.7610 |
| 逐图平均IoU | 0.7828 | 0.6160 |

其中正脸8张平均人脸IoU=0.7135，侧面16张结果见报告。主测试全有人脸，无脸误检率不适用；验证背景误检16/20。33张外部原图的累计肤色/人脸基线IoU为0.1951/0.1332，两个纯手部裁剪误检2/2。场景标签重叠：复杂背景33、多人11、手脸干扰30、偏色或不均匀光照3。参考是项目辅助标注，边缘近似，未经独立专家精修；指标只反映对该参考的一致程度。

人物编号只在可获得时辅助实验分组，不是检测输入或所有数据的准入条件；无编号数据按照片/场景分组并披露身份重叠风险。现有24张测试和33张场景的结果已报告；如使用反馈继续开发，新结果须标为探索性，或另设未用数据盲测，不将已用于开发的数据再称为未见测试。

## B/C 共用接口（待实现）

当前src下仅有training模块，尚无 `src/segmentation/`、`config/pipeline.json`、`src/app/`、`src/evaluation/` 或完整GUI。B提供 `segment_image(image_bgr, model, config) -> dict`，C按以下契约接入：

| 项目 | 约定 |
| --- | --- |
| 输入 | EXIF转正后的(H,W,3) uint8 BGR照片、已加载模型、配置 |
| skin_mask、face_mask | uint8、(H,W)、值仅0/1，尺寸与转正原图一致 |
| candidates | list[dict]，每项含bbox=(x,y,width,height)及area前景像素数 |
| timing_ms | 非负处理耗时，不含读取、UI及导出 |

bbox零起始、左上原点、右下边界不包含。无候选时face_mask全零、candidates为空，skin_mask保留实际分类结果；非法输入及模型/配置错配须明确报错。配置使用distance_squared_threshold、model_revision、postprocess。B改变阈值或后处理时保存新配置及算法版本，由C另存实际B算法报告。

A的assessment.face_proposal仅是最大连通域诊断基线，会丢多脸，不能充当完整分割入口。assess与stress_assessment内部固定调用该基线，C须另接B入口评估。

## 安装与恢复数据

在仓库根目录 PowerShell 执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-A.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests/data -v
.\.venv\Scripts\python.exe -m src.training collect
.\.venv\Scripts\python.exe -m src.training rasterize --manifest data/manifest.csv
.\.venv\Scripts\python.exe -m src.training validate --manifest data/manifest.csv --purpose evaluation
```

`collect` 依据固定来源恢复缺失原图并检查 SHA256，不修改划分和标注字段；齐全时不联网，损坏时报错。需访问作者 OSF 与 Berkeley 站点，遵守 [使用条件](DATA_SOURCES.md)。

`rasterize` 从 Git 内的参考及记录重建 488 张 skin/face PNG，不需要辅助模型和 MediaPipe。压缩参考为位编码 0/1/2/3，不可直接当二值图；生成 PNG 为 0/255，`read_mask` 读取后为 0/1。32 项检查属于 A 的数据与模型单元测试，不等于 B/C 集成或界面验收。

B/C 接入现有交付无需重新训练或覆盖报告。需要核验原实验时，可明确执行以下复现命令；它们会重写默认模型、评估和审计产物：

```powershell
.\.venv\Scripts\python.exe -m src.training train --manifest data/manifest.csv --samples-per-image 2000 --seed 42 --regularization 1 --mask-margin 15
.\.venv\Scripts\python.exe -m src.training assess --manifest data/manifest.csv
.\.venv\Scripts\python.exe -m src.training.audit
```

`assess` 用验证集定阈值，先保存 operating_point 再报告测试；它调用 A 的简易人脸基线，不能用于宣称 B 算法已评估。重复执行仅适合固定配置复现，不要利用已报告测试结果调参。实际依赖版本保存在模型和评估报告中。通过 Python API 复现训练时须显式传入 `mask_margin=15`，`fit_model` 的 API 默认值为 0，CLI 默认值为 15。

## 数据与查看

224 张 HFD 对应 48 位作者编码人物：原 40 人各五张，新增 8 人各三张。原测试已移入验证；新人物在看预测前固定为最终留出。主清单训练/验证/测试 120/100/24 张，人物 24/16/8 人；20 张背景全部在验证集。分组与完整性见 data/split.json。

另有 33 张真实 WIDER-FAIR 场景，作者逐脸感知族群标为 Asian，无人物编号。当前冻结实验将其单独用于外部压力测试，不进入拟合或选参；这不是无编号数据不得用于人脸检测的通用限制。恢复原图：

```powershell
.\.venv\Scripts\python.exe -m src.training.public_sources restore-scenes
```

已交付压力结果在 `models/stress_evaluation.json`。如需重跑 A 基线，执行 `.\.venv\Scripts\python.exe -m src.training.stress_assessment`，会重写该报告并生成两张手部裁剪；它不选择阈值，必须使用同版本 skin_model 与 operating_point。裁剪坐标与父图来源在报告中，不计为独立原图。B 的算法需 C 另接评估入口。

主清单所有 244 张都有整图双层参考。人像由独立模型初始化后 Codex 逐图复核，尚无独立人工/专家审核；不是官方像素真值。定义见 [标注规范](annotation_guide.md)。检查拒绝人物跨集合、完全重复或近重复 RGB 图跨集合、尺寸与编码错误。受控人像易发生粗感知哈希碰撞，保留相关警告，不凭碰撞判定身份；人物分组依据作者编码。

```powershell
.\.venv\Scripts\python.exe -m src.training contact-sheet --manifest data/manifest.csv --output artifacts/review/overlay_000.png --offset 0 --limit 5 --overlays --tile-size 600 --columns 3
```

`offset=0、5……195` 配合 `limit=5` 查看原 200 张人像；背景使用 `offset=200、210`、`limit=10`；新增 24 张人像使用 `offset=220、226、232、238`、`limit=6`。蓝色为 face、绿色为 skin，交集颜色叠加。33 张场景不在主清单中，需按场景来源清单另行查看。重训时工具默认在 `artifacts/model/` 生成分布图及统计，在 `artifacts/predictions/` 生成预测预览；这些中间产物不纳入Git，也不作为保留交付物。本轮分布图单独交付于 `models/cbcr_distribution.png`，训练统计以模型JSON为准。

每张训练图的 skin 向内腐蚀 15 原始像素，均衡无放回取 2000 像素，共 240000 像素；特征 `[Cb, Cr]`，样本协方差 ddof=1。仅在距离计算时加 λI，不重复加到导出协方差。不对输入整图重新拟合肤色。

## 可选标注工具

只在新增标注时安装 `requirements-annotation-tools.txt`，再运行 `python -m src.training.public_sources annotation-assets` 获取固定权重。`assisted_annotations --manifest <仅含待标注人像的清单>` 产生 sampled/draft；逐图查看，记录哈希与修正后才可用 `--finalize-review <记录> --output <同目录输出清单>` 定稿。复核记录应如实写明审核者及限制。

草稿不能直接评估；不要对含背景的总清单运行人像初始化。新增批次须用 `--record-name` 指定新的记录名，避免默认名称覆盖现有记录；先运行该子命令的 `--help` 核对参数。
