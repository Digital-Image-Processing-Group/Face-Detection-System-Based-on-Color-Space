# 基于 YCbCr 颜色空间的人脸区域检测系统

A 的真实数据与高斯模型、B 的分割算法和 C 的桌面界面、导出、校准及评估已完成整合。输入彩色照片，输出肤色掩膜、近似人脸区域掩膜与候选叠加图。检测使用传统颜色统计和几何规则。

## 完整版分支

`full-project` 保存完整工程及最终课程提交源码。`src/` 包含训练、分割、界面和评估模块，`tests/` 包含自动检查，`models/`、`config/` 和 `reports/` 保存模型、参数与实验结果。`submission/` 保留精简提交版的单文件 Python 程序及参考掩膜，运行方式见 [提交版说明](submission/README.md)。

最新实验报告：[Word](reports/course/袁远冬24325312_闫若彤24325287_刘梓靖24325192_第三次project.docx) · [PDF](reports/course/袁远冬24325312_闫若彤24325287_刘梓靖24325192_第三次project.pdf)。

| 分工 | 成员 | 主要工作 |
| --- | --- | --- |
| A | 刘梓靖 24325192 | 数据、标注、划分与肤色统计模型 |
| B | 闫若彤 24325287 | 颜色分类、形态学与候选区域筛选 |
| C | 袁远冬 24325312 | 读取、界面、导出、评价、整合与报告 |

项目要求见 [TASKBOOK.md](TASKBOOK.md)，实际指标和限制见 [最终实验报告](reports/FINAL_REPORT.md)，联调记录见 [集成交付说明](docs/INTEGRATION.md)。

## 安装和启动

当前在 Windows、Python 3.14.2、Tk 8.6 上实测。建议使用带 Tkinter 的官方 Python 3.11 或更新版本；Tkinter 属于 Python 安装组件，不通过 pip 安装。首次安装在本文件夹的 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

然后双击 `run.bat`，或运行：

```powershell
.\.venv\Scripts\python.exe -m src.app
```

选择 JPG、JPEG 或 PNG 照片，界面同时显示原图、肤色、人脸及轮廓叠加；点击“保存检测结果”选择保存位置。支持中文路径，读取时先处理 EXIF 方向。透明图、灰度图、损坏文件和模型错配会明确报错；无候选是正常结果。检测与导出在工作线程中运行，窗口可缩放。

普通单图检测只需已交付的模型和配置，**无需下载课程数据或辅助标注模型**。Python 依赖范围在 requirements.txt；本轮实测的完整精确版本在 requirements-lock.txt。需要复现本轮依赖可在空虚拟环境执行 `python -m pip install -r requirements-lock.txt`。

## 命令行检测与导出

```powershell
.\.venv\Scripts\python.exe -m src.app --image "D:\照片\合照.jpg" --output "exports\合照结果"
```

输出目录必须是一个尚不存在的新目录，防止覆盖之前的结果。每次完整保存六个文件：

| 文件 | 内容 |
| --- | --- |
| skin_mask.npy / face_mask.npy | 转正原尺寸 uint8，只有 0 和 1 |
| skin_mask.png / face_mask.png | 转正原尺寸黑白预览，只有 0 和 255 |
| overlay.png | 原尺寸轮廓、编号框与近似面部区域 |
| result.json | 原图哈希、尺寸、候选坐标、面积、模型/配置版本与哈希、算法及读取耗时、导出文件哈希 |

坐标为左上原点、零起始，bbox 中的 x/y/width/height 对应半开区间。面积是候选填洞后的前景像素数。算法耗时不含读取、绘图或保存。结果先写临时目录，再整体发布；失败时不留下不完整的结果目录。

独立通道模型必须使用其单独校准的配置：

```powershell
.\.venv\Scripts\python.exe -m src.app --image "D:\照片\合照.jpg" --output "exports\对角模型结果" --diagonal --config config/baseline.json
```

## 恢复实验数据

使用前阅读 [来源与使用条件](docs/DATA_SOURCES.md)。人物原图不纳入 Git、不打包重新公开分发。恢复工具按固定来源下载缺失文件、校验 SHA256，保留清单和划分；全部齐全时无需联网。中断后可重复执行。

```powershell
.\.venv\Scripts\python.exe -m src.training collect
.\.venv\Scripts\python.exe -m src.training rasterize --manifest data/manifest.csv
.\.venv\Scripts\python.exe -m src.training.public_sources restore-scenes
.\.venv\Scripts\python.exe -m src.training validate --manifest data/manifest.csv --purpose evaluation --report reports/data_validation.json
```

实际范围：主清单 244 张（120 训练 / 100 验证 / 24 测试）、48 位作者编号人物；另有 33 张真实复杂场景，以及从其中两张派生的手部裁剪。20 张无脸背景均在验证集，测试无脸误检率因此不适用。多人场景无身份编号，单列压力评估。参考属于项目辅助标注，并非官方或独立专家像素真值。

## 重跑评估

已交付配置可直接运行冻结评估，无需重新选参：

```powershell
.\.venv\Scripts\python.exe -m src.evaluation run
```

明确需要重现选参时，再执行：

```powershell
.\.venv\Scripts\python.exe -m src.evaluation calibrate
.\.venv\Scripts\python.exe -m src.evaluation run
```

校准分别为完整协方差和独立通道模型比较 6 个阈值 × 3 套后处理配置，只读验证集参考；选择包含无脸误检像素的累计人脸 IoU 最大者。冻结配置后才运行评估。阈值字段为 `distance_squared_threshold`，含义是**马氏距离平方**阈值。`distance_threshold` 仅为兼容 B 原始配置的旧别名，两者同时出现且不同会拒绝。

评估结果保存在 reports/：

- `calibration.json`：全部候选、验证集逐文件哈希与选参依据。
- `freeze.json`：评估前冻结的模型、配置、数据清单哈希与目标子集。
- `evaluation.json` / `per_image.csv`：两层累计指标、逐图结果、重叠场景分类、无脸误检、硬件和计时。
- `FINAL_REPORT.md`：设计思路、实验过程、真实结果、失败案例和实验总结。
- `evidence/`：本地六联预览；`gui/`：真实窗口截图。含人物示例，只在本地保留，默认不进入 Git。

A 已公布过现有测试集结果，因此本轮明确为**探索性集成评估**，不宣称新增盲测。固定 `frontal` 子集仅代表受控正脸，不能概括未经标定的小角度或任意光照；任务书 0.65 是实验目标，实际是否达到见报告。若继续根据这些结果调整算法，应保持探索性标记，或另采未使用的最终留出数据。

## 训练与必要检查

A 的训练只使用训练集标注皮肤，均衡每张抽取 2000 像素、种子 42、标注内缩 15 像素、λ=1。训练输出另存，保护已交付模型：

```powershell
.\.venv\Scripts\python.exe -m src.training train --manifest data/manifest.csv --output models/retrained_skin_model.json --diagnostics artifacts/retrained --samples-per-image 2000 --seed 42 --regularization 1 --mask-margin 15
```

重新训练后的模型版本可能随依赖元数据改变；不要直接替换配套的正式模型和配置。如需评估新模型，使用独立配置和报告目录：

```powershell
.\.venv\Scripts\python.exe -m src.evaluation calibrate --model models/retrained_skin_model.json --config-dir artifacts/retrained/config --output artifacts/retrained/reports
.\.venv\Scripts\python.exe -m src.evaluation run --model models/retrained_skin_model.json --config-dir artifacts/retrained/config --output artifacts/retrained/reports
```

全部自动检查（无需下载实验照片）：

```powershell
powershell -ExecutionPolicy Bypass -File .\verify.ps1
```

检查覆盖 A 数据/训练、B 分割、C 指标算例、EXIF/BGR、透明与损坏图、模型/配置校验、多候选、空检测、导出回读与失败清理、测试数据不参与选参。测试中人工构造图片只用于行为验证，不进入正式指标。

## 模块接口

```python
from src.app.service import load_pipeline, load_photo
from src.segmentation import segment_image

model, config = load_pipeline()
bgr = load_photo("photo.jpg")
result = segment_image(bgr, model, config)
# result: skin_mask, face_mask, candidates, timing_ms
```

分割入口遵循任务书的字典返回契约；B 原始的四元组返回已在整合时统一，旧调用需改为按键读取。输入为转正后的 H×W×3 uint8 BGR；两层结果均为原尺寸 uint8 0/1。无候选时人脸掩膜全零、候选列表为空，肤色仍保留实际结果。

颜色和简单几何不能稳定区分脸、手、脖子和相似背景。合格连通域全部保留，但彼此连接或完全漏分的面部不保证被拆分/检出；实际复杂场景效果应查看报告和样例。

## 本分支的形态学实验

`codex/morphology-refinement` 从原始完整版独立比较减弱闭运算、提前填洞和候选尺度自适应。测试人像平均IoU从0.5878升至0.6135，复杂场景平均IoU从0.1482降至0.1324。实验入口、逐项对照、配置及完整结论见[形态学实验说明](experiments/morphology_refinement/README.md)。
