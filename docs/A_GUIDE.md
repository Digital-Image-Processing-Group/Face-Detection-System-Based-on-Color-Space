# A 分支数据与模型使用说明

A 分支交付 RADIATE 数据整理、双层多边形标注、人物独立划分、YCrCb 肤色统计模型及独立人物验证结果。唯一数据清单为 `data/manifest.csv`，唯一统计模型为 `models/skin_model.json`。

## 环境

在仓库根目录的 PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-A.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests/data -v
```

## 数据与标注

当前本地包含 22 个作者匿名人物组、44 张 1500×1500 彩色照片。训练集 13 组 26 张、验证集 5 组 10 张、测试集 4 组 8 张；同一人物的 NC、HC 两张照片始终在同一集合。固定分组见 `data/split.json`，不要重新划分已使用的测试集。

图片位于 `data/images/radiate/`。44 张照片均有完整覆盖的 skin、face 多边形，位于 `data/annotations/polygons/batch_*.json`，每个批次 4 张。它们由 Codex 逐图查看后绘制并检查叠加图，是项目自制的近似多边形参考标注，来源未提供像素真值，也未经过独立专家复核。`complete` 表示覆盖范围已检查，不代表像素边缘完全准确。细则见 [标注规范](annotation_guide.md)。

本地数据齐全时，采集命令核对来源 SHA256 后直接复用，不访问网络；在新环境中依据官方 ZIP 成员恢复照片并保留清单现有划分和标注字段。下载只支持这一来源，不会合并其他数据集。

```powershell
.\.venv\Scripts\python.exe -m src.training collect-radiate
.\.venv\Scripts\python.exe -m src.training rasterize --manifest data/manifest.csv --polygons data/annotations/polygons
.\.venv\Scripts\python.exe -m src.training contact-sheet --manifest data/manifest.csv --output artifacts/radiate_review/overlay_00.png --offset 0 --limit 4 --overlays --tile-size 600 --columns 2
.\.venv\Scripts\python.exe -m src.training validate --manifest data/manifest.csv --purpose evaluation --report artifacts/radiate_model/dataset_validation.json
```

对 offset=0、4、8……40 分别生成叠加图即可检查全部照片。源码、多边形、清单、来源摘要和统计模型纳入 Git；原图、可重建 PNG 和本地检查图由 Git 忽略。原图使用条件见 [数据来源](DATA_SOURCES.md)，不随公开仓库重新分发。

## 训练与验证

```powershell
.\.venv\Scripts\python.exe -m src.training train --manifest data/manifest.csv --output models/skin_model.json --diagnostics artifacts/radiate_model --samples-per-image 2000 --seed 42 --regularization 1 --mask-margin 15
.\.venv\Scripts\python.exe -m src.training assess --manifest data/manifest.csv --model models/skin_model.json --output models/evaluation.json --operating-point models/operating_point.json --previews artifacts/radiate_predictions
```

训练入口要求人物分组已核验、两类标注完整，并检查跨集合人物及重复图片。只从 `train` 读取正皮肤像素。训练时将肤色区域内缩 15 个原图像素以减小边界误差影响，评估参考掩膜保持原尺寸和完整范围。每图固定无放回抽样 2000 个像素，26 图共 52000 个。模型记录种子、逐图来源哈希、掩膜哈希和内缩参数，重复训练应得到相同版本。

OpenCV 返回 Y、Cr、Cb，实际模型显式取 `[Cb, Cr]`。模型存原始样本协方差，推理时使用 `covariance + regularization_lambda * I`。均值、方差、散点图和训练检查输出到 `artifacts/radiate_model/`；方差为协方差矩阵的对角元素。正则项 λ=1 在测试前固定。

评估在预先固定的 T=0.5 至 20、步长 0.25 的网格上，按验证集累计肤色 Dice 选择马氏距离平方阈值，平局取更小值。写出 operating_point 后再运行测试指标，不用测试标签选择参数。当前 T=6.5。评估拒绝清单版本不符或训练图片、训练掩膜已变更的模型。

肤色结果为 `d² <= T`。人脸区域评估另使用固定的开闭运算、最大连通区域和填洞作为受控单人照片基线。它用于检查模型交付，B 的完整多人候选算法仍按任务书开发；不能将此基线当作多人检测成品。

## 输出及交接

| 文件或目录 | 内容 |
| --- | --- |
| data/manifest.csv、data/split.json | 唯一数据清单、冻结人物分组及清单哈希 |
| data/sources/radiate_downloads.json | 官方成员路径、作者编号、许可说明和原文件 SHA256 |
| data/annotations/polygons/ | 可重建两类参考掩膜的逐图多边形及标注来源 |
| models/skin_model.json | 均值、原始协方差、λ、版本、采样和训练文件哈希 |
| models/operating_point.json | 验证集选择的距离平方阈值、版本和固定后处理说明 |
| models/evaluation.json | 验证与测试 Precision、Recall、IoU、Dice、逐图结果、标注哈希、环境和耗时 |
| artifacts/radiate_model/ | Cb–Cr 分布图、统计摘要和数据校验报告 |
| artifacts/radiate_predictions/ | 验证与测试预测 PNG，0/255 预览 |

算法内部掩膜为二维 uint8 0/1；PNG 预览为 0/255。肤色与面部定义不同，B、C 必须分别读取。原始照片恢复和多边形栅格化足以在新环境重建掩膜，交付不依赖下载缓存。

测试结果已查看。此后若根据测试失败案例修改参数或模型，必须说明测试集已参与开发，另取未使用人物评估，不能继续宣称它是新的独立最终结果。当前结果及覆盖边界见 [交付状态](A_STATUS.md)。
