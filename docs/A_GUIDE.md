# A 分支数据与模型工具使用说明

A 分支负责公开样本采集、标注管理、数据分组校验、肤色均值与协方差训练及模型导出。训练入口是 `python -m src.training`，不包含 B 的分割算法或 C 的产品界面。

## 安装与验证

在仓库根目录运行以下 PowerShell 命令。已有 `.venv` 时直接使用该环境；独立安装无需使用系统包。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-A.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests/data -v
.\.venv\Scripts\python.exe -m src.training --help
```

工具使用 NumPy、OpenCV、Pillow 和 Matplotlib，测试使用 Python 自带 unittest。当前本地环境为 Python 3.13，项目虚拟环境复用了已安装的数值计算库，并单独安装 OpenCV。

## 获取与恢复公开样本

```powershell
.\.venv\Scripts\python.exe -m src.training collect-fairface --manifest data/manifest.csv --count 200 --pages 20
.\.venv\Scripts\python.exe -m src.training collect-beans --manifest data/manifest.csv --count 20
```

人脸来自 FairFace 的 Hugging Face 镜像，按上游作者的 East Asian 与 Southeast Asian 标签和成人年龄组筛选，本项目不从照片推断族群。这个子集只覆盖东亚与东南亚来源标签，不代表整个亚洲人群。原始照片及缓存保留在本地并由 Git 忽略，来源、许可、上游行号和下载哈希记录在 `data/sources/`。

叶片来源用于补充候选无脸图片；采集程序不会自动赋予全零真值，必须先查看照片，确定其中是否有脸或手部皮肤，再标注。此类图片不足以代表真实复杂场景负样本。

采集支持重新运行：保留已有图片及标注字段。网络失败时可重试同一条命令；`--pages` 限制 FairFace 元数据查询页数，每页最多 100 条。签名图片地址只存在本地缓存，不写入交付的来源清单；缓存中的图片地址过期后需重新获取相关元数据缓存文件。

FairFace 镜像缺少可靠人物身份字段，所以照片统一放入训练集合中的未核验来源组。它们可训练联调模型，不能直接构成宣称人物独立的验证集或测试集。不得把上游行号当作人物编号。

## 查看和修改标注

```powershell
.\.venv\Scripts\python.exe -m src.training contact-sheet --manifest data/manifest.csv --output data/contact_sheets/first20.png --limit 20
.\.venv\Scripts\python.exe -m src.training rasterize --manifest data/manifest.csv --polygons data/annotations/polygons
.\.venv\Scripts\python.exe -m src.training contact-sheet --manifest data/manifest.csv --output data/contact_sheets/first20_overlay.png --limit 20 --overlays
```

依照 `docs/annotation_guide.md` 编辑对应图片的多边形 JSON，再重新生成掩膜。栅格化命令会更新清单中的状态和掩膜路径。首批人脸皮肤采样仅为正样本，粗略面部轮廓保持 draft 状态。

批次标注位于 polygons/batches 下，栅格化自动递归处理。目前 200 张人脸均有轮廓草稿，199 张有皮肤采样。1 张面部涂抹照片及 3 张明显去色照片排除出颜色训练，实际用于模型拟合的为 196 张。新增椭圆草稿不能用于正式面部像素指标。

## 校验与训练

```powershell
.\.venv\Scripts\python.exe -m src.training validate --manifest data/manifest.csv --purpose train --report artifacts/A/dataset_validation.json
.\.venv\Scripts\python.exe -m src.training train --manifest data/manifest.csv --output models/skin_model.json --diagnostics artifacts/A --samples-per-image 2000 --seed 42 --regularization 1
```

训练只读取 split=train 且已标注的正皮肤像素。每张有效图片抽取相同数量、不重复的像素，数量取请求上限与各图可用像素数最小值。相同图像重复项不会重复进入训练。随机种子与图片编号共同确定抽样，固定数据和参数可以重复得到相同模型。

带 exclude_color_training 标签的图片不会进入拟合。标注区域内至少 98% 像素的通道最大差不超过 2 时，也会以缺少颜色信息为由跳过，避免把 RGB 格式保存的纯黑白皮肤混入统计。跳过原因记录在模型 training_info.skipped 中。

训练输出包含 `models/skin_model.json`、Cb–Cr 分布图、统计摘要和数据检查报告。二维协方差采用样本估计；模型保存原始矩阵，不预先加正则项。图中的二倍标准差轮廓仅用于观察分布，不是 B 校准后的分割阈值。默认 λ=1 只是初始数值设置，后续与 B 在验证集上联合确定。

## 正式分组与评估的前提

只有在有可靠人物标识和采集组、两类完整标注之后，才能生成正式划分。保持一份明确 split 为空的独立清单，使用不同输出文件保留原数据版本。

```powershell
.\.venv\Scripts\python.exe -m src.training split --manifest data/manifest.unsplit.csv --output data/manifest.verified.csv --seed 42 --ratios 0.6 0.2 0.2
.\.venv\Scripts\python.exe -m src.training validate --manifest data/manifest.verified.csv --purpose evaluation --report artifacts/A/evaluation_readiness.json
```

划分按共同人物及采集组的传递关系执行，不能强行拆分关联组。工具拒绝未核验身份、已有划分、少于三个独立组以及不完整评估标注，校验时也检查解码后完全重复和感知哈希相近的跨集合图片。感知哈希只能提示部分近似重复，不能证明人物身份独立；有误报时需复核、合并关联组或选取新的独立样本。

输出划分比例受组大小约束，实际各集合数量写入旁边的 split JSON。验证集供 B 调参，测试集在模型及配置冻结后交给 C 评估。

## B 与 C 的交接

| 接收方 | 交付内容与约定 |
| --- | --- |
| B | JSON 模型；OpenCV 返回顺序 Y、Cr、Cb，特征必须显式取 Cb、Cr；推理使用原始 covariance 加 λI；使用马氏距离平方；阈值由 B 校准 |
| C | 图片、两类标注、清单及状态；PNG 0/255 读取为 0/1；不能用 sampled 或 draft 标注计算正式像素指标 |
| 三人共用 | 模型版本、数据清单版本、来源与标注规范；模型的 bootstrap_only 标记需保留 |

模型字段为 schema_version、model_revision、color_space、feature_order、mean、covariance、regularization_lambda、training_info。`training_info` 记录采样数量、种子、图片及掩膜哈希、数据版本、环境版本和跳过原因；A 不写 B 的阈值或后处理配置。

## 来源

- [FairFace 作者仓库](https://github.com/joojs/fairface)：作者声明 CC BY 4.0，并提供论文引用。
- [FairFace 镜像及标签定义](https://huggingface.co/datasets/HuggingFaceM4/FairFace)：样本下载入口。
- [Makerere Beans 数据集](https://huggingface.co/datasets/AI-Lab-Makerere/beans)及[作者许可](https://github.com/AI-Lab-Makerere/ibean/blob/master/LICENSE)：MIT。
- [Hugging Face 数据行接口](https://huggingface.co/docs/dataset-viewer/rows)：按需下载少量图片，避免下载整个数据集。
- [OpenCV 颜色转换说明](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)：核对 YCrCb 通道与 8 位数值范围。
