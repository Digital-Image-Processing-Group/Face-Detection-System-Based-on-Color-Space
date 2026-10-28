# 人物分组与像素标注的数据来源

更新日期：2026 年 10 月 28 日。已通过替换来源获得可靠匿名人物分组，以及现成的人工肤色掩膜。两种来源分别解决不同问题，未把肤色掩膜或轮廓草稿描述为完整人脸区域真值。

## 已下载的 RADIATE 彩色子集

RADIATE 作者论文描述了 109 位成人的彩色及黑白表情照片，其中有 22 位亚洲受试者。官方发布目录通过 AF、AM 分组和匿名人物编号组织文件；本项目仅按这些作者元数据选择彩色版本，没有根据照片推断族群或识别人物真实身份。

- 作者论文：[The Racially Diverse Affective Expression Face Stimulus Set](https://pmc.ncbi.nlm.nih.gov/articles/PMC6446554/)。
- 官方发布页及使用条件：[ABCD Study fMRI Tasks and Tools](https://abcdstudy.org/scientists/abcd-fmri-tasks-and-tools/)。
- 官方文件：[RADIATE JPEGS 1](https://abcdstudy.org/newsletter/fMRI_files/RADIATE_JPEGS1.zip)。
- 已下载 22 个作者匿名人物组、44 张彩色照片，每组取中性闭口 NC 与高兴闭口 HC 各一张。
- 图片为 1500 × 1500；保留原始文件，逐张记录 SHA256、ZIP 路径和作者编号。
- 原始人物清单为 `data/manifest.radiate.unsplit.csv`，分组后为 `data/manifest.radiate.csv`。
- 随机种子 42，训练集为 13 组 26 张，验证集为 5 组 10 张，测试集为 4 组 8 张；人物编号不跨集合。
- 实际只有 22 位亚洲受试者，不宣称已达到早期计划中的至少 30 位目标。
- 来源没有提供皮肤或完整人脸像素掩膜。两类 mask 状态保持 missing，分组记录明确为 preparation_only、evaluation_ready=false。

这些照片用于教育和研究，遵循来源关于不得识别照片中个人的要求。原图由 Git 忽略，不重新发布。项目只使用匿名组标识防止数据交叉，不运行人脸识别或身份聚类。

下载采用官方 ZIP 的有界字节请求，只获取选定成员，校验长度、CRC 与图片解码，不下载整个 1.47 GB 包，不运行压缩包中的任何代码。已成功的数据会缓存，可恢复下载。

```powershell
.\.venv\Scripts\python.exe -m src.training collect-radiate --manifest data/manifest.radiate.unsplit.csv
.\.venv\Scripts\python.exe -m src.training split --manifest data/manifest.radiate.unsplit.csv --output data/manifest.radiate.csv --seed 42 --ratios 0.6 0.2 0.2 --prepare-only
```

分组输出已存在时，split 命令拒绝覆盖，避免无意改变留出集合。prepare-only 仅允许在精细标注完成前整理已核验人物组，正式 evaluation 校验仍要求完整标注。

## 已下载的外部肤色真值

已取得包含 78 对原图与人工肤色掩膜的 SkinColor 压缩包，目录为 FacePhoto 32 张与 FamilyPhoto 46 张。该材料由阿姆斯特丹大学图像处理课程公开发布，其目录及图片名称与 Pratheepan 数据一致；作者原站当前请求超时，因此使用有明确教学说明的大学镜像，并记录镜像与原始来源。

- 原始来源及非商业使用说明：[Pratheepan Dataset](https://cs-chan.com/downloads_skin_dataset.html)。
- 镜像说明与读取规则：[University of Amsterdam Skin Color Detection](https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/20172018/LabExercises/SkinColorDetection.html)。
- 下载：[SkinColor.zip](https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/_downloads/SkinColor.zip)。
- 下载文件为 5,886,054 字节，SHA256 为 `ec1c897f6acb8fd3feaffdd77bc6097b11118c80e42354c49db36923c0d5d26c`。
- 配对清单为 `data/benchmarks/pratheepan/manifest.csv`，来源摘要为同目录 provenance.json。
- 图片与掩膜逐张核对尺寸。部分旧 PNG 有少量非黑白噪声，严格沿用课程公开读取规则：第一 RGB 通道等于 255 才记为皮肤，忽略 alpha；转换规则写入每行来源记录。
- 这是混合人群的外部肤色分割数据，没有可验证的人物编号，也没有完整面部区域标签。不能宣称为亚洲人专用、人物独立的人脸测试集。

```powershell
Invoke-WebRequest -Uri 'https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/_downloads/SkinColor.zip' -OutFile data/cache/SkinColor.zip
.\.venv\Scripts\python.exe -m src.training import-skin-benchmark --archive data/cache/SkinColor.zip --output data/benchmarks/pratheepan
```

该清单与训练清单格式分开，防止把肤色真值当作脸部真值，或将混合人群的像素加入亚洲肤色训练却不说明来源。

## 当前可交接与待完成内容

| 内容 | 状态 |
| --- | --- |
| 原有 FairFace 肤色训练样本与统计模型 | 保留，用于联调；人物分组未核验 |
| 新增亚洲受试者的可靠匿名编号 | 已下载并核对作者文件结构，共 22 组 |
| 新增数据的人物独立划分 | 已完成，13 组训练、5 组验证、4 组测试 |
| 外部人工肤色掩膜 | 已下载并转换，共 78 对 |
| RADIATE 的完整皮肤及面部区域掩膜 | 来源没有提供，仍需制作和复核 |
| 亚洲人专用完整面部分割测试结果 | 尚未生成，不能由外部肤色掩膜代替 |

正式的人物独立实验应使用 RADIATE 训练集合单独拟合模型，再在其验证集合选参数、测试集合做一次最终评估。当前 FairFace 来源与 RADIATE 之间没有身份对应表，因此不能直接声称混用两个来源仍然人物独立。
