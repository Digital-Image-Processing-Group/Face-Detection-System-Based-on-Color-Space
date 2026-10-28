# 数据来源与使用范围

更新日期：2026 年 10 月 28 日。本项目当前唯一数据来源为 RADIATE 官方发布的彩色照片子集，所有训练、验证、测试均使用该子集。

## 来源与下载

- [作者论文：The Racially Diverse Affective Expression Face Stimulus Set](https://pmc.ncbi.nlm.nih.gov/articles/PMC6446554/)。
- [官方发布页及使用条件：ABCD Study fMRI Tasks and Tools](https://abcdstudy.org/scientists/abcd-fmri-tasks-and-tools/)。
- [官方 RADIATE JPEGS 1 文件](https://abcdstudy.org/newsletter/fMRI_files/RADIATE_JPEGS1.zip)。

本项目仅依据作者 AF、AM 目录标签和匿名参与者编号选取 22 个人物组的彩色 NC、HC 照片，共 44 张，每张 1500×1500。不根据外观推断族群，不识别真实身份。作者标签限定了研究子集，不能据此声称覆盖亚洲全部人群。

`data/sources/radiate_downloads.json` 逐图记录官方 ZIP 成员、作者编号、来源链接、原文件 SHA256、尺寸及使用范围。下载器对所选 ZIP 成员使用有界字节请求，校验长度、CRC，并解码检查；恢复已有照片时核对来源 SHA256。当前本地照片已经齐备，不依赖缓存或重试才能运行训练。

```powershell
.\.venv\Scripts\python.exe -m src.training collect-radiate
```

照片用于研究及教育，遵守来源关于不得识别照片中个人的条件。人物原图保留在本地，由 Git 忽略，不重新发布；清单和脚本供有权使用者从官方来源恢复文件。公开发布或改变用途时应重新核对官方条款。

## 人物分组与标注

唯一清单为 `data/manifest.csv`，冻结划分见 `data/split.json`。分组使用作者匿名参与者编号，不使用文件行号、相似度猜测或人脸识别结果。

| 集合 | 人物组 | 照片 |
| --- | ---: | ---: |
| train | 13 | 26 |
| validation | 5 | 10 |
| test | 4 | 8 |
| 合计 | 22 | 44 |

随机种子 42，目标比例 60/20/20，实际数量受整个人物组约束。同一参与者的两张照片不跨集合。跨集合检查同时涵盖解码后完全重复、粗略感知哈希及 RGB 缩略图差异；哈希只用于重复提示，不作为人物身份依据。

来源没有提供皮肤或完整面部像素掩膜。当前 44 对掩膜全部由项目自制：Codex 查看 600 像素预览逐图绘制面部折线和皮肤区域，栅格化到原图并检查叠加图。多边形、方法和复核说明位于 `data/annotations/polygons/`。这些是覆盖完整的近似参考标注，未经过独立专家逐像素复核；指标不能表述成官方像素真值基准成绩。

## 当前覆盖范围

本次数据是 22 位参与者、44 张受控背景单人照片。没有多人场景和无脸负样本，也不能证明复杂背景、强偏色、遮挡或脸颈分离效果。当前数据规模小于早期扩展目标，完成的是此明确子集上的 A 数据与统计模型交付。后续扩展应单独设立未使用人物与场景，保持测试用途清楚。
