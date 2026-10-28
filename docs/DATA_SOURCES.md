# 数据来源与使用边界

核验日期：2026-10-28。正式清单只有 `data/manifest.csv`，共 220 张原始照片：200 张 HFD 人像、20 张无人物背景。所有含人的正式样本均来自作者明确分类为 Asian、East Asian、Southeast Asian 的目录及元数据。项目不根据外貌判断族群，不训练族群识别器。

## Hawai‘i Face Database：40 人、200 张

- [作者实验室](https://www.diversitysocialcoglab.com/resources)、[作者 OSF](https://osf.io/fkn7y/)、[论文](https://doi.org/10.1080/15298868.2024.2417664)。作者提供参与者自报与分类信息。
- 只选择 Asian faces、East Asian faces、Southeast Asian faces；不使用 Multiracial、Multiracial Asian 等目录。63 位拥有完整五版本的参与者中，按种子 42 固定选择 40 位。
- 每人五张原始彩照：露齿笑、闭口笑、中性、右侧面、左侧面，作者编号 2、3、6、7、8。侧面并非统一精确转角。
- 分组依据作者文件命名中的照片编号和参与者编码；所有表情及侧面同组。`hawaii_participants.json` 保存所选匿名编码、作者分类和自报族群原文，记录官方 norming XLSX 的下载地址及 SHA256。原表一项文件名末尾空格在匹配时去除。
- `hawaii_selection.json` 固定选择范围；`hawaii_downloads.json` 逐图记录 OSF 文件 ID、原始文件名、作者目录、下载地址和 SHA256。200 张原文件均与 OSF 作者发布 SHA256 一致。
- 作者允许课堂教育、研究刺激及演示/出版物示例刺激用途，其他用途不在授权内。原图不纳入 Git、不由本项目重新公开分发；恢复工具直接访问作者地址。使用者须遵守上述条件。

引用：Meyers, C., Garay, M. M., & Pauker, K. (2024). Hawai‘i face database: a racially and ethnically diverse set of facial stimuli. Self and Identity. DOI: 10.1080/15298868.2024.2417664。

## BSDS500：20 张无人物背景

- [UC Berkeley 官方资源页](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)、[官方原始包](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)。
- 逐图查看后保留 20 张未发现人物、人脸或人体皮肤的自然景物、动物和交通物体照片。两层参考均为空。视觉检查不构成对极小不可辨识物体的保证。
- 没有人物，故无族群标签，也不虚构人物编号。场景组分配为验证/测试各 10 张，不参与肤色拟合。
- `bsds_downloads.json` 保存官方包 SHA256、精确成员路径和逐图 SHA256；`bsds_visual_review.json` 保存选择和复核范围。
- 这里采用本项目负样本划分和空参考，并未复用官方边界分割真值；结果不属于 BSDS 官方基准、不与官方榜单比较。版权仍归原权利人，引用官方说明及 Arbelaez 等 TPAMI 2011 论文。

## 像素参考的来源

HFD 没有本项目需要的官方 skin/face 像素真值。本项目用独立 BiSeNet 和 MediaPipe 多类分割初始化，再由 Codex 逐图查看全部 200 张叠加图，修正 12 张中的明显衣物、发饰、背景误标或漏标。没有使用待评估的 YCrCb 结果生成参考。

这些属于项目辅助参考（silver-standard），未经过独立人工/专家逐像素审核。细边缘、耳朵、饰物、阴影仍有近似误差。指标仅表示与这套参考的一致程度，不能充当官方真值精度；complete 是整图覆盖状态。`annotations/records/hawaii.json`、`annotations/review.json` 保存方法、原图/辅助模型/草稿/最终参考哈希与复核记录。最终压缩参考 PNG 纳入 Git，可以脱离辅助模型重建掩膜。

辅助模型只用于标注。训练、肤色分割和基线评估使用传统颜色统计，不调用神经网络。工具来源：[BiSeNet 实现及 MIT 许可证](https://github.com/yakhyo/face-parsing)、[Google 官方分割说明](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter)。固定权重 URL 与 SHA256 在 `src/training/public_sources.py` 及标注记录中。

## 场景缺口及排除来源

已覆盖真实单人、表情、左右侧面、不同服装和脸颈相连，另有真实无人物背景。未覆盖作者明确标注亚洲人的真实多人同框、复杂正样本背景、强偏色光照和完整原始手部干扰场景，不用合成拼接或未知族群图冒充。

[SHAPE 作者页](https://users.soict.hust.edu.vn/linhdt/dataset/)与[作者论文](https://doi.org/10.1007/s00521-022-07651-1)是已核验的亚洲参与者手势候选。完整原图需邮件申请；公开网页 16 张手势说明图已单独保存，不能视为独立原始场景，不纳入 220 张、人物计数、训练或正式指标。部分说明图含口罩遮挡的脸，不能统一标无脸。记录为 `shape_examples.json`。

HGR 无明确作者族群标签，已排除并删除；所检查的 NCKU 原图是灰度，未进入彩色统计。被替换的小规模旧来源、标注、模型及探索缓存已经删除，没有归档。Git 既有历史未重写。

完整手部数据尚未获得，也未联系作者。具体申请材料见 [剩余数据申请](DATA_ACCESS_REQUEST.md)。
