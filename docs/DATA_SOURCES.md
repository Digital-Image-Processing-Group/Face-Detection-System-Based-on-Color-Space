# 数据来源与使用边界

核验日期：2026-10-29。交付共 277 张原始照片：主清单 `data/manifest.csv` 含 224 张 HFD 人像、20 张背景；另有 33 张 WIDER-FAIR 外部场景，单独记录、评估。HFD 使用作者明确分类为 Asian、East Asian、Southeast Asian 的目录及元数据；WIDER-FAIR 使用作者逐脸感知族群 Asian 标注，两者标签来源不同。项目不自行根据外貌判断族群，不训练族群识别器。

## Hawai‘i Face Database：48 人、224 张

- [作者实验室](https://www.diversitysocialcoglab.com/resources)、[作者 OSF](https://osf.io/fkn7y/)、[论文](https://doi.org/10.1080/15298868.2024.2417664)。作者提供参与者自报与分类信息。
- 只选择 Asian faces、East Asian faces、Southeast Asian faces；不使用 Multiracial、Multiracial Asian 等目录。63 位拥有完整五版本的参与者中，按种子 42 固定选择 40 位。
- 原 40 人每人五张原始彩照：露齿笑、闭口笑、中性、右侧面、左侧面，作者编号 2、3、6、7、8；新增 8 人每人中性及左右侧面三张。侧面并非统一精确转角。
- 分组依据作者文件命名中的照片编号和参与者编码；所有表情及侧面同组。`hawaii_participants.json` 保存所选匿名编码、作者分类和自报族群原文，记录官方 norming XLSX 的下载地址及 SHA256。原表一项文件名末尾空格在匹配时去除。
- `hawaii_selection.json` 固定原 200 张选择范围，`hawaii_reserve.json` 固定新增 24 张选择；`hawaii_downloads.json` 逐图记录 OSF 文件 ID、原始文件名、作者目录、下载地址和 SHA256。224 张原文件均与 OSF 作者发布 SHA256 一致。上述来源 JSON 均位于 `data/sources/`。
- 作者允许课堂教育、研究刺激及演示/出版物示例刺激用途，其他用途不在授权内。原图不纳入 Git、不由本项目重新公开分发；恢复工具直接访问作者地址。使用者须遵守上述条件。

引用：Meyers, C., Garay, M. M., & Pauker, K. (2024). Hawai‘i face database: a racially and ethnically diverse set of facial stimuli. Self and Identity. DOI: 10.1080/15298868.2024.2417664。

## BSDS500：20 张无人物背景

- [UC Berkeley 官方资源页](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)、[官方原始包](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)。
- 逐图查看后保留 20 张未发现人物、人脸或人体皮肤的自然景物、动物和交通物体照片。两层参考均为空。视觉检查不构成对极小不可辨识物体的保证。
- 没有人物，故无族群标签，也不虚构人物编号。当前 20 张全部在验证集，不参与肤色拟合；早期各 10 张的验证/测试划分已退出默认流程。
- `bsds_downloads.json` 保存官方包 SHA256、精确成员路径和逐图 SHA256；`bsds_visual_review.json` 保存选择和复核范围。
- 这里采用本项目负样本划分和空参考，并未复用官方边界分割真值；结果不属于 BSDS 官方基准、不与官方榜单比较。版权仍归原权利人，引用官方说明及 Arbelaez 等 TPAMI 2011 论文。

## 像素参考的来源

HFD 没有本项目需要的官方 skin/face 像素真值。本项目用独立 BiSeNet 和 MediaPipe 多类分割初始化。原 200 张由 Codex 逐图查看，先修正 12 张明显衣物、发饰、背景误标或漏标，后修正 30 张侧脸的离散 face 误标；新增 24 张也全部复核，其中 4 张显式修正。没有使用待评估的 YCrCb 结果生成参考。

这些属于项目辅助参考（silver-standard），未经过独立人工/专家逐像素审核。细边缘、耳朵、饰物、阴影仍有近似误差。指标仅表示与这套参考的一致程度，不能充当官方真值精度；complete 是整图覆盖状态。`annotations/records/hawaii.json`、`annotations/review.json` 保存方法、原图/辅助模型/草稿/最终参考哈希与复核记录。最终压缩参考 PNG 纳入 Git，可以脱离辅助模型重建掩膜。

辅助模型只用于标注。训练、肤色分割和基线评估使用传统颜色统计，不调用神经网络。工具来源：[BiSeNet 实现及 MIT 许可证](https://github.com/yakhyo/face-parsing)、[Google 官方分割说明](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter)。固定权重 URL 与 SHA256 在 `src/training/public_sources.py` 及标注记录中。

## 场景覆盖与排除来源

已补入 33 张真实多人/复杂背景/手脸干扰/红色及暖色光场景，见下节。仍不能声称覆盖任意复杂场景。纯手部测试取自两张真实原图的裁剪，不作为新的独立原始场景。

[SHAPE 作者页](https://users.soict.hust.edu.vn/linhdt/dataset/)与[作者论文](https://doi.org/10.1007/s00521-022-07651-1)是已核验的亚洲参与者手势候选。完整原图需邮件申请；公开网页说明图未纳入本轮交付，也不作为独立原始场景。

HGR 无明确作者族群标签，已排除并删除；所检查的 NCKU 原图是灰度，未进入彩色统计。被替换的小规模旧来源、标注、模型及探索缓存已经删除，没有归档。Git 既有历史未重写。

完整 SHAPE 手部数据尚未获得，也未联系作者；这属于可选扩展，不是当前 A 交付的阻塞项。具体申请材料见 [可选数据申请](DATA_ACCESS_REQUEST.md)。

## 本轮补全（2026-10-29）

HFD 新增 8 位作者编号人物，每人中性/左右侧面三张，共 24 张。固定选择见 hawaii_reserve.json，原文件均核对 OSF 作者 SHA256；norming 表已核对至 48 人。此前使用过的测试人物和背景移入验证集，新 24 张人物图是未用于开发的最终人物留出集。主清单 120/100/24 张，人物 24/16/8 人；20 张已看过的无脸背景全部在验证集，最终人物测试不再宣称有未使用过的无脸背景。

[WIDER-FAIR 作者仓库](https://github.com/bronval/Wider-Fair-Dataset)、[作者论文](https://proceedings.mlr.press/v350/moussi26a.html)公开了逐脸人工感知族群标签；标注来自单个作者审核者，并非参与者自报。本项目只采用作者全部标为 Asian、Valid=True 的候选，再逐图排除未标注背景人物、疑似拼接、目录宣传画及像素参考质量不足的图。固定提交为 92b0df27ac7cb7b203317d7c9f88a9afbed18984，三个原 CSV 的 Git blob SHA 已校验；原图直接从作者该提交获取，保存项目下载 SHA256（不是作者发布的 SHA256）。

33 张入选原图及逐脸作者行记录见 wider_fair_downloads.json；完整来源说明与 CSV blob 哈希见 wider_fair_provenance.json。两层像素参考由独立 BiSeNet/MediaPipe 初始化、逐图查看并修正，记录为 annotations/scene_review.json，压缩参考为 annotations/scenes/。它们仍是有近似误差的项目辅助参考，不是 WIDER 官方像素真值。原图版权归原权利人；本地仅作非商业研究，不随 Git 公开分发。

WIDER-FAIR 没有人物身份编号，不能虚构人物 ID 或保证与其他来源绝无人物重叠。当前冻结实验将整批单列为外部场景压力测试，不进入肤色拟合、阈值选择或人物独立主指标。人物编号不是检测或数据准入的必需条件；未来可按原图/场景分组使用此类数据，但须披露身份重叠风险，并在用于开发后重新安排最终评估。两个纯手部裁剪保存原图 ID 与裁剪坐标，不增加原始照片数或独立人物数。33 张是目的性场景样本，不能代表总体分布。

原有 30 张 HFD 侧脸的离散脸部误标经逐图查看后修正，保留逐图修改前后哈希与移除像素数。新留出集也在读取预测之前复核。当前参考未经独立人工逐像素精修；进一步精修是质量提升方向，不另增为 A 未完成的硬性任务，也不把复核覆盖状态当作专家精度认证。
