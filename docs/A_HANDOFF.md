# A 向 B、C 的交接

交付数据和最终肤色统计模型，完整系统验收仍未完成；详见 A_STATUS.md。

| 文件 | 用途 |
| --- | --- |
| data/manifest.csv、data/split.json | 唯一清单与冻结人物/场景分组，120/100/24 |
| data/sources/*.json | 作者地址、群体/人物依据、SHA256、选择记录 |
| data/annotations/records/、reference/、review.json | 双层参考及无损压缩、辅助方法和逐图复核修正 |
| models/skin_model.json | 最终统计模型与训练图/掩膜哈希 |
| models/operating_point.json | 与模型版本绑定的验证集阈值 |
| models/evaluation.json | 累计/逐图/场景指标、无脸误检、环境与耗时 |
| models/delivery_checks.json | 来源、参考、拟合复现和掩膜重建检查 |
| artifacts/model/、predictions/ | 可重建分布图/统计及验证测试预测 |

原图按 A_GUIDE.md 直接从作者恢复，不经本项目重新公开分发。Git 内参考重建不需要神经网络。SHAPE 网页说明图独立保存，不在正式清单。

## B 的参数契约

- schema_version=1，model_revision=skin-d239a8aaf4ae7f05。加载使用 src.training.model.check_model，并核对配置版本。
- 输入为转正的 uint8 BGR。OpenCV YCrCb 输出 [Y, Cr, Cb]，必须选通道 [2, 1] 得 [Cb, Cr]。
- μ=[113.86623333333333,143.3184875]。
- 原始 Σ=[[13.11251944771992,-7.345890096625399],[-7.345890096625399,16.315821194932038]]，对角为方差。推理时只加一次 λI，λ=1。
- d²=(x−μ)ᵀ(Σ+λI)⁻¹(x−μ)。T=4，采用 d²≤4，不再平方，也不称为人脸置信概率。
- A 配置字段 distance_squared_threshold；任务书 pipeline 的 distance_threshold 如被采用，B 须显式映射为同一平方距离含义。
- T 专为完整协方差肤色 micro Dice 校准，不自动适用于对角对照及后处理；B 的对照和几何筛选只在验证集调参。

assessment.distance_map 提供距离计算；face_proposal 是最大连通域诊断基线，会丢多脸、背景误检高，不作最终 segment_image。完整多候选、面积/几何筛选、无脸空返回和脸颈分离由 B 完成。原图尺寸、内部两层 uint8 0/1。

## C 的读取与评估

dataset.read_bgr 统一 EXIF，拒绝灰度和透明输入。skin 用于肤色评估，face 用于脸区域评估；reference 位编码图须经 rasterize/read_mask，不能直接作前景。NPY 导出 0/1，PNG 预览 0/255。

主指标累计全测试 TP/FP/FN，含无脸误检。另报 24 张新人物图平均 IoU；无脸诊断见20张验证背景和独立来源手部裁剪；不能把空真值/空预测静默记成 IoU=1。须注明辅助参考限制。

测试结果已公开，不能借其修改算法再称一次性最终测试。B 以验证集开发；如使用测试反馈，应另设未使用留出集或标明探索性结果。当前未运行 B/C 跨模块集成。

32 项必要单元测试通过。复现命令见 A_GUIDE.md；整套来源/参考与训练复现检查为 `python -m src.training.audit`。audit 不重新选择阈值或输出新的测试性能。


2026-10-29：增加33张作者感知Asian标签真实场景，使用 `python -m src.training.public_sources restore-scenes` 恢复；`python -m src.training.stress_assessment` 固定模型评估。该来源没有身份编号，只作外部压力测试。原始照片共277，纯手部裁剪2张不增加原图数。新留出主结果与场景结果见A_STATUS.md，不沿用旧版本指标。
