# 给 B：分割算法交接

目标：基于A的真实肤色模型完成肤色分割、区域筛选及人脸0/1掩膜，保留所有合格候选。当前数据、模型版本和共用返回结构见[A交付与使用说明](A_GUIDE.md)。不需要识别人物身份。

## 开工输入

先按[A_GUIDE](A_GUIDE.md)安装requirements-A.txt、恢复主原图及488张掩膜，按需要恢复33张场景。使用models/skin_model.json和models/operating_point.json；当前版本skin-d239a8aaf4ae7f05、d²阈值4.0。无需MediaPipe或辅助标注权重即可开发运行时算法。

可以复用：

下表模块完整名称为 `src.training.dataset`、`src.training.model`、`src.training.assessment`，均从仓库根目录导入。`distance_map` 本身不做完整模型校验，调用前执行 `check_model`。

| 实现 | 用途 |
| --- | --- |
| dataset.read_bgr(path) | EXIF转正、RGB解码转BGR、拒绝灰度/透明图 |
| model.check_model(model) | 校验模型格式、特征顺序、有限值和协方差 |
| assessment.distance_map(image,model) | 完整协方差平方马氏距离，已加一次正则 |
| dataset.read_mask(path,shape) | 读取参考为0/1，供离线评估 |
| assessment.confusion / metrics | 累计TP/FP/FN及指标 |

推理只接收照片、模型和配置；作者人脸框、族群标签和参考掩膜仅用于离线数据/标注/评估，不作为检测输入。

## 实施内容

1. 加载JSON，执行check_model；核对模型与配置model_revision。沿用A数据时核对清单哈希，避免配置错配。
2. 输入必须是转正的(H,W,3) uint8 BGR。OpenCV YCrCb输出[Y,Cr,Cb]，特征取[2,1]为[Cb,Cr]；先转浮点再做减法，避免uint8溢出。
3. 计算d²=(x−μ)ᵀ(Σ+λI)⁻¹(x−μ)，用d²≤T生成肤色掩膜。协方差是原始Σ，λ只加一次；阈值4不是再平方成16，也不是概率。
4. 完成尺寸相关的区域清理、连通候选、面积/长宽比/填充率等筛选及受限填洞。处理脸颈相连、手、近肤色背景；多个合格区域全部保留，无合格区域正常返回空脸结果。
5. 对照使用相同μ和Σ对角元素，保持数据/采样一致，对照阈值仅在验证集单独校准。不要把对角近似写回A原始模型。
6. 在验证集开发并冻结最终pipeline配置、算法版本和选择记录，再交C集成。配置使用distance_squared_threshold、model_revision、postprocess；具体postprocess结构由B实现后同步C。

assessment.face_proposal只做开闭运算、最大连通域和填洞，不能直接作为完整segment_image；它会丢弃多脸。assessment.assess内部也固定调用这套基线，不能直接用其报告代表B的新算法；C应让评估器调用B入口。

## 验证重点与已知失败

必要行为检查：颜色通道/浮点计算正确、正则不重复、边界等号、两个分离候选均保留、无候选face_mask全零、手部不强行当脸、孔洞填充不跨大遮挡、小图/坏模型报错、原图尺寸与0/1类型符合契约。内部缩放后用最近邻恢复掩膜，bbox也恢复原图坐标。

当前基线新人物平均脸IoU为0.6160；33张复杂场景累计脸IoU为0.1332，纯手部裁剪误检2/2。优先改进多人保留、手与背景抑制和脸颈连接。偏色条件样本只有3张，不能宣传稳健覆盖任意光照。

当前24张主测试和33张场景结果已报告。固定配置可以复现；如使用其反馈开发，报告新结果为探索性，正式盲测需另有未使用数据。缺少人物编号不妨碍运行或开发；按照片/场景分组说明实验范围即可。

## 交付 C 的内容

- src/segmentation/内可调用的segment_image及其共用返回结构。
- config/pipeline.json、模型版本、算法版本、参数含义和验证集选参记录。
- 对角协方差对照配置与结果、必要行为测试及失败示例。
- 调用示例和异常约定；说明当前支持范围。无需自行实现UI，B/C接口契约见[A_GUIDE](A_GUIDE.md)。
