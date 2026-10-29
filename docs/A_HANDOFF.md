# A 交接入口

核对日期：2026-10-29；交付基线提交：`f77c33a`（本次文档修订之前）。A 的数据、参考标注、统计模型及报告已经交付。B、C 从下面对应文档开工：

- [给 B：分割算法交接](HANDOFF_B.md)
- [给 C：界面、评估和集成交接](HANDOFF_C.md)
- [环境与数据恢复](A_GUIDE.md)
- [本轮状态及指标](A_STATUS.md)
- [来源和使用条件](DATA_SOURCES.md)、[像素标注定义](annotation_guide.md)
- [文档漂移核对记录](DOCUMENT_AUDIT.md)

## 当前事实与权威文件

| 项目 | 当前值／权威文件 |
| --- | --- |
| 模型版本 | `skin-d239a8aaf4ae7f05`；`models/skin_model.json` |
| 模型格式 | schema_version=1；color_space=`OpenCV-YCrCb-uint8-full-range`；feature_order=`[Cb, Cr]` |
| 高斯参数 | 从模型 mean、covariance、regularization_lambda 读取；原始协方差只在推理时加一次 λI，当前 λ=1 |
| 平方距离阈值 | `models/operating_point.json` 的 distance_squared_threshold=4.0；与模型版本及清单哈希绑定 |
| 主数据 | `data/manifest.csv`：244 张＝224 HFD 人像＋20 背景；48 位作者匿名人物 |
| 主划分 | `data/split.json`：训练120张/24人；验证100张（80人像/16人＋20背景）；测试24张/8人 |
| 外部场景 | `data/sources/wider_fair_downloads.json` 的 records：33 张；不在主CSV中 |
| 原图总数 | 277；另有2张可重建纯手部裁剪，不增加原图或人物数 |
| 参考标注 | 主数据488张0/255掩膜由Git内参考重建；场景33张位编码PNG在annotations/scenes/ |
| 主结果 | `models/evaluation.json`：新人物测试肤色IoU=0.7921、人脸基线IoU=0.6142 |
| 场景结果 | `models/stress_evaluation.json`：肤色IoU=0.1951、人脸基线IoU=0.1332；手部裁剪误检2/2 |
| 检查范围 | `models/delivery_checks.json`核验244张主数据、488张掩膜及拟合复现；场景哈希检查由stress_assessment执行 |

表内 data、models 路径均相对仓库根目录；annotations 路径相对 data/。发生冲突时以模型/阈值JSON、清单及相应报告为依据，同步修正文档。不要从历史聊天或旧文档抄参数。数据/参考变化后，须更新对应哈希、模型及报告版本。

## 本轮范围与实现边界

系统输入彩照，输出原尺寸人脸二值掩膜；不识别人名、身份或族群。人物编号用于已有数据分组，属于可用时采用的实验辅助信息，不是检测输入或所有数据的准入条件。无身份编号的数据可按照片/场景分组、查重，并如实说明身份重叠无法排除。

现有33张场景已经以外部压力测试口径报告，因此保持当前记录，不把它们补写成身份独立测试。它们可以用于后续开发；一旦用其失败案例调参，后续结果须标为开发/探索性结果，不能再声称该批为未见测试。作者Asian标签仍是含人物样本的来源要求。

参考均为独立工具辅助初始化、Codex视觉复核的项目参考，有边界误差；没有外部专家审核这一附加验收条件。纯手部裁剪有父图和坐标，不冒充独立拍摄场景。SHAPE说明图已删除，完整SHAPE申请属于可选扩展，见DATA_ACCESS_REQUEST.md。

当前src下仅有training模块。`segment_image`、`src/segmentation/`、`config/pipeline.json`、`src/app/`、`src/evaluation/`及完整界面尚待B/C实现。A的face_proposal仅是诊断基线，不作为完整多人分割入口。32项现有数据/模型测试通过，不等同于B/C集成测试通过。

## 共同接口约定（B/C待实现）

建议由B提供 `segment_image(image_bgr, model, config) -> dict`，C按同一返回结构集成；此处是施工契约，当前仓库没有该函数。

| 返回字段 | 约定 |
| --- | --- |
| skin_mask、face_mask | uint8、形状(H,W)、值仅0/1，尺寸与转正原图一致 |
| candidates | list[dict]；每项bbox=(x,y,width,height)、area为区域前景像素数 |
| timing_ms | 非负处理耗时，排除读取、UI与导出 |

bbox零起始、左上原点、右下边界不包含。无候选返回两个有效掩膜、全零face_mask和空candidates；skin_mask保留实际肤色分类结果。非法输入或模型/配置错配明确报错。配置统一使用 `distance_squared_threshold`，沿用A阈值字段；旧文档的distance_threshold不作为并存别名。B若改变阈值/后处理，保存新配置与算法版本，不能覆盖A基线报告。
