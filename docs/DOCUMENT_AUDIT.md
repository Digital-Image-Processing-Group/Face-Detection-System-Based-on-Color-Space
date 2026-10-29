# 文档漂移核对记录

日期：2026-10-29；代码、数据与模型基线提交：`f77c33a`，A 分支。本次仅修订任务书和交接说明，不变更数据、模型、阈值或既有指标。

## 发现与修正

| 漂移或歧义 | 修正后口径 | 核对依据 |
| --- | --- | --- |
| A3 仍写 220 张参考，使用说明仍写 440 张生成掩膜 | 244 张主数据、488 张主掩膜；另有 33 张场景参考，合计 277 张原图 | manifest.csv、场景 records、delivery_checks.json |
| HFD 全部人物均被写为每人五张 | 原 40 人各五张；新增 8 人各三张，共 224 张、48 人 | manifest.csv、hawaii_selection.json、hawaii_reserve.json |
| 背景仍写验证/测试各 10 张，任务书仍要求重新按 60/20/20 划分 | 当前固定 120/100/24，20 张背景全部在验证集 | split.json、manifest.csv、evaluation.json |
| 查看命令未覆盖新 24 张人像 | 补充 offset 220/226/232/238、limit 6；背景 limit 10 | 主清单实际行顺序、contact-sheet 参数 |
| 标注说明仅描述原 200 张与首次 12 张修正 | 补充 30 张侧脸修正、新 24 张复核及 4 张修正、33 张场景独立参考路径 | refinement_20261029.json、reserve_review.json、scene_review.json |
| 将人物编号写成所有数据必须具备的条件 | 编号仅在可获得时用于实验分组；未编号场景可用但须披露身份重叠风险；当前实验分区保持冻结 | 用户已确认的检测范围；现有主验证器与场景评估器的实际差异 |
| 任务书配置字段 distance_threshold 与产物不一致 | 统一为 distance_squared_threshold；当前 T=4，表示平方距离阈值 | operating_point.json、assessment.distance_map |
| 无候选时“全零掩膜”未区分肤色和人脸 | face_mask 全零、candidates 为空，skin_mask 保留实际分类 | B/C 待实现接口契约；A 基线不能替代该契约 |
| SHAPE 说明图和已删除来源仍出现在旧交接中 | 移除失效恢复说明；完整 SHAPE 仅列为可选申请，未联系作者 | public_sources 的实际命令、当前来源清单 |
| 场景近重复检查被描述为已完成 | 只报告现有来源/参考哈希及解码完全重复检查，不补写近重复能力 | stress_evaluation.json、stress_assessment.py |
| A 基线与 B/C 实现边界不清 | 当前仅有 src/training；segment_image、GUI、B 算法评估接入待实现 | src 目录、assessment.assess、stress_assessment.py |
| 安装步骤默认重跑训练并覆盖交付报告 | 拆分数据恢复与可选原实验复现，写明覆盖范围 | training CLI 默认输出路径 |

## 交接文件与责任

[A_HANDOFF.md](A_HANDOFF.md)作为共用事实与接口入口；[HANDOFF_B.md](HANDOFF_B.md)说明 B 的输入、复用 API、算法内容及交给 C 的产物；[HANDOFF_C.md](HANDOFF_C.md)说明 C 的数据解码、界面导出、实际 B 算法评估与联调验收。

本轮 A 任务完成状态保持不变。独立专家精修及新增独立拍摄手部数据属于可选扩展；现有近似参考、外部场景未知身份及手部裁剪范围须继续如实披露。B/C 完整系统尚未验收。

## 本次核对方式

- 读取主清单、来源清单和 JSON 产物，核对数量、分组、模型版本、清单 SHA256、阈值、报告指标及场景标签；核对 B/C 文档引用的 API 签名与基线调用路径。
- 核对任务书及 docs/ 下共 10 份 Markdown；用现有 CLI 解析器检查 6 条核心命令与参数，公开来源 CLI 帮助可正常运行；检查相对文档链接、文件路径与 Git 空白错误。
- 未重新下载数据、训练或运行性能评估，未覆盖现有报告；32 项 A 测试通过记录沿用前次代码交付结果，本次不将其宣称为新增集成验收。

后续改动数据/标注、模型、配置或接口时，同步更新共用入口和对应角色交接；冻结实验的来源/版本/指标以配套 JSON、清单和报告为依据，功能是否实现以当前代码为依据。
