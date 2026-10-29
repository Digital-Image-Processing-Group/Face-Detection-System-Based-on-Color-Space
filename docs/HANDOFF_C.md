# 给 C：界面、评估和集成交接

目标：接入A数据和B分割入口，实现选图、两层结果展示、原图叠加、保存和可复核报告。A已完成约定交付；当前只存在training模块，B完整入口和GUI均未交付。参数、返回结构和文件权威关系见[A交付与使用说明](A_GUIDE.md)。

## 环境和最短接入路径

在仓库根目录PowerShell执行[A_GUIDE](A_GUIDE.md)的环境安装、collect、rasterize和validate命令。先读取已交付模型/阈值及报告即可，不必重新训练或覆盖A评估报告。原图、生成掩膜、artifacts和手部派生图不在Git中；277张原图需按来源恢复。GUI采用任务书中的Tkinter，C检查当前Python能否import tkinter并启动窗口；A的离线训练检查未证明GUI可用。

B入口未到位时，可用明确标为模拟的掩膜开发布局与导出；最终演示切换B真实结果。assessment.face_proposal是A的单候选诊断基线，不是B的交付入口。

## 数据读取与编码

下表 `dataset` 指 `src.training.dataset`，路径参数传 `pathlib.Path`；`data_path` 的首参是主清单路径，次参是清单中相对路径。场景不伪造成主 CSV 人物记录，直接按独立来源 JSON 读取。

| 数据 | 读取方法及用途 |
| --- | --- |
| 主清单244张 | dataset.read_manifest(Path('data/manifest.csv'))；路径相对data/，用dataset.data_path解析 |
| 输入照片 | dataset.read_bgr，统一EXIF；UI显示前BGR转RGB |
| 主skin/face参考 | rasterize从records/及reference/重建488张PNG；dataset.read_mask(path,image.shape[:2])得到0/1 |
| 场景33张 | sources/wider_fair_downloads.json的records；image_path、reference_path也相对data/ |
| 场景位编码PNG | 灰度uint8，skin=(packed & 1)!=0，face=(packed & 2)!=0；先核对来源/参考SHA256、尺寸和复核状态 |
| 两张手部裁剪 | stress_evaluation.json的derived_hand_only.records包含父图、crop_xyxy、图像/参考路径；运行stress_assessment可重建 |

主reference/和场景scenes/里的PNG为0/1/2/3位编码，不能用非零直接作face。派生手部reference同为位编码；主生成skin/face PNG为0/255。B返回的两层数组才是uint8 0/1。原图尺寸以EXIF转正后的(H,W)为准。

## 界面和导出

1. 选择单张彩照，加载并校验模型/配置，调用B的segment_image。
2. 展示原图、肤色、人脸及候选叠加；显示处理耗时，保证界面可操作并明确错误。
3. 保存两层NPY（0/1）、PNG预览（0/255）和叠加图；为结果记录图像ID/哈希、model_revision、算法/配置版本、尺寸、候选和耗时。不得只保存缩略图。
4. 无候选时显示全零face_mask及空列表，skin_mask仍显示真实分类结果。分开处理“检测为空”与“读取/模型错误”。

任务书指定C负责src/app/、src/evaluation/、tests/integration/、reports/和最终README/依赖说明。不要把训练参数、辅助标注工具细节塞进普通使用者界面；界面也不要求输入姓名、人物编号或族群。

## 评估口径

评估器必须调用B实际入口。A的assess和stress_assessment固定使用distance_map＋face_proposal，只复现A基线；没有可传入B回调的现成开关。C可复用指标和读取工具，另写B报告，保留A基线文件供对照。

分别算skin与face的Precision、Recall、IoU、Dice，主结果累计像素TP/FP/FN，并另列有脸图逐图平均IoU。多人脸mask包含全部合格面部，不只评最大脸。原始作者矩形框是标注辅助，不能直接当像素真值。

- 有前景真值但预测为空：IoU、Dice为0。
- 真值与预测都空：IoU、Dice不适用，不当满分，不加入有脸图均值。
- 无脸图预测非空：累计FP，并统计非空输出图片比例。
- 当前主测试24张全有人脸，所以测试无脸误检率为不适用，不能写0%；20张验证背景误检16/20，另有手部裁剪2/2误检。
- 场景33张单独报告；其中11多人、30手脸干扰、3偏色/不均匀光照，标签重叠不能相加作总数。两张裁剪不算两张独立原始照片。

HFD人物分组是现有实验安排，不是产品要求。场景没有身份编号，当前按外部场景分区报告，不能声称身份绝无重叠。后续开发可按照片/拍摄场景查重分组并说明限制；使用已公开测试反馈后须标探索性结果，或另留未用数据进行盲测。参考为项目辅助标注，未经独立专家精修，报告须说明边界近似。

## 联调与交付检查

与B确认segment_image签名、dict返回键、bbox=(x,y,width,height)的半开边界、area单位、配置字段distance_squared_threshold和model_revision。用一张正常图、无脸图和多候选图打通读取→算法→展示→NPY/PNG导出；核对尺寸、dtype、值域、BGR/RGB及EXIF。

对损坏/透明图、模型错配、保存失败给出可理解提示。计时区分算法和端到端耗时，记录样本量、分辨率、硬件和软件版本；A报告中的124张中位205.43ms/P95 244.98ms只是其基线，不是GUI或B算法耗时。

最后交付可运行演示、README、完整依赖、B实际算法的配置/报告、集成检查与失败案例。32项A测试不代替C的跨模块集成验收。
