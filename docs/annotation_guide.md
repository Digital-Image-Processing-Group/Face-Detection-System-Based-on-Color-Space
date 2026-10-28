# 肤色与人脸标注规范

坐标以 EXIF 转正后的原尺寸照片为准；当前 RADIATE 全部为 1500×1500。标注独立于肤色模型预测绘制，禁止直接将预测结果当作参考答案。

## 两类参考标注

| 层 | 包含 | 排除 |
| --- | --- | --- |
| skin | 可见面部皮肤、耳部及颈部皮肤 | 头发、浓密胡须、眉毛、眼睛、嘴唇、衣服、饰物、背景和遮挡 |
| face | 可见面部及内部眼睛、眉毛、嘴巴 | 头发、耳朵、脖子、背景和外部遮挡 |

稀疏胡茬下仍可见的皮肤计入 skin，明显浓密胡须区扣除。眼睛、眉毛、嘴唇及个别鼻孔通过排除多边形扣除。细小边缘存在近似，需按当前参考标注的精度解释指标。

## 本次标注方法与状态

44 张照片逐图绘制面部边界折线和额外可见皮肤，使用曲线采样点近似眼睛、眉毛、嘴唇区域，并逐批检查原图和叠加图。参考预览为 600×600，顶点换算到原图坐标后栅格化；没有把统一椭圆或模型输出用作答案。

标注者为 `codex_visual_polygon_annotation`。它们是项目自制的多边形参考标注，来源不提供像素掩膜，也未经过独立专家审核。`skin_status=complete`、`face_status=complete` 表示整图范围均已处理、叠加图已查看，不等于每个边界像素绝对准确。指标以该参考为准。

当前交付全部 `identity_status=verified`，依据作者匿名编号；此字段仅指分组依据，不表示像素标注质量。编辑工具仍能记录 `missing`、`sampled`、`draft`，便于发现未完成编辑；训练及评估入口均拒绝未核验分组或不完整的两类标注。

## JSON 与掩膜

`data/annotations/polygons/batch_*.json` 的 `annotations` 列表每批包含 4 张照片。逐图保存 image_id、width、height、annotator、method、review_note，以及 skin/face 的 status、include、exclude。

每个区域由至少三个 `[x,y]` 顶点组成，坐标为原图像素，不能越界。先填充 include，再扣除 exclude。肤色可包含多个耳颈多边形；面部标注不因眉眼嘴而挖孔。栅格化递归读取 JSON，拒绝重复或未知 image_id，并更新清单。

PNG 是单通道 0/255，读取后转为 uint8 0/1。NPY 仅接受二维 0/1。掩膜尺寸必须与转正后的图片相同，禁止 JPEG、彩色掩膜或混合 1 与 255。

```powershell
.\.venv\Scripts\python.exe -m src.training rasterize --manifest data/manifest.csv --polygons data/annotations/polygons
.\.venv\Scripts\python.exe -m src.training contact-sheet --manifest data/manifest.csv --output artifacts/radiate_review/overlay_00.png --offset 0 --limit 4 --tile-size 600 --columns 2 --overlays
```

检查所有 offset=0、4……40 的批次。训练时的 15 像素内缩只作用于采样区域，不改动保存的完整参考掩膜，不用于缩减评估范围。

## 清单字段与修改规则

`data/manifest.csv` 为 UTF-8 CSV，字段顺序由 `Record` 定义。图片与掩膜路径相对于 data 目录，禁止越出该目录。subject_ids 与 scene_tags 用英文分号分隔；人物代码只能来自可靠来源元数据。

清单记录 image_id、image_path、subject_ids、group_id、split、skin_mask_path、face_mask_path、scene_tags、source_url、source_label、license、identity_status、skin_status、face_status。当前照片统一为 controlled_portrait、author_color_version。

修改多边形后重新栅格化、检查尺寸与叠加效果，再重训和校验。不得看过测试效果后悄悄修改测试标注或划分；需公开变更、说明测试用途，并另设未使用的最终留出数据。
