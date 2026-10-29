# 双层标注规范

尺寸和坐标以纠正 EXIF 后的原图为准，两层分开定义。

| 层 | 前景 | 排除 |
| --- | --- | --- |
| skin | 面部、耳朵、颈胸和四肢的可见皮肤 | 眉眼、嘴唇、口腔、头发、衣物、饰物、遮挡与背景 |
| face | 可见面部及内部眉眼、嘴部 | 耳朵、颈部、头发、衣物、外部遮挡与背景 |

侧脸只标可见部分，不补被挡区域；阴影皮肤仍属于 skin。无人物背景两层全零，动物皮毛不属于人体皮肤。

当前人像参考由独立 BiSeNet/MediaPipe 初始化，Codex 查看全部 200 张、每张 600 像素宽叠加图，显式修正 12 张后定稿。属于项目辅助参考（silver-standard），仍有边缘、阴影、耳朵、饰物和局部漏标误差。未经过独立人工/专家逐像素审核，不能宣传为精确或官方真值；complete 仅表示有整图覆盖，visually_reviewed 指上述 Codex 视觉复核。

正式记录位于 `data/annotations/records/`。人像最终压缩参考为 `reference/*.png`：最低位 skin、第二位 face，值域 0/1/2/3；记录保存尺寸、参考/原图/辅助模型 SHA256、方法版本和修正。`review.json` 记录逐图 ID、查看图哈希、草稿哈希和修正。废弃草稿不作为备用数据保留。

`python -m src.training rasterize --manifest data/manifest.csv` 从记录和压缩参考重建两层 0/255 PNG，`read_mask` 读为 uint8 0/1；会检查参考哈希、尺寸、编码和复核状态。背景用明确的空多边形规范重建。

missing 为缺标，skin sampled 为局部取样/草稿，face draft 为未定稿。正式评估需两层 complete、来源分组 verified 并通过完整检查。以后人工复核应记录审核身份、参考版本及新评估方案，不能只改状态。

2026-10-29 补全：原 30 张侧脸去除离散后颈/耳后/头发脸部误标，记录 refinement_20261029.json；新增 24 张人物参考记录为 reserve.json。33 张场景两层位编码参考在 annotations/scenes/，不由主清单 rasterize 重建，直接由 stress_assessment 校验读取。纯手部裁剪保留父图及坐标。均未使用被评估的颜色模型作为标注来源。
