# 课程提交版

这里保留最终提交的 `main.py`、依赖清单、启动脚本和像素参考。模型均值、协方差及处理参数在 `main.py` 中，算法结果与 `src/` 中的完整工程一致。实验报告位于 [reports/course](../reports/course/)。

## 准备验证图片

在仓库根目录安装依赖并恢复来源图片，然后生成提交版所需的 79 个输入：

```powershell
python -m pip install -r requirements.txt
python -m src.training collect
python -m src.training.public_sources restore-scenes
python submission/prepare_images.py
```

已有 `data/images/` 时复用本地照片；`prepare_images.py` 从中复制 24 张测试人像、20 张无脸背景和 33 张外部场景，再按冻结坐标生成 2 张手部裁剪。图片来源见 [数据说明](../docs/DATA_SOURCES.md)。

## 运行

```powershell
python submission/main.py
python submission/main.py --all
python submission/main.py --gui
```

默认运行 8 张代表图片，`--all` 验证全部 79 个输入并比较完整协方差与独立通道模型，`--gui` 打开选图界面。也可以双击本目录的 `运行.bat`。结果生成在 `submission/results/`；`images/`、结果和本地依赖目录不进入 Git。
