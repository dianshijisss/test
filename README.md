# 基于深度学习的皮肤病检测系统

本项目提供了一个使用 PyTorch 实现的皮肤病分类示例。数据集需按照 `train/类别`、`val/类别` 的目录结构放置。

## 依赖

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py --train-dir path/to/train --val-dir path/to/val --epochs 20 --output model.pth
```

## 预测

```bash
python predict.py --model model.pth --num-classes 3 --image test.jpg
```

以上命令将在命令行输出预测的类别编号。

## 可视化

```bash
python visualize.py --model model.pth --num-classes 3 --data-dir path/to/val
```

该命令将随机选择几张图片并显示真实标签与预测标签。
