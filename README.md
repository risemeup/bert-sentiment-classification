# bert-sentiment-classification

中文情感二分类（BERT）。基于 Hugging Face Transformers 与 Datasets，训练与评估均可通过 `config/params.ini` 配置完成。

## 1. 环境与安装

- **Python**: >= 3.8
- **依赖**:
  - torch (建议按你的 CUDA 版本安装)
  - transformers
  - datasets
  - scikit-learn
  - matplotlib

推荐通过 `requirements.txt` 一键安装：
```bash
pip install -r requirements.txt
```

若需 GPU，请参考 PyTorch 官网选择与你 CUDA 对应的命令安装 `torch` 后再安装其余依赖。

## 2. 数据与模型
- 默认数据集：`lansinuote/ChnSentiCorp`（Hugging Face Hub 自动下载）
- 预训练模型：`bert-base-chinese`

以上均可在 `config/params.ini` 中修改。

## 3. 配置说明（config/params.ini）
```ini
[data]
batch_size = 64
max_length = 500
dataset_name = lansinuote/ChnSentiCorp

[model]
num_classes = 2
pretrained_model_name = bert-base-chinese
freeze_bert = true

[train]
epochs = 30
lr = 5e-5
save_dir = checkpoints
log_interval = 5
val_interval = 50

[viz]
figures_dir = figures

[test]
checkpoint_dir = checkpoints
checkpoint_name = checkpoint_epoch_29.pt
results_dir = results
```
- **data**: 批大小、序列最长长度、数据集名称
- **model**: 类别数、预训练模型、是否冻结 BERT 编码器参数
- **train**: 训练轮次、学习率、日志与验证打印间隔、权重保存目录
- **viz**: 可视化图保存目录
- **test**: 评估所用的检查点与输出目录

## 4. 训练与验证
1) 修改 `config/params.ini` 中的训练参数（如 `epochs`, `lr` 等）。
2) 运行训练：
```bash
python train.py
```
- 训练过程中会定期在验证集上评估，日志输出到 `logs/train.txt`。
- 每个 epoch 结束会保存检查点到 `checkpoints/`，并在 `figures/` 绘制并更新 `loss.png`, `accuracy.png`, `f1.png`。

## 5. 测试/评估
- 设置 `config/params.ini` 中 `[test]` 的 `checkpoint_name` 为需要评估的检查点。
- 运行：
```bash
python test.py
```
- 评估日志写入 `logs/test.txt`，包含 Accuracy、Precision、Recall、F1、分类报告与混淆矩阵统计。

## 6. 结果展示（30 epochs）
在验证/测试集上，使用 `bert-base-chinese`、`freeze_bert=true`、`epochs=30` 的典型结果如下（来自 `logs/test.txt`）：
- **Accuracy**: 0.8958
- **Precision**: 0.8962
- **Recall**: 0.8956
- **F1-Score**: 0.8958

训练过程中生成的可视化曲线：
- `figures/loss.png`: 训练损失曲线
- `figures/accuracy.png`: 训练与验证准确率曲线
- `figures/f1.png`: 训练与验证 macro-F1 曲线

注：由于随机种子、显卡/CPU、依赖版本差异，实际数值可能有轻微波动。

## 7. 项目结构
```text
d:
└─AI\emotion-classify
  ├─checkpoints/                 # 训练权重（每个 epoch 保存）
  ├─config/
  │  ├─common.ini                # 日志等通用配置（可选）
  │  └─params.ini                # 主要参数配置
  ├─figures/                     # 训练可视化曲线
  ├─logs/
  │  ├─train.txt                 # 训练日志
  │  └─test.txt                  # 测试/评估日志
  ├─results/                     # 评估产出（如需要）
  ├─dataset.py                   # 数据集与 DataLoader 构建
  ├─train.py                     # 训练入口
  ├─test.py                      # 评估入口
  ├─requirements.txt             # Python 依赖
  └─utils/
     ├─logger.py                 # 日志封装
     └─figure.py                 # 绘图工具
```

## 8. 常见问题
- 第一次运行较慢：需下载预训练模型与数据集。
- CUDA 相关错误：请确认 PyTorch 安装版本与 CUDA 驱动匹配。
- 中文分词长度：如遇显存不足，尝试降低 `max_length` 或 `batch_size`。

## 9. 许可
此项目仅供学习与研究使用。
