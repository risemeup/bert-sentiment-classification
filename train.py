import time
import os
import logging
import configparser
import torch
import torch.nn as nn
from transformers import BertModel
from dataset import create_dataloaders
from sklearn.metrics import f1_score
from utils.logger import get_logger
from figure import draw_loss_curve, draw_accuracy_curve, draw_f1_curve

logger = get_logger(name="train")

class BertForMultiClassification(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='bert-base-chinese', freeze_bert=True):
        super().__init__()
        # 加载预训练BERT模型（仅包含Encoder）
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        # 冻结BERT参数（可选，小数据集可冻结以减少过拟合）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 分类头：全连接层（输入维度为BERT的hidden_size，输出维度为类别数）
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        # 初始化分类头参数
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT输出：last_hidden_state是所有token的隐藏状态，pooler_output是[CLS]的池化结果
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 取[CLS]的隐藏状态（也可用pooler_output，效果类似）
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_size]
        
        # 分类头输出类别logits（未经过softmax）
        logits = self.classifier(cls_embedding)  # shape: [batch_size, num_classes]
        return logits

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in data_loader:
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
            preds = logits.argmax(dim=1).detach().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().tolist())
    acc = sum(int(p==y) for p, y in zip(all_preds, all_labels)) / max(1, len(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    model.train()
    return acc, f1


def train_model(model, train_loader, val_loader, device,
                epochs=1, lr=5e-4, save_dir: str = 'checkpoints',
                log_interval: int = 5, val_interval: int = 20,
                figures_dir: str = 'figures'):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    # 指标历史
    steps, train_losses, train_accs, train_f1s = [], [], [], []
    val_steps, val_accs, val_f1s = [], [], []
    for epoch in range(epochs):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            start_time = time.time()
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % max(1, log_interval) == 0:
                preds = logits.argmax(dim=1)
                accuracy = (preds == labels).sum().item() / len(labels)
                train_f1 = f1_score(labels.detach().cpu().tolist(), preds.detach().cpu().tolist(), average='macro')
                cost = time.time() - start_time
                logger.info(f'epoch {epoch} step {global_step} | 损失 {loss.item():.4f} | 准确率 {accuracy:.4f} | 训练macro-F1 {train_f1:.4f} | 耗时 {cost:.4f}s')
                steps.append(global_step)
                train_losses.append(float(loss.item()))
                train_accs.append(float(accuracy))
                train_f1s.append(float(train_f1))

            if global_step % max(1, val_interval) == 0:
                val_st = time.time()
                val_acc, val_f1 = evaluate(model, val_loader, device)
                cost = time.time() - val_st
                logger.info(f'epoch {epoch} step {global_step} | 验证集: 准确率 {val_acc:.4f} | macro-F1 {val_f1:.4f} | 耗时 {cost:.4f}s')
                val_steps.append(global_step)
                val_accs.append(float(val_acc))
                val_f1s.append(float(val_f1))

        # 每个epoch结束后保存一次检查点
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'num_classes': model.classifier.out_features,
                'lr': lr,
            }
        }
        ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(ckpt, ckpt_path)
        logger.info(f'保存检查点到 {ckpt_path}')

    # 绘制曲线
    try:
        os.makedirs(figures_dir, exist_ok=True)
        draw_loss_curve(steps, train_losses, out_path=os.path.join(figures_dir, 'loss.png'))
        draw_accuracy_curve(steps, train_accs, val_steps, val_accs, out_path=os.path.join(figures_dir, 'accuracy.png'))
        draw_f1_curve(steps, train_f1s, val_steps, val_f1s, out_path=os.path.join(figures_dir, 'f1.png'))
        logger.info('训练曲线已保存到 figures/ 目录')
    except Exception as e:
        logger.error(f'绘图失败: {e}')



if __name__ == '__main__':
    # 读取配置
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join('config', 'params.ini'), encoding='utf-8')

    # 通用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 数据
    batch_size = cfg.getint('data', 'batch_size', fallback=64)
    max_length = cfg.getint('data', 'max_length', fallback=500)
    dataset_name = cfg.get('data', 'dataset_name', fallback='lansinuote/ChnSentiCorp')

    # 模型
    num_classes = cfg.getint('model', 'num_classes', fallback=2)
    pretrained_model_name = cfg.get('model', 'pretrained_model_name', fallback='bert-base-chinese')
    freeze_bert = cfg.getboolean('model', 'freeze_bert', fallback=True)

    # 训练
    epochs = cfg.getint('train', 'epochs', fallback=1)
    lr = cfg.getfloat('train', 'lr', fallback=5e-4)
    save_dir = cfg.get('train', 'save_dir', fallback='checkpoints')
    log_interval = cfg.getint('train', 'log_interval', fallback=5)
    val_interval = cfg.getint('train', 'val_interval', fallback=20)

    # 可视化
    figures_dir = cfg.get('viz', 'figures_dir', fallback='figures')

    # 创建数据
    train_loader, val_loader = create_dataloaders(batch_size=batch_size,
                                                  max_length=max_length,
                                                  dataset_name=dataset_name,
                                                  device=device)

    # 创建模型
    model = BertForMultiClassification(num_classes=num_classes,
                                       pretrained_model_name=pretrained_model_name,
                                       freeze_bert=freeze_bert)

    # 训练
    train_model(model, train_loader, val_loader, device=device,
                epochs=epochs, lr=lr, save_dir=save_dir,
                log_interval=log_interval, val_interval=val_interval,
                figures_dir=figures_dir)