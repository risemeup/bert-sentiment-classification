import time
import os
import logging
import configparser
import torch
import torch.nn as nn
from transformers import BertModel
from dataset import create_dataloaders
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from utils.logger import get_logger

logger = get_logger(name="test")

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

def load_model(checkpoint_path, num_classes, pretrained_model_name, freeze_bert, device):
    """加载训练好的模型"""
    model = BertForMultiClassification(num_classes=num_classes, 
                                     pretrained_model_name=pretrained_model_name, 
                                     freeze_bert=freeze_bert)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f'成功加载模型: {checkpoint_path}')
    logger.info(f'模型训练轮次: {checkpoint.get("epoch", "unknown")}')
    
    return model

def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info('开始模型评估...')
    start_time = time.time()
    
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
            logits = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
            preds = logits.argmax(dim=1).detach().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            
            if (i + 1) % 10 == 0:
                logger.info(f'已处理 {i + 1}/{len(test_loader)} 个批次')
    
    eval_time = time.time() - start_time
    logger.info(f'评估完成，耗时: {eval_time:.2f}s')
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 详细分类报告
    class_names = ['负面', '正面']
    class_report = classification_report(all_labels, all_preds, 
                                       target_names=class_names, 
                                       digits=4)
    
    # 记录混淆矩阵到日志
    logger.info('混淆矩阵:')
    logger.info(f'真实\\预测    0(负面)    1(正面)')
    logger.info(f'0(负面)      {cm[0,0]:8d}  {cm[0,1]:8d}')
    logger.info(f'1(正面)      {cm[1,0]:8d}  {cm[1,1]:8d}')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_report': class_report,
        'predictions': all_preds,
        'labels': all_labels
    }

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
    
    # 测试配置
    checkpoint_dir = cfg.get('test', 'checkpoint_dir', fallback='checkpoints')
    checkpoint_name = cfg.get('test', 'checkpoint_name', fallback='checkpoint_epoch_0.pt')
    results_dir = cfg.get('test', 'results_dir', fallback='results')
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # 检查检查点文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f'检查点文件不存在: {checkpoint_path}')
        logger.info('可用的检查点文件:')
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    logger.info(f'  - {f}')
        exit(1)
    
    # 创建测试数据加载器
    logger.info('创建测试数据集...')
    _, _, test_loader = create_dataloaders(batch_size=batch_size,
                                       max_length=max_length,
                                       dataset_name=dataset_name,
                                       device=device)
    
    # 获取原始测试集数量
    from dataset import TextSentimentDataset
    test_dataset = TextSentimentDataset('test', dataset_name)
    logger.info(f'测试集原始样本数量: {len(test_dataset)}')
    logger.info(f'测试集批次数量: {len(test_loader)}')
    
    # 加载模型
    logger.info('加载训练好的模型...')
    model = load_model(checkpoint_path, num_classes, pretrained_model_name, freeze_bert, device)
    
    # 评估模型
    results = evaluate_model(model, test_loader)
    
    # 输出结果
    logger.info('=' * 50)
    logger.info('测试结果:')
    logger.info(f'准确率 (Accuracy): {results["accuracy"]:.4f}')
    logger.info(f'精确率 (Precision): {results["precision"]:.4f}')
    logger.info(f'召回率 (Recall): {results["recall"]:.4f}')
    logger.info(f'F1分数 (F1-Score): {results["f1"]:.4f}')
    logger.info('=' * 50)
    logger.info('详细分类报告:')
    logger.info('\n'+results['class_report'])
    logger.info('=' * 50)
    
    # 输出混淆矩阵统计信息
    cm = results['confusion_matrix']
    total_samples = cm.sum()
    correct_predictions = cm.trace()
    logger.info('混淆矩阵统计:')
    logger.info(f'总样本数: {total_samples}')
    logger.info(f'正确预测数: {correct_predictions}')
    logger.info(f'错误预测数: {total_samples - correct_predictions}')
    logger.info(f'各类别准确率:')
    for i, class_name in enumerate(['负面', '正面']):
        class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        logger.info(f'  {class_name}: {class_accuracy:.4f}')
    logger.info('=' * 50)