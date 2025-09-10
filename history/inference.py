import torch
import time
from datasets import load_dataset
from sklearn.metrics import f1_score
from dataset import create_dataloaders

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label


dataset = Dataset('train')
_, val_loader = create_dataloaders(batch_size=64, max_length=500, device=device)

print(f'数据集大小:{len(dataset)}, 样例：{dataset[0]}')

from transformers import BertTokenizer

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, token_type_ids, labels


#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=64,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

print(len(loader))
print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

from transformers import BertModel

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)

#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

#模型试算
out = pretrained(input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids)

print(out.last_hidden_state.shape)

#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out

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

model = Model().to(device)

print(model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)

from torch.optim import AdamW

#训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    st_time = time.time()
    # 数据已经在collate_fn中移动到GPU了
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        cost_time = time.time() - st_time

        print(f'步骤 {i}, 损失: {loss.item():.4f}, 准确率: {accuracy:.4f}, 耗时: {cost_time:.4f}s')
    if i % 20 == 0:
        val_st = time.time()
        val_acc, val_f1 = evaluate(model, val_loader, device)
        cost = time.time() - val_st
        print(f'step {i} | 验证集: 准确率 {val_acc:.4f} | macro-F1 {val_f1:.4f} | 耗时 {cost:.4f}s')

