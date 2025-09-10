import torch
from datasets import load_dataset
from transformers import BertTokenizer


class TextSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, dataset_name: str = 'lansinuote/ChnSentiCorp'):
        self.dataset = load_dataset(path=dataset_name, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        return item['text'], item['label']


def create_dataloaders(batch_size: int = 64,
                       max_length: int = 500,
                       dataset_name: str = 'lansinuote/ChnSentiCorp',
                       device: torch.device = torch.device('cpu')):
    token = BertTokenizer.from_pretrained('bert-base-chinese')

    def collate_fn(data):
        sents = [i[0] for i in data]
        labels = [i[1] for i in data]

        encoded = token.batch_encode_plus(
            batch_text_or_text_pairs=sents,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_length=True
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        token_type_ids = encoded['token_type_ids'].to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        return input_ids, attention_mask, token_type_ids, labels_tensor

    train_dataset = TextSentimentDataset('train', dataset_name)
    val_dataset = TextSentimentDataset('validation', dataset_name)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


