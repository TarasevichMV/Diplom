import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report


# Загрузка текстов и тегов
texts = []
with open(os.path.join(os.path.dirname(__file__), '..', 'texts', 'texts3.txt'), 'r', encoding='utf-8') as f:
    for item in f:
        texts.append(item)
df = pd.read_csv('tags/saved_tags.csv', header=None)
labels = df.values.tolist()

# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# Класс Dataset для работы с данными
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


# Параметры
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4

# Инициализация токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Создание DataLoader
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Инициализация модели
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=21)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

# Обучение модели
model.train()

for epoch in range(EPOCHS):
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Loss after epoch {epoch + 1}: {loss.item()}")


# Оценка модели
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions.append(logits)
            true_labels.append(labels)

    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().numpy()

    return predictions, true_labels


# Оценка на тестовой выборке
predictions, true_labels = evaluate(model, test_dataloader)

# Преобразование предсказаний в бинарный формат
predicted_labels = (torch.sigmoid(torch.tensor(predictions)) > 0.5).int().numpy()

# Метрики оценки
print(classification_report(true_labels, predicted_labels))
