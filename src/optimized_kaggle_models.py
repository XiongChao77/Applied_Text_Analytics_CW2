import random
import re
import string
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 环境与路径配置（保留你的方式）
# ==========================================
# 假设脚本在 src/ 目录下，向上退一级到达项目根目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / 'train.csv'
KAGGLE_FILE = DATA_DIR / 'test.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")


# ==========================================
# 2. 随机种子与工具函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def basic_clean(text):
    if not isinstance(text, str):
        return "unk"
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "unk"


def print_metrics(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n===== {title} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return acc, f1


def save_submission(path, ids, preds_zero_based):
    sub = pd.DataFrame({
        'Id': ids,
        'Rating': np.asarray(preds_zero_based) + 1,
    })
    sub.to_csv(path, index=False)
    print(f"Saved: {path}")


# ==========================================
# 3. 数据加载
# ==========================================
def load_data():
    print("Loading and preparing datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(KAGGLE_FILE)

    train_df['text'] = train_df['text'].fillna('unk').astype(str)
    test_df['text'] = test_df['text'].fillna('unk').astype(str)

    # 直接映射到 0..4，更透明且可复现
    train_df['label'] = train_df['rating'].astype(int) - 1

    train_df['clean'] = train_df['text'].apply(basic_clean)
    test_df['clean'] = test_df['text'].apply(basic_clean)
    return train_df, test_df


# ==========================================
# 4. RNN 数据与模型
# ==========================================
class RNNData(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=128):
        self.seqs = []
        for text in texts:
            ids = [word2idx.get(w, 1) for w in str(text).split()][:max_len]
            if not ids:
                ids = [1]
            self.seqs.append(torch.tensor(ids, dtype=torch.long))
        self.labels = labels.to_numpy(dtype=np.int64) if labels is not None else None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.seqs[idx]
        return self.seqs[idx], torch.tensor(self.labels[idx], dtype=torch.long)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, bidirectional=False, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        if self.rnn.bidirectional:
            out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            out = hidden[-1]
        out = self.dropout(out)
        return self.fc(out)


def rnn_collate(batch):
    if isinstance(batch[0], tuple):
        items, labels = zip(*batch)
        padded = pad_sequence(items, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(x) for x in items], dtype=torch.long)
        labels = torch.stack(labels)
        return padded, labels, lengths
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
    return padded, lengths


def build_vocab(texts, max_vocab=15000, min_freq=2):
    counts = {}
    for text in texts:
        for w in str(text).split():
            counts[w] = counts.get(w, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in sorted_items:
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[word] = len(vocab)
    return vocab


# ==========================================
# 5. BERT 数据
# ==========================================
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ==========================================
# 6. 训练/评估通用逻辑
# ==========================================
def run_eval_rnn(model, loader, criterion):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)
            logits = model(texts, lengths)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return np.mean(losses), np.array(trues), np.array(preds)


def train_rnn_model(model, train_loader, val_loader, name, lr=1e-3, epochs=5):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses, tr_preds, tr_trues = [], [], []

        for texts, labels, lengths in tqdm(train_loader, desc=f"{name} Epoch {epoch}", leave=False):
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model(texts, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_losses.append(loss.item())
            tr_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            tr_trues.extend(labels.cpu().numpy())

        tr_loss = np.mean(tr_losses)
        tr_acc = accuracy_score(tr_trues, tr_preds)
        tr_f1 = f1_score(tr_trues, tr_preds, average='macro')

        val_loss, y_true, y_pred = run_eval_rnn(model, val_loader, criterion)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='macro')

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Train Macro-F1: {tr_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_true, y_pred = run_eval_rnn(model, val_loader, criterion)
    print_metrics(f"{name} Final Validation", y_true + 1, y_pred + 1)
    return model


def predict_rnn(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for texts, lengths in loader:
            texts = texts.to(DEVICE)
            lengths = lengths.to(DEVICE)
            logits = model(texts, lengths)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return preds


def run_eval_bert(model, loader):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(batch['labels'].cpu().numpy())
    return np.mean(losses), np.array(trues), np.array(preds)


def train_bert_model(model, train_loader, val_loader, name, lr=2e-5, epochs=2):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses, tr_preds, tr_trues = [], [], []

        for batch in tqdm(train_loader, desc=f"{name} Epoch {epoch}", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tr_losses.append(loss.item())
            tr_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            tr_trues.extend(batch['labels'].cpu().numpy())

        tr_loss = np.mean(tr_losses)
        tr_acc = accuracy_score(tr_trues, tr_preds)
        tr_f1 = f1_score(tr_trues, tr_preds, average='macro')

        val_loss, y_true, y_pred = run_eval_bert(model, val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='macro')

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Train Macro-F1: {tr_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_true, y_pred = run_eval_bert(model, val_loader)
    print_metrics(f"{name} Final Validation", y_true + 1, y_pred + 1)
    return model


def predict_bert(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="DistilBERT Predicting", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
    return preds


# ==========================================
# 7. 主流程：训练三种模型并导出三份结果
# ==========================================
if __name__ == "__main__":
    train_df, test_df = load_data()

    df_tr, df_val = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label'],
    )

    print(f"Train split: {df_tr.shape}, Validation split: {df_val.shape}, Test: {test_df.shape}")

    # ---------- LSTM / BiLSTM ----------
    vocab = build_vocab(df_tr['clean'].tolist(), max_vocab=15000, min_freq=2)
    print(f"RNN vocab size: {len(vocab)}")

    rnn_train_loader = DataLoader(
        RNNData(df_tr['clean'], df_tr['label'], vocab, max_len=128),
        batch_size=32,
        shuffle=True,
        collate_fn=rnn_collate,
    )
    rnn_val_loader = DataLoader(
        RNNData(df_val['clean'], df_val['label'], vocab, max_len=128),
        batch_size=64,
        shuffle=False,
        collate_fn=rnn_collate,
    )
    rnn_test_loader = DataLoader(
        RNNData(test_df['clean'], None, vocab, max_len=128),
        batch_size=64,
        shuffle=False,
        collate_fn=rnn_collate,
    )

    print("\nStarting LSTM Experiment")
    lstm_model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        output_dim=5,
        bidirectional=False,
        dropout=0.3,
    )
    lstm_model = train_rnn_model(lstm_model, rnn_train_loader, rnn_val_loader, name="LSTM", lr=1e-3, epochs=4)
    lstm_preds = predict_rnn(lstm_model, rnn_test_loader)
    save_submission(OUTPUTS_DIR / 'submission_lstm.csv', test_df['Id'], lstm_preds)

    print("\nStarting BiLSTM Experiment")
    bilstm_model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        output_dim=5,
        bidirectional=True,
        dropout=0.3,
    )
    bilstm_model = train_rnn_model(bilstm_model, rnn_train_loader, rnn_val_loader, name="BiLSTM", lr=1e-3, epochs=4)
    bilstm_preds = predict_rnn(bilstm_model, rnn_test_loader)
    save_submission(OUTPUTS_DIR / 'submission_bilstm.csv', test_df['Id'], bilstm_preds)

    # ---------- DistilBERT ----------
    print("\nStarting DistilBERT Experiment")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    bert_train_loader = DataLoader(
        BertDataset(df_tr['text'].tolist(), df_tr['label'].tolist(), tokenizer, max_len=128),
        batch_size=16,
        shuffle=True,
    )
    bert_val_loader = DataLoader(
        BertDataset(df_val['text'].tolist(), df_val['label'].tolist(), tokenizer, max_len=128),
        batch_size=32,
        shuffle=False,
    )
    bert_test_loader = DataLoader(
        BertDataset(test_df['text'].tolist(), None, tokenizer, max_len=128),
        batch_size=32,
        shuffle=False,
    )

    bert_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=5,
    )
    bert_model = train_bert_model(bert_model, bert_train_loader, bert_val_loader, name="DistilBERT", lr=2e-5, epochs=4)
    bert_preds = predict_bert(bert_model, bert_test_loader)
    save_submission(OUTPUTS_DIR / 'submission_distilbert.csv', test_df['Id'], bert_preds)

    print(f"\nAll models finished. Files saved in: {OUTPUTS_DIR}")
