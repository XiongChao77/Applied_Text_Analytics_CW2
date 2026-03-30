import os
import re
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Text preprocessing
# =========================
def simple_clean(text: str) -> str:
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def whitespace_tokenize(text: str) -> List[str]:
    return simple_clean(text).split()


# =========================
# Vocabulary and encoding
# =========================
class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, min_freq: int = 2, max_size: Optional[int] = 30000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi = {self.PAD: 0, self.UNK: 1}
        self.itos = [self.PAD, self.UNK]

    def build(self, texts: List[str]):
        freq = {}
        for text in texts:
            for tok in whitespace_tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1

        items = [(tok, c) for tok, c in freq.items() if c >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_size is not None:
            items = items[: max(0, self.max_size - len(self.itos))]

        for tok, _ in items:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text: str, max_len: int) -> Tuple[List[int], int]:
        tokens = whitespace_tokenize(text)
        ids = [self.stoi.get(tok, self.stoi[self.UNK]) for tok in tokens[:max_len]]
        length = len(ids)
        if length < max_len:
            ids += [self.stoi[self.PAD]] * (max_len - length)
        return ids, min(length, max_len)

    def __len__(self):
        return len(self.itos)


# =========================
# Datasets
# =========================
class SequenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], vocab: Vocab, max_len: int):
        self.texts = list(texts)
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, length = self.vocab.encode(self.texts[idx], self.max_len)
        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
        if self.labels is not None:
            # map rating 1-5 -> class 0-4
            item["labels"] = torch.tensor(int(self.labels[idx]) - 1, dtype=torch.long)
        return item


# =========================
# Models
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, pad_idx: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, length):
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, length.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = self.dropout(hidden[-1])
        return self.fc(out)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, pad_idx: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, length):
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, length.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(out)
        return self.fc(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        num_classes: int,
        pad_idx: int,
        max_len: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, length):
        mask = input_ids.eq(self.pad_idx)
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        # masked mean pooling
        non_pad = (~mask).unsqueeze(-1)
        summed = (x * non_pad).sum(dim=1)
        counts = non_pad.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        pooled = self.dropout(pooled)
        return self.fc(pooled)


# =========================
# DistilBERT helpers
# =========================
def ensure_transformers():
    try:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            "DistilBERT requires transformers. Install it with: pip install transformers"
        ) from e


class BertTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            simple_clean(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]) - 1, dtype=torch.long)
        return item


# =========================
# Training / Evaluation
# =========================
@dataclass
class TrainConfig:
    model_name: str
    train_file: str
    kaggle_file: Optional[str]
    output_dir: str
    batch_size: int = 32
    epochs: int = 4
    lr: float = 1e-3
    max_len: int = 128
    embed_dim: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 2
    min_freq: int = 2
    max_vocab_size: int = 30000
    test_size: float = 0.2
    random_state: int = 42


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, criterion, device, is_bert: bool = False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        if is_bert:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
        else:
            logits = model(batch["input_ids"], batch["length"])
            loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_model(model, loader, device, is_bert: bool = False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if is_bert:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
            else:
                logits = model(batch["input_ids"], batch["length"])
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, digits=4)
    return acc, f1, report


def predict_sequence_model(model, loader, device, is_bert: bool = False) -> List[int]:
    model.eval()
    preds_all = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if is_bert:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
            else:
                logits = model(batch["input_ids"], batch["length"])
            preds = torch.argmax(logits, dim=1) + 1  # back to rating 1-5
            preds_all.extend(preds.cpu().numpy().tolist())
    return preds_all


# =========================
# Main pipeline
# =========================
def load_train_data(train_file: str) -> pd.DataFrame:
    df = pd.read_csv(train_file)
    if "text" not in df.columns or "rating" not in df.columns:
        raise ValueError("train.csv must contain columns: text, rating")
    df = df[["text", "rating"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["rating"] = df["rating"].astype(int)
    return df


def load_kaggle_data(kaggle_file: str) -> pd.DataFrame:
    df = pd.read_csv(kaggle_file)
    if "Id" not in df.columns or "text" not in df.columns:
        raise ValueError("KAGGLE_FILE must contain columns: Id, text")
    df = df[["Id", "text"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    return df


def run_lstm_family(cfg: TrainConfig, bidirectional: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_train_data(cfg.train_file)
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["rating"].tolist(),
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["rating"],
    )

    vocab = Vocab(min_freq=cfg.min_freq, max_size=cfg.max_vocab_size)
    vocab.build(X_train)

    train_ds = SequenceDataset(X_train, y_train, vocab, cfg.max_len)
    val_ds = SequenceDataset(X_val, y_val, vocab, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if bidirectional:
        model = BiLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=5,
            pad_idx=vocab.stoi[Vocab.PAD],
        )
    else:
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=5,
            pad_idx=vocab.stoi[Vocab.PAD],
        )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, is_bert=False)
        acc, f1, _ = evaluate_model(model, val_loader, device, is_bert=False)
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | val_acc={acc:.4f} | val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    acc, f1, report = evaluate_model(model, val_loader, device, is_bert=False)
    print("\nFinal validation results")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(report)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_path = os.path.join(cfg.output_dir, f"{cfg.model_name}_model.pt")
    vocab_path = os.path.join(cfg.output_dir, f"{cfg.model_name}_vocab.pt")
    torch.save(model.state_dict(), model_path)
    torch.save(vocab, vocab_path)
    print(f"Saved model to {model_path}")

    if cfg.kaggle_file:
        kaggle_df = load_kaggle_data(cfg.kaggle_file)
        test_ds = SequenceDataset(kaggle_df["text"].tolist(), None, vocab, cfg.max_len)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        preds = predict_sequence_model(model, test_loader, device, is_bert=False)
        sub = pd.DataFrame({"Id": kaggle_df["Id"], "Rating": preds})
        sub_path = os.path.join(cfg.output_dir, f"submission_{cfg.model_name}.csv")
        sub.to_csv(sub_path, index=False)
        print(f"Saved submission to {sub_path}")



def run_transformer(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_train_data(cfg.train_file)
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["rating"].tolist(),
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["rating"],
    )

    vocab = Vocab(min_freq=cfg.min_freq, max_size=cfg.max_vocab_size)
    vocab.build(X_train)

    train_ds = SequenceDataset(X_train, y_train, vocab, cfg.max_len)
    val_ds = SequenceDataset(X_val, y_val, vocab, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_layers=cfg.num_layers,
        num_classes=5,
        pad_idx=vocab.stoi[Vocab.PAD],
        max_len=cfg.max_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, is_bert=False)
        acc, f1, _ = evaluate_model(model, val_loader, device, is_bert=False)
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | val_acc={acc:.4f} | val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    acc, f1, report = evaluate_model(model, val_loader, device, is_bert=False)
    print("\nFinal validation results")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(report)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_path = os.path.join(cfg.output_dir, f"{cfg.model_name}_model.pt")
    vocab_path = os.path.join(cfg.output_dir, f"{cfg.model_name}_vocab.pt")
    torch.save(model.state_dict(), model_path)
    torch.save(vocab, vocab_path)
    print(f"Saved model to {model_path}")

    if cfg.kaggle_file:
        kaggle_df = load_kaggle_data(cfg.kaggle_file)
        test_ds = SequenceDataset(kaggle_df["text"].tolist(), None, vocab, cfg.max_len)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        preds = predict_sequence_model(model, test_loader, device, is_bert=False)
        sub = pd.DataFrame({"Id": kaggle_df["Id"], "Rating": preds})
        sub_path = os.path.join(cfg.output_dir, f"submission_{cfg.model_name}.csv")
        sub.to_csv(sub_path, index=False)
        print(f"Saved submission to {sub_path}")



def run_distilbert(cfg: TrainConfig):
    ensure_transformers()
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_train_data(cfg.train_file)
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["rating"].tolist(),
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["rating"],
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_ds = BertTextDataset(X_train, y_train, tokenizer, cfg.max_len)
    val_ds = BertTextDataset(X_val, y_val, tokenizer, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_state = None
    best_f1 = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, None, device, is_bert=True)
        acc, f1, _ = evaluate_model(model, val_loader, device, is_bert=True)
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | val_acc={acc:.4f} | val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    acc, f1, report = evaluate_model(model, val_loader, device, is_bert=True)
    print("\nFinal validation results")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(report)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_dir = os.path.join(cfg.output_dir, f"{cfg.model_name}_model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Saved DistilBERT model to {model_dir}")

    if cfg.kaggle_file:
        kaggle_df = load_kaggle_data(cfg.kaggle_file)
        test_ds = BertTextDataset(kaggle_df["text"].tolist(), None, tokenizer, cfg.max_len)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        preds = predict_sequence_model(model, test_loader, device, is_bert=True)
        sub = pd.DataFrame({"Id": kaggle_df["Id"], "Rating": preds})
        sub_path = os.path.join(cfg.output_dir, f"submission_{cfg.model_name}.csv")
        sub.to_csv(sub_path, index=False)
        print(f"Saved submission to {sub_path}")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM / BiLSTM / Transformer / DistilBERT for rating prediction.")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "bilstm", "transformer", "distilbert"])
    parser.add_argument("--train_file", type=str, required=True, help="Path to train.csv with columns text,rating")
    parser.add_argument("--kaggle_file", type=str, default=None, help="Path to Kaggle/test CSV with columns Id,text")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_state)

    if args.lr is None:
        if args.model == "distilbert":
            args.lr = 2e-5
        elif args.model == "transformer":
            args.lr = 5e-4
        else:
            args.lr = 1e-3

    cfg = TrainConfig(
        model_name=args.model,
        train_file=args.train_file,
        kaggle_file=args.kaggle_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if cfg.model_name == "lstm":
        run_lstm_family(cfg, bidirectional=False)
    elif cfg.model_name == "bilstm":
        run_lstm_family(cfg, bidirectional=True)
    elif cfg.model_name == "transformer":
        run_transformer(cfg)
    elif cfg.model_name == "distilbert":
        run_distilbert(cfg)
    else:
        raise ValueError(f"Unsupported model: {cfg.model_name}")


if __name__ == "__main__":
    main()
