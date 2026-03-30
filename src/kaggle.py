import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re, string, os
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. 环境与路径配置
# ==========================================
# 假设脚本在 src/ 目录下，向上退一级到达项目根目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / 'train.csv'
KAGGLE_FILE = DATA_DIR / 'test.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {device}")

# ==========================================
# 2. 数据处理与清洗
# ==========================================
def basic_clean(text):
    if not isinstance(text, str): return "unk"
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text if len(text) > 0 else "unk" # 确保长度 > 0 防止 LSTM 崩溃

def load_data():
    print("📊 Loading and preparing datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(KAGGLE_FILE)
    
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['rating'])
    
    train_df['clean'] = train_df['text'].apply(basic_clean)
    test_df['clean'] = test_df['text'].apply(basic_clean)
    
    return train_df, test_df, le

# ==========================================
# 3. 模型定义 (LSTM/BiLSTM)
# ==========================================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        # 拼接双向或取单向最后状态
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.rnn.bidirectional else hidden[-1,:,:]
        return self.fc(out)

class RNNData(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=128):
        self.seqs = [torch.tensor([word2idx.get(w, 1) for w in t.split()][:max_len] or [1]) for t in texts]
        self.labels = labels.values if labels is not None else None
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        return (self.seqs[i], torch.tensor(self.labels[i])) if self.labels is not None else self.seqs[i]

def rnn_collate(batch):
    if isinstance(batch[0], tuple):
        items, labels = zip(*batch)
        return pad_sequence(items, batch_first=True, padding_value=0), torch.stack(labels), torch.tensor([len(x) for x in items])
    return pad_sequence(batch, batch_first=True, padding_value=0), torch.tensor([len(x) for x in batch])

# ==========================================
# 4. 训练与评估引擎
# ==========================================
def run_eval(model, loader, criterion, is_bert=False):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            if is_bert:
                input_ids, att_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=att_mask)
                loss, logits = criterion(outputs.logits, labels), outputs.logits
            else:
                texts, labels, lengths = [b.to(device) for b in batch]
                logits = model(texts, lengths)
                loss = criterion(logits, labels)
            
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return np.mean(losses), accuracy_score(trues, preds), f1_score(trues, preds, average='macro')

def train_engine(model, train_loader, val_loader, optimizer, criterion, epochs, is_bert=False):
    for epoch in range(epochs):
        model.train()
        tr_losses, tr_preds, tr_trues = [], [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            if is_bert:
                input_ids, att_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=att_mask)
                loss, logits = criterion(outputs.logits, labels), outputs.logits
            else:
                texts, labels, lengths = [b.to(device) for b in batch]
                logits = model(texts, lengths)
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            tr_losses.append(loss.item())
            tr_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            tr_trues.extend(labels.cpu().numpy())
        
        # 结果汇总打印
        tr_loss = np.mean(tr_losses)
        tr_acc = accuracy_score(tr_trues, tr_preds)
        tr_f1 = f1_score(tr_trues, tr_preds, average='macro')
        
        te_loss, te_acc, te_f1 = run_eval(model, val_loader, criterion, is_bert)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Train Macro-F1: {tr_f1:.4f} | "
              f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f} | Test Macro-F1: {te_f1:.4f}")

# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    train_df, test_df, le = load_data()
    # 按照 8:2 划分验证集
    df_tr, df_val = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
    
    # --- RNN 部分 ---
    all_words = " ".join(df_tr['clean']).split()
    vocab = {w: i+2 for i, (w, _) in enumerate(pd.Series(all_words).value_counts()[:15000].items())}
    vocab["<PAD>"], vocab["<UNK>"] = 0, 1
    
    for name, is_bi in [("LSTM", False), ("BiLSTM", True)]:
        print(f"\n⚡ Starting {name} Experiment")
        t_loader = DataLoader(RNNData(df_tr['clean'], df_tr['label'], vocab), batch_size=32, shuffle=True, collate_fn=rnn_collate)
        v_loader = DataLoader(RNNData(df_val['clean'], df_val['label'], vocab), batch_size=32, collate_fn=rnn_collate)
        
        model = RNNClassifier(len(vocab), 128, 256, len(le.classes_), is_bi).to(device)
        train_engine(model, t_loader, v_loader, optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss(), epochs=3)
        
        # 推理并保存
        model.eval()
        test_loader = DataLoader(RNNData(test_df['clean'], None, vocab), batch_size=32, collate_fn=rnn_collate)
        p = []
        for texts, lengths in test_loader:
            p.extend(torch.argmax(model(texts.to(device), lengths.to(device)), dim=1).cpu().numpy())
        pd.DataFrame({'Id': test_df['Id'], 'Rating': le.inverse_transform(p)}).to_csv(OUTPUTS_DIR/f'submission_{name.lower()}.csv', index=False)

    # --- DistilBERT 部分 ---
    print(f"\n⚡ Starting DistilBERT Experiment")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    def get_bert_loader(df, shuffle=False):
        enc = tokenizer(df['text'].fillna("unk").tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
        dataset = [{
            'input_ids': enc['input_ids'][i], 
            'attention_mask': enc['attention_mask'][i], 
            'labels': torch.tensor(df['label'].iloc[i]) if 'label' in df else torch.tensor(0)
        } for i in range(len(df))]
        return DataLoader(dataset, batch_size=16, shuffle=shuffle)

    bt_loader = get_bert_loader(df_tr, shuffle=True)
    bv_loader = get_bert_loader(df_val)
    
    model_b = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le.classes_)).to(device)
    train_engine(model_b, bt_loader, bv_loader, optim.AdamW(model_b.parameters(), lr=2e-5), nn.CrossEntropyLoss(), epochs=2, is_bert=True)
    
    # 推理并保存 (修正语法错误)
    model_b.eval()
    bp = []
    test_texts = test_df['text'].fillna("unk").tolist()
    for i in tqdm(range(0, len(test_texts), 16), desc="BERT Predicting"):
        inputs = tokenizer(test_texts[i:i+16], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            bp.extend(torch.argmax(model_b(**inputs).logits, dim=1).cpu().numpy())
    
    pd.DataFrame({'Id': test_df['Id'], 'Rating': le.inverse_transform(bp)}).to_csv(OUTPUTS_DIR/'submission_distilbert.csv', index=False)

    print(f"\n🎉 All models trained and submissions saved in: {OUTPUTS_DIR}")