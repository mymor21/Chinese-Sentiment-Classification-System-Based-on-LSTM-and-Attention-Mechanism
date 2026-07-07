"""
统一训练循环 + Early Stopping + 模型保存
"""

import os
import json
import time
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from config import (
    MODEL_DIR, PROCESSED_DIR, DEVICE, RANDOM_SEED,
    MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE,
    GRAD_CLIP, HIDDEN_DIM, NUM_LAYERS, DROPOUT, EMBED_DIM, NUM_CLASSES,
)
from dataset import create_dataloaders, load_processed_data


def set_seed(seed: int = RANDOM_SEED):
    """固定随机种子。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """训练一个 epoch。返回 (平均损失, 准确率)。"""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for sequences, labels in loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """评估模型。返回 (平均损失, 准确率, macro F1)。"""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for sequences, labels in loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(sequences)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
) -> dict:
    """
    训练一个模型，返回训练历史。

    Args:
        model: PyTorch 模型
        model_name: 模型名称 (用于保存)
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader
        lr: 初始学习率
        max_epochs: 最大 epoch 数
        early_stop_patience: early stopping 容忍度

    Returns:
        训练历史 dict: {train_loss, val_loss, val_acc, val_f1, train_acc}
    """
    print(f"\n{'='*60}")
    print(f"[train] 训练模型: {model_name}")
    print(f"{'='*60}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  设备: {DEVICE}")
    print(f"  学习率: {lr}, Early Stop Patience: {early_stop_patience}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        # 验证
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        # 学习率调度
        scheduler.step(val_loss)

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:2d}/{max_epochs} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  [train] Early stopping at epoch {epoch}")
                break

    # 恢复最佳模型
    model.load_state_dict(best_model_state)

    # 保存模型（附带配置，方便加载时重建）
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    model_config = {
        "vocab_size": getattr(model, "vocab_size", None),
        "embed_dim": getattr(model, "embed_dim", None),
        "max_seq_len": getattr(model, "max_seq_len", None),
        "num_layers": getattr(model, "num_layers", None),
        "num_heads": getattr(model, "num_heads", None),
        "ff_dim": getattr(model, "ff_dim", None),
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "history": history,
        "model_config": model_config,
    }, model_path)
    print(f"  [train] 模型已保存至 {model_path}")

    # 保存训练历史
    history_path = os.path.join(MODEL_DIR, f"{model_name}_history.json")
    # 转 Python float
    hist_json = {k: [float(x) for x in v] for k, v in history.items()}
    with open(history_path, "w") as f:
        json.dump(hist_json, f, indent=2)

    return history


def train_all_models() -> dict[str, dict]:
    """
    训练所有三个模型。

    Returns:
        {model_name: history} 字典
    """
    set_seed()

    # 加载数据
    print("[train] 加载预处理数据...")
    data = load_processed_data()
    train_loader, val_loader, _ = create_dataloaders(data)

    # 嵌入矩阵
    embedding_matrix = data.get("embedding_matrix", None)
    if embedding_matrix is not None:
        embedding_matrix = torch.FloatTensor(embedding_matrix)

    vocab_size = data["vocab_size"]
    max_seq_len = data.get("max_seq_len", 128)

    # 导入模型
    from models.rnn import SentimentRNN
    from models.lstm import SentimentLSTM
    from models.attention_lstm import SentimentAttentionLSTM
    from models.cnn_lstm import SentimentCNNBiLSTM

    # 从零训练的模型
    model_specs = {
        "rnn": (SentimentRNN, {}),
        "lstm": (SentimentLSTM, {}),
        "attention_lstm": (SentimentAttentionLSTM, {}),
        "cnn_lstm": (SentimentCNNBiLSTM, {}),
    }

    all_histories = {}

    for name, (ModelClass, extra_kwargs) in model_specs.items():
        # 检查是否已训练
        model_path = os.path.join(MODEL_DIR, f"{name}.pt")
        base_kwargs = dict(
            vocab_size=vocab_size, embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES, dropout=DROPOUT,
            pretrained_embeddings=embedding_matrix,
        )
        base_kwargs.update(extra_kwargs)

        if os.path.exists(model_path):
            print(f"\n[train] {name}.pt 已存在，跳过训练")
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model = ModelClass(**base_kwargs)
            model.load_state_dict(checkpoint["model_state_dict"])
            all_histories[name] = checkpoint.get("history", {})
            continue

        model = ModelClass(**base_kwargs)

        history = train_model(model, name, train_loader, val_loader)
        all_histories[name] = history

    # ── BERT 预训练模型（单独训练管线） ──
    bert_path = os.path.join(MODEL_DIR, "bert.pt")
    if not os.path.exists(bert_path):
        print(f"\n{'='*60}")
        print(f"[train] 训练 BERT (预训练对照组)")
        print(f"{'='*60}")
        train_bert(data, train_loader, val_loader)
    else:
        print(f"\n[train] bert.pt 已存在，跳过训练")

    return all_histories


def train_bert(data, _, val_loader):
    """
    BERT 微调。使用原始文本 + BERT tokenizer，学习率更低。
    """
    from models.bert import SentimentBERT, get_bert_tokenizer
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split

    # 加载原始文本和标签
    texts = data["texts"]
    labels = data["labels"]

    # 划分训练/验证（与 create_dataloaders 一致）
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.176, stratify=labels, random_state=RANDOM_SEED
    )

    # BERT tokenizer
    tokenizer = get_bert_tokenizer()

    # Tokenize
    train_enc = tokenizer(
        list(X_train), padding=True, truncation=True,
        max_length=256, return_tensors="pt"
    )
    val_enc = tokenizer(
        list(X_val), padding=True, truncation=True,
        max_length=256, return_tensors="pt"
    )

    class BERTDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.LongTensor(labels.values if hasattr(labels, 'values') else list(labels))
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    bert_train_ds = BERTDataset(train_enc, y_train)
    bert_val_ds = BERTDataset(val_enc, y_val)
    bert_train_loader = DataLoader(bert_train_ds, batch_size=16, shuffle=True)
    bert_val_loader = DataLoader(bert_val_ds, batch_size=16, shuffle=False)

    # 模型
    model = SentimentBERT(num_classes=NUM_CLASSES, dropout=0.2).to(DEVICE)
    print(f"  BERT 参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  训练样本: {len(bert_train_ds)}, 验证: {len(bert_val_ds)}")

    # 训练（BERT 用更小的学习率）
    history = train_model_bert(model, "bert", bert_train_loader, bert_val_loader,
                               lr=2e-5, max_epochs=5, early_stop_patience=3)
    return history


def train_model_bert(model, model_name, train_loader, val_loader,
                     lr=2e-5, max_epochs=5, early_stop_patience=3):
    """BERT 专用训练循环。"""
    import copy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(lbls)
            correct += (logits.argmax(1) == lbls).sum().item()
            total += len(lbls)

        train_loss = total_loss / total
        train_acc = correct / total

        # 验证
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        all_preds, all_lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["labels"].to(DEVICE)
                logits = model(input_ids, attention_mask)
                v_loss += criterion(logits, lbls).item() * len(lbls)
                v_correct += (logits.argmax(1) == lbls).sum().item()
                v_total += len(lbls)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_lbls.extend(lbls.cpu().numpy())

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        val_f1 = f1_score(all_lbls, all_preds, average="macro")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch}/{max_epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"  [train] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    torch.save({"model_state_dict": model.state_dict(), "model_name": model_name, "history": history}, model_path)
    print(f"  [train] 模型已保存至 {model_path}")
    return history


if __name__ == "__main__":
    train_all_models()
