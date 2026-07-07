"""
模型评估：测试集评估、混淆矩阵计算、模型对比
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
)

from config import (
    MODEL_DIR, EVALUATION_DIR, DEVICE, PROCESSED_DIR,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, EMBED_DIM, NUM_CLASSES,
    TEST_RATIO, RANDOM_SEED,
)
from dataset import create_dataloaders, load_processed_data


def load_trained_model(model_name: str, vocab_size: int, embedding_matrix: np.ndarray = None):
    """加载训练好的模型（RNN/LSTM/Attention-LSTM/CNN-BiLSTM）。"""
    from models.rnn import SentimentRNN
    from models.lstm import SentimentLSTM
    from models.attention_lstm import SentimentAttentionLSTM
    from models.cnn_lstm import SentimentCNNBiLSTM

    model_classes = {
        "rnn": SentimentRNN, "lstm": SentimentLSTM,
        "attention_lstm": SentimentAttentionLSTM, "cnn_lstm": SentimentCNNBiLSTM,
    }

    ModelClass = model_classes[model_name]
    emb = torch.FloatTensor(embedding_matrix) if embedding_matrix is not None else None

    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    saved_config = checkpoint.get("model_config", {})

    kwargs = dict(vocab_size=vocab_size, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                  dropout=DROPOUT, pretrained_embeddings=emb, hidden_dim=HIDDEN_DIM,
                  num_layers=NUM_LAYERS)
    model = ModelClass(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


def load_bert_model():
    """加载 BERT 模型。"""
    from models.bert import SentimentBERT
    model_path = os.path.join(MODEL_DIR, "bert.pt")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = SentimentBERT(num_classes=NUM_CLASSES, dropout=0.2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """获取模型在数据上的预测和真实标签。"""
    model.eval()
    all_preds, all_labels = [], []

    for sequences, labels in loader:
        sequences = sequences.to(DEVICE)
        logits = model(sequences)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def evaluate_model(
    model: nn.Module, model_name: str, test_loader: DataLoader
) -> dict:
    """
    评估单个模型，返回详细报告。
    """
    print(f"\n[evaluate] ===== {model_name} =====")

    preds, labels = get_predictions(model, test_loader)

    # 分类报告
    label_names = ["负面", "正面"]
    report = classification_report(
        labels, preds,
        target_names=label_names,
        digits=4,
        output_dict=True,
    )

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)

    # 基本指标
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall: {recall:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  混淆矩阵:\n{cm}")

    result = {
        "model_name": model_name,
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
    }

    # 保存
    report_path = os.path.join(EVALUATION_DIR, f"{model_name}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def measure_inference_time(model: nn.Module, loader: DataLoader, n_batches: int = 50) -> float:
    """测量推理时间 (ms/batch)。"""
    model.eval()
    times = []

    with torch.no_grad():
        for i, (sequences, _) in enumerate(loader):
            if i >= n_batches:
                break
            sequences = sequences.to(DEVICE)
            t0 = time.perf_counter()
            model(sequences)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            times.append(elapsed)

    return np.mean(times)


def compare_models(
    results: dict[str, dict],
    test_loader: DataLoader,
) -> dict:
    """
    横向对比三个模型。
    """
    print(f"\n[evaluate] ===== 模型对比 =====")
    print(f"{'模型':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)

    comparison = {}

    for name in ["rnn", "lstm", "attention_lstm", "cnn_lstm"]:
        result = results.get(name, {})
        acc = result.get("accuracy", 0)
        p = result.get("macro_precision", 0)
        r = result.get("macro_recall", 0)
        f1 = result.get("macro_f1", 0)
        print(f"{name:<20} {acc:>10.4f} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")
        comparison[name] = {
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        }

    # 保存对比结果
    with open(os.path.join(EVALUATION_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    return comparison


def main():
    """评估主流程。"""
    # 加载数据
    data = load_processed_data()
    _, _, test_loader = create_dataloaders(data)
    embedding_matrix = data.get("embedding_matrix")
    vocab_size = data["vocab_size"]

    results = {}

    for model_name in ["rnn", "lstm", "attention_lstm", "cnn_lstm"]:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        if not os.path.exists(model_path):
            print(f"[evaluate] 跳过 {model_name}：模型文件 {model_path} 不存在")
            continue

        model = load_trained_model(model_name, vocab_size, embedding_matrix)
        result = evaluate_model(model, model_name, test_loader)
        results[model_name] = result

        # 测量推理时间
        avg_time = measure_inference_time(model, test_loader)
        print(f"  平均推理时间: {avg_time:.2f} ms/batch")
        result["inference_time_ms"] = avg_time

    # BERT 单独评估
    bert_path = os.path.join(MODEL_DIR, "bert.pt")
    if os.path.exists(bert_path):
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, Dataset
        from models.bert import get_bert_tokenizer

        texts = data["texts"]
        labels = data["labels"]
        _, X_test, _, y_test = train_test_split(
            texts, labels, test_size=TEST_RATIO, stratify=labels, random_state=RANDOM_SEED
        )

        tokenizer = get_bert_tokenizer()
        test_enc = tokenizer(list(X_test), padding=True, truncation=True, max_length=256, return_tensors="pt")

        class BERTTestDS(Dataset):
            def __init__(self, enc, lbls):
                self.enc = enc
                self.lbls = torch.LongTensor(list(lbls))
            def __len__(self): return len(self.lbls)
            def __getitem__(self, i):
                return self.enc["input_ids"][i], self.enc["attention_mask"][i], self.lbls[i]

        bert_test_loader = DataLoader(BERTTestDS(test_enc, y_test), batch_size=16)

        model = load_bert_model()
        preds, trues = [], []
        with torch.no_grad():
            for input_ids, attn_mask, lbls in bert_test_loader:
                input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
                logits = model(input_ids, attn_mask)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(lbls.numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        acc = accuracy_score(trues, preds)
        p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="macro")
        cm = confusion_matrix(trues, preds)

        print(f"\n[evaluate] ===== bert =====")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {f1:.4f}")
        print(f"  混淆矩阵:\n{cm}")

        results["bert"] = {
            "model_name": "bert", "accuracy": acc,
            "macro_precision": p, "macro_recall": r, "macro_f1": f1,
            "confusion_matrix": cm.tolist(), "predictions": preds.tolist(), "labels": trues.tolist(),
        }
        json.dump(results["bert"], open(os.path.join(EVALUATION_DIR, "bert_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if results:
        compare_models(results, test_loader)

    return results


if __name__ == "__main__":
    main()
