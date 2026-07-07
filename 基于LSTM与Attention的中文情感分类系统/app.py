"""
Flask 交互式 Demo — 中文情感分类系统
"""

import os
import sys
import json
import pickle
import random
import threading

import numpy as np
import torch
import jieba
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    PROCESSED_DIR, MODEL_DIR, DEVICE, RANDOM_SEED,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, EMBED_DIM, NUM_CLASSES, MAX_SEQ_LEN,
)

app = Flask(__name__)
random.seed(RANDOM_SEED)

# ── 全局变量 ──────────────────────────────────────────────

_models = {}
_bert_tokenizer = None
_word2idx = {}
_idx2word = {}
_vocab_size = 0
_max_seq_len = MAX_SEQ_LEN
_test_samples = []
_loaded = False
_bert_status = "waiting"  # waiting | loading | ready | failed

LABEL_NAMES = {0: "负面", 1: "正面"}
LABEL_COLORS = {0: "#ef4444", 1: "#22c55e", -1: "#94a3b8"}


def _load_bert_background():
    """后台线程加载 BERT。"""
    global _bert_status, _bert_tokenizer
    _bert_status = "loading"
    print("[app] 后台加载 BERT...")
    try:
        from models.bert import SentimentBERT, get_bert_tokenizer
        model = SentimentBERT(num_classes=NUM_CLASSES, dropout=0.2)
        ckpt = torch.load(os.path.join(MODEL_DIR, "bert.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()
        _models["bert"] = model
        _bert_tokenizer = get_bert_tokenizer()
        _bert_status = "ready"
        print("[app] BERT 加载完成")
    except Exception as e:
        _bert_status = "failed"
        print(f"[app] BERT 加载失败: {e}")


def load_all():
    """加载词表 + 4 个轻量模型，BERT 后台异步加载。"""
    global _models, _word2idx, _idx2word, _vocab_size, _max_seq_len, _test_samples, _loaded, _bert_status
    if _loaded:
        return

    print("[app] 加载预处理数据...")

    # 测试样本缓存
    samples_path = os.path.join(PROCESSED_DIR, "test_samples.json")
    if os.path.exists(samples_path):
        with open(samples_path, "r", encoding="utf-8") as f:
            _test_samples = json.load(f)[:200]
        print(f"[app] 从缓存加载 {len(_test_samples)} 条测试样本")

    # 词表
    pkl_path = os.path.join(PROCESSED_DIR, "processed_data.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        _word2idx = data["word2idx"]
        _idx2word = data["idx2word"]
        _vocab_size = data["vocab_size"]
        _max_seq_len = data["max_seq_len"]

    if not _word2idx:
        vocab_path = os.path.join(PROCESSED_DIR, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_info = json.load(f)
            _word2idx = vocab_info.get("word2idx", {})
            _vocab_size = vocab_info.get("vocab_size", 10000)
            _max_seq_len = vocab_info.get("max_seq_len", MAX_SEQ_LEN)
            _idx2word = {v: k for k, v in _word2idx.items()}

    print(f"[app] 词表大小: {_vocab_size}, max_seq_len: {_max_seq_len}")

    # 加载 4 个轻量模型
    from models.rnn import SentimentRNN
    from models.lstm import SentimentLSTM
    from models.attention_lstm import SentimentAttentionLSTM
    from models.cnn_lstm import SentimentCNNBiLSTM

    for name, ModelClass in [
        ("rnn", SentimentRNN), ("lstm", SentimentLSTM),
        ("attention_lstm", SentimentAttentionLSTM), ("cnn_lstm", SentimentCNNBiLSTM),
    ]:
        model_path = os.path.join(MODEL_DIR, f"{name}.pt")
        if not os.path.exists(model_path):
            continue
        model = ModelClass(
            vocab_size=_vocab_size or 30000, embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES, dropout=DROPOUT,
        )
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE).eval()
        _models[name] = model
        print(f"[app] 已加载: {name}")

    _loaded = True
    print(f"[app] 就绪: {len(_models)} 个模型, BERT 后台加载中...")

    # 后台线程加载 BERT
    threading.Thread(target=_load_bert_background, daemon=True).start()


def tokenize_and_encode(text: str) -> tuple[np.ndarray, list[str]]:
    """分词并转为模型输入。"""
    import re
    from config import CHINESE_STOPWORDS
    stopwords = set(CHINESE_STOPWORDS)
    extra = set("，。！？、；：""''（）【】《》…—·～ \t\n\r　​" +
                "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" +
                "0123456789０１２３４５６７８９" +
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    stopwords.update(extra)

    words = jieba.lcut(text)
    tokens = []
    for w in words:
        w = w.strip()
        if not w or w in stopwords or w.isdigit():
            continue
        if not re.search(r'[一-鿿]', w):
            continue
        tokens.append(w)

    indices = [_word2idx.get(t, 1) for t in tokens]
    if len(indices) > _max_seq_len:
        indices = indices[-_max_seq_len:]
    else:
        indices = [0] * (_max_seq_len - len(indices)) + indices
    return np.array([indices], dtype=np.int64), tokens


@torch.no_grad()
def predict(text: str) -> dict:
    """对输入文本做情感预测。"""
    load_all()

    if not _models:
        return {"error": "没有可用的模型，请先训练模型"}

    seq, tokens = tokenize_and_encode(text)
    seq_tensor = torch.LongTensor(seq).to(DEVICE)
    results = {}

    for name, model in _models.items():
        if name == "bert" and _bert_tokenizer:
            enc = _bert_tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        elif name == "attention_lstm":
            logits, alpha = model(seq_tensor, return_attention=True)
            n_valid = len(tokens) if len(tokens) <= _max_seq_len else _max_seq_len
            attn = alpha.squeeze(0).cpu().numpy()
            attn_weights = attn[-n_valid:].tolist() if n_valid <= len(attn) else attn[:n_valid].tolist()
            results[name] = {"attention": attn_weights}
        else:
            logits = model(seq_tensor)

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(torch.argmax(logits, dim=1).item())
        conf = float(probs[pred])

        if conf >= 0.85:
            conf_level = "high"
        elif conf >= 0.65:
            conf_level = "medium"
        else:
            conf_level = "low"
            pred = -1

        results[name] = {
            **results.get(name, {}),
            "prediction": pred,
            "label": LABEL_NAMES.get(pred, "不确定"),
            "color": LABEL_COLORS.get(pred, "#94a3b8"),
            "confidence": conf,
            "confidence_level": conf_level,
            "probabilities": {"负面": float(probs[0]), "正面": float(probs[1])},
        }

    return {
        "text": text,
        "tokens": tokens,
        "results": results,
        "bert_status": _bert_status,
    }


# ── 路由 ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "请提供 text 字段"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "文本不能为空"}), 400
    if len(text) < 3:
        return jsonify({"error": "文本太短，至少需要 3 个字符"}), 400
    return jsonify(predict(text))


@app.route("/api/sample")
def api_sample():
    load_all()
    if not _test_samples:
        return jsonify({"error": "没有可用的测试样本"}), 404
    sample = random.choice(_test_samples)
    return jsonify({
        "text": sample["text"],
        "label": int(sample["label"]),
        "label_name": LABEL_NAMES.get(int(sample["label"]), "未知"),
    })


@app.route("/api/status")
def api_status():
    """返回模型加载状态，前端轮询 BERT 是否就绪。"""
    load_all()
    return jsonify({
        "models_ready": list(_models.keys()),
        "bert_status": _bert_status,
    })


@app.route("/api/gallery")
def api_gallery():
    images = []
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    for root, _, files in os.walk(results_dir):
        for f in sorted(files):
            if f.endswith(".png"):
                rel_path = os.path.relpath(os.path.join(root, f), os.path.dirname(__file__))
                category = os.path.basename(os.path.dirname(os.path.join(root, f)))
                images.append({
                    "name": f.replace(".png", "").replace("_", " ").title(),
                    "path": "/static/results/" + "/".join(rel_path.split(os.sep)[1:]),
                    "category": category,
                })
    return jsonify(images)


@app.route("/results-img/<path:filename>")
def serve_results_img(filename):
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    for root, _, files in os.walk(results_dir):
        if filename in files:
            return send_from_directory(root, filename)
    return jsonify({"error": "图片不存在"}), 404


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  中文情感分析系统 Demo")
    print("  打开浏览器访问: http://localhost:5000")
    print("  (BERT 将在后台加载，约 50 秒后就绪)")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
