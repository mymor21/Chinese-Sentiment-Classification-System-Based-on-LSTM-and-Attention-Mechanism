"""
文本预处理：jieba 分词、去停用词、构建词表、文本转向量
"""

import os
import json
import pickle
import re
from collections import Counter

import jieba
import numpy as np
from tqdm import tqdm

from config import (
    RAW_DIR, PROCESSED_DIR, MAX_SEQ_LEN, MIN_FREQ, VOCAB_SIZE,
    CHINESE_STOPWORDS, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)

# ── 停用词加载 ────────────────────────────────────────────

def load_stopwords() -> set:
    """加载中文停用词表。"""
    stopwords = set(CHINESE_STOPWORDS)

    # 尝试加载 jieba 内置停用词表
    jieba_stopwords_paths = [
        os.path.join(os.path.dirname(jieba.__file__), "analyse", "stop_words.txt"),
        "stopwords.txt",
    ]

    for path in jieba_stopwords_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w:
                        stopwords.add(w)
            print(f"[preprocess] 加载停用词表: {path} ({len(stopwords)} 个)")
            break
    else:
        print(f"[preprocess] 使用内置停用词表 ({len(stopwords)} 个)")

    # 追加标点符号和特殊字符
    extra = set("，。！？、；：""''（）【】《》…—·～ \t\n\r　​" +
                "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" +
                "0123456789０１２３４５６７８９" +
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    stopwords.update(extra)

    return stopwords


# ── 文本清洗 ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """清洗文本：去除 URL、@ 提及、多余空白。"""
    # 去 URL
    text = re.sub(r'https?://\S+', '', text)
    # 去 @ 提及
    text = re.sub(r'@\S+', '', text)
    # 去 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 合并空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str, stopwords: set) -> list[str]:
    """
    jieba 分词 + 去停用词 + 过滤。
    停用词表负责过滤虚词，不再按字数一刀切，保留"好""坏""差"等单字情感词。
    """
    text = clean_text(text)
    words = jieba.lcut(text)
    result = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if w in stopwords:
            continue
        # 过滤纯数字
        if w.isdigit():
            continue
        # 至少包含一个中文字符
        if not re.search(r'[一-鿿]', w):
            continue
        result.append(w)
    return result


# ── 构建词表 ──────────────────────────────────────────────

def build_vocab(tokenized_texts: list[list[str]]) -> tuple[dict, dict, int]:
    """
    构建 word2idx 和 idx2word 映射。
    idx=0: <PAD>, idx=1: <UNK>
    """
    word_counts = Counter()
    for tokens in tokenized_texts:
        word_counts.update(tokens)

    # 过滤低频词
    vocab_words = [w for w, c in word_counts.items() if c >= MIN_FREQ]

    # 限制词表大小
    if VOCAB_SIZE and len(vocab_words) > VOCAB_SIZE - 2:
        vocab_words = [w for w, _ in word_counts.most_common(VOCAB_SIZE - 2)]

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(vocab_words, start=2):
        word2idx[w] = i

    idx2word = {v: k for k, v in word2idx.items()}

    print(f"[preprocess] 词表大小: {len(word2idx)}")
    print(f"[preprocess] 高频词示例: {vocab_words[:30]}")

    return word2idx, idx2word, len(word2idx)


# ── 文本转向量 ────────────────────────────────────────────

def texts_to_sequences(
    tokenized_texts: list[list[str]],
    word2idx: dict,
    max_len: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """
    将分词后的文本转为固定长度的 index 序列。
    Padding: 前补 0 (<PAD>)
    Truncation: 保留后 max_len 个词
    """
    sequences = np.zeros((len(tokenized_texts), max_len), dtype=np.int64)
    for i, tokens in enumerate(tokenized_texts):
        indices = [word2idx.get(t, 1) for t in tokens]  # 未知词 → <UNK>
        if len(indices) > max_len:
            indices = indices[-max_len:]  # 截断：保留末尾
        # 前向 padding
        seq_len = len(indices)
        sequences[i, max_len - seq_len:] = indices
    return sequences


# ── 主流程 ────────────────────────────────────────────────

def load_raw_data() -> tuple[list[str], list[int]]:
    """加载原始 TSV 数据。"""
    tsv_path = os.path.join(RAW_DIR, "sentiment_dataset.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"数据文件 {tsv_path} 不存在！请先运行 download_data.py 下载数据集。"
        )

    texts, labels = [], []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过表头
        for line in tqdm(f, desc="加载数据"):
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label_str, text = parts
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                if text.strip():
                    texts.append(text.strip())
                    labels.append(label)

    print(f"[preprocess] 加载 {len(texts)} 条数据")
    return texts, labels


def save_processed(data: dict):
    """保存预处理结果。"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 保存为 pickle（快速加载）
    with open(os.path.join(PROCESSED_DIR, "processed_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    # 保存词表为 JSON（可读）
    vocab_info = {
        "word2idx": data["word2idx"],
        "vocab_size": data["vocab_size"],
        "max_seq_len": data["max_seq_len"],
    }
    with open(os.path.join(PROCESSED_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)

    # 保存统计信息
    stats = {
        "num_samples": len(data["labels"]),
        "vocab_size": data["vocab_size"],
        "max_seq_len": data["max_seq_len"],
        "label_distribution": {
            "负面": int(sum(1 for l in data["labels"] if l == 0)),
            "正面": int(sum(1 for l in data["labels"] if l == 1)),
        },
        "avg_text_length": float(np.mean([len(t) for t in data["tokenized"]])),
        "median_text_length": float(np.median([len(t) for t in data["tokenized"]])),
    }
    with open(os.path.join(PROCESSED_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[preprocess] 预处理数据已保存至 {PROCESSED_DIR}/")
    print(f"[preprocess] 统计: {json.dumps(stats, ensure_ascii=False)}")


def main():
    """预处理主流程。"""
    processed_pkl = os.path.join(PROCESSED_DIR, "processed_data.pkl")
    if os.path.exists(processed_pkl):
        print(f"[preprocess] {processed_pkl} 已存在，跳过预处理")
        return

    print("[preprocess] ===== 开始文本预处理 =====\n")

    # 1. 加载原始数据
    texts, labels = load_raw_data()

    # 2. 加载停用词
    stopwords = load_stopwords()

    # 3. jieba 分词
    print("\n[preprocess] jieba 分词中...")
    tokenized = []
    valid_labels = []
    valid_texts = []   # 保留原始文本
    for text, label in tqdm(zip(texts, labels), total=len(texts), desc="分词"):
        tokens = tokenize(text, stopwords)
        if len(tokens) >= 1:  # 仅排除空文本
            tokenized.append(tokens)
            valid_labels.append(label)
            valid_texts.append(text)

    print(f"[preprocess] 有效样本: {len(tokenized)} (过滤掉 {len(texts) - len(tokenized)} 条)")

    # 4. 分析文本长度，确定截断长度
    lengths = [len(t) for t in tokenized]
    max_seq_len = int(np.percentile(lengths, 95))
    max_seq_len = min(max_seq_len, MAX_SEQ_LEN)  # 不超过上限
    print(f"[preprocess] 95% 文本长度: {np.percentile(lengths, 95):.0f}, "
          f"实际 max_seq_len: {max_seq_len}")

    # 5. 构建词表
    word2idx, idx2word, vocab_size = build_vocab(tokenized)

    # 6. 文本转向量
    sequences = texts_to_sequences(tokenized, word2idx, max_seq_len)

    # 7. 打包保存
    data = {
        "sequences": sequences,
        "labels": np.array(valid_labels, dtype=np.int64),
        "tokenized": tokenized,
        "texts": valid_texts,       # 原始文本
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
    }
    save_processed(data)

    print(f"\n[preprocess] ===== 预处理完成 =====")
    print(f"  sequences 形状: {sequences.shape}")
    print(f"  标签分布: 负面={sum(1 for l in valid_labels if l==0)}, "
          f"正面={sum(1 for l in valid_labels if l==1)}")

    return data


if __name__ == "__main__":
    main()
