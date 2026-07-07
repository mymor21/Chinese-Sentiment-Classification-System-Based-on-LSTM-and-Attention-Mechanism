"""
Word2Vec 词向量训练 (gensim Skip-gram)
"""

import os
import pickle

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from config import (
    PROCESSED_DIR, MODEL_DIR,
    WV_VECTOR_SIZE, WV_WINDOW, WV_MIN_COUNT, WV_SG, WV_EPOCHS,
    WV_WORKERS, WV_SEED, EMBED_DIM,
)


class LossLogger(CallbackAny2Vec):
    """gensim 训练进度回调。"""

    def __init__(self):
        self.epoch = 0
        self.loss = 0.0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        current_loss = loss - self.loss
        self.loss = loss
        print(f"  Epoch {self.epoch + 1}/{WV_EPOCHS}, loss: {current_loss:.4f}")
        self.epoch += 1


def train_word2vec(sentences: list[list[str]]) -> Word2Vec:
    """
    训练 Word2Vec Skip-gram 模型。

    Args:
        sentences: 分词后的句子列表（list of list of str）

    Returns:
        训练好的 gensim Word2Vec 模型
    """
    print(f"[word2vec] 开始训练 Word2Vec (Skip-gram, {WV_VECTOR_SIZE}d)...")
    print(f"  语料: {len(sentences)} 句, vector_size={WV_VECTOR_SIZE}, "
          f"window={WV_WINDOW}, epochs={WV_EPOCHS}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=WV_VECTOR_SIZE,
        window=WV_WINDOW,
        min_count=WV_MIN_COUNT,
        sg=WV_SG,
        workers=WV_WORKERS,
        seed=WV_SEED,
        epochs=WV_EPOCHS,
        compute_loss=True,
        callbacks=[LossLogger()],
    )

    print(f"[word2vec] 训练完成，词表大小: {len(model.wv)}")
    return model


def save_model(model: Word2Vec):
    """保存模型和词向量。"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "word2vec.model")
    model.save(model_path)

    vectors_path = os.path.join(MODEL_DIR, "word2vec_vectors.txt")
    model.wv.save_word2vec_format(vectors_path)

    print(f"[word2vec] 模型已保存至 {model_path}")
    print(f"[word2vec] 词向量已保存至 {vectors_path}")


def build_embedding_matrix(
    model: Word2Vec, word2idx: dict, embed_dim: int = EMBED_DIM
) -> np.ndarray:
    """
    将 gensim 词向量转为 PyTorch Embedding 权重矩阵。

    对于 word2idx 中每个词:
    - 如果在 Word2Vec 中存在 → 使用预训练向量
    - 否则 → 随机初始化 (uniform(-0.25, 0.25))

    Args:
        model: gensim Word2Vec 模型
        word2idx: 预处理阶段构建的词表
        embed_dim: 词向量维度

    Returns:
        (vocab_size, embed_dim) 的 numpy 权重矩阵
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim)).astype(np.float32)

    # <PAD> 置零
    embedding_matrix[0] = np.zeros(embed_dim, dtype=np.float32)

    hit, miss = 0, 0
    for word, idx in word2idx.items():
        if word in ("<PAD>", "<UNK>"):
            continue
        if word in model.wv:
            embedding_matrix[idx] = model.wv[word]
            hit += 1
        else:
            miss += 1

    print(f"[word2vec] 嵌入矩阵: 命中 {hit}, 未命中 {miss} "
          f"(覆盖率 {hit/(hit+miss)*100:.1f}%)")
    return embedding_matrix


def main():
    """Word2Vec 训练主流程。"""
    embedding_path = os.path.join(PROCESSED_DIR, "embedding_matrix.npy")

    if os.path.exists(embedding_path):
        print(f"[word2vec] {embedding_path} 已存在，跳过训练")
        return

    # 加载预处理数据
    with open(os.path.join(PROCESSED_DIR, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    # 训练 Word2Vec
    model = train_word2vec(data["tokenized"])

    # 保存模型
    save_model(model)

    # 构建嵌入矩阵
    embedding_matrix = build_embedding_matrix(model, data["word2idx"])

    # 保存嵌入矩阵
    np.save(embedding_path, embedding_matrix)
    print(f"[word2vec] 嵌入矩阵已保存至 {embedding_path} ({embedding_matrix.shape})")

    # 更新 data
    data["embedding_matrix"] = embedding_matrix
    with open(os.path.join(PROCESSED_DIR, "processed_data.pkl"), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
