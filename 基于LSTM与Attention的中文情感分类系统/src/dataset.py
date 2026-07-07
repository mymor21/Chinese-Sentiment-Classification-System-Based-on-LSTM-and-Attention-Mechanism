"""
PyTorch Dataset + DataLoader 封装
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_DIR, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, DEVICE,
)


class SentimentDataset(Dataset):
    """中文情感分类 Dataset。"""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_processed_data() -> dict:
    """加载预处理后的数据。"""
    pkl_path = os.path.join(PROCESSED_DIR, "processed_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"{pkl_path} 不存在！请先运行 preprocess.py。"
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def create_dataloaders(
    data: dict = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader。

    Args:
        data: 预处理后的数据字典。若为 None 则自动加载。
        batch_size: batch 大小

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if data is None:
        data = load_processed_data()

    sequences = data["sequences"]
    labels = data["labels"]

    # 分层划分
    # 先分出 train+val 和 test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences, labels,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # 再从 train+val 中分出 train 和 val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        stratify=y_train_val,
        random_state=RANDOM_SEED,
    )

    print(f"\n[dataset] 数据划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")

    train_ds = SentimentDataset(X_train, y_train)
    val_ds = SentimentDataset(X_val, y_val)
    test_ds = SentimentDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_test_data() -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """
    获取测试集原始数据（用于可视化: t-SNE、注意力、混淆矩阵）。

    Returns:
        (sequences, labels, tokenized_texts)
    """
    data = load_processed_data()
    sequences = data["sequences"]
    labels = data["labels"]
    tokenized = data["tokenized"]

    _, X_test, _, y_test, _, tokenized_test = train_test_split(
        sequences, labels, tokenized,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    return X_test, y_test, tokenized_test


if __name__ == "__main__":
    # 快速测试
    train_loader, val_loader, test_loader = create_dataloaders()
    for seqs, lbls in train_loader:
        print(f"Batch: seqs={seqs.shape}, labels={lbls.shape}")
        print(f"Label distribution: {torch.bincount(lbls)}")
        break
