"""
双向 LSTM 情感分类模型
Architecture: Embedding(pretrained) → BiLSTM(2-layer) → Dropout → FC → 2-class
"""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """
    双向 LSTM 文本分类器。

    Args:
        vocab_size: 词表大小
        embed_dim: 词向量维度
        hidden_dim: 隐层维度
        num_layers: LSTM 层数
        num_classes: 分类类别数
        dropout: dropout 比率
        bidirectional: 是否双向
        pretrained_embeddings: 预训练词向量权重 (可选)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        bidirectional: bool = True,
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) 词索引序列

        Returns:
            logits: (batch, num_classes)
        """
        embedded = self.embedding(x)

        # LSTM: output (batch, seq_len, hidden_dim * num_directions)
        #        h_n   (num_layers * num_directions, batch, hidden_dim)
        #        c_n   (num_layers * num_directions, batch, hidden_dim)
        _, (h_n, _) = self.lstm(embedded)

        # 取最后一层 hidden state
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_last = h_n[-1, :, :]

        out = self.dropout(h_last)
        logits = self.fc(out)
        return logits
