"""
Attention-LSTM 情感分类模型
Architecture: Embedding(pretrained) → BiLSTM(2-layer) → Self-Attention → FC → 2-class

Attention 机制 (加性 / Bahdanau-style):
    u_t = tanh(W * h_t + b)
    α_t = softmax(u_t^T · u_w)
    context = Σ α_t * h_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    加性 Self-Attention 层。

    对所有时间步的隐状态做加权求和, 得到一个固定长度的上下文向量。

    Args:
        hidden_dim: 输入隐状态维度
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_w = nn.Parameter(torch.randn(hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.u_w, mean=0, std=0.1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, seq_len, hidden_dim) BiLSTM 所有时间步的输出
            mask: (batch, seq_len) 布尔 mask, True 表示有效位置

        Returns:
            context: (batch, hidden_dim) 加权上下文向量
            alpha: (batch, seq_len) 注意力权重
        """
        # u_t = tanh(W * h_t)
        u = torch.tanh(self.W(h))  # (batch, seq_len, hidden_dim)

        # score_t = u_t · u_w  (未归一化的注意力分数)
        scores = torch.matmul(u, self.u_w)  # (batch, seq_len)

        # Mask: 将 padding 位置的分数置为极小值
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        # α = softmax(scores)
        alpha = F.softmax(scores, dim=1)  # (batch, seq_len)

        # context = Σ α_t * h_t
        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (batch, hidden_dim)

        return context, alpha


class SentimentAttentionLSTM(nn.Module):
    """
    BiLSTM + Self-Attention 文本分类器。

    Args:
        vocab_size: 词表大小
        embed_dim: 词向量维度
        hidden_dim: 隐层维度 (每个方向)
        num_layers: LSTM 层数
        num_classes: 分类类别数
        dropout: dropout 比率
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
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2  # 固定双向
        self.lstm_output_dim = hidden_dim * self.num_directions

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Self-Attention (作用于 LSTM 输出维度 = hidden_dim * 2)
        self.attention = AdditiveAttention(self.lstm_output_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (batch, seq_len) 词索引序列
            return_attention: 是否返回注意力权重

        Returns:
            若 return_attention=False:
                logits: (batch, num_classes)
            若 return_attention=True:
                (logits, attention_weights)
        """
        # Mask: 标记非 padding 位置
        mask = (x != 0)  # (batch, seq_len), True = 有效词

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)

        # Self-Attention
        context, alpha = self.attention(lstm_out, mask)  # context: (batch, hidden_dim*2)

        # 分类
        out = self.dropout(context)
        logits = self.fc(out)  # (batch, num_classes)

        if return_attention:
            return logits, alpha
        return logits
