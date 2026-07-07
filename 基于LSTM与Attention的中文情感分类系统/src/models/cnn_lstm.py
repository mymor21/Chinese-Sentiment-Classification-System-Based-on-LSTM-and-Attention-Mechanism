"""
CNN-BiLSTM 情感分类模型

核心思路：在 Embedding 和 BiLSTM 之间插入多尺度 1D 卷积，
让模型在进入序列建模之前先学会捕获局部 n-gram 模式。

kernel_size=2 → "不热情" "很低" "好吃" 等二元组
kernel_size=3 → "性价比很低" "不怎么样" "很不错" 等三元组
kernel_size=4 → "一问三不知" "物超所值" 等四元组

卷积权重通过训练自动发现哪些局部组合改变情感极性，
无需预设否定词列表。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentCNNBiLSTM(nn.Module):
    """
    Embedding → Multi-Scale Conv1D → BiLSTM → Attention → FC

    Args:
        vocab_size: 词表大小
        embed_dim: 词向量维度
        hidden_dim: BiLSTM 隐层维度（每个方向）
        num_layers: LSTM 层数
        num_classes: 分类数
        dropout: dropout 比率
        kernel_sizes: 卷积核大小列表
        num_filters: 每个 kernel size 的卷积核数
        pretrained_embeddings: 预训练词向量
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        kernel_sizes: list = None,
        num_filters: int = 100,
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]

        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.num_directions = 2

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # Multi-scale 1D Convolutions
        # 每个 kernel size 有 num_filters 个卷积核
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k - 1,   # 保持序列长度不变（左 padding）
            )
            for k in kernel_sizes
        ])

        # CNN 输出维度：num_filters * len(kernel_sizes)
        cnn_output_dim = num_filters * len(kernel_sizes)

        # 将 CNN 多尺度特征投影回统一维度，送入 BiLSTM
        self.cnn_proj = nn.Linear(cnn_output_dim, embed_dim)

        # BiLSTM
        lstm_input_dim = embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_dim = hidden_dim * self.num_directions

        # Additive Self-Attention（复用 attention_lstm 的 Attention）
        from .attention_lstm import AdditiveAttention
        self.attention = AdditiveAttention(lstm_output_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (batch, seq_len) 词索引序列
            return_attention: 是否返回注意力权重
        """
        mask = (x != 0)
        batch_size, seq_len = x.shape

        # 1. Embedding: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # 2. Multi-Scale CNN
        # 转为 Conv1d 格式: (batch, embed_dim, seq_len)
        embedded_t = embedded.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            # conv_out: (batch, num_filters, seq_len + k - 1)
            conv_out = conv(embedded_t)
            # 截断右端多余的 padding（取前 seq_len 个位置）
            conv_out = conv_out[:, :, :seq_len]
            conv_out = F.relu(conv_out)
            conv_outputs.append(conv_out)

        # 拼接所有 kernel size 的输出: (batch, num_filters * n_kernels, seq_len)
        cnn_cat = torch.cat(conv_outputs, dim=1)

        # 转回 (batch, seq_len, cnn_output_dim)
        cnn_out = cnn_cat.transpose(1, 2)

        # 投影到 embed_dim: (batch, seq_len, embed_dim)
        cnn_proj = self.cnn_proj(cnn_out)
        cnn_proj = F.relu(cnn_proj)
        cnn_proj = self.dropout(cnn_proj)

        # 3. BiLSTM: (batch, seq_len, lstm_output_dim)
        lstm_out, _ = self.lstm(cnn_proj)

        # 4. Self-Attention: context (batch, lstm_output_dim)
        context, alpha = self.attention(lstm_out, mask)

        # 5. 分类
        out = self.dropout(context)
        logits = self.fc(out)

        if return_attention:
            return logits, alpha
        return logits
