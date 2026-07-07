"""
BERT 情感分类模型（预训练对照组）

使用 bert-base-chinese，从 HF 镜像下载。
作为预训练基线：不参与训练曲线对比，仅在最终评估中展示性能上限。
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class SentimentBERT(nn.Module):
    """
    BERT-base-chinese + 分类头。

    与从零训练的模型不在同一比较维度——BERT 引入外部预训练知识，
    代表当前数据规模下微调范式的性能天花板。
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.2,
        bert_model_name: str = "bert-base-chinese",
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size  # 768
        self.fc = nn.Linear(hidden_size, num_classes)

        # 保存以便 checkpoint 重建
        self.num_classes = num_classes
        self.bert_model_name = bert_model_name

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output  # (batch, 768), CLS token
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits


def get_bert_tokenizer(model_name: str = "bert-base-chinese"):
    """获取 BERT tokenizer（供 evaluate.py 测试集推理使用）。"""
    return BertTokenizer.from_pretrained(model_name)


def get_bert_max_length(model_name: str = "bert-base-chinese"):
    """BERT 最大序列长度（bert-base-chinese = 512）。"""
    return 512
