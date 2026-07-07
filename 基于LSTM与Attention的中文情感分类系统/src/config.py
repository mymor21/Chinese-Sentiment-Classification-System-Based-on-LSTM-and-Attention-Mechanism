"""
全局配置与超参数
"""

import os
import torch

# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRAINING_DIR = os.path.join(RESULTS_DIR, "training")
EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# 确保所有目录存在
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR,
          TRAINING_DIR, EVALUATION_DIR, ANALYSIS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 设备配置 ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 随机种子 ──────────────────────────────────────────────
RANDOM_SEED = 42

# ── 数据集配置 ────────────────────────────────────────────
# HuggingFace / 下载源
DATASET_SOURCES = {
    "chnsenticorp": "seamew/ChnSentiCorp",
    "waimai": "waimai_10k",  # 从 GitHub 下载
}

# 标签映射
LABEL_MAP = {
    0: "负面",
    1: "正面",
}
NUM_CLASSES = 2

# 训练/验证/测试划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── 文本预处理配置 ────────────────────────────────────────
MAX_SEQ_LEN = 128          # 最大序列长度（padding/truncation）
MIN_FREQ = 3               # 词表最小词频
VOCAB_SIZE = 30000         # 最大词表大小（null 则不限制）

# ── Word2Vec 配置 ─────────────────────────────────────────
WV_VECTOR_SIZE = 300
WV_WINDOW = 5
WV_MIN_COUNT = 3
WV_SG = 1                  # 1 = Skip-gram
WV_EPOCHS = 30
WV_WORKERS = 4
WV_SEED = RANDOM_SEED

# ── 模型通用配置 ──────────────────────────────────────────
EMBED_DIM = 300            # 词向量维度（须与 WV_VECTOR_SIZE 一致）
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True

# ── 训练配置 ──────────────────────────────────────────────
BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LR_PATIENCE = 3            # ReduceLROnPlateau 的 patience
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 7    # val_loss 不降时终止
GRAD_CLIP = 5.0

# ── 中文字体候选（matplotlib） ────────────────────────────
FONT_CANDIDATES = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
]

# ── 可视化配色 ────────────────────────────────────────────
COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "primary": "#6366f1",
    "rnn": "#f59e0b",
    "lstm": "#3b82f6",
    "attention_lstm": "#8b5cf6",
    "cnn_lstm": "#06b6d4",
    "bert": "#ec4899",
}

# ── 停用词（基础列表，会与载入的外部词表合并） ────────────
CHINESE_STOPWORDS = {
    # 虚词
    "的", "了", "在", "是", "我", "有", "和", "就", "人", "都", "一",
    "一个", "上", "也", "到", "说", "要", "去", "你", "会", "着",
    "看", "自己", "这", "他", "她", "它", "们", "那", "些",
    "所", "为", "所以", "因为", "但是", "然而", "而且", "还是", "只是",
    "可以", "这个", "那个", "什么", "怎么", "哪里", "哪",
    "的", "地", "得", "之", "以", "及", "与", "或", "但", "而", "且",
    "虽", "然", "如", "若", "使", "向", "从", "对", "把", "被", "将",
    "能", "会", "可", "已", "还", "又", "再", "才", "刚", "正", "在",
    "有", "常",
    # 语气词
    "吗", "啊", "吧", "呢", "哦", "嗯", "哈", "呀", "哇", "嘛", "呗", "啦", "哟",
    # 注意：否定词(不/没/没有/非)和程度词(很/太/最/极/更/多/少)故意保留在词表中，
    # 因为它们携带关键情感信息，过滤掉会导致语义反转。
}
