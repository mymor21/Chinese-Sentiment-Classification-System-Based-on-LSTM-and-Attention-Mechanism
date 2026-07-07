"""
可视化模块 — 8 类静态 matplotlib 图表

1. 训练损失曲线 (三模型对比)
2. 准确率 + F1 曲线
3. 混淆矩阵 (每模型 3×3)
4. 模型对比柱状图
5. 情感分布图
6. 词云图
7. t-SNE 嵌入可视化
8. 注意力权重热力图
"""

import os
import json
import pickle
import re
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from font")

from config import (
    TRAINING_DIR, EVALUATION_DIR, ANALYSIS_DIR, PROCESSED_DIR, MODEL_DIR,
    COLORS, FONT_CANDIDATES, NUM_CLASSES, DEVICE,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, EMBED_DIM,
)

# ── 中文字体设置 ──────────────────────────────────────────

def setup_chinese_font():
    """设置 matplotlib 中文字体。"""
    # 先清除字体缓存，强制重建
    import glob
    cache_dir = matplotlib.get_cachedir()
    for f in glob.glob(os.path.join(cache_dir, "fontlist*.json")):
        try:
            os.remove(f)
        except Exception:
            pass
    fm._load_fontmanager(try_read_cache=False)

    font_path = None
    for fname in FONT_CANDIDATES:
        for f in fm.fontManager.ttflist:
            if fname.lower() in f.name.lower():
                font_path = f.fname
                break
        if font_path:
            break

    if font_path:
        fm.fontManager.addfont(font_path)
        family = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [family] + plt.rcParams["font.sans-serif"]
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    print(f"[visualize] 字体: sans-serif[0]={plt.rcParams['font.sans-serif'][0]}")


sns.set_style("whitegrid")
setup_chinese_font()


# ── 加载数据工具 ──────────────────────────────────────────

def load_history(model_name: str) -> dict:
    """加载模型训练历史。"""
    path = os.path.join(MODEL_DIR, f"{model_name}_history.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_eval_report(model_name: str) -> dict:
    """加载模型评估报告。"""
    path = os.path.join(EVALUATION_DIR, f"{model_name}_report.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── 图表 1: 训练损失曲线 ──────────────────────────────────

def plot_loss_curves():
    """三模型 train/val loss 对比图。"""
    print("[visualize] 绘制训练损失曲线...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = [("rnn", "RNN"), ("lstm", "LSTM"), ("attention_lstm", "Attention-LSTM"), ("cnn_lstm", "CNN-BiLSTM"), ("bert", "BERT")]
    color_map = {"rnn": COLORS["rnn"], "lstm": COLORS["lstm"], "attention_lstm": COLORS["attention_lstm"], "cnn_lstm": COLORS["cnn_lstm"], "bert": COLORS["bert"]}

    for key, label in model_names:
        hist = load_history(key)
        if not hist:
            continue
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], color=color_map[key], linestyle="--",
                     linewidth=1.2, alpha=0.7)
        axes[0].plot(epochs, hist["val_loss"], color=color_map[key], linewidth=2,
                     label=f"{label} (val)")

    axes[0].set_title("训练 & 验证损失对比", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9)

    # 右侧：只画 val_loss 清晰对比
    for key, label in model_names:
        hist = load_history(key)
        if not hist:
            continue
        epochs = range(1, len(hist["val_loss"]) + 1)
        axes[1].plot(epochs, hist["val_loss"], color=color_map[key], linewidth=2.5, label=label)

    axes[1].set_title("验证损失对比 (Val Loss)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(TRAINING_DIR, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 图表 2: 准确率 + F1 曲线 ──────────────────────────────

def plot_accuracy_f1_curves():
    """三模型 val_acc + val_f1 随 epoch 变化。"""
    print("[visualize] 绘制准确率/F1 曲线...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = [("rnn", "RNN"), ("lstm", "LSTM"), ("attention_lstm", "Attention-LSTM"), ("cnn_lstm", "CNN-BiLSTM"), ("bert", "BERT")]
    color_map = {"rnn": COLORS["rnn"], "lstm": COLORS["lstm"], "attention_lstm": COLORS["attention_lstm"], "cnn_lstm": COLORS["cnn_lstm"], "bert": COLORS["bert"]}

    for key, label in model_names:
        hist = load_history(key)
        if not hist:
            continue
        epochs = range(1, len(hist["val_acc"]) + 1)

        axes[0].plot(epochs, hist["val_acc"], color=color_map[key], linewidth=2.5, label=label)

        if "val_f1" in hist:
            axes[1].plot(epochs, hist["val_f1"], color=color_map[key], linewidth=2.5, label=label)

    axes[0].set_title("验证准确率 (Val Accuracy)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_title("验证 F1 分数 (Val F1)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 (Macro)")
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(TRAINING_DIR, "accuracy_f1_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 图表 3: 混淆矩阵 ──────────────────────────────────────

def plot_confusion_matrices():
    """每个模型一个混淆矩阵。"""
    print("[visualize] 绘制混淆矩阵...")

    model_names = [("rnn", "RNN"), ("lstm", "LSTM"), ("attention_lstm", "Attention-LSTM"), ("cnn_lstm", "CNN-BiLSTM"), ("bert", "BERT")]

    fig, axes = plt.subplots(1, 5, figsize=(30, 5.5))
    label_names = ["负面", "正面"]

    for ax, (key, label) in zip(axes, model_names):
        report = load_eval_report(key)
        cm = np.array(report.get("confusion_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names,
                    ax=ax, cbar=False, linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": 14, "fontweight": "bold"})
        ax.set_title(f"{label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")

    plt.tight_layout()
    path = os.path.join(EVALUATION_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 图表 4: 模型对比柱状图 ──────────────────────────────────

def plot_model_comparison():
    """三模型四指标并排柱状图。"""
    print("[visualize] 绘制模型对比柱状图...")

    comparison_path = os.path.join(EVALUATION_DIR, "model_comparison.json")
    if not os.path.exists(comparison_path):
        print("  跳过：model_comparison.json 不存在")
        return

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison = json.load(f)

    models = list(comparison.keys())
    model_labels = {"rnn": "RNN", "lstm": "LSTM", "attention_lstm": "Attention-\nLSTM", "cnn_lstm": "CNN-\nBiLSTM", "bert": "BERT"}
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(metrics))
    n_models = len(models)
    width = 0.20 if n_models > 3 else 0.25
    colors = [COLORS.get(m, "#999999") for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (model_name, c) in enumerate(zip(models, colors)):
        values = [comparison[model_name].get(m, 0) for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model_labels.get(model_name, model_name),
                      color=c, edgecolor="white", linewidth=0.8)
        # 标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("模型性能对比", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(EVALUATION_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 图表 5: 情感分布图 ────────────────────────────────────

def plot_sentiment_distribution():
    """绘制数据集中二分类的分布。"""
    print("[visualize] 绘制情感分布图...")

    with open(os.path.join(PROCESSED_DIR, "stats.json"), "r", encoding="utf-8") as f:
        stats = json.load(f)

    dist = stats["label_distribution"]
    labels = list(dist.keys())
    values = list(dist.values())
    colors_pie = [COLORS["negative"], COLORS["positive"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    wedges, texts, autotexts = axes[0].pie(
        values, labels=labels, autopct="%1.1f%%", colors=colors_pie,
        startangle=90, explode=(0.02, 0.02),
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)
    axes[0].set_title("情感类别分布 (饼图)", fontsize=14, fontweight="bold")

    # 柱状图
    bars = axes[1].bar(labels, values, color=colors_pie, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                     str(v), ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("样本数", fontsize=12)
    axes[1].set_title("情感类别分布 (柱状图)", fontsize=14, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "sentiment_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 图表 6: 词云图 ────────────────────────────────────────

def plot_wordclouds():
    """
    情感区分词云图。

    核心改进：不使用原始词频，而是计算每个词对特定情感类别的"区分度"。
    只有在该类中出现频率显著高于其他类的词才被保留。
    公式: distinctiveness(w, class) = freq_in_class / mean(freq_in_other_classes)
    阈值: distinctiveness > 2.0 且 freq >= 5
    """
    print("[visualize] 绘制词云图...")

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  跳过：wordcloud 未安装")
        return

    # 加载分词数据
    with open(os.path.join(PROCESSED_DIR, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    tokenized = data["tokenized"]
    labels = data["labels"]

    # 按类别分别计算词频（二分类）
    neg_counter = Counter()
    pos_counter = Counter()
    for tokens, label in zip(tokenized, labels):
        unique_tokens = set(tokens)
        if label == 0:
            neg_counter.update(unique_tokens)
        elif label == 1:
            pos_counter.update(unique_tokens)

    all_words = set(neg_counter.keys()) | set(pos_counter.keys())
    all_words = {w for w in all_words if neg_counter[w] + pos_counter[w] >= 5}

    print(f"  候选词数: {len(all_words)}")

    # 跨类共性过滤 + 正负对比
    neg_freq = {}
    pos_freq = {}

    for word in all_words:
        f_neg = neg_counter.get(word, 0)
        f_pos = pos_counter.get(word, 0)

        # 跨类共性过滤：min/max > 0.25 则是领域通用词
        if f_neg > 0 and f_pos > 0:
            ratio = min(f_neg, f_pos) / max(f_neg, f_pos)
            if ratio > 0.25:
                continue

        d_pos = f_pos / (f_neg + 1)
        d_neg = f_neg / (f_pos + 1)

        if d_neg > 1.5 and f_neg >= 3:
            neg_freq[word] = f_neg
        if d_pos > 1.5 and f_pos >= 3:
            pos_freq[word] = f_pos

    print(f"  负面区分词: {len(neg_freq)}, 正面区分词: {len(pos_freq)}")

    for label, freq_dict in [("负面", neg_freq), ("正面", pos_freq)]:
        top = sorted(freq_dict.items(), key=lambda x: -x[1])[:15]
        print(f"  [{label}] {' '.join(w for w,_ in top)}")

    # 中文字体路径
    font_path = None
    for f in fm.fontManager.ttflist:
        if any(name in f.name for name in ["SimHei", "Microsoft YaHei", "PingFang", "Noto Sans CJK"]):
            font_path = f.fname
            break

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def make_wordcloud(freq_dict, ax, title, bg_color, cmap_name):
        if not freq_dict:
            ax.text(0.5, 0.5, "无足够区分词", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#94a3b8")
            ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
            return
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib as mpl
        # 截取 colormap 后半段（50%-100%），跳过浅色区域
        full_cmap = mpl.colormaps[cmap_name]
        truncated = LinearSegmentedColormap.from_list(
            f"{cmap_name}_deep",
            [full_cmap(i) for i in [0.45, 0.65, 1.0]]
        )
        kwargs = dict(
            width=400, height=300, background_color=bg_color,
            max_words=80, collocations=False, random_state=42,
            colormap=truncated, min_font_size=10,
        )
        if font_path:
            kwargs["font_path"] = font_path
        wc = WordCloud(**kwargs).generate_from_frequencies(freq_dict)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)

    make_wordcloud(neg_freq, axes[0], "负面情感特征词", "#fef2f2", "Reds")
    make_wordcloud(pos_freq, axes[1], "正面情感特征词", "#f0fdf4", "Greens")

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "wordclouds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 图表 7: t-SNE 嵌入可视化 ───────────────────────────────

def plot_tsne_embeddings():
    """
    提取 Attention-LSTM 在测试集上的 FC 层输入表示，用 t-SNE 降维可视化。
    """
    print("[visualize] 绘制 t-SNE 嵌入可视化...")

    from sklearn.manifold import TSNE
    from dataset import create_dataloaders, load_processed_data
    from evaluate import load_trained_model

    data = load_processed_data()
    _, _, test_loader = create_dataloaders(data)
    embedding_matrix = data.get("embedding_matrix")
    vocab_size = data["vocab_size"]

    model_path = os.path.join(MODEL_DIR, "attention_lstm.pt")
    if not os.path.exists(model_path):
        print("  跳过：attention_lstm.pt 不存在")
        return

    model = load_trained_model("attention_lstm", vocab_size, embedding_matrix)
    model.eval()

    # 提取 FC 前的 context 向量 (LSTM + Attention 输出)
    @torch.no_grad()
    def extract_features(loader, max_samples=500):
        features_list, labels_list = [], []
        for sequences, labels in loader:
            sequences = sequences.to(DEVICE)
            x = sequences
            mask = (x != 0)
            embedded = model.embedding(x)
            lstm_out, _ = model.lstm(embedded)
            context, _ = model.attention(lstm_out, mask)
            features_list.append(context.cpu().numpy())
            labels_list.append(labels.numpy())
            if len(features_list) * sequences.size(0) >= max_samples:
                break
        features = np.concatenate(features_list, axis=0)[:max_samples]
        labels = np.concatenate(labels_list, axis=0)[:max_samples]
        return features, labels

    features, labels = extract_features(test_loader)

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, min(30, len(features) // 5)), metric="cosine")
    tsne_result = tsne.fit_transform(features)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    label_names = ["负面", "正面"]
    colors_tsne = [COLORS["negative"], COLORS["positive"]]

    for i, (name, c) in enumerate(zip(label_names, colors_tsne)):
        mask = labels == i
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   c=c, label=name, alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

    ax.set_title("t-SNE 嵌入可视化 (Attention-LSTM 特征空间)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "tsne_embeddings.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 图表 8: 注意力权重热力图 ───────────────────────────────

def plot_attention_heatmaps():
    """
    对几条测试样本可视化 Attention-LSTM 的注意力权重。
    """
    print("[visualize] 绘制注意力热力图...")

    from dataset import create_dataloaders, load_processed_data
    from evaluate import load_trained_model

    data = load_processed_data()
    _, _, test_loader = create_dataloaders(data)
    embedding_matrix = data.get("embedding_matrix")
    vocab_size = data["vocab_size"]

    model_path = os.path.join(MODEL_DIR, "attention_lstm.pt")
    if not os.path.exists(model_path):
        print("  跳过：attention_lstm.pt 不存在")
        return

    model = load_trained_model("attention_lstm", vocab_size, embedding_matrix)
    model.eval()

    idx2word = data["idx2word"]
    label_names = ["负面", "正面"]

    # 取 2 条样本（每类一条）
    samples_per_class = {}
    for sequences, labels in test_loader:
        for seq, lbl in zip(sequences, labels):
            lbl_int = lbl.item()
            if lbl_int not in samples_per_class and len(samples_per_class) < 2:
                samples_per_class[lbl_int] = seq
            if len(samples_per_class) == 2:
                break
        if len(samples_per_class) == 2:
            break

    if len(samples_per_class) < 2:
        print("  跳过：样本不足")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (label, seq) in zip(axes, sorted(samples_per_class.items())):
        seq_tensor = seq.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, alpha = model(seq_tensor, return_attention=True)

        pred = torch.argmax(logits, dim=1).item()
        weights = alpha.squeeze(0).cpu().numpy()

        # 获取非 padding 的词
        tokens = []
        for idx in seq.tolist():
            if idx == 0:  # <PAD>
                continue
            word = idx2word.get(idx, "<UNK>")
            tokens.append(word)

        # 取对应的权重（去掉 padding 部分）
        n_tokens = len(tokens)
        # 注意力权重对应的是整个序列的末尾部分（因为我们做了前 padding）
        token_weights = weights[-n_tokens:] if n_tokens <= len(weights) else weights[:n_tokens]

        # 显示不超过 25 个词
        if n_tokens > 25:
            tokens = tokens[-25:]
            token_weights = token_weights[-25:]

        # 热力图
        token_weights_2d = token_weights.reshape(1, -1)
        im = ax.imshow(token_weights_2d, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=max(token_weights.max(), 0.01))
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_yticks([])

        # 标注数值
        for j, w in enumerate(token_weights):
            if w > token_weights.max() * 0.3:  # 只标注重要的
                ax.text(j, 0, f"{w:.3f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color="white" if w > 0.5 else "black")

        true_label = label_names[label]
        pred_label = label_names[pred]
        ax.set_title(f"真实: {true_label} → 预测: {pred_label}", fontsize=13, fontweight="bold")

    fig.suptitle("Attention-LSTM 注意力权重可视化", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "attention_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存 {path}")


# ── 主函数 ────────────────────────────────────────────────

def main():
    """生成所有可视化图表。"""
    print("[visualize] ===== 开始生成可视化图表 =====\n")

    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    plot_loss_curves()
    plot_accuracy_f1_curves()
    plot_confusion_matrices()
    plot_model_comparison()
    plot_sentiment_distribution()
    plot_wordclouds()
    plot_tsne_embeddings()
    plot_attention_heatmaps()

    print(f"\n[visualize] ===== 所有图表已生成至 {TRAINING_DIR}/, {EVALUATION_DIR}/, {ANALYSIS_DIR}/ =====")


if __name__ == "__main__":
    main()
