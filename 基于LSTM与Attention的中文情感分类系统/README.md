# 基于 LSTM 与 Attention 的中文情感分类系统

## 项目简介

对中文评论进行情感二分类（**正面 / 负面**），对比五种模型的性能差异：

- **RNN** — 基线模型
- **BiLSTM** — 双向长短期记忆网络
- **Attention-LSTM** — BiLSTM + 加性 Self-Attention
- **CNN-BiLSTM** — 多尺度卷积 + BiLSTM + Attention
- **BERT** — 预训练对照组（bert-base-chinese）

数据规模：80,000 条（正负各 40,000），覆盖酒店、外卖、电商三个领域。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install transformers  # BERT 额外依赖
```

### 2. 获取模型权重

训练好的模型文件（`.pt`）较大，请从以下方式获取：

- **百度网盘**：
  - 预处理缓存（`processed.zip`）：https://pan.baidu.com/s/1zfEUz_77bduHHNTUIbKMoA?pwd=wmxq 提取码: wmxq
  - 模型权重（`models.zip`）：https://pan.baidu.com/s/13XJA0qev9Hj7ma4gcnJnVA?pwd=9t6a 提取码: 9t6a
  - 下载后解压到项目根目录即可
- **自行训练**（需要 GPU，约 4~6 小时）：

```bash
python main.py --step 1   # 下载数据集
python main.py --step 2   # 文本预处理
python main.py --step 3   # 训练 Word2Vec 词向量
python main.py --step 4   # 训练全部模型（含 BERT）
python main.py --step 5   # 模型评估
python main.py --step 6   # 生成可视化图表
```

或一键执行：

```bash
python main.py
```

### 3. 启动交互式 Demo

```bash
python app.py
```

浏览器访问 `http://localhost:5000`。

## 实验结果

| 模型 | 16K Acc | 80K Acc | 参数量 | 特点 |
|------|:--:|:--:|:--:|------|
| RNN | 81.79% | **89.82%** | 9.5M | 基线，最轻量 |
| BiLSTM | 87.46% | **92.44%** | 11.5M | 从零训练中最优 |
| Attention-LSTM | 87.71% | 92.37% | 11.8M | 注意力可解释 |
| CNN-BiLSTM | 87.83% | 92.12% | 12.1M | 自动 n-gram |
| BERT | — | **94.66%** | 102M | 预训练天花板 |

**核心发现**：五倍数据令 RNN 涨幅 +8 pp，超过 16K 下任何复杂模型——数据规模 > 模型复杂度。

## 项目结构

```
├── data/
│   ├── raw/                # 原始数据集（TSV/CSV）
│   └── processed/          # 预处理缓存（.pkl/.npy，运行生成）
├── models/                 # 训练好的 .pt 权重（需单独下载）
├── results/                # 可视化图表（运行后生成）
├── src/
│   ├── config.py           # 全局超参数、停用词表
│   ├── download_data.py    # 数据下载与合并
│   ├── preprocess.py       # jieba 分词 + 词表构建
│   ├── word2vec.py         # Skip-gram 词向量训练
│   ├── dataset.py          # PyTorch DataLoader
│   ├── train.py            # 训练循环 + 早停
│   ├── evaluate.py         # 评估 + 混淆矩阵
│   ├── visualize.py        # 8 类可视化图表
│   └── models/
│       ├── rnn.py
│       ├── lstm.py
│       ├── attention_lstm.py
│       ├── cnn_lstm.py
│       └── bert.py
├── static/                 # 前端 JS/CSS
├── templates/              # HTML 模板
├── app.py                  # Flask Demo
├── main.py                 # CLI 全流程入口
├── requirements.txt
└── README.md
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 中文分词 | jieba |
| 词向量 | gensim Word2Vec (Skip-gram, 300d) |
| 深度学习框架 | PyTorch |
| 预训练模型 | bert-base-chinese (HuggingFace) |
| Web Demo | Flask + Chart.js |
| 可视化 | matplotlib + seaborn + wordcloud |

---

**课程项目 — 自然语言处理**
