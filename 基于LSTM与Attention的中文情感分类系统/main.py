"""
基于 LSTM 与 Attention 的中文情感分类系统 — 主入口

用法:
    python main.py                # 运行全流程
    python main.py --step 1       # 仅下载数据
    python main.py --step 2       # 仅预处理
    python main.py --step 3       # 仅训练词向量
    python main.py --step 4       # 仅训练模型
    python main.py --step 5       # 仅评估
    python main.py --step 6       # 仅生成可视化
    python main.py --skip-download  # 跳过下载，从已有数据开始
"""

import os
import sys
import argparse

# 将 src 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(
        description="基于 LSTM 与 Attention 的中文情感分类系统"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="仅运行指定步骤: 1=下载 2=预处理 3=词向量 4=训练 5=评估 6=可视化"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="跳过数据集下载步骤"
    )
    args = parser.parse_args()

    steps = []

    if args.step is not None:
        steps = [args.step]
    else:
        steps = [1, 2, 3, 4, 5, 6]

    print("=" * 65)
    print("  基于 LSTM 与 Attention 的中文情感分类系统")
    print("=" * 65)

    for step in steps:
        if step == 1:
            if args.skip_download:
                print("\n[main] 跳过第 1 步：数据集下载")
                continue
            print("\n" + "─" * 50)
            print("  第 1 步：下载数据集")
            print("─" * 50)
            from download_data import main as step_fn
            try:
                step_fn()
            except Exception as e:
                print(f"[main] 下载失败: {e}")
                print("[main] 将继续执行后续步骤（使用已有数据）")

        elif step == 2:
            print("\n" + "─" * 50)
            print("  第 2 步：文本预处理")
            print("─" * 50)
            from preprocess import main as step_fn
            step_fn()

        elif step == 3:
            print("\n" + "─" * 50)
            print("  第 3 步：Word2Vec 词向量训练")
            print("─" * 50)
            from word2vec import main as step_fn
            step_fn()

        elif step == 4:
            print("\n" + "─" * 50)
            print("  第 4 步：模型训练 (RNN, LSTM, Attention-LSTM, CNN-BiLSTM, BERT)")
            print("─" * 50)
            from train import train_all_models
            train_all_models()

        elif step == 5:
            print("\n" + "─" * 50)
            print("  第 5 步：模型评估")
            print("─" * 50)
            from evaluate import main as step_fn
            step_fn()

        elif step == 6:
            print("\n" + "─" * 50)
            print("  第 6 步：生成可视化图表")
            print("─" * 50)
            from visualize import main as step_fn
            step_fn()

    print("\n" + "=" * 65)
    print("  全流程完成！")
    print(f"  模型文件: {os.path.join('models', '')}")
    print(f"  评估报告: {os.path.join('results', 'evaluation', '')}")
    print(f"  可视化图表: {os.path.join('results', '')}")
    print(f"  启动 Demo: python app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
