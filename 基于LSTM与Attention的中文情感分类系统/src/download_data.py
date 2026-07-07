"""
联网下载中文情感分析数据集（二分类）

数据来源:
1. ChnSentiCorp — 酒店评论 (9600条, 正面/负面)
2. waimai_10k  — 外卖评论 (11987条, 正面/负面)

标签: 0=负面, 1=正面 (使用原始人工标注，不做任何修改)
"""

import os
import random
import csv

from config import RAW_DIR, RANDOM_SEED

random.seed(RANDOM_SEED)
HF_ENDPOINT = "https://hf-mirror.com"


def download_chnsenticorp() -> list[tuple[str, int]]:
    """从 HuggingFace 镜像下载 ChnSentiCorp。"""
    print("[download] 正在从 HuggingFace 镜像下载 ChnSentiCorp...")
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        from datasets import load_dataset
        ds = load_dataset("lansinuote/ChnSentiCorp", split="train")
    except Exception as e:
        raise RuntimeError(f"ChnSentiCorp 下载失败: {e}")

    data = []
    for item in ds:
        text = item.get("text", "").strip()
        label = item.get("label", -1)
        if text and len(text) >= 10 and label in (0, 1):
            data.append((text, label))

    neg = sum(1 for _, l in data if l == 0)
    pos = sum(1 for _, l in data if l == 1)
    print(f"[download] ChnSentiCorp: {len(data)} 条 (负面={neg}, 正面={pos})")
    return data


def download_waimai() -> list[tuple[str, int]]:
    """从阿里云 OSS 下载 waimai_10k。"""
    print("[download] 正在下载 waimai_10k (阿里云 OSS)...")

    urls = [
        "https://labfile.oss.aliyuncs.com/courses/3205/waimai_10k.csv",
        "https://sandbox-expriment-files.obs.cn-north-1.myhuaweicloud.com/20221019/waimai_10_mapro.csv",
        "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
    ]

    import urllib.request

    for url in urls:
        try:
            print(f"  尝试: {url[:60]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            content = resp.read().decode("utf-8")
            break
        except Exception as e:
            print(f"  失败: {e}")
    else:
        print("[download] waimai_10k 所有源均不可用，跳过")
        return []

    # 解析 CSV
    data = []
    reader = csv.reader(content.splitlines())
    header = next(reader, None)
    for row in reader:
        if len(row) != 2:
            continue
        label_str, text = row
        try:
            label = int(label_str.strip())
        except ValueError:
            continue
        text = text.strip()
        if text and len(text) >= 5 and label in (0, 1):
            data.append((text, label))

    neg = sum(1 for _, l in data if l == 0)
    pos = sum(1 for _, l in data if l == 1)
    print(f"[download] waimai_10k: {len(data)} 条 (负面={neg}, 正面={pos})")
    return data


def balance_and_save(all_data: list, output_path: str):
    """平衡正负样本并保存为 TSV。"""
    neg = [(t, l) for t, l in all_data if l == 0]
    pos = [(t, l) for t, l in all_data if l == 1]

    target = min(len(neg), len(pos), 8000)  # 每类不超过 8000
    target = max(target, 2000)

    print(f"[download] 平衡目标: 每类 {target} 条")

    if len(neg) > target:
        neg = random.sample(neg, target)
    if len(pos) > target:
        pos = random.sample(pos, target)

    all_data = neg + pos
    random.shuffle(all_data)

    n_neg = sum(1 for _, l in all_data if l == 0)
    n_pos = sum(1 for _, l in all_data if l == 1)
    print(f"[download] 最终: 负面={n_neg}, 正面={n_pos}, 总计={len(all_data)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("label\ttext\n")
        for text, label in all_data:
            clean = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
            f.write(f"{label}\t{clean}\n")

    print(f"[download] 数据已保存至: {output_path}")


def load_onlineshopping() -> list[tuple[str, int]]:
    """加载本地 online_shopping_10_cats.csv（需要手动下载放置到 data/raw/）。"""
    csv_path = os.path.join(RAW_DIR, "online_shopping_10_cats.csv")
    if not os.path.exists(csv_path):
        print("[download] online_shopping_10_cats.csv 未找到，跳过。")
        print("  如需加入，请将 CSV 文件放到 data/raw/ 目录。")
        return []

    print("[download] 加载 online_shopping_10_cats.csv...")
    data = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过 header: cat,label,review
        for row in reader:
            if len(row) < 3:
                continue
            try:
                label = int(row[1].strip())
            except ValueError:
                continue
            text = row[2].strip()
            if text and len(text) >= 5 and label in (0, 1):
                data.append((text, label))

    neg = sum(1 for _, l in data if l == 0)
    pos = sum(1 for _, l in data if l == 1)
    print(f"[download] online_shopping_10_cats: {len(data)} 条 (负面={neg}, 正面={pos})")
    return data


def main():
    """下载/加载 ChnSentiCorp + waimai_10k + online_shopping，合并为二分类数据集。"""
    output_path = os.path.join(RAW_DIR, "sentiment_dataset.tsv")

    if os.path.exists(output_path):
        print(f"[download] {output_path} 已存在，跳过下载")
        return

    chn = download_chnsenticorp()
    waimai = download_waimai()
    shopping = load_onlineshopping()

    all_data = chn + waimai + shopping
    if not all_data:
        raise RuntimeError("所有数据源加载失败！")

    print(f"\n[download] 合并前总计: {len(all_data)} 条")
    balance_and_save(all_data, output_path)


if __name__ == "__main__":
    main()
