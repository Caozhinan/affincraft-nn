#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import re


def parse_losses(log_path):
    """
    解析日志里：
      2025-xx-xx ... | INFO | train | epoch 017 | loss 0.004299 | ...
      2025-xx-xx ... | INFO | valid | epoch 017 | valid on 'valid' subset | loss 1.00589 | ...
    这样的 train / valid 行
    """
    train_losses = {}
    valid_losses = {}

    # 简单匹配 valid 行：| valid | epoch N | ... | loss X |
    valid_pattern = re.compile(
        r"\|\s*valid\s*\|\s*epoch\s+(\d+)\s*\|.*?\|\s*loss\s+([0-9.]+)",
        re.IGNORECASE,
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # ---- 先看是不是 train 行 ----
            if "| train | epoch" in line and "| loss" in line:
                # 例如：... | train | epoch 017 | loss 0.004299 | ...
                parts = [p.strip() for p in line.split("|")]

                # parts 典型结构：
                # [时间, 'INFO', 'train', 'epoch 017', 'loss 0.004299', ...]
                epoch = None
                loss = None
                for p in parts:
                    if p.startswith("epoch"):
                        # "epoch 017" -> 17
                        try:
                            epoch = int(p.split()[1])
                        except Exception:
                            pass
                    elif p.startswith("loss"):
                        # "loss 0.004299" -> 0.004299
                        try:
                            loss = float(p.split()[1])
                        except Exception:
                            pass

                if epoch is not None and loss is not None:
                    train_losses[epoch] = loss
                continue  # 这一行已经处理过了

            # ---- 再匹配 valid 行 ----
            m_valid = valid_pattern.search(line)
            if m_valid:
                epoch = int(m_valid.group(1))
                loss = float(m_valid.group(2))
                valid_losses[epoch] = loss
                continue

    if not train_losses:
        print("警告：没有解析到任何 train loss（train_losses 为空）")
    if not valid_losses:
        print("警告：没有解析到任何 valid loss（valid_losses 为空）")
    if not train_losses and not valid_losses:
        raise RuntimeError("日志里既没有 train 也没有 valid 的 loss，请再检查一下路径和内容。")

    return train_losses, valid_losses


def plot_losses(train_losses, valid_losses, out_path="loss_curve.png"):
    """
    画出 train / valid loss 并保存
    """
    plt.style.use("seaborn-v0_8-darkgrid")

    mpl.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.figsize": (8, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    epochs = sorted(set(train_losses.keys()) | set(valid_losses.keys()))
    train_y = [train_losses.get(e, None) for e in epochs]
    valid_y = [valid_losses.get(e, None) for e in epochs]

    fig, ax = plt.subplots()

    if any(v is not None for v in train_y):
        ax.plot(
            epochs,
            train_y,
            marker="o",
            color="#1f77b4",
            linewidth=2.0,
            markersize=5,
            label="Train Loss",
        )

    if any(v is not None for v in valid_y):
        ax.plot(
            epochs,
            valid_y,
            marker="s",
            color="#d62728",
            linewidth=2.0,
            markersize=5,
            label="Valid Loss",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss Curve (fairseq)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(epochs)

    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=300)
    print(f"保存图片到: {out_path.resolve()}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("用法: python plot.py run_debug.out [loss_curve.png]")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = "loss_curve.png"

    train_losses, valid_losses = parse_losses(log_path)

    print("解析到的 epoch 数量：")
    print("  train epochs:", sorted(train_losses.keys()))
    print("  valid epochs:", sorted(valid_losses.keys()))

    plot_losses(train_losses, valid_losses, out_path)


if __name__ == "__main__":
    main()