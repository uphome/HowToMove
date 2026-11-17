"""把已有 SFT/GRPO 日志画成曲线，便于报告复盘。

    python scripts/plot_metrics.py \
        --sft-log /tmp/opencode/sft_l0_100.log \
        --sft-log /tmp/opencode/sft_l1_600.log \
        --grpo-json /tmp/opencode/grpo_l1_sft600.json \
        --out reports/curves
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SFT_RE = re.compile(r"epoch=(\d+)\s+step=(\d+)\s+loss=([0-9.eE+-]+)")


def parse_sft(path: Path) -> tuple[list[int], list[float], str]:
    steps, losses = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = SFT_RE.search(line)
        if m:
            steps.append(int(m.group(2)))
            losses.append(float(m.group(3)))
    return steps, losses, path.stem


def plot_sft(logs: list[Path], out: Path) -> None:
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for path in logs:
        steps, losses, name = parse_sft(path)
        if not steps:
            continue
        ax.plot(steps, losses, linewidth=1.4, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy loss")
    ax.set_title("SFT loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sft_loss.png", dpi=150)
    plt.close(fig)


def plot_grpo(path: Path, out: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("train", data if isinstance(data, list) else [])
    if not rows:
        return
    updates = [r.get("update", i + 1) for i, r in enumerate(rows)]
    success = [100.0 * r.get("success_rate", 0.0) for r in rows]
    reward = [r.get("mean_reward", 0.0) for r in rows]
    loss = [r.get("loss") for r in rows]
    collisions = [r.get("collisions", 0) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(updates, success, marker="o")
    ax.set_ylim(-5, 105)
    ax.set_ylabel("success %")
    ax.set_title("GRPO success rate")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(updates, reward, marker="o")
    ax.set_ylabel("mean reward")
    ax.set_title("GRPO mean reward")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    valid = [(u, v) for u, v in zip(updates, loss) if v is not None]
    if valid:
        ax.plot([x[0] for x in valid], [x[1] for x in valid], marker="o")
    ax.set_ylabel("loss")
    ax.set_title("GRPO loss (updates with samples)")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(updates, collisions, marker="o")
    ax.set_ylabel("collisions / update")
    ax.set_title("GRPO collisions")
    ax.grid(alpha=0.25)

    for ax in axes.flat:
        ax.set_xlabel("update")
    fig.tight_layout()
    fig.savefig(out / "grpo_metrics.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-log", action="append", default=[],
                    help="可重复；SFT 训练日志")
    ap.add_argument("--grpo-json", default=None,
                    help="train_grpo 产出的 JSON")
    ap.add_argument("--out", default="reports/curves")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_sft([Path(x) for x in args.sft_log], out)
    if args.grpo_json:
        plot_grpo(Path(args.grpo_json), out)
    print(f"curves -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
