"""
DrawTread.py
Load a specific avg_reward_trend.csv and plot reward trend.
python DrawTread.py --seed 42 --lr 0.0001 --gamma 0.9 --batch-size 32

"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(
    "C:/Users/Charlotte/NewStart/EXP/GaosiLeak/marl_leakage_search/experiments/Train_network"
)


def _file_tag(seed: int, lr: float, gamma: float, batch_size: int) -> str:
    return f"seed{seed}_lr{lr:.6f}_gamma{gamma:.4f}_bs{batch_size}"


def _csv_path(seed: int, lr: float, gamma: float, batch_size: int) -> Path:
    return EXPERIMENT_DIR / f"{_file_tag(seed, lr, gamma, batch_size)}_avg_reward_trend.csv"


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if rows.size == 0:
        return np.array([]), np.array([])
    if rows.ndim == 1:
        return np.array([rows[0]]), np.array([rows[1]])
    return rows[:, 0], rows[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Avg Reward trend from CSV.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    csv_path = _csv_path(args.seed, args.lr, args.gamma, args.batch_size)
    episodes, rewards = _load_csv(csv_path)

    if episodes.size == 0:
        print(f"No data in {csv_path}")
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards, marker="o", linewidth=1.5)
    plt.title(f"Avg Reward Trend\n{_file_tag(args.seed, args.lr, args.gamma, args.batch_size)}")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
