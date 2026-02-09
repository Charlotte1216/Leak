"""
2000Tread.py
Plot smoothed avg_reward trend for the first 2000 episodes.
python 2000Tread.py --seed 42 --agent-algorithm dqn --network-type ffnn --marl-algorithm qmix --lr 0.0001 --gamma 0.9 --batch-size 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(
    "C:/Users/Charlotte/NewStart/EXP/GaosiLeak/marl_leakage_search/experiments/Train_network"
)

DEFAULT_MIN_EPISODES = 500
DEFAULT_MAX_EPISODES = 2000
DEFAULT_SMOOTH_WINDOW = 25


def _file_tag(
    seed: int,
    agent_algorithm: str,
    network_type: str,
    marl_algorithm: str,
    lr: float,
    gamma: float,
    batch_size: int,
) -> str:
    return (
        f"seed{seed}_agent{agent_algorithm}_net{network_type}_marl{marl_algorithm}"
        f"_lr{lr:.6f}_gamma{gamma:.4f}_bs{batch_size}"
    )


def _csv_path(
    seed: int,
    agent_algorithm: str,
    network_type: str,
    marl_algorithm: str,
    lr: float,
    gamma: float,
    batch_size: int,
) -> Path:
    return EXPERIMENT_DIR / (
        f"{_file_tag(seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size)}_avg_reward_trend.csv"
    )


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if rows.size == 0:
        return np.array([]), np.array([])
    if rows.ndim == 1:
        return np.array([rows[0]]), np.array([rows[1]])
    return rows[:, 0], rows[:, 1]


def _smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    if window <= 1 or values.size <= 1:
        return values, 1
    window = min(window, values.size)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid"), window


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot smoothed Avg Reward trend (first 2000 episodes).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--agent-algorithm", type=str, default="ppo")
    parser.add_argument("--network-type", type=str, default="ffnn")
    parser.add_argument("--marl-algorithm", type=str, default="mappo")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--min-episodes", type=int, default=DEFAULT_MIN_EPISODES)
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    args = parser.parse_args()

    agent_algorithm = args.agent_algorithm.strip().lower()
    network_type = args.network_type.strip().lower()
    marl_algorithm = args.marl_algorithm.strip().lower()
    csv_path = _csv_path(
        args.seed,
        agent_algorithm,
        network_type,
        marl_algorithm,
        args.lr,
        args.gamma,
        args.batch_size,
    )
    episodes, rewards = _load_csv(csv_path)

    if episodes.size == 0:
        print(f"No data in {csv_path}")
        return

    min_episodes = int(args.min_episodes)
    max_episodes = int(args.max_episodes)
    if min_episodes > max_episodes:
        min_episodes, max_episodes = max_episodes, min_episodes

    mask = (episodes >= min_episodes) & (episodes <= max_episodes)
    if not np.any(mask):
        print(f"No episodes in [{min_episodes}, {max_episodes}] for {csv_path}")
        return

    episodes = episodes[mask]
    rewards = rewards[mask]

    smooth_rewards, window = _smooth(rewards, int(args.smooth_window))
    smooth_episodes = episodes if smooth_rewards.size == rewards.size else episodes[window - 1 :]

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards, color="gray", alpha=0.35, linewidth=1.0, label="raw")
    plt.plot(
        smooth_episodes,
        smooth_rewards,
        color="tab:blue",
        linewidth=2.0,
        label=f"smoothed (window={window})",
    )
    plt.title(
        f"Avg Reward Trend (Episodes {min_episodes}-{max_episodes})\n"
        f"{_file_tag(args.seed, agent_algorithm, network_type, marl_algorithm, args.lr, args.gamma, args.batch_size)}"
    )
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
