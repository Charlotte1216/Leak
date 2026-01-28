"""
Auto_train.py
Grid-search over seed/lr/gamma/batch_size and run train.py.
Stops early when reward trend looks increasing.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "marl_leakage_search").exists():
            return parent
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
TRAIN_SCRIPT = Path("C:/Users/Charlotte/NewStart/EXP/GaosiLeak/train.py")
DEFAULT_TRAIN_CFG = REPO_ROOT / "marl_leakage_search" / "configs" / "train_config.yaml"
DEFAULT_AGENT_CFG = REPO_ROOT / "marl_leakage_search" / "configs" / "agent_config.yaml"
EXPERIMENT_DIR = REPO_ROOT / "marl_leakage_search" / "experiments" / "Train_network"

SEEDS = [42, 123, 256, 512, 1024]
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
GAMMAS = [0.9, 0.95, 0.99]
BATCH_SIZES = [32, 64, 128]

MIN_POINTS = 5
MIN_DELTA = 0.1
MIN_SLOPE = 0.0


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _file_tag(seed: int, lr: float, gamma: float, batch_size: int) -> str:
    return f"seed{seed}_lr{lr:.6f}_gamma{gamma:.4f}_bs{batch_size}"


def _avg_reward_csv_path(seed: int, lr: float, gamma: float, batch_size: int) -> Path:
    return EXPERIMENT_DIR / f"{_file_tag(seed, lr, gamma, batch_size)}_avg_reward_trend.csv"


def _read_avg_rewards(csv_path: Path) -> List[float]:
    if not csv_path.exists():
        return []
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if rows.size == 0:
        return []
    if rows.ndim == 1:
        return [float(rows[1])]
    return [float(x) for x in rows[:, 1]]


def _reward_trend_ok(rewards: List[float]) -> Tuple[bool, float]:
    if len(rewards) < MIN_POINTS:
        return False, float("-inf")
    
    # 使用移动平均来平滑奖励波动
    rewards_smooth = np.convolve(rewards, np.ones(5)/5, mode='valid')
    
    # 重新计算奖励的斜率和差值
    x = np.arange(len(rewards_smooth))
    slope = float(np.polyfit(x, rewards_smooth, 1)[0])
    delta = rewards_smooth[-1] - rewards_smooth[0]
    
    # 判断趋势的逻辑
    ok = slope > MIN_SLOPE and delta > MIN_DELTA
    return ok, slope


def _run_train(train_cfg: Path, agent_cfg: Path) -> int:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-config",
        str(train_cfg),
        "--agent-config",
        str(agent_cfg),
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _prepare_configs(
    seed: int,
    lr: float,
    gamma: float,
    batch_size: int,
    work_dir: Path,
) -> Tuple[Path, Path]:
    train_cfg = _load_yaml(DEFAULT_TRAIN_CFG)
    agent_cfg = _load_yaml(DEFAULT_AGENT_CFG)

    train_cfg["seed"] = int(seed)
    agent_cfg.setdefault("learning", {})
    agent_cfg["learning"]["lr"] = float(lr)
    agent_cfg["learning"]["gamma"] = float(gamma)
    agent_cfg["learning"]["batch_size"] = int(batch_size)

    tag = _file_tag(seed, lr, gamma, batch_size)
    train_cfg_path = work_dir / f"train_config_{tag}.yaml"
    agent_cfg_path = work_dir / f"agent_config_{tag}.yaml"

    _save_yaml(train_cfg_path, train_cfg)
    _save_yaml(agent_cfg_path, agent_cfg)
    return train_cfg_path, agent_cfg_path


def main() -> None:
    work_dir = EXPERIMENT_DIR / "auto_configs"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for seed, lr, gamma, batch_size in itertools.product(SEEDS, LEARNING_RATES, GAMMAS, BATCH_SIZES):
        train_cfg_path, agent_cfg_path = _prepare_configs(seed, lr, gamma, batch_size, work_dir)
        print(f"Running: seed={seed}, lr={lr}, gamma={gamma}, batch_size={batch_size}")

        code = _run_train(train_cfg_path, agent_cfg_path)
        if code != 0:
            print(f"Train failed with code {code}, skipping.")
            continue

        csv_path = _avg_reward_csv_path(seed, lr, gamma, batch_size)
        rewards = _read_avg_rewards(csv_path)
        ok, slope = _reward_trend_ok(rewards)
        results.append(
            {
                "seed": seed,
                "lr": lr,
                "gamma": gamma,
                "batch_size": batch_size,
                "trend_ok": ok,
                "slope": slope,
                "points": len(rewards),
            }
        )
        print(f"Trend ok={ok}, slope={slope:.6f}, points={len(rewards)}")

        if ok:
            print("Found acceptable trend, stopping search.")
            break

    results_path = EXPERIMENT_DIR / "auto_train_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results:
        best = max(results, key=lambda r: (r["trend_ok"], r["slope"]))
        print("Best config:", best)
    else:
        print("No successful runs were recorded.")


if __name__ == "__main__":
    main()
