"""
Auto_train.py
Grid-search over seed/lr/gamma/batch_size and run train.py.
Stops early when reward trend looks increasing.
"""
from __future__ import annotations

import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# SEEDS = [42, 123, 256, 512, 1024]
# LEARNING_RATES = [1e-3, 5e-3, 1e-2, 5e-2]
# GAMMAS = [0.9, 0.95, 0.99]
# BATCH_SIZES = [64]

SEEDS = [42]
LEARNING_RATES = [1e-4]
GAMMAS = [0.9]
BATCH_SIZES = [64]

MIN_POINTS = 5
MIN_DELTA = 0.1
MIN_SLOPE = 0.0
PARALLEL_JOBS = 4


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _normalize_algorithm(value: object, default: str) -> str:
    text = str(value).strip().lower()
    return text if text else default


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


def _avg_reward_csv_path(
    seed: int,
    agent_algorithm: str,
    network_type: str,
    marl_algorithm: str,
    lr: float,
    gamma: float,
    batch_size: int,
) -> Path:
    return EXPERIMENT_DIR / f"{_file_tag(seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size)}_avg_reward_trend.csv"


def _experiment_log_path(
    seed: int,
    agent_algorithm: str,
    network_type: str,
    marl_algorithm: str,
    lr: float,
    gamma: float,
    batch_size: int,
) -> Path:
    return EXPERIMENT_DIR / f"{_file_tag(seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size)}.log"


def _clear_previous_run_artifacts(
    seed: int,
    agent_algorithm: str,
    network_type: str,
    marl_algorithm: str,
    lr: float,
    gamma: float,
    batch_size: int,
) -> None:
    for path in (
        _avg_reward_csv_path(seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size),
        _experiment_log_path(seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _read_avg_rewards(csv_path: Path) -> List[float]:
    if not csv_path.exists():
        return []
    rewards: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if not row:
                continue
            if not header_skipped:
                header_skipped = True
                continue
            if len(row) < 2:
                continue
            try:
                rewards.append(float(row[1]))
            except ValueError:
                continue
    return rewards


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
) -> Tuple[Path, Path, str, str, str]:
    train_cfg = _load_yaml(DEFAULT_TRAIN_CFG)
    agent_cfg = _load_yaml(DEFAULT_AGENT_CFG)

    train_cfg["seed"] = int(seed)
    marl_cfg = train_cfg.setdefault("marl", {})
    marl_algorithm = _normalize_algorithm(marl_cfg.get("algorithm"), "mappo")

    agent_cfg.setdefault("learning", {})
    agent_cfg["learning"]["lr"] = float(lr)
    agent_cfg["learning"]["gamma"] = float(gamma)
    agent_cfg["learning"]["batch_size"] = int(batch_size)

    # Make gamma effective for the active MARL trainer (MAPPO/QMIX), not only the agent config.
    if marl_algorithm == "mappo":
        marl_cfg.setdefault("mappo", {})
        marl_cfg["mappo"]["gamma"] = float(gamma)
    elif marl_algorithm == "qmix":
        marl_cfg.setdefault("qmix", {})
        marl_cfg["qmix"]["gamma"] = float(gamma)

    agent_algorithm = _normalize_algorithm(agent_cfg.get("algorithm"), "ppo")
    network_type = _normalize_algorithm(
        agent_cfg.get("network", {}).get("type", agent_cfg.get("network_type", "ffnn")),
        "ffnn",
    )

    tag = _file_tag(seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size)
    train_cfg_path = work_dir / f"train_config_{tag}.yaml"
    agent_cfg_path = work_dir / f"agent_config_{tag}.yaml"

    _save_yaml(train_cfg_path, train_cfg)
    _save_yaml(agent_cfg_path, agent_cfg)
    return train_cfg_path, agent_cfg_path, agent_algorithm, network_type, marl_algorithm


def custom_json_encoder(obj):
    if isinstance(obj, np.bool_):  # 处理 NumPy bool_ 类型
        return bool(obj)
    elif isinstance(obj, np.int64):  # 处理 NumPy int64 类型
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):  # 处理 NumPy float 类型
        return float(obj)
    elif isinstance(obj, np.ndarray):  # 处理 NumPy ndarray 类型
        return obj.tolist()  # 转换为普通的 Python 列表
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main() -> None:
    work_dir = EXPERIMENT_DIR / "auto_configs"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []

    combos = list(itertools.product(SEEDS, LEARNING_RATES, GAMMAS, BATCH_SIZES))

    def _run_combo(seed: int, lr: float, gamma: float, batch_size: int):
        train_cfg_path, agent_cfg_path, agent_algorithm, network_type, marl_algorithm = _prepare_configs(
            seed, lr, gamma, batch_size, work_dir
        )
        _clear_previous_run_artifacts(
            seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size
        )
        print(
            "Running: "
            f"seed={seed}, agent={agent_algorithm}, net={network_type}, marl={marl_algorithm}, "
            f"lr={lr}, gamma={gamma}, batch_size={batch_size}"
        )
        code = _run_train(train_cfg_path, agent_cfg_path)
        return seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size, code

    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        combo_iter = iter(combos)
        pending = []

        def submit_next():
            try:
                seed, lr, gamma, batch_size = next(combo_iter)
            except StopIteration:
                return
            future = executor.submit(_run_combo, seed, lr, gamma, batch_size)
            pending.append(future)

        for _ in range(PARALLEL_JOBS):
            submit_next()

        while pending:
            for future in as_completed(list(pending)):
                pending.remove(future)
                seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size, code = future.result()

                if code != 0:
                    print(f"Train failed with code {code}, skipping.")
                else:
                    csv_path = _avg_reward_csv_path(
                        seed, agent_algorithm, network_type, marl_algorithm, lr, gamma, batch_size
                    )
                    rewards = _read_avg_rewards(csv_path)
                    ok, slope = _reward_trend_ok(rewards)
                    results.append(
                        {
                            "seed": seed,
                            "agent_algorithm": agent_algorithm,
                            "network_type": network_type,
                            "marl_algorithm": marl_algorithm,
                            "lr": lr,
                            "gamma": gamma,
                            "batch_size": batch_size,
                            "trend_ok": ok,
                            "slope": slope,
                            "points": len(rewards),
                        }
                    )
                    print(f"Trend ok={ok}, slope={slope:.6f}, points={len(rewards)}")
                submit_next()

    results_path = EXPERIMENT_DIR / "auto_train_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=custom_json_encoder)

    ranked_results = sorted(
        results,
        key=lambda r: (r.get("trend_ok", False), r.get("slope", float("-inf")), r.get("points", 0)),
        reverse=True,
    )
    ranked_payload = [
        dict(item, rank=idx + 1) for idx, item in enumerate(ranked_results)
    ]
    ranked_path = EXPERIMENT_DIR / "auto_train_results_ranked.json"
    with ranked_path.open("w", encoding="utf-8") as f:
        json.dump(ranked_payload, f, indent=2, default=custom_json_encoder)

    if results:
        best = ranked_results[0]
        print("Best config:", best)
    else:
        print("No successful runs were recorded.")


if __name__ == "__main__":
    main()
