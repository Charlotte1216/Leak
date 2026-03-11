"""
run_ablation.py
Run ablation study for:
1) FFNN + no aux head
2) FFNN + aux head
3) LSTM + no aux head
4) LSTM + aux head

Usage:
    python run_ablation.py
"""
from __future__ import annotations

import copy
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "marl_leakage_search").exists():
            return parent
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
TRAIN_SCRIPT = REPO_ROOT / "train.py"
DEFAULT_TRAIN_CFG = REPO_ROOT / "marl_leakage_search" / "configs" / "train_config.yaml"
DEFAULT_AGENT_CFG = REPO_ROOT / "marl_leakage_search" / "configs" / "agent_config.yaml"
EXPERIMENT_DIR = REPO_ROOT / "marl_leakage_search" / "experiments" / "Train_network"


@dataclass(frozen=True)
class AblationVariant:
    name: str
    network_type: str
    aux_enabled: bool
    lstm_training_enabled: bool


VARIANTS: List[AblationVariant] = [
    AblationVariant(
        name="ffnn_no_aux",
        network_type="ffnn",
        aux_enabled=False,
        lstm_training_enabled=False,
    ),
    AblationVariant(
        name="ffnn_aux",
        network_type="ffnn",
        aux_enabled=True,
        lstm_training_enabled=False,
    ),
    AblationVariant(
        name="lstm_no_aux",
        network_type="lstm",
        aux_enabled=False,
        lstm_training_enabled=True,
    ),
    AblationVariant(
        name="lstm_aux",
        network_type="lstm",
        aux_enabled=True,
        lstm_training_enabled=True,
    ),
]

# ---------------------------------------------------------------------------
# Direct-run settings (edit here; no CLI args required)
# ---------------------------------------------------------------------------
RUN_TRAIN_CONFIG: Path = DEFAULT_TRAIN_CFG
RUN_AGENT_CONFIG: Path = DEFAULT_AGENT_CFG
RUN_FIELD_DIR: str | None = None
RUN_SEEDS: List[int] = [42]
RUN_AUX_WEIGHT: float = 0.1
RUN_TAIL_WINDOW: int = 200
RUN_NUM_EPISODES: int | None = None
RUN_MAX_STEPS: int | None = None
RUN_FORCE_MAPPO: bool = True
RUN_PYTHON_EXE: str = sys.executable
MAX_PARALLEL_JOBS: int = 4


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Please run with an environment that has yaml installed, "
            "e.g. `conda run -n PriorText python run_ablation.py`."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Please run with an environment that has yaml installed, "
            "e.g. `conda run -n PriorText python run_ablation.py`."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _parse_seed_list(text: str) -> List[int]:
    parts = [x.strip() for x in text.split(",")]
    seeds = [int(x) for x in parts if x]
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def _build_file_tag(train_cfg: Dict[str, Any], agent_cfg: Dict[str, Any]) -> str:
    seed = int(train_cfg.get("seed", 42))

    marl_cfg = train_cfg.get("marl", {})
    if not isinstance(marl_cfg, dict):
        marl_cfg = {}
    marl_algorithm = str(marl_cfg.get("algorithm", "mappo")).strip().lower()

    env_cfg = train_cfg.get("environment", {})
    if not isinstance(env_cfg, dict):
        env_cfg = {}
    num_agents = int(env_cfg.get("num_agents", 2))

    communication_cfg = env_cfg.get("communication", {})
    if not isinstance(communication_cfg, dict):
        communication_cfg = {}
    comm_enabled = bool(communication_cfg.get("enabled", False))
    comm_top_k = max(0, int(communication_cfg.get("top_k", 0)))
    comm_tag = f"commk{comm_top_k}" if comm_enabled else "commoff"

    comm_channel_cfg = communication_cfg.get("channel", {})
    if not isinstance(comm_channel_cfg, dict):
        comm_channel_cfg = {}
    comm_channel_enabled = bool(comm_channel_cfg.get("enabled", False))
    comm_channel_mode = str(comm_channel_cfg.get("mode", "none")).strip().lower()
    channel_tag = f"ch{comm_channel_mode}" if comm_channel_enabled else "chnone"

    agent_algorithm = str(agent_cfg.get("algorithm", "ppo")).strip().lower()
    network_type = str(
        agent_cfg.get("network", {}).get(
            "type", agent_cfg.get("network_type", "ffnn")
        )
    ).strip().lower()
    learning_cfg = agent_cfg.get("learning", {})
    if not isinstance(learning_cfg, dict):
        learning_cfg = {}
    lr = float(learning_cfg.get("lr", 0.0))
    agent_gamma = float(learning_cfg.get("gamma", 0.0))
    batch_size = int(learning_cfg.get("batch_size", 0))

    if marl_algorithm == "mappo":
        gamma = float(marl_cfg.get("mappo", {}).get("gamma", agent_gamma))
    elif marl_algorithm == "qmix":
        gamma = float(marl_cfg.get("qmix", {}).get("gamma", agent_gamma))
    else:
        gamma = agent_gamma

    aux_enabled = bool(
        learning_cfg.get("ppo", {}).get("aux", {}).get("enabled", False)
    )
    aux_tag = "auxon" if aux_enabled else "auxoff"

    return (
        f"seed{seed}_agent{agent_algorithm}_net{network_type}_marl{marl_algorithm}_na{num_agents}"
        f"_{comm_tag}_{channel_tag}_lr{lr:.6f}_gamma{gamma:.4f}_bs{batch_size}_{aux_tag}"
    )


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _collect_metrics_from_training_stats(stats_path: Path, tail_window: int) -> Dict[str, Any]:
    if not stats_path.exists():
        return {}

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    rewards = payload.get("episode_rewards", [])
    lengths = payload.get("episode_lengths", [])
    success = payload.get("success_episodes", [])
    partial = payload.get("partial_success_episodes", [])
    found_ratio = payload.get("found_source_ratios", [])

    n = min(len(rewards), len(lengths), len(success), len(partial), len(found_ratio))
    if n == 0:
        return {}

    rewards = rewards[:n]
    lengths = lengths[:n]
    success = success[:n]
    partial = partial[:n]
    found_ratio = found_ratio[:n]

    tail = min(tail_window, n)
    rewards_tail = rewards[-tail:]
    lengths_tail = lengths[-tail:]
    success_tail = success[-tail:]
    partial_tail = partial[-tail:]
    found_ratio_tail = found_ratio[-tail:]

    return {
        "episodes": n,
        "tail_window": tail,
        "avg_reward_all": _safe_mean(rewards),
        "avg_reward_tail": _safe_mean(rewards_tail),
        "avg_length_all": _safe_mean(lengths),
        "avg_length_tail": _safe_mean(lengths_tail),
        "success_rate_all": _safe_mean(success),
        "success_rate_tail": _safe_mean(success_tail),
        "partial_success_rate_all": _safe_mean(partial),
        "partial_success_rate_tail": _safe_mean(partial_tail),
        "avg_found_ratio_all": _safe_mean(found_ratio),
        "avg_found_ratio_tail": _safe_mean(found_ratio_tail),
    }


def _prepare_variant_configs(
    base_train_cfg: Dict[str, Any],
    base_agent_cfg: Dict[str, Any],
    *,
    seed: int,
    variant: AblationVariant,
    aux_weight: float,
    force_mappo: bool,
    num_episodes: int | None,
    max_steps: int | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    train_cfg = copy.deepcopy(base_train_cfg)
    agent_cfg = copy.deepcopy(base_agent_cfg)

    train_cfg["seed"] = int(seed)

    if force_mappo:
        train_cfg.setdefault("marl", {})
        train_cfg["marl"]["algorithm"] = "mappo"
        agent_cfg["algorithm"] = "ppo"

    if num_episodes is not None:
        train_cfg.setdefault("training", {})
        train_cfg["training"]["num_episodes"] = int(num_episodes)
    if max_steps is not None:
        train_cfg.setdefault("training", {})
        train_cfg["training"]["max_steps_per_episode"] = int(max_steps)

    agent_cfg.setdefault("network", {})
    agent_cfg["network"]["type"] = variant.network_type
    agent_cfg["network_type"] = variant.network_type

    agent_cfg.setdefault("learning", {})
    agent_cfg["learning"].setdefault("ppo", {})
    ppo_cfg = agent_cfg["learning"]["ppo"]

    ppo_cfg.setdefault("aux", {})
    ppo_cfg["aux"]["enabled"] = bool(variant.aux_enabled)
    ppo_cfg["aux"]["weight"] = float(aux_weight if variant.aux_enabled else 0.0)

    ppo_cfg.setdefault("lstm_training", {})
    ppo_cfg["lstm_training"]["enabled"] = bool(variant.lstm_training_enabled)

    return train_cfg, agent_cfg


def _run_train(
    python_exe: str,
    train_cfg_path: Path,
    agent_cfg_path: Path,
    field_dir: str | None,
) -> int:
    cmd = [
        python_exe,
        str(TRAIN_SCRIPT),
        "--train-config",
        str(train_cfg_path),
        "--agent-config",
        str(agent_cfg_path),
    ]
    if field_dir:
        cmd.extend(["--field-dir", field_dir])
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def _run_single_ablation(
    *,
    index: int,
    total: int,
    variant: AblationVariant,
    seed: int,
    train_cfg_base: Dict[str, Any],
    agent_cfg_base: Dict[str, Any],
    cfg_dir: Path,
    aux_weight: float,
    force_mappo: bool,
    num_episodes: int | None,
    max_steps: int | None,
    python_exe: str,
    field_dir: str | None,
    tail_window: int,
) -> Dict[str, Any]:
    print(f"[{index}/{total}] Running variant={variant.name}, seed={seed}")

    train_cfg, agent_cfg = _prepare_variant_configs(
        train_cfg_base,
        agent_cfg_base,
        seed=seed,
        variant=variant,
        aux_weight=float(aux_weight),
        force_mappo=bool(force_mappo),
        num_episodes=num_episodes,
        max_steps=max_steps,
    )

    tag = _build_file_tag(train_cfg, agent_cfg)
    train_cfg_path = cfg_dir / f"train_config_{variant.name}_seed{seed}.yaml"
    agent_cfg_path = cfg_dir / f"agent_config_{variant.name}_seed{seed}.yaml"
    _save_yaml(train_cfg_path, train_cfg)
    _save_yaml(agent_cfg_path, agent_cfg)

    code = _run_train(
        python_exe=python_exe,
        train_cfg_path=train_cfg_path,
        agent_cfg_path=agent_cfg_path,
        field_dir=field_dir,
    )

    stats_path = REPO_ROOT / "logs" / tag / "training_stats.json"
    metrics = _collect_metrics_from_training_stats(stats_path, int(tail_window))
    status = "ok" if code == 0 else f"failed_{code}"

    record: Dict[str, Any] = {
        "variant": variant.name,
        "seed": int(seed),
        "file_tag": tag,
        "status": status,
        "return_code": code,
        "train_config_path": str(train_cfg_path),
        "agent_config_path": str(agent_cfg_path),
        "stats_path": str(stats_path),
        **metrics,
    }
    if metrics:
        print(
            "  "
            f"status={status}, "
            f"success_tail={metrics.get('success_rate_tail', 0.0):.2%}, "
            f"found_ratio_tail={metrics.get('avg_found_ratio_tail', 0.0):.2%}, "
            f"reward_tail={metrics.get('avg_reward_tail', 0.0):.2f}"
        )
    else:
        print(f"  status={status}, metrics unavailable (stats file missing or empty)")

    return record


def main() -> None:
    seeds = list(RUN_SEEDS)
    if not seeds:
        raise ValueError("RUN_SEEDS is empty.")

    train_cfg_base = _load_yaml(Path(RUN_TRAIN_CONFIG))
    agent_cfg_base = _load_yaml(Path(RUN_AGENT_CONFIG))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_dir = EXPERIMENT_DIR / "ablation_configs" / timestamp
    cfg_dir.mkdir(parents=True, exist_ok=True)

    run_results: List[Dict[str, Any]] = []
    tasks = [(variant, seed) for variant in VARIANTS for seed in seeds]
    total_runs = len(tasks)

    max_workers = max(1, int(MAX_PARALLEL_JOBS))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, (variant, seed) in enumerate(tasks, start=1):
            future = executor.submit(
                _run_single_ablation,
                index=idx,
                total=total_runs,
                variant=variant,
                seed=seed,
                train_cfg_base=train_cfg_base,
                agent_cfg_base=agent_cfg_base,
                cfg_dir=cfg_dir,
                aux_weight=RUN_AUX_WEIGHT,
                force_mappo=RUN_FORCE_MAPPO,
                num_episodes=RUN_NUM_EPISODES,
                max_steps=RUN_MAX_STEPS,
                python_exe=RUN_PYTHON_EXE,
                field_dir=RUN_FIELD_DIR,
                tail_window=RUN_TAIL_WINDOW,
            )
            futures.append(future)

        for future in as_completed(futures):
            run_results.append(future.result())

    summary_rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [r for r in run_results if r["variant"] == variant.name and r.get("episodes")]
        if not rows:
            summary_rows.append(
                {
                    "variant": variant.name,
                    "n_runs": 0,
                    "success_rate_tail_mean": 0.0,
                    "success_rate_tail_std": 0.0,
                    "found_ratio_tail_mean": 0.0,
                    "found_ratio_tail_std": 0.0,
                    "avg_reward_tail_mean": 0.0,
                    "avg_reward_tail_std": 0.0,
                }
            )
            continue

        success_vals = [float(r.get("success_rate_tail", 0.0)) for r in rows]
        found_vals = [float(r.get("avg_found_ratio_tail", 0.0)) for r in rows]
        reward_vals = [float(r.get("avg_reward_tail", 0.0)) for r in rows]
        summary_rows.append(
            {
                "variant": variant.name,
                "n_runs": len(rows),
                "success_rate_tail_mean": _safe_mean(success_vals),
                "success_rate_tail_std": _safe_std(success_vals),
                "found_ratio_tail_mean": _safe_mean(found_vals),
                "found_ratio_tail_std": _safe_std(found_vals),
                "avg_reward_tail_mean": _safe_mean(reward_vals),
                "avg_reward_tail_std": _safe_std(reward_vals),
            }
        )

    summary_rows.sort(key=lambda x: x["success_rate_tail_mean"], reverse=True)

    results_json = EXPERIMENT_DIR / f"ablation_results_{timestamp}.json"
    summary_json = EXPERIMENT_DIR / f"ablation_summary_{timestamp}.json"
    summary_csv = EXPERIMENT_DIR / f"ablation_summary_{timestamp}.csv"

    results_latest = EXPERIMENT_DIR / "ablation_results_latest.json"
    summary_latest_json = EXPERIMENT_DIR / "ablation_summary_latest.json"
    summary_latest_csv = EXPERIMENT_DIR / "ablation_summary_latest.csv"

    results_json.write_text(json.dumps(run_results, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    results_latest.write_text(json.dumps(run_results, indent=2), encoding="utf-8")
    summary_latest_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    csv_fields = [
        "variant",
        "n_runs",
        "success_rate_tail_mean",
        "success_rate_tail_std",
        "found_ratio_tail_mean",
        "found_ratio_tail_std",
        "avg_reward_tail_mean",
        "avg_reward_tail_std",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    with summary_latest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("")
    print(f"Saved per-run results: {results_json}")
    print(f"Saved summary (json): {summary_json}")
    print(f"Saved summary (csv):  {summary_csv}")
    print("Latest aliases:")
    print(f"  {results_latest}")
    print(f"  {summary_latest_json}")
    print(f"  {summary_latest_csv}")


if __name__ == "__main__":
    main()
