"""
Run and summarize a comparison suite over the registered baseline algorithms.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "marl_leakage_search").exists():
            return parent
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marl_leakage_search.comparison_algorithms import ALGORITHM_REGISTRY

TRAIN_SCRIPT = REPO_ROOT / "train.py"
DEFAULT_FIELD_DIR = REPO_ROOT / "marl_leakage_search" / "envs" / "generated_fields"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "cmp"
DEFAULT_SUITE_CONFIG_PATH = (
    REPO_ROOT / "marl_leakage_search" / "configs" / "comparison_suite_config.yaml"
)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_env_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    if "environment" in raw:
        env_cfg = raw.get("environment", {})
        return {"environment": env_cfg if isinstance(env_cfg, dict) else {}}
    return {"environment": raw}


def _resolve_repo_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _pick_config_value(
    cli_value: Any,
    config: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def _normalize_requested_algorithms(values: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        for token in str(value).split(","):
            text = token.strip().lower()
            if text:
                tokens.append(text)
    if not tokens or tokens == ["all"]:
        return list(ALGORITHM_REGISTRY.keys())
    invalid = [token for token in tokens if token not in ALGORITHM_REGISTRY]
    if invalid:
        raise ValueError(
            f"Unknown algorithms: {invalid}. Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )
    return tokens


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _mean_last(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    width = max(1, min(int(window), len(values)))
    return float(np.mean(values[-width:]))


def _max_value(values: List[float]) -> float:
    return float(max(values)) if values else 0.0


def _resolve_report_window(num_items: int, requested_window: int) -> int:
    if num_items <= 0:
        return 0
    return max(1, min(int(requested_window), int(num_items)))


def _build_suite_tag(suite_tag: str | None) -> str:
    if suite_tag:
        return str(suite_tag).strip()
    return datetime.now().strftime("comparison_suite_%Y%m%d_%H%M%S")


def _prepare_train_payload(
    base_cfg: Dict[str, Any],
    *,
    log_dir: Path,
    save_dir: Path,
    seed: int | None,
    episodes: int | None,
    max_steps: int | None,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("training", {})
    cfg.setdefault("output", {})
    cfg["output"]["log_dir"] = str(log_dir.resolve())
    cfg["output"]["save_dir"] = str(save_dir.resolve())
    if seed is not None:
        cfg["seed"] = int(seed)
    if episodes is not None:
        cfg["training"]["num_episodes"] = int(episodes)
    if max_steps is not None:
        cfg["training"]["max_steps_per_episode"] = int(max_steps)
    return cfg


def _prepare_infotaxis_train_payload(
    base_cfg: Dict[str, Any],
    *,
    log_dir: Path,
    seed: int | None,
    episodes: int | None,
    max_steps: int | None,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("training", {})
    cfg.setdefault("output", {})
    cfg["output"]["log_dir"] = str(log_dir.resolve())
    if seed is not None:
        cfg["seed"] = int(seed)
    if episodes is not None:
        cfg["training"]["num_episodes"] = int(episodes)
    if max_steps is not None:
        cfg["training"]["max_steps_per_episode"] = int(max_steps)
    return cfg


def _write_episode_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "episode",
                    "episode_reward",
                    "episode_length",
                    "found_sources",
                    "found_ratio",
                    "success",
                    "partial_success",
                    "loss",
                ]
            )
        return

    fieldnames = [
        "episode",
        "episode_reward",
        "episode_length",
        "found_sources",
        "found_ratio",
        "success",
        "partial_success",
        "loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report_window_csv(path: Path, rows: List[Dict[str, Any]], report_window: int) -> None:
    width = _resolve_report_window(len(rows), report_window)
    tail_rows = rows[-width:] if width > 0 else []
    _write_episode_csv(path, tail_rows)


def _run_command(command: List[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(result.returncode), float(time.time() - started)


def _summarize_training_run(
    algorithm_key: str,
    algorithm_name: str,
    *,
    suite_tag: str,
    report_window: int,
    train_config_path: Path,
    agent_config_path: Path,
    env_config_path: Path,
    command_log_path: Path,
    artifacts_root: Path,
) -> Dict[str, Any]:
    train_cfg = _load_yaml(train_config_path)
    log_dir = Path(train_cfg.get("output", {}).get("log_dir", artifacts_root / "logs"))
    stats = _load_json(log_dir / "training_stats.json")

    rewards = [float(v) for v in stats.get("episode_rewards", [])]
    lengths = [float(v) for v in stats.get("episode_lengths", [])]
    losses = [float(v) for v in stats.get("losses", [])]
    found_sources = [float(v) for v in stats.get("found_sources", [])]
    found_ratios = [float(v) for v in stats.get("found_source_ratios", [])]
    success = [float(v) for v in stats.get("success_episodes", [])]
    partial = [float(v) for v in stats.get("partial_success_episodes", [])]

    num_episodes = len(rewards)
    actual_report_window = _resolve_report_window(num_episodes, report_window)
    episode_rows: List[Dict[str, Any]] = []
    for idx in range(num_episodes):
        episode_rows.append(
            {
                "episode": idx + 1,
                "episode_reward": rewards[idx] if idx < len(rewards) else 0.0,
                "episode_length": lengths[idx] if idx < len(lengths) else 0.0,
                "found_sources": found_sources[idx] if idx < len(found_sources) else 0.0,
                "found_ratio": found_ratios[idx] if idx < len(found_ratios) else 0.0,
                "success": int(success[idx]) if idx < len(success) else 0,
                "partial_success": int(partial[idx]) if idx < len(partial) else 0,
                "loss": losses[idx] if idx < len(losses) else 0.0,
            }
        )

    episode_csv_path = artifacts_root / "results" / "episode_metrics.csv"
    report_csv_path = artifacts_root / "results" / "report_window_metrics.csv"
    _write_episode_csv(episode_csv_path, episode_rows)
    _write_report_window_csv(report_csv_path, episode_rows, actual_report_window)

    run_dirs = sorted([path for path in log_dir.iterdir() if path.is_dir()]) if log_dir.exists() else []
    run_log_dir = run_dirs[-1] if run_dirs else log_dir
    summary = {
        "suite_tag": suite_tag,
        "algorithm_key": algorithm_key,
        "algorithm_name": algorithm_name,
        "run_type": "train",
        "num_episodes": num_episodes,
        "report_window": actual_report_window,
        "report_reward": _mean_last(rewards, actual_report_window),
        "report_found_ratio": _mean_last(found_ratios, actual_report_window),
        "report_success_rate": _mean_last(success, actual_report_window),
        "report_partial_success_rate": _mean_last(partial, actual_report_window),
        "report_loss": _mean_last(losses, actual_report_window),
        "avg_episode_reward": _mean(rewards),
        "final_window_reward": _mean_last(rewards, actual_report_window),
        "best_episode_reward": _max_value(rewards),
        "avg_episode_length": _mean(lengths),
        "avg_found_sources": _mean(found_sources),
        "avg_found_ratio": _mean(found_ratios),
        "final_window_found_ratio": _mean_last(found_ratios, actual_report_window),
        "best_found_ratio": _max_value(found_ratios),
        "success_rate": _mean(success),
        "final_window_success_rate": _mean_last(success, actual_report_window),
        "partial_success_rate": _mean(partial),
        "final_window_partial_success_rate": _mean_last(partial, actual_report_window),
        "avg_loss": _mean(losses),
        "final_window_loss": _mean_last(losses, actual_report_window),
        "train_config_path": str(train_config_path.resolve()),
        "agent_or_policy_config_path": str(agent_config_path.resolve()),
        "env_config_path": str(env_config_path.resolve()),
        "stdout_log_path": str(command_log_path.resolve()),
        "artifacts_root": str(artifacts_root.resolve()),
        "run_log_dir": str(run_log_dir.resolve()) if run_log_dir.exists() else "",
        "episode_metrics_csv": str(episode_csv_path.resolve()),
        "report_window_metrics_csv": str(report_csv_path.resolve()),
    }
    _save_json(artifacts_root / "results" / "summary.json", summary)
    return summary


def _summarize_infotaxis_run(
    algorithm_key: str,
    algorithm_name: str,
    *,
    suite_tag: str,
    report_window: int,
    train_config_path: Path,
    policy_config_path: Path,
    env_config_path: Path,
    command_log_path: Path,
    artifacts_root: Path,
) -> Dict[str, Any]:
    train_cfg = _load_yaml(train_config_path)
    log_dir = Path(train_cfg.get("output", {}).get("log_dir", artifacts_root / "logs"))
    run_dir = log_dir / "infotaxis_baseline"
    summary_payload = _load_json(run_dir / "summary.json")
    episodes_payload = _load_json(run_dir / "episode_stats.json")
    episodes = episodes_payload.get("episodes", []) if isinstance(episodes_payload, dict) else []

    rewards = [float(item.get("team_reward", 0.0)) for item in episodes]
    lengths = [float(item.get("episode_length", 0.0)) for item in episodes]
    found_sources = [float(item.get("found_sources", 0.0)) for item in episodes]
    found_ratios = [float(item.get("found_ratio", 0.0)) for item in episodes]
    success = [1.0 if item.get("success", False) else 0.0 for item in episodes]
    partial = [1.0 if item.get("partial_success", False) else 0.0 for item in episodes]

    actual_report_window = _resolve_report_window(len(episodes), report_window)
    episode_rows = [
        {
            "episode": int(item.get("episode", idx + 1)),
            "episode_reward": float(item.get("team_reward", 0.0)),
            "episode_length": float(item.get("episode_length", 0.0)),
            "found_sources": float(item.get("found_sources", 0.0)),
            "found_ratio": float(item.get("found_ratio", 0.0)),
            "success": int(bool(item.get("success", False))),
            "partial_success": int(bool(item.get("partial_success", False))),
            "loss": 0.0,
        }
        for idx, item in enumerate(episodes)
    ]

    episode_csv_path = artifacts_root / "results" / "episode_metrics.csv"
    report_csv_path = artifacts_root / "results" / "report_window_metrics.csv"
    _write_episode_csv(episode_csv_path, episode_rows)
    _write_report_window_csv(report_csv_path, episode_rows, actual_report_window)

    summary = {
        "suite_tag": suite_tag,
        "algorithm_key": algorithm_key,
        "algorithm_name": algorithm_name,
        "run_type": "rollout",
        "num_episodes": int(summary_payload.get("num_episodes", len(episodes))),
        "report_window": actual_report_window,
        "report_reward": _mean_last(rewards, actual_report_window),
        "report_found_ratio": _mean_last(found_ratios, actual_report_window),
        "report_success_rate": _mean_last(success, actual_report_window),
        "report_partial_success_rate": _mean_last(partial, actual_report_window),
        "report_loss": 0.0,
        "avg_episode_reward": float(summary_payload.get("avg_team_reward", _mean(rewards))),
        "final_window_reward": _mean_last(rewards, actual_report_window),
        "best_episode_reward": _max_value(rewards),
        "avg_episode_length": float(summary_payload.get("avg_episode_length", _mean(lengths))),
        "avg_found_sources": float(summary_payload.get("avg_found_sources", _mean(found_sources))),
        "avg_found_ratio": float(summary_payload.get("avg_found_ratio", _mean(found_ratios))),
        "final_window_found_ratio": _mean_last(found_ratios, actual_report_window),
        "best_found_ratio": _max_value(found_ratios),
        "success_rate": float(summary_payload.get("success_rate", _mean(success))),
        "final_window_success_rate": _mean_last(success, actual_report_window),
        "partial_success_rate": float(summary_payload.get("partial_success_rate", _mean(partial))),
        "final_window_partial_success_rate": _mean_last(partial, actual_report_window),
        "avg_loss": 0.0,
        "final_window_loss": 0.0,
        "train_config_path": str(train_config_path.resolve()),
        "agent_or_policy_config_path": str(policy_config_path.resolve()),
        "env_config_path": str(env_config_path.resolve()),
        "stdout_log_path": str(command_log_path.resolve()),
        "artifacts_root": str(artifacts_root.resolve()),
        "run_log_dir": str(run_dir.resolve()) if run_dir.exists() else "",
        "episode_metrics_csv": str(episode_csv_path.resolve()),
        "report_window_metrics_csv": str(report_csv_path.resolve()),
    }
    _save_json(artifacts_root / "results" / "summary.json", summary)
    return summary


def _execute_job(job: Dict[str, Any], report_window: int) -> Dict[str, Any]:
    return_code, duration_sec = _run_command(job["command"], job["stdout_log_path"])
    row = {
        "suite_tag": job["suite_tag"],
        "algorithm_key": job["algorithm_key"],
        "algorithm_name": job["algorithm_name"],
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
        "duration_sec": round(duration_sec, 3),
        "run_type": job["run_type"],
        "train_config_path": str(job["train_config_path"].resolve()),
        "agent_or_policy_config_path": str(job["agent_or_policy_config_path"].resolve()),
        "env_config_path": str(job["env_config_path"].resolve()),
        "stdout_log_path": str(job["stdout_log_path"].resolve()),
        "artifacts_root": str(job["artifacts_root"].resolve()),
    }
    if return_code != 0:
        return row

    if job["run_type"] == "rollout":
        row.update(
            _summarize_infotaxis_run(
                job["algorithm_key"],
                job["algorithm_name"],
                suite_tag=job["suite_tag"],
                report_window=report_window,
                train_config_path=job["train_config_path"],
                policy_config_path=job["agent_or_policy_config_path"],
                env_config_path=job["env_config_path"],
                command_log_path=job["stdout_log_path"],
                artifacts_root=job["artifacts_root"],
            )
        )
    else:
        row.update(
            _summarize_training_run(
                job["algorithm_key"],
                job["algorithm_name"],
                suite_tag=job["suite_tag"],
                report_window=report_window,
                train_config_path=job["train_config_path"],
                agent_config_path=job["agent_or_policy_config_path"],
                env_config_path=job["env_config_path"],
                command_log_path=job["stdout_log_path"],
                artifacts_root=job["artifacts_root"],
            )
        )
    return row


def _write_suite_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite_tag",
        "algorithm_key",
        "algorithm_name",
        "status",
        "return_code",
        "duration_sec",
        "run_type",
        "num_episodes",
        "report_window",
        "report_reward",
        "report_found_ratio",
        "report_success_rate",
        "report_partial_success_rate",
        "report_loss",
        "avg_episode_reward",
        "final_window_reward",
        "best_episode_reward",
        "avg_episode_length",
        "avg_found_sources",
        "avg_found_ratio",
        "final_window_found_ratio",
        "best_found_ratio",
        "success_rate",
        "final_window_success_rate",
        "partial_success_rate",
        "final_window_partial_success_rate",
        "avg_loss",
        "final_window_loss",
        "train_config_path",
        "agent_or_policy_config_path",
        "env_config_path",
        "stdout_log_path",
        "artifacts_root",
        "run_log_dir",
        "episode_metrics_csv",
        "report_window_metrics_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a unified comparison suite over baseline algorithms.")
    parser.add_argument("--suite-config", type=str, default=str(DEFAULT_SUITE_CONFIG_PATH), help="Path to the suite YAML config file.")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Algorithm keys to run. Default comes from suite config.")
    parser.add_argument("--field-dir", type=str, default=None, help="Override field directory.")
    parser.add_argument("--shared-env-config", type=str, default=None, help="Optional shared environment override YAML applied to every algorithm.")
    parser.add_argument("--episodes", type=int, default=None, help="Override training.num_episodes for all algorithms.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps_per_episode for all algorithms.")
    parser.add_argument("--final-window", type=int, default=None, help="Use the last N episodes as the reported final result window.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of algorithms to run in parallel. Default comes from suite config or min(num_algorithms, 4).")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for all algorithms.")
    parser.add_argument("--suite-tag", type=str, default=None, help="Name of the output suite directory.")
    parser.add_argument("--output-root", type=str, default=None, help="Override output root directory.")
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true", help="Continue to the next algorithm when one run fails.")
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false", help="Stop immediately when one algorithm fails.")
    parser.set_defaults(continue_on_error=None)
    args = parser.parse_args()

    suite_config_path = _resolve_repo_path(args.suite_config) or DEFAULT_SUITE_CONFIG_PATH
    suite_cfg = _load_yaml(suite_config_path)

    algorithms = _normalize_requested_algorithms(
        _pick_config_value(args.algorithms, suite_cfg, "algorithms", ["all"])
    )
    field_dir = _resolve_repo_path(
        _pick_config_value(args.field_dir, suite_cfg, "field_dir", str(DEFAULT_FIELD_DIR))
    ) or DEFAULT_FIELD_DIR
    shared_env_config_path = _resolve_repo_path(
        _pick_config_value(args.shared_env_config, suite_cfg, "shared_env_config", None)
    )
    episodes = _pick_config_value(args.episodes, suite_cfg, "episodes", None)
    max_steps = _pick_config_value(args.max_steps, suite_cfg, "max_steps", None)
    report_window = max(1, int(_pick_config_value(args.final_window, suite_cfg, "final_window", 200)))
    max_workers_arg = _pick_config_value(args.max_workers, suite_cfg, "max_workers", None)
    seed = _pick_config_value(args.seed, suite_cfg, "seed", None)
    continue_on_error = bool(_pick_config_value(args.continue_on_error, suite_cfg, "continue_on_error", False))
    suite_tag = _build_suite_tag(_pick_config_value(args.suite_tag, suite_cfg, "suite_tag", None))
    output_root = _resolve_repo_path(
        _pick_config_value(args.output_root, suite_cfg, "output_root", str(DEFAULT_OUTPUT_ROOT))
    ) or DEFAULT_OUTPUT_ROOT

    suite_root = Path(output_root) / suite_tag
    generated_root = suite_root / "cfg"
    console_root = suite_root / "stdout"
    results_root = suite_root / "res"

    if not field_dir.exists():
        raise FileNotFoundError(f"Field directory not found: {field_dir}")
    if shared_env_config_path is not None and not shared_env_config_path.exists():
        raise FileNotFoundError(f"Shared environment config not found: {shared_env_config_path}")

    shared_env_override = _extract_env_config(_load_yaml(shared_env_config_path))
    suite_manifest = {
        "suite_tag": suite_tag,
        "suite_config_path": str(suite_config_path.resolve()),
        "algorithms": algorithms,
        "field_dir": str(field_dir.resolve()),
        "shared_env_config": str(shared_env_config_path.resolve()) if shared_env_config_path is not None else "",
        "episodes_override": episodes,
        "max_steps_override": max_steps,
        "report_window": report_window,
        "max_workers": max_workers_arg,
        "seed_override": seed,
        "continue_on_error": continue_on_error,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_json(results_root / "suite_manifest.json", suite_manifest)

    jobs: List[Dict[str, Any]] = []
    for job_index, algorithm_key in enumerate(algorithms):
        spec = ALGORITHM_REGISTRY[algorithm_key]
        algorithm_name = str(spec["name"])

        generated_dir = generated_root / algorithm_key
        artifacts_root = suite_root / "a" / algorithm_key
        log_dir = artifacts_root / "l"
        save_dir = artifacts_root / "c"
        console_log_path = console_root / f"{algorithm_key}.log"

        env_cfg = _load_yaml(Path(spec["env_profile"]))
        env_cfg = _deep_update(env_cfg, shared_env_override)
        env_cfg_path = generated_dir / "env_config.yaml"
        _save_yaml(env_cfg_path, env_cfg)

        if "runner" in spec:
            train_cfg = _prepare_infotaxis_train_payload(
                _load_yaml(Path(spec["train_profile"])),
                log_dir=log_dir,
                seed=seed,
                episodes=episodes,
                max_steps=max_steps,
            )
            policy_cfg = _load_yaml(Path(spec["policy_profile"]))
            train_cfg_path = generated_dir / "train_config.yaml"
            policy_cfg_path = generated_dir / "policy_config.yaml"
            _save_yaml(train_cfg_path, train_cfg)
            _save_yaml(policy_cfg_path, policy_cfg)
            command = [
                sys.executable,
                str(Path(spec["runner"]).resolve()),
                "--train-config",
                str(train_cfg_path.resolve()),
                "--policy-config",
                str(policy_cfg_path.resolve()),
                "--env-config",
                str(env_cfg_path.resolve()),
                "--field-dir",
                str(field_dir.resolve()),
            ]
            jobs.append(
                {
                    "job_index": job_index,
                    "suite_tag": suite_tag,
                    "algorithm_key": algorithm_key,
                    "algorithm_name": algorithm_name,
                    "run_type": "rollout",
                    "command": command,
                    "train_config_path": train_cfg_path,
                    "agent_or_policy_config_path": policy_cfg_path,
                    "env_config_path": env_cfg_path,
                    "stdout_log_path": console_log_path,
                    "artifacts_root": artifacts_root,
                }
            )
        else:
            train_cfg = _prepare_train_payload(
                _load_yaml(Path(spec["train_profile"])),
                log_dir=log_dir,
                save_dir=save_dir,
                seed=seed,
                episodes=episodes,
                max_steps=max_steps,
            )
            agent_cfg = _load_yaml(Path(spec["agent_profile"]))
            train_cfg_path = generated_dir / "train_config.yaml"
            agent_cfg_path = generated_dir / "agent_config.yaml"
            _save_yaml(train_cfg_path, train_cfg)
            _save_yaml(agent_cfg_path, agent_cfg)
            command = [
                sys.executable,
                str(TRAIN_SCRIPT.resolve()),
                "--train-config",
                str(train_cfg_path.resolve()),
                "--agent-config",
                str(agent_cfg_path.resolve()),
                "--env-config",
                str(env_cfg_path.resolve()),
                "--field-dir",
                str(field_dir.resolve()),
            ]
            jobs.append(
                {
                    "job_index": job_index,
                    "suite_tag": suite_tag,
                    "algorithm_key": algorithm_key,
                    "algorithm_name": algorithm_name,
                    "run_type": "train",
                    "command": command,
                    "train_config_path": train_cfg_path,
                    "agent_or_policy_config_path": agent_cfg_path,
                    "env_config_path": env_cfg_path,
                    "stdout_log_path": console_log_path,
                    "artifacts_root": artifacts_root,
                }
            )

    max_workers = int(max_workers_arg) if max_workers_arg is not None else min(len(jobs), 4)
    max_workers = max(1, max_workers)
    print(
        f"[suite] Launching {len(jobs)} algorithms in parallel | "
        f"max_workers={max_workers} | report_window={report_window}"
    )
    for job in jobs:
        print(f"[suite] queued {job['algorithm_key']} ({job['algorithm_name']})")

    results_by_index: Dict[int, Dict[str, Any]] = {}
    failure_return_code = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(_execute_job, job, report_window): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "suite_tag": suite_tag,
                    "algorithm_key": job["algorithm_key"],
                    "algorithm_name": job["algorithm_name"],
                    "status": "failed",
                    "return_code": 1,
                    "duration_sec": 0.0,
                    "run_type": job["run_type"],
                    "train_config_path": str(job["train_config_path"].resolve()),
                    "agent_or_policy_config_path": str(job["agent_or_policy_config_path"].resolve()),
                    "env_config_path": str(job["env_config_path"].resolve()),
                    "stdout_log_path": str(job["stdout_log_path"].resolve()),
                    "artifacts_root": str(job["artifacts_root"].resolve()),
                    "error": repr(exc),
                }

            results_by_index[int(job["job_index"])] = row
            ordered_rows = [results_by_index[idx] for idx in sorted(results_by_index)]
            _save_json(results_root / "suite_results.json", {"results": ordered_rows})
            _write_suite_csv(results_root / "suite_results.csv", ordered_rows)

            if row["status"] != "ok" and failure_return_code == 0:
                failure_return_code = int(row.get("return_code", 1) or 1)

            print(
                f"[suite] {row['algorithm_key']} finished | "
                f"status={row['status']} | duration={float(row.get('duration_sec', 0.0)):.1f}s"
            )

    if failure_return_code != 0 and not continue_on_error:
        raise SystemExit(failure_return_code)

    print(f"[suite] Results saved to {results_root.resolve()}")


if __name__ == "__main__":
    main()
