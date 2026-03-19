"""
Run the Infotaxis baseline on the plume-search environment.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

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

from marl_leakage_search.comparison_algorithms.infotaxis import (
    InfotaxisController,
    run_infotaxis_episode,
)
from marl_leakage_search.envs.marl_env import PlumeEnv


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "training": {
        "num_episodes": 100,
        "max_steps_per_episode": 500,
        "log_interval": 10,
    },
    "baseline": {
        "algorithm": "infotaxis",
    },
    "environment": {
        "num_agents": 2,
        "action_dim": 8,
        "source_find_radius": 5.0,
        "collision_penalty": 1.0,
        "battery_penalty": 0.01,
        "found_source_bonus": 5.0,
        "done_bonus": 0.0,
        "success_done_bonus": 20.0,
        "failure_done_penalty": 0.0,
        "enable_collision": True,
        "stop_on_collision": True,
        "observe_wind": False,
        "observe_velocity": False,
        "distance_reward_scale": 0.0,
        "found_source_concentration_scale": 1.0,
        "found_source_stay_penalty": 0.0,
        "found_source_stay_radius_scale": 1.0,
        "communication": {
            "enabled": False,
            "top_k": 2,
            "radius": 0.0,
            "include_concentration": True,
            "include_battery": True,
            "include_velocity": False,
            "normalize_relative": True,
            "channel": {
                "enabled": False,
                "mode": "none",
                "nlos_gain": 0.35,
                "nlos_noise_std": 0.0,
                "los_drop_prob": 0.0,
                "nlos_drop_prob": 0.0,
                "add_link_flag": False,
            },
        },
        "uav_params": {},
        "init_pos_mode": "random",
        "dynamic_field": {
            "enabled": False,
            "dt": 1.0,
            "keep_plume_behind_obstacle": True,
            "wind": {
                "enabled": False,
                "speed_amplitude": 0.0,
                "speed_frequency": 0.0,
                "speed_phase": 0.0,
                "dir_amplitude": 0.0,
                "dir_frequency": 0.0,
                "dir_phase": 0.0,
            },
            "vortex": {
                "use_strouhal": False,
                "strouhal": 0.2,
                "min_wind_speed": 0.001,
                "min_diameter": 0.001,
            },
        },
    },
    "output": {
        "log_dir": "./logs",
    },
    "seed": 42,
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _extract_environment_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    if "environment" in raw:
        env_cfg = raw.get("environment", {})
        if env_cfg is None:
            return {}
        if not isinstance(env_cfg, dict):
            raise TypeError("environment must be a mapping/dict in env_config.yaml")
        return {"environment": env_cfg}

    known_top_level_keys = {"training", "marl", "agent", "output", "baseline", "seed"}
    if any(key in raw for key in known_top_level_keys):
        return {}
    return {"environment": raw}


def _resolve_profile_env_config(
    train_config_path: str,
    env_config_path: str | None,
) -> Path | None:
    if env_config_path:
        return Path(env_config_path)

    train_cfg_path = Path(train_config_path)
    if not train_cfg_path.exists():
        return None

    candidates = sorted(
        list(train_cfg_path.parent.glob("env_config*.yaml"))
        + list(train_cfg_path.parent.glob("env_config*.yml"))
    )
    if len(candidates) == 1:
        return candidates[0]
    return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_env(config: Dict[str, Any], field_dir: str, seed: int) -> PlumeEnv:
    env_cfg = config["environment"]
    uav_params = env_cfg.get("uav_params", {})
    if not isinstance(uav_params, dict):
        raise TypeError("environment.uav_params must be a mapping/dict")

    return PlumeEnv(
        field_dir=field_dir,
        num_agents=int(env_cfg.get("num_agents", 2)),
        source_find_radius=float(env_cfg.get("source_find_radius", 5.0)),
        collision_penalty=float(env_cfg.get("collision_penalty", 1.0)),
        battery_penalty=float(env_cfg.get("battery_penalty", 0.01)),
        found_source_bonus=float(env_cfg.get("found_source_bonus", 5.0)),
        done_bonus=float(env_cfg.get("done_bonus", 0.0)),
        success_done_bonus=(
            None
            if env_cfg.get("success_done_bonus", None) is None
            else float(env_cfg.get("success_done_bonus"))
        ),
        failure_done_penalty=float(env_cfg.get("failure_done_penalty", 0.0)),
        enable_collision=bool(env_cfg.get("enable_collision", True)),
        stop_on_collision=bool(env_cfg.get("stop_on_collision", True)),
        observe_wind=bool(env_cfg.get("observe_wind", False)),
        observe_velocity=bool(env_cfg.get("observe_velocity", False)),
        distance_reward_scale=float(env_cfg.get("distance_reward_scale", 0.0)),
        found_source_concentration_scale=float(env_cfg.get("found_source_concentration_scale", 1.0)),
        found_source_stay_penalty=float(env_cfg.get("found_source_stay_penalty", 0.0)),
        found_source_stay_radius_scale=float(env_cfg.get("found_source_stay_radius_scale", 1.0)),
        init_pos_mode=str(env_cfg.get("init_pos_mode", "random")),
        seed=seed,
        uav_params=uav_params,
        dynamic_field_config=env_cfg.get("dynamic_field", {}),
        communication_config=env_cfg.get("communication", {}),
    )


def _save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def main() -> None:
    default_field_dir = REPO_ROOT / "marl_leakage_search" / "envs" / "generated_fields"
    default_train_cfg = (
        REPO_ROOT / "marl_leakage_search" / "configs" / "profiles" / "05_infotaxis" / "train_config_infotaxis.yaml"
    )
    default_policy_cfg = (
        REPO_ROOT / "marl_leakage_search" / "configs" / "profiles" / "05_infotaxis" / "policy_config_infotaxis.yaml"
    )

    parser = argparse.ArgumentParser(description="Run the Infotaxis baseline.")
    parser.add_argument("--train-config", type=str, default=str(default_train_cfg))
    parser.add_argument("--policy-config", type=str, default=str(default_policy_cfg))
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help="Optional environment override YAML. If omitted, a sibling env_config*.yaml is auto-loaded when available.",
    )
    parser.add_argument("--field-dir", type=str, default=str(default_field_dir))
    args = parser.parse_args()

    resolved_env_config = _resolve_profile_env_config(args.train_config, args.env_config)
    config = json.loads(json.dumps(DEFAULT_TRAIN_CONFIG))
    config = _deep_update(config, _load_yaml(args.train_config))
    config = _deep_update(
        config,
        _extract_environment_overrides(
            _load_yaml(str(resolved_env_config) if resolved_env_config is not None else None)
        ),
    )
    policy_cfg = _load_yaml(args.policy_config)

    seed = int(config.get("seed", 42))
    _set_seed(seed)

    env = _build_env(config, args.field_dir, seed)
    controller = InfotaxisController(policy_cfg)

    training_cfg = config.get("training", {})
    num_episodes = int(training_cfg.get("num_episodes", 100))
    max_steps = int(training_cfg.get("max_steps_per_episode", 500))
    log_interval = max(1, int(training_cfg.get("log_interval", 10)))

    results: list[Dict[str, Any]] = []
    for episode_idx in range(1, num_episodes + 1):
        episode_stats = run_infotaxis_episode(env, controller, max_steps=max_steps)
        episode_stats["episode"] = episode_idx
        results.append(episode_stats)

        if episode_idx % log_interval == 0:
            print(
                f"Episode {episode_idx}/{num_episodes} | "
                f"Team Reward: {episode_stats['team_reward']:.2f} | "
                f"Length: {episode_stats['episode_length']} | "
                f"Found: {episode_stats['found_sources']}/{episode_stats['total_sources']} | "
                f"Success: {int(episode_stats['success'])}"
            )

    avg_reward = float(np.mean([item["team_reward"] for item in results])) if results else 0.0
    avg_length = float(np.mean([item["episode_length"] for item in results])) if results else 0.0
    avg_found_sources = float(np.mean([item["found_sources"] for item in results])) if results else 0.0
    avg_found_ratio = float(np.mean([item["found_ratio"] for item in results])) if results else 0.0
    success_rate = float(np.mean([1.0 if item["success"] else 0.0 for item in results])) if results else 0.0
    partial_success_rate = float(
        np.mean([1.0 if item["partial_success"] else 0.0 for item in results])
    ) if results else 0.0

    output_dir = Path(config.get("output", {}).get("log_dir", "./logs")) / "infotaxis_baseline"
    summary = {
        "algorithm": "infotaxis",
        "seed": seed,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps,
        "avg_team_reward": avg_reward,
        "avg_episode_length": avg_length,
        "avg_found_sources": avg_found_sources,
        "avg_found_ratio": avg_found_ratio,
        "success_rate": success_rate,
        "partial_success_rate": partial_success_rate,
        "train_config_path": str(Path(args.train_config).resolve()),
        "policy_config_path": str(Path(args.policy_config).resolve()),
        "env_config_path": (
            str(resolved_env_config.resolve())
            if resolved_env_config is not None
            else ""
        ),
    }
    _save_json(summary, output_dir / "summary.json")
    _save_json({"episodes": results}, output_dir / "episode_stats.json")


if __name__ == "__main__":
    main()
