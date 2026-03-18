"""
Run one training experiment, export per-episode rewards, and draw a convergence plot.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Run this script in the same environment as train.py."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "marl_leakage_search" / "configs" / "train_config.yaml"
DEFAULT_AGENT_CONFIG = REPO_ROOT / "marl_leakage_search" / "configs" / "agent_config.yaml"
DEFAULT_FIELD_DIR = REPO_ROOT / "marl_leakage_search" / "envs" / "generated_fields"

# Zero-arg execution uses this preset so the script can be run directly.
COMMON_PRESET_NAME = "paper_ffnn_auxon"
COMMON_TRAIN_OVERRIDES: Dict[str, Any] = {
    "training": {
        "num_episodes": 3000,
        "max_steps_per_episode": 1000,
        "save_interval": 100,
        "log_interval": 10,
        "pretrained_dir": "",
    },
    "marl": {
        "algorithm": "mappo",
        "mappo": {
            "gamma": 0.90,
            "gae_lambda": 0.95,
        },
    },
    "environment": {
        "num_agents": 4,
        "observe_wind": True,
        "observe_velocity": True,
        "communication": {
            "enabled": True,
            "top_k": 2,
            "channel": {
                "enabled": True,
                "mode": "los_nlos",
            },
        },
        "dynamic_field": {
            "enabled": True,
        },
    },
    "seed": 42,
}
COMMON_AGENT_OVERRIDES: Dict[str, Any] = {
    "algorithm": "ppo",
    "network": {
        "type": "ffnn",
    },
    "learning": {
        "lr": 1e-4,
        "gamma": 0.90,
        "batch_size": 64,
        "ppo": {
            "aux": {
                "enabled": True,
                "weight": 0.1,
            }
        },
    },
}

# Minimal defaults needed to match train.py's run-tag generation.
FILE_TAG_DEFAULTS: Dict[str, Any] = {
    "training": {
        "num_episodes": 200,
        "max_steps_per_episode": 500,
        "save_interval": 50,
        "log_interval": 10,
        "pretrained_dir": "",
    },
    "marl": {
        "algorithm": "mappo",
        "mappo": {"gamma": 0.99},
        "qmix": {"gamma": 0.99},
    },
    "environment": {
        "num_agents": 2,
        "communication": {
            "enabled": False,
            "top_k": 2,
            "channel": {
                "enabled": False,
                "mode": "none",
            },
        },
    },
    "agent": {
        "algorithm": "ppo",
        "network_type": "ffnn",
        "network": {"type": "ffnn"},
        "learning": {
            "lr": 3e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "ppo": {
                "aux": {
                    "enabled": False,
                    "weight": 0.0,
                }
            },
        },
    },
    "output": {
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
    },
    "seed": 42,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _build_file_tag(cfg: Dict[str, Any]) -> str:
    seed = int(cfg.get("seed", 42))

    marl_cfg = cfg.get("marl", {})
    marl_algorithm = str(marl_cfg.get("algorithm", "mappo")).lower()

    agent_cfg = cfg.get("agent", {})
    agent_algorithm = str(agent_cfg.get("algorithm", "ppo")).lower()
    network_type = str(
        agent_cfg.get("network", {}).get("type", agent_cfg.get("network_type", "ffnn"))
    ).lower()

    learning_cfg = agent_cfg.get("learning", {})
    trainer_cfg = marl_cfg.get(marl_algorithm, {}) if isinstance(marl_cfg, dict) else {}

    lr_value = float(learning_cfg.get("lr", 0.0))
    agent_gamma_value = float(learning_cfg.get("gamma", 0.0))
    gamma_value = float(trainer_cfg.get("gamma", agent_gamma_value))
    batch_size = int(learning_cfg.get("batch_size", 0))

    ppo_cfg = learning_cfg.get("ppo", {})
    aux_enabled = bool(ppo_cfg.get("aux", {}).get("enabled", False))
    aux_tag = "auxon" if aux_enabled else "auxoff"

    env_cfg = cfg.get("environment", {})
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

    return (
        f"seed{seed}_agent{agent_algorithm}_net{network_type}_marl{marl_algorithm}_na{num_agents}"
        f"_{comm_tag}_{channel_tag}_lr{lr_value:.6f}_gamma{gamma_value:.4f}_bs{batch_size}_{aux_tag}"
    )


def _apply_overrides(train_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = copy.deepcopy(train_cfg)
    training_cfg = updated.setdefault("training", {})

    if args.episodes is not None:
        training_cfg["num_episodes"] = int(args.episodes)
    if args.max_steps is not None:
        training_cfg["max_steps_per_episode"] = int(args.max_steps)
    if args.save_interval is not None:
        training_cfg["save_interval"] = int(args.save_interval)
    if args.log_interval is not None:
        training_cfg["log_interval"] = int(args.log_interval)
    if args.seed is not None:
        updated["seed"] = int(args.seed)
    if args.pretrained_dir is not None:
        training_cfg["pretrained_dir"] = str(args.pretrained_dir)

    return updated


def _apply_preset(
    train_cfg: Dict[str, Any],
    agent_cfg: Dict[str, Any],
    preset_name: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    train_updated = copy.deepcopy(train_cfg)
    agent_updated = copy.deepcopy(agent_cfg)

    if preset_name == "common":
        _deep_update(train_updated, copy.deepcopy(COMMON_TRAIN_OVERRIDES))
        _deep_update(agent_updated, copy.deepcopy(COMMON_AGENT_OVERRIDES))
        return train_updated, agent_updated

    if preset_name == "config":
        return train_updated, agent_updated

    raise ValueError(f"Unsupported preset: {preset_name}")


def _merge_effective_config(train_cfg: Dict[str, Any], agent_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(FILE_TAG_DEFAULTS)
    _deep_update(merged, copy.deepcopy(train_cfg))
    _deep_update(merged, {"agent": copy.deepcopy(agent_cfg)})
    return merged


def _find_latest_stats(log_dir: Path) -> Path:
    candidates = sorted(
        log_dir.glob("*/training_stats.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No training_stats.json found under {log_dir}")
    return candidates[0]


def _find_recent_stats(log_dir: Path, started_at: float) -> Path:
    candidates = sorted(
        (
            item
            for item in log_dir.glob("*/training_stats.json")
            if item.stat().st_mtime >= started_at - 1.0
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return _find_latest_stats(log_dir)


def _normalize_series(
    values: Sequence[Any] | None,
    target_len: int,
    *,
    cast,
    default: Any,
) -> List[Any]:
    if values is None:
        return [default for _ in range(target_len)]
    result = [cast(value) for value in list(values)[:target_len]]
    if len(result) < target_len:
        result.extend(default for _ in range(target_len - len(result)))
    return result


def _load_training_stats(stats_path: Path) -> Dict[str, Any]:
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    rewards = [float(value) for value in payload.get("episode_rewards", [])]
    if not rewards:
        raise ValueError(f"No episode_rewards found in {stats_path}")

    episode_count = len(rewards)
    reward_components_raw = payload.get("reward_components", {})
    if not isinstance(reward_components_raw, dict):
        reward_components_raw = {}

    reward_components = {
        str(key): _normalize_series(values, episode_count, cast=float, default=0.0)
        for key, values in reward_components_raw.items()
    }

    return {
        "episode_rewards": rewards,
        "episode_lengths": _normalize_series(
            payload.get("episode_lengths"), episode_count, cast=int, default=0
        ),
        "found_sources": _normalize_series(
            payload.get("found_sources"), episode_count, cast=int, default=0
        ),
        "total_sources": _normalize_series(
            payload.get("total_sources"), episode_count, cast=int, default=0
        ),
        "found_source_ratios": _normalize_series(
            payload.get("found_source_ratios"), episode_count, cast=float, default=0.0
        ),
        "success_episodes": _normalize_series(
            payload.get("success_episodes"), episode_count, cast=int, default=0
        ),
        "partial_success_episodes": _normalize_series(
            payload.get("partial_success_episodes"), episode_count, cast=int, default=0
        ),
        "losses": _normalize_series(payload.get("losses"), episode_count, cast=float, default=0.0),
        "reward_components": reward_components,
    }


def _write_episode_reward_csv(stats: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    component_keys = sorted(stats["reward_components"].keys())

    header = [
        "episode",
        "episode_reward",
        "episode_length",
        "loss",
        "found_sources",
        "total_sources",
        "found_source_ratio",
        "success",
        "partial_success",
    ]
    header.extend(component_keys)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for index, reward in enumerate(stats["episode_rewards"], start=1):
            row = [
                index,
                f"{float(reward):.6f}",
                stats["episode_lengths"][index - 1],
                f"{float(stats['losses'][index - 1]):.6f}",
                stats["found_sources"][index - 1],
                stats["total_sources"][index - 1],
                f"{float(stats['found_source_ratios'][index - 1]):.6f}",
                stats["success_episodes"][index - 1],
                stats["partial_success_episodes"][index - 1],
            ]
            row.extend(
                f"{float(stats['reward_components'][key][index - 1]):.6f}" for key in component_keys
            )
            writer.writerow(row)


def _moving_average(values: Sequence[float], window: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr, 1
    used_window = max(1, min(int(window), int(arr.size)))
    if used_window == 1:
        return arr, 1
    kernel = np.ones(used_window, dtype=float) / float(used_window)
    return np.convolve(arr, kernel, mode="valid"), used_window


def _line_slope(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _build_summary(
    stats_path: Path,
    csv_path: Path,
    plot_path: Path,
    stats: Dict[str, Any],
    smooth_window: int,
    tail_window: int,
) -> Dict[str, Any]:
    rewards = np.asarray(stats["episode_rewards"], dtype=float)
    episodes = np.arange(1, rewards.size + 1, dtype=int)
    smooth_rewards, used_window = _moving_average(rewards, smooth_window)
    smooth_episodes = episodes[used_window - 1 :] if smooth_rewards.size else episodes

    tail = max(1, min(int(tail_window), int(rewards.size)))
    tail_rewards = rewards[-tail:]
    best_index = int(np.argmax(rewards))

    summary = {
        "stats_path": str(stats_path.resolve()),
        "episode_reward_csv": str(csv_path.resolve()),
        "reward_plot": str(plot_path.resolve()),
        "num_episodes": int(rewards.size),
        "final_reward": float(rewards[-1]),
        "best_reward": float(rewards[best_index]),
        "best_reward_episode": int(episodes[best_index]),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "tail_window": int(tail),
        "tail_mean_reward": float(np.mean(tail_rewards)),
        "tail_std_reward": float(np.std(tail_rewards)),
        "tail_slope": _line_slope(tail_rewards),
        "smooth_window": int(used_window),
        "final_smooth_reward": float(smooth_rewards[-1]) if smooth_rewards.size else float(rewards[-1]),
        "smooth_tail_slope": _line_slope(
            smooth_rewards[-tail:] if smooth_rewards.size else tail_rewards
        ),
        "final_found_ratio": float(stats["found_source_ratios"][-1]),
        "tail_success_rate": float(np.mean(stats["success_episodes"][-tail:])),
        "tail_partial_success_rate": float(np.mean(stats["partial_success_episodes"][-tail:])),
    }

    if smooth_rewards.size:
        best_smooth_index = int(np.argmax(smooth_rewards))
        summary["best_smooth_reward"] = float(smooth_rewards[best_smooth_index])
        summary["best_smooth_episode"] = int(smooth_episodes[best_smooth_index])

    return summary


def _polyline_points(xs: Iterable[float], ys: Iterable[float]) -> str:
    return " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in zip(xs, ys))


def _plot_reward_svg(
    *,
    episodes: Sequence[int],
    rewards: Sequence[float],
    smooth_rewards: Sequence[float],
    smooth_episodes: Sequence[int],
    tail_mean: float,
    best_episode: int,
    best_reward: float,
    title: str,
    output_path: Path,
) -> Path:
    width = 1200
    height = 700
    left = 80
    right = 30
    top = 50
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_vals = [float(x) for x in episodes]
    y_vals = [float(y) for y in rewards]
    if len(smooth_rewards) > 0:
        y_vals.extend(float(y) for y in smooth_rewards)
    y_vals.append(float(tail_mean))
    y_vals.append(float(best_reward))

    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    grid_lines: List[str] = []
    labels: List[str] = []
    for idx in range(6):
        x = left + idx * plot_w / 5
        y = top + idx * plot_h / 5
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        labels.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#4b5563">'
            f"{int(round(x_min + idx * (x_max - x_min) / 5))}</text>"
        )
        labels.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#4b5563">'
            f"{y_max - idx * (y_max - y_min) / 5:.2f}</text>"
        )

    raw_points = _polyline_points((sx(x) for x in episodes), (sy(y) for y in rewards))
    smooth_points = (
        _polyline_points(
            (sx(x) for x in smooth_episodes),
            (sy(y) for y in smooth_rewards),
        )
        if len(smooth_rewards) > 0
        else ""
    )

    best_x = sx(float(best_episode))
    best_y = sy(float(best_reward))
    tail_y = sy(float(tail_mean))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#ffffff" />
<text x="{width / 2:.2f}" y="28" font-size="20" text-anchor="middle" fill="#111827">{title}</text>
<text x="{width / 2:.2f}" y="50" font-size="12" text-anchor="middle" fill="#6b7280">Raw reward + moving average + tail mean</text>
{''.join(grid_lines)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#9ca3af" stroke-width="1" />
<polyline fill="none" stroke="#9ca3af" stroke-width="1.2" points="{raw_points}" />
<polyline fill="none" stroke="#2563eb" stroke-width="2.4" points="{smooth_points}" />
<line x1="{left}" y1="{tail_y:.2f}" x2="{left + plot_w}" y2="{tail_y:.2f}" stroke="#059669" stroke-width="1.6" stroke-dasharray="8,5" />
<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="4.5" fill="#dc2626" />
<text x="{best_x + 10:.2f}" y="{best_y - 8:.2f}" font-size="12" fill="#dc2626">best: ep {best_episode}, {best_reward:.2f}</text>
<text x="{left + plot_w - 8:.2f}" y="{tail_y - 8:.2f}" text-anchor="end" font-size="12" fill="#059669">tail mean: {tail_mean:.2f}</text>
<text x="{width / 2:.2f}" y="{height - 18}" font-size="13" text-anchor="middle" fill="#111827">Episode</text>
<text x="20" y="{top + plot_h / 2:.2f}" font-size="13" text-anchor="middle" fill="#111827" transform="rotate(-90 20 {top + plot_h / 2:.2f})">Reward</text>
{''.join(labels)}
<g transform="translate({left + 8}, {top + 8})">
<rect x="0" y="0" width="212" height="58" rx="8" fill="#f9fafb" stroke="#d1d5db" stroke-width="1" />
<line x1="12" y1="18" x2="42" y2="18" stroke="#9ca3af" stroke-width="2" />
<text x="50" y="22" font-size="12" fill="#374151">Episode reward</text>
<line x1="12" y1="34" x2="42" y2="34" stroke="#2563eb" stroke-width="2.4" />
<text x="50" y="38" font-size="12" fill="#374151">Moving average</text>
<line x1="12" y1="50" x2="42" y2="50" stroke="#059669" stroke-width="2" stroke-dasharray="8,5" />
<text x="50" y="54" font-size="12" fill="#374151">Tail mean</text>
</g>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def _plot_reward_curve(
    *,
    rewards: Sequence[float],
    smooth_window: int,
    tail_window: int,
    title: str,
    output_path: Path,
    show: bool,
) -> Path:
    reward_array = np.asarray(rewards, dtype=float)
    episodes = np.arange(1, reward_array.size + 1, dtype=int)
    smooth_rewards, used_window = _moving_average(reward_array, smooth_window)
    smooth_episodes = episodes[used_window - 1 :] if smooth_rewards.size else episodes
    tail = max(1, min(int(tail_window), int(reward_array.size)))
    tail_mean = float(np.mean(reward_array[-tail:]))
    best_index = int(np.argmax(reward_array))
    best_episode = int(episodes[best_index])
    best_reward = float(reward_array[best_index])

    try:
        if not show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        svg_output = output_path if output_path.suffix.lower() == ".svg" else output_path.with_suffix(".svg")
        return _plot_reward_svg(
            episodes=episodes,
            rewards=reward_array,
            smooth_rewards=smooth_rewards,
            smooth_episodes=smooth_episodes,
            tail_mean=tail_mean,
            best_episode=best_episode,
            best_reward=best_reward,
            title=title,
            output_path=svg_output,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 6.5), dpi=160)
    axis.plot(
        episodes,
        reward_array,
        color="#9ca3af",
        linewidth=1.2,
        alpha=0.75,
        label="Episode reward",
    )
    axis.plot(
        smooth_episodes,
        smooth_rewards,
        color="#2563eb",
        linewidth=2.3,
        label=f"Moving average ({used_window})",
    )
    axis.axhline(
        tail_mean,
        color="#059669",
        linestyle="--",
        linewidth=1.6,
        label=f"Tail mean ({tail})",
    )
    axis.scatter([best_episode], [best_reward], color="#dc2626", s=35, zorder=4, label="Best episode")
    axis.annotate(
        f"best={best_reward:.2f} @ ep {best_episode}",
        xy=(best_episode, best_reward),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="#dc2626",
    )
    axis.set_title(title)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward")
    axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _run_training(
    *,
    train_config_path: Path,
    agent_config_path: Path,
    field_dir: Path,
    pretrained_dir: str | None,
) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--train-config",
        str(train_config_path),
        "--agent-config",
        str(agent_config_path),
        "--field-dir",
        str(field_dir),
    ]
    if pretrained_dir:
        command.extend(["--pretrained-dir", str(pretrained_dir)])

    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one experiment, export per-episode reward CSV, and draw a convergence plot. "
            "Running with no arguments uses the built-in common preset."
        )
    )
    parser.add_argument("--train-config", type=str, default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--agent-config", type=str, default=str(DEFAULT_AGENT_CONFIG))
    parser.add_argument("--field-dir", type=str, default=str(DEFAULT_FIELD_DIR))
    parser.add_argument(
        "--preset",
        type=str,
        choices=["common", "config"],
        default="common",
        help="`common` uses the built-in direct-run preset; `config` uses the yaml files as-is.",
    )
    parser.add_argument("--stats-path", type=str, default=None, help="Analyze an existing training_stats.json.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for CSV/plot/summary outputs.")
    parser.add_argument("--pretrained-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument("--tail-window", type=int, default=100)
    parser.add_argument("--plot-name", type=str, default="episode_reward_curve.png")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only analyze stats.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    args = parser.parse_args()

    if args.stats_path and not args.skip_train:
        parser.error("--stats-path can only be used together with --skip-train.")

    train_cfg_path = _resolve_repo_path(args.train_config)
    agent_cfg_path = _resolve_repo_path(args.agent_config)
    field_dir = _resolve_repo_path(args.field_dir)

    base_train_cfg = _load_yaml(train_cfg_path)
    base_agent_cfg = _load_yaml(agent_cfg_path)
    train_cfg, agent_cfg = _apply_preset(base_train_cfg, base_agent_cfg, args.preset)
    train_cfg = _apply_overrides(train_cfg, args)
    effective_cfg = _merge_effective_config(train_cfg, agent_cfg)
    file_tag = _build_file_tag(effective_cfg)

    log_dir = _resolve_repo_path(effective_cfg.get("output", {}).get("log_dir", "./logs"))
    expected_stats_path = log_dir / file_tag / "training_stats.json"

    output_dir_override = _resolve_repo_path(args.output_dir) if args.output_dir else None
    if args.skip_train:
        if args.stats_path:
            stats_path = _resolve_repo_path(args.stats_path)
        elif expected_stats_path.exists():
            stats_path = expected_stats_path
        else:
            stats_path = _find_latest_stats(log_dir)
        analysis_dir = output_dir_override if output_dir_override else stats_path.parent / "reward_analysis"
    else:
        stats_path = expected_stats_path
        analysis_dir = output_dir_override if output_dir_override else expected_stats_path.parent / "reward_analysis"

    analysis_dir.mkdir(parents=True, exist_ok=True)

    effective_train_cfg_path = analysis_dir / "effective_train_config.yaml"
    effective_agent_cfg_path = analysis_dir / "effective_agent_config.yaml"
    _save_yaml(effective_train_cfg_path, train_cfg)
    _save_yaml(effective_agent_cfg_path, agent_cfg)

    if not args.skip_train:
        started_at = time.time()
        _run_training(
            train_config_path=effective_train_cfg_path,
            agent_config_path=effective_agent_cfg_path,
            field_dir=field_dir,
            pretrained_dir=args.pretrained_dir,
        )
        if not stats_path.exists():
            stats_path = _find_recent_stats(log_dir, started_at)

    stats = _load_training_stats(stats_path)
    run_tag = stats_path.parent.name if stats_path.parent.name else file_tag

    csv_path = analysis_dir / "episode_reward_history.csv"
    _write_episode_reward_csv(stats, csv_path)

    plot_path = _plot_reward_curve(
        rewards=stats["episode_rewards"],
        smooth_window=max(1, int(args.smooth_window)),
        tail_window=max(1, int(args.tail_window)),
        title=f"Episode Reward Convergence | {run_tag}",
        output_path=analysis_dir / args.plot_name,
        show=not args.no_show,
    )

    summary = _build_summary(
        stats_path=stats_path,
        csv_path=csv_path,
        plot_path=plot_path,
        stats=stats,
        smooth_window=max(1, int(args.smooth_window)),
        tail_window=max(1, int(args.tail_window)),
    )

    summary_path = analysis_dir / "reward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    preset_label = COMMON_PRESET_NAME if args.preset == "common" else "config"
    print(f"Run tag: {run_tag}")
    print(f"Preset: {preset_label}")
    print(f"Training stats: {stats_path}")
    print(f"Episode reward CSV: {csv_path}")
    print(f"Reward plot: {plot_path}")
    print(f"Reward summary: {summary_path}")
    print(
        "Final reward: "
        f"{summary['final_reward']:.3f}, "
        f"tail mean: {summary['tail_mean_reward']:.3f}, "
        f"tail slope: {summary['tail_slope']:.6f}"
    )


if __name__ == "__main__":
    main()
