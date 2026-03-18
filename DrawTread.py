"""
DrawTread.py
Plot one reward curve or a built-in set of recommended reward curves.

Examples
--------
python DrawTread.py --seed 42 --agent-algorithm ppo --network-type ffnn --marl-algorithm mappo --lr 0.0001 --gamma 0.9 --batch-size 64
python DrawTread.py --preset recommended_two --output debug_renders/recommended_two.svg --no-show
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = REPO_ROOT / "marl_leakage_search" / "experiments" / "Train_network"
DEFAULT_PRESET_MAX_EPISODE = 3000
DEFAULT_PRESET_RESAMPLE_STEP = 1
SVG_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
)


@dataclass(frozen=True)
class CurveSpec:
    csv_path: Path
    label: str
    smooth_window: int | None = None


@dataclass(frozen=True)
class CurveSeries:
    label: str
    episodes: np.ndarray
    rewards: np.ndarray
    used_window: int
    source_stem: str


PRESET_CURVES: dict[str, tuple[CurveSpec, ...]] = {
    "recommended_two": (
        CurveSpec(
            EXPERIMENT_DIR
            / "seed42_agentppo_netffnn_marlmappo_na4_commk2_chlos_nlos_lr0.000100_gamma0.9000_bs64_auxon_avg_reward_trend.csv",
            "Paper setting (seed42, FFNN, aux on)",
        ),
        CurveSpec(
            EXPERIMENT_DIR
            / "seed1024_agentppo_netffnn_marlmappo_na4_lr0.000100_gamma0.9000_bs64_auxoff_avg_reward_trend.csv",
            "Best overall (seed1024, FFNN, aux off)",
        ),
    ),
}


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

    episodes: list[float] = []
    rewards: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append(float(row["episode"]))
                rewards.append(float(row["avg_reward"]))
            except (KeyError, TypeError, ValueError):
                continue

    return np.asarray(episodes, dtype=float), np.asarray(rewards, dtype=float)


def _dedupe_last_by_episode(
    episodes: np.ndarray,
    rewards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    last_value_by_episode: dict[int, float] = {}
    for episode, reward in zip(episodes, rewards):
        last_value_by_episode[int(round(float(episode)))] = float(reward)

    ordered = sorted(last_value_by_episode.items())
    if not ordered:
        return np.array([], dtype=float), np.array([], dtype=float)

    return (
        np.asarray([episode for episode, _ in ordered], dtype=float),
        np.asarray([reward for _, reward in ordered], dtype=float),
    )


def _clip_and_resample(
    episodes: np.ndarray,
    rewards: np.ndarray,
    *,
    max_episode: int | None,
    resample_step: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if episodes.size == 0 or rewards.size == 0:
        return episodes, rewards

    if max_episode is not None:
        mask = episodes <= float(max_episode)
        episodes = episodes[mask]
        rewards = rewards[mask]

    if episodes.size == 0 or rewards.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if resample_step is None or resample_step <= 0 or episodes.size == 1:
        return episodes, rewards

    step = int(resample_step)
    start_episode = int(np.ceil(float(episodes[0]) / step) * step)
    end_episode = int(np.floor(float(episodes[-1]) / step) * step)
    if end_episode < start_episode:
        return episodes, rewards

    target_episodes = np.arange(start_episode, end_episode + step, step, dtype=float)
    target_rewards = np.interp(target_episodes, episodes, rewards)
    return target_episodes, target_rewards


def _smooth(
    episodes: np.ndarray,
    rewards: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if rewards.size == 0:
        return episodes, rewards, 1
    if window <= 1 or rewards.size == 1:
        return episodes, rewards, 1

    window = min(int(window), int(rewards.size))
    kernel = np.ones(window, dtype=float) / float(window)
    return episodes[window - 1 :], np.convolve(rewards, kernel, mode="valid"), window


def _collect_series(
    curves: Iterable[CurveSpec],
    *,
    max_episode: int | None,
    resample_step: int | None,
    smooth_window: int | None,
) -> list[CurveSeries]:
    series: list[CurveSeries] = []
    for curve in curves:
        episodes, rewards = _load_csv(curve.csv_path)
        episodes, rewards = _dedupe_last_by_episode(episodes, rewards)
        episodes, rewards = _clip_and_resample(
            episodes,
            rewards,
            max_episode=max_episode,
            resample_step=resample_step,
        )
        if episodes.size == 0 or rewards.size == 0:
            print(f"No data in {curve.csv_path}")
            continue
        curve_smooth_window = (
            curve.smooth_window if smooth_window is None else smooth_window
        )
        episodes, rewards, used_window = _smooth(
            episodes,
            rewards,
            max(1, int(curve_smooth_window or 1)),
        )
        series.append(
            CurveSeries(
                label=curve.label,
                episodes=episodes,
                rewards=rewards,
                used_window=used_window,
                source_stem=curve.csv_path.stem,
            )
        )
    return series


def _plot_curves_svg(
    series: list[CurveSeries],
    *,
    output_path: Path,
    title: str,
) -> None:
    width, height = 1100, 560
    left, right, top, bottom = 90, 40, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_vals = [float(x) for curve in series for x in curve.episodes]
    y_vals = [float(y) for curve in series for y in curve.rewards]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    def polyline_points(xs: np.ndarray, ys: np.ndarray) -> str:
        return " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(xs, ys))

    grid_lines: list[str] = []
    axis_labels: list[str] = []
    for i in range(6):
        x = left + i * plot_w / 5
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            'stroke="#e6e6e6" stroke-width="1" />'
        )
        x_val = x_min + i * (x_max - x_min) / 5
        axis_labels.append(
            f'<text x="{x:.2f}" y="{height - 25}" font-size="11" text-anchor="middle" fill="#444">{int(round(x_val))}</text>'
        )
    for i in range(6):
        y = top + i * plot_h / 5
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e6e6e6" stroke-width="1" />'
        )
        y_val = y_max - i * (y_max - y_min) / 5
        axis_labels.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="11" text-anchor="end" fill="#444">{y_val:.1f}</text>'
        )

    legend_items: list[str] = []
    curve_items: list[str] = []
    for idx, curve in enumerate(series):
        color = SVG_COLORS[idx % len(SVG_COLORS)]
        legend_label = curve.label
        if curve.used_window > 1:
            legend_label = f"{curve.label} | MA({curve.used_window})"
        curve_items.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
            f'points="{polyline_points(curve.episodes, curve.rewards)}" />'
        )
        curve_items.append(
            f'<circle cx="{sx(float(curve.episodes[-1])):.2f}" cy="{sy(float(curve.rewards[-1])):.2f}" '
            f'r="4" fill="{color}" />'
        )
        legend_y = 32 + idx * 18
        legend_items.append(
            f'<line x1="110" y1="{legend_y}" x2="140" y2="{legend_y}" stroke="{color}" stroke-width="2.5" />'
            f'<text x="148" y="{legend_y + 4}" font-size="12" fill="#222">{escape(legend_label)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2:.2f}" y="28" font-size="18" text-anchor="middle" fill="#111">{escape(title)}</text>
<text x="{width / 2:.2f}" y="{height - 10}" font-size="12" text-anchor="middle" fill="#333">Episode</text>
<text x="24" y="{height / 2:.2f}" font-size="12" text-anchor="middle" fill="#333" transform="rotate(-90 24 {height / 2:.2f})">Avg Reward</text>
{''.join(grid_lines)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#888" stroke-width="1" />
{''.join(curve_items)}
{''.join(axis_labels)}
{''.join(legend_items)}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def _plot_curves(
    series: list[CurveSeries],
    *,
    output_path: Path | None,
    show: bool,
    title: str,
) -> None:
    if not series:
        raise ValueError("No valid curves to plot.")

    if not HAS_MATPLOTLIB:
        if output_path is None:
            raise RuntimeError(
                "matplotlib is not installed. Use --output to save an SVG file instead."
            )
        svg_output = output_path.with_suffix(".svg")
        _plot_curves_svg(series, output_path=svg_output, title=title)
        print("matplotlib not found, saved SVG instead.")
        print(f"Saved plot to: {svg_output}")
        return

    plt.figure(figsize=(10, 5))
    for curve in series:
        legend_label = curve.label
        if curve.used_window > 1:
            legend_label = f"{curve.label} | MA({curve.used_window})"
        plt.plot(curve.episodes, curve.rewards, linewidth=2.0, label=legend_label)
        plt.scatter([curve.episodes[-1]], [curve.rewards[-1]], s=24, zorder=3)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to: {output_path}")
    if show:
        plt.show()
    plt.close()


def _single_curve_from_args(args: argparse.Namespace) -> CurveSpec:
    required = {
        "seed": args.seed,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"Missing arguments for single-curve mode: {', '.join(missing)}"
        )

    agent_algorithm = args.agent_algorithm.strip().lower()
    network_type = args.network_type.strip().lower()
    marl_algorithm = args.marl_algorithm.strip().lower()
    tag = _file_tag(
        int(args.seed),
        agent_algorithm,
        network_type,
        marl_algorithm,
        float(args.lr),
        float(args.gamma),
        int(args.batch_size),
    )
    return CurveSpec(
        _csv_path(
            int(args.seed),
            agent_algorithm,
            network_type,
            marl_algorithm,
            float(args.lr),
            float(args.gamma),
            int(args.batch_size),
        ),
        tag,
        None,
    )


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "curve"


def _split_output_path(output_path: Path | None, curve: CurveSeries) -> Path | None:
    if output_path is None:
        return None

    suffix = output_path.suffix or ".svg"
    base_name = output_path.stem if output_path.suffix else output_path.name
    file_name = f"{base_name}_{_slugify(curve.source_stem)}{suffix}"
    return output_path.parent / file_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Avg Reward trend from CSV.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CURVES),
        default=None,
        help="Plot a built-in set of recommended curves.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--agent-algorithm", type=str, default="ppo")
    parser.add_argument("--network-type", type=str, default="ffnn")
    parser.add_argument("--marl-algorithm", type=str, default="mappo")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Moving-average window.",
    )
    parser.add_argument(
        "--max-episode",
        type=int,
        default=None,
        help="Only keep data up to this episode.",
    )
    parser.add_argument(
        "--resample-step",
        type=int,
        default=None,
        help="Resample to one point every N episodes. The source CSV is logged every 10 episodes, so finer spacing uses interpolation.",
    )
    parser.add_argument(
        "--combine-curves",
        action="store_true",
        help="Combine multiple preset curves into one figure instead of separate figures.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.preset is not None:
        curves = list(PRESET_CURVES[args.preset])
        smooth_window = args.smooth_window
        max_episode = (
            args.max_episode
            if args.max_episode is not None
            else DEFAULT_PRESET_MAX_EPISODE
        )
        resample_step = (
            args.resample_step
            if args.resample_step is not None
            else DEFAULT_PRESET_RESAMPLE_STEP
        )
        split_curves = len(curves) > 1 and not args.combine_curves
        title = args.title or "Recommended Reward Curves"
    else:
        curves = [_single_curve_from_args(args)]
        smooth_window = args.smooth_window if args.smooth_window is not None else 1
        max_episode = args.max_episode
        resample_step = args.resample_step
        split_curves = False
        title = args.title or f"Avg Reward Trend\n{curves[0].label}"

    output_path = Path(args.output) if args.output else None
    show_plot = not args.no_show
    if output_path is None and not show_plot:
        raise ValueError("Nothing to do: enable show or provide --output.")

    series = _collect_series(
        curves,
        max_episode=max_episode,
        resample_step=resample_step,
        smooth_window=smooth_window,
    )

    if split_curves:
        for curve in series:
            curve_title = args.title or f"Avg Reward Trend\n{curve.label}"
            curve_output = _split_output_path(output_path, curve)
            _plot_curves([curve], output_path=curve_output, show=show_plot, title=curve_title)
    else:
        _plot_curves(series, output_path=output_path, show=show_plot, title=title)

    for curve in curves:
        print(f"CSV: {curve.csv_path}")


if __name__ == "__main__":
    main()
