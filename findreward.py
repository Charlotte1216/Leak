from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from xml.sax.saxutils import escape

try:
    import matplotlib
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = REPO_ROOT / "marl_leakage_search" / "experiments" / "Train_network"


def find_latest_reward_csv(search_dir: Path) -> Path:
    candidates = list(search_dir.glob("*_avg_reward_trend.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_avg_reward_trend.csv found in {search_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_reward_csv(csv_path: Path) -> Tuple[List[int], List[float]]:
    episodes: List[int] = []
    rewards: List[float] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append(int(float(row["episode"])))
                rewards.append(float(row["avg_reward"]))
            except (KeyError, TypeError, ValueError):
                continue

    if not episodes:
        raise ValueError(f"No valid reward rows found in {csv_path}")

    return episodes, rewards


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return []
    out: List[float] = []
    running = sum(values[:window])
    out.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out.append(running / window)
    return out


def plot_reward_trend(
    csv_path: Path,
    output_path: Path | None = None,
    smooth_window: int = 5,
    show: bool = True,
) -> None:
    episodes, rewards = read_reward_csv(csv_path)

    if not HAS_MATPLOTLIB:
        if output_path is None:
            raise RuntimeError(
                "matplotlib is not installed, cannot show plot directly. "
                "Use --output to save an SVG fallback."
            )
        _plot_reward_trend_svg(csv_path, output_path, episodes, rewards, smooth_window)
        return

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Avg Reward", linewidth=1.5, alpha=0.8)

    smooth_rewards = moving_average(rewards, smooth_window)
    if smooth_rewards:
        smooth_episodes = episodes[smooth_window - 1 :]
        plt.plot(
            smooth_episodes,
            smooth_rewards,
            label=f"MA({smooth_window})",
            linewidth=2.0,
        )

    plt.scatter([episodes[-1]], [rewards[-1]], s=40, zorder=3)
    plt.annotate(
        f"{rewards[-1]:.2f}",
        (episodes[-1], rewards[-1]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9,
    )

    plt.title(f"Reward Trend: {csv_path.name}")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _plot_reward_trend_svg(
    csv_path: Path,
    output_path: Path,
    episodes: List[int],
    rewards: List[float],
    smooth_window: int,
) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 70, 20, 50, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = float(min(episodes)), float(max(episodes))
    y_min, y_max = float(min(rewards)), float(max(rewards))
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

    def polyline_points(xs: List[float], ys: List[float]) -> str:
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))

    grid_lines: List[str] = []
    for i in range(6):
        x = left + i * plot_w / 5
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            'stroke="#e6e6e6" stroke-width="1" />'
        )
    for i in range(6):
        y = top + i * plot_h / 5
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e6e6e6" stroke-width="1" />'
        )

    labels: List[str] = []
    for i in range(6):
        x_val = x_min + i * (x_max - x_min) / 5
        x = left + i * plot_w / 5
        labels.append(
            f'<text x="{x:.2f}" y="{height - 20}" font-size="11" text-anchor="middle" fill="#444">'
            f'{int(round(x_val))}</text>'
        )
    for i in range(6):
        y_val = y_max - i * (y_max - y_min) / 5
        y = top + i * plot_h / 5
        labels.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end" fill="#444">'
            f'{y_val:.1f}</text>'
        )

    raw_line = (
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" stroke-opacity="0.85" '
        f'points="{polyline_points([float(x) for x in episodes], rewards)}" />'
    )

    smooth_rewards = moving_average(rewards, smooth_window)
    smooth_line = ""
    legend_smooth = ""
    if smooth_rewards:
        smooth_episodes = [float(x) for x in episodes[smooth_window - 1 :]]
        smooth_line = (
            f'<polyline fill="none" stroke="#d62728" stroke-width="2.5" '
            f'points="{polyline_points(smooth_episodes, smooth_rewards)}" />'
        )
        legend_smooth = (
            '<line x1="190" y1="28" x2="220" y2="28" stroke="#d62728" stroke-width="2.5" />'
            f'<text x="226" y="32" font-size="12" fill="#222">MA({smooth_window})</text>'
        )

    last_x = sx(float(episodes[-1]))
    last_y = sy(float(rewards[-1]))
    last_marker = (
        f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="4" fill="#1f77b4" />'
        f'<text x="{last_x + 8:.2f}" y="{last_y - 8:.2f}" font-size="12" fill="#222">{rewards[-1]:.2f}</text>'
    )

    title = escape(f"Reward Trend: {csv_path.name}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2:.2f}" y="24" font-size="16" text-anchor="middle" fill="#111">{title}</text>
<text x="{width/2:.2f}" y="{height-6}" font-size="12" text-anchor="middle" fill="#333">Episode</text>
<text x="18" y="{height/2:.2f}" font-size="12" text-anchor="middle" fill="#333" transform="rotate(-90 18 {height/2:.2f})">Avg Reward</text>
{''.join(grid_lines)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#888" stroke-width="1" />
{raw_line}
{smooth_line}
{last_marker}
<line x1="90" y1="28" x2="120" y2="28" stroke="#1f77b4" stroke-width="2" />
<text x="126" y="32" font-size="12" fill="#222">Avg Reward</text>
{legend_smooth}
{''.join(labels)}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward trend for the latest training run.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to a specific *_avg_reward_trend.csv file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path (only saves when provided)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for overlay line",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_reward_csv(EXPERIMENT_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path: Path | None = Path(args.output) if args.output else None
    show_plot = not args.no_show

    if output_path is not None and (not HAS_MATPLOTLIB) and output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
        print("matplotlib not found, saving SVG instead.")

    if output_path is None and not show_plot:
        raise ValueError("Nothing to do: enable show or provide --output.")

    plot_reward_trend(
        csv_path,
        output_path,
        smooth_window=max(1, args.smooth_window),
        show=show_plot,
    )
    print(f"CSV: {csv_path}")
    if show_plot:
        print("Plot shown.")
    if output_path is not None:
        print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
