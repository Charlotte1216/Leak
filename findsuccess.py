from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from xml.sax.saxutils import escape

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_STATS_JSON = REPO_ROOT / "logs" / "training_stats.json"
DEFAULT_CONFIG_JSON = REPO_ROOT / "logs" / "config.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_bucket_size(config_path: Path, default: int = 10) -> int:
    if not config_path.exists():
        return default
    try:
        cfg = _load_json(config_path)
        return int(cfg.get("training", {}).get("log_interval", default))
    except Exception:
        return default


def read_success_series(stats_path: Path) -> Tuple[List[int], List[int], List[int] | None]:
    data = _load_json(stats_path)
    success = data.get("success_episodes")
    if not isinstance(success, list) or not success:
        raise ValueError(f"'success_episodes' missing or empty in {stats_path}")

    partial = data.get("partial_success_episodes")
    partial_series: List[int] | None = None
    if isinstance(partial, list) and len(partial) == len(success):
        partial_series = [int(x) for x in partial]

    episodes = list(range(1, len(success) + 1))
    success_series = [int(x) for x in success]
    return episodes, success_series, partial_series


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    if len(values) < window:
        return []
    out: List[float] = []
    running = sum(values[:window])
    out.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out.append(running / window)
    return out


def bucket_average(values: List[int], bucket_size: int) -> List[float]:
    if bucket_size <= 1:
        return [float(v) for v in values]
    n = len(values) // bucket_size
    if n == 0:
        return []
    out: List[float] = []
    for i in range(n):
        chunk = values[i * bucket_size : (i + 1) * bucket_size]
        out.append(sum(chunk) / len(chunk))
    return out


def bucket_end_episodes(total_episodes: int, bucket_size: int) -> List[int]:
    if bucket_size <= 1:
        return list(range(1, total_episodes + 1))
    n = total_episodes // bucket_size
    return [bucket_size * (i + 1) for i in range(n)]


def plot_success_trend(
    stats_path: Path,
    config_path: Path,
    output_path: Path | None = None,
    show: bool = True,
    window: int = 100,
    bucket_size: int | None = None,
    include_partial: bool = True,
) -> None:
    episodes, success, partial = read_success_series(stats_path)
    bucket = int(bucket_size) if bucket_size and bucket_size > 0 else _infer_bucket_size(config_path, default=10)
    bucket = max(1, bucket)

    x_bucket = bucket_end_episodes(len(success), bucket)
    y_bucket = bucket_average(success, bucket)
    y_roll = moving_average([float(v) for v in success], max(1, window))
    x_roll = episodes[max(1, window) - 1 :] if y_roll else []

    y_bucket_partial: List[float] = []
    if include_partial and partial is not None:
        y_bucket_partial = bucket_average(partial, bucket)

    if not HAS_MATPLOTLIB:
        if output_path is None:
            raise RuntimeError(
                "matplotlib is not installed, cannot show plot directly. "
                "Use --output to save an SVG fallback."
            )
        _plot_success_svg(
            stats_path=stats_path,
            output_path=output_path,
            x_bucket=x_bucket,
            y_bucket=y_bucket,
            x_roll=x_roll,
            y_roll=y_roll,
            y_bucket_partial=y_bucket_partial,
            bucket=bucket,
            window=window,
            include_partial=include_partial and partial is not None,
        )
        return

    plt.figure(figsize=(10, 5))

    if y_bucket:
        plt.plot(
            x_bucket,
            y_bucket,
            label=f"Success Rate (bucket={bucket})",
            linewidth=1.8,
            alpha=0.9,
        )

    if y_roll:
        plt.plot(
            x_roll,
            y_roll,
            label=f"Success Rate MA({window})",
            linewidth=2.2,
        )

    if include_partial and partial is not None and y_bucket_partial:
        plt.plot(
            x_bucket,
            y_bucket_partial,
            label=f"Partial Success Rate (bucket={bucket})",
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
        )

    # Mark the latest bucketed value when available.
    if y_bucket:
        plt.scatter([x_bucket[-1]], [y_bucket[-1]], s=40, zorder=3)
        plt.annotate(
            f"{100.0 * y_bucket[-1]:.1f}%",
            (x_bucket[-1], y_bucket[-1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    plt.title(f"Success Trend: {stats_path.name}")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _plot_success_svg(
    *,
    stats_path: Path,
    output_path: Path,
    x_bucket: List[int],
    y_bucket: List[float],
    x_roll: List[int],
    y_roll: List[float],
    y_bucket_partial: List[float],
    bucket: int,
    window: int,
    include_partial: bool,
) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 70, 20, 50, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_series = x_bucket if x_bucket else (x_roll if x_roll else [0, 1])
    x_min, x_max = float(min(x_series)), float(max(x_series))
    if x_max == x_min:
        x_max = x_min + 1.0
    y_min, y_max = 0.0, 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    def polyline_points(xs: List[int], ys: List[float]) -> str:
        return " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(xs, ys))

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
            f"{int(round(x_val))}</text>"
        )
    for i in range(6):
        y_val = y_max - i * (y_max - y_min) / 5
        y = top + i * plot_h / 5
        labels.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end" fill="#444">'
            f"{100.0 * y_val:.0f}%</text>"
        )

    bucket_line = ""
    roll_line = ""
    partial_line = ""
    if x_bucket and y_bucket:
        bucket_line = (
            f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" stroke-opacity="0.9" '
            f'points="{polyline_points(x_bucket, y_bucket)}" />'
        )
    if x_roll and y_roll:
        roll_line = (
            f'<polyline fill="none" stroke="#d62728" stroke-width="2.4" '
            f'points="{polyline_points(x_roll, y_roll)}" />'
        )
    if include_partial and x_bucket and y_bucket_partial:
        partial_line = (
            f'<polyline fill="none" stroke="#2ca02c" stroke-width="1.8" stroke-dasharray="6,4" '
            f'points="{polyline_points(x_bucket, y_bucket_partial)}" />'
        )

    marker = ""
    if x_bucket and y_bucket:
        mx = sx(float(x_bucket[-1]))
        my = sy(float(y_bucket[-1]))
        marker = (
            f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="4" fill="#1f77b4" />'
            f'<text x="{mx + 8:.2f}" y="{my - 8:.2f}" font-size="12" fill="#222">{100.0*y_bucket[-1]:.1f}%</text>'
        )

    legend = (
        '<line x1="90" y1="28" x2="120" y2="28" stroke="#1f77b4" stroke-width="2" />'
        f'<text x="126" y="32" font-size="12" fill="#222">Success (bucket={bucket})</text>'
        '<line x1="300" y1="28" x2="330" y2="28" stroke="#d62728" stroke-width="2.4" />'
        f'<text x="336" y="32" font-size="12" fill="#222">Success MA({window})</text>'
    )
    if include_partial:
        legend += (
            '<line x1="500" y1="28" x2="530" y2="28" stroke="#2ca02c" stroke-width="1.8" stroke-dasharray="6,4" />'
            f'<text x="536" y="32" font-size="12" fill="#222">Partial (bucket={bucket})</text>'
        )

    title = escape(f"Success Trend: {stats_path.name}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2:.2f}" y="24" font-size="16" text-anchor="middle" fill="#111">{title}</text>
<text x="{width/2:.2f}" y="{height-6}" font-size="12" text-anchor="middle" fill="#333">Episode</text>
<text x="18" y="{height/2:.2f}" font-size="12" text-anchor="middle" fill="#333" transform="rotate(-90 18 {height/2:.2f})">Success Rate</text>
{''.join(grid_lines)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#888" stroke-width="1" />
{bucket_line}
{roll_line}
{partial_line}
{marker}
{legend}
{''.join(labels)}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot success-rate trend for the latest training run.")
    parser.add_argument(
        "--stats-json",
        type=str,
        default=str(DEFAULT_STATS_JSON),
        help="Path to training_stats.json (default: logs/training_stats.json)",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=str(DEFAULT_CONFIG_JSON),
        help="Path to config.json for inferring log_interval bucket size",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling window size for success rate moving average",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=0,
        help="Bucket size for success-rate trend (0 = infer from config log_interval)",
    )
    parser.add_argument(
        "--no-partial",
        action="store_true",
        help="Do not plot partial success rate",
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
    args = parser.parse_args()

    stats_path = Path(args.stats_json)
    config_path = Path(args.config_json)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats JSON not found: {stats_path}")

    output_path: Path | None = Path(args.output) if args.output else None
    show_plot = not args.no_show
    if output_path is not None and (not HAS_MATPLOTLIB) and output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
        print("matplotlib not found, saving SVG instead.")
    if output_path is None and not show_plot:
        raise ValueError("Nothing to do: enable show or provide --output.")

    plot_success_trend(
        stats_path=stats_path,
        config_path=config_path,
        output_path=output_path,
        show=show_plot,
        window=max(1, args.window),
        bucket_size=args.bucket_size if args.bucket_size > 0 else None,
        include_partial=not args.no_partial,
    )
    print(f"Stats: {stats_path}")
    if show_plot:
        print("Plot shown.")
    if output_path is not None:
        print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
