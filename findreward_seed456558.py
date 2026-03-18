from __future__ import annotations

from pathlib import Path

from findreward import HAS_MATPLOTLIB, plot_reward_trend


REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "marl_leakage_search" / "experiments" / "Train_network" / (
    "seed456558_agentppo_netffnn_marlmappo_na4_commk2_chlos_nlos_lr0.000100_gamma0.9000_bs64_auxon_avg_reward_trend.csv"
)
OUTPUT_PATH: Path | None = REPO_ROOT / "debug_renders" / "seed456558_reward_curve.svg"
SHOW_PLOT = True
SMOOTH_WINDOW = 5
PLOT_START_EPISODE: int | None = 450
PLOT_END_EPISODE: int | None = 2450
EPISODE_AXIS_ORIGIN: int | None = PLOT_START_EPISODE
PLOT_TITLE: str | None = ""
X_LABEL = "Training Episodes"
Y_LABEL = "Average Episodic Return"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    output_path = OUTPUT_PATH
    if output_path is None and not SHOW_PLOT:
        raise ValueError("Nothing to do: enable SHOW_PLOT or provide OUTPUT_PATH.")

    if output_path is not None and (not HAS_MATPLOTLIB) and output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
        print("matplotlib not found, saving SVG instead.")

    plot_reward_trend(
        csv_path=CSV_PATH,
        output_path=output_path,
        smooth_window=max(1, SMOOTH_WINDOW),
        show=SHOW_PLOT,
        start_episode=PLOT_START_EPISODE,
        end_episode=PLOT_END_EPISODE,
        episode_axis_origin=EPISODE_AXIS_ORIGIN,
        title=PLOT_TITLE,
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
    )
    print(f"CSV: {CSV_PATH}")
    if SHOW_PLOT:
        print("Plot shown.")
    if output_path is not None:
        print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
