from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import AnalysisConfig, EXPERIMENTS
from .primitives import PrimitiveExecutor
from .stats import get_global_stats_frame, reset_global_stats, summarize_stats
from .workloads import simulate_full_workload


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_single_experiment(config: AnalysisConfig, experiment_name: str) -> pd.DataFrame:
    capability = next(profile for profile in EXPERIMENTS if profile.name == experiment_name)
    reset_global_stats()
    executor = PrimitiveExecutor(config=config, capabilities=capability)
    simulate_full_workload(executor)
    frame = get_global_stats_frame()
    return frame


def run_all_experiments(config: AnalysisConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    experiment_frames: dict[str, pd.DataFrame] = {}
    latency_rows: list[dict[str, float | str]] = []

    for capability in EXPERIMENTS:
        frame = run_single_experiment(config, capability.name)
        experiment_frames[capability.name] = frame
        total_latency = float(frame["estimated_latency"].sum()) if not frame.empty else 0.0
        scalar_latency = float(
            frame.loc[frame["primitive_kind"] == "scalar", "estimated_latency"].sum()
        )
        vector_latency = float(
            frame.loc[frame["primitive_kind"] == "vector", "estimated_latency"].sum()
        )
        matrix_latency = float(
            frame.loc[frame["primitive_kind"] == "matrix", "estimated_latency"].sum()
        )
        latency_rows.append(
            {
                "experiment": capability.name,
                "total_latency": total_latency,
                "scalar_latency": scalar_latency,
                "vector_latency": vector_latency,
                "matrix_latency": matrix_latency,
            }
        )

    summary = pd.DataFrame(latency_rows).sort_values("total_latency", ascending=False)
    return experiment_frames, summary


def write_results(experiment_frames: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in experiment_frames.items():
        frame.to_csv(RESULTS_DIR / f"{name}_primitive_stats.csv", index=False)
        per_kind = summarize_stats(frame)
        per_kind.to_csv(RESULTS_DIR / f"{name}_primitive_summary.csv", index=False)
    summary.to_csv(RESULTS_DIR / "latency_summary.csv", index=False)


def make_total_latency_plot(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("total_latency", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(ordered["experiment"], ordered["total_latency"], color=["#5B6C5D", "#C27B58", "#B7A26A", "#6A7FB3"])
    ax.set_ylabel("Estimated Latency")
    ax.set_title("Regge-Wheeler v2 Primitive Capability Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:,.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_totals.png", dpi=160)
    plt.close(fig)


def make_ablation_plot(summary: pd.DataFrame) -> None:
    plot_order = ["all_primitives", "scalar_vector_only", "scalar_matrix_only", "scalar_only"]
    ordered = summary.set_index("experiment").loc[plot_order].reset_index()
    baseline = float(ordered.loc[ordered["experiment"] == "all_primitives", "total_latency"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(ordered["experiment"], ordered["total_latency"], color=["#2F4B3C", "#4F6D7A", "#C17C74", "#D9BF77"])
    ax.set_ylabel("Estimated Latency")
    ax.set_title("Ablation of Vector and Matrix Primitives")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    x_positions = [bar.get_x() + bar.get_width() / 2.0 for bar in bars]
    y_positions = [bar.get_height() for bar in bars]
    for idx in range(1, len(x_positions)):
        delta = y_positions[idx] - baseline
        ax.annotate(
            f"+{delta:,.0f}",
            xy=(x_positions[idx], y_positions[idx]),
            xytext=(x_positions[0], baseline),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333333"},
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_ablation.png", dpi=160)
    plt.close(fig)


def main() -> None:
    config = AnalysisConfig()
    config.validate()
    experiment_frames, summary = run_all_experiments(config)
    write_results(experiment_frames, summary)
    make_total_latency_plot(summary)
    make_ablation_plot(summary)
    print("Wrote primitive statistics, summaries, and plots to", RESULTS_DIR)


if __name__ == "__main__":
    main()