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
        compute_total = float(frame["compute_latency"].sum()) if "compute_latency" in frame.columns else total_latency
        memory_total = float(frame["memory_latency"].sum()) if "memory_latency" in frame.columns else 0.0
        latency_rows.append(
            {
                "experiment": capability.name,
                "total_latency": total_latency,
                "scalar_latency": scalar_latency,
                "vector_latency": vector_latency,
                "matrix_latency": matrix_latency,
                "compute_latency": compute_total,
                "memory_latency": memory_total,
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
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(ordered["experiment"], ordered["total_latency"], color=["#5B6C5D", "#C27B58", "#B7A26A", "#6A7FB3"])

    ax.set_ylabel("Estimated Latency (Cycles)")
    # ax.set_title("Regge-Wheeler v2 Primitive Capability Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar in bars:
        height = bar.get_height()

        # Format in T,G,M,K
        if height >= 1e12:
            text = f"{height / 1e12:.1f}T"
        elif height >= 1e9:
            text = f"{height / 1e9:.1f}G"
        elif height >= 1e6:
            text = f"{height / 1e6:.1f}M"
        elif height >= 1e3:
            text = f"{height / 1e3:.1f}K"
        else:
            text = f"{height:.0f}"
        
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, text, ha="center", va="bottom")
    
    # Also format y-axis ticks in T,G,M,K
    # def format_yticks(value, pos):
    #     if value >= 1e12:
    #         return f"{value / 1e12:.1f}T"
    #     elif value >= 1e9:
    #         return f"{value / 1e9:.1f}G"
    #     elif value >= 1e6:
    #         return f"{value / 1e6:.1f}M"
    #     elif value >= 1e3:
    #         return f"{value / 1e3:.1f}K"
    #     else:
    #         return f"{value:.0f}"
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(format_yticks))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=20, ha="right")

    ax.set_yscale("log")

    # Raise y limit to make space for annotations while staying valid on a log axis.
    positive_totals = ordered.loc[ordered["total_latency"] > 0, "total_latency"]
    if not positive_totals.empty:
        lower = max(float(positive_totals.min()) / 3.0, 1e-3)
        upper = float(positive_totals.max()) * (10**0.5)
        ax.set_ylim(lower, upper)
    
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_totals.png", dpi=300)
    fig.savefig(RESULTS_DIR / "latency_totals.pdf")
    plt.close(fig)


def make_latency_composition_plots(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("total_latency", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(
        ordered["experiment"],
        ordered["scalar_latency"],
        color="#5B6C5D",
        label="Scalar",
    )
    ax.bar(
        ordered["experiment"],
        ordered["vector_latency"],
        bottom=ordered["scalar_latency"],
        color="#C27B58",
        label="Vector",
    )
    ax.bar(
        ordered["experiment"],
        ordered["matrix_latency"],
        bottom=ordered["scalar_latency"] + ordered["vector_latency"],
        color="#B7A26A",
        label="Matrix",
    )
    ax.set_ylabel("Estimated Latency (Cycles)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")
    ax.set_yscale("log")

    positive_totals = ordered.loc[ordered["total_latency"] > 0, "total_latency"]
    if not positive_totals.empty:
        lower = max(float(positive_totals.min()) / 3.0, 1e-3)
        upper = float(positive_totals.max()) * (10**0.5)
        ax.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_composition.png", dpi=300)
    fig.savefig(RESULTS_DIR / "latency_composition.pdf")
    plt.close(fig)

    composition = ordered[["scalar_latency", "vector_latency", "matrix_latency"]].div(
        ordered["total_latency"], axis=0
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(
        ordered["experiment"],
        composition["scalar_latency"],
        color="#5B6C5D",
        label="Scalar",
    )
    ax.bar(
        ordered["experiment"],
        composition["vector_latency"],
        bottom=composition["scalar_latency"],
        color="#C27B58",
        label="Vector",
    )
    ax.bar(
        ordered["experiment"],
        composition["matrix_latency"],
        bottom=composition["scalar_latency"] + composition["vector_latency"],
        color="#B7A26A",
        label="Matrix",
    )
    ax.set_ylabel("Latency Composition")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_composition_fraction.png", dpi=300)
    fig.savefig(RESULTS_DIR / "latency_composition_fraction.pdf")
    plt.close(fig)


def make_latency_multibar_plot(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("total_latency", ascending=False).copy()

    x_positions = list(range(len(ordered)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        [x - 1.5 * width for x in x_positions],
        ordered["scalar_latency"],
        width=width,
        color="#5B6C5D",
        label="Scalar",
    )
    ax.bar(
        [x - 0.5 * width for x in x_positions],
        ordered["vector_latency"],
        width=width,
        color="#C27B58",
        label="Vector",
    )
    ax.bar(
        [x + 0.5 * width for x in x_positions],
        ordered["matrix_latency"],
        width=width,
        color="#B7A26A",
        label="Matrix",
    )
    ax.bar(
        [x + 1.5 * width for x in x_positions],
        ordered["total_latency"],
        width=width,
        color="#6A7FB3",
        label="Total",
    )

    ax.set_ylabel("Estimated Cycles")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered["experiment"])
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    ax.set_yscale("log")

    positive_values = ordered[
        ["scalar_latency", "vector_latency", "matrix_latency", "total_latency"]
    ].to_numpy()
    positive_values = positive_values[positive_values > 0]
    if positive_values.size:
        lower = max(float(positive_values.min()) / 3.0, 1e-3)
        upper = float(positive_values.max()) * (10**0.5)
        ax.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_multibar.png", dpi=300)
    fig.savefig(RESULTS_DIR / "latency_multibar.pdf")
    plt.close(fig)


def make_latency_distribution_pie_charts(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("total_latency", ascending=False).copy()
    labels = ["Scalar", "Vector", "Matrix"]
    colors = ["#5B6C5D", "#C27B58", "#B7A26A"]

    for row in ordered.itertuples(index=False):
        values = [float(row.scalar_latency), float(row.vector_latency), float(row.matrix_latency)]
        total = sum(values)

        fig, ax = plt.subplots(figsize=(4, 3))
        if total > 0:
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
            )
            for text in texts:
                text.set_fontsize(9)
            for text in autotexts:
                text.set_fontsize(9)
                text.set_color("white")
        else:
            ax.text(0.5, 0.5, "No latency data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"{row.experiment} Composition")
        ax.axis("equal")

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"latency_distribution_pie_{row.experiment}.png", dpi=300)
        fig.savefig(RESULTS_DIR / f"latency_distribution_pie_{row.experiment}.pdf")
        plt.close(fig)

def make_speedup_plot(summary: pd.DataFrame) -> None:
    baseline = float(summary.loc[summary["experiment"] == "scalar_only", "total_latency"].iloc[0])
    ordered = summary.sort_values("total_latency", ascending=False).copy()
    ordered["speedup"] = baseline / ordered["total_latency"]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(ordered["experiment"], ordered["speedup"], color=["#5B6C5D", "#C27B58", "#B7A26A", "#6A7FB3"])
    ax.set_ylabel("Speedup")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar in bars:
        height = bar.get_height()
        text = f"{height:.1f}x"
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, text, ha="center", va="bottom")

    plt.xticks(rotation=20, ha="right")

    ax.set_yscale("log")
    positive_speedups = ordered.loc[ordered["speedup"] > 0, "speedup"]
    if not positive_speedups.empty:
        lower = max(float(positive_speedups.min()) / 3.0, 1e-3)
        upper = float(positive_speedups.max()) * (10**0.5)
        ax.set_ylim(lower, upper)

    # Clean the labels to be more readable
    label_mapping = {
        "all_primitives": "Full Accelerator",
        "scalar_vector_only": "Scalar + Vector",
        "scalar_matrix_only": "Scalar + Matrix",
        "scalar_only": "Scalar Only",
    }
    ax.set_xticklabels([label_mapping.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()])

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "speedup.png", dpi=300)
    fig.savefig(RESULTS_DIR / "speedup.pdf")
    plt.close(fig)


def make_ablation_plot(summary: pd.DataFrame) -> None:
    plot_order = ["all_primitives", "scalar_vector_only", "scalar_matrix_only", "scalar_only"]
    ordered = summary.set_index("experiment").loc[plot_order].reset_index()
    baseline = float(ordered.loc[ordered["experiment"] == "all_primitives", "total_latency"].iloc[0])

    fig, ax = plt.subplots(figsize=(4, 3))
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
    # make_latency_composition_plots(summary)
    # make_latency_multibar_plot(summary)
    # make_latency_distribution_pie_charts(summary)
    make_speedup_plot(summary)
    # make_ablation_plot(summary)
    print("Wrote primitive statistics, summaries, and plots to", RESULTS_DIR)


if __name__ == "__main__":
    main()