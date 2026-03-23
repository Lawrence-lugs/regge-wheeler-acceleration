from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from . import run_analysis as analysis_runner
from .config import AnalysisConfig, LatencyConfig, MatrixConfig, MemoryConfig, VectorConfig


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_sweep_config(max_lanes: int, tpu_size: int, l1_bandwidth_bits: int) -> AnalysisConfig:
    return AnalysisConfig(
        vector=VectorConfig(
            max_lanes=max_lanes,
            supported_lanes=(1, 2, 4, 8, 16),
            dtype_name="fp32",
        ),
        latencies=LatencyConfig(
            scalar_default=1.0,
            vector_default=1.0,
            scalar_overrides=(),
            vector_overrides=(
                ("div", 5.0),
                ("sqrt", 5.0),
            ),
        ),
        memory=MemoryConfig(
            l1_bandwidth_bits=l1_bandwidth_bits,
            dtype_bits=32,
            weights_local_to_matrix_unit=True,
        ),
        matrix=MatrixConfig(
            tile_m=tpu_size,
            tile_k=tpu_size,
            tile_n=tpu_size,
            dtype_name="fp32",
        ),
    )


def run_sweep(
    max_lane_sweeps: list[int],
    tpu_size_sweeps: list[int],
    bandwidth_sweeps: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for max_lanes in max_lane_sweeps:
        for tpu_size in tpu_size_sweeps:
            for bandwidth in bandwidth_sweeps:
                config = build_sweep_config(
                    max_lanes=max_lanes,
                    tpu_size=tpu_size,
                    l1_bandwidth_bits=bandwidth,
                )
                config.validate()
                _, summary = analysis_runner.run_all_experiments(config)
                totals = summary.set_index("experiment")["total_latency"]

                scalar_only = float(totals.loc["scalar_only"])
                scalar_vector_only = float(totals.loc["scalar_vector_only"])
                scalar_matrix_only = float(totals.loc["scalar_matrix_only"])
                all_primitives = float(totals.loc["all_primitives"])

                rows.append(
                    {
                        "max_lanes": max_lanes,
                        "tpu_size": tpu_size,
                        "l1_bandwidth_bits": bandwidth,
                        "scalar_only_latency": scalar_only,
                        "scalar_vector_only_latency": scalar_vector_only,
                        "scalar_matrix_only_latency": scalar_matrix_only,
                        "all_primitives_latency": all_primitives,
                        "speedup_all_primitives_vs_scalar": (
                            scalar_only / all_primitives if all_primitives > 0 else 0.0
                        ),
                        "speedup_vector_vs_scalar": (
                            scalar_only / scalar_vector_only if scalar_vector_only > 0 else 0.0
                        ),
                        "speedup_matrix_vs_scalar": (
                            scalar_only / scalar_matrix_only if scalar_matrix_only > 0 else 0.0
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["speedup_all_primitives_vs_scalar", "max_lanes", "tpu_size", "l1_bandwidth_bits"],
        ascending=[False, True, True, True],
    )


def write_sweep_results(frame: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "sweep_latency_summary.csv", index=False)

    best_per_bandwidth = (
        frame.sort_values("speedup_all_primitives_vs_scalar", ascending=False)
        .groupby("l1_bandwidth_bits", as_index=False)
        .first()
        .sort_values("l1_bandwidth_bits")
    )
    best_per_bandwidth.to_csv(RESULTS_DIR / "sweep_best_per_bandwidth.csv", index=False)


def make_sweep_scatter_plot(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    marker_sizes = frame["l1_bandwidth_bits"] / frame["l1_bandwidth_bits"].min() * 50.0
    scatter = ax.scatter(
        frame["tpu_size"],
        frame["max_lanes"],
        c=frame["speedup_all_primitives_vs_scalar"],
        s=marker_sizes,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )

    ax.set_xlabel("TPU Tile Size (M=K=N)")
    ax.set_ylabel("Vector Max Lanes")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(sorted(frame["tpu_size"].unique()))
    ax.set_yticks(sorted(frame["max_lanes"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, linestyle="--", alpha=0.35)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Speedup (All Primitives vs Scalar Only)")

    for bandwidth in sorted(frame["l1_bandwidth_bits"].unique()):
        size = bandwidth / frame["l1_bandwidth_bits"].min() * 50.0
        ax.scatter([], [], s=size, c="gray", alpha=0.6, edgecolors="black", linewidths=0.3, label=f"{bandwidth}b")
    ax.legend(title="L1 Bandwidth", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sweep_speedup_scatter.png", dpi=300)
    fig.savefig(RESULTS_DIR / "sweep_speedup_scatter.pdf")
    plt.close(fig)


def main() -> None:

    max_lane_sweeps = [1, 2, 4, 8, 16]
    tpu_size_sweeps = [16, 32, 64, 128, 256]
    bandwidth_sweeps = [128, 256, 512, 1024]

    sweep_frame = run_sweep(
        max_lane_sweeps=max_lane_sweeps,
        tpu_size_sweeps=tpu_size_sweeps,
        bandwidth_sweeps=bandwidth_sweeps,
    )
    write_sweep_results(sweep_frame)
    make_sweep_scatter_plot(sweep_frame)

    top_rows = sweep_frame.head(10)
    print(top_rows.to_string(index=False))
    print("\nWrote sweep outputs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
