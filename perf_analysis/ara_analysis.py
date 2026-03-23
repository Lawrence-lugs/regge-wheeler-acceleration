from __future__ import annotations

from pathlib import Path

from . import run_analysis as analysis_runner
from .config import AnalysisConfig, LatencyConfig, MatrixConfig, MemoryConfig, VectorConfig


def build_ara_config() -> AnalysisConfig:
    # Ara defaults to 4 lanes with 64-bit per-lane datapath. For fp32 modeling,
    # this maps to 4 fp32 lanes and 256-bit aggregate per-cycle bandwidth.
    return AnalysisConfig(
        vector=VectorConfig(
            max_lanes=8,
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
            l1_bandwidth_bits=1024,
            dtype_bits=32,
            weights_local_to_matrix_unit=True,
        ),
        # TPU config to a single 128x128 systolic array like TPUv1
        matrix=MatrixConfig(
            tile_m=128,
            tile_k=128,
            tile_n=128,
            dtype_name="fp32",
        )
    )


def main() -> None:
    config = build_ara_config()
    config.validate()

    ara_results_dir = Path(__file__).resolve().parent / "results" / "ara"
    ara_results_dir.mkdir(parents=True, exist_ok=True)

    # Reuse plotting/writing utilities from run_analysis with an Ara-specific output directory.
    analysis_runner.RESULTS_DIR = ara_results_dir

    experiment_frames, summary = analysis_runner.run_all_experiments(config)
    analysis_runner.write_results(experiment_frames, summary)
    analysis_runner.make_total_latency_plot(summary)
    # analysis_runner.make_latency_composition_plots(summary)
    # analysis_runner.make_latency_multibar_plot(summary)
    # analysis_runner.make_latency_distribution_pie_charts(summary)
    analysis_runner.make_speedup_plot(summary)
    analysis_runner.make_ablation_plot(summary)

    print(summary.to_string(index=False))
    print("\nWrote Ara primitive statistics, summaries, and plots to", ara_results_dir)


if __name__ == "__main__":
    main()
