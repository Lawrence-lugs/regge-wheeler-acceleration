from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from stream_sim.config import EXPERIMENTS, RESULTS_DIR


def _iter_nodes(workload: Any) -> list[Any]:
    if hasattr(workload, "node_list"):
        return list(workload.node_list)
    if hasattr(workload, "nodes"):
        return list(workload.nodes())
    raise TypeError("Unsupported workload node container")


def _safe_num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _core_class(core_id: int | None, experiment_name: str) -> str:
    experiment = next(exp for exp in EXPERIMENTS if exp.name == experiment_name)
    if core_id in experiment.matrix_core_ids:
        return "matrix"
    if core_id in experiment.vector_core_ids:
        return "vector"
    if core_id in experiment.scalar_core_ids:
        return "scalar"
    return "other"


def _plot_bar(summary: pd.DataFrame, column: str, ylabel: str, filename: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.bar(summary["label"], summary[column], color=color)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=160)
    plt.close(fig)


def analyze_all() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []

    for experiment in EXPERIMENTS:
        scme_path = experiment.hardware_path.parents[2] / "outputs" / experiment.experiment_id / "scme.pickle"
        if not scme_path.exists():
            summary_rows.append(
                {
                    "experiment": experiment.name,
                    "label": experiment.label,
                    "latency_cycles": float("nan"),
                    "energy_total_pj": float("nan"),
                    "energy_cn_onchip_pj": float("nan"),
                    "energy_cn_offchip_memory_pj": float("nan"),
                    "matrix_mapped_nodes": 0,
                    "vector_mapped_nodes": 0,
                }
            )
            continue

        with scme_path.open("rb") as handle:
            scme = pickle.load(handle)

        nodes = sorted(_iter_nodes(scme.workload), key=lambda n: (getattr(n, "id", -1), getattr(n, "sub_id", -1)))
        matrix_count = 0
        vector_count = 0

        for node in nodes:
            chosen_core = getattr(node, "chosen_core_allocation", None)
            core_class = _core_class(chosen_core, experiment.name)
            if core_class == "matrix":
                matrix_count += 1
            if core_class == "vector":
                vector_count += 1
            node_rows.append(
                {
                    "experiment": experiment.name,
                    "label": experiment.label,
                    "node_id": getattr(node, "id", None),
                    "sub_id": getattr(node, "sub_id", None),
                    "name": getattr(node, "name", ""),
                    "operator_type": getattr(node, "type", ""),
                    "chosen_core": chosen_core,
                    "core_class": core_class,
                    "runtime_cycles": _safe_num(getattr(node, "runtime", 0)),
                    "onchip_energy_pj": _safe_num(getattr(node, "onchip_energy", 0)),
                    "offchip_energy_pj": _safe_num(getattr(node, "offchip_energy", 0)),
                }
            )

        summary_rows.append(
            {
                "experiment": experiment.name,
                "label": experiment.label,
                "latency_cycles": _safe_num(getattr(scme, "latency", 0)),
                "energy_total_pj": _safe_num(getattr(scme, "energy", 0)),
                "energy_cn_onchip_pj": _safe_num(getattr(scme, "total_cn_onchip_energy", 0)),
                "energy_cn_offchip_memory_pj": _safe_num(getattr(scme, "total_cn_offchip_memory_energy", 0)),
                "matrix_mapped_nodes": matrix_count,
                "vector_mapped_nodes": vector_count,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    order = {exp.name: idx for idx, exp in enumerate(EXPERIMENTS)}
    summary_df = summary_df.sort_values("experiment", key=lambda s: s.map(order)).reset_index(drop=True)
    if node_rows:
        nodes_df = pd.DataFrame(node_rows)
        nodes_df = nodes_df.sort_values(["experiment", "node_id", "sub_id"]).reset_index(drop=True)
        op_summary_df = (
            nodes_df.groupby(["experiment", "label", "operator_type", "core_class"], dropna=False, as_index=False)[
                ["runtime_cycles", "onchip_energy_pj", "offchip_energy_pj"]
            ]
            .sum()
            .sort_values(["experiment", "runtime_cycles"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        nodes_df = pd.DataFrame(
            columns=[
                "experiment",
                "label",
                "node_id",
                "sub_id",
                "name",
                "operator_type",
                "chosen_core",
                "core_class",
                "runtime_cycles",
                "onchip_energy_pj",
                "offchip_energy_pj",
            ]
        )
        op_summary_df = pd.DataFrame(
            columns=[
                "experiment",
                "label",
                "operator_type",
                "core_class",
                "runtime_cycles",
                "onchip_energy_pj",
                "offchip_energy_pj",
            ]
        )

    summary_df.to_csv(RESULTS_DIR / "stream_summary.csv", index=False)
    nodes_df.to_csv(RESULTS_DIR / "stream_node_stats.csv", index=False)
    op_summary_df.to_csv(RESULTS_DIR / "stream_operator_summary.csv", index=False)

    _plot_bar(summary_df, "latency_cycles", "Latency (cycles)", "ablation_latency.png", "#2F5A78")
    _plot_bar(summary_df, "energy_total_pj", "Energy (pJ)", "ablation_energy.png", "#A55C3B")
    _plot_bar(summary_df, "matrix_mapped_nodes", "Matrix-mapped Nodes", "matrix_mapped_node_counts.png", "#6D8F4E")
    _plot_bar(summary_df, "vector_mapped_nodes", "Vector-mapped Nodes", "vector_mapped_node_counts.png", "#8C6BB1")

    print("Wrote", RESULTS_DIR / "stream_summary.csv")
    print("Wrote", RESULTS_DIR / "stream_node_stats.csv")
    print("Wrote", RESULTS_DIR / "stream_operator_summary.csv")
    print("Wrote", RESULTS_DIR / "ablation_latency.png")
    print("Wrote", RESULTS_DIR / "ablation_energy.png")
    print("Wrote", RESULTS_DIR / "matrix_mapped_node_counts.png")
    print("Wrote", RESULTS_DIR / "vector_mapped_node_counts.png")


def main() -> None:
    analyze_all()


if __name__ == "__main__":
    main()
