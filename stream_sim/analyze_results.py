from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"


def _latest_scme_path() -> Path:
    candidates = sorted(OUTPUTS_DIR.glob("*/scme.pickle"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No Stream run found. Execute stream_sim.run_stream_sim first.")
    return candidates[0]


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


def analyze(scme_path: Path) -> None:
    with scme_path.open("rb") as handle:
        scme = pickle.load(handle)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {
            "experiment_dir": scme_path.parent.name,
            "latency_cycles": _safe_num(getattr(scme, "latency", 0)),
            "energy_total_pj": _safe_num(getattr(scme, "energy", 0)),
            "energy_cn_onchip_pj": _safe_num(getattr(scme, "total_cn_onchip_energy", 0)),
            "energy_cn_offchip_link_pj": _safe_num(getattr(scme, "total_cn_offchip_link_energy", 0)),
            "energy_cn_offchip_memory_pj": _safe_num(getattr(scme, "total_cn_offchip_memory_energy", 0)),
            "energy_eviction_offchip_link_pj": _safe_num(getattr(scme, "total_eviction_to_offchip_link_energy", 0)),
            "energy_eviction_offchip_memory_pj": _safe_num(getattr(scme, "total_eviction_to_offchip_memory_energy", 0)),
            "energy_sink_offchip_link_pj": _safe_num(getattr(scme, "total_sink_layer_output_offchip_link_energy", 0)),
            "energy_sink_offchip_memory_pj": _safe_num(getattr(scme, "total_sink_layer_output_offchip_memory_energy", 0)),
            "energy_core_to_core_link_pj": _safe_num(getattr(scme, "total_core_to_core_link_energy", 0)),
            "energy_core_to_core_memory_pj": _safe_num(getattr(scme, "total_core_to_core_memory_energy", 0)),
        }
    ]
    summary_df = pd.DataFrame(summary_rows)

    node_rows: list[dict[str, Any]] = []
    for node in sorted(_iter_nodes(scme.workload), key=lambda n: (getattr(n, "id", -1), getattr(n, "sub_id", -1))):
        chosen_core = getattr(node, "chosen_core_allocation", None)
        node_rows.append(
            {
                "node_id": getattr(node, "id", None),
                "sub_id": getattr(node, "sub_id", None),
                "name": getattr(node, "name", ""),
                "operator_type": getattr(node, "type", ""),
                "chosen_core": chosen_core,
                "runtime_cycles": _safe_num(getattr(node, "runtime", 0)),
                "onchip_energy_pj": _safe_num(getattr(node, "onchip_energy", 0)),
                "offchip_energy_pj": _safe_num(getattr(node, "offchip_energy", 0)),
            }
        )

    nodes_df = pd.DataFrame(node_rows)
    op_summary_df = (
        nodes_df.groupby(["operator_type", "chosen_core"], dropna=False, as_index=False)[
            ["runtime_cycles", "onchip_energy_pj", "offchip_energy_pj"]
        ]
        .sum()
        .sort_values(["runtime_cycles", "onchip_energy_pj"], ascending=False)
    )

    summary_df.to_csv(RESULTS_DIR / "stream_summary.csv", index=False)
    nodes_df.to_csv(RESULTS_DIR / "stream_node_stats.csv", index=False)
    op_summary_df.to_csv(RESULTS_DIR / "stream_operator_summary.csv", index=False)

    print("Wrote", RESULTS_DIR / "stream_summary.csv")
    print("Wrote", RESULTS_DIR / "stream_node_stats.csv")
    print("Wrote", RESULTS_DIR / "stream_operator_summary.csv")


def main() -> None:
    scme_path = _latest_scme_path()
    analyze(scme_path)


if __name__ == "__main__":
    main()
