from __future__ import annotations

from typing import Any

import pandas as pd


_GLOBAL_ROWS: list[dict[str, Any]] = []
_ROW_COUNTER = 0


def reset_global_stats() -> None:
    global _GLOBAL_ROWS, _ROW_COUNTER
    _GLOBAL_ROWS = []
    _ROW_COUNTER = 0


def append_stat_row(row: dict[str, Any]) -> None:
    global _ROW_COUNTER
    _ROW_COUNTER += 1
    enriched = {"row_id": _ROW_COUNTER}
    enriched.update(row)
    _GLOBAL_ROWS.append(enriched)


def get_global_stats_frame() -> pd.DataFrame:
    if not _GLOBAL_ROWS:
        return pd.DataFrame(
            columns=[
                "row_id",
                "experiment",
                "section",
                "primitive_kind",
                "primitive_name",
                "logical_operation",
                "shape",
                "chunk_shape",
                "elements_per_invocation",
                "invocations",
                "latency_per_invocation",
                "compute_latency",
                "memory_latency",
                "estimated_latency",
                "fallback_from",
                "notes",
            ]
        )
    return pd.DataFrame(_GLOBAL_ROWS)


def summarize_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["experiment", "primitive_kind", "compute_latency", "memory_latency", "estimated_latency"]
        )

    agg_cols = {"estimated_latency": "sum"}
    if "compute_latency" in frame.columns:
        agg_cols["compute_latency"] = "sum"
    if "memory_latency" in frame.columns:
        agg_cols["memory_latency"] = "sum"

    summary = (
        frame.groupby(["experiment", "primitive_kind"], dropna=False, as_index=False)
        .agg(agg_cols)
        .sort_values(["experiment", "primitive_kind"])
    )
    return summary