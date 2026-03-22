from __future__ import annotations

from pathlib import Path

import onnx
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

from stream_sim.config import EXPERIMENTS, OUTPUTS_DIR, WORKLOAD_PATH
from stream_sim.scripts.build_pinn_workload_onnx import build_workload


def run_single_experiment(experiment_name: str):
    experiment = next(exp for exp in EXPERIMENTS if exp.name == experiment_name)
    build_workload(WORKLOAD_PATH)
    model = onnx.load(str(WORKLOAD_PATH))
    layer_stacks = [(idx,) for idx in range(len(model.graph.node))]

    scme = optimize_allocation_ga(
        hardware=str(experiment.hardware_path),
        workload=str(WORKLOAD_PATH),
        mapping=str(experiment.mapping_path),
        mode="lbl",
        layer_stacks=layer_stacks,
        nb_ga_generations=1,
        nb_ga_individuals=1,
        experiment_id=experiment.experiment_id,
        output_path=str(OUTPUTS_DIR),
        skip_if_exists=False,
    )

    run_dir = OUTPUTS_DIR / experiment.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cost_lut = CostModelEvaluationLUT(str(run_dir / "cost_lut.pickle"))
    try:
        plot_memory_usage(scme, section_start_percent=(0,), percent_shown=(100,), fig_path=str(run_dir / "memory.png"))
    except ValueError as exc:
        print(f"[{experiment.label}] Skipping memory plot: {exc}")

    convert_scme_to_perfetto_json(scme, cost_lut, json_path=str(run_dir / "scme.json"))

    print(f"[{experiment.label}] Experiment ID: {experiment.experiment_id}")
    print(f"[{experiment.label}] Latency (cycles): {scme.latency}")
    print(f"[{experiment.label}] Energy (pJ): {scme.energy}")
    print(f"[{experiment.label}] Run outputs in {run_dir}")
    return scme


def main() -> None:
    build_workload(WORKLOAD_PATH)
    for experiment in EXPERIMENTS:
        try:
            run_single_experiment(experiment.name)
        except Exception as exc:  # noqa: BLE001
            run_dir = OUTPUTS_DIR / experiment.experiment_id
            run_dir.mkdir(parents=True, exist_ok=True)
            failure_path = run_dir / "run_failure.txt"
            failure_path.write_text(str(exc) + "\n", encoding="utf-8")
            print(f"[{experiment.label}] FAILED: {exc}")
            print(f"[{experiment.label}] Wrote failure marker to {failure_path}")


if __name__ == "__main__":
    main()
