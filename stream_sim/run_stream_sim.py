from __future__ import annotations

import re
from pathlib import Path

import onnx
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

from .scripts.build_pinn_workload_onnx import build_workload


ROOT = Path(__file__).resolve().parent
HARDWARE_PATH = ROOT / "inputs" / "hardware" / "pinn_vector_tpu_shared_l1.yaml"
MAPPING_PATH = ROOT / "inputs" / "mapping" / "pinn_vector_tpu_mapping.yaml"
WORKLOAD_PATH = ROOT / "inputs" / "workload" / "pinn_workload.onnx"
OUTPUTS_DIR = ROOT / "outputs"


def _experiment_id() -> str:
    hw_name = HARDWARE_PATH.stem
    wl_name = re.split(r"/|\\.", str(WORKLOAD_PATH))[-2]
    return f"{hw_name}-{wl_name}-lbl-genetic_algorithm"


def main() -> None:
    build_workload(WORKLOAD_PATH)

    model = onnx.load(str(WORKLOAD_PATH))
    layer_stacks = [(idx,) for idx in range(len(model.graph.node))]
    experiment_id = _experiment_id()

    scme = optimize_allocation_ga(
        hardware=str(HARDWARE_PATH),
        workload=str(WORKLOAD_PATH),
        mapping=str(MAPPING_PATH),
        mode="lbl",
        layer_stacks=layer_stacks,
        nb_ga_generations=4,
        nb_ga_individuals=4,
        experiment_id=experiment_id,
        output_path=str(OUTPUTS_DIR),
        skip_if_exists=False,
    )

    run_dir = OUTPUTS_DIR / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cost_lut = CostModelEvaluationLUT(str(run_dir / "cost_lut.pickle"))
    try:
        plot_memory_usage(scme, section_start_percent=(0,), percent_shown=(100,), fig_path=str(run_dir / "memory.png"))
    except ValueError as exc:
        print("Skipping memory plot:", exc)

    convert_scme_to_perfetto_json(scme, cost_lut, json_path=str(run_dir / "scme.json"))

    print("Experiment ID:", experiment_id)
    print("Latency (cycles):", scme.latency)
    print("Energy (pJ):", scme.energy)
    print("Run outputs in", run_dir)


if __name__ == "__main__":
    main()
