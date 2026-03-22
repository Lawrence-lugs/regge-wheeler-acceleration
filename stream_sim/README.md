# Stream Simulation Subproject

This subproject instantiates a Stream-based model of a coupled accelerator:

- Core 0: TPU-like matrix core (GEMM focused)
- Core 1: SIMD/vector core (elementwise focused)
- Core 2: Shared L1 modeled as Stream "offchip" memory

The coupling between core 0 and core 1 is only through the shared L1 (modeled as the offchip core in Stream).

## Layout

- `inputs/hardware/pinn_vector_tpu_shared_l1.yaml`: top-level hardware graph.
- `inputs/hardware/cores/tpu_like_matmul_core.yaml`: matrix core.
- `inputs/hardware/cores/vector_simd_core.yaml`: vector core.
- `inputs/hardware/cores/shared_l1_offchip.yaml`: shared L1 memory modeled as offchip memory core.
- `inputs/mapping/pinn_vector_tpu_mapping.yaml`: operator-to-core mapping.
- `inputs/workload/pinn_workload.yaml`: preliminary manual workload draft for reference.
- `scripts/build_pinn_workload_onnx.py`: builds Stream-compatible ONNX graph for the Regge-Wheeler surrogate flow.
- `run_stream_sim.py`: executes Stream with GA allocation.
- `analyze_results.py`: extracts latency and energy summaries similar to `perf_analysis`.

## Why ONNX Instead Of Manual YAML Workload

The currently published Stream package path for manual YAML workloads is not complete (`WorkloadFactoryStream` raises `NotImplementedError`).
This subproject therefore uses ONNX for execution while preserving YAML for hardware and mapping definitions.

## Run

From workspace root:

```bash
python -m stream_sim.run_stream_sim
python -m stream_sim.analyze_results
```

Outputs are written to:

- `stream_sim/outputs/<experiment_id>/` (native Stream artifacts)
- `stream_sim/results/stream_summary.csv`
- `stream_sim/results/stream_node_stats.csv`
- `stream_sim/results/stream_operator_summary.csv`
