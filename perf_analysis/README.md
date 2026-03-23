# Regge-Wheeler v2 Performance Analysis

This package models the toy `regger_wheeler_v2.py` workload in terms of scalar, vector, and matrix primitives and estimates latency under four capability profiles:

- `scalar_matrix_only`
- `scalar_vector_only`
- `scalar_only`
- `all_primitives`

## What It Covers

- Finite-difference potential construction and time stepping.
- PINN forward/training surrogate based on the original Fourier embedding and dense layers.
- Observer extraction and a DFT-style surrogate for the strain spectrum stage.

## Primitive Model

- Scalar primitives use names such as `fp32_scalar_add`.
- Vector primitives use names such as `8x_fp32_vector_add`.
- Matrix primitives use names such as `16x16x16_fp32_matrix_multiply`.
- Every primitive call appends a row to the global statistics table with invocation count and estimated latency.
- Vector operations explicitly account for chunking and tail loops when the data length exceeds the configured vector width.

## PINN Training Model Versions

Two training-cost models are available, selected via `AnalysisConfig.pinn_model_version`:

| Version | `pinn_model_version` | Description |
|---------|----------------------|-------------|
| v2 autograd | `"v2_autograd"` (default) | Mirrors `regge_wheeler_v2.py`. PDE derivatives obtained via `torch.autograd.grad(create_graph=True)`, which requires building and traversing a higher-order computation graph. The overhead is amortised into the forward-pass surrogate via `pinn_forward_pass_equivalents` (default 5). One standard backward pass for the parameter update. |
| v3 RK1 FD | `"v3_rk1_fd"` | Mirrors `regge_wheeler_v3.py`. PDE derivatives are approximated with a central finite-difference stencil packed into the batch dimension: center, t+Δt, t-Δt, x+Δx, x-Δx. This keeps one forward pass and one backward pass per training step. No `create_graph` autograd is required. `pinn_rk1_fd_stencil_evals` (default 5) is a batch expansion factor, not extra forward repetitions. |

To run with the v3 model, pass the config explicitly:

```python
from perf_analysis.config import AnalysisConfig
from perf_analysis.run_analysis import run_all_experiments, write_results, make_total_latency_plot

config = AnalysisConfig(pinn_model_version="v3_rk1_fd")
config.validate()
frames, summary = run_all_experiments(config)
write_results(frames, summary)
make_total_latency_plot(summary)
```

Stats output sections that distinguish the two models:

- v2: `pinn.forward_surrogate.*`, `pinn.backward.*`, `pinn.optimizer`
- v3: `pinn_rk1.stencil_forward.*`, `pinn_rk1.fd_combine.*`, `pinn_rk1.backward.*`, `pinn.optimizer`

## Global Parameters

The main parameters are in [config.py](/workspaces/torch_dev/perf_analysis/config.py):

- Vector width is controlled by `VectorConfig.max_lanes` and defaults to `8` lanes of `fp32`.
- Latency defaults are scalar `1`, vector `6`, matrix `40`.
- Workload sizes default to the toy script values: `Nx=200`, `Nt=400`, `N_colloc=3000`, `epochs=2000`.

## Assumptions

- The Lambert W inverse in the Regge-Wheeler potential is approximated with a Newton solve expressed only in elementary operations.
- v2: PyTorch autograd internals are approximated with explicit forward/backward surrogate kernels for the dense network and residual construction.  `pinn_forward_pass_equivalents = 5` encodes 1 actual forward pass plus 4 forward-equivalent costs for the create_graph derivative chain.
- v3: Finite-difference stencil points are packed into the batch as `effective_batch = pinn_collocation_points * pinn_rk1_fd_stencil_evals` with one forward call per step and no create_graph overhead.
- The FFT stage is represented as a DFT-style matrix surrogate because the target primitive set is scalar/vector/matrix rather than butterfly-specific FFT primitives.

## Run

From the workspace root:

```bash
python -m perf_analysis.run_analysis
```

Outputs are written into `perf_analysis/results/`:

- Four detailed primitive statistics CSVs.
- Four per-experiment primitive summaries.
- One latency summary CSV.
- Two plots:
  - `latency_totals.png`
  - `latency_ablation.png`