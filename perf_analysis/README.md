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

## Global Parameters

The main parameters are in [config.py](/workspaces/torch_dev/perf_analysis/config.py):

- Vector width is controlled by `VectorConfig.max_lanes` and defaults to `8` lanes of `fp32`.
- Latency defaults are scalar `1`, vector `6`, matrix `40`.
- Workload sizes default to the toy script values: `Nx=200`, `Nt=400`, `N_colloc=3000`, `epochs=2000`.

## Assumptions

- The Lambert W inverse in the Regge-Wheeler potential is approximated with a Newton solve expressed only in elementary operations.
- PyTorch autograd internals are approximated with explicit forward/backward surrogate kernels for the dense network and residual construction.
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