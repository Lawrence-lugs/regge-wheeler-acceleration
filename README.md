# Modelling PINN Accelerator w/ SIMD and TPU-like Array sharing L1 memory

- `regge_wheeler_v2.py` contains a Pytorch/Numpy/Scipy simulation of using PINNs to solve the Regge-Wheeler equations.
- `regge_wheeler_v3.py` - the v2 version uses 5 backward passes. 4 for autodifferentiation and 1 for loss propagation. We replace that in v3 by enforcing finite difference stencil points to be in a single input batch. Then, we use these points to compute the gradient via RK1.

- `perf_analyis` contains a subproject that models the execution latency of the Regge-Wheeler simulation when limited to any combination of Scalar, Vector, and Matrix primitives.
    - Models latency of reading the shared data from L1
    - Systolic array delay model for TPU
    - TPU assumed to already contain the weight hierarchy (consistent with an assumption that we train the weights on the spot).
    - Tiling is performed if Matmul dim is mismatched. The software overhead of managing the tiling is not accounted for. 

- `stream_sim` models the accelerator in [Stream]().
    - Currently WIP.
    - Bug in trying to support SIMD-like nodes.
    - Example: Random Fourier projection MVM or Sine for single vector can't be mapped if W=1,H=H...

# Issues (in README for now)

- [ ] Fix problems with stream_sim
- [ ] Regge Wheeler v3 with FD stencils for loss computation isn't working so well...

# Sample Outputs
## Regge-Wheeler Comparison

![Regge-Wheeler Comparison](./regge_wheeler_comparison.pdf)

## Performance Analysis Visualizations

![Concatenated Heatmap](./concatenated_heatmap.png)

![Ara Scatter Plot](./ara_scatter.png)
