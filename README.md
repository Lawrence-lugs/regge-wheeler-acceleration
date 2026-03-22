# Modelling PINN Accelerator w/ SIMD and TPU-like Array sharing L1 memory

- `regge_wheeler_v2.py` contains a Pytorch/Numpy/Scipy simulation of using PINNs to solve the Regge-Wheeler equations.

- `perf_analyis` contains a subproject that models the execution latency of the Regge-Wheeler simulation when limited to any combination of Scalar, Vector, and Matrix primitives.
    - Models latency of reading the shared data from L1
    - TPU assumed to already contain the weight hierarchy (consistent with an assumption that we train the weights on the spot).

# Issues (in README for now)

- [ ] Updating the weights should be done by the Vector processor
- [ ] No tiling performed to account for mismatch of TPU size vs Matrix size... 

