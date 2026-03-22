from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorConfig:
    max_lanes: int = 8
    supported_lanes: tuple[int, ...] = (4, 8, 16)
    dtype_name: str = "fp32"

    def validate(self) -> None:
        if self.max_lanes not in self.supported_lanes:
            raise ValueError(
                f"Unsupported vector length {self.max_lanes}. "
                f"Choose one of {self.supported_lanes}."
            )

@dataclass(frozen=True)
class MatrixConfig:
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    dtype_name: str = "fp32"


@dataclass(frozen=True)
class LatencyConfig:
    """Per-instruction latency model.

    Most pipelined vector/scalar instructions run at 1 cycle.  Stalling
    instructions (div, sqrt) carry an explicit per-op override.  Matrix
    tile latency is computed from the systolic drain formula
    ``tile_m + tile_k + tile_n - 2``.
    """

    scalar_default: float = 1.0
    vector_default: float = 1.0
    scalar_overrides: tuple[tuple[str, float], ...] = ()
    vector_overrides: tuple[tuple[str, float], ...] = (
        ("div", 5.0),
        ("sqrt", 5.0),
    )

    def scalar_latency(self, op_name: str) -> float:
        for name, lat in self.scalar_overrides:
            if name == op_name:
                return lat
        return self.scalar_default

    def vector_latency(self, op_name: str) -> float:
        for name, lat in self.vector_overrides:
            if name == op_name:
                return lat
        return self.vector_default

    def matrix_tile_latency(self, tile_m: int, tile_k: int, tile_n: int) -> float:
        return float(tile_m + tile_k + tile_n - 2)


@dataclass(frozen=True)
class MemoryConfig:
    """L1 memory-bandwidth model.

    All starting tensors are assumed to reside in shared L1.  The
    bandwidth (``l1_bandwidth_bits``) limits how fast data can be
    transferred to/from the compute cores.  Weights inside the TPU/GEMM
    unit are assumed local and do *not* traverse the L1 bus
    (``weights_local_to_matrix_unit``).
    """

    l1_bandwidth_bits: int = 1024
    dtype_bits: int = 32
    weights_local_to_matrix_unit: bool = True


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    has_vector: bool
    has_matrix: bool


@dataclass(frozen=True)
class AnalysisConfig:
    mass: float = 1.0
    angular_l: int = 2
    spin_s: int = 2
    r_star_min: float = 2.0
    r_star_max: float = 30.0
    t_max: float = 20.0
    nx: int = 200
    nt: int = 400
    observer_r_star: float = 20.0
    pinn_collocation_points: int = 3000
    pinn_epochs: int = 2000
    embedding_features: int = 64
    hidden_width: int = 128
    hidden_layers: int = 2
    output_width: int = 1
    lambert_newton_steps: int = 6
    pinn_forward_pass_equivalents: int = 5
    pinn_backward_pass_equivalents: int = 1
    adam_elementwise_ops: int = 8
    dft_surrogate_repetitions: int = 2
    random_seed: int = 7
    vector: VectorConfig = field(default_factory=VectorConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    latencies: LatencyConfig = field(default_factory=LatencyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def validate(self) -> None:
        self.vector.validate()
        if self.embedding_features % 2 != 0:
            raise ValueError("embedding_features must be even for sin/cos Fourier features.")


EXPERIMENTS: tuple[CapabilityProfile, ...] = (
    CapabilityProfile(name="scalar_matrix_only", has_vector=False, has_matrix=True),
    CapabilityProfile(name="scalar_vector_only", has_vector=True, has_matrix=False),
    CapabilityProfile(name="scalar_only", has_vector=False, has_matrix=False),
    CapabilityProfile(name="all_primitives", has_vector=True, has_matrix=True),
)