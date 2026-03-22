from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
HARDWARE_DIR = INPUTS_DIR / "hardware"
MAPPING_DIR = INPUTS_DIR / "mapping"
WORKLOAD_DIR = INPUTS_DIR / "workload"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
WORKLOAD_PATH = WORKLOAD_DIR / "regge_wheeler_v2_surrogate.onnx"


@dataclass(frozen=True)
class WorkloadConfig:
    nx: int = 200
    nt: int = 400
    pinn_collocation_points: int = 300
    pinn_epochs: int = 200
    embedding_features: int = 64
    hidden_width: int = 128
    hidden_layers: int = 2
    output_width: int = 1
    lambert_newton_steps: int = 6
    mass: float = 1.0
    angular_l: int = 2
    spin_s: int = 2
    r_star_min: float = 2.0
    r_star_max: float = 30.0
    t_max: float = 20.0

    @property
    def fourier_half(self) -> int:
        return self.embedding_features // 2

    @property
    def fft_bins(self) -> int:
        return max((self.nt // 2) - 1, 1)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    label: str
    hardware_path: Path
    mapping_path: Path
    scalar_core_ids: tuple[int, ...]
    vector_core_ids: tuple[int, ...]
    matrix_core_ids: tuple[int, ...]

    @property
    def experiment_id(self) -> str:
        return f"{self.name}-{WORKLOAD_PATH.stem}-lbl-genetic_algorithm"


WORKLOAD_CONFIG = WorkloadConfig()

EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        name="scalar_only",
        label="Scalar",
        hardware_path=HARDWARE_DIR / "rw_scalar_only_shared_l1.yaml",
        mapping_path=MAPPING_DIR / "rw_scalar_only.yaml",
        scalar_core_ids=(0,),
        vector_core_ids=(),
        matrix_core_ids=(),
    ),
    ExperimentConfig(
        name="scalar_tpu",
        label="Scalar+TPU",
        hardware_path=HARDWARE_DIR / "rw_scalar_tpu_shared_l1.yaml",
        mapping_path=MAPPING_DIR / "rw_scalar_tpu.yaml",
        scalar_core_ids=(0,),
        vector_core_ids=(),
        matrix_core_ids=(1,),
    ),
    ExperimentConfig(
        name="vector_only",
        label="Vector",
        hardware_path=HARDWARE_DIR / "rw_vector_only_shared_l1.yaml",
        mapping_path=MAPPING_DIR / "rw_vector_only.yaml",
        scalar_core_ids=(),
        vector_core_ids=(0,),
        matrix_core_ids=(),
    ),
    ExperimentConfig(
        name="vector_tpu",
        label="Vector+TPU",
        hardware_path=HARDWARE_DIR / "rw_vector_tpu_shared_l1.yaml",
        mapping_path=MAPPING_DIR / "rw_vector_tpu.yaml",
        scalar_core_ids=(2,),
        vector_core_ids=(0,),
        matrix_core_ids=(1,),
    ),
)
