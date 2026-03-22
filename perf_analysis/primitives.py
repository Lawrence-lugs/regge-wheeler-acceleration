from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import AnalysisConfig, CapabilityProfile
from .stats import append_stat_row


BinaryOp = Callable[[np.ndarray, np.ndarray], np.ndarray]
UnaryOp = Callable[[np.ndarray], np.ndarray]


def _to_fp32(value: np.ndarray | float | int) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


@dataclass
class PrimitiveExecutor:
    config: AnalysisConfig
    capabilities: CapabilityProfile

    def __post_init__(self) -> None:
        self.vector_lanes = self.config.vector.max_lanes
        self.vector_name_prefix = f"{self.vector_lanes}x_{self.config.vector.dtype_name}"
        matrix = self.config.matrix
        self.matrix_name_prefix = (
            f"{matrix.tile_m}x{matrix.tile_n}x{matrix.tile_k}_{matrix.dtype_name}"
        )

    # ------------------------------------------------------------------
    # Memory-bandwidth helper
    # ------------------------------------------------------------------

    def _memory_cycles(self, total_elements: int) -> float:
        mem = self.config.memory
        return math.ceil(total_elements * mem.dtype_bits / mem.l1_bandwidth_bits)

    # ------------------------------------------------------------------
    # Public primitive API
    # ------------------------------------------------------------------

    def binary(
        self,
        op_name: str,
        left: np.ndarray | float | int,
        right: np.ndarray | float | int,
        *,
        section: str,
        repetitions: int = 1,
    ) -> np.ndarray:
        left_arr = _to_fp32(left)
        right_arr = _to_fp32(right)
        result = self._binary_numpy(op_name, left_arr, right_arr)
        logical_shape = np.broadcast_shapes(left_arr.shape, right_arr.shape)
        total_elements = int(np.prod(logical_shape)) if logical_shape else 1
        if self.capabilities.has_vector and total_elements > 1:
            self._log_vector_chunks(
                op_name=op_name,
                section=section,
                total_elements=total_elements,
                repetitions=repetitions,
                logical_shape=logical_shape,
                num_input_operands=2,
                fallback_from="",
                notes="elementwise",
            )
        else:
            self._log_scalar(
                op_name=op_name,
                section=section,
                invocations=total_elements * repetitions,
                total_data_elements=total_elements * 3,
                logical_shape=logical_shape,
                fallback_from="vector" if total_elements > 1 else "",
                notes="elementwise",
            )
        return result.astype(np.float32, copy=False)

    def unary(
        self,
        op_name: str,
        value: np.ndarray | float | int,
        *,
        section: str,
        repetitions: int = 1,
    ) -> np.ndarray:
        value_arr = _to_fp32(value)
        result = self._unary_numpy(op_name, value_arr)
        logical_shape = value_arr.shape
        total_elements = int(value_arr.size)
        if self.capabilities.has_vector and total_elements > 1:
            self._log_vector_chunks(
                op_name=op_name,
                section=section,
                total_elements=total_elements,
                repetitions=repetitions,
                logical_shape=logical_shape,
                num_input_operands=1,
                fallback_from="",
                notes="unary",
            )
        else:
            self._log_scalar(
                op_name=op_name,
                section=section,
                invocations=total_elements * repetitions,
                total_data_elements=total_elements * 2,
                logical_shape=logical_shape,
                fallback_from="vector" if total_elements > 1 else "",
                notes="unary",
            )
        return result.astype(np.float32, copy=False)

    def reduce_sum(
        self,
        value: np.ndarray,
        *,
        section: str,
        repetitions: int = 1,
    ) -> np.ndarray:
        value_arr = _to_fp32(value)
        length = int(value_arr.shape[-1]) if value_arr.ndim else 1
        outer = int(value_arr.size / max(length, 1))
        total_repetitions = outer * repetitions
        result = np.sum(value_arr, axis=-1, dtype=np.float32)

        if self.capabilities.has_vector and length > 1:
            self._log_vector_chunks(
                op_name="reduce_sum",
                section=section,
                total_elements=length,
                repetitions=total_repetitions,
                logical_shape=(length,),
                num_input_operands=1,
                fallback_from="",
                notes="reduction",
            )
            chunk_count = math.ceil(length / self.vector_lanes)
            partial_adds = max(chunk_count - 1, 0) * total_repetitions
            if partial_adds:
                self._log_scalar(
                    op_name="add",
                    section=section,
                    invocations=partial_adds,
                    total_data_elements=partial_adds * 3,
                    logical_shape=(1,),
                    fallback_from="vector_reduce_finalize",
                    notes="combine vector partial sums",
                )
        else:
            scalar_invocations = max(length - 1, 0) * total_repetitions
            self._log_scalar(
                op_name="add",
                section=section,
                invocations=scalar_invocations,
                total_data_elements=length * total_repetitions + total_repetitions,
                logical_shape=(length,),
                fallback_from="vector_reduce" if length > 1 else "",
                notes="scalar reduction",
            )
        return result.astype(np.float32, copy=False)

    def matmul(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        section: str,
        repetitions: int = 1,
    ) -> np.ndarray:
        left_arr = _to_fp32(left)
        right_arr = _to_fp32(right)
        result = left_arr @ right_arr
        if left_arr.ndim != 2 or right_arr.ndim != 2:
            raise ValueError("matmul expects rank-2 operands.")
        m_dim, k_dim = left_arr.shape
        k_rhs, n_dim = right_arr.shape
        if k_dim != k_rhs:
            raise ValueError("matmul operands have incompatible shapes.")

        if self.capabilities.has_matrix:
            tile = self.config.matrix
            tile_calls = (
                math.ceil(m_dim / tile.tile_m)
                * math.ceil(n_dim / tile.tile_n)
                * math.ceil(k_dim / tile.tile_k)
                * repetitions
            )
            lat_per_tile = self.config.latencies.matrix_tile_latency(
                tile.tile_m, tile.tile_k, tile.tile_n,
            )
            compute_latency = tile_calls * lat_per_tile

            # Memory: activations (m*k) + output (m*n) always from L1.
            # Weights (k*n) only if NOT local to the matrix unit.
            mem_elements = m_dim * k_dim + m_dim * n_dim
            if not self.config.memory.weights_local_to_matrix_unit:
                mem_elements += k_dim * n_dim
            memory_latency = float(self._memory_cycles(mem_elements) * repetitions)
            estimated = max(compute_latency, memory_latency)

            append_stat_row(
                {
                    "experiment": self.capabilities.name,
                    "section": section,
                    "primitive_kind": "matrix",
                    "primitive_name": f"{self.matrix_name_prefix}_matrix_multiply",
                    "logical_operation": "matmul",
                    "shape": f"({m_dim}, {k_dim}) x ({k_dim}, {n_dim})",
                    "chunk_shape": f"({tile.tile_m}, {tile.tile_k}) x ({tile.tile_k}, {tile.tile_n})",
                    "elements_per_invocation": tile.tile_m * tile.tile_n,
                    "invocations": tile_calls,
                    "latency_per_invocation": lat_per_tile,
                    "compute_latency": compute_latency,
                    "memory_latency": memory_latency,
                    "estimated_latency": estimated,
                    "fallback_from": "",
                    "notes": "tiled matrix primitive",
                }
            )
            return result.astype(np.float32, copy=False)

        dot_products = m_dim * n_dim
        if self.capabilities.has_vector:
            sample = np.ones((dot_products, k_dim), dtype=np.float32)
            self.binary(
                "multiply",
                sample,
                sample,
                section=section,
                repetitions=repetitions,
            )
            self.reduce_sum(sample, section=section, repetitions=repetitions)
            return result.astype(np.float32, copy=False)

        scalar_mul_invocations = dot_products * k_dim * repetitions
        scalar_add_invocations = dot_products * max(k_dim - 1, 0) * repetitions
        total_data = (m_dim * k_dim + k_dim * n_dim + m_dim * n_dim) * repetitions
        self._log_scalar(
            op_name="multiply",
            section=section,
            invocations=scalar_mul_invocations,
            total_data_elements=total_data,
            logical_shape=(m_dim, n_dim, k_dim),
            fallback_from="matrix",
            notes="scalar matmul multiply fallback",
        )
        self._log_scalar(
            op_name="add",
            section=section,
            invocations=scalar_add_invocations,
            total_data_elements=0,
            logical_shape=(m_dim, n_dim, k_dim),
            fallback_from="matrix",
            notes="scalar matmul accumulation fallback",
        )
        return result.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------

    def _log_vector_chunks(
        self,
        *,
        op_name: str,
        section: str,
        total_elements: int,
        repetitions: int,
        logical_shape: tuple[int, ...],
        num_input_operands: int,
        fallback_from: str,
        notes: str,
    ) -> None:
        lat_per_inv = self.config.latencies.vector_latency(op_name)
        data_elements_per_rep = total_elements * (num_input_operands + 1)
        memory_latency_total = float(self._memory_cycles(data_elements_per_rep) * repetitions)

        full_chunks, remainder = divmod(total_elements, self.vector_lanes)

        total_invocations = full_chunks * repetitions
        if remainder:
            total_invocations += repetitions
        compute_latency_total = total_invocations * lat_per_inv
        estimated = max(compute_latency_total, memory_latency_total)

        if full_chunks:
            invocations = full_chunks * repetitions
            frac = invocations / total_invocations if total_invocations else 1.0
            append_stat_row(
                {
                    "experiment": self.capabilities.name,
                    "section": section,
                    "primitive_kind": "vector",
                    "primitive_name": f"{self.vector_name_prefix}_vector_{op_name}",
                    "logical_operation": op_name,
                    "shape": str(logical_shape),
                    "chunk_shape": str((self.vector_lanes,)),
                    "elements_per_invocation": self.vector_lanes,
                    "invocations": invocations,
                    "latency_per_invocation": lat_per_inv,
                    "compute_latency": invocations * lat_per_inv,
                    "memory_latency": memory_latency_total * frac,
                    "estimated_latency": estimated * frac,
                    "fallback_from": fallback_from,
                    "notes": notes,
                }
            )
        if remainder:
            invocations = repetitions
            frac = invocations / total_invocations if total_invocations else 1.0
            append_stat_row(
                {
                    "experiment": self.capabilities.name,
                    "section": section,
                    "primitive_kind": "vector",
                    "primitive_name": f"{self.vector_name_prefix}_vector_{op_name}",
                    "logical_operation": op_name,
                    "shape": str(logical_shape),
                    "chunk_shape": str((remainder,)),
                    "elements_per_invocation": remainder,
                    "invocations": invocations,
                    "latency_per_invocation": lat_per_inv,
                    "compute_latency": invocations * lat_per_inv,
                    "memory_latency": memory_latency_total * frac,
                    "estimated_latency": estimated * frac,
                    "fallback_from": fallback_from,
                    "notes": f"{notes}; tail loop",
                }
            )

    def _log_scalar(
        self,
        *,
        op_name: str,
        section: str,
        invocations: int,
        total_data_elements: int,
        logical_shape: tuple[int, ...],
        fallback_from: str,
        notes: str,
    ) -> None:
        if invocations <= 0:
            return
        lat_per_inv = self.config.latencies.scalar_latency(op_name)
        compute_latency = invocations * lat_per_inv
        memory_latency = float(self._memory_cycles(total_data_elements))
        estimated = max(compute_latency, memory_latency)
        append_stat_row(
            {
                "experiment": self.capabilities.name,
                "section": section,
                "primitive_kind": "scalar",
                "primitive_name": f"fp32_scalar_{op_name}",
                "logical_operation": op_name,
                "shape": str(logical_shape),
                "chunk_shape": str((1,)),
                "elements_per_invocation": 1,
                "invocations": invocations,
                "latency_per_invocation": lat_per_inv,
                "compute_latency": compute_latency,
                "memory_latency": memory_latency,
                "estimated_latency": estimated,
                "fallback_from": fallback_from,
                "notes": notes,
            }
        )

    @staticmethod
    def _binary_numpy(op_name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        operations: dict[str, BinaryOp] = {
            "add": np.add,
            "sub": np.subtract,
            "multiply": np.multiply,
            "div": np.divide,
        }
        if op_name not in operations:
            raise ValueError(f"Unsupported binary operation: {op_name}")
        return operations[op_name](left, right, dtype=np.float32)

    @staticmethod
    def _unary_numpy(op_name: str, value: np.ndarray) -> np.ndarray:
        operations: dict[str, UnaryOp] = {
            "exp": np.exp,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "abs": np.abs,
        }
        if op_name not in operations:
            raise ValueError(f"Unsupported unary operation: {op_name}")
        return operations[op_name](value).astype(np.float32, copy=False)