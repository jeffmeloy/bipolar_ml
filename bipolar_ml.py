import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import Counter, defaultdict
from typing import (
    List,
    Tuple,
    Optional,
    Set,
    Dict,
    Union,
    overload,
    Any,
    FrozenSet,
)
from dataclasses import dataclass, field
import time
from itertools import combinations
import logging
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
import copy
from functools import lru_cache
from scipy import stats
import random

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MemorySafetyError(Exception):
    """Custom exception for memory safety limits."""

    pass


class Bipolar:
    """A bipolar (-1/+1) vector type optimized for HDC operations with bit-packed storage."""

    __slots__ = ["_data", "_shape", "_buffer"]

    def __init__(
        self,
        data: Union[torch.Tensor, "Bipolar", List[bool], List[int], List[float]],
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize Bipolar object with bit-packed storage."""
        # Reusable buffer for operations
        self._buffer = None

        # Fast path for Bipolar copying
        if isinstance(data, Bipolar):
            self._data = data._data.clone()
            self._shape = data._shape
            if device is not None:
                self._data = self._data.to(device)
            return

        # Handle tensor inputs
        if isinstance(data, torch.Tensor):
            # Fast path for packed int8 tensors with shape provided
            if data.dtype == torch.int8 and shape is not None:
                self._data = data.to(device) if device is not None else data.clone()
                self._shape = shape
                return

            # Determine target shape
            if shape is None:
                if data.dtype == torch.bool:
                    self._shape = data.shape
                else:
                    raise ValueError(
                        "Shape must be provided for non-boolean tensor initialization"
                    )
            else:
                self._shape = shape

            # Convert to boolean representation for packing
            if data.dtype == torch.bool:
                bool_tensor = data
            elif data.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                bool_tensor = data > 0
            elif data.dtype in (torch.float16, torch.float32, torch.float64):
                bool_tensor = torch.sign(data) > 0
            else:
                raise TypeError(f"Unsupported tensor dtype for Bipolar: {data.dtype}")

            # Move to target device before packing
            if device is not None:
                bool_tensor = bool_tensor.to(device)

            # Pack the boolean tensor
            self._data = self._pack_efficient(bool_tensor)
            return

        # Handle list inputs - convert to tensor first
        if isinstance(data, list):
            # Determine the type and convert to boolean tensor
            if all(isinstance(x, bool) for x in data):
                bool_tensor = torch.tensor(data, dtype=torch.bool, device=device)
            else:
                bool_tensor = torch.tensor(
                    [x > 0 for x in data], dtype=torch.bool, device=device
                )

            # Use provided shape or infer from list
            self._shape = shape if shape is not None else (len(data),)

            # Pack the boolean tensor
            self._data = self._pack_efficient(bool_tensor)
            return

        raise TypeError(f"Cannot convert {type(data)} to Bipolar")

    def _pack_efficient(self, bool_tensor: torch.Tensor) -> torch.Tensor:
        """Pack a boolean tensor into an int8 tensor with minimal memory allocations."""
        flattened = bool_tensor.reshape(-1)
        total_bits = flattened.numel()

        # Calculate padded size and prepare bit weights
        padding = (
            8 - (total_bits % 8)
        ) % 8  # Avoid unnecessary padding when total_bits is multiple of 8
        padded_size = total_bits + padding

        # Pre-allocate padded tensor (only when needed)
        if padding > 0:
            padded = torch.zeros(
                padded_size, dtype=torch.bool, device=bool_tensor.device
            )
            padded[:total_bits] = flattened
        else:
            padded = flattened

        # Reshape and use efficient bit packing
        reshaped = padded.reshape(-1, 8)
        bit_weights = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int8, device=bool_tensor.device
        )

        # Use matrix multiplication for efficient packing
        return torch.matmul(reshaped.int(), bit_weights).to(torch.int8)

    def _unpack_efficient(self) -> torch.Tensor:
        """Unpack the int8 tensor into a boolean tensor."""
        device = self._data.device
        packed_size = self._data.numel()
        unpacked_bits = torch.zeros((packed_size, 8), dtype=torch.bool, device=device)

        # Vectorized unpacking using bitwise operations
        data_expanded = self._data.unsqueeze(1).expand(-1, 8)
        bit_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
        unpacked_bits = (data_expanded >> bit_positions) & 1

        # Reshape and trim to original size
        flattened = unpacked_bits.reshape(-1)
        original_size = torch.prod(torch.tensor(self._shape)).item()
        trimmed = flattened[:original_size]

        return trimmed.reshape(self._shape)

    def bind(self, other: "Bipolar", inplace: bool = False) -> "Bipolar":
        """Bind two vectors using bitwise XOR"""
        if not isinstance(other, Bipolar):
            raise TypeError("other must be a Bipolar object")

        # Direct operation on packed representation when shapes match
        if self._shape == other._shape:
            if inplace:
                self._data ^= other._data
                return self
            else:
                return Bipolar(
                    self._data ^ other._data,
                    shape=self._shape,
                    device=self._data.device,
                )

        # Fallback to unpacked operation for different shapes
        unpacked_self = self._unpack_efficient()
        unpacked_other = other._unpack_efficient()
        result_unpacked = unpacked_self ^ unpacked_other

        if inplace:
            self._data = self._pack_efficient(result_unpacked)
            self._shape = result_unpacked.shape
            return self
        else:
            return Bipolar(result_unpacked, device=self._data.device)

    def unbind(self, binder: "Bipolar", inplace: bool = False) -> "Bipolar":
        """Unbind operation is identical to bind for XOR."""
        return self.bind(binder, inplace=inplace)

    def distance(self, other: "Bipolar") -> torch.Tensor:
        """Calculate normalized Hamming distance"""
        if not isinstance(other, Bipolar):
            raise TypeError("other must be a Bipolar object")

        # Fast path for exact shape match - XOR in packed space and count bits
        if self._shape == other._shape:
            xor_result = self._data ^ other._data
            hamming_distance = self._count_bits_packed(xor_result) / self.__len__()
            return hamming_distance

        # Fallback to unpacked comparison for different shapes
        unpacked_self = self._unpack_efficient()
        unpacked_other = other._unpack_efficient()
        return (unpacked_self != unpacked_other).float().mean()

    def _count_bits_packed(self, packed_data: torch.Tensor) -> torch.Tensor:
        """Efficiently count bits in packed representation using popcount algorithm."""
        # Convert to uint8 for bit manipulation
        data = packed_data.to(torch.uint8)

        # Parallel bit counting algorithm (popcount)
        # Implementation of Hamming weight using SWAR algorithm
        data = data - ((data >> 1) & 0x55)
        data = (data & 0x33) + ((data >> 2) & 0x33)
        data = (data + (data >> 4)) & 0x0F

        # Sum all byte popcounts
        return data.sum().to(torch.float32)

    def similarity(self, other: "Bipolar") -> torch.Tensor:
        """Calculate similarity (1 - Hamming distance)."""
        return 1.0 - self.distance(other)

    def superposition(
        self, other: "Bipolar", alpha: float = 0.5, inplace: bool = False
    ) -> "Bipolar":
        """Superposition using weighted averaging, then thresholding."""
        if not isinstance(other, Bipolar):
            raise TypeError("other must be a Bipolar object")

        # Unpack, apply weighted average and threshold
        unpacked_self = self._unpack_efficient().float()
        unpacked_other = other._unpack_efficient().float()

        # Convert from boolean to bipolar (-1/+1) for weighted average
        bipolar_self = unpacked_self * 2 - 1
        bipolar_other = unpacked_other * 2 - 1

        # Apply weighted average and threshold back to boolean
        weighted_sum = alpha * bipolar_self + (1 - alpha) * bipolar_other
        result_unpacked = weighted_sum > 0

        if inplace:
            self._data = self._pack_efficient(result_unpacked)
            self._shape = result_unpacked.shape
            return self
        else:
            return Bipolar(result_unpacked, device=self._data.device)

    def randomize(self, flip_prob: float = 0.1) -> "Bipolar":
        """Randomly flip bits with specified probability (in-place)."""
        unpacked = self._unpack_efficient()

        # Generate random mask and apply XOR to flip bits
        device = self._data.device
        flip_mask = (
            torch.rand_like(unpacked, dtype=torch.float32, device=device) < flip_prob
        )
        unpacked[flip_mask] = ~unpacked[flip_mask]

        self._data = self._pack_efficient(unpacked)
        return self

    @classmethod
    def random(
        cls, size: Tuple[int, ...], device: Optional[Union[str, torch.device]] = None
    ) -> "Bipolar":
        """Create a Bipolar vector with random bits."""
        bool_tensor = torch.randint(0, 2, size, device=device, dtype=torch.bool)
        return cls(bool_tensor, device=device)

    def to(self, device: Union[str, torch.device]) -> "Bipolar":
        """Move the Bipolar vector to the specified device."""
        if self._data.device == device:
            return self

        self._data = self._data.to(device)

        # Move buffer if it exists
        if self._buffer is not None:
            self._buffer = self._buffer.to(device)

        return self

    def batch_distance(self, others: List["Bipolar"]) -> torch.Tensor:
        """Calculate distances between this vector and a batch of vectors."""
        if not all(isinstance(o, Bipolar) for o in others):
            raise TypeError("All elements in others must be Bipolar objects")

        # Fast path for exact shape match - operate in packed space
        if all(self._shape == o._shape for o in others):
            device = self._data.device
            batch_size = len(others)
            distances = torch.zeros(batch_size, device=device)

            # Vectorized XOR and bit counting
            packed_self = self._data
            total_bits = self.__len__()

            for i, other in enumerate(others):
                xor_result = packed_self ^ other._data
                distances[i] = self._count_bits_packed(xor_result) / total_bits

            return distances

        # Fallback to unpacked operation
        unpacked_self = self._unpack_efficient()
        batch_tensor_unpacked = torch.stack([o._unpack_efficient() for o in others])
        return (unpacked_self.unsqueeze(0) != batch_tensor_unpacked).float().mean(dim=1)

    def circular_shift(self, shift: int) -> "Bipolar":
        """Cyclic shift (circular convolution) of a Bipolar vector."""
        unpacked_vector = self._unpack_efficient()
        shifted_tensor = torch.roll(unpacked_vector, shifts=shift, dims=(-1,))
        return Bipolar(shifted_tensor, device=self.device)

    def convolve(
        self, sequence_vectors: List["Bipolar"], filter_vector: "Bipolar"
    ) -> List[float]:
        """Convolve using Hamming similarity with batch processing."""
        if not sequence_vectors:
            return []

        # Get dimensions
        filter_len = (
            filter_vector.shape[-1]
            if len(filter_vector.shape) > 0
            else len(filter_vector)
        )
        sequence_len = (
            sequence_vectors[0].shape[-1]
            if sequence_vectors and len(sequence_vectors[0].shape) > 0
            else (len(sequence_vectors[0]) if sequence_vectors else 0)
        )

        # Check for valid convolution dimensions
        if sequence_len < filter_len:
            return []

        # Preallocate results tensor
        batch_size = len(sequence_vectors)
        num_positions = sequence_len - filter_len + 1
        results = torch.zeros((num_positions, batch_size), device=filter_vector.device)

        # Compute similarities for each position
        for i in range(num_positions):
            # Extract segments for each sequence
            segments = [
                seq_vec[..., i : i + filter_len] for seq_vec in sequence_vectors
            ]

            # Calculate batch distances
            distances = filter_vector.batch_distance(segments)

            # Convert to similarities and store
            results[i] = 1.0 - distances

        # Return as list format according to original API
        if batch_size > 1:
            return [results[:, b].tolist() for b in range(batch_size)]
        else:
            return results[:, 0].tolist()

    def max_pool(
        self, feature_map: List["Bipolar"], window_size: int = 4
    ) -> List["Bipolar"]:
        """Max pooling for Bipolar feature maps using element-wise logical OR with batch processing."""
        if not feature_map:
            return []

        # Handle case with fewer vectors than window size
        if len(feature_map) < window_size:
            if not feature_map:
                return []

            # For small feature maps, logical OR all vectors
            device = feature_map[0].device
            unpacked_vectors = [fm._unpack_efficient() for fm in feature_map]
            pooled_tensor = torch.zeros_like(unpacked_vectors[0])

            # Apply logical OR across all vectors
            for vec in unpacked_vectors:
                pooled_tensor |= vec

            return [Bipolar(pooled_tensor, device=device)]

        # Process windows in batch when possible
        num_windows = len(feature_map) // window_size
        pooled_vectors = []

        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_vectors = feature_map[start_idx:end_idx]

            # Stack vectors for batch processing
            unpacked_vectors = [vec._unpack_efficient() for vec in window_vectors]
            stacked = torch.stack(unpacked_vectors)

            # Logical OR across window (dim=0)
            pooled_tensor = torch.any(stacked, dim=0)
            pooled_vectors.append(Bipolar(pooled_tensor, device=feature_map[0].device))

        return pooled_vectors

    def clone(self) -> "Bipolar":
        """Create a copy of this Bipolar vector"""
        return Bipolar(self)

    # Properties
    @property
    def tensor(self) -> torch.Tensor:
        """Get unpacked boolean tensor representation."""
        return self._unpack_efficient()

    @property
    def packed_tensor(self) -> torch.Tensor:
        """Get packed int8 tensor representation."""
        return self._data

    @property
    def shape(self) -> Tuple:
        """Get shape of the Bipolar vector."""
        return self._shape

    @property
    def device(self) -> torch.device:
        """Get device of the Bipolar vector."""
        return self._data.device

    def __len__(self) -> int:
        """Get total number of elements in the Bipolar vector."""
        return torch.Size(self._shape).numel()

    @overload
    def __getitem__(self, idx: int) -> "Bipolar": ...

    @overload
    def __getitem__(self, idx: slice) -> "Bipolar": ...

    @overload
    def __getitem__(self, idx: torch.Tensor) -> "Bipolar": ...

    def __getitem__(self, idx):
        """Index into the Bipolar vector."""
        unpacked = self._unpack_efficient()
        sliced_tensor = unpacked[idx]
        return Bipolar(sliced_tensor, device=self.device)

    def __mul__(self, other: "Bipolar") -> "Bipolar":
        """Implement * operator for binding."""
        return self.bind(other)

    def __imul__(self, other: "Bipolar") -> "Bipolar":
        """Implement *= operator for in-place binding."""
        return self.bind(other, inplace=True)

    def __matmul__(self, other: "Bipolar") -> torch.Tensor:
        """Implement @ operator for similarity computation."""
        return self.similarity(other)

    def __xor__(self, other: "Bipolar") -> "Bipolar":
        """Implement ^ operator for binding (XOR semantics)."""
        return self.bind(other)

    def __neg__(self) -> "Bipolar":
        """Implement - operator for negating all bits."""
        # Fast negation using packed representation
        # For bit-packed int8, we can just negate and add 1 to flip all bits
        return Bipolar(~self._data, shape=self._shape, device=self._data.device)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Bipolar(shape={self.shape}, device={self.device})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        sample = self._unpack_efficient().flatten()[:5].tolist()
        return f"Bipolar(shape={self.shape}, sample={sample}...)"


class AdaptiveBipolar:
    """Adaptive toolkit for bipolar hyperdimensional computing with vectorized operations."""

    # ---- Convergence Detection Methods ----
    @staticmethod
    def detect_convergence(
        loss_history: List[float],
        window_size: int = 10,
        snr_threshold: float = 2.0,
        min_iterations: int = 20,
        use_momentum: bool = True,
        momentum: float = 0.9,
        min_patience: int = 5,
    ) -> bool:
        """Detect convergence using either SNR or momentum-based approach."""
        if not loss_history:
            return False

        if use_momentum:
            return AdaptiveBipolar._check_momentum_convergence(
                loss_history, min_patience, momentum
            )
        else:
            return AdaptiveBipolar._check_snr_convergence(
                loss_history, window_size, snr_threshold, min_iterations
            )

    @staticmethod
    def _check_momentum_convergence(
        loss_history: List[float], min_patience: int, momentum: float
    ) -> bool:
        """Check convergence using momentum-based smoothing."""
        if len(loss_history) < max(10, min_patience):
            return False

        # Convert to tensor for vectorized operations
        losses = torch.tensor(loss_history, dtype=torch.float32)
        smoothed = torch.zeros_like(losses)
        smoothed[0] = losses[0]

        # Efficient computation with PyTorch operations
        for i in range(1, len(loss_history)):
            smoothed[i] = momentum * smoothed[i - 1] + (1 - momentum) * losses[i]

        # Check if change over the last min_patience steps is minimal
        if len(smoothed) >= min_patience + 1:
            if abs(smoothed[-min_patience - 1] - smoothed[-1]) < 1e-6:
                return True
        return False

    @staticmethod
    def _compute_momentum_values(
        loss_history: List[float], momentum: float
    ) -> torch.Tensor:
        """Compute momentum-smoothed values from loss history."""
        losses = torch.tensor(loss_history, dtype=torch.float32)
        smoothed = torch.zeros_like(losses)
        smoothed[0] = losses[0]

        # Vectorized computation when possible
        for i in range(1, len(loss_history)):
            smoothed[i] = momentum * smoothed[i - 1] + (1 - momentum) * losses[i]

        return smoothed

    @staticmethod
    def _check_snr_convergence(
        loss_history: List[float],
        window_size: int,
        snr_threshold: float,
        min_iterations: int,
    ) -> bool:
        """Check convergence using signal-to-noise ratio analysis."""
        if len(loss_history) < min_iterations:
            return False

        recent_losses = loss_history[-window_size:]
        if len(recent_losses) < window_size:
            return False

        # Calculate trend metrics
        slope, noise, snr = AdaptiveBipolar._calculate_trend_metrics(recent_losses)

        # Adjust threshold based on oscillation
        adjusted_threshold = AdaptiveBipolar._adjust_threshold_for_oscillation(
            loss_history, recent_losses, snr_threshold
        )

        # Converged when SNR drops below threshold
        return snr < adjusted_threshold

    @staticmethod
    def _calculate_trend_metrics(
        recent_losses: List[float],
    ) -> Tuple[float, float, float]:
        """Calculate slope, noise, and SNR from recent losses."""
        # Convert to numpy for scipy stats which doesn't work directly with PyTorch
        losses = np.array(recent_losses)
        x = np.arange(len(losses))

        # Use linear regression to get slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, losses)

        # Switch back to torch for remaining calculations
        x_tensor = torch.arange(len(recent_losses), dtype=torch.float32)
        losses_tensor = torch.tensor(recent_losses, dtype=torch.float32)
        predicted = slope * x_tensor + intercept
        residuals = losses_tensor - predicted
        noise = torch.std(residuals).item()
        signal = abs(slope * len(recent_losses))
        snr = float("inf") if noise < 1e-10 else signal / noise
        return slope, noise, snr

    @staticmethod
    def _adjust_threshold_for_oscillation(
        loss_history: List[float], recent_losses: List[float], base_threshold: float
    ) -> float:
        """Adjust threshold based on oscillation detection."""
        if len(loss_history) < len(recent_losses) + 2:
            return base_threshold

        # Vectorized calculation of diffs, velocity, and acceleration using PyTorch
        losses_tensor = torch.tensor(recent_losses, dtype=torch.float32)
        diffs = torch.diff(losses_tensor)
        velocity = torch.mean(torch.abs(diffs)).item()
        acceleration = torch.std(diffs).item()

        # Detect oscillation and adjust threshold
        oscillation_factor = acceleration / (velocity + 1e-10)
        return base_threshold * (1.0 - min(0.5, oscillation_factor))

    @staticmethod
    def analyze_loss_trend(loss_history: List[float], window_size: int = 10) -> str:
        """Analyze loss trend pattern."""
        if not loss_history:
            return "decreasing"  # Default for empty list

        if len(loss_history) < window_size:
            return "decreasing"  # Default early in training

        # Convert to tensor for vectorized operations
        recent = torch.tensor(loss_history[-window_size:], dtype=torch.float32)
        first_deriv = torch.diff(recent)

        # Check trend patterns
        if AdaptiveBipolar._is_plateau(first_deriv):
            return "plateau"
        elif AdaptiveBipolar._is_oscillating(first_deriv):
            return "oscillating"
        else:
            # Determine direction using vectorized mean
            avg_slope = torch.mean(first_deriv).item()
            return "decreasing" if avg_slope < 0 else "increasing"

    @staticmethod
    def _is_plateau(derivatives: torch.Tensor, threshold: float = 1e-4) -> bool:
        """Check if derivatives indicate a plateau."""
        return torch.mean(torch.abs(derivatives)).item() < threshold

    @staticmethod
    def _is_oscillating(derivatives: torch.Tensor) -> bool:
        """Check if derivatives indicate oscillation."""
        # Calculate sign changes
        sign_changes = torch.sum((derivatives[1:] * derivatives[:-1]) < 0).item()
        return sign_changes > len(derivatives) // 3

    # ---- Dimension Reduction Methods ----
    @staticmethod
    def compute_information_density(
        vectors: List[Any], max_segments: int = 8
    ) -> torch.Tensor:
        """Compute information density across vector segments."""
        if not vectors:
            return torch.tensor([])

        # Prepare segments
        dim = vectors[0].shape[0]
        segment_size, num_segments = AdaptiveBipolar._calculate_segment_params(
            dim, max_segments
        )

        # Initialize scores
        device = vectors[0].tensor.device
        density_scores = torch.zeros(num_segments, device=device)

        # Prepare tensors for vectorized operations
        # Stack all vectors into a single batch for efficient processing
        stacked_tensors = torch.stack([v.tensor for v in vectors])

        # Calculate density for each segment in a vectorized manner
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min(start_idx + segment_size, dim)
            segment_data = stacked_tensors[:, start_idx:end_idx]

            # Calculate variance of the segment as information density
            segment_float = segment_data.float()
            density_scores[i] = torch.var(segment_float)

        # Normalize scores using vectorized operations
        sum_density = torch.sum(density_scores)
        if sum_density > 0:
            return density_scores / sum_density
        return density_scores

    @staticmethod
    def _calculate_segment_params(dim: int, max_segments: int) -> Tuple[int, int]:
        """Calculate segment size and number of segments."""
        segment_size = max(1, dim // min(max_segments, dim))
        num_segments = dim // segment_size
        return segment_size, num_segments

    @staticmethod
    def xor_entropy_segmentation(
        vectors: List[Any], threshold_factor: float = 1.5
    ) -> List[int]:
        """Find natural segment boundaries using XOR entropy patterns."""
        if len(vectors) < 2:
            return []

        # Get dimensions and device
        dim = vectors[0].shape[0]
        device = vectors[0].tensor.device

        # Preallocate entropy tensor
        dimension_entropy = torch.zeros(dim, device=device)

        # Calculate XOR patterns and dimension entropy efficiently
        for i in range(len(vectors) - 1):
            # Get raw tensors for direct XOR calculation
            xor_pattern = vectors[i].tensor != vectors[i + 1].tensor
            dimension_entropy += xor_pattern.float()

        # Normalize by number of patterns
        dimension_entropy /= len(vectors) - 1

        # Calculate entropy gradients and find significant changes
        entropy_gradients = torch.abs(dimension_entropy[1:] - dimension_entropy[:-1])
        mean_gradient = torch.mean(entropy_gradients).item()
        std_gradient = torch.std(entropy_gradients).item()
        threshold = mean_gradient + threshold_factor * std_gradient

        # Find indices where gradient exceeds threshold
        boundary_indices = torch.where(entropy_gradients > threshold)[0].cpu().tolist()
        return boundary_indices

    @staticmethod
    def calculate_adaptive_window_sizes(
        segment_info_density: torch.Tensor,
        original_dim: int,
        target_compression_ratio: float = 0.5,
        min_dim: int = 256,
    ) -> List[int]:
        """Calculate adaptive window sizes based on information density."""
        # Ensure segment_info_density is a torch tensor
        if not isinstance(segment_info_density, torch.Tensor):
            segment_info_density = torch.tensor(
                segment_info_density, dtype=torch.float32
            )

        segment_size = original_dim // len(segment_info_density)

        # Calculate target dimension and compression
        target_dim = max(min_dim, int(original_dim * target_compression_ratio))
        total_compression = original_dim / target_dim

        # Vectorized computation of inverse densities
        inverse_densities = 1.0 / (segment_info_density + 0.01)
        inverse_density_sum = torch.sum(inverse_densities).item()

        # Calculate window sizes using vectorized operations
        raw_window_sizes = torch.clamp(
            torch.round(inverse_densities * total_compression / inverse_density_sum),
            min=1,
            max=segment_size,
        ).int()

        # Convert to list for processing
        raw_window_sizes_list = raw_window_sizes.cpu().tolist()

        # Expand window sizes to cover segments
        window_sizes = []
        for window_size in raw_window_sizes_list:
            # Calculate number of windows needed for this segment
            num_windows = segment_size // window_size
            window_sizes.extend([window_size] * num_windows)

        return window_sizes

    @staticmethod
    def track_binding_stability(
        vectors: List[Any],
        iterations: int = 100,
        top_k: Optional[Union[int, float]] = None,
    ) -> torch.Tensor:
        """Track which dimensions remain stable under binding operations."""
        if not vectors or len(vectors) < 2:
            return None

        # Set up tracking
        dim = vectors[0].shape[0]
        device = vectors[0].tensor.device
        stability_scores = torch.zeros(dim, device=device)

        # Generate all random pairs for binding simulation at once
        num_vectors = len(vectors)

        # Set fixed seed for reproducibility
        torch.manual_seed(42)

        for _ in range(iterations):
            # Generate a single pair of indices
            i, j = random.sample(range(num_vectors), 2)

            # Bind them and recover original
            bound_vector = vectors[i].bind(vectors[j])
            recovered = bound_vector.unbind(vectors[j])

            # Update stability scores
            stability_mask = recovered.tensor == vectors[i].tensor
            stability_scores += stability_mask.float()

        # Normalize by iterations
        stability_scores /= iterations

        # Return top-k or all stability scores
        if top_k is not None:
            # Calculate k value (either direct or as fraction)
            k = int(dim * top_k) if isinstance(top_k, float) else int(top_k)
            k = max(1, min(k, dim))  # Ensure valid k value
            return torch.topk(stability_scores, k).indices

        return stability_scores

    @staticmethod
    def detect_shift_invariant_dimensions(
        vectors: List[Any], max_shift: int = 5
    ) -> torch.Tensor:
        """Find dimensions that preserve information under circular shifts."""
        if not vectors:
            return None

        dim = vectors[0].shape[0]
        device = vectors[0].tensor.device
        shift_variance = torch.zeros(dim, device=device)

        # Vectorized computation when possible
        for vec in vectors:
            for shift in range(1, max_shift + 1):
                # Perform circular shift
                shifted = vec.circular_shift(shift)

                # Record which dimensions changed
                changed_dims = shifted.tensor != vec.tensor
                shift_variance += changed_dims.float()

        # Normalize by vectors and shifts
        return shift_variance / (len(vectors) * max_shift)

    @staticmethod
    def find_optimal_compression_ratio(
        vectors: List[Any],
        test_pairs: Optional[List[Tuple[int, int]]] = None,
        ratios: Optional[List[float]] = None,
    ) -> float:
        """Find optimal compression ratio via superposition testing."""
        if not vectors or len(vectors) < 2:
            return 0.5  # Default fallback

        # Default ratios and test pairs
        if ratios is None:
            ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        if test_pairs is None:
            # Generate random test pairs efficiently
            n_vectors = len(vectors)
            max_pairs = min(50, n_vectors * (n_vectors - 1) // 2)

            # Generate unique random pairs
            random.seed(42)  # For reproducibility

            indices = list(range(n_vectors))
            test_pairs = []
            for _ in range(max_pairs):
                i, j = random.sample(indices, 2)
                test_pairs.append((i, j))

        # Prepare for vectorized computation
        test_accuracies = []

        # Test different compression ratios
        for ratio in ratios:
            # Calculate window sizes for this ratio
            info_density = AdaptiveBipolar.compute_information_density(vectors)
            window_sizes = AdaptiveBipolar.calculate_adaptive_window_sizes(
                info_density, vectors[0].shape[0], ratio
            )

            # Test compression with these window sizes
            similarities = []
            for i, j in test_pairs:
                # Original superposition
                orig_super = vectors[i].superposition(vectors[j])

                # Compressed vectors
                compressed_i = AdaptiveBipolar.reduce_dimension_with_max_pool(
                    vectors[i], window_sizes
                )
                compressed_j = AdaptiveBipolar.reduce_dimension_with_max_pool(
                    vectors[j], window_sizes
                )

                # Compressed superposition
                comp_super = compressed_i.superposition(compressed_j)

                # Measure similarity
                similarities.append(1.0 - orig_super.distance(comp_super).item())

            # Calculate mean accuracy using torch
            test_accuracies.append(torch.tensor(similarities).mean().item())

        # Convert to tensor for efficient calculation
        accuracy_tensor = torch.tensor(test_accuracies)
        ratios_tensor = torch.tensor(ratios)

        # Calculate gradients and second derivatives
        gradients = torch.diff(accuracy_tensor) / torch.diff(ratios_tensor)
        second_derivs = torch.diff(gradients)

        # Find index of minimum absolute second derivative
        elbow_idx = torch.argmin(torch.abs(second_derivs)).item() + 1

        return ratios[elbow_idx]

    @staticmethod
    def reduce_dimension_with_max_pool(vector: Any, window_sizes: List[int]) -> Any:
        """Reduce dimension of a Bipolar vector using adaptive max pooling."""
        # Handle empty case
        if not window_sizes:
            return vector

        # Split vector into segments based on window sizes
        segments = []
        start_idx = 0

        for window_size in window_sizes:
            end_idx = start_idx + window_size
            if end_idx <= vector.shape[0]:
                segments.append(vector[start_idx:end_idx])
                start_idx = end_idx
            else:
                break

        if not segments:
            return vector

        # Apply max_pool to segments
        pooled_vectors = vector.max_pool(segments)

        # Handle case where no pooled vectors were created
        if not pooled_vectors:
            return type(vector).random((len(segments),), device=vector.device)

        return pooled_vectors[0]

    @staticmethod
    def multi_bind_resilience(
        vectors: List[Any], binding_depth: int = 5, n_tests: int = 20
    ) -> torch.Tensor:
        """Test dimension resilience under multiple binding operations."""
        if not vectors or len(vectors) < binding_depth + 1:
            return None

        dim = vectors[0].shape[0]
        device = vectors[0].tensor.device
        resilience_scores = torch.zeros(dim, device=device)

        # Set fixed seed for reproducibility
        random.seed(42)

        # Run tests in vectorized manner where possible
        for _ in range(n_tests):
            # Select random start vector
            start_idx = random.randrange(len(vectors))
            vec = vectors[start_idx]

            # Create binding chain
            bound = vec
            bind_chain = [vec]

            for _ in range(binding_depth):
                # Bind with random vector
                random_idx = random.randrange(len(vectors))
                bound = bound.bind(vectors[random_idx])
                bind_chain.append(vectors[random_idx])

            # Recover through chain
            recovered = bound
            for i in range(binding_depth, 0, -1):
                recovered = recovered.unbind(bind_chain[i])

            # Check dimensions
            accurate_dims = recovered.tensor == vec.tensor
            resilience_scores += accurate_dims.float()

        # Normalize
        resilience_scores /= n_tests
        return resilience_scores

    # ---- Hyperparameter Adaptation Methods ----
    @staticmethod
    def adapt_learning_rate(
        iteration: int, loss_trend: str, grad_variance: float, base_lr: float = 0.1
    ) -> float:
        """Adapt learning rate based on loss trend and gradient variance."""
        # Define multipliers for each trend type in a dict for efficient lookup
        trend_multipliers = {
            "plateau": 2.0,  # Boost on plateaus
            "oscillating": 0.1,  # Reduce on oscillation
            "increasing": 0.5,  # Reduce if loss is increasing
            "decreasing": 1.0,  # Keep same if decreasing
        }

        # Calculate adaptations using simple multiplications
        variance_factor = 1.0 / (1.0 + grad_variance)
        trend_multiplier = trend_multipliers.get(loss_trend, 1.0)
        decay_factor = 0.9 ** (iteration // 10)

        # Apply adaptations
        lr = base_lr * trend_multiplier * variance_factor * decay_factor

        # Clamp to reasonable range
        return max(0.001, min(0.5, lr))

    @staticmethod
    def adapt_batch_size(
        iteration: int,
        available_memory_gb: float,
        min_size: int = 16,
        max_size: int = 512,
    ) -> int:
        """Dynamically adapt batch size based on iteration and available memory."""
        # Calculate memory-based maximum
        memory_fraction = 0.1  # Use 10% of available memory
        memory_max = min(max_size, int(available_memory_gb * 1024 * memory_fraction))

        # Calculate schedule-based size (start small and increase gradually)
        schedule_size = min_size + int(iteration * 2)

        # Use the smaller of the two constraints
        return min(memory_max, schedule_size)

    @staticmethod
    def adaptive_convergence_threshold(vocab_size: int, dimension: int) -> float:
        """Calculate adaptive convergence threshold based on model size."""
        # Calculate parameter count
        param_count = vocab_size * dimension

        # Base threshold (smaller models can use larger thresholds)
        base_threshold = 1e-4

        # Scale logarithmically with parameter count
        return base_threshold / (1.0 + 0.1 * math.log10(param_count + 1))

    @staticmethod
    def entropy_guided_binding(
        vectors: List[Any], entropy_threshold: float = 0.7
    ) -> Any:
        """Create a function that selectively binds dimensions based on information content."""
        # Vectorized computation of information content
        info_density = AdaptiveBipolar.compute_information_density(vectors)
        high_info_dims = info_density > entropy_threshold

        # Create a selective binding function
        def selective_bind(vec_a, vec_b):
            # Clone the tensor to avoid modifying the original
            result = vec_a.clone()

            # Only XOR the dimensions with high information content
            tensor_a = vec_a.tensor
            tensor_b = vec_b.tensor

            # Use vectorized operations with mask
            result.tensor[high_info_dims] = (
                tensor_a[high_info_dims] ^ tensor_b[high_info_dims]
            )

            return result

        return selective_bind


class STE(torch.autograd.Function):
    """Generic Straight-Through Estimator with customizable forward and backward."""

    @staticmethod
    def forward(ctx, input: Tensor, forward_fn, backward_fn):
        ctx.forward_fn = forward_fn
        ctx.backward_fn = backward_fn
        ctx.save_for_backward(input)
        return forward_fn(input)

    @staticmethod
    def backward(
        ctx, grad_output: Tensor, input: Tensor
    ):  # Standardized backward signature
        return (
            ctx.backward_fn(grad_output, input),
            None,
            None,
        )  # Pass input to backward_fn

    @staticmethod
    def ste_bool(x: Tensor) -> Tensor:
        """Straight-Through Estimator for boolean conversion."""

        def forward_fn(input):
            return (input.float() * 2) - 1  # bool -> bipolar

        def backward_fn(grad_output, input):
            return grad_output.clamp_(-1, 1)  # Standardized backward_fn

        return STE.apply(x, forward_fn, backward_fn)

    @staticmethod
    def ste_sign(x):
        """Straight-Through Estimator for sign function with gradient clipping."""
        forward_fn = torch.sign

        def backward_fn(grad_output, input):
            return grad_output.clamp(-1, 1)

        return STE.apply(x, forward_fn, backward_fn)


class BipolarLinear(nn.Module):
    """Linear layer with bit-packed bipolar weights."""

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.nn.Parameter
    bias: Optional[torch.nn.Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            Bipolar.random(
                (out_features, in_features)
            ).packed_tensor  # Use Bipolar.random
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data = Bipolar.random(  # Use Bipolar.random
            (self.out_features, self.in_features)
        ).packed_tensor
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight.unsqueeze(-1)
            )  # fake last dimension
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        unpacked_weight = Bipolar(  # Use Bipolar class
            self.weight, shape=(self.out_features, self.in_features)
        ).tensor

        if isinstance(input, Bipolar):  # Use Bipolar class
            unpacked_input = input.tensor
        else:
            raise TypeError("Input must be Bipolar")

        # Use STEBool for the forward and backward pass
        bipolar_weight = STE.ste_bool(
            unpacked_weight
        )  # Boolean -> Bipolar (-1, +1) , using generic STE
        bipolar_input = STE.ste_bool(unpacked_input)
        output_float = F.linear(bipolar_input, bipolar_weight, self.bias)
        output_bipolar = Bipolar(  # Use Bipolar class
            (output_float > 0), shape=output_float.shape
        )  # Float -> Boolean, and pack
        return output_bipolar

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


@dataclass
class SignSGDConfig:
    """Configuration for the SignSGD optimizer."""

    initial_flip_prob: float = 0.1  # Initial global probability of flipping a bit
    use_loss_trend: bool = False  # Enables loss trend to adjust flip probability
    adaptive_window_size: int = 10  # Window size for loss trend analysis
    cooling_threshold: float = 0.8  # Threshold for cooling the flip probability
    warming_threshold: float = 1.2  # Threshold for warming the flip probability
    min_flip_prob: float = 0.0001  # Minimum allowed flip probability
    max_flip_prob: float = 0.5  # Maximum allowed flip probability
    stability_floor: float = 0.01  # Minimum stability value to avoid division by zero
    batch_params: bool = True  # Whether to batch parameter updates for vectorization

    def __repr__(self) -> str:
        return (
            f"SignSGDConfig(initial_flip_prob={self.initial_flip_prob}, "
            f"use_loss_trend={self.use_loss_trend}, "
            f"adaptive_window_size={self.adaptive_window_size}, "
            f"cooling_threshold={self.cooling_threshold}, "
            f"warming_threshold={self.warming_threshold}, "
            f"min_flip_prob={self.min_flip_prob}, "
            f"max_flip_prob={self.max_flip_prob}, "
            f"stability_floor={self.stability_floor}, "
            f"batch_params={self.batch_params})"
        )


class SignSGD:
    """Simplified Hybrid SignSGD: Disagreement Rate and System Stability Focused."""

    def __init__(self, params: List[torch.nn.Parameter], config: SignSGDConfig) -> None:
        """Initializes the simplified hybrid SignSGD optimizer."""
        self.params: List[torch.nn.Parameter] = list(params)
        self.config: SignSGDConfig = config
        self.flip_prob: float = config.initial_flip_prob
        self.param_groups: List[Dict[str, List[torch.nn.Parameter]]] = [
            {"params": self.params}
        ]
        self.iterations: int = 0
        self.prev_flips: Optional[float] = None
        self.loss_history: List[float] = []  # Track loss history

        # Group parameters by dtype for batch processing
        self.int8_params = []
        self.float_params = []

        for p in self.params:
            history_cap: int = max(5, min(50, int(math.log(p.numel()) * 3)))
            p.register_buffer("history", torch.zeros(history_cap, device=p.device))
            p.history_idx: int = 0
            p.history_cap: int = history_cap
            p.stability_floor: float = config.stability_floor * math.log(p.numel())

            # Group parameters by dtype for vectorized operations
            if p.dtype == torch.int8:
                self.int8_params.append(p)
            else:  # float parameters
                self.float_params.append(p)

    def __repr__(self) -> str:
        return (
            f"SignSGD(flip_prob={self.flip_prob}, "
            f"iterations={self.iterations}, "
            f"config={repr(self.config)})"
        )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dictionary."""
        return {
            "flip_prob": self.flip_prob,
            "iterations": self.iterations,
            "loss_history": self.loss_history,
            "prev_flips": self.prev_flips,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the optimizer from a dictionary."""
        self.flip_prob = state_dict["flip_prob"]
        self.iterations = state_dict["iterations"]
        self.loss_history = state_dict["loss_history"]
        self.prev_flips = state_dict["prev_flips"]

    def _process_param(self, p: torch.nn.Parameter) -> Tuple[Tensor, Tensor]:
        """Helper function to process parameter based on dtype."""
        if p.dtype == torch.int8:
            shape: Tuple[int, ...] = (
                (p.shape[0], -1) if len(p.shape) == 2 else p.shape[:-1]
            )
            unpacked_p: Tensor = Bipolar(p, shape=shape).tensor
            unpacked_grad: Tensor = p.grad > 0
        else:  # float parameters
            unpacked_p: Tensor = torch.sign(p.data)
            unpacked_grad: Tensor = torch.sign(p.grad.data)
        return unpacked_p, unpacked_grad

    def _update_history_and_stability(
        self, p: torch.nn.Parameter, disagree_rate: float
    ) -> float:
        """Helper function to update history and calculate stability."""
        p.history[p.history_idx % p.history_cap] = disagree_rate
        p.history_idx += 1
        history_window: Tensor = p.history[: min(p.history_idx, p.history_cap)]
        return max(p.stability_floor, history_window.std().item())

    def step(self, current_loss: float) -> None:
        """Perform a single optimization step."""
        with torch.no_grad():
            self.loss_history.append(current_loss)  # Track loss history

            # step 1: get layer metrics and system stability
            layer_metrics: List[Dict[str, Any]] = self._calculate_layer_metrics()
            system_stability: float = self._calculate_system_stability(layer_metrics)
            self._adjust_global_flip_prob(system_stability, layer_metrics)

            # step 2: adjust flip probability based on flip rate
            total_flips, total_bits = self._apply_parameter_updates(layer_metrics)
            flip_rate: float = total_flips / max(total_bits, 1)
            self._adjust_flip_prob_with_rate(flip_rate, system_stability)

            # step 3: log progress and update state
            self._log_progress(flip_rate, system_stability)
            self.prev_flips: float = flip_rate
            self.iterations += 1

    def _adjust_global_flip_prob(
        self, system_stability: float, layer_metrics: List[Dict[str, Any]]
    ) -> None:
        """Simplified global flip probability adjustment."""
        if system_stability > 0.9:
            return  # Skip adjustment if system is very stable

        # Vectorized calculation of average disagree rate
        disagree_rates = torch.tensor([m["disagree_rate"] for m in layer_metrics])
        avg_disagree_rate = disagree_rates.mean().item()

        global_flip_factor_base: float = 2.0 - system_stability
        disagree_factor: float = 1.0 + 0.5 * avg_disagree_rate

        # Optional loss trend factor
        if (
            self.config.use_loss_trend
            and len(self.loss_history) >= self.config.adaptive_window_size
        ):
            recent_losses: List[float] = self.loss_history[
                -self.config.adaptive_window_size :
            ]
            slope, noise, snr = AdaptiveBipolar._calculate_trend_metrics(recent_losses)
            trend_factor: float = max(0.1, (1.0 - slope * 2.0) * (0.5 + 0.5 * snr))
            global_flip_factor_base *= trend_factor

        global_flip_factor: float = global_flip_factor_base * disagree_factor
        self.flip_prob *= global_flip_factor
        self.flip_prob = max(
            self.config.min_flip_prob, min(self.config.max_flip_prob, self.flip_prob)
        )  # Clamp

    def _adjust_flip_prob_with_rate(
        self, flip_rate: float, system_stability: float
    ) -> None:
        """Adjust global flip probability based on flip rate."""
        if self.prev_flips is not None:
            cooling_factor: float = 1.0 - (system_stability / 4)
            warming_factor: float = 1.0 + system_stability
            if flip_rate < self.prev_flips * self.config.cooling_threshold:
                self.flip_prob *= warming_factor
            elif flip_rate > self.prev_flips * self.config.warming_threshold:
                self.flip_prob *= cooling_factor
            self.flip_prob = max(
                self.config.min_flip_prob,
                min(self.config.max_flip_prob, self.flip_prob),
            )  # Clamp

    def _calculate_local_flip_prob(
        self, local_stability: float, disagree_rate: float
    ) -> float:
        """Calculates local flip probability."""
        return self.flip_prob * (1.0 - local_stability) * (1.0 + 0.5 * disagree_rate)

    def _log_progress(self, flip_rate: float, system_stability: float) -> None:
        """Logs training progress."""
        if self.iterations % 10 == 0:
            print(
                f"Iter {self.iterations}: Flip rate: {flip_rate:.4f}, "
                f"Prob: {self.flip_prob:.4f}, Stability: {system_stability:.4f}"
            )

    def _calculate_layer_metrics(self) -> List[Dict[str, Any]]:
        """Calculates metrics for each layer (parameter group) with vectorization."""
        layer_metrics: List[Dict[str, Any]] = []

        for p in self.params:
            if p.grad is None:
                continue

            # Process parameter
            unpacked_p, unpacked_grad = self._process_param(p)

            # Vectorized computation of disagree mask and rate
            disagree_mask: Tensor = unpacked_grad != unpacked_p
            disagree_rate: float = disagree_mask.float().mean().item()

            # Update history and calculate stability
            stability: float = self._update_history_and_stability(p, disagree_rate)
            disagree_multiplier: float = 1.0 + disagree_rate

            layer_metrics.append(
                {
                    "param": p,
                    "disagree_rate": disagree_rate,
                    "stability": stability,
                    "disagree_multiplier": disagree_multiplier,
                    "unpacked_p": unpacked_p,
                    "unpacked_grad": unpacked_grad,
                    "disagree_mask": disagree_mask,  # Store for reuse in parameter updates
                }
            )
        return layer_metrics

    def _calculate_system_stability(self, layer_metrics: List[Dict[str, Any]]) -> float:
        """Calculates the overall system stability with vectorization."""
        if not layer_metrics:
            return 0.1  # Default if no layers with gradients
        stability_tensor = torch.tensor([m["stability"] for m in layer_metrics])
        return stability_tensor.mean().item()

    def _apply_parameter_updates(
        self, layer_metrics: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """Applies parameter updates (bit flips)"""
        total_flips = 0
        total_bits = 0
        for metric in layer_metrics:
            p = metric["param"]
            local_stability = metric["stability"]
            disagree_rate = metric["disagree_rate"]
            unpacked_p = metric["unpacked_p"]
            unpacked_grad = metric["unpacked_grad"]
            local_flip_prob = self._calculate_local_flip_prob(
                local_stability, disagree_rate
            )
            local_flip_prob = min(max(local_flip_prob, 1.0 / p.numel()), 0.5)

            if p.dtype == torch.int8:
                flip_mask = (
                    torch.rand_like(unpacked_p, dtype=torch.float32) < local_flip_prob
                )
                disagree_mask = unpacked_grad != unpacked_p
                updated_unpacked_p = unpacked_p.clone()
                updated_unpacked_p[flip_mask & disagree_mask] = ~updated_unpacked_p[
                    flip_mask & disagree_mask
                ]
                p.data = Bipolar(
                    updated_unpacked_p, shape=updated_unpacked_p.shape
                ).packed_tensor
                bit_flips = torch.sum(flip_mask & disagree_mask).item()
                total_flips += bit_flips
                total_bits += p.numel()
            else:  # float parameters
                flip_mask = torch.rand_like(p.data) < local_flip_prob
                disagree_mask = torch.sign(p.grad) != torch.sign(p.data)
                p.data[flip_mask & disagree_mask] *= -1.0
                bit_flips = torch.sum(flip_mask & disagree_mask).item()
                total_flips += bit_flips
                total_bits += p.numel()

        return total_flips, total_bits

    def _calculate_total_bits(self, layer_metrics: List[Dict[str, Any]]) -> int:
        """Calculates the total number of bits across all parameters."""
        return sum(
            Bipolar(
                m["param"],
                shape=(m["param"].shape[0], -1)
                if len(m["param"].shape) == 2
                else m["param"].shape[:-1],
            ).__len__()
            if m["param"].dtype == torch.int8
            else m["param"].numel()
            for m in layer_metrics
        )

    def zero_grad(self) -> None:
        """Clears gradients for all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def reset_flip_prob(self) -> None:
        """Resets the global flip probability to the initial value."""
        self.flip_prob: float = self.config.initial_flip_prob

    def set_flip_prob(self, flip_prob: float) -> None:
        """Sets the global flip probability to a specific value."""
        self.flip_prob = max(
            self.config.min_flip_prob, min(self.config.max_flip_prob, self.flip_prob)
        )

    def get_flip_prob(self) -> float:
        """Returns the current global flip probability."""
        return self.flip_prob

    def reset_history(self) -> None:
        """Resets the history for all parameters."""
        for p in self.params:
            p.history.zero_()
            p.history_idx: int = 0


@dataclass
class TextProcessorConfig:
    """Streamlined config for text processing."""

    # Core settings
    tokenizer: str = "whitespace"
    clean_ops: List[str] = field(
        default_factory=lambda: ["lowercase", "remove_punctuation"]
    )
    doc_sep: str = r"\n+"
    min_freq: int = 3
    max_words: int = 5000

    # BM25 params
    k1: float = 1.5
    b: float = 0.75

    # Adaptive settings
    adaptive: bool = True
    info_density: bool = True
    entropy_seg: bool = False
    window_range: Tuple[int, int] = (2, 4)

    # Cache settings
    cache_size: int = 100  # Size for LRU cache

    @property
    def min_window(self) -> int:
        return self.window_range[0]

    @property
    def max_window(self) -> int:
        return self.window_range[1]


class TextProcessor:
    """Compact text processor with semantic relationship extraction."""

    def __init__(self, config: TextProcessorConfig = None):
        self.cfg = config or TextProcessorConfig()
        self._doc_term_cache = {}

        # Create the cache with the configured size
        self._create_doc_term_matrix = lru_cache(maxsize=self.cfg.cache_size)(
            self._create_doc_term_matrix_impl
        )

    def process_text(self, text: str) -> Tuple[List[FrozenSet[str]], Set[str], Dict]:
        """One-shot text processing pipeline."""
        clean = self._clean(text)
        docs = self._tokenize(clean)
        vocab = self._extract_vocab(docs)
        w2w, w2p, p2p = self.build_relationships(docs, vocab)
        return docs, vocab, {"w2w": w2w, "w2p": w2p, "p2p": p2p}

    def _clean(self, text: str) -> str:
        """Clean text based on configured operations."""
        txt = text
        ops = self.cfg.clean_ops

        if "lowercase" in ops:
            txt = txt.lower()
        if "remove_punctuation" in ops:
            txt = re.sub(r"[^\w\s]", " ", txt)
        if "remove_numbers" in ops:
            txt = re.sub(r"\d+", "", txt)

        return re.sub(r"\s+", " ", txt).strip()

    def _tokenize(self, text: str) -> List[FrozenSet[str]]:
        """Split text into document token sets."""
        docs = re.split(self.cfg.doc_sep, text)
        return [frozenset(doc.split()) for doc in docs if doc.strip()]

    def _extract_vocab(self, docs: List[FrozenSet[str]]) -> Set[str]:
        """Extract vocabulary using frequency or information density."""
        # Get frequency-filtered candidates
        word_counts = Counter(word for doc in docs for word in doc)
        candidates = {
            word
            for word, count in word_counts.most_common(self.cfg.max_words * 2)
            if count >= self.cfg.min_freq
        }

        # If we don't need info-density filtering or have few enough words, return early
        if not self.cfg.info_density or len(candidates) <= self.cfg.max_words:
            return set(list(candidates)[: self.cfg.max_words])

        # Create word presence matrix directly using boolean operations
        candidate_list = list(candidates)
        n_docs = len(docs)

        # Preallocate matrix once (optimization)
        word_doc_matrix = torch.zeros((len(candidate_list), n_docs), dtype=torch.bool)

        # Use boolean indexing for faster matrix creation
        for j, doc in enumerate(docs):
            doc_set = set(doc)  # Convert once for faster lookups
            for i, word in enumerate(candidate_list):
                word_doc_matrix[i, j] = word in doc_set

        # Create Bipolar vectors and calculate density - reuse tensor data
        bipolar_vecs = [Bipolar(word_doc_matrix[i]) for i in range(len(candidate_list))]

        # Calculate density and sort by information value
        info_density = self._compute_info_density(bipolar_vecs)

        # Convert to tensor for efficient sorting and indexing
        word_density = torch.tensor(info_density, dtype=torch.float32)

        # Get indices of top words by density
        top_indices = torch.argsort(-word_density)[: self.cfg.max_words].tolist()

        # Return top words
        return {candidate_list[i] for i in top_indices}

    def _compute_info_density(self, vectors: List[Bipolar]) -> np.ndarray:
        """Calculate information density for vectors."""
        if not vectors:
            return np.array([])

        # Extract shapes once
        dim = vectors[0].shape[0]

        # Stack tensors for vectorized operations
        stacked_tensors = torch.stack([v.tensor for v in vectors]).cpu()

        # Calculate segments efficiently
        seg_size = max(1, dim // 8)
        segments = dim // seg_size

        # Preallocate density array
        density = np.zeros(segments)

        # Calculate variance in segments
        for i in range(segments):
            start = i * seg_size
            end = min((i + 1) * seg_size, dim)
            segment = stacked_tensors[:, start:end].float().numpy()
            # Use numpy variance for the CPU operation
            density[i] = np.var(segment)

        # Safe normalization
        sum_density = np.sum(density)
        if sum_density > 0:
            return density / sum_density
        return density

    def build_relationships(
        self, docs: List[FrozenSet[str]], vocab: Set[str]
    ) -> Tuple[Dict, Dict, Dict]:
        """Extract word and phrase relationships from documents."""
        if not docs or not vocab:
            logger.warning("No documents or vocabulary provided.")
            return {}, {}, {}

        # Filter docs by vocabulary
        filtered = [doc & vocab for doc in docs if doc]
        if not filtered:
            logger.warning("No documents after vocabulary filtering.")
            return {}, {}, {}

        # Calculate BM25 scores and prepare vocabulary mapping
        bm25, words, word_idx = self._compute_bm25(filtered, vocab)

        # Prepare hashable version of parameters for cache
        docs_tuple = tuple(filtered)
        word_idx_tuple = tuple(word_idx.items())

        # Cache document term matrix for reuse
        doc_term_matrix = self._create_doc_term_matrix(docs_tuple, word_idx_tuple)
        self._doc_term_cache["matrix"] = doc_term_matrix
        self._doc_term_cache["word_idx"] = word_idx

        # Get window sizes (adaptive or fixed)
        windows = (
            self._get_windows(filtered, word_idx, doc_term_matrix)
            if self.cfg.adaptive
            else {"w2w": 2, "w2p": 3, "p2p": 4}
        )

        # Extract each relationship type
        w2w = self._extract_relationships(
            filtered, bm25, word_idx, words, "w2w", windows, doc_term_matrix
        )
        w2p = self._extract_relationships(
            filtered, bm25, word_idx, words, "w2p", windows, doc_term_matrix
        )
        p2p = self._extract_relationships(
            filtered, bm25, word_idx, words, "p2p", windows, doc_term_matrix
        )

        # Clear cache after use
        self.clear_cache()

        return w2w, w2p, p2p

    def _create_doc_term_matrix(
        self,
        docs_tuple: Tuple[FrozenSet[str], ...],
        word_idx_tuple: Tuple[Tuple[str, int], ...],
    ) -> torch.Tensor:
        """Cache wrapper for document-term matrix creation."""
        return self._create_doc_term_matrix_impl(docs_tuple, word_idx_tuple)

    def _create_doc_term_matrix_impl(
        self,
        docs_tuple: Tuple[FrozenSet[str], ...],
        word_idx_tuple: Tuple[Tuple[str, int], ...],
    ) -> torch.Tensor:
        """Create document-term matrix with caching for reuse."""
        # Convert back to original types
        docs = list(docs_tuple)
        word_idx = dict(word_idx_tuple)

        # Early return for empty case
        if not docs or not word_idx:
            return torch.zeros((0, 0), dtype=torch.bool)

        max_idx = max(word_idx.values()) + 1
        n_docs = len(docs)

        # Create matrix once
        doc_term_matrix = torch.zeros((n_docs, max_idx), dtype=torch.bool)

        # Vectorized fill using lists of indices
        for i, doc in enumerate(docs):
            indices = [word_idx[w] for w in doc if w in word_idx]
            if indices:
                doc_term_matrix[i, indices] = True

        return doc_term_matrix

    def clear_cache(self):
        """Clear the document-term matrix cache."""
        self._doc_term_cache.clear()
        if hasattr(self._create_doc_term_matrix, "cache_clear"):
            self._create_doc_term_matrix.cache_clear()

    def _compute_bm25(
        self, docs: List[FrozenSet[str]], vocab: Set[str]
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """Compute BM25 scores efficiently using sparse PyTorch operations."""
        words = list(vocab)
        word_idx = {word: i for i, word in enumerate(words)}
        N = len(docs)

        # Get document lengths as tensor
        doc_lens = torch.tensor([len(d) for d in docs], dtype=torch.float32)
        avg_len = doc_lens.mean().item() or 1.0  # Avoid div by zero

        # Collect indices and values for sparse tensor in one pass
        indices, values = [], []
        for doc_i, doc in enumerate(docs):
            # Use counter for efficient term frequency calculation
            counts = Counter(w for w in doc if w in word_idx)
            for word, tf in counts.items():
                indices.append([doc_i, word_idx[word]])
                values.append(float(tf))

        # Handle empty case - use the compact tensor creation approach
        if not indices:
            return torch.zeros((N, len(words))), words, word_idx

        # Create sparse tensor directly
        indices_tensor = torch.tensor(indices).t()
        values_tensor = torch.tensor(values, dtype=torch.float32)
        tf_sparse = torch.sparse_coo_tensor(
            indices_tensor, values_tensor, (N, len(words))
        )

        # Convert to dense for operations that require it
        tf = tf_sparse.to_dense()

        # Vectorized IDF calculation
        df = (tf > 0).sum(dim=0)
        idf = torch.log1p((N - df + 0.5) / (df + 0.5))

        # Vectorized BM25 calculation
        k1, b = self.cfg.k1, self.cfg.b
        len_norm = 1 - b + b * (doc_lens.unsqueeze(1) / avg_len)
        denom = tf + k1 * len_norm
        bm25 = tf * (k1 + 1) * idf / denom

        return bm25, words, word_idx

    def _get_windows(
        self,
        docs: List[FrozenSet[str]],
        word_idx: Dict[str, int],
        doc_term_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        """Calculate adaptive window sizes based on information density."""
        if not docs:
            return {"w2w": 2, "w2p": 3, "p2p": 4}

        # Reuse doc_term_matrix if provided
        if doc_term_matrix is None:
            docs_tuple = tuple(docs)
            word_idx_tuple = tuple(word_idx.items())
            doc_term_matrix = self._create_doc_term_matrix(docs_tuple, word_idx_tuple)

        # Skip if no valid documents
        if not doc_term_matrix.any():
            return {"w2w": 2, "w2p": 3, "p2p": 4}

        # Efficient creation of Bipolar vectors
        vecs = []
        for i in range(doc_term_matrix.shape[0]):
            row = doc_term_matrix[i]
            if row.any():
                vecs.append(Bipolar(row, shape=row.shape))

        if not vecs:
            return {"w2w": 2, "w2p": 3, "p2p": 4}

        # Calculate information density
        density_array = self._compute_info_density(vecs)
        density = torch.tensor(density_array, dtype=torch.float32)
        avg_density = density.mean().item() or 0.1  # Avoid div by zero

        # Scale window sizes inversely with density - vectorized calculation
        scale = 1.0 / (avg_density + 0.1)
        min_w, max_w = self.cfg.min_window, self.cfg.max_window

        # Calculate all window sizes at once using tensor operations
        base_sizes = torch.tensor([2, 3, 4], dtype=torch.float32)
        scaled_sizes = torch.round(base_sizes * scale).int()

        # Apply bounds with tensor operations
        min_tensor = torch.tensor([min_w, min_w + 1, min_w + 2], dtype=torch.int32)
        max_tensor = torch.tensor([max_w, max_w, max_w], dtype=torch.int32)

        # Use element-wise min and max operations
        bounded_sizes = torch.max(min_tensor, torch.min(scaled_sizes, max_tensor))

        # Extract values
        w2w = bounded_sizes[0].item()
        w2p = bounded_sizes[1].item()
        p2p = bounded_sizes[2].item()

        return {"w2w": w2w, "w2p": w2p, "p2p": p2p}

    def _extract_relationships(
        self,
        docs: List[FrozenSet[str]],
        bm25: torch.Tensor,
        word_idx: Dict[str, int],
        words: List[str],
        rel_type: str,
        windows: Dict[str, int],
        doc_term_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[Any, float]:
        """Extract semantic relationships with vectorized operations."""
        # Get window size for this relationship type
        combo_size = windows.get(rel_type, 2)

        # Reuse doc_term_matrix if provided
        if doc_term_matrix is None and self.cfg.info_density:
            docs_tuple = tuple(docs)
            word_idx_tuple = tuple(word_idx.items())
            doc_term_matrix = self._create_doc_term_matrix(docs_tuple, word_idx_tuple)

        # Calculate document importance scores and valid docs in one pass
        doc_scores = []
        valid_doc_indices = []

        # Process in batches for better vectorization
        for doc_i, doc in enumerate(docs):
            if len(doc) < combo_size:
                continue

            indices = [word_idx[w] for w in doc if w in word_idx]
            if not indices:
                continue

            # Convert indices to tensor once
            idx_tensor = torch.tensor(indices, dtype=torch.long)

            # Calculate score using tensor indexing
            doc_score = bm25[doc_i, idx_tensor].mean().item()
            doc_scores.append(doc_score)
            valid_doc_indices.append(doc_i)

        # Early return if no valid documents
        if not valid_doc_indices:
            return {}

        # Apply density adjustment if needed
        if self.cfg.info_density and doc_term_matrix is not None:
            # Extract valid rows from doc_term_matrix
            valid_indices = torch.tensor(valid_doc_indices, dtype=torch.long)
            valid_rows = doc_term_matrix[valid_indices]

            # Create Bipolar vectors once
            vecs = [Bipolar(valid_rows[i]) for i in range(len(valid_doc_indices))]

            # Calculate density
            density_array = self._compute_info_density(vecs)
            density = torch.tensor(density_array, dtype=torch.float32)

            # Apply adjustment vectorized
            doc_scores_tensor = torch.tensor(doc_scores, dtype=torch.float32)
            density_factor = 1.0 + density
            adjusted_scores = doc_scores_tensor * density_factor.to(
                doc_scores_tensor.device
            )
            doc_scores = adjusted_scores.tolist()

        # Create doc_importance with scores
        doc_importance = list(zip(valid_doc_indices, doc_scores))

        # Sort by importance
        doc_importance.sort(key=lambda x: x[1], reverse=True)

        # Efficient combination collection
        combos = defaultdict(list)
        for rank, (doc_i, _) in enumerate(doc_importance):
            doc = docs[doc_i]
            indices = [word_idx[w] for w in doc if w in word_idx]

            # Generate combinations - this is hard to fully vectorize due to frozenset creation
            # but we can optimize the extraction
            for combo in combinations(indices, combo_size):
                # Create keys based on relationship type - extract once for efficiency
                if rel_type == "w2w":
                    key = frozenset([words[combo[0]], words[combo[1]]])
                elif rel_type == "w2p":
                    key = (words[combo[0]], frozenset(words[i] for i in combo[1:3]))
                elif rel_type == "p2p":
                    key = frozenset(
                        [
                            frozenset(words[i] for i in combo[:2]),
                            frozenset(words[i] for i in combo[2:4]),
                        ]
                    )
                else:
                    continue

                combos[key].append(rank)

        # Calculate relationship strengths with vectorized operations
        keys = list(combos.keys())
        if not keys:
            return {}

        # Convert to numpy arrays for vectorization
        ranks_array = [np.array(combos[k], dtype=np.float32) for k in keys]

        # Vectorized frequency calculation
        freqs = np.array([len(ranks) for ranks in ranks_array])
        freq_boosts = np.log1p(freqs)

        # Vectorized rank scoring
        if self.cfg.info_density and "density" in locals() and len(density) > 0:
            # Get CPU numpy array to ensure device compatibility
            density_np = density.cpu().numpy()

            # Vectorized decay factor
            decay = max(0.5, 1.0 - density_np.mean())

            # Vectorized rank score calculation for each array of ranks
            rank_scores = np.array([np.sum(decay**ranks) for ranks in ranks_array])
        else:
            # Vectorized calculation for each rank array
            rank_scores = np.array(
                [np.sum(1.0 / (1.0 + ranks)) for ranks in ranks_array]
            )

        # Vectorized final score calculation
        scores = rank_scores * freq_boosts
        max_score = np.max(scores) if scores.size > 0 else 0.0

        # Create normalized dictionary in one pass
        if max_score > 0:
            return {keys[i]: (scores[i] / max_score) for i in range(len(keys))}
        return {keys[i]: 0.0 for i in range(len(keys))}

    def relationships_to_distances(
        self, relationships: Dict[FrozenSet[str], float], word_idx: Dict[str, int]
    ) -> Dict[Tuple[int, int], float]:
        """Convert word relationships to distance pairs efficiently."""
        distances = {}

        # Filter valid word pairs once
        word_pairs = [
            (pair, strength)
            for pair, strength in relationships.items()
            if len(pair) == 2 and all(w in word_idx for w in pair)
        ]

        # Process in one pass
        for pair, strength in word_pairs:
            w1, w2 = tuple(pair)
            idx1, idx2 = word_idx[w1], word_idx[w2]
            dist = 1.0 - strength

            # Set both directions at once
            distances[(idx1, idx2)] = dist
            distances[(idx2, idx1)] = dist

        return distances


@dataclass
class SignLSHConfig:
    """Configuration for Locality-Sensitive Hashing (LSH)."""

    num_tables: int = 8
    bits_per_table: int = 16
    device: Union[str, torch.device] = "cpu"


class SignLSH:
    """Locality-Sensitive Hashing (LSH) for Bipolar vectors."""

    def __init__(self, config: SignLSHConfig):
        self.config = config
        self.projection_matrices = [
            torch.randn(config.bits_per_table, config.dimension, device=config.device)
            for _ in range(config.num_tables)
        ]
        self.hash_tables = [defaultdict(list) for _ in range(config.num_tables)]

    def hash_vector(self, vector: torch.Tensor, table_idx: int) -> Tuple[int, ...]:
        """Hash a Bipolar vector using the specified table index."""
        proj = self.projection_matrices[table_idx]
        projections = torch.sign(torch.matmul(proj, vector))
        bits = ((projections + 1) / 2).int()
        return tuple(bits.tolist())

    def index(self, vectors: List[torch.Tensor], ids: List[int]) -> None:
        """Index the Bipolar vectors for fast similarity search."""
        for i, vec in enumerate(vectors):
            for table_idx in range(self.config.num_tables):
                hash_key = self.hash_vector(vec, table_idx)
                self.hash_tables[table_idx][hash_key].append(ids[i])

    def query(self, vector: torch.Tensor, k: int = 10) -> List[int]:
        """Query the LSH index for similar vectors."""
        candidates = set()
        for table_idx in range(self.config.num_tables):
            hash_key = self.hash_vector(vector, table_idx)
            candidates.update(self.hash_tables[table_idx][hash_key])
        return list(candidates)[:k] if k > 0 else list(candidates)


@dataclass
class AdaptiveConfig:
    """Unified configuration for Adaptive Semantic model training and LSH."""

    dimension: int = 2048
    max_seq_len: int = 512
    min_freq: int = 3
    special_tokens: List[str] = field(
        default_factory=lambda: ["[CLS]", "[SEP]", "[REL]", "[PAD]", "[MASK]"]
    )
    device: Union[str, torch.device] = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # LSH parameters
    lsh_tables: int = 8
    lsh_bits: int = 16

    # Training parameters
    max_iterations: int = 500
    convergence_tol: float = 1e-4
    batch_size: Optional[int] = None
    initial_flip_prob: Optional[float] = None
    learning_rate: float = 0.1  # Added learning rate

    target_memory_fraction: float = 0.7
    min_batch_size: int = 16
    max_batch_size: int = 512

    # Adaptive dimension parameters
    use_adaptive_dimensions: bool = True
    adaptive_dim_ratio: float = 0.5
    min_dimension: int = 256

    # Vocabulary parameters
    min_words: int = 5000
    max_words_limit: int = 10000
    target_vocab_memory_fraction: float = 0.2

    # Training state
    vocab: Set[str] = field(default_factory=set)
    token_to_id: Dict[str, int] = field(default_factory=dict)
    converged: bool = False
    current_epoch: int = 0
    training_time: float = 0.0
    best_loss: float = float("inf")
    iterations_completed: int = 0
    loss_history: list = field(default_factory=list)

    def reset_training_state(self):
        self.converged = False
        self.current_epoch = 0
        self.training_time = 0.0
        self.best_loss = float("inf")
        self.iterations_completed = 0
        self.loss_history = []


class AdaptiveBitSemantic(nn.Module):
    """Adaptive Semantic model with integrated LSH and training."""

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        super().__init__()
        self.config = config or AdaptiveConfig()
        self.adaptive_bipolar = AdaptiveBipolar()  # Instance of AdaptiveBipolar
        self.text_processor = TextProcessor()

        self.dimension = self.config.dimension
        self.embedding = nn.Embedding(0, self.dimension).to(self.config.device)
        self.topics = []
        self.topic_to_idx = {}
        self.lsh = self._initialize_lsh()  # Initialize LSH directly

        def _init_model_architecture(self, w2w, w2p, p2p):
            """Initialize model architecture based on data."""
            sample_size = min(100, len(w2w) + len(w2p) + len(p2p))
            sample_vectors = self._get_sample_vectors(sample_size)
            info_density = self.adaptive_bipolar.compute_information_density(
                sample_vectors
            )
            num_segments = len(
                info_density
            )  # Num Segments is determined by info_density

            self.model = BipolarTransformer(
                BipolarTransformerConfig(
                    vocab_size=len(self.topics),
                    dimension=self.config.dimension,
                    heads=max(
                        1, num_segments // 2
                    ),  # Adaptive heads - NOW USING num_segments
                    num_layers=max(
                        1, int(math.log2(num_segments))
                    ),  # Adaptive layers - NOW USING num_segments
                    max_seq_len=self.config.max_seq_len,
                    use_bipolar_attention=True,
                )
            ).to(self.config.device)

    def fit(self, text: str) -> AdaptiveConfig:
        """Fit model to text data."""
        self.config.reset_training_state()  # Reset training state at start
        start_time = time.time()

        docs = self.text_processor.tokenize_docs(text)
        self.config.vocab = self.text_processor.extract_vocab(
            docs, self.config.min_freq, self.config.min_words
        )
        logger.info(f"Extracted {len(self.config.vocab)} words")

        w2w, w2p, p2p = self.text_processor.build_bm25_relationships(
            docs, self.config.vocab, relationship_types=["w2w", "w2p", "p2p"]
        )
        logger.info("Generated word relationships")

        self.topics = list(self.config.vocab)
        self.topic_to_idx = {w: i for i, w in enumerate(self.topics)}
        self.embedding = nn.Embedding(len(self.topics), self.config.dimension).to(
            self.config.device
        )  # Re-initialize embedding

        self._init_model_architecture(w2w, w2p, p2p)  # Initialize model architecture

        target_distances = self.text_processor.relationships_to_distances(
            w2w, self.topic_to_idx
        )  # Use w2w for training targets

        self._build_lsh_index()  # Build LSH Index - moved LSH index build here

        self._train_model(target_distances)  # Train Model

        self.config.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.config.training_time:.2f} seconds")
        return self.config

    def _init_model_architecture(self, w2w, w2p, p2p):
        """Initialize model architecture based on data."""
        sample_size = min(100, len(w2w) + len(w2p) + len(p2p))
        sample_vectors = self._get_sample_vectors(sample_size)
        info_density = self.adaptive_bipolar.compute_information_density(sample_vectors)
        num_segments = len(info_density)  # Num Segments is determined by info_density

        self.model = BipolarTransformer(
            BipolarTransformerConfig(
                vocab_size=len(self.topics),
                dimension=self.config.dimension,
                heads=max(1, num_segments // 2),  # Adaptive heads
                num_layers=max(1, int(math.log2(num_segments))),  # Adaptive layers
                max_seq_len=self.config.max_seq_len,
                use_bipolar_attention=True,
            )
        ).to(self.config.device)

    def _build_lsh_index(self):
        """Build LSH index after vocabulary and embedding are set."""
        info_density = self.adaptive_bipolar.compute_information_density(
            [self._get_vector(i) for i in range(len(self.topics))]
        )
        num_segments = len(info_density)  # Num Segments is determined by info_density
        self.lsh = SignLSH(
            self.config.dimension,
            self.config.lsh_bits,  # LSH bits - kept as before (heuristic based on vocab size)
            self.config.device,
            num_tables=max(
                1, num_segments // 4
            ),  # Adaptive num_tables - NOW USING num_segments!
        )
        self._build_index_vectors()  # Call internal index build - renamed for clarity

    def _train_model(self, target_distances):
        """Train the semantic model."""
        optimizer_config = SignSGDConfig(
            initial_flip_prob=self.config.initial_flip_prob, use_loss_trend=True
        )  # Create SignSGDConfig
        optimizer = SignSGD(
            self.model.parameters(), optimizer_config
        )  # Pass config instance
        data_loader = self._create_dataloader(target_distances)

        while not self.config.converged:
            epoch_loss = self._train_epoch_logic(data_loader, optimizer)
            self.config.loss_history.append(epoch_loss)

            if len(self.config.loss_history) > 2:
                self.config.converged = AdaptiveBipolar.detect_convergence(
                    self.config.loss_history
                )

            self.config.current_epoch += 1

    def _create_dataloader(self, target_distances):
        """Create DataLoader for training."""
        idx_pairs = torch.tensor(
            list(target_distances.keys()), device=self.config.device
        )
        targets = torch.tensor(
            list(target_distances.values()),
            dtype=torch.float32,
            device=self.config.device,
        )
        dataset = TensorDataset(idx_pairs, targets)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size or self._calculate_dynamic_batch_size(),
            shuffle=True,
        )

    def _train_epoch_logic(self, data_loader, optimizer):
        """Logic for a single training epoch."""
        epoch_loss = 0.0
        num_samples = 0
        for batch in data_loader:
            batch_pairs, batch_targets = batch
            num_samples += len(batch_pairs)

            optimizer.zero_grad()
            vecs_i = self.model.token_embed(
                batch_pairs[:, 0]
            )  # Access embedding directly from model
            vecs_j = self.model.token_embed(
                batch_pairs[:, 1]
            )  # Access embedding directly from model
            distances = (
                (STE.ste_sign(vecs_i) != STE.ste_sign(vecs_j)).float().mean(dim=1)
            )  # Use inline distance calculation
            loss = F.mse_loss(distances, batch_targets)
            loss.backward()
            optimizer.step(loss.item())  # Pass loss to optimizer step

            with torch.no_grad():
                self.model.token_embed.weight.data.copy_(
                    STE.ste_sign(self.model.token_embed.weight.data)
                )  # Access embedding directly from model

            epoch_loss += loss.item() * len(batch_pairs)
        return epoch_loss / num_samples

    def _build_index_vectors(
        self,
    ) -> None:  # Renamed from _build_index to clarify scope
        """Build LSH index for fast similarity search."""
        if self.lsh is None:
            raise ValueError("LSH index not initialized.")

        with torch.no_grad():
            indices = torch.arange(len(self.topics), device=self.config.device)
            vectors = [self._get_vector(idx.item()).to("cpu") for idx in indices]
            ids = indices.cpu().tolist()
            self.lsh.index(vectors, ids)

    def _calculate_dynamic_batch_size(self) -> int:
        """Dynamically calculate batch_size based on available GPU memory and vocab size."""
        available_memory_gb = self._get_available_memory()
        vocab_size = len(self.topics)
        if (
            available_memory_gb <= 0 or vocab_size == 0
        ):  # Handle cases where memory or vocab is not available
            return self.config.min_batch_size  # Fallback to min batch size

        sample_batch_size = min(
            32, vocab_size
        )  # Use smaller sample size if vocab is small
        sample_batch_pairs = torch.randint(
            0, vocab_size, (sample_batch_size, 2), device=self.config.device
        )  # Create sample batch pairs

        # Estimate memory usage based on a sample batch
        memory_per_sample_mb = self._measure_memory_batch(sample_batch_pairs)
        if memory_per_sample_mb <= 0:  # Fallback if memory measurement fails
            return self.config.min_batch_size

        target_memory_mb = (
            available_memory_gb * 1024 * self.config.target_memory_fraction
        )  # Target memory based on config
        batch_size = max(
            self.config.min_batch_size,
            min(
                self.config.max_batch_size, int(target_memory_mb / memory_per_sample_mb)
            ),
        )  # Calculate batch size

        return batch_size

    def _measure_memory_batch(self, batch_pairs: torch.Tensor) -> float:
        """Estimate memory usage for a sample batch in MB."""
        try:
            num_samples = batch_pairs.size(0)
            embedding_memory_mb = (num_samples * 2 * self.config.dimension * 4) / (
                1024**2
            )  # Estimate embedding memory
            distance_memory_mb = (num_samples * 4) / (
                1024**2
            )  # Estimate distance memory
            return max(
                0, embedding_memory_mb + distance_memory_mb
            )  # Return total memory, ensure non-negative
        except Exception:
            return 0.0  # Return 0 on failure

    def _get_sample_vectors(self, sample_size):
        """Utility to get sample Bipolar vectors for analysis."""
        sample_indices = torch.randperm(len(self.topics))[:sample_size]
        return [self._get_vector(idx) for idx in sample_indices]

    def _prepare_training_examples(self, w2w, w2p, p2p):
        """Prepare training examples from relationships."""
        examples = []
        examples.extend(
            {"type": "w2w", "pair": pair, "target": strength}
            for pair, strength in w2w.items()
        )  # Word-Word examples
        # You can extend for w2p, p2p if needed, adjusting input preparation in _prepare_inputs
        return examples

    def _batch_iterator(self, data, batch_size):
        """Generic batch iterator."""
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def _prepare_inputs(self, batch_examples):
        """Prepare model inputs from batch examples (currently for w2w)."""
        indices = [
            self.topic_to_idx[word]
            for ex in batch_examples
            for word in ex["pair"]  # Assuming 'pair' is frozenset of words
        ]
        return torch.tensor(indices, device=self.config.device).reshape(
            -1, 2
        )  # Reshape into pairs

    def _extract_predictions(self, outputs, rel_type):
        """Extract model predictions based on relationship type."""
        # For now, assuming outputs are direct distances for all types
        return outputs.squeeze(1)  # Adjust if your model outputs something different

    # Semantic operations - similar API as before, now methods of AdaptiveBitSemantic
    def find_similar(self, word: str, n: int = 10) -> List[Tuple[str, float]]:
        """Find similar words using LSH index."""
        if (
            not self.lsh or not self.config.vocab
        ):  # Check if LSH is initialized and vocab exists
            logger.warning("LSH index or vocabulary not initialized.")
            return []
        if word not in self.topic_to_idx:
            logger.warning(f"Word '{word}' not in vocabulary.")
            return []

        query_vector = self._get_word_vector(word)
        if query_vector is None:
            return []  # Handle case where word vector couldn't be retrieved

        candidate_ids = self.lsh.query(query_vector)  # Query LSH index
        if not candidate_ids:
            return []  # Return empty list if no candidates

        distances = []
        for candidate_idx in candidate_ids:
            if candidate_idx != self.topic_to_idx[word]:  # Avoid self-similarity
                candidate_vector = self._get_vector(candidate_idx)
                if (
                    candidate_vector is not None
                ):  # Handle cases where vector retrieval fails
                    distance = query_vector.distance(candidate_vector).item()
                    distances.append((self.topics[candidate_idx], distance))

        distances.sort(key=lambda x: x[1])  # Sort by distance
        return [
            (word, dist) for word, dist in distances[:n]
        ]  # Return top N similar words

    def analogy(self, a: str, b: str, c: str, n: int = 5) -> List[Tuple[str, float]]:
        """Solve analogy queries (a:b::c:?)."""
        if (
            not all(w in self.topic_to_idx for w in [a, b, c]) or not self.lsh
        ):  # Check vocab and LSH
            return []

        vectors = [self._get_word_vector(w) for w in (a, b, c)]
        if not all(vectors):  # Ensure all vectors are retrieved
            return []

        resilience = self.adaptive_bipolar.multi_bind_resilience(
            vectors
        )  # Resilience calculation
        weights = torch.softmax(resilience, dim=0)  # Softmax weights

        bind_op = self.adaptive_bipolar.entropy_guided_binding(
            vectors
        )  # Entropy guided binding
        target = bind_op(vectors[0], vectors[1])
        target = bind_op(target, vectors[2])

        candidate_ids = self.lsh.query(target)  # Query LSH index
        candidate_ids_filtered = [
            idx
            for idx in candidate_ids
            if idx
            not in {self.topic_to_idx[w] for w in (a, b, c)}  # Filter out query words
        ]

        distances = []
        for idx in candidate_ids_filtered:
            cand_vec = self._get_vector(idx)
            if cand_vec is not None:  # Handle vector retrieval failures
                diff = (target.tensor != cand_vec.tensor).float()
                dist = torch.sum(diff * weights).item()  # Weighted distance
                distances.append((self.topics[idx], dist))

        distances.sort(key=lambda x: x[1])  # Sort by distance
        return [
            (word, dist) for word, dist in distances[:n]
        ]  # Return top N analogy words

    def visualize_semantic_space(
        self,
        words: Optional[List[str]] = None,
        n_words: int = 10,
        filename: Optional[str] = None,
    ) -> str:
        """Visualize semantic space as bit patterns and distances."""
        if not self.topics:
            return "No words in vocabulary to visualize."

        words_to_visualize = (
            words or self.topics[:n_words]
        )  # Use provided words or top N
        valid_words = [
            w for w in words_to_visualize if w in self.topic_to_idx
        ]  # Filter valid words
        if not valid_words:
            return "No valid words provided for visualization."
        words_to_visualize = valid_words[:n_words]  # Limit to max words

        output = ["Semantic Bitspace Visualization:"]
        output.append("=" * 50)
        output.append(f"{'Word':<15} | {'Bit Pattern (first 25 dims)':<30}")
        output.append("-" * 50)

        for word in words_to_visualize:
            vec = self._get_word_vector(word)
            if vec is not None:  # Handle cases where vector is None
                bit_pattern = "".join(
                    "1" if b else "0" for b in vec.tensor.cpu()[:25].tolist()
                )
                output.append(f"{word:<15} | {bit_pattern}")

        output.append("\nSemantic Distances (first few words):")
        output.append("-" * 50)
        comparison_words = words_to_visualize[
            : min(5, len(words_to_visualize))
        ]  # Limit comparisons for readability

        for i in range(len(comparison_words)):
            for j in range(i + 1, len(comparison_words)):
                vec_i = self._get_word_vector(comparison_words[i])
                vec_j = self._get_word_vector(comparison_words[j])
                if (
                    vec_i is not None and vec_j is not None
                ):  # Handle cases where vectors are None
                    dist = vec_i.distance(vec_j).item()
                    output.append(
                        f"{comparison_words[i]:<15} <-> {comparison_words[j]:<15}: {dist:.4f}"
                    )

        visualization = "\n".join(output)
        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(visualization)
                logger.info(f"Visualization saved to '{filename}'")
            except Exception as e:
                logger.error(f"Error saving visualization to file: {e}")
        return visualization


@dataclass
class BipolarTransformerConfig:
    """Configuration for the BipolarTransformer."""

    vocab_size: int
    dimension: int = 512
    heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 100
    use_bipolar_attention: bool = True
    ff_expansion: int = 4


class BipolarTransformer(nn.Module):
    """A unified transformer class supporting bipolar attention."""

    def __init__(self, config: BipolarTransformerConfig) -> None:
        """
        Initialize the BipolarTransformer."""
        super().__init__()
        self.config = config

        # Token and positional embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.dimension)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.dimension)
        self.embed_scale = nn.Parameter(torch.ones(1))

        # Initialize embeddings with bipolar values
        with torch.no_grad():
            self.token_embed.weight.data = torch.sign(
                torch.randn(config.vocab_size, config.dimension)
            )
            self.pos_embed.weight.data = torch.sign(
                torch.randn(config.max_seq_len, config.dimension)
            )

        # Transformer blocks - now defined directly within BipolarTransformer
        self.blocks = nn.ModuleList(
            [self._create_transformer_block() for _ in range(config.num_layers)]
        )

        # Output projection
        self.output_proj = BipolarLinear(config.dimension, config.dimension)

    def _create_transformer_block(self) -> nn.Module:
        """Create a transformer block (now defined inline)."""
        dimension = self.config.dimension
        heads = self.config.heads
        ff_expansion = self.config.ff_expansion
        use_bipolar_attention = self.config.use_bipolar_attention

        if dimension % heads != 0:
            raise ValueError(
                f"Dimension {dimension} must be divisible by heads {heads}"
            )

        block = nn.Module()  # Use a generic Module to hold layers
        block.dimension = dimension  # Store dimension for attention method
        block.heads = heads
        block.head_dim = dimension // heads
        block.use_bipolar_attention = use_bipolar_attention

        # Attention projections
        block.query_proj = BipolarLinear(dimension, dimension, bias=False)
        block.key_proj = BipolarLinear(dimension, dimension, bias=False)
        block.value_proj = BipolarLinear(dimension, dimension, bias=False)
        block.output_proj = BipolarLinear(dimension, dimension, bias=False)

        # Feed-forward network
        block.ff1 = BipolarLinear(dimension, dimension * ff_expansion)
        block.ff2 = BipolarLinear(dimension * ff_expansion, dimension)

        # Scaling factors
        block.attn_scale = nn.Parameter(torch.ones(1))
        block.ff_scale = nn.Parameter(torch.ones(1))

        # Define the forward pass for the block inline
        def block_forward(x: torch.Tensor) -> torch.Tensor:
            # Self-attention
            attended = self._attention(x, block)  # Pass block instance to attention
            x = STE.ste_sign(x + block.attn_scale * attended)  # Use generic ste_sign

            # Feed-forward network
            ff_hidden = torch.relu(block.ff1(x))
            ff_output = block.ff2(ff_hidden)
            output = STE.ste_sign(
                x + block.ff_scale * ff_output
            )  # Use generic ste_sign
            return output

        block.forward = block_forward.__get__(block, nn.Module)  # Bind forward method

        # Define the attention method inline (now needs block as argument)
        def _attention(x: torch.Tensor, block_instance: nn.Module) -> torch.Tensor:
            """Compute attention using bipolar (Hamming distance-based) attention."""
            batch_size, seq_len, _ = x.shape

            # Project inputs
            q = STE.ste_sign(block_instance.query_proj(x))  # Use generic ste_sign
            k = STE.ste_sign(block_instance.key_proj(x))  # Use generic ste_sign
            v = STE.ste_sign(block_instance.value_proj(x))  # Use generic ste_sign

            # Bipolar attention (Hamming distance-based)
            q_bipolar = Bipolar(q)
            k_bipolar = Bipolar(k)
            v_bipolar = Bipolar(v)

            # Compute Hamming similarity
            xor_result = (
                torch.einsum("bhsd,bhtd->bhst", [q_bipolar.tensor, k_bipolar.tensor])
                < 0
            )
            hamming_similarity = 1.0 - xor_result.float().mean(dim=-1)
            attention_scores = hamming_similarity

            # Apply softmax to attention scores
            attention_weights = F.softmax(
                block_instance.attn_scale * attention_scores, dim=-1
            )

            # Compute context using attention weights
            context = torch.einsum(
                "bhst,bhtd->bhsd", [attention_weights, v_bipolar.tensor]
            )

            # Reshape and project output
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, block_instance.dimension)
            )
            return STE.ste_sign(
                block_instance.output_proj(context)
            )  # Use generic ste_sign

        block._attention = _attention.__get__(block, nn.Module)  # Bind attention method

        return block  # Return the created block module

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer."""
        batch_size, seq_len = token_ids.shape

        # Token embeddings
        token_embeds = STE.ste_sign(
            self.token_embed(token_ids) * self.embed_scale
        )  # Use generic ste_sign

        # Positional embeddings
        positions = (
            torch.arange(seq_len, device=token_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_embeds = STE.ste_sign(self.pos_embed(positions))  # Use generic ste_sign

        # Combine token and positional embeddings
        x = token_embeds * pos_embeds

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)  # Now calling the inline forward method of the block

        # Output projection
        output = self.output_proj(x)
        return output

    def get_binary_params_percent(self) -> float:
        """Calculate the percentage of binary parameters."""
        total_params = 0
        binary_params = 0
        for name, param in self.named_parameters():
            param_size = param.numel()
            total_params += param_size
            if "proj" in name or "ff" in name or "embed" in name:
                binary_params += param_size
        return binary_params / total_params * 100


class ProgressiveHDCDistiller:
    """ProgressiveHDCDistiller class."""

    def __init__(self, model, dataloader):
        """Initialize ProgressiveHDCDistiller."""
        self.model = model
        self.dataloader = dataloader
        self.transformed_tensors = set()

    def transform_next_layer(self):
        """Transform the next most promising layer to HDC sign space."""
        untransformed = [
            name
            for name, param in self.model.named_parameters()
            if name not in self.transformed_tensors
        ]
        if not untransformed:
            return False
        baseline_score = self._evaluate_model()  # Get baseline performance
        impacts = []
        original_states = {}
        for tensor_name in untransformed:
            param = dict(self.model.named_parameters())[tensor_name]
            original_states[tensor_name] = param.data.clone()  # Store original state
            param.data = torch.sign(param.data)  # Binarize the layer
            new_score = (
                self._evaluate_model()
            )  # Evaluate performance after binarization
            impact = baseline_score - new_score  # Calculate performance impact
            impacts.append((tensor_name, impact))
            param.data = original_states[tensor_name]  # Restore original state
        impacts.sort(key=lambda x: x[1])  # Sort by impact (smaller impact is better)
        best_tensor = impacts[0][0]  # Choose the layer with least negative impact
        param = dict(self.model.named_parameters())[best_tensor]
        param.data = torch.sign(param.data)  # Permanently binarize the best layer
        self.transformed_tensors.add(best_tensor)  # Mark as transformed
        self._targeted_finetune(best_tensor)  # Finetune the binarized layer
        return True

    def _targeted_finetune(self, tensor_name):
        """Finetune specific tensor using SignSGD."""
        target_param = dict(self.model.named_parameters())[tensor_name]
        optimizer = SignSGD(
            [target_param], initial_flip_prob=0.1
        )  # You might need to adjust flip_prob
        for _ in range(10):  # Number of finetuning iterations - adjust as needed
            for batch in self.dataloader:
                loss = self._compute_task_loss(batch)  # Compute task-specific loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def _evaluate_model(self):
        """
        Evaluate the model's performance on a validation/test dataset.

        THIS METHOD IS A PLACEHOLDER AND MUST BE IMPLEMENTED BASED ON THE SPECIFIC TASK AND MODEL.

        It should:
        1. Set the model to evaluation mode (model.eval()).
        2. Iterate through a validation/test dataloader (you might need a separate one).
        3. For each batch, perform a forward pass and calculate relevant metrics
           (e.g., accuracy, loss, F1-score).
        4. Aggregate the metrics across all batches to get an overall performance score.
        5. Return a SINGLE SCALAR VALUE representing the performance.
           Higher values should generally indicate better performance (e.g., accuracy, F1-score).
           If using loss, you might return the *negative* loss so that higher values are still "better" in the impact calculation.
        6. Set the model back to training mode (model.train()) if needed after evaluation.

        Example (for a classification task, placeholder):
        ```python
        def _evaluate_model(self):
            self.model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad(): # Important for evaluation
                for batch in validation_dataloader: # Assuming you have a validation_dataloader
                    inputs, labels = batch
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / total_samples
            self.model.train() # Back to training mode if needed
            return accuracy # Return accuracy as the metric
        ```

        Returns:
            float: A scalar value representing the model's performance.
                   Implement this based on your task.
        """
        raise NotImplementedError(
            "Implement the _evaluate_model method based on your task and evaluation metric."
        )
        return 0.0  # Dummy return value - replace with your actual metric

    def _compute_task_loss(self, batch):
        """
        Compute the loss for a single batch of data.

        THIS METHOD IS A PLACEHOLDER AND MUST BE IMPLEMENTED BASED ON THE SPECIFIC TASK AND MODEL.

        It should:
        1. Perform a forward pass through the model with the given batch.
        2. Calculate the appropriate loss function for your task (e.g., CrossEntropyLoss, MSELoss).
        3. Return the calculated loss.

        Example (for a classification task using CrossEntropyLoss, placeholder):
        ```python
        import torch.nn as nn

        def _compute_task_loss(self, batch):
            inputs, labels = batch
            outputs = self.model(inputs)
            criterion = nn.CrossEntropyLoss() # Or your task-specific loss
            loss = criterion(outputs, labels)
            return loss
        ```

        Args:
            batch: A batch of data from the dataloader.

        Returns:
            torch.Tensor: The calculated loss for the batch.
                          Implement this based on your task and loss function.
        """
        raise NotImplementedError(
            "Implement the _compute_task_loss method based on your task and loss function."
        )
        return torch.tensor(0.0)  # Dummy return value - replace with your actual loss


class BipolarNetwork(nn.Module):
    """Unified bipolar network supporting configurable architectures for classification."""

    def __init__(
        self,
        input_size: int,
        architecture: str = "wide",
        hidden_size: int = None,
        num_layers: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size

        # Configure architecture
        if architecture == "wide":
            hidden_size = hidden_size or input_size * 3
            self.layers = nn.ModuleList(
                [
                    self._make_layer(input_size, hidden_size, bias),
                    self._make_layer(hidden_size, 1, bias),
                ]
            )
        else:  # deep
            hidden_size = hidden_size or input_size
            self.layers = nn.ModuleList(
                [self._make_layer(input_size, hidden_size, bias)]
                + [
                    self._make_layer(hidden_size, hidden_size, bias)
                    for _ in range(num_layers - 1)
                ]
                + [self._make_layer(hidden_size, 1, bias)]
            )

    def _make_layer(self, in_features: int, out_features: int, bias: bool) -> nn.Module:
        """Create a bipolar linear layer."""
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.sign(torch.randn(out_features, in_features)))
        layer.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        layer.forward = lambda x: self._layer_forward(x, layer.weight, layer.bias)
        return layer

    @staticmethod
    def _layer_forward(
        x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through a bipolar layer using Hamming distance."""
        x_bipolar = x if hasattr(x, "sign") else torch.sign(x)  # Ensure bipolar input
        xor_result = x_bipolar.unsqueeze(1) != weight.unsqueeze(0)
        similarity = 1.0 - xor_result.float().mean(dim=2)
        output = similarity * 2.0 - 1.0  # Convert similarity to bipolar
        return output + bias.unsqueeze(0) if bias is not None else output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.sign(x)  # Ensure input is bipolar
        for i, layer in enumerate(self.layers[:-1]):
            x = STE.ste_sign(layer(x))  # Apply STE to maintain bipolar outputs
        x = self.layers[-1](x)  # Final layer output
        return x  # Return bipolar output


def compare_bipolar_classifiers(
    X, y, architectures=None, cv_folds=5, train_ratio=0.7, report=True
):
    """Compare bipolar neural network architectures for binary classification.

    Args:
        X: Feature tensor of shape [n_samples, n_features]
        y: Target tensor of shape [n_samples, 1]
        architectures: Dictionary of architecture configs to compare (default: wide and deep)
        cv_folds: Number of cross-validation folds
        train_ratio: Train/test split ratio
        report: Whether to print and plot results

    Returns:
        tuple: (best_model, test_accuracy)
    """
    # Set default architectures if none provided
    if architectures is None:
        architectures = {
            "wide": {"architecture": "wide"},
            "deep": {"architecture": "deep", "num_layers": 3},
        }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, shuffle=True, random_state=42
    )

    # Set up cross-validation
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Track scores per architecture
    scores = {name: [] for name in architectures}

    # Run cross-validation for each architecture
    for name, config in architectures.items():
        for train_idx, val_idx in kfold.split(X_train, y_train.squeeze()):
            # Create and train model
            model = BipolarNetwork(input_size=X.shape[1], **config)
            val_acc = train_bipolar_classifier(
                model=model,
                X_train=X_train[train_idx],
                y_train=y_train[train_idx],
                X_val=X_train[val_idx],
                y_val=y_train[val_idx],
                verbose=False,
            )
            scores[name].append(val_acc)
            if report:
                print(f"Fold accuracy ({name}): {val_acc:.4f}")

    # Report CV results
    if report:
        print("\nCross-validation results:")
        for name, accs in scores.items():
            print(
                f"{name.capitalize()}: {np.mean(accs) * 100:.2f}% (±{np.std(accs) * 100:.2f}%)"
            )

    # Select best architecture
    best_arch = max(scores.items(), key=lambda x: np.mean(x[1]))[0]

    # Train final model with best architecture
    final_model = BipolarNetwork(input_size=X.shape[1], **architectures[best_arch])
    _ = train_bipolar_classifier(
        model=final_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        verbose=report,
    )

    # Evaluate on test set
    final_model.eval()
    with torch.no_grad():
        bipolar_output = final_model(X_test)
        binary_predictions = bipolar_to_binary(bipolar_output, output_type="01")
        test_acc = (binary_predictions == y_test).float().mean().item()

        if report:
            print(f"\nTest accuracy: {test_acc * 100:.2f}%")

            # Plot ROC curve
            prob_output = bipolar_to_binary(bipolar_output, output_type="prob")
            fpr, tpr, _ = roc_curve(y_test.numpy(), prob_output.numpy())
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {best_arch.capitalize()} Bipolar Network")
            plt.grid(True)
            plt.show()

    return final_model, test_acc


def train_bipolar_classifier(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    max_epochs=300,
    batch_size=10,
    flip_prob=0.1,
    patience=20,
    monitor="accuracy",  # Options: "accuracy", "loss", "both"
    verbose=True,
):
    """Train a bipolar neural network for classification.

    Args:
        model: BipolarNetwork model
        X_train: Training feature tensor
        y_train: Training target tensor (binary 0/1)
        X_val: Validation feature tensor
        y_val: Validation target tensor (binary 0/1)
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        flip_prob: Initial flip probability for SignSGD
        patience: Early stopping patience
        monitor: Metric to monitor for early stopping ("accuracy", "loss", or "both")
        verbose: Whether to print progress

    Returns:
        tuple: (best_validation_accuracy, validation_loss_at_best_accuracy)
    """
    # Initialize optimizer and loss
    optimizer = SignSGD(model.parameters(), SignSGDConfig(initial_flip_prob=flip_prob))
    loss_fn = nn.MSELoss()

    # Convert binary targets (0/1) to bipolar targets (-1/+1)
    y_train_bipolar = y_train * 2 - 1
    y_val_bipolar = y_val * 2 - 1

    # Track best model
    best_acc = -np.inf
    best_loss = np.inf
    best_weights = None
    patience_counter = 0

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Process mini-batches
        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            x_batch = X_train[start:end]
            y_batch = y_train_bipolar[start:end]

            # Forward pass
            bipolar_output = model(x_batch)

            # Compute loss
            loss = loss_fn(bipolar_output, y_batch)
            epoch_loss += loss.item()
            n_batches += 1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            bipolar_output = model(X_val)
            val_loss = loss_fn(bipolar_output, y_val_bipolar).item()
            binary_predictions = bipolar_to_binary(bipolar_output, output_type="01")
            val_acc = (binary_predictions == y_val).float().mean().item()

            # Determine if this is the best model
            improved = False

            if monitor == "accuracy":
                if val_acc > best_acc:
                    improved = True
                    best_acc = val_acc
                    best_loss = val_loss  # Track loss at best accuracy
            elif monitor == "loss":
                if val_loss < best_loss:
                    improved = True
                    best_loss = val_loss
                    best_acc = val_acc  # Track accuracy at best loss
            else:  # "both" - use a weighted combination
                # Calculate a combined score (higher is better)
                current_score = val_acc - 0.1 * val_loss
                best_score = best_acc - 0.1 * best_loss
                if current_score > best_score:
                    improved = True
                    best_acc = val_acc
                    best_loss = val_loss

            # Update best model if improved
            if improved:
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Verbose reporting
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{max_epochs}, "
                    f"Train Loss: {epoch_loss / n_batches:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    model.load_state_dict(best_weights)

    # Final evaluation of best model
    model.eval()
    with torch.no_grad():
        final_output = model(X_val)
        final_loss = loss_fn(final_output, y_val_bipolar).item()
        final_preds = bipolar_to_binary(final_output, output_type="01")
        final_acc = (final_preds == y_val).float().mean().item()

    if verbose:
        print(f"Best model - Val Acc: {final_acc:.4f}, Val Loss: {final_loss:.4f}")

    return final_acc, final_loss


def bipolar_to_binary(bipolar_output, output_type="bool"):
    """Convert bipolar values (-1/+1) to binary representation.

    Args:
        bipolar_output: Tensor or Bipolar object with values in {-1, 1} or {False, True}
        output_type: Conversion type - "bool", "01", or "prob"

    Returns:
        torch.Tensor: Converted tensor according to output_type
    """
    # Handle Bipolar objects
    if isinstance(bipolar_output, Bipolar):
        unpacked_tensor = bipolar_output.tensor
    else:
        # Handle regular tensors (assuming bipolar values)
        unpacked_tensor = bipolar_output

    # Convert based on output type
    if output_type == "bool":
        return unpacked_tensor > 0  # True for +1, False for -1
    elif output_type == "01":
        return (unpacked_tensor > 0).float()  # 1.0 for +1, 0.0 for -1
    elif output_type == "prob":
        return (unpacked_tensor + 1) / 2  # Maps -1 to 0, +1 to 1, linearly in between
    else:
        raise ValueError(
            f"Unsupported output_type: {output_type}. Use 'bool', '01', or 'prob'."
        )


def train_semantic_model(text, config=None, text_processor=None):
    """Train a semantic model on text data.

    Args:
        text: Input text corpus
        config: AdaptiveConfig object or dict of config parameters
        text_processor: Custom TextProcessor instance

    Returns:
        tuple: (model, config)
    """
    # Handle config
    if config is None:
        config = AdaptiveConfig()
    elif isinstance(config, dict):
        config = AdaptiveConfig(**config)

    # Create model
    model = AdaptiveBitSemantic(config).to(config.device)

    # Fit model
    config = model.fit(text, text_processor=text_processor)

    return model, config


def train_transformer(vocab_size, config=None):
    """Create and initialize a BipolarTransformer model.

    Args:
        vocab_size: Size of vocabulary
        config: BipolarTransformerConfig object or dict of config parameters

    Returns:
        BipolarTransformer: Initialized transformer model
    """
    # Handle config
    if config is None:
        config = BipolarTransformerConfig(vocab_size=vocab_size)
    elif isinstance(config, dict):
        config_dict = config.copy()
        config_dict["vocab_size"] = vocab_size
        config = BipolarTransformerConfig(**config_dict)
    else:
        # Ensure vocab_size is set
        config.vocab_size = vocab_size

    # Create and return model
    return BipolarTransformer(config)


def generic_trainer(model, data_loader, optimizer, distance_fn, sign_fn, config):
    """Universal trainer for models using bipolar representations.

    Args:
        model: Model to train
        data_loader: DataLoader with training data
        optimizer: Optimizer instance
        distance_fn: Function to compute distances between vectors
        sign_fn: Function to convert to bipolar representation
        config: Training configuration with loss_history and other fields

    Returns:
        config: Updated configuration with training results
    """
    # Initialize adaptive utilities
    adaptive = AdaptiveBipolar()

    # Reset training state
    config.loss_history = []
    config.best_loss = float("inf")
    config.iterations_completed = 0
    config.converged = False

    # Extract model and data characteristics
    vocab_size = getattr(
        model,
        "vocab_size",
        getattr(
            model.embedding,
            "num_embeddings",
            data_loader.dataset.tensors[0].max().item() + 1,
        ),
    )
    data_size = len(data_loader.dataset)
    dim = getattr(config, "dimension", model.embedding.weight.size(1))

    # Adaptive parameters
    sample_size = max(1, min(100, int(math.log2(vocab_size + 1))))
    analysis_interval = max(1, int(math.sqrt(data_size / max(1, vocab_size))))
    report_interval = max(1, int(math.log2(config.max_iterations + 1)))

    # Sample vectors for analysis if possible
    sample_vectors = []
    try:
        if hasattr(model, "embedding") and sample_size > 0:
            with torch.no_grad():
                sample_indices = torch.randperm(vocab_size)[:sample_size]
                sample_vectors = [
                    Bipolar(model.embedding.weight[idx]) for idx in sample_indices
                ]
    except (RuntimeError, IndexError):
        pass  # Skip if sampling fails

    # Memory-based adaptation
    try:
        available_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
            if torch.cuda.is_available()
            else 4.0
        )  # Default to 4GB if CPU
    except Exception as e:
        logger.warning(f"Memory detection failed: {e}")
        available_memory = 4.0  # Fallback

    memory_scale = max(
        0.1, min(1.0, available_memory / max(0.1, vocab_size * dim / 1e6))
    )

    # Dynamic batch size if not fixed
    if not hasattr(config, "batch_size") or not config.batch_size:
        config.batch_size = adaptive.adapt_batch_size(
            0,
            available_memory,
            min_size=max(1, int(vocab_size / 100)),
            max_size=max(16, int(vocab_size / 10)),
        )

    # Adaptive convergence threshold
    conv_threshold = adaptive.adaptive_convergence_threshold(vocab_size, dim)
    min_dim = max(1, int(math.sqrt(vocab_size)))
    min_ratio = min_dim / dim if dim > 0 else 0.1

    # Training loop
    for iteration in range(config.max_iterations):
        total_loss = 0.0
        num_samples = 0

        # Process batches
        for batch in data_loader:
            batch_pairs, batch_targets = batch
            num_samples += len(batch_pairs)

            # Training step
            optimizer.zero_grad()
            vecs_i = model.embedding(batch_pairs[:, 0])
            vecs_j = model.embedding(batch_pairs[:, 1])
            distances = distance_fn(vecs_i, vecs_j)
            loss = F.mse_loss(distances, batch_targets)
            loss.backward()
            optimizer.step(loss.item())

            # Apply sign function to weights
            with torch.no_grad():
                model.embedding.weight.data.copy_(sign_fn(model.embedding.weight.data))

            total_loss += loss.item() * len(batch_pairs)

        # Update metrics
        avg_loss = total_loss / num_samples
        config.loss_history.append(avg_loss)
        config.iterations_completed = iteration + 1

        if avg_loss < config.best_loss:
            config.best_loss = avg_loss

        # Adaptive analysis and optimization
        if iteration % analysis_interval == 0 and sample_vectors:
            # Analyze loss trend
            loss_trend = adaptive.analyze_loss_trend(config.loss_history)

            # Get gradient variance
            grad_var = 0.0
            if (
                hasattr(model.embedding, "weight")
                and model.embedding.weight.grad is not None
            ):
                grad_var = torch.var(model.embedding.weight.grad).item()

            # Adapt optimizer parameters
            if hasattr(optimizer, "flip_prob"):
                optimizer.flip_prob = adaptive.adapt_learning_rate(
                    iteration,
                    loss_trend,
                    grad_var,
                    base_lr=getattr(optimizer, "flip_prob", 0.1),
                )

            # Consider dimension reduction
            if getattr(config, "use_adaptive_dimensions", False) and iteration > 0:
                progress_ratio = iteration / config.max_iterations
                early_reduction_threshold = 1.0 / math.log2(data_size + vocab_size)

                if progress_ratio > early_reduction_threshold and loss_trend in [
                    "plateau",
                    "decreasing",
                ]:
                    if hasattr(model, "_update_embedding_with_reduced_dim"):
                        try:
                            # Find optimal compression
                            optimal_ratio = adaptive.find_optimal_compression_ratio(
                                sample_vectors
                            )

                            # Determine compression threshold
                            loss_improvement = 1.0 - (avg_loss / config.loss_history[0])
                            compression_threshold = (
                                1.0 - loss_improvement * memory_scale
                            )

                            # Only compress if beneficial and above minimum
                            if (
                                optimal_ratio < compression_threshold
                                and optimal_ratio > min_ratio
                            ):
                                info_density = adaptive.compute_information_density(
                                    sample_vectors
                                )
                                window_sizes = adaptive.calculate_adaptive_window_sizes(
                                    info_density, dim, optimal_ratio
                                )
                                old_dim = dim
                                model._update_embedding_with_reduced_dim(window_sizes)
                                dim = config.dimension  # Update after reduction

                                # Update sample vectors
                                sample_vectors = []
                                if sample_size > 0:
                                    with torch.no_grad():
                                        sample_indices = torch.randperm(
                                            min(
                                                vocab_size,
                                                model.embedding.weight.size(0),
                                            )
                                        )[:sample_size]
                                        sample_vectors = [
                                            Bipolar(model.embedding.weight[idx])
                                            for idx in sample_indices
                                        ]

                                logger.info(f"Reduced dimensions: {old_dim} → {dim}")
                        except Exception as e:
                            logger.warning(f"Dimension reduction failed: {e}")

        # Convergence check
        if len(config.loss_history) > analysis_interval:
            window_size = min(
                len(config.loss_history),
                max(3, int(math.log2(iteration + 2) * analysis_interval / 2)),
            )

            if adaptive.detect_convergence(
                config.loss_history,
                window_size=window_size,
                snr_threshold=conv_threshold,
                min_iterations=analysis_interval,
            ):
                config.converged = True
                break

        # Reporting
        if (
            iteration == 0
            or (iteration + 1) % report_interval == 0
            or iteration + 1 == config.max_iterations
        ):
            if len(config.loss_history) > 1:
                trend_window = min(
                    len(config.loss_history),
                    int(math.log2(len(config.loss_history) + 1)),
                )
                trend = adaptive.analyze_loss_trend(config.loss_history[-trend_window:])
                logger.info(
                    f"Iter {iteration + 1}/{config.max_iterations}, Loss: {avg_loss:.6f}, Trend: {trend}"
                )
            else:
                logger.info(
                    f"Iter {iteration + 1}/{config.max_iterations}, Loss: {avg_loss:.6f}"
                )

    return config


def sonar_experiment(filepath="sonar.csv", report=True):
    """Run experiment classifying sonar data with bipolar networks.

    Args:
        filepath: Path to sonar dataset CSV
        report: Whether to print results

    Returns:
        tuple: (best_model, accuracy, parameter_count)
    """
    try:
        # Load data
        data = pd.read_csv(filepath, header=None)
        X = data.iloc[:, 0:60]
        y = data.iloc[:, 60]

        # Binary encoding of labels
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        # Convert to PyTorch tensors
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        # Define architectures to compare
        architectures = {
            "wide": {"architecture": "wide"},
            "deep": {"architecture": "deep", "num_layers": 3, "hidden_size": 30},
        }

        # Compare and get best model
        best_model, accuracy = compare_bipolar_classifiers(
            X, y, architectures, report=report
        )

        # Calculate model size
        param_count = sum(p.numel() for p in best_model.parameters())
        memory_kb = param_count / 8 / 1024  # 8 bits per parameter

        if report:
            print(f"\nModel parameter count: {param_count:,}")
            print(f"Memory footprint: {memory_kb:.2f} KB")

        return best_model, accuracy, param_count

    except FileNotFoundError:
        print(f"Error: Dataset file '{filepath}' not found.")
        return None, 0.0, 0
    except Exception as e:
        print(f"Error in sonar experiment: {e}")
        return None, 0.0, 0


def vector_operations_demo(dimension=1024, device=None):
    """Demonstrate Bipolar vector operations.

    Args:
        dimension: Dimension of vectors to create
        device: Device to use (None for auto-detection)

    Returns:
        dict: Dictionary containing example vectors and results
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running Bipolar Vector Operations Demo on: {device}")

    # Create random vectors
    vec1 = Bipolar.random((dimension,), device=device)
    vec2 = Bipolar.random((dimension,), device=device)

    # Run demonstrations
    results = {}

    # Binding and unbinding
    results["vec1"] = vec1
    results["vec2"] = vec2

    # Binding (XOR)
    bound_vec = vec1 * vec2
    results["bound"] = bound_vec

    # Unbinding
    unbound_vec = bound_vec.unbind(vec2)
    results["unbound"] = unbound_vec

    # Distance and similarity
    results["distance"] = vec1.distance(vec2).item()
    results["similarity"] = vec1.similarity(vec2).item()
    results["recovery_distance"] = vec1.distance(unbound_vec).item()

    # Superposition
    alpha = 0.7
    superposed_vec = vec1.superposition(vec2, alpha=alpha)
    results["superposed"] = superposed_vec
    results["superposed_distance1"] = vec1.distance(superposed_vec).item()
    results["superposed_distance2"] = vec2.distance(superposed_vec).item()

    # Randomization
    flip_prob = 0.2
    randomized_vec = vec1.clone().randomize(flip_prob=flip_prob)
    results["randomized"] = randomized_vec
    results["randomized_distance"] = vec1.distance(randomized_vec).item()

    # Print results
    print("\nBipolar Vector Operations:")
    print("-" * 50)
    print(f"Vector 1 sample: {vec1[:5]}")
    print(f"Vector 2 sample: {vec2[:5]}")
    print(f"Bound Vector (vec1 * vec2) sample: {bound_vec[:5]}")
    print(f"Unbound Vector (bound_vec unbind vec2) sample: {unbound_vec[:5]}")
    print(f"Distance between vec1 and unbound_vec: {results['recovery_distance']:.4f}")

    print(f"\nDistance between vec1 and vec2: {results['distance']:.4f}")
    print(f"Similarity between vec1 and vec2: {results['similarity']:.4f}")

    print(
        f"\nSuperposed Vector ({alpha}*vec1 + {1 - alpha}*vec2) sample: {superposed_vec[:5]}"
    )
    print(
        f"Distance between vec1 and superposed_vec: {results['superposed_distance1']:.4f}"
    )
    print(
        f"Distance between vec2 and superposed_vec: {results['superposed_distance2']:.4f}"
    )

    print(
        f"\nRandomized Vector (vec1 with {flip_prob * 100}% flip probability) sample: {randomized_vec[:5]}"
    )
    print(
        f"Distance between vec1 and randomized_vec: {results['randomized_distance']:.4f}"
    )

    print("\nDevice and Shape Information:")
    print(f"Device: {vec1.device}")
    print(f"Shape: {vec1.shape}")
    print(f"Length: {len(vec1)}")

    return results


def transformer_demo(
    vocab_size=1000, dimension=128, heads=4, layers=2, seq_len=16, batch_size=2
):
    """Demonstrate BipolarTransformer instantiation and usage.

    Args:
        vocab_size: Size of vocabulary
        dimension: Model dimension
        heads: Number of attention heads
        layers: Number of transformer layers
        seq_len: Sequence length for demo input
        batch_size: Batch size for demo input

    Returns:
        tuple: (transformer, output, parameter_count)
    """
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Bipolar Transformer Demo on: {device}")

    # Create configuration
    config = BipolarTransformerConfig(
        vocab_size=vocab_size,
        dimension=dimension,
        heads=heads,
        num_layers=layers,
        max_seq_len=seq_len * 2,  # Allow for longer sequences
        use_bipolar_attention=True,
        ff_expansion=2,
    )

    # Create transformer
    transformer = BipolarTransformer(config).to(device)

    # Generate random input
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Forward pass
    with torch.no_grad():
        output = transformer(token_ids)

    # Calculate size
    param_count = sum(p.numel() for p in transformer.parameters())
    memory_kb = param_count / 8 / 1024  # 8 bits per parameter

    # Print results
    print("\nTransformer Configuration:")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Dimension: {dimension}")
    print(f"Attention Heads: {heads}")
    print(f"Layers: {layers}")

    print("\nInput Shape:", token_ids.shape)
    print("Output Shape:", output.shape)
    print(
        f"Binary Parameter Percentage: {transformer.get_binary_params_percent():.2f}%"
    )
    print(f"Parameter Count: {param_count:,}")
    print(f"Memory Footprint: {memory_kb:.2f} KB")

    return transformer, output, param_count


def progressive_distiller_demo(input_size=10, hidden_size=20, dataset_size=100):
    """Demonstrate Progressive HDC Distiller with a simple classification model.

    Args:
        input_size: Input dimension
        hidden_size: Hidden layer size
        dataset_size: Number of samples in dummy dataset

    Returns:
        tuple: (distiller, model, final_accuracy)
    """
    print("Running Progressive HDC Distiller Demo")

    # Create a simple classification model
    class DummyModel(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return torch.sigmoid(x)

    # Create a dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, input_size):
            self.size = size
            self.data = torch.randn(size, input_size)
            self.labels = torch.randint(0, 2, (size, 1)).float()

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # Initialize model and dataset
    model = DummyModel(input_size, hidden_size)
    dataset = DummyDataset(dataset_size, input_size)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Create distiller
    distiller = ProgressiveHDCDistiller(model, dataloader)

    # Implement evaluation and loss methods
    def _evaluate_model(self):
        """Evaluate model accuracy."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                outputs = self.model(inputs)
                predicted = (outputs > 0.5).float()
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / total_samples
        self.model.train()
        return accuracy

    def _compute_task_loss(self, batch):
        """Compute binary cross-entropy loss."""
        inputs, labels = batch
        outputs = self.model(inputs)
        criterion = nn.BCELoss()
        loss = criterion(outputs, labels)
        return loss

    # Bind methods to distiller instance
    distiller._evaluate_model = _evaluate_model.__get__(
        distiller, ProgressiveHDCDistiller
    )
    distiller._compute_task_loss = _compute_task_loss.__get__(
        distiller, ProgressiveHDCDistiller
    )

    # Initial accuracy
    initial_accuracy = distiller._evaluate_model()
    print(f"\nInitial model accuracy: {initial_accuracy:.4f}")

    # Run distillation process
    print("\nProgressive Distillation Steps:")
    print("-" * 50)

    final_accuracy = initial_accuracy
    for i in range(2):  # Transform up to 2 layers (matches our model)
        print(f"\nDistillation Step {i + 1}:")
        transformed = distiller.transform_next_layer()
        if transformed:
            current_accuracy = distiller._evaluate_model()
            final_accuracy = current_accuracy
            print("Transformed a layer to bipolar representation")
            print(f"  Current Accuracy: {current_accuracy:.4f}")
        else:
            print("  No more layers to transform")
            break

    # Report parameter savings
    total_params = sum(p.numel() for p in model.parameters())
    bipolar_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name in distiller.transformed_tensors
    )
    bipolar_percentage = bipolar_params / total_params * 100

    print("\nDistillation Results:")
    print(f"Total Parameters: {total_params}")
    print(f"Bipolar Parameters: {bipolar_params} ({bipolar_percentage:.2f}%)")
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Accuracy Change: {final_accuracy - initial_accuracy:.4f}")


def run_semantic_demo(
    text=None,
    dimension=512,
    min_freq=2,
    max_words=200,
    word_list=None,
    show_analogy=True,
    visualize=True,
):
    """Run a demonstration of the semantic modeling capabilities.

    Args:
        text: Text corpus (uses default example if None)
        dimension: Vector dimension for semantic model
        min_freq: Minimum word frequency
        max_words: Maximum vocabulary size
        word_list: Words to analyze and visualize (uses defaults if None)
        show_analogy: Whether to show analogy examples
        visualize: Whether to visualize semantic space

    Returns:
        tuple: (model, config)
    """
    if text is None:
        text = """Machine learning is a field of study in artificial intelligence concerned with the development of algorithms that can learn from and make decisions based on data. Deep learning is a subset of machine learning based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Machine learning algorithms build a mathematical model based on sample data to make predictions without being explicitly programmed to perform the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers.

        Neural networks were inspired by information processing and distributed communication nodes in biological systems. Neural networks have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic and analog. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unlimited width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions.
        """

    # Use device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create custom text processor
    text_processor = TextProcessor(
        TextProcessorConfig(
            tokenizer_type="whitespace",
            cleaning_operations=["lowercase", "remove_punctuation"],
            doc_separator=r"\.\s+|\n+",  # Split on periods or newlines
            min_freq=min_freq,
            max_words=max_words,
        )
    )

    # Create and train model
    config = AdaptiveConfig(
        max_iterations=100,
        device=device,
        dimension=dimension,
        use_adaptive_dimensions=True,
        adaptive_dim_ratio=0.5,
        min_freq=min_freq,
        max_words_limit=max_words,
    )

    model = AdaptiveBitSemantic(config).to(device)

    try:
        # Train model
        start_time = time.time()
        config = model.fit(text, text_processor=text_processor)
        training_time = time.time() - start_time

        # Report vocabulary size and training metrics
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Vocabulary size: {len(model.topics):,} words")
        print(f"Final dimension: {config.dimension:,}")
        print(f"Iterations: {config.iterations_completed:,}")

        # Use default word list if none provided
        if word_list is None:
            word_list = ["learning", "neural", "network", "deep", "data"]
            # Filter to words actually in vocabulary
            word_list = [w for w in word_list if w in model.topic_to_idx]

        # Show similar words
        if word_list:
            target_word = word_list[0]
            print(f"\nWords similar to '{target_word}':")
            similar = model.find_similar(target_word, n=5)
            for word, dist in similar:
                print(f"  {word}: {dist:.4f}")

        # Show analogies if requested and possible
        if show_analogy and len(word_list) >= 3:
            a, b, c = word_list[:3]
            if all(w in model.topic_to_idx for w in [a, b, c]):
                print(f"\nAnalogy - {a}:{b}::{c}:?")
                results = model.analogy(a, b, c, n=3)
                for word, dist in results:
                    print(f"  {word}: {dist:.4f}")

        # Visualize semantic space if requested
        if visualize and word_list:
            vis = model.visualize_semantic_space(
                words=word_list, filename="semantic_space.txt", n_words=len(word_list)
            )
            print(f"\nSemantic Space Visualization:\n{'-' * 30}")
            print(vis.split("\n\n")[0])  # Show just the bit pattern part

        return model, config

    except Exception as e:
        print(f"Error in semantic demo: {e}")
        return None, config


def run_all_demos():
    """Run all available demonstrations.

    Returns:
        dict: Dictionary with results from each demo
    """
    results = {}

    print("=" * 60)
    print("BIPOLAR COMPUTING DEMONSTRATIONS")
    print("=" * 60)

    # 1. Vector Operations
    print("\n1. BIPOLAR VECTOR OPERATIONS")
    print("-" * 60)
    try:
        results["vector_ops"] = vector_operations_demo(dimension=512)
    except Exception as e:
        print(f"Vector operations demo failed: {e}")

    # 2. Semantic Model
    print("\n\n2. SEMANTIC MODELING")
    print("-" * 60)
    try:
        results["semantic"] = run_semantic_demo(dimension=512, max_words=100)
    except Exception as e:
        print(f"Semantic modeling demo failed: {e}")

    # 3. Transformer
    print("\n\n3. BIPOLAR TRANSFORMER")
    print("-" * 60)
    try:
        results["transformer"] = transformer_demo(
            vocab_size=500, dimension=64, heads=2, layers=1
        )
    except Exception as e:
        print(f"Transformer demo failed: {e}")

    # 4. Sonar Classification (only if file exists)
    if os.path.exists("sonar.csv"):
        print("\n\n4. SONAR CLASSIFICATION")
        print("-" * 60)
        try:
            results["sonar"] = sonar_experiment(report=True)
        except Exception as e:
            print(f"Sonar experiment failed: {e}")

    # 5. Progressive Distillation
    print("\n\n5. PROGRESSIVE DISTILLATION")
    print("-" * 60)
    try:
        results["distiller"] = progressive_distiller_demo()
    except Exception as e:
        print(f"Progressive distillation demo failed: {e}")

    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("=" * 60)

    return results


def main():
    """Main function for command-line execution.

    Usage: python bitfinder.py [mode]

    Modes:
        all - Run all demos
        semantic - Run semantic modeling demo
        vector - Run vector operations demo
        transformer - Run transformer demo
        sonar - Run sonar classification (requires sonar.csv)
        distiller - Run progressive distillation demo
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Bipolar Computing Examples")
    parser.add_argument(
        "mode",
        nargs="?",  # Make argument optional
        choices=["all", "semantic", "vector", "transformer", "sonar", "distiller"],
        default="semantic",  # Default to semantic demo
        help="Demo mode to run",
    )
    args = parser.parse_args()

    # Run selected demo
    if args.mode == "all":
        run_all_demos()
    elif args.mode == "semantic":
        run_semantic_demo()
    elif args.mode == "vector":
        vector_operations_demo()
    elif args.mode == "transformer":
        transformer_demo()
    elif args.mode == "sonar":
        if os.path.exists("sonar.csv"):
            sonar_experiment()
        else:
            print("Error: sonar.csv file not found.")
            print("Download the dataset or run a different demo.")
    elif args.mode == "distiller":
        progressive_distiller_demo()


if __name__ == "__main__":
    main()
