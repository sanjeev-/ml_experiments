"""Metrics and profiling utilities for ML experiments."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from cleanfid import fid
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from clip_score import clip_score
    CLIP_SCORE_AVAILABLE = True
except ImportError:
    CLIP_SCORE_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console()


class MetricResult(BaseModel):
    """Result from a metric calculation."""

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LatencyStats(BaseModel):
    """Statistics for latency measurements."""

    mean: float = Field(..., description="Mean latency")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum latency")
    max: float = Field(..., description="Maximum latency")
    p50: float = Field(..., description="50th percentile (median)")
    p95: float = Field(..., description="95th percentile")
    p99: float = Field(..., description="99th percentile")
    unit: str = Field(default="ms", description="Unit of measurement")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


class MemoryStats(BaseModel):
    """GPU memory statistics."""

    allocated_gb: float = Field(..., description="Allocated memory in GB")
    reserved_gb: float = Field(..., description="Reserved memory in GB")
    max_allocated_gb: float = Field(..., description="Max allocated memory in GB")
    max_reserved_gb: float = Field(..., description="Max reserved memory in GB")
    free_gb: float = Field(..., description="Free memory in GB")
    total_gb: float = Field(..., description="Total memory in GB")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
            "max_allocated_gb": self.max_allocated_gb,
            "max_reserved_gb": self.max_reserved_gb,
            "free_gb": self.free_gb,
            "total_gb": self.total_gb,
        }


# ============================================================================
# Image Quality Metrics
# ============================================================================


def calculate_fid(
    real_images_path: Union[str, Path],
    generated_images_path: Union[str, Path],
    batch_size: int = 50,
    device: str = "cuda",
    num_workers: int = 4,
) -> MetricResult:
    """Calculate Frechet Inception Distance (FID) score.

    Args:
        real_images_path: Path to real images directory
        generated_images_path: Path to generated images directory
        batch_size: Batch size for feature extraction
        device: Device to use
        num_workers: Number of workers for data loading

    Returns:
        FID metric result
    """
    logger.info("Calculating FID score...")
    start_time = time.time()

    try:
        fid_score = fid.compute_fid(
            str(real_images_path),
            str(generated_images_path),
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )

        elapsed = time.time() - start_time
        logger.info(f"FID calculation completed in {elapsed:.2f}s: {fid_score:.4f}")

        return MetricResult(
            name="fid",
            value=float(fid_score),
            unit="score",
            metadata={"computation_time": elapsed}
        )

    except Exception as e:
        logger.error(f"Error calculating FID: {e}")
        raise


def calculate_lpips(
    images1: torch.Tensor,
    images2: torch.Tensor,
    net: str = "alex",
    device: str = "cuda",
) -> MetricResult:
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity) metric.

    Args:
        images1: First set of images [B, C, H, W], values in [-1, 1]
        images2: Second set of images [B, C, H, W], values in [-1, 1]
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device to use

    Returns:
        LPIPS metric result
    """
    if not LPIPS_AVAILABLE:
        raise ImportError("lpips not available. Install with: pip install lpips")

    logger.info("Calculating LPIPS...")
    start_time = time.time()

    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=net).to(device)

    with torch.no_grad():
        lpips_values = loss_fn(images1.to(device), images2.to(device))
        mean_lpips = lpips_values.mean().item()

    elapsed = time.time() - start_time
    logger.info(f"LPIPS calculation completed in {elapsed:.2f}s: {mean_lpips:.4f}")

    return MetricResult(
        name="lpips",
        value=float(mean_lpips),
        unit="distance",
        metadata={
            "network": net,
            "num_images": len(images1),
            "computation_time": elapsed
        }
    )


def calculate_clip_score(
    images: Union[List[str], torch.Tensor],
    prompts: List[str],
    device: str = "cuda",
) -> MetricResult:
    """Calculate CLIP score between images and text prompts.

    Args:
        images: List of image paths or tensor of images
        prompts: List of text prompts
        device: Device to use

    Returns:
        CLIP score metric result
    """
    if not CLIP_SCORE_AVAILABLE:
        logger.warning("clip_score not available. Returning placeholder.")
        return MetricResult(name="clip_score", value=0.0, unit="score")

    logger.info("Calculating CLIP score...")
    start_time = time.time()

    try:
        score = clip_score(images, prompts, device=device)
        elapsed = time.time() - start_time
        logger.info(f"CLIP score calculation completed in {elapsed:.2f}s: {score:.4f}")

        return MetricResult(
            name="clip_score",
            value=float(score),
            unit="score",
            metadata={
                "num_images": len(images),
                "computation_time": elapsed
            }
        )

    except Exception as e:
        logger.error(f"Error calculating CLIP score: {e}")
        raise


# ============================================================================
# Performance Profiling
# ============================================================================


class LatencyProfiler:
    """Profile latency of function calls with percentile statistics."""

    def __init__(self, name: str = "profiler"):
        """Initialize latency profiler.

        Args:
            name: Profiler name
        """
        self.name = name
        self.measurements: List[float] = []
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timing and record measurement.

        Returns:
            Elapsed time in milliseconds
        """
        if self._start_time is None:
            raise RuntimeError("Profiler not started. Call start() first.")

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self.measurements.append(elapsed_ms)
        self._start_time = None
        return elapsed_ms

    @contextmanager
    def measure(self):
        """Context manager for measuring latency."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def get_stats(self) -> LatencyStats:
        """Get latency statistics.

        Returns:
            LatencyStats with percentiles
        """
        if not self.measurements:
            raise ValueError("No measurements recorded")

        measurements = np.array(self.measurements)
        return LatencyStats(
            mean=float(np.mean(measurements)),
            std=float(np.std(measurements)),
            min=float(np.min(measurements)),
            max=float(np.max(measurements)),
            p50=float(np.percentile(measurements, 50)),
            p95=float(np.percentile(measurements, 95)),
            p99=float(np.percentile(measurements, 99)),
            unit="ms"
        )

    def reset(self) -> None:
        """Reset measurements."""
        self.measurements.clear()

    def print_stats(self) -> None:
        """Print statistics as a rich table."""
        stats = self.get_stats()

        table = Table(title=f"Latency Statistics: {self.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Mean", f"{stats.mean:.2f} ms")
        table.add_row("Std Dev", f"{stats.std:.2f} ms")
        table.add_row("Min", f"{stats.min:.2f} ms")
        table.add_row("Max", f"{stats.max:.2f} ms")
        table.add_row("P50 (Median)", f"{stats.p50:.2f} ms")
        table.add_row("P95", f"{stats.p95:.2f} ms")
        table.add_row("P99", f"{stats.p99:.2f} ms")
        table.add_row("Samples", str(len(self.measurements)))

        console.print(table)


def measure_latency(
    fn: Callable,
    num_runs: int = 100,
    warmup_runs: int = 10,
    *args,
    **kwargs
) -> LatencyStats:
    """Measure latency of a function with warmup.

    Args:
        fn: Function to measure
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        LatencyStats with percentiles
    """
    logger.info(f"Measuring latency: {warmup_runs} warmup + {num_runs} runs")

    # Warmup
    for _ in range(warmup_runs):
        fn(*args, **kwargs)

    # Synchronize CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    profiler = LatencyProfiler(fn.__name__)
    for _ in range(num_runs):
        with profiler.measure():
            fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    return profiler.get_stats()


def measure_throughput(
    fn: Callable,
    batch_size: int,
    num_runs: int = 100,
    warmup_runs: int = 10,
    *args,
    **kwargs
) -> float:
    """Measure throughput (items/second).

    Args:
        fn: Function to measure
        batch_size: Batch size
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Throughput in items/second
    """
    stats = measure_latency(fn, num_runs, warmup_runs, *args, **kwargs)
    throughput = (batch_size * 1000) / stats.mean  # Convert ms to seconds
    logger.info(f"Throughput: {throughput:.2f} items/sec (batch_size={batch_size})")
    return throughput


# ============================================================================
# GPU Memory Profiling
# ============================================================================


def get_gpu_memory_stats(device: Union[int, str] = 0) -> MemoryStats:
    """Get current GPU memory statistics.

    Args:
        device: GPU device index or 'cuda'

    Returns:
        MemoryStats with current memory usage
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if isinstance(device, str):
        device = 0

    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    max_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    # Get total and free memory
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    free = total - reserved

    return MemoryStats(
        allocated_gb=allocated,
        reserved_gb=reserved,
        max_allocated_gb=max_allocated,
        max_reserved_gb=max_reserved,
        free_gb=free,
        total_gb=total,
    )


def print_gpu_memory_stats(device: Union[int, str] = 0) -> None:
    """Print GPU memory statistics as a rich table.

    Args:
        device: GPU device index or 'cuda'
    """
    stats = get_gpu_memory_stats(device)

    table = Table(title=f"GPU Memory Statistics (Device {device})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Allocated", f"{stats.allocated_gb:.2f} GB")
    table.add_row("Reserved", f"{stats.reserved_gb:.2f} GB")
    table.add_row("Max Allocated", f"{stats.max_allocated_gb:.2f} GB")
    table.add_row("Max Reserved", f"{stats.max_reserved_gb:.2f} GB")
    table.add_row("Free", f"{stats.free_gb:.2f} GB")
    table.add_row("Total", f"{stats.total_gb:.2f} GB")

    utilization = (stats.allocated_gb / stats.total_gb) * 100
    table.add_row("Utilization", f"{utilization:.1f}%")

    console.print(table)


@contextmanager
def profile_memory(device: Union[int, str] = 0, reset: bool = True):
    """Context manager for profiling GPU memory usage.

    Args:
        device: GPU device index or 'cuda'
        reset: Whether to reset memory stats before profiling

    Yields:
        Dictionary with 'before', 'after', and 'peak' MemoryStats
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if isinstance(device, str):
        device = 0

    if reset:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    before = get_gpu_memory_stats(device)

    result = {"before": before}

    try:
        yield result
    finally:
        after = get_gpu_memory_stats(device)
        result["after"] = after
        result["peak"] = MemoryStats(
            allocated_gb=after.max_allocated_gb,
            reserved_gb=after.max_reserved_gb,
            max_allocated_gb=after.max_allocated_gb,
            max_reserved_gb=after.max_reserved_gb,
            free_gb=after.free_gb,
            total_gb=after.total_gb,
        )

        # Print summary
        delta = after.allocated_gb - before.allocated_gb
        console.print(f"\n[bold]Memory Usage Summary:[/bold]")
        console.print(f"  Before: {before.allocated_gb:.2f} GB")
        console.print(f"  After: {after.allocated_gb:.2f} GB")
        console.print(f"  Delta: {delta:+.2f} GB")
        console.print(f"  Peak: {after.max_allocated_gb:.2f} GB")


class MemoryProfiler:
    """Profile memory usage of model forward passes."""

    def __init__(self, model: nn.Module, device: Union[int, str] = 0):
        """Initialize memory profiler.

        Args:
            model: PyTorch model to profile
            device: GPU device index or 'cuda'
        """
        self.model = model
        self.device = device if isinstance(device, int) else 0
        self.measurements: List[Tuple[int, float]] = []  # (batch_size, memory_gb)

    def profile_batch_sizes(
        self,
        batch_sizes: List[int],
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Dict[int, float]:
        """Profile memory usage for different batch sizes.

        Args:
            batch_sizes: List of batch sizes to test
            input_shape: Input shape without batch dimension (C, H, W)
            dtype: Input data type

        Returns:
            Dictionary mapping batch_size to memory in GB
        """
        results = {}

        for batch_size in batch_sizes:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

            try:
                # Create dummy input
                dummy_input = torch.randn(
                    batch_size, *input_shape,
                    dtype=dtype,
                    device=f"cuda:{self.device}"
                )

                # Forward pass
                with torch.no_grad():
                    _ = self.model(dummy_input)

                # Get peak memory
                peak_memory = torch.cuda.max_memory_allocated(self.device) / 1e9
                results[batch_size] = peak_memory
                self.measurements.append((batch_size, peak_memory))

                logger.info(f"Batch size {batch_size}: {peak_memory:.2f} GB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at batch size {batch_size}")
                    results[batch_size] = float('inf')
                else:
                    raise

            finally:
                del dummy_input
                torch.cuda.empty_cache()

        return results

    def print_results(self) -> None:
        """Print profiling results as a rich table."""
        table = Table(title="Memory Profiling Results")
        table.add_column("Batch Size", style="cyan")
        table.add_column("Peak Memory", style="magenta")

        for batch_size, memory in sorted(self.measurements):
            if memory == float('inf'):
                table.add_row(str(batch_size), "OOM")
            else:
                table.add_row(str(batch_size), f"{memory:.2f} GB")

        console.print(table)


# ============================================================================
# Integration with Experiment class
# ============================================================================


class MetricsTracker:
    """Track and aggregate metrics for experiments."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Any] = {}

    def add_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Add a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def add_metric_result(self, result: MetricResult, step: Optional[int] = None) -> None:
        """Add a MetricResult.

        Args:
            result: MetricResult to add
            step: Optional step number
        """
        self.add_metric(result.name, result.value, step)
        if result.metadata:
            self.metadata[result.name] = result.metadata

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics.

        Returns:
            Dictionary of metric statistics
        """
        summary = {}
        for name, values in self.metrics.items():
            values_array = np.array(values)
            summary[name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "count": len(values),
            }
        return summary

    def print_summary(self) -> None:
        """Print metrics summary as a rich table."""
        summary = self.get_summary()

        table = Table(title="Metrics Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="magenta")
        table.add_column("Std", style="magenta")
        table.add_column("Min", style="green")
        table.add_column("Max", style="red")
        table.add_column("Count", style="yellow")

        for name, stats in summary.items():
            table.add_row(
                name,
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                str(stats['count'])
            )

        console.print(table)
