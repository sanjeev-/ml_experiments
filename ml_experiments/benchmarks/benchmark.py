"""
Benchmarking utilities for measuring latency, throughput, and memory usage.

This module provides comprehensive benchmarking tools for ML models, including:
- Latency measurement with statistical analysis (mean, std, percentiles)
- Throughput calculation (samples/second, tokens/second)
- GPU memory profiling
- Warmup runs and configurable benchmark parameters
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        num_warmup: Number of warmup iterations before measurement
        num_iterations: Number of measurement iterations
        measure_memory: Whether to measure GPU memory usage
        measure_throughput: Whether to calculate throughput metrics
        sync_cuda: Whether to synchronize CUDA before timing (more accurate but slower)
        clear_cache: Whether to clear CUDA cache between iterations
        batch_sizes: List of batch sizes to test (if applicable)
    """
    num_warmup: int = 5
    num_iterations: int = 100
    measure_memory: bool = True
    measure_throughput: bool = True
    sync_cuda: bool = True
    clear_cache: bool = False
    batch_sizes: Optional[List[int]] = None


class LatencyStats(BaseModel):
    """Statistics for latency measurements."""

    mean: float = Field(..., description="Mean latency in milliseconds")
    std: float = Field(..., description="Standard deviation in milliseconds")
    min: float = Field(..., description="Minimum latency in milliseconds")
    max: float = Field(..., description="Maximum latency in milliseconds")
    median: float = Field(..., description="Median latency in milliseconds")
    p50: float = Field(..., description="50th percentile in milliseconds")
    p95: float = Field(..., description="95th percentile in milliseconds")
    p99: float = Field(..., description="99th percentile in milliseconds")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_ms": self.mean,
            "std_ms": self.std,
            "min_ms": self.min,
            "max_ms": self.max,
            "median_ms": self.median,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
        }


class MemoryStats(BaseModel):
    """GPU memory statistics."""

    allocated_mb: float = Field(..., description="Allocated memory in MB")
    reserved_mb: float = Field(..., description="Reserved memory in MB")
    max_allocated_mb: float = Field(..., description="Max allocated memory in MB")
    max_reserved_mb: float = Field(..., description="Max reserved memory in MB")
    peak_mb: float = Field(..., description="Peak memory usage in MB")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "max_allocated_mb": self.max_allocated_mb,
            "max_reserved_mb": self.max_reserved_mb,
            "peak_mb": self.peak_mb,
        }


class ThroughputStats(BaseModel):
    """Throughput statistics."""

    samples_per_second: float = Field(..., description="Samples processed per second")
    tokens_per_second: Optional[float] = Field(
        default=None, description="Tokens processed per second"
    )
    batches_per_second: Optional[float] = Field(
        default=None, description="Batches processed per second"
    )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {"samples_per_second": self.samples_per_second}
        if self.tokens_per_second is not None:
            result["tokens_per_second"] = self.tokens_per_second
        if self.batches_per_second is not None:
            result["batches_per_second"] = self.batches_per_second
        return result


class BenchmarkResult(BaseModel):
    """Complete benchmark results."""

    latency: LatencyStats
    memory: Optional[MemoryStats] = None
    throughput: Optional[ThroughputStats] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"latency": self.latency.to_dict()}
        if self.memory is not None:
            result["memory"] = self.memory.to_dict()
        if self.throughput is not None:
            result["throughput"] = self.throughput.to_dict()
        result["metadata"] = self.metadata
        return result

    def print_summary(self) -> None:
        """Print a formatted summary table."""
        table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        # Latency metrics
        table.add_row("Mean Latency", f"{self.latency.mean:.2f} ms")
        table.add_row("Std Dev", f"{self.latency.std:.2f} ms")
        table.add_row("Min Latency", f"{self.latency.min:.2f} ms")
        table.add_row("Max Latency", f"{self.latency.max:.2f} ms")
        table.add_row("Median (P50)", f"{self.latency.p50:.2f} ms")
        table.add_row("P95", f"{self.latency.p95:.2f} ms")
        table.add_row("P99", f"{self.latency.p99:.2f} ms")

        # Memory metrics
        if self.memory is not None:
            table.add_section()
            table.add_row("Allocated Memory", f"{self.memory.allocated_mb:.2f} MB")
            table.add_row("Peak Memory", f"{self.memory.peak_mb:.2f} MB")
            table.add_row("Max Allocated", f"{self.memory.max_allocated_mb:.2f} MB")

        # Throughput metrics
        if self.throughput is not None:
            table.add_section()
            table.add_row("Samples/Second", f"{self.throughput.samples_per_second:.2f}")
            if self.throughput.tokens_per_second is not None:
                table.add_row("Tokens/Second", f"{self.throughput.tokens_per_second:.2f}")

        console.print(table)


class ModelBenchmark:
    """Benchmarking utility for ML models.

    This class provides comprehensive benchmarking capabilities including:
    - Warmup runs to stabilize performance
    - Statistical latency analysis with percentiles
    - GPU memory profiling
    - Throughput calculation
    - Multiple batch size testing

    Example:
        >>> benchmark = ModelBenchmark(config=BenchmarkConfig(num_iterations=100))
        >>> result = benchmark.run(model_fn, inputs)
        >>> result.print_summary()
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark with configuration.

        Args:
            config: Benchmark configuration. If None, uses defaults.
        """
        self.config = config or BenchmarkConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _warmup(self, fn: Callable, *args, **kwargs) -> None:
        """Run warmup iterations.

        Args:
            fn: Function to benchmark
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn
        """
        logger.info(f"Running {self.config.num_warmup} warmup iterations...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Warming up...", total=self.config.num_warmup)
            for _ in range(self.config.num_warmup):
                with torch.no_grad():
                    fn(*args, **kwargs)
                if self.config.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                progress.advance(task)

    def _measure_latency(self, fn: Callable, *args, **kwargs) -> List[float]:
        """Measure latency over multiple iterations.

        Args:
            fn: Function to benchmark
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            List of latency measurements in milliseconds
        """
        latencies = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Measuring latency ({self.config.num_iterations} iterations)...",
                total=self.config.num_iterations,
            )

            for _ in range(self.config.num_iterations):
                if self.config.clear_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Measure time
                if self.config.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                with torch.no_grad():
                    fn(*args, **kwargs)

                if self.config.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()

                # Convert to milliseconds
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                progress.advance(task)

        return latencies

    def _calculate_latency_stats(self, latencies: List[float]) -> LatencyStats:
        """Calculate statistical metrics from latency measurements.

        Args:
            latencies: List of latency measurements in milliseconds

        Returns:
            LatencyStats object with statistical analysis
        """
        latencies_array = np.array(latencies)

        return LatencyStats(
            mean=float(np.mean(latencies_array)),
            std=float(np.std(latencies_array)),
            min=float(np.min(latencies_array)),
            max=float(np.max(latencies_array)),
            median=float(np.median(latencies_array)),
            p50=float(np.percentile(latencies_array, 50)),
            p95=float(np.percentile(latencies_array, 95)),
            p99=float(np.percentile(latencies_array, 99)),
        )

    def _measure_memory(self) -> Optional[MemoryStats]:
        """Measure GPU memory usage.

        Returns:
            MemoryStats object or None if CUDA is not available
        """
        if not torch.cuda.is_available():
            return None

        # Get memory stats in bytes, convert to MB
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2

        return MemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            max_reserved_mb=max_reserved,
            peak_mb=max(max_allocated, max_reserved),
        )

    def _calculate_throughput(
        self,
        latency_stats: LatencyStats,
        batch_size: int = 1,
        num_tokens: Optional[int] = None,
    ) -> ThroughputStats:
        """Calculate throughput metrics.

        Args:
            latency_stats: Latency statistics
            batch_size: Batch size used for inference
            num_tokens: Number of tokens processed (for text models)

        Returns:
            ThroughputStats object
        """
        # Use mean latency for throughput calculation
        latency_seconds = latency_stats.mean / 1000.0
        samples_per_second = batch_size / latency_seconds

        tokens_per_second = None
        if num_tokens is not None:
            tokens_per_second = (batch_size * num_tokens) / latency_seconds

        batches_per_second = 1.0 / latency_seconds

        return ThroughputStats(
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            batches_per_second=batches_per_second,
        )

    def run(
        self,
        fn: Callable,
        *args,
        batch_size: int = 1,
        num_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Run complete benchmark suite.

        Args:
            fn: Function to benchmark (should accept *args, **kwargs)
            *args: Positional arguments for fn
            batch_size: Batch size for throughput calculation
            num_tokens: Number of tokens (for text models)
            metadata: Additional metadata to include in results
            **kwargs: Keyword arguments for fn

        Returns:
            BenchmarkResult with complete statistics
        """
        console.print("\n[bold cyan]Starting Benchmark[/bold cyan]")

        # Reset memory stats if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        if self.config.num_warmup > 0:
            self._warmup(fn, *args, **kwargs)

        # Measure latency
        latencies = self._measure_latency(fn, *args, **kwargs)
        latency_stats = self._calculate_latency_stats(latencies)

        # Measure memory
        memory_stats = None
        if self.config.measure_memory:
            memory_stats = self._measure_memory()

        # Calculate throughput
        throughput_stats = None
        if self.config.measure_throughput:
            throughput_stats = self._calculate_throughput(
                latency_stats, batch_size=batch_size, num_tokens=num_tokens
            )

        # Compile results
        result = BenchmarkResult(
            latency=latency_stats,
            memory=memory_stats,
            throughput=throughput_stats,
            metadata=metadata or {},
        )

        console.print("\n[bold green]Benchmark Complete![/bold green]")
        return result

    def run_multi_batch(
        self,
        fn: Callable,
        batch_sizes: Optional[List[int]] = None,
        input_generator: Optional[Callable[[int], Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, BenchmarkResult]:
        """Run benchmarks across multiple batch sizes.

        Args:
            fn: Function to benchmark
            batch_sizes: List of batch sizes to test. If None, uses config.batch_sizes
            input_generator: Function that generates inputs for a given batch size
            metadata: Additional metadata to include in results

        Returns:
            Dictionary mapping batch_size -> BenchmarkResult
        """
        batch_sizes = batch_sizes or self.config.batch_sizes or [1, 2, 4, 8]
        results = {}

        console.print(
            f"\n[bold cyan]Running Multi-Batch Benchmark[/bold cyan] "
            f"(batch sizes: {batch_sizes})"
        )

        for batch_size in batch_sizes:
            console.print(f"\n[yellow]Batch Size: {batch_size}[/yellow]")

            # Generate inputs for this batch size if generator provided
            if input_generator is not None:
                inputs = input_generator(batch_size)
                if isinstance(inputs, tuple):
                    args = inputs
                    kwargs = {}
                elif isinstance(inputs, dict):
                    args = ()
                    kwargs = inputs
                else:
                    args = (inputs,)
                    kwargs = {}
            else:
                args = ()
                kwargs = {}

            # Run benchmark for this batch size
            result = self.run(
                fn,
                *args,
                batch_size=batch_size,
                metadata={**(metadata or {}), "batch_size": batch_size},
                **kwargs,
            )
            results[batch_size] = result

        # Print comparison table
        self._print_multi_batch_summary(results)

        return results

    def _print_multi_batch_summary(self, results: Dict[int, BenchmarkResult]) -> None:
        """Print a comparison table for multi-batch results.

        Args:
            results: Dictionary mapping batch_size -> BenchmarkResult
        """
        table = Table(
            title="Multi-Batch Benchmark Summary",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Batch Size", style="cyan")
        table.add_column("Mean Latency (ms)", justify="right", style="green")
        table.add_column("P95 (ms)", justify="right", style="yellow")
        table.add_column("Throughput (samples/s)", justify="right", style="blue")
        table.add_column("Peak Memory (MB)", justify="right", style="red")

        for batch_size in sorted(results.keys()):
            result = results[batch_size]

            throughput = (
                f"{result.throughput.samples_per_second:.2f}"
                if result.throughput
                else "N/A"
            )
            memory = f"{result.memory.peak_mb:.2f}" if result.memory else "N/A"

            table.add_row(
                str(batch_size),
                f"{result.latency.mean:.2f}",
                f"{result.latency.p95:.2f}",
                throughput,
                memory,
            )

        console.print("\n")
        console.print(table)


@contextmanager
def benchmark_context(name: str = "Operation", verbose: bool = True):
    """Context manager for quick benchmarking of code blocks.

    Example:
        >>> with benchmark_context("Model inference"):
        ...     output = model(input)

    Args:
        name: Name of the operation being benchmarked
        verbose: Whether to print results

    Yields:
        Dictionary that will be populated with timing results
    """
    result = {}

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        elapsed_ms = (end_time - start_time) * 1000
        memory_used_mb = (end_memory - start_memory) / 1024**2

        result["elapsed_ms"] = elapsed_ms
        result["memory_used_mb"] = memory_used_mb

        if verbose:
            console.print(
                f"[cyan]{name}[/cyan]: "
                f"[green]{elapsed_ms:.2f} ms[/green]"
                + (
                    f", [red]{memory_used_mb:.2f} MB[/red]"
                    if torch.cuda.is_available()
                    else ""
                )
            )


def quick_benchmark(fn: Callable, *args, num_iterations: int = 10, **kwargs) -> Dict[str, float]:
    """Quick benchmark helper for simple use cases.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for fn
        num_iterations: Number of iterations (default: 10)
        **kwargs: Keyword arguments for fn

    Returns:
        Dictionary with mean, std, min, max latencies in milliseconds
    """
    benchmark = ModelBenchmark(
        config=BenchmarkConfig(
            num_warmup=min(3, num_iterations // 2),
            num_iterations=num_iterations,
            measure_memory=False,
            measure_throughput=False,
        )
    )

    result = benchmark.run(fn, *args, **kwargs)
    return {
        "mean_ms": result.latency.mean,
        "std_ms": result.latency.std,
        "min_ms": result.latency.min,
        "max_ms": result.latency.max,
        "p95_ms": result.latency.p95,
    }
