"""
Benchmarking utilities for ML experiments.

This module provides comprehensive benchmarking, quantization, and comparison tools:
- benchmark: Latency, throughput, and memory profiling
- quantize: Model quantization and optimization with bitsandbytes and torch.compile
- compare: Side-by-side model comparison with Rich tables
"""

from ml_experiments.benchmarks.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    LatencyStats,
    MemoryStats,
    ModelBenchmark,
    ThroughputStats,
    benchmark_context,
    quick_benchmark,
)
from ml_experiments.benchmarks.compare import (
    ComparisonConfig,
    ModelComparison,
    ModelMetrics,
    compare_models,
)
from ml_experiments.benchmarks.quantize import (
    CompileConfig,
    CompileMode,
    ModelOptimizer,
    OptimizationResult,
    QuantizationComparison,
    QuantizationConfig,
    QuantizationType,
    compile_model,
    optimize_model,
    quantize_model,
)

__all__ = [
    # Benchmark
    "BenchmarkConfig",
    "BenchmarkResult",
    "LatencyStats",
    "MemoryStats",
    "ModelBenchmark",
    "ThroughputStats",
    "benchmark_context",
    "quick_benchmark",
    # Compare
    "ComparisonConfig",
    "ModelComparison",
    "ModelMetrics",
    "compare_models",
    # Quantize
    "CompileConfig",
    "CompileMode",
    "ModelOptimizer",
    "OptimizationResult",
    "QuantizationComparison",
    "QuantizationConfig",
    "QuantizationType",
    "compile_model",
    "optimize_model",
    "quantize_model",
]
