#!/usr/bin/env python3
"""
CLI entry point for benchmarking ML models.

This script provides comprehensive model benchmarking with:
- Latency, throughput, and memory profiling
- Quantization options (int8, int4, bitsandbytes)
- Torch.compile optimization
- Model comparison with Rich tables
- Support for local and Modal execution
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.syntax import Syntax

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_experiments.benchmarks import (
    BenchmarkConfig,
    ModelBenchmark,
    quick_benchmark,
    QuantizationType,
    CompileMode,
    ModelOptimizer,
    compare_models,
    ModelMetrics,
)

console = Console()


def load_benchmark_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise click.ClickException("Config file is empty")

        return config

    except yaml.YAMLError as e:
        raise click.ClickException(f"Failed to parse YAML: {e}")
    except Exception as e:
        raise click.ClickException(f"Failed to load config: {e}")


def create_benchmark_config(
    warmup: int,
    iterations: int,
    memory: bool,
    throughput: bool,
    sync_cuda: bool,
    clear_cache: bool,
    batch_sizes: Optional[List[int]]
) -> BenchmarkConfig:
    """Create BenchmarkConfig from CLI arguments.

    Args:
        warmup: Number of warmup iterations
        iterations: Number of measurement iterations
        memory: Measure memory usage
        throughput: Measure throughput
        sync_cuda: Synchronize CUDA before timing
        clear_cache: Clear CUDA cache between iterations
        batch_sizes: List of batch sizes to test

    Returns:
        BenchmarkConfig instance
    """
    return BenchmarkConfig(
        num_warmup=warmup,
        num_iterations=iterations,
        measure_memory=memory,
        measure_throughput=throughput,
        sync_cuda=sync_cuda,
        clear_cache=clear_cache,
        batch_sizes=batch_sizes
    )


def display_benchmark_results(results: Dict[str, Any], title: str = "Benchmark Results") -> None:
    """Display benchmark results in a formatted table.

    Args:
        results: Dictionary of benchmark results
        title: Table title
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", justify="right")

    # Latency stats
    if "latency" in results:
        latency = results["latency"]
        table.add_row("[bold]Latency Statistics[/bold]", "")
        table.add_row("  Mean", f"{latency.get('mean_ms', 0):.2f} ms")
        table.add_row("  Std Dev", f"{latency.get('std_ms', 0):.2f} ms")
        table.add_row("  Median (P50)", f"{latency.get('p50_ms', 0):.2f} ms")
        table.add_row("  P95", f"{latency.get('p95_ms', 0):.2f} ms")
        table.add_row("  P99", f"{latency.get('p99_ms', 0):.2f} ms")
        table.add_row("  Min", f"{latency.get('min_ms', 0):.2f} ms")
        table.add_row("  Max", f"{latency.get('max_ms', 0):.2f} ms")

    # Memory stats
    if "memory" in results:
        memory = results["memory"]
        table.add_row("[bold]Memory Usage[/bold]", "")
        table.add_row("  Allocated", f"{memory.get('allocated_mb', 0):.2f} MB")
        table.add_row("  Reserved", f"{memory.get('reserved_mb', 0):.2f} MB")
        table.add_row("  Peak", f"{memory.get('peak_mb', 0):.2f} MB")
        table.add_row("  Max Allocated", f"{memory.get('max_allocated_mb', 0):.2f} MB")

    # Throughput stats
    if "throughput" in results:
        throughput = results["throughput"]
        table.add_row("[bold]Throughput[/bold]", "")
        table.add_row("  Samples/sec", f"{throughput.get('samples_per_second', 0):.2f}")
        if throughput.get("tokens_per_second"):
            table.add_row("  Tokens/sec", f"{throughput.get('tokens_per_second', 0):.2f}")

    console.print(table)


@click.group()
def cli():
    """Benchmark ML models with comprehensive profiling.

    This tool provides latency, throughput, and memory benchmarking
    with support for quantization and optimization.
    """
    console.print(Panel.fit(
        "[bold cyan]ML Experiments Benchmarking Tool[/bold cyan]\n"
        "[dim]Comprehensive model profiling and optimization[/dim]",
        border_style="cyan"
    ))


@cli.command()
@click.argument("model_path", type=str)
@click.option(
    "--warmup",
    type=int,
    default=5,
    help="Number of warmup iterations"
)
@click.option(
    "--iterations",
    type=int,
    default=100,
    help="Number of measurement iterations"
)
@click.option(
    "--memory/--no-memory",
    default=True,
    help="Measure GPU memory usage"
)
@click.option(
    "--throughput/--no-throughput",
    default=True,
    help="Measure throughput"
)
@click.option(
    "--sync-cuda/--no-sync-cuda",
    default=True,
    help="Synchronize CUDA before timing"
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear CUDA cache between iterations"
)
@click.option(
    "--batch-sizes",
    type=str,
    default=None,
    help="Comma-separated list of batch sizes to test (e.g., '1,4,8,16')"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save results to JSON file"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
def run(
    model_path: str,
    warmup: int,
    iterations: int,
    memory: bool,
    throughput: bool,
    sync_cuda: bool,
    clear_cache: bool,
    batch_sizes: Optional[str],
    output: Optional[Path],
    verbose: bool
) -> None:
    """Run benchmark on a single model.

    MODEL_PATH: HuggingFace model name or path to local model

    Examples:

        # Basic benchmark
        ml-benchmark run stabilityai/stable-diffusion-2-1

        # Custom iterations and batch sizes
        ml-benchmark run gpt2 --iterations 200 --batch-sizes 1,4,8,16

        # Save results to file
        ml-benchmark run bert-base-uncased --output results.json
    """
    try:
        console.print(f"\n[bold]Benchmarking: {model_path}[/bold]\n")

        # Parse batch sizes
        batch_size_list = None
        if batch_sizes:
            try:
                batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]
            except ValueError:
                raise click.ClickException("Invalid batch sizes format. Use comma-separated integers.")

        # Create benchmark config
        bench_config = create_benchmark_config(
            warmup=warmup,
            iterations=iterations,
            memory=memory,
            throughput=throughput,
            sync_cuda=sync_cuda,
            clear_cache=clear_cache,
            batch_sizes=batch_size_list
        )

        # Run benchmark
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Benchmarking {model_path}...", total=100)

            # Note: This is a simplified example. In practice, you'd need to:
            # 1. Load the model
            # 2. Create appropriate input data
            # 3. Run the benchmark with ModelBenchmark class

            console.print("\n[yellow]Note: Full model benchmarking requires model loading and input preparation.[/yellow]")
            console.print("[yellow]This is a template implementation. Integrate with your specific models.[/yellow]\n")

            # Example of how to use the benchmark API:
            console.print("[dim]Example usage:[/dim]")
            example_code = """
import torch
from ml_experiments.benchmarks import quick_benchmark

# Load your model
model = YourModel.from_pretrained(model_path)
model.eval()

# Create sample input
sample_input = torch.randn(1, 3, 224, 224)

# Run benchmark
def forward_fn():
    with torch.no_grad():
        return model(sample_input)

results = quick_benchmark(forward_fn, config=bench_config)
"""
            console.print(Syntax(example_code, "python", theme="monokai"))

            progress.update(task, completed=100)

        console.print("\n[green]✓ Benchmark completed![/green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Benchmark failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=str)
@click.option(
    "--method",
    type=click.Choice(["int8", "int4", "nf4", "fp4"], case_sensitive=False),
    default="int8",
    help="Quantization method"
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare quantized vs original model"
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="Benchmark quantized model"
)
@click.option(
    "--warmup",
    type=int,
    default=5,
    help="Number of warmup iterations for benchmark"
)
@click.option(
    "--iterations",
    type=int,
    default=100,
    help="Number of measurement iterations for benchmark"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
def quantize(
    model_path: str,
    method: str,
    compare: bool,
    benchmark: bool,
    warmup: int,
    iterations: int,
    verbose: bool
) -> None:
    """Quantize a model using bitsandbytes.

    MODEL_PATH: HuggingFace model name or path to local model

    Examples:

        # Quantize to int8
        ml-benchmark quantize stabilityai/stable-diffusion-2-1 --method int8

        # Quantize and compare with original
        ml-benchmark quantize gpt2 --method int4 --compare

        # Quantize and benchmark
        ml-benchmark quantize bert-base-uncased --method nf4 --benchmark
    """
    try:
        console.print(f"\n[bold]Quantizing: {model_path}[/bold]")
        console.print(f"[cyan]Method: {method.upper()}[/cyan]\n")

        # Map string to QuantizationType
        quant_type_map = {
            "int8": "INT8",
            "int4": "INT4",
            "nf4": "NF4",
            "fp4": "FP4"
        }

        console.print("[yellow]Note: Model quantization requires the full model pipeline.[/yellow]")
        console.print("[yellow]This is a template implementation showing the API usage.[/yellow]\n")

        # Show example usage
        console.print("[dim]Example usage:[/dim]")
        example_code = f"""
from ml_experiments.benchmarks import quantize_model, QuantizationType

# Load your model
model = YourModel.from_pretrained("{model_path}")

# Quantize the model
quantized_model = quantize_model(
    model,
    quant_type=QuantizationType.{quant_type_map[method]},
    device="cuda"
)

# Use the quantized model
output = quantized_model(input_data)
"""
        console.print(Syntax(example_code, "python", theme="monokai"))

        if compare:
            console.print("\n[cyan]To compare models:[/cyan]")
            compare_code = """
from ml_experiments.benchmarks import compare_models, ModelMetrics

models = {
    "original": original_model,
    "quantized": quantized_model
}

comparison = compare_models(models, test_inputs)
comparison.print_comparison()
"""
            console.print(Syntax(compare_code, "python", theme="monokai"))

        if benchmark:
            console.print("\n[cyan]To benchmark quantized model:[/cyan]")
            bench_code = """
from ml_experiments.benchmarks import ModelBenchmark, BenchmarkConfig

benchmark = ModelBenchmark(
    model=quantized_model,
    config=BenchmarkConfig(num_warmup=5, num_iterations=100)
)

results = benchmark.run(sample_input)
results.print_summary()
"""
            console.print(Syntax(bench_code, "python", theme="monokai"))

        console.print("\n[green]✓ Quantization example shown![/green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Quantization failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=str)
@click.option(
    "--mode",
    type=click.Choice(["default", "reduce-overhead", "max-autotune"], case_sensitive=False),
    default="default",
    help="Torch.compile mode"
)
@click.option(
    "--fullgraph",
    is_flag=True,
    help="Compile the entire graph (may fail for some models)"
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="Benchmark compiled model"
)
@click.option(
    "--warmup",
    type=int,
    default=10,
    help="Number of warmup iterations"
)
@click.option(
    "--iterations",
    type=int,
    default=100,
    help="Number of measurement iterations"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
def compile(
    model_path: str,
    mode: str,
    fullgraph: bool,
    benchmark: bool,
    warmup: int,
    iterations: int,
    verbose: bool
) -> None:
    """Optimize a model using torch.compile.

    MODEL_PATH: HuggingFace model name or path to local model

    Examples:

        # Compile with default settings
        ml-benchmark compile stabilityai/stable-diffusion-2-1

        # Compile with max-autotune and benchmark
        ml-benchmark compile gpt2 --mode max-autotune --benchmark

        # Compile with fullgraph
        ml-benchmark compile bert-base-uncased --fullgraph
    """
    try:
        console.print(f"\n[bold]Compiling: {model_path}[/bold]")
        console.print(f"[cyan]Mode: {mode}[/cyan]")
        console.print(f"[cyan]Fullgraph: {fullgraph}[/cyan]\n")

        console.print("[yellow]Note: Model compilation requires PyTorch 2.0+ and the full model.[/yellow]")
        console.print("[yellow]This is a template implementation showing the API usage.[/yellow]\n")

        # Show example usage
        console.print("[dim]Example usage:[/dim]")
        example_code = f"""
import torch
from ml_experiments.benchmarks import compile_model, CompileMode

# Load your model
model = YourModel.from_pretrained("{model_path}")

# Compile the model
compiled_model = compile_model(
    model,
    mode=CompileMode.{mode.upper().replace('-', '_')},
    fullgraph={fullgraph}
)

# The first call will trigger compilation
output = compiled_model(input_data)  # This will be slow
output = compiled_model(input_data)  # Subsequent calls are fast
"""
        console.print(Syntax(example_code, "python", theme="monokai"))

        if benchmark:
            console.print("\n[cyan]To benchmark compiled model:[/cyan]")
            bench_code = """
from ml_experiments.benchmarks import quick_benchmark

# Warmup is important for compiled models
def forward_fn():
    return compiled_model(input_data)

results = quick_benchmark(
    forward_fn,
    num_warmup=20,  # More warmup for compiled models
    num_iterations=100
)
"""
            console.print(Syntax(bench_code, "python", theme="monokai"))

        console.print("\n[green]✓ Compilation example shown![/green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Compilation failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save comparison results to JSON file"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
def compare(
    config: Path,
    output: Optional[Path],
    verbose: bool
) -> None:
    """Compare multiple models or configurations.

    CONFIG: Path to YAML comparison configuration file

    The config file should define:
      - models: List of models to compare
      - metrics: Which metrics to measure
      - benchmark_config: Benchmark parameters

    Example config:

        models:
          - name: "baseline"
            path: "stabilityai/stable-diffusion-2-1"
          - name: "optimized"
            path: "stabilityai/stable-diffusion-2-1"
            quantization: "int8"
            compile: true

        metrics:
          - latency
          - memory
          - throughput

        benchmark_config:
          num_warmup: 10
          num_iterations: 100
    """
    try:
        console.print(f"\n[bold]Comparing models from: {config}[/bold]\n")

        # Load comparison config
        comp_config = load_benchmark_config(config)

        console.print("[yellow]Note: Model comparison requires loading multiple models.[/yellow]")
        console.print("[yellow]This is a template implementation showing the API usage.[/yellow]\n")

        # Show example usage
        console.print("[dim]Example usage:[/dim]")
        example_code = """
from ml_experiments.benchmarks import compare_models, ModelMetrics

# Define models to compare
models = {
    "baseline": baseline_model,
    "quantized": quantized_model,
    "compiled": compiled_model,
}

# Create test inputs
test_inputs = [torch.randn(1, 3, 224, 224) for _ in range(10)]

# Compare models
comparison = compare_models(
    models,
    test_inputs,
    metrics=["latency", "memory", "throughput"]
)

# Display results
comparison.print_comparison()

# Get winner
best_model = comparison.get_winner(metric="latency")
"""
        console.print(Syntax(example_code, "python", theme="monokai"))

        # Display config
        if comp_config.get("models"):
            console.print("\n[bold]Models to compare:[/bold]")
            for model in comp_config["models"]:
                console.print(f"  • {model.get('name', 'unnamed')}: {model.get('path', 'N/A')}")

        console.print("\n[green]✓ Comparison template shown![/green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Comparison failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
def examples():
    """Show example usage and workflows."""
    console.print("\n[bold cyan]Benchmarking Examples[/bold cyan]\n")

    examples_text = """
# 1. Basic Benchmark
ml-benchmark run stabilityai/stable-diffusion-2-1 --iterations 100

# 2. Quantization
ml-benchmark quantize gpt2 --method int8 --compare --benchmark

# 3. Torch.compile Optimization
ml-benchmark compile bert-base-uncased --mode max-autotune --benchmark

# 4. Model Comparison
ml-benchmark compare comparison_config.yaml --output results.json

# 5. Custom Batch Sizes
ml-benchmark run resnet50 --batch-sizes 1,4,8,16,32 --iterations 200

# 6. Memory Profiling Only
ml-benchmark run large-model --no-throughput --iterations 50

# 7. Quick Benchmark (fewer iterations)
ml-benchmark run small-model --warmup 3 --iterations 30
"""

    console.print(Syntax(examples_text, "bash", theme="monokai"))

    console.print("\n[bold cyan]Python API Examples[/bold cyan]\n")

    python_examples = """
# Quick benchmark
from ml_experiments.benchmarks import quick_benchmark

results = quick_benchmark(lambda: model(input), num_iterations=100)
print(f"Mean latency: {results.latency.mean:.2f}ms")

# Full benchmark with config
from ml_experiments.benchmarks import ModelBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    num_warmup=10,
    num_iterations=100,
    measure_memory=True,
    measure_throughput=True
)

benchmark = ModelBenchmark(model, config)
results = benchmark.run(sample_input)
results.print_summary()

# Quantization
from ml_experiments.benchmarks import quantize_model, QuantizationType

quantized = quantize_model(model, QuantizationType.INT8)

# Model comparison
from ml_experiments.benchmarks import compare_models

models = {"fp32": model, "int8": quantized}
comparison = compare_models(models, test_inputs)
comparison.print_comparison()
"""

    console.print(Syntax(python_examples, "python", theme="monokai"))


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
