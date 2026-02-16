# ML Experiments CLI Scripts

This directory contains the command-line interface entry points for the ML Experiments framework.

## Scripts

### `run_experiment.py`

Main CLI for running ML experiments locally or on Modal GPU clusters.

**Features:**
- YAML config loading and validation
- Local or Modal execution modes
- Rich console output with progress tracking
- GPU configuration options (T4, A10G, A100, H100, L4)
- Experiment registry integration

**Usage:**

```bash
# Run experiment locally
ml-experiment configs/example_experiment.yaml

# Run on Modal with A100 GPU
ml-experiment configs/example_experiment.yaml --mode modal --gpu a100

# Run with verbose output
ml-experiment configs/example_experiment.yaml -v

# List available experiment types
ml-experiment --list-experiments

# Show help
ml-experiment --help
```

**Options:**
- `--mode [local|modal]`: Execution mode (default: local)
- `--gpu [t4|a10g|a100|a100-80gb|h100|l4]`: GPU type for Modal (default: a10g)
- `--timeout INT`: Timeout in seconds for Modal execution (default: 3600)
- `-v, --verbose`: Enable verbose output
- `--list-experiments`: List available experiment types

### `benchmark.py`

Comprehensive model benchmarking tool with quantization and optimization support.

**Features:**
- Latency, throughput, and memory profiling
- Quantization (int8, int4, NF4, FP4) via bitsandbytes
- Torch.compile optimization
- Model comparison with Rich tables
- Statistical analysis (mean, std, percentiles)

**Usage:**

```bash
# Basic benchmark
ml-benchmark run stabilityai/stable-diffusion-2-1

# Benchmark with custom iterations
ml-benchmark run gpt2 --iterations 200 --warmup 10

# Benchmark with multiple batch sizes
ml-benchmark run resnet50 --batch-sizes 1,4,8,16,32

# Quantize model
ml-benchmark quantize gpt2 --method int8 --compare --benchmark

# Compile model with torch.compile
ml-benchmark compile bert-base-uncased --mode max-autotune --benchmark

# Compare multiple models
ml-benchmark compare comparison_config.yaml

# Show examples
ml-benchmark examples

# Show help
ml-benchmark --help
```

**Commands:**

#### `run`
Run benchmark on a single model.

Options:
- `--warmup INT`: Number of warmup iterations (default: 5)
- `--iterations INT`: Number of measurement iterations (default: 100)
- `--memory/--no-memory`: Measure GPU memory (default: true)
- `--throughput/--no-throughput`: Measure throughput (default: true)
- `--sync-cuda/--no-sync-cuda`: Synchronize CUDA before timing (default: true)
- `--clear-cache`: Clear CUDA cache between iterations
- `--batch-sizes STR`: Comma-separated batch sizes (e.g., "1,4,8,16")
- `--output PATH`: Save results to JSON file
- `-v, --verbose`: Verbose output

#### `quantize`
Quantize a model using bitsandbytes.

Options:
- `--method [int8|int4|nf4|fp4]`: Quantization method (default: int8)
- `--compare`: Compare quantized vs original
- `--benchmark`: Benchmark quantized model
- `--warmup INT`: Warmup iterations (default: 5)
- `--iterations INT`: Measurement iterations (default: 100)
- `-v, --verbose`: Verbose output

#### `compile`
Optimize model with torch.compile.

Options:
- `--mode [default|reduce-overhead|max-autotune]`: Compile mode (default: default)
- `--fullgraph`: Compile entire graph
- `--benchmark`: Benchmark compiled model
- `--warmup INT`: Warmup iterations (default: 10)
- `--iterations INT`: Measurement iterations (default: 100)
- `-v, --verbose`: Verbose output

#### `compare`
Compare multiple models or configurations.

Requires a YAML config file defining:
- `models`: List of models to compare
- `metrics`: Metrics to measure
- `benchmark_config`: Benchmark parameters

Options:
- `--output PATH`: Save results to JSON
- `-v, --verbose`: Verbose output

#### `examples`
Show example usage and workflows.

## Installation

The scripts are automatically installed as console entry points when you install the package:

```bash
# Install in development mode
pip install -e .

# Or install from source
pip install .
```

This creates two commands:
- `ml-experiment` → `ml_experiments.scripts.run_experiment:main`
- `ml-benchmark` → `ml_experiments.scripts.benchmark:main`

## Configuration Files

### Experiment Config (for `run_experiment.py`)

Example `configs/experiment.yaml`:

```yaml
experiment_name: "my_experiment"
experiment_type: "text_to_image"
description: "Example experiment"

# Output settings
output_dir: "./outputs/my_experiment"
seed: 42
device: "auto"
log_level: "INFO"

# Checkpointing
checkpoint_dir: "./outputs/my_experiment/checkpoints"
checkpoint_frequency: 100
checkpoint_to_s3: false

# Metrics
use_wandb: false
log_frequency: 10

# Task-specific config
# Add your task-specific parameters here
```

### Benchmark Comparison Config (for `benchmark.py compare`)

Example `comparison_config.yaml`:

```yaml
models:
  - name: "baseline"
    path: "stabilityai/stable-diffusion-2-1"
  - name: "quantized"
    path: "stabilityai/stable-diffusion-2-1"
    quantization: "int8"
  - name: "compiled"
    path: "stabilityai/stable-diffusion-2-1"
    compile: true
    compile_mode: "max-autotune"

metrics:
  - latency
  - memory
  - throughput

benchmark_config:
  num_warmup: 10
  num_iterations: 100
  measure_memory: true
  measure_throughput: true
```

## Python API Integration

Both scripts are built on top of the benchmarking and experiment modules. You can also use the Python API directly:

```python
# Running experiments programmatically
from ml_experiments.tasks.text_to_image import TextToImageExperiment, TextToImageConfig

config = TextToImageConfig(
    experiment_name="my_experiment",
    experiment_type="text_to_image",
    # ... other config
)

experiment = TextToImageExperiment(config)
results = experiment.run()

# Benchmarking programmatically
from ml_experiments.benchmarks import quick_benchmark, ModelBenchmark

# Quick benchmark
results = quick_benchmark(lambda: model(input), num_iterations=100)
print(f"Mean latency: {results.latency.mean:.2f}ms")

# Full benchmark
from ml_experiments.benchmarks import BenchmarkConfig

config = BenchmarkConfig(num_warmup=10, num_iterations=100)
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
```

## Examples

### 1. Run Text-to-Image Experiment Locally

```bash
ml-experiment configs/text_to_image.yaml
```

### 2. Run on Modal with H100 GPU

```bash
ml-experiment configs/training.yaml --mode modal --gpu h100
```

### 3. Benchmark a Model

```bash
ml-benchmark run stabilityai/stable-diffusion-2-1 \
  --iterations 200 \
  --warmup 10 \
  --batch-sizes 1,4,8
```

### 4. Quantize and Compare

```bash
ml-benchmark quantize gpt2 \
  --method int8 \
  --compare \
  --benchmark \
  --iterations 100
```

### 5. Optimize with Torch.compile

```bash
ml-benchmark compile bert-base-uncased \
  --mode max-autotune \
  --benchmark \
  --warmup 20
```

### 6. Compare Multiple Configurations

```bash
ml-benchmark compare comparison_config.yaml --output results.json
```

## Requirements

- Python 3.9+
- click >= 8.1.0
- pyyaml >= 6.0
- rich >= 13.0.0
- torch >= 2.0.0
- modal >= 0.55.0 (for Modal execution)
- All dependencies from `pyproject.toml`

## Development

To modify the scripts:

1. Edit the scripts in this directory
2. Test locally: `python scripts/run_experiment.py --help`
3. Reinstall package: `pip install -e .`
4. Run via console script: `ml-experiment --help`

## Troubleshooting

### Import Errors

If you get import errors, ensure the package is installed:

```bash
pip install -e .
```

### Modal Connection Issues

For Modal execution, ensure you're logged in:

```bash
modal token new
```

### GPU Not Available

For local GPU execution, ensure CUDA is properly installed:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## License

MIT License - see project LICENSE file for details.
