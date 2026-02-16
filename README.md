# ML Experiments Framework

A modular, production-ready framework for running machine learning experiments on Modal GPU clusters with comprehensive benchmarking, optimization, and data streaming capabilities.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **🎯 Task-Specific Experiments**: Pre-built experiment classes for text-to-image, segmentation, matting, VAE training, and more
- **☁️ Modal GPU Integration**: Run experiments on cloud GPUs (T4, A10G, A100, H100) with automatic scaling
- **📊 Comprehensive Benchmarking**: Latency, throughput, memory profiling, and model comparison tools
- **⚡ Optimization Built-in**: Quantization (int8/int4), torch.compile, and bitsandbytes integration
- **🗂️ S3 WebDataset Streaming**: Efficient streaming of large datasets from S3 with automatic sharding
- **📈 Metric Tracking**: FID, LPIPS, CLIP-score, with Weights & Biases integration
- **🔧 Configuration-Driven**: YAML-based configuration with Pydantic validation
- **💾 Automatic Checkpointing**: Local and S3 checkpoint management
- **🎨 Rich CLI**: Beautiful terminal output with progress tracking

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Benchmarking Models](#benchmarking-models)
  - [Data Streaming](#data-streaming)
  - [Modal GPU Execution](#modal-gpu-execution)
- [Configuration](#configuration)
- [Available Experiments](#available-experiments)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (for local GPU execution)
- Modal account (for cloud GPU execution)
- AWS credentials (for S3 data streaming)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml_experiments.git
cd ml_experiments

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev]"
```

### Modal Setup

For cloud GPU execution, set up Modal:

```bash
# Install Modal
pip install modal

# Authenticate
modal token new
```

### AWS Setup

For S3 data streaming, configure AWS credentials:

```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

## 🚀 Quick Start

### 1. Run Your First Experiment (Locally)

```bash
# Create a config file
cp configs/base.yaml configs/my_experiment.yaml

# Edit the config to customize your experiment
# Then run locally
ml-experiment configs/my_experiment.yaml
```

### 2. Run on Modal GPU

```bash
# Run on A100 GPU
ml-experiment configs/my_experiment.yaml --mode modal --gpu a100

# Run on H100 with longer timeout
ml-experiment configs/my_experiment.yaml --mode modal --gpu h100 --timeout 7200
```

### 3. Benchmark a Model

```bash
# Quick benchmark
ml-benchmark run stabilityai/stable-diffusion-2-1 --iterations 100

# Benchmark with quantization
ml-benchmark quantize gpt2 --method int8 --compare --benchmark

# Compare multiple models
ml-benchmark compare configs/comparison.yaml
```

### 4. Generate Images (Text-to-Image)

```python
from ml_experiments.tasks.text_to_image import TextToImageExperiment, TextToImageConfig

# Configure experiment
config = TextToImageConfig(
    experiment_name="demo",
    experiment_type="text_to_image",
    model_id="stabilityai/stable-diffusion-2-1",
    prompts=["A cat sitting on a windowsill at sunset"],
    num_inference_steps=50,
    guidance_scale=7.5
)

# Run experiment
experiment = TextToImageExperiment(config)
results = experiment.run()

# Results include generated images, metrics, and timing info
print(f"Generated {len(results['images'])} images")
print(f"Average latency: {results['metrics']['latency_mean']:.2f}ms")
```

## 🏗️ Architecture

The framework is organized into several core modules:

```
ml_experiments/
├── core/              # Core framework components
│   ├── experiment.py      # Base Experiment class
│   ├── modal_runner.py    # Modal GPU integration
│   ├── data.py            # WebDataset S3 streaming
│   └── metrics.py         # Evaluation metrics
├── tasks/             # Task-specific experiments
│   ├── text_to_image.py   # Stable Diffusion experiments
│   ├── segmentation.py    # SAM segmentation
│   ├── matting.py         # Image matting
│   ├── vae.py             # VAE training
│   ├── text_encoder.py    # CLIP/T5 training
│   └── synthetic_data.py  # Blender pipeline
├── benchmarks/        # Benchmarking utilities
│   ├── benchmark.py       # Performance measurement
│   ├── quantize.py        # Quantization tools
│   └── compare.py         # Model comparison
├── utils/             # Utility modules
│   ├── s3.py              # S3 operations
│   ├── hf.py              # HuggingFace helpers
│   └── viz.py             # Visualization tools
└── scripts/           # CLI entry points
    ├── run_experiment.py  # Main experiment runner
    └── benchmark.py       # Benchmarking CLI
```

### Key Concepts

#### Experiment Base Class

All experiments inherit from the `Experiment` base class, which provides:
- Configuration loading and validation (Pydantic)
- Structured logging (Rich)
- Automatic checkpointing
- Metric tracking (W&B integration)
- Abstract methods for train/eval/inference

#### Modal Integration

The `modal_runner.py` module provides:
- Pre-configured GPU images (CUDA + PyTorch + HuggingFace)
- Volume mounting for HuggingFace model cache
- Generic `run_experiment()` function for any experiment class
- Automatic GPU selection and configuration

#### WebDataset Streaming

The `data.py` module enables:
- Streaming large datasets from S3 without local storage
- Automatic shard discovery and loading
- Standard image preprocessing pipelines
- Error handling for network issues

## 📖 Usage

### Running Experiments

#### Command Line

```bash
# Basic usage
ml-experiment configs/my_experiment.yaml

# With options
ml-experiment configs/my_experiment.yaml \
  --mode modal \
  --gpu a100 \
  --timeout 3600 \
  --verbose

# List available experiment types
ml-experiment --list-experiments
```

#### Python API

```python
from ml_experiments.core import Experiment, ExperimentConfig

# Load config from YAML
config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")

# Or create programmatically
config = ExperimentConfig(
    experiment_name="my_experiment",
    experiment_type="text_to_image",
    output_dir="./outputs",
    seed=42
)

# Get experiment class and run
experiment_cls = ExperimentFactory.get_experiment_class(config.experiment_type)
experiment = experiment_cls(config)
results = experiment.run()
```

### Benchmarking Models

#### Quick Benchmark

```bash
# Benchmark a model
ml-benchmark run stabilityai/stable-diffusion-2-1 \
  --iterations 100 \
  --warmup 10 \
  --memory \
  --throughput

# With multiple batch sizes
ml-benchmark run resnet50 \
  --batch-sizes 1,4,8,16,32 \
  --output results.json
```

#### Quantization

```bash
# Quantize to int8
ml-benchmark quantize gpt2 \
  --method int8 \
  --compare \
  --benchmark

# Quantize to int4 with NF4
ml-benchmark quantize llama-2-7b \
  --method nf4 \
  --compare
```

#### Torch.compile Optimization

```bash
# Compile with default settings
ml-benchmark compile bert-base-uncased \
  --benchmark

# Compile with max-autotune
ml-benchmark compile stable-diffusion-2-1 \
  --mode max-autotune \
  --benchmark \
  --warmup 20
```

#### Model Comparison

Create a comparison config:

```yaml
# comparison_config.yaml
models:
  - name: "baseline"
    path: "stabilityai/stable-diffusion-2-1"

  - name: "int8_quantized"
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
```

Run comparison:

```bash
ml-benchmark compare comparison_config.yaml --output results.json
```

### Data Streaming

#### WebDataset from S3

```python
from ml_experiments.core.data import get_dataset, DatasetConfig

# Configure dataset
config = DatasetConfig(
    source="webdataset",
    s3_path="s3://my-bucket/datasets/imagenet/{00000..01281}.tar",
    batch_size=32,
    num_workers=4,
    image_size=512
)

# Get streaming dataset
dataset = get_dataset(config)

# Iterate over batches
for batch in dataset:
    images = batch["images"]  # Tensor of shape [B, C, H, W]
    labels = batch.get("labels")  # Optional labels
    # Process batch...
```

#### Local Dataset

```python
config = DatasetConfig(
    source="local",
    local_path="./data/my_dataset",
    batch_size=32,
    image_size=512
)

dataset = get_dataset(config)
```

#### HuggingFace Dataset

```python
config = DatasetConfig(
    source="huggingface",
    hf_dataset="lambdalabs/pokemon-blip-captions",
    batch_size=32
)

dataset = get_dataset(config)
```

### Modal GPU Execution

#### Using the CLI

```bash
# Run on different GPU types
ml-experiment configs/my_experiment.yaml --mode modal --gpu t4      # Tesla T4
ml-experiment configs/my_experiment.yaml --mode modal --gpu a10g    # A10G
ml-experiment configs/my_experiment.yaml --mode modal --gpu a100    # A100 40GB
ml-experiment configs/my_experiment.yaml --mode modal --gpu h100    # H100 80GB
ml-experiment configs/my_experiment.yaml --mode modal --gpu l4      # L4

# With custom timeout (in seconds)
ml-experiment configs/my_experiment.yaml --mode modal --gpu a100 --timeout 7200
```

#### Python API

```python
from ml_experiments.core.modal_runner import run_experiment_on_modal

# Run experiment on Modal
config_dict = {
    "experiment_name": "my_experiment",
    "experiment_type": "text_to_image",
    # ... other config
}

results = run_experiment_on_modal(
    config_dict=config_dict,
    gpu_type="a100",
    timeout=3600
)
```

#### Custom Modal Functions

```python
import modal
from ml_experiments.core.modal_runner import app, gpu_image

@app.function(
    image=gpu_image,
    gpu="a100",
    timeout=3600
)
def my_custom_function():
    # Your code here
    pass
```

## ⚙️ Configuration

### Configuration File Structure

The framework uses YAML configuration files with Pydantic validation. See [`configs/base.yaml`](configs/base.yaml) for a comprehensive template.

#### Core Settings

```yaml
experiment_name: "my_experiment_001"
experiment_type: "text_to_image"
description: "My first experiment"
output_dir: "./outputs/my_experiment"
seed: 42
device: "auto"
log_level: "INFO"
```

#### Checkpointing

```yaml
checkpoint_dir: "./outputs/checkpoints"
checkpoint_frequency: 100  # Save every 100 steps
checkpoint_to_s3: true
s3_bucket: "my-ml-experiments"
s3_prefix: "checkpoints/my_experiment"
```

#### Metrics and Logging

```yaml
use_wandb: true
wandb_project: "ml-experiments"
wandb_entity: "my-team"
wandb_tags: ["experiment", "baseline"]
log_frequency: 10
```

#### Data Configuration

```yaml
data:
  source: "webdataset"
  s3_path: "s3://my-bucket/datasets/my_data/{00000..00099}.tar"
  batch_size: 32
  num_workers: 4
  image_size: 512
  shuffle: true
```

#### Model Configuration (Text-to-Image)

```yaml
model:
  model_id: "stabilityai/stable-diffusion-2-1"
  variant: "fp16"
  use_safetensors: true
  enable_xformers: true

generation:
  num_inference_steps: 50
  guidance_scale: 7.5
  height: 512
  width: 512
  scheduler: "ddim"
```

#### Training Configuration

```yaml
training:
  num_epochs: 10
  batch_size: 8
  learning_rate: 5.0e-6
  lr_scheduler: "cosine"
  optimizer: "adamw"
  mixed_precision: "fp16"
  gradient_checkpointing: true
```

#### Optimization

```yaml
optimization:
  use_compile: true
  compile_mode: "max-autotune"
  quantization: "int8"  # or "int4", "nf4", "fp4"
```

### Environment Variables

You can use environment variables in config files:

```yaml
s3_bucket: ${S3_BUCKET}
wandb_project: ${WANDB_PROJECT}
hf_token: ${HF_TOKEN}
```

## 🎯 Available Experiments

### Text-to-Image

Stable Diffusion text-to-image generation with support for fine-tuning.

```python
from ml_experiments.tasks.text_to_image import TextToImageExperiment, TextToImageConfig

config = TextToImageConfig(
    experiment_name="txt2img_demo",
    experiment_type="text_to_image",
    model_id="stabilityai/stable-diffusion-2-1",
    prompts=["A beautiful sunset over mountains"],
    num_inference_steps=50,
    guidance_scale=7.5
)

experiment = TextToImageExperiment(config)
results = experiment.run()
```

### Segmentation

Image segmentation using Segment Anything Model (SAM).

```python
from ml_experiments.tasks.segmentation import SegmentationExperiment, SegmentationConfig

config = SegmentationConfig(
    experiment_name="segmentation_demo",
    experiment_type="segmentation",
    model_id="facebook/sam-vit-huge",
    # ... other config
)

experiment = SegmentationExperiment(config)
results = experiment.run()
```

### Matting

Image matting for foreground/background separation.

```python
from ml_experiments.tasks.matting import MattingExperiment, MattingConfig

config = MattingConfig(
    experiment_name="matting_demo",
    experiment_type="matting",
    # ... config
)

experiment = MattingExperiment(config)
results = experiment.run()
```

### VAE Training

Train Variational Autoencoders for image compression and generation.

```python
from ml_experiments.tasks.vae import VAEExperiment, VAEConfig

config = VAEConfig(
    experiment_name="vae_training",
    experiment_type="vae",
    latent_dim=128,
    # ... config
)

experiment = VAEExperiment(config)
results = experiment.run()
```

### Text Encoder Training

Fine-tune CLIP or T5 text encoders.

```python
from ml_experiments.tasks.text_encoder import TextEncoderExperiment, TextEncoderConfig

config = TextEncoderConfig(
    experiment_name="text_encoder_finetune",
    experiment_type="text_encoder",
    model_id="openai/clip-vit-large-patch14",
    # ... config
)

experiment = TextEncoderExperiment(config)
results = experiment.run()
```

### Synthetic Data Generation

Generate synthetic training data using Blender.

```python
from ml_experiments.tasks.synthetic_data import SyntheticDataExperiment, SyntheticDataConfig

config = SyntheticDataConfig(
    experiment_name="synthetic_data_gen",
    experiment_type="synthetic_data",
    num_samples=10000,
    # ... config
)

experiment = SyntheticDataExperiment(config)
results = experiment.run()
```

## 💡 Examples

### Example 1: Text-to-Image Generation with Benchmarking

```python
from ml_experiments.tasks.text_to_image import TextToImageExperiment, TextToImageConfig

config = TextToImageConfig(
    experiment_name="sdxl_benchmark",
    experiment_type="text_to_image",
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    prompts=[
        "A majestic lion in the savanna at sunset",
        "A futuristic city with flying cars",
        "An underwater scene with colorful coral reefs"
    ],
    num_inference_steps=50,
    guidance_scale=7.5,
    output_dir="./outputs/sdxl_benchmark",
    use_wandb=True,
    wandb_project="text-to-image",
    evaluation={
        "benchmark_performance": True,
        "benchmark_warmup": 5,
        "benchmark_iterations": 100
    }
)

experiment = TextToImageExperiment(config)
results = experiment.run()

# Results include images, metrics, and performance data
print(f"Generated {len(results['images'])} images")
print(f"Average latency: {results['benchmark']['latency_mean']:.2f}ms")
print(f"Throughput: {results['benchmark']['throughput']:.2f} imgs/sec")
print(f"Peak memory: {results['benchmark']['memory_peak_mb']:.2f}MB")
```

### Example 2: Multi-GPU Training on Modal

```python
from ml_experiments.core.modal_runner import run_experiment_on_modal

config_dict = {
    "experiment_name": "sdxl_finetune",
    "experiment_type": "text_to_image",
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "data": {
        "source": "webdataset",
        "s3_path": "s3://my-bucket/training-data/{00000..00099}.tar",
        "batch_size": 4
    },
    "training": {
        "num_epochs": 10,
        "learning_rate": 1e-5,
        "mixed_precision": "fp16"
    },
    "checkpoint_to_s3": True,
    "s3_bucket": "my-checkpoints",
    "use_wandb": True,
    "wandb_project": "sdxl-finetune"
}

# Run on dual A100s
results = run_experiment_on_modal(
    config_dict=config_dict,
    gpu_type="a100",
    gpu_count=2,
    timeout=7200
)
```

### Example 3: Model Quantization and Comparison

```bash
# Create comparison config
cat > comparison.yaml << EOF
models:
  - name: "fp32_baseline"
    path: "stabilityai/stable-diffusion-2-1"

  - name: "fp16"
    path: "stabilityai/stable-diffusion-2-1"
    variant: "fp16"

  - name: "int8_quantized"
    path: "stabilityai/stable-diffusion-2-1"
    quantization: "int8"

  - name: "int4_nf4"
    path: "stabilityai/stable-diffusion-2-1"
    quantization: "nf4"

metrics:
  - latency
  - memory
  - throughput

benchmark_config:
  num_warmup: 10
  num_iterations: 100
EOF

# Run comparison
ml-benchmark compare comparison.yaml --output comparison_results.json
```

Results will show a comparison table:

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Model          ┃ Latency   ┃ Memory (GB) ┃ Throughput    ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ fp32_baseline  │ 523.4ms   │ 8.2GB       │ 1.91 imgs/sec │
│ fp16           │ 287.6ms   │ 4.1GB       │ 3.48 imgs/sec │
│ int8_quantized │ 198.3ms   │ 2.9GB       │ 5.04 imgs/sec │
│ int4_nf4       │ 156.7ms   │ 2.1GB       │ 6.38 imgs/sec │
└────────────────┴───────────┴─────────────┴───────────────┘
```

### Example 4: Streaming Large Datasets from S3

```python
from ml_experiments.core.data import get_dataset, DatasetConfig
from ml_experiments.tasks.vae import VAEExperiment, VAEConfig

# Configure streaming dataset
data_config = DatasetConfig(
    source="webdataset",
    s3_path="s3://my-bucket/large-dataset/{00000..09999}.tar",
    batch_size=64,
    num_workers=8,
    image_size=256,
    shuffle=True
)

# Get streaming dataset
dataset = get_dataset(data_config)

# Configure VAE training
vae_config = VAEConfig(
    experiment_name="vae_large_scale",
    experiment_type="vae",
    latent_dim=256,
    training={
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "mixed_precision": "fp16"
    },
    checkpoint_frequency=1000,
    checkpoint_to_s3=True
)

# Run training with streaming data
experiment = VAEExperiment(vae_config)
experiment.train(dataset)
```

## 📚 API Reference

### Core Classes

#### `Experiment`

Base class for all experiments.

```python
class Experiment(ABC):
    def __init__(self, config: ExperimentConfig)
    def run(self) -> Dict[str, Any]

    @abstractmethod
    def train(self, dataset) -> Dict[str, Any]

    @abstractmethod
    def evaluate(self, dataset) -> Dict[str, Any]

    @abstractmethod
    def infer(self, inputs) -> Dict[str, Any]

    def save_checkpoint(self, step: int)
    def load_checkpoint(self, checkpoint_path: str)
    def log_metrics(self, metrics: Dict[str, Any], step: int)
```

#### `ExperimentConfig`

Base configuration class (Pydantic model).

```python
class ExperimentConfig(BaseModel):
    experiment_name: str
    experiment_type: str
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"
    # ... many more fields

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig"

    def to_yaml(self, path: str)
```

### Benchmarking Functions

```python
from ml_experiments.benchmarks import (
    quick_benchmark,
    ModelBenchmark,
    quantize_model,
    compare_models
)

# Quick benchmark
results = quick_benchmark(
    func=lambda: model(input),
    num_iterations=100
)

# Full benchmark
benchmark = ModelBenchmark(model, config)
results = benchmark.run(sample_input)

# Quantization
quantized = quantize_model(model, method="int8")

# Comparison
comparison = compare_models(models_dict, test_inputs)
```

### Data Functions

```python
from ml_experiments.core.data import (
    get_dataset,
    DatasetConfig,
    list_s3_shards
)

# Get dataset
config = DatasetConfig(source="webdataset", s3_path="...")
dataset = get_dataset(config)

# List S3 shards
shards = list_s3_shards("s3://bucket/dataset/{00000..00099}.tar")
```

### Utility Functions

```python
from ml_experiments.utils import (
    load_model_from_hf,
    save_image_grid,
    upload_to_s3,
    download_from_s3
)

# Load HuggingFace model
model = load_model_from_hf("stabilityai/stable-diffusion-2-1")

# Save image grid
save_image_grid(images, path="output.png", nrow=4)

# S3 operations
upload_to_s3(local_path, "s3://bucket/key")
download_from_s3("s3://bucket/key", local_path)
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/ml_experiments.git
cd ml_experiments

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_experiments --cov-report=html

# Run specific test
pytest tests/test_experiment.py::test_experiment_init
```

### Code Formatting

```bash
# Format with black
black ml_experiments/ tests/

# Lint with ruff
ruff check ml_experiments/ tests/

# Type checking
mypy ml_experiments/
```

### Creating a Custom Experiment

1. Create a new file in `ml_experiments/tasks/`:

```python
# ml_experiments/tasks/my_task.py
from ml_experiments.core import Experiment, ExperimentConfig
from pydantic import Field

class MyTaskConfig(ExperimentConfig):
    # Add task-specific config fields
    my_param: int = Field(default=10, description="My parameter")

class MyTaskExperiment(Experiment):
    def __init__(self, config: MyTaskConfig):
        super().__init__(config)
        # Initialize task-specific components

    def train(self, dataset):
        # Implement training logic
        pass

    def evaluate(self, dataset):
        # Implement evaluation logic
        pass

    def infer(self, inputs):
        # Implement inference logic
        pass
```

2. Register in `ml_experiments/tasks/__init__.py`:

```python
from .my_task import MyTaskExperiment, MyTaskConfig

__all__ = [..., "MyTaskExperiment", "MyTaskConfig"]
```

3. Use your experiment:

```bash
ml-experiment configs/my_task.yaml --experiment-type my_task
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Enable gradient checkpointing
training:
  gradient_checkpointing: true

# Reduce batch size
training:
  batch_size: 4

# Enable CPU offloading
model:
  enable_cpu_offload: true
```

#### Modal Connection Issues

```bash
# Re-authenticate
modal token new

# Check Modal status
modal status
```

#### S3 Access Denied

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check bucket permissions
aws s3 ls s3://my-bucket/
```

#### Import Errors

```bash
# Reinstall package
pip uninstall ml-experiments
pip install -e .

# Clear cache
pip cache purge
```

### Getting Help

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ml_experiments/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ml_experiments/discussions)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Modal](https://modal.com/) for GPU infrastructure
- [HuggingFace](https://huggingface.co/) for model hub and libraries
- [PyTorch](https://pytorch.org/) for deep learning framework
- [WebDataset](https://github.com/webdataset/webdataset) for efficient data streaming

## 📞 Contact

- Project Maintainer: ML Experiments Team
- Email: ml-experiments@example.com
- Twitter: [@ml_experiments](https://twitter.com/ml_experiments)

---

**Built with ❤️ for the ML community**
