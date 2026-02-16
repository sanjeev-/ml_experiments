# Modal Runner Documentation

This module provides Modal integration for running ML experiments on GPU clusters.

## Features

- **Pre-configured GPU Image**: CUDA + PyTorch + HuggingFace + all dependencies
- **Persistent Model Cache**: Modal Volume for HuggingFace models to avoid re-downloading
- **Flexible GPU Selection**: Support for T4, A10G, A100, H100, L4 GPUs
- **Generic Experiment Runner**: Run any experiment class on Modal GPUs
- **Cache Management**: List and clear cached models
- **Secret Management**: Automatic integration with HuggingFace, Weights & Biases, and AWS secrets

## Quick Start

### 1. Install Modal

```bash
pip install modal
modal token new  # Authenticate with Modal
```

### 2. Set up secrets (optional)

```bash
# HuggingFace token
modal secret create huggingface-secret HF_TOKEN=hf_...

# Weights & Biases
modal secret create wandb-secret WANDB_API_KEY=...

# AWS credentials
modal secret create aws-secret AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
```

### 3. Run an experiment

```python
from ml_experiments.core.modal_runner import run_experiment_local

config = {
    "model_name": "stabilityai/stable-diffusion-2-1",
    "prompt": "A beautiful sunset over mountains",
    "num_images": 10,
}

result = run_experiment_local(
    experiment_config=config,
    experiment_class_path="ml_experiments.tasks.text_to_image.TextToImageExperiment",
    gpu_type="a10g",
    timeout=3600,
)

print(result)
```

## GPU Types

Available GPU configurations:

- `t4`: NVIDIA T4 (16GB) - Budget option
- `a10g`: NVIDIA A10G (24GB) - Good balance (default)
- `a100`: NVIDIA A100 (40GB) - High performance
- `a100-80gb`: NVIDIA A100 (80GB) - Very large models
- `h100`: NVIDIA H100 (80GB) - Cutting edge
- `l4`: NVIDIA L4 (24GB) - Inference optimized

## API Reference

### `run_experiment_local()`

Run an experiment from your local machine on Modal GPUs.

**Parameters:**
- `experiment_config` (dict): Configuration for the experiment
- `experiment_class_path` (str): Python import path to experiment class
- `gpu_type` (str): GPU type to use (default: "a10g")
- `timeout` (int): Timeout in seconds (default: 14400 = 4 hours)

**Returns:**
- dict: Results with "status", "results", and "experiment_class" keys

### `list_cached_models()`

List all models in the HuggingFace cache volume.

**Returns:**
- list: Cached model names

**Example:**
```python
from ml_experiments.core.modal_runner import app, list_cached_models

with app.run():
    models = list_cached_models.remote()
    print(models)
```

### `clear_cache()`

Clear the HuggingFace model cache.

**Parameters:**
- `pattern` (str, optional): Glob pattern to match models to delete
- `force` (bool): If True, actually delete. Otherwise dry-run (default: False)

**Returns:**
- dict: Statistics with "deleted", "freed_mb", and "dry_run" keys

**Example:**
```python
from ml_experiments.core.modal_runner import app, clear_cache

with app.run():
    # Dry run
    stats = clear_cache.remote(force=False)
    print(f"Would free {stats['freed_mb']:.2f} MB")

    # Actually delete
    stats = clear_cache.remote(force=True)
    print(f"Freed {stats['freed_mb']:.2f} MB")
```

### `create_gpu_function()`

Create a custom GPU function with specified resources.

**Parameters:**
- `gpu_type` (str): GPU type (default: "a10g")
- `timeout` (int): Timeout in seconds (default: 3600)
- `cpu` (float, optional): Number of CPU cores
- `memory` (int, optional): Memory in MB

**Returns:**
- Function decorator for Modal functions

**Example:**
```python
from ml_experiments.core.modal_runner import create_gpu_function

@create_gpu_function(gpu_type="a100", timeout=7200, memory=32768)
def my_custom_training():
    # Your code here
    pass
```

## Cache Volume

The HuggingFace cache volume (`hf-model-cache`) persists across runs:

- **Location**: `/cache/huggingface` in Modal containers
- **Contents**: Downloaded models, tokenizers, and datasets
- **Benefits**:
  - Faster experiment startup (no re-downloading)
  - Reduced bandwidth usage
  - Consistent model versions

The volume is automatically mounted to all GPU functions.

## Environment Variables

The GPU image sets these environment variables:

- `HF_HOME=/cache/huggingface` - HuggingFace home directory
- `TRANSFORMERS_CACHE=/cache/huggingface/transformers` - Transformers cache
- `HF_DATASETS_CACHE=/cache/huggingface/datasets` - Datasets cache
- `TORCH_HOME=/cache/torch` - PyTorch home directory

## Advanced Usage

### Custom GPU Configuration

```python
from ml_experiments.core.modal_runner import app, gpu_image, hf_cache_volume
import modal

@app.function(
    image=gpu_image,
    gpu=modal.gpu.A100(count=2),  # Multi-GPU
    volumes={"/cache": hf_cache_volume},
    timeout=7200,
)
def multi_gpu_training():
    # Your multi-GPU training code
    pass
```

### Running Modal CLI Commands

```bash
# Deploy the app
modal deploy ml_experiments/core/modal_runner.py

# Run a function directly
modal run ml_experiments/core/modal_runner.py::list_cached_models

# View app in web UI
modal app list
```

## Troubleshooting

### Import Errors

If you get import errors for your experiment classes, ensure:
1. Your experiment module is included in the Modal image
2. The Python path is correctly set
3. All dependencies are installed in the GPU image

### Out of Memory

If you run out of GPU memory:
1. Use a larger GPU type (e.g., a100-80gb)
2. Reduce batch size in your experiment config
3. Enable gradient checkpointing
4. Use quantization (int8/int4)

### Timeout Issues

If experiments timeout:
1. Increase the `timeout` parameter
2. Use a faster GPU (e.g., h100 vs a10g)
3. Reduce dataset size or number of iterations

### Cache Issues

If models aren't being cached:
1. Check volume is mounted: `volumes={"/cache": hf_cache_volume}`
2. Verify environment variables are set correctly
3. Use `list_cached_models()` to verify cache contents

## Examples

See the `examples/` directory for complete examples:
- `examples/text_to_image_modal.py` - Running Stable Diffusion on Modal
- `examples/benchmark_modal.py` - Benchmarking models on different GPUs
- `examples/cache_management.py` - Managing the HuggingFace cache
