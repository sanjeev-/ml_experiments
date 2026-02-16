"""Modal integration for running ML experiments on GPU clusters.

This module provides a Modal app definition with GPU image configuration
and a generic runner for executing experiments on Modal's infrastructure.
"""

import modal
from typing import Dict, Any, Optional, List
from pathlib import Path


# Define the Modal app
app = modal.App("ml-experiments")

# Create a Modal volume for HuggingFace model cache
# This persists downloaded models across runs to avoid re-downloading
hf_cache_volume = modal.Volume.from_name(
    "hf-model-cache",
    create_if_missing=True
)

# Define the GPU image with all necessary dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
    )
    .pip_install(
        # Core ML frameworks
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchaudio>=2.1.0",
        # HuggingFace ecosystem
        "transformers>=4.36.0",
        "diffusers>=0.25.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        # Data and storage
        "webdataset>=0.2.0",
        "boto3>=1.34.0",
        "s3fs>=2024.0.0",
        # Quantization and optimization
        "bitsandbytes>=0.41.0",
        "triton>=2.1.0",
        # Config and utilities
        "pydantic>=2.5.0",
        "pyyaml>=6.0",
        "rich>=13.7.0",
        # Metrics and evaluation
        "torchmetrics>=1.2.0",
        "clean-fid>=0.1.35",
        "wandb>=0.16.0",
        # Additional dependencies
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
        "TORCH_HOME": "/cache/torch",
    })
)


# GPU configuration options
GPU_CONFIGS = {
    "t4": modal.gpu.T4(),
    "a10g": modal.gpu.A10G(),
    "a100": modal.gpu.A100(),
    "a100-80gb": modal.gpu.A100(size="80GB"),
    "h100": modal.gpu.H100(),
    "l4": modal.gpu.L4(),
}


@app.function(
    image=gpu_image,
    gpu=modal.gpu.A10G(),  # Default GPU
    volumes={"/cache": hf_cache_volume},
    timeout=3600 * 4,  # 4 hours default timeout
    secrets=[
        modal.Secret.from_name("huggingface-secret", required=False),
        modal.Secret.from_name("wandb-secret", required=False),
        modal.Secret.from_name("aws-secret", required=False),
    ],
)
def run_experiment(
    experiment_config: Dict[str, Any],
    experiment_class_path: str,
) -> Dict[str, Any]:
    """Run an experiment on Modal GPUs.

    This function dynamically imports and instantiates an experiment class,
    then runs it with the provided configuration.

    Args:
        experiment_config: Configuration dictionary for the experiment
        experiment_class_path: Python import path to the experiment class
                              (e.g., "ml_experiments.tasks.text_to_image.TextToImageExperiment")

    Returns:
        Dictionary containing experiment results and metrics
    """
    import importlib
    import sys
    from pathlib import Path

    # Add the workspace to the Python path if not already there
    workspace_path = Path("/root")
    if str(workspace_path) not in sys.path:
        sys.path.insert(0, str(workspace_path))

    try:
        # Parse the experiment class path
        module_path, class_name = experiment_class_path.rsplit(".", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the experiment class
        experiment_class = getattr(module, class_name)

        # Instantiate the experiment with config
        experiment = experiment_class(config=experiment_config)

        # Run the experiment
        print(f"🚀 Starting experiment: {experiment_class_path}")
        print(f"📊 Config: {experiment_config}")

        results = experiment.run()

        print(f"✅ Experiment completed successfully")

        return {
            "status": "success",
            "results": results,
            "experiment_class": experiment_class_path,
        }

    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "experiment_class": experiment_class_path,
        }


@app.function(
    image=gpu_image,
    gpu=None,  # No GPU needed for this
    volumes={"/cache": hf_cache_volume},
    timeout=600,
)
def list_cached_models() -> List[str]:
    """List all models cached in the HuggingFace cache volume.

    Returns:
        List of cached model names/paths
    """
    from pathlib import Path

    cache_dir = Path("/cache/huggingface")
    if not cache_dir.exists():
        return []

    cached_models = []

    # List models in transformers cache
    transformers_cache = cache_dir / "transformers"
    if transformers_cache.exists():
        for model_dir in transformers_cache.iterdir():
            if model_dir.is_dir():
                cached_models.append(f"transformers/{model_dir.name}")

    # List models in diffusers cache
    diffusers_cache = cache_dir / "diffusers"
    if diffusers_cache.exists():
        for model_dir in diffusers_cache.iterdir():
            if model_dir.is_dir():
                cached_models.append(f"diffusers/{model_dir.name}")

    return cached_models


@app.function(
    image=gpu_image,
    gpu=None,
    volumes={"/cache": hf_cache_volume},
    timeout=600,
)
def clear_cache(
    pattern: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """Clear the HuggingFace model cache.

    Args:
        pattern: Optional glob pattern to match models to delete
        force: If True, actually delete files. Otherwise dry-run.

    Returns:
        Dictionary with deletion statistics
    """
    from pathlib import Path
    import shutil

    cache_dir = Path("/cache/huggingface")
    if not cache_dir.exists():
        return {"deleted": 0, "freed_mb": 0, "dry_run": not force}

    deleted_count = 0
    freed_bytes = 0

    for item in cache_dir.rglob("*"):
        if pattern and not item.match(pattern):
            continue

        if item.is_file():
            size = item.stat().st_size
            if force:
                item.unlink()
            deleted_count += 1
            freed_bytes += size
        elif item.is_dir() and force:
            try:
                shutil.rmtree(item)
            except Exception as e:
                print(f"Failed to delete {item}: {e}")

    return {
        "deleted": deleted_count,
        "freed_mb": freed_bytes / (1024 * 1024),
        "dry_run": not force,
    }


def run_experiment_local(
    experiment_config: Dict[str, Any],
    experiment_class_path: str,
    gpu_type: str = "a10g",
    timeout: int = 3600 * 4,
) -> Dict[str, Any]:
    """Run an experiment on Modal from local machine.

    This is the main entry point for running experiments from your local machine.
    It handles Modal app deployment and execution.

    Args:
        experiment_config: Configuration dictionary for the experiment
        experiment_class_path: Python import path to the experiment class
        gpu_type: Type of GPU to use (t4, a10g, a100, a100-80gb, h100, l4)
        timeout: Timeout in seconds

    Returns:
        Dictionary containing experiment results
    """
    # Update the GPU type if specified
    if gpu_type not in GPU_CONFIGS:
        raise ValueError(
            f"Invalid GPU type: {gpu_type}. "
            f"Must be one of: {list(GPU_CONFIGS.keys())}"
        )

    # Deploy and run
    with app.run():
        # Call the remote function
        result = run_experiment.remote(
            experiment_config=experiment_config,
            experiment_class_path=experiment_class_path,
        )

        return result


def get_gpu_config(gpu_type: str) -> modal.gpu.GPU:
    """Get Modal GPU configuration by type.

    Args:
        gpu_type: Type of GPU (t4, a10g, a100, a100-80gb, h100, l4)

    Returns:
        Modal GPU configuration object

    Raises:
        ValueError: If gpu_type is not recognized
    """
    if gpu_type not in GPU_CONFIGS:
        raise ValueError(
            f"Invalid GPU type: {gpu_type}. "
            f"Must be one of: {list(GPU_CONFIGS.keys())}"
        )

    return GPU_CONFIGS[gpu_type]


# Convenience function for creating custom GPU functions
def create_gpu_function(
    gpu_type: str = "a10g",
    timeout: int = 3600,
    cpu: Optional[float] = None,
    memory: Optional[int] = None,
):
    """Create a custom GPU function decorator with specified resources.

    Args:
        gpu_type: Type of GPU to use
        timeout: Timeout in seconds
        cpu: Number of CPU cores (optional)
        memory: Memory in MB (optional)

    Returns:
        Modal function decorator
    """
    kwargs = {
        "image": gpu_image,
        "gpu": get_gpu_config(gpu_type),
        "volumes": {"/cache": hf_cache_volume},
        "timeout": timeout,
        "secrets": [
            modal.Secret.from_name("huggingface-secret", required=False),
            modal.Secret.from_name("wandb-secret", required=False),
            modal.Secret.from_name("aws-secret", required=False),
        ],
    }

    if cpu is not None:
        kwargs["cpu"] = cpu

    if memory is not None:
        kwargs["memory"] = memory

    return app.function(**kwargs)


# For backward compatibility and testing
if __name__ == "__main__":
    print("Modal app defined. Use modal deploy or modal run to deploy.")
    print(f"Available GPU types: {list(GPU_CONFIGS.keys())}")
