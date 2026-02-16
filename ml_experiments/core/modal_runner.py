"""Modal integration for running experiments on GPUs."""

from typing import Any, Dict, Optional

import modal

# Create Modal app
app = modal.App("ml-experiments")

# Define GPU image with all dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "webdataset>=0.2.0",
        "boto3>=1.29.0",
        "pydantic>=2.5.0",
        "rich>=13.7.0",
        "bitsandbytes>=0.41.0",
        "torchmetrics>=1.2.0",
        "clean-fid>=0.1.35",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "lpips>=0.1.4",
    )
    .apt_install("git")
)

# Create volume for HuggingFace cache
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=gpu_image,
    gpu="A10G",  # Default GPU, can be overridden
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=3600,  # 1 hour default timeout
)
def run_experiment(
    experiment_class: str,
    config: Dict[str, Any],
    gpu_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Run an experiment on Modal GPU.

    Args:
        experiment_class: Fully qualified experiment class name (e.g., 'ml_experiments.tasks.text_to_image.TextToImage')
        config: Experiment configuration dictionary
        gpu_type: Optional GPU type override (A10G, A100, H100)

    Returns:
        Experiment results dictionary
    """
    import importlib
    from ml_experiments.core.experiment import ExperimentConfig

    # Import the experiment class dynamically
    module_name, class_name = experiment_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    ExperimentClass = getattr(module, class_name)

    # Create config and instantiate experiment
    exp_config = ExperimentConfig(**config)
    experiment = ExperimentClass(exp_config)

    # Run the experiment
    try:
        experiment.train()
        results = experiment.eval()
    finally:
        experiment.cleanup()

    return results


def run_experiment_local(
    experiment_class: str,
    config: Dict[str, Any],
    gpu_type: str = "A10G",
) -> Dict[str, Any]:
    """Run an experiment on Modal from local machine.

    Args:
        experiment_class: Fully qualified experiment class name
        config: Experiment configuration dictionary
        gpu_type: GPU type (A10G, A100, H100)

    Returns:
        Experiment results dictionary
    """
    # Update function with specified GPU type
    fn = run_experiment.with_options(gpu=gpu_type)
    return fn.remote(experiment_class, config, gpu_type)


def get_modal_app() -> modal.App:
    """Get the Modal app instance.

    Returns:
        Modal app instance
    """
    return app
