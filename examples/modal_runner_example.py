"""Example usage of Modal runner for ML experiments.

This demonstrates how to run experiments on Modal GPUs from your local machine.
"""

from ml_experiments.core.modal_runner import (
    run_experiment_local,
    app,
    list_cached_models,
    clear_cache,
)


def example_run_experiment():
    """Example: Run an experiment on Modal."""
    config = {
        "model_name": "CompVis/stable-diffusion-v1-4",
        "prompt": "A futuristic city at sunset, digital art",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "num_images": 4,
        "seed": 42,
    }

    print("🚀 Running experiment on Modal A10G GPU...")

    result = run_experiment_local(
        experiment_config=config,
        experiment_class_path="ml_experiments.tasks.text_to_image.TextToImageExperiment",
        gpu_type="a10g",
        timeout=3600,
    )

    if result["status"] == "success":
        print("✅ Experiment completed successfully!")
        print(f"Results: {result['results']}")
    else:
        print("❌ Experiment failed!")
        print(f"Error: {result['error']}")


def example_list_cached_models():
    """Example: List cached models in Modal volume."""
    print("📦 Listing cached models...")

    with app.run():
        models = list_cached_models.remote()

    print(f"Found {len(models)} cached models:")
    for model in models:
        print(f"  - {model}")


def example_clear_cache():
    """Example: Clear cache (dry-run)."""
    print("🧹 Checking cache size (dry-run)...")

    with app.run():
        stats = clear_cache.remote(force=False)

    print(f"Would delete {stats['deleted']} items")
    print(f"Would free {stats['freed_mb']:.2f} MB")


def example_custom_gpu():
    """Example: Run on different GPU types."""
    config = {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": "An astronaut riding a horse on Mars",
        "num_inference_steps": 30,
    }

    # Run on H100 for maximum performance
    print("🚀 Running on H100 GPU...")

    result = run_experiment_local(
        experiment_config=config,
        experiment_class_path="ml_experiments.tasks.text_to_image.TextToImageExperiment",
        gpu_type="h100",
        timeout=1800,
    )

    print(f"Status: {result['status']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python modal_runner_example.py <example>")
        print("\nAvailable examples:")
        print("  run        - Run an experiment on Modal")
        print("  list       - List cached models")
        print("  cache      - Check cache size")
        print("  gpu        - Run on different GPU types")
        sys.exit(1)

    example = sys.argv[1]

    if example == "run":
        example_run_experiment()
    elif example == "list":
        example_list_cached_models()
    elif example == "cache":
        example_clear_cache()
    elif example == "gpu":
        example_custom_gpu()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)
