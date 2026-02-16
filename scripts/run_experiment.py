#!/usr/bin/env python3
"""
CLI entry point for running ML experiments locally or on Modal.

This script provides a unified interface for launching experiments with:
- YAML config loading and validation
- Local or Modal execution modes
- Rich console output and progress tracking
- GPU configuration options
- Experiment registry integration
"""

import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_experiments.core import Experiment, ExperimentFactory
from ml_experiments.core.modal_runner import app as modal_app, GPU_CONFIGS

console = Console()


def load_config(config_path: Path) -> dict:
    """Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        click.ClickException: If config is invalid
    """
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise click.ClickException("Config file is empty")

        if not isinstance(config, dict):
            raise click.ClickException("Config must be a YAML dictionary")

        # Validate required fields
        required_fields = ["experiment_name", "experiment_type"]
        missing_fields = [f for f in required_fields if f not in config]
        if missing_fields:
            raise click.ClickException(
                f"Missing required fields: {', '.join(missing_fields)}"
            )

        return config

    except yaml.YAMLError as e:
        raise click.ClickException(f"Failed to parse YAML: {e}")
    except Exception as e:
        raise click.ClickException(f"Failed to load config: {e}")


def display_config(config: dict) -> None:
    """Display experiment configuration in a formatted table.

    Args:
        config: Configuration dictionary
    """
    table = Table(title="Experiment Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", style="green")

    # Display key configuration items
    key_items = [
        ("Experiment Name", config.get("experiment_name", "N/A")),
        ("Experiment Type", config.get("experiment_type", "N/A")),
        ("Output Directory", config.get("output_dir", "./outputs")),
        ("Device", config.get("device", "auto")),
        ("Seed", config.get("seed", 42)),
        ("Log Level", config.get("log_level", "INFO")),
    ]

    for key, value in key_items:
        table.add_row(key, str(value))

    # Add checkpointing info if enabled
    if config.get("checkpoint_frequency", 0) > 0:
        table.add_row("Checkpoint Frequency", str(config["checkpoint_frequency"]))
        if config.get("checkpoint_to_s3"):
            table.add_row("S3 Bucket", config.get("s3_bucket", "N/A"))

    # Add W&B info if enabled
    if config.get("use_wandb"):
        table.add_row("W&B Project", config.get("wandb_project", "N/A"))

    console.print(table)


def run_local(config_path: Path, verbose: bool = False) -> None:
    """Run experiment locally.

    Args:
        config_path: Path to configuration file
        verbose: Enable verbose output
    """
    config = load_config(config_path)

    console.print(Panel.fit(
        "[bold green]Running Experiment Locally[/bold green]",
        border_style="green"
    ))

    if verbose:
        display_config(config)

    try:
        # Get experiment type and find the appropriate class
        experiment_type = config.get("experiment_type")

        # Try to get experiment class from factory
        experiment_class = None
        try:
            experiment_class = ExperimentFactory.get(experiment_type)
        except KeyError:
            # Try to import from tasks module
            try:
                module_name = f"ml_experiments.tasks.{experiment_type}"
                class_name = "".join([word.capitalize() for word in experiment_type.split("_")]) + "Experiment"

                import importlib
                module = importlib.import_module(module_name)
                experiment_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise click.ClickException(
                    f"Could not find experiment class for type '{experiment_type}'. "
                    f"Available types: {', '.join(ExperimentFactory.list())}"
                )

        # Instantiate and run experiment
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing experiment...", total=None)

            # Create experiment instance
            experiment = experiment_class(config=config)

            progress.update(task, description="Running experiment...")

            # Run the experiment
            results = experiment.run()

            progress.update(task, description="Experiment completed!", completed=True)

        # Display results
        console.print("\n[bold green]✓ Experiment completed successfully![/bold green]\n")

        if results:
            console.print("[bold]Results:[/bold]")
            results_table = Table(show_header=True, header_style="bold cyan")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")

            for key, value in results.items():
                if isinstance(value, (int, float, str, bool)):
                    results_table.add_row(key, str(value))

            console.print(results_table)

    except Exception as e:
        console.print(f"\n[bold red]✗ Experiment failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def run_modal(
    config_path: Path,
    gpu: str = "a10g",
    timeout: int = 3600,
    verbose: bool = False
) -> None:
    """Run experiment on Modal with GPU acceleration.

    Args:
        config_path: Path to configuration file
        gpu: GPU type (t4, a10g, a100, a100-80gb, h100, l4)
        timeout: Timeout in seconds
        verbose: Enable verbose output
    """
    config = load_config(config_path)

    console.print(Panel.fit(
        f"[bold blue]Running Experiment on Modal ({gpu.upper()})[/bold blue]",
        border_style="blue"
    ))

    if verbose:
        display_config(config)

    # Validate GPU type
    if gpu not in GPU_CONFIGS:
        raise click.ClickException(
            f"Invalid GPU type: {gpu}. Available: {', '.join(GPU_CONFIGS.keys())}"
        )

    try:
        # Import Modal runner
        from ml_experiments.core.modal_runner import run_experiment

        # Determine experiment class path
        experiment_type = config.get("experiment_type")

        # Build class path
        class_name = "".join([word.capitalize() for word in experiment_type.split("_")]) + "Experiment"
        experiment_class_path = f"ml_experiments.tasks.{experiment_type}.{class_name}"

        console.print(f"\n[cyan]Launching on Modal with {gpu.upper()} GPU...[/cyan]")
        console.print(f"[dim]Experiment class: {experiment_class_path}[/dim]\n")

        # Create a Modal function with the specified GPU
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Submitting to Modal...", total=None)

            # Run on Modal
            result = run_experiment.remote(
                experiment_config=config,
                experiment_class_path=experiment_class_path,
            )

            progress.update(task, description="Experiment completed!", completed=True)

        # Display results
        if result.get("status") == "success":
            console.print("\n[bold green]✓ Experiment completed successfully on Modal![/bold green]\n")

            if result.get("results"):
                console.print("[bold]Results:[/bold]")
                results_table = Table(show_header=True, header_style="bold cyan")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Value", style="green")

                for key, value in result["results"].items():
                    if isinstance(value, (int, float, str, bool)):
                        results_table.add_row(key, str(value))

                console.print(results_table)
        else:
            console.print(f"\n[bold red]✗ Experiment failed on Modal[/bold red]")
            console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            if verbose and result.get("traceback"):
                console.print("\n[bold]Traceback:[/bold]")
                console.print(Syntax(result["traceback"], "python", theme="monokai"))
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to run on Modal: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--mode",
    type=click.Choice(["local", "modal"], case_sensitive=False),
    default="local",
    help="Execution mode: local or Modal GPU cluster"
)
@click.option(
    "--gpu",
    type=click.Choice(["t4", "a10g", "a100", "a100-80gb", "h100", "l4"], case_sensitive=False),
    default="a10g",
    help="GPU type for Modal execution (ignored in local mode)"
)
@click.option(
    "--timeout",
    type=int,
    default=3600,
    help="Timeout in seconds for Modal execution"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--list-experiments",
    is_flag=True,
    help="List available experiment types"
)
def main(
    config: Path,
    mode: str,
    gpu: str,
    timeout: int,
    verbose: bool,
    list_experiments: bool
) -> None:
    """Run ML experiments locally or on Modal GPU cluster.

    CONFIG: Path to YAML configuration file

    Examples:

        # Run locally
        ml-experiment configs/example_experiment.yaml

        # Run on Modal with A100 GPU
        ml-experiment configs/example_experiment.yaml --mode modal --gpu a100

        # List available experiments
        ml-experiment --list-experiments
    """
    # Handle list experiments flag
    if list_experiments:
        console.print("[bold]Available Experiment Types:[/bold]\n")

        experiments = ExperimentFactory.list()

        if experiments:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Experiment Type", style="cyan")
            table.add_column("Description", style="green")

            for exp_type in sorted(experiments):
                table.add_row(exp_type, f"{exp_type.replace('_', ' ').title()} experiment")

            console.print(table)
        else:
            console.print("[yellow]No experiments registered. Available task types:[/yellow]")
            console.print("  • text_to_image")
            console.print("  • matting")
            console.print("  • segmentation")
            console.print("  • synthetic_data")
            console.print("  • vae")
            console.print("  • text_encoder")

        return

    # Display banner
    console.print(Panel.fit(
        "[bold cyan]ML Experiments Framework[/bold cyan]\n"
        "[dim]Modular GPU-accelerated ML experiments[/dim]",
        border_style="cyan"
    ))

    # Run experiment based on mode
    if mode == "local":
        run_local(config, verbose)
    else:
        run_modal(config, gpu, timeout, verbose)


if __name__ == "__main__":
    main()
