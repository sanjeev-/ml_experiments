"""
Base Experiment class for ML experiments.

This module provides the foundational Experiment class that all task-specific
experiments should inherit from. It handles configuration loading, logging,
checkpointing, and metric tracking.
"""

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import torch
import yaml
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Configure rich console for structured logging
console = Console()


class ExperimentConfig(BaseModel):
    """Base configuration for all experiments.

    This Pydantic model defines the core configuration structure that all
    experiments must follow. Subclasses can extend this with task-specific
    configuration fields.

    Attributes:
        experiment_name: Unique identifier for the experiment
        experiment_type: Type of experiment (e.g., 'text_to_image', 'segmentation')
        output_dir: Local directory for outputs and checkpoints
        seed: Random seed for reproducibility
        device: Device to run on ('cuda', 'cpu', or 'auto')
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

        # Checkpointing
        checkpoint_dir: Directory for saving checkpoints
        checkpoint_frequency: Save checkpoint every N steps (0 = disabled)
        checkpoint_to_s3: Whether to upload checkpoints to S3
        s3_bucket: S3 bucket name for checkpoints
        s3_prefix: S3 key prefix for checkpoints

        # Metrics and logging
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        wandb_tags: List of tags for W&B run
        log_frequency: Log metrics every N steps

        # Additional metadata
        description: Human-readable experiment description
        tags: List of tags for organization
    """

    # Core settings
    experiment_name: str = Field(..., description="Unique experiment identifier")
    experiment_type: str = Field(..., description="Type of experiment")
    output_dir: str = Field(default="./outputs", description="Output directory path")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: str = Field(default="auto", description="Device: 'cuda', 'cpu', or 'auto'")
    log_level: str = Field(default="INFO", description="Logging level")

    # Checkpointing configuration
    checkpoint_dir: Optional[str] = Field(default=None, description="Checkpoint directory")
    checkpoint_frequency: int = Field(default=0, description="Checkpoint save frequency")
    checkpoint_to_s3: bool = Field(default=False, description="Upload checkpoints to S3")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for checkpoints")
    s3_prefix: Optional[str] = Field(default="checkpoints", description="S3 key prefix")

    # Metrics and logging
    use_wandb: bool = Field(default=False, description="Enable W&B logging")
    wandb_project: Optional[str] = Field(default=None, description="W&B project name")
    wandb_entity: Optional[str] = Field(default=None, description="W&B entity name")
    wandb_tags: List[str] = Field(default_factory=list, description="W&B tags")
    log_frequency: int = Field(default=10, description="Metric logging frequency")

    # Metadata
    description: str = Field(default="", description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Organization tags")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {v}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device is valid."""
        valid_devices = ["cuda", "cpu", "auto", "mps"]
        if v not in valid_devices and not v.startswith("cuda:"):
            raise ValueError(f"device must be one of {valid_devices} or 'cuda:N', got {v}")
        return v

    @field_validator("checkpoint_dir")
    @classmethod
    def set_checkpoint_dir(cls, v: Optional[str], info) -> Optional[str]:
        """Set default checkpoint dir if not provided."""
        if v is None and "output_dir" in info.data:
            return str(Path(info.data["output_dir"]) / "checkpoints")
        return v


class Experiment(ABC):
    """Abstract base class for all ML experiments.

    This class provides the infrastructure for running ML experiments including:
    - Configuration loading and validation
    - Structured logging with rich
    - Checkpointing to local storage and S3
    - Metric tracking with W&B integration
    - Device management and reproducibility

    Subclasses must implement the abstract methods for training, evaluation,
    and inference specific to their task.

    Example:
        ```python
        class MyExperiment(Experiment):
            def train(self):
                for step in range(self.config.num_steps):
                    loss = self._train_step()
                    self.log_metrics({"loss": loss}, step=step)

                    if step % self.config.checkpoint_frequency == 0:
                        self.save_checkpoint(step=step)

            def evaluate(self):
                metrics = self._compute_metrics()
                return metrics

            def inference(self, inputs):
                return self.model(inputs)
        ```

    Usage:
        ```python
        config = ExperimentConfig.from_yaml("config.yaml")
        experiment = MyExperiment(config)
        experiment.setup()
        experiment.train()
        metrics = experiment.evaluate()
        experiment.cleanup()
        ```
    """

    def __init__(self, config: Union[ExperimentConfig, Dict[str, Any], str, Path]):
        """Initialize the experiment.

        Args:
            config: Experiment configuration as ExperimentConfig, dict, or path to YAML file

        Raises:
            ValueError: If config is invalid or cannot be loaded
            FileNotFoundError: If config file path doesn't exist
        """
        # Load and validate configuration
        self.config = self._load_config(config)

        # Initialize core components
        self.logger: Optional[logging.Logger] = None
        self.device: torch.device = None
        self.s3_client: Optional[Any] = None
        self.wandb_run: Optional[Any] = None

        # Tracking
        self._step = 0
        self._metrics_history: Dict[str, List[float]] = {}
        self._is_setup = False

        # Setup logging first
        self._setup_logging()

        self.logger.info(f"Initialized experiment: {self.config.experiment_name}")
        self.logger.info(f"Experiment type: {self.config.experiment_type}")

    def _load_config(self, config: Union[ExperimentConfig, Dict, str, Path]) -> ExperimentConfig:
        """Load and validate experiment configuration.

        Args:
            config: Configuration as object, dict, or file path

        Returns:
            Validated ExperimentConfig object

        Raises:
            ValueError: If config is invalid
            FileNotFoundError: If config file doesn't exist
        """
        if isinstance(config, ExperimentConfig):
            return config

        if isinstance(config, dict):
            try:
                return ExperimentConfig(**config)
            except Exception as e:
                raise ValueError(f"Invalid configuration dict: {e}") from e

        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            try:
                with open(config_path, "r") as f:
                    config_dict = yaml.safe_load(f)
                return ExperimentConfig(**config_dict)
            except Exception as e:
                raise ValueError(f"Failed to load config from {config_path}: {e}") from e

        raise ValueError(f"Invalid config type: {type(config)}")

    def _setup_logging(self) -> None:
        """Configure structured logging with rich."""
        # Create logger
        self.logger = logging.getLogger(f"ml_experiments.{self.config.experiment_name}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Add rich handler for console output
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
        )
        rich_handler.setLevel(getattr(logging, self.config.log_level))
        formatter = logging.Formatter(
            "%(message)s",
            datefmt="[%X]"
        )
        rich_handler.setFormatter(formatter)
        self.logger.addHandler(rich_handler)

        # Add file handler
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"{self.config.experiment_name}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Logging to: {log_file}")

    def _setup_device(self) -> None:
        """Configure and validate compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        self.logger.info(f"Using device: {self.device}")

        if self.device.type == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name(self.device)}")
            self.logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.2f} GB"
            )

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        import random
        import numpy as np

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            # Enable deterministic operations (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.logger.info(f"Set random seed to: {self.config.seed}")

    def _setup_s3(self) -> None:
        """Initialize S3 client for checkpoint uploads."""
        if self.config.checkpoint_to_s3:
            if not self.config.s3_bucket:
                raise ValueError("s3_bucket must be specified when checkpoint_to_s3=True")

            try:
                self.s3_client = boto3.client("s3")
                self.logger.info(f"Initialized S3 client for bucket: {self.config.s3_bucket}")
            except Exception as e:
                self.logger.error(f"Failed to initialize S3 client: {e}")
                raise

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if self.config.use_wandb:
            if not WANDB_AVAILABLE:
                self.logger.warning("wandb not installed, disabling W&B logging")
                self.config.use_wandb = False
                return

            if not self.config.wandb_project:
                raise ValueError("wandb_project must be specified when use_wandb=True")

            try:
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=self.config.experiment_name,
                    config=self.config.model_dump(),
                    tags=self.config.wandb_tags,
                    reinit=True,
                )
                self.logger.info(f"Initialized W&B: {self.wandb_run.url}")
            except Exception as e:
                self.logger.error(f"Failed to initialize W&B: {e}")
                raise

    def setup(self) -> None:
        """Setup experiment infrastructure.

        This method should be called before running training, evaluation, or inference.
        It sets up the device, seeds, output directories, S3, and W&B.

        Raises:
            RuntimeError: If setup has already been called
        """
        if self._is_setup:
            raise RuntimeError("Experiment.setup() has already been called")

        console.print(Panel.fit(
            f"[bold blue]Setting up experiment: {self.config.experiment_name}[/bold blue]\n"
            f"Type: {self.config.experiment_type}\n"
            f"Output: {self.config.output_dir}",
            title="Experiment Setup",
            border_style="blue",
        ))

        # Setup core infrastructure
        self._setup_device()
        self._set_seed()

        # Create output directories
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.checkpoint_dir:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Checkpoint directory: {checkpoint_dir}")

        # Setup external services
        self._setup_s3()
        self._setup_wandb()

        # Save configuration to output directory
        config_path = output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.model_dump(), f, default_flow_style=False)
        self.logger.info(f"Saved config to: {config_path}")

        self._is_setup = True
        self.logger.info("Setup complete")

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to console, W&B, and internal history.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training/eval step (uses internal counter if None)
            commit: Whether to commit metrics to W&B (default: True)
        """
        if step is None:
            step = self._step
            self._step += 1

        # Convert tensors to Python scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            processed_metrics[key] = value

        # Update history
        for key, value in processed_metrics.items():
            if key not in self._metrics_history:
                self._metrics_history[key] = []
            self._metrics_history[key].append(value)

        # Log to console (respecting log frequency)
        if step % self.config.log_frequency == 0:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in processed_metrics.items()])
            self.logger.info(f"Step {step} | {metrics_str}")

        # Log to W&B
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.log(processed_metrics, step=step, commit=commit)

    def save_checkpoint(
        self,
        state_dict: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save experiment checkpoint locally and optionally to S3.

        Args:
            state_dict: State dictionary to save (model weights, optimizer state, etc.)
            step: Current step (for checkpoint naming)
            metadata: Additional metadata to include in checkpoint

        Returns:
            Path to saved checkpoint file

        Raises:
            RuntimeError: If experiment hasn't been setup
        """
        if not self._is_setup:
            raise RuntimeError("Must call setup() before saving checkpoints")

        if step is None:
            step = self._step

        # Prepare checkpoint data
        checkpoint = {
            "step": step,
            "config": self.config.model_dump(),
            "metrics_history": self._metrics_history,
            "timestamp": datetime.now().isoformat(),
        }

        if state_dict is not None:
            checkpoint["state_dict"] = state_dict

        if metadata is not None:
            checkpoint["metadata"] = metadata

        # Save locally
        checkpoint_dir = Path(self.config.checkpoint_dir or self.config.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = checkpoint_dir / checkpoint_name

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

        # Upload to S3 if configured
        if self.config.checkpoint_to_s3 and self.s3_client:
            try:
                s3_key = f"{self.config.s3_prefix}/{self.config.experiment_name}/{checkpoint_name}"
                self.s3_client.upload_file(
                    str(checkpoint_path),
                    self.config.s3_bucket,
                    s3_key,
                )
                self.logger.info(f"Uploaded checkpoint to S3: s3://{self.config.s3_bucket}/{s3_key}")
            except Exception as e:
                self.logger.error(f"Failed to upload checkpoint to S3: {e}")
                # Don't raise - local checkpoint still saved

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_state_dict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint from local file or S3.

        Args:
            checkpoint_path: Local path or S3 URI (s3://bucket/key)
            load_state_dict: Whether to return state_dict for loading into model

        Returns:
            Checkpoint dictionary containing config, metrics, state_dict, etc.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid
        """
        checkpoint_path = str(checkpoint_path)

        # Handle S3 URIs
        if checkpoint_path.startswith("s3://"):
            if not self.s3_client:
                self._setup_s3()

            # Parse S3 URI
            parts = checkpoint_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1]

            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                tmp_path = tmp_file.name

            try:
                self.s3_client.download_file(bucket, key, tmp_path)
                self.logger.info(f"Downloaded checkpoint from S3: {checkpoint_path}")
                checkpoint_path = tmp_path
            except Exception as e:
                self.logger.error(f"Failed to download checkpoint from S3: {e}")
                raise

        # Load checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")

            # Restore metrics history
            if "metrics_history" in checkpoint:
                self._metrics_history = checkpoint["metrics_history"]

            # Restore step
            if "step" in checkpoint:
                self._step = checkpoint["step"]

            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise ValueError(f"Invalid checkpoint file: {e}") from e

    def print_summary(self) -> None:
        """Print a formatted summary of the experiment configuration and status."""
        table = Table(title=f"Experiment: {self.config.experiment_name}", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")

        # Core settings
        table.add_row("Type", self.config.experiment_type)
        table.add_row("Output Dir", self.config.output_dir)
        table.add_row("Device", str(self.device) if self._is_setup else self.config.device)
        table.add_row("Seed", str(self.config.seed))

        # Checkpointing
        if self.config.checkpoint_frequency > 0:
            table.add_row("Checkpoint Frequency", str(self.config.checkpoint_frequency))
            if self.config.checkpoint_to_s3:
                table.add_row("S3 Bucket", self.config.s3_bucket or "N/A")

        # W&B
        if self.config.use_wandb:
            table.add_row("W&B Project", self.config.wandb_project or "N/A")

        # Status
        table.add_row("Current Step", str(self._step))
        table.add_row("Setup Complete", str(self._is_setup))

        console.print(table)

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all logged metrics.

        Returns:
            Dictionary mapping metric names to their statistics (mean, std, min, max, latest)
        """
        import numpy as np

        summary = {}
        for metric_name, values in self._metrics_history.items():
            if not values:
                continue

            summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "latest": float(values[-1]),
                "count": len(values),
            }

        return summary

    def cleanup(self) -> None:
        """Cleanup resources and finalize experiment.

        This should be called at the end of an experiment run to properly
        close W&B runs, save final state, etc.
        """
        self.logger.info("Cleaning up experiment...")

        # Close W&B run
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.finish()
            self.logger.info("Closed W&B run")

        # Save final metrics summary
        if self._metrics_history:
            summary = self.get_metrics_summary()
            summary_path = Path(self.config.output_dir) / "metrics_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Saved metrics summary to: {summary_path}")

        console.print("[bold green]Experiment complete![/bold green]")

    # Abstract methods that subclasses must implement

    @abstractmethod
    def train(self) -> None:
        """Execute training loop.

        Subclasses must implement this method with their specific training logic.
        Should use self.log_metrics() to track progress and self.save_checkpoint()
        for checkpointing.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Execute evaluation and return metrics.

        Subclasses must implement this method with their specific evaluation logic.

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def inference(self, *args, **kwargs) -> Any:
        """Execute inference on inputs.

        Subclasses must implement this method with their specific inference logic.

        Returns:
            Model outputs (format depends on task)
        """
        pass


class ExperimentFactory:
    """Factory for creating experiments from configuration files.

    This utility class helps instantiate the correct experiment subclass
    based on the experiment_type specified in the configuration.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, experiment_type: str):
        """Decorator to register an experiment class.

        Usage:
            ```python
            @ExperimentFactory.register("text_to_image")
            class TextToImageExperiment(Experiment):
                ...
            ```
        """
        def decorator(experiment_class: type) -> type:
            cls._registry[experiment_type] = experiment_class
            return experiment_class
        return decorator

    @classmethod
    def create(cls, config: Union[ExperimentConfig, Dict, str, Path]) -> Experiment:
        """Create an experiment instance from configuration.

        Args:
            config: Experiment configuration

        Returns:
            Instantiated experiment subclass

        Raises:
            ValueError: If experiment_type is not registered
        """
        # Load config to get experiment_type
        if isinstance(config, ExperimentConfig):
            config_obj = config
        elif isinstance(config, dict):
            config_obj = ExperimentConfig(**config)
        else:
            config_path = Path(config)
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            config_obj = ExperimentConfig(**config_dict)

        experiment_type = config_obj.experiment_type

        if experiment_type not in cls._registry:
            raise ValueError(
                f"Unknown experiment_type: {experiment_type}. "
                f"Registered types: {list(cls._registry.keys())}"
            )

        experiment_class = cls._registry[experiment_type]
        return experiment_class(config)

    @classmethod
    def list_types(cls) -> List[str]:
        """List all registered experiment types."""
        return list(cls._registry.keys())
