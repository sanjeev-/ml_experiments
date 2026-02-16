"""Base experiment class with config management, logging, and checkpointing."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import torch
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentConfig(BaseModel):
    """Base configuration for experiments."""

    name: str = Field(..., description="Experiment name")
    seed: int = Field(default=42, description="Random seed")
    device: str = Field(default="cuda", description="Device to use")
    checkpoint_dir: str = Field(default="./checkpoints", description="Checkpoint directory")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for checkpoints")
    s3_prefix: Optional[str] = Field(default=None, description="S3 prefix for checkpoints")
    wandb_project: Optional[str] = Field(default=None, description="W&B project name")
    wandb_entity: Optional[str] = Field(default=None, description="W&B entity")
    log_level: str = Field(default="INFO", description="Logging level")


class Experiment(ABC):
    """Base class for ML experiments with structured logging and checkpointing."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.console = Console()
        self.metrics: Dict[str, Any] = {}
        self._setup_logging()
        self._setup_seed()
        self._setup_wandb()

    def _setup_logging(self) -> None:
        """Set up structured logging with rich."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(message)s",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _setup_wandb(self) -> None:
        """Initialize W&B if configured."""
        if WANDB_AVAILABLE and self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.name,
                config=self.config.model_dump()
            )
            self.logger.info(f"Initialized W&B project: {self.config.wandb_project}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Experiment":
        """Load experiment from YAML configuration.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Initialized experiment instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = ExperimentConfig(**config_dict)
        return cls(config)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to console and W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        self.metrics.update(metrics)

        # Log to console with rich
        self.console.log(f"[bold green]Metrics (step {step}):[/bold green]")
        for key, value in metrics.items():
            self.console.log(f"  {key}: {value}")

        # Log to W&B if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=step)

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        filename: str,
        upload_to_s3: bool = True
    ) -> Path:
        """Save checkpoint locally and optionally to S3.

        Args:
            state_dict: State dictionary to save
            filename: Checkpoint filename
            upload_to_s3: Whether to upload to S3

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / filename
        torch.save(state_dict, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        if upload_to_s3 and self.config.s3_bucket:
            self._upload_to_s3(checkpoint_path)

        return checkpoint_path

    def _upload_to_s3(self, local_path: Path) -> None:
        """Upload checkpoint to S3.

        Args:
            local_path: Local path to checkpoint
        """
        if not self.config.s3_bucket:
            return

        s3_client = boto3.client('s3')
        s3_key = f"{self.config.s3_prefix}/{local_path.name}" if self.config.s3_prefix else local_path.name

        try:
            s3_client.upload_file(
                str(local_path),
                self.config.s3_bucket,
                s3_key
            )
            self.logger.info(f"Uploaded checkpoint to s3://{self.config.s3_bucket}/{s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to upload to S3: {e}")

    def load_checkpoint(self, filename: str, from_s3: bool = False) -> Dict[str, Any]:
        """Load checkpoint from local or S3.

        Args:
            filename: Checkpoint filename
            from_s3: Whether to download from S3

        Returns:
            Loaded state dictionary
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        if from_s3 and self.config.s3_bucket:
            self._download_from_s3(filename)

        state_dict = torch.load(checkpoint_path, map_location=self.config.device)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return state_dict

    def _download_from_s3(self, filename: str) -> None:
        """Download checkpoint from S3.

        Args:
            filename: Checkpoint filename
        """
        if not self.config.s3_bucket:
            return

        s3_client = boto3.client('s3')
        s3_key = f"{self.config.s3_prefix}/{filename}" if self.config.s3_prefix else filename
        local_path = Path(self.config.checkpoint_dir) / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3_client.download_file(
                self.config.s3_bucket,
                s3_key,
                str(local_path)
            )
            self.logger.info(f"Downloaded checkpoint from s3://{self.config.s3_bucket}/{s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to download from S3: {e}")
            raise

    @abstractmethod
    def train(self) -> None:
        """Train the model. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> Dict[str, Any]:
        """Evaluate the model. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def inference(self, *args, **kwargs) -> Any:
        """Run inference. Must be implemented by subclasses."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Cleanup resources."""
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
