"""
Example demonstrating how to create a custom experiment using the base Experiment class.

This example shows:
1. Creating a custom experiment by inheriting from Experiment
2. Defining task-specific configuration
3. Implementing train, evaluate, and inference methods
4. Using logging, checkpointing, and metrics
"""

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from pydantic import Field

from ml_experiments.core import Experiment, ExperimentConfig, ExperimentFactory


class SimpleModelConfig(ExperimentConfig):
    """Configuration for the simple model experiment.

    Extends the base ExperimentConfig with task-specific parameters.
    """

    # Model hyperparameters
    input_dim: int = Field(default=10, description="Input dimension")
    hidden_dim: int = Field(default=64, description="Hidden layer dimension")
    output_dim: int = Field(default=1, description="Output dimension")

    # Training hyperparameters
    num_epochs: int = Field(default=10, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")


@ExperimentFactory.register("simple_model")
class SimpleModelExperiment(Experiment):
    """A simple feedforward neural network experiment for demonstration.

    This experiment trains a simple MLP on synthetic data to demonstrate
    the usage of the Experiment base class.
    """

    def __init__(self, config: SimpleModelConfig):
        """Initialize the experiment."""
        super().__init__(config)

        # Type hint for IDE support
        self.config: SimpleModelConfig = self.config

        # Model and optimizer (initialized in setup)
        self.model = None
        self.optimizer = None
        self.criterion = None

    def setup(self) -> None:
        """Setup the experiment (extends base setup)."""
        # Call parent setup
        super().setup()

        # Initialize model
        self.model = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        ).to(self.device)

        self.logger.info(f"Model: {self.model}")
        self.logger.info(
            f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.criterion = nn.MSELoss()

    def train(self) -> None:
        """Train the model on synthetic data."""
        self.logger.info("Starting training...")
        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = []

            # Simulate multiple batches
            num_batches = 100
            for batch_idx in range(num_batches):
                # Generate synthetic data
                x = torch.randn(self.config.batch_size, self.config.input_dim).to(
                    self.device
                )
                y = (x.sum(dim=1, keepdim=True) * 0.5).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

                # Calculate global step
                step = epoch * num_batches + batch_idx

                # Log metrics
                if step % self.config.log_frequency == 0:
                    self.log_metrics(
                        {
                            "loss": loss.item(),
                            "epoch": epoch,
                        },
                        step=step,
                    )

            # Log epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - Avg Loss: {avg_loss:.4f}"
            )

            # Save checkpoint
            if (
                self.config.checkpoint_frequency > 0
                and epoch % self.config.checkpoint_frequency == 0
            ):
                self.save_checkpoint(
                    state_dict={
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    step=epoch * num_batches,
                    metadata={"epoch": epoch, "avg_loss": avg_loss},
                )

        self.logger.info("Training complete!")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on synthetic validation data."""
        self.logger.info("Starting evaluation...")
        self.model.eval()

        total_loss = 0.0
        num_batches = 50

        with torch.no_grad():
            for _ in range(num_batches):
                # Generate synthetic validation data
                x = torch.randn(self.config.batch_size, self.config.input_dim).to(
                    self.device
                )
                y = (x.sum(dim=1, keepdim=True) * 0.5).to(self.device)

                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        metrics = {"eval_loss": avg_loss}

        self.logger.info(f"Evaluation Loss: {avg_loss:.4f}")

        # Log to W&B if enabled
        if self.config.use_wandb:
            self.log_metrics(metrics)

        return metrics

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on input tensor.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Model predictions of shape (batch_size, output_dim)
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
        return outputs


def main():
    """Main function demonstrating experiment usage."""
    # Create configuration
    config = SimpleModelConfig(
        experiment_name="simple_mlp_demo",
        experiment_type="simple_model",
        output_dir="./outputs/simple_mlp_demo",
        seed=42,
        device="auto",
        log_level="INFO",
        # Task-specific config
        input_dim=10,
        hidden_dim=64,
        output_dim=1,
        num_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        checkpoint_frequency=2,  # Save every 2 epochs
        log_frequency=20,
    )

    # Create experiment using factory
    experiment = ExperimentFactory.create(config)

    # Or create directly
    # experiment = SimpleModelExperiment(config)

    # Print experiment summary
    experiment.print_summary()

    # Setup experiment
    experiment.setup()

    # Train
    experiment.train()

    # Evaluate
    eval_metrics = experiment.evaluate()
    print(f"\nEvaluation metrics: {eval_metrics}")

    # Inference example
    test_input = torch.randn(5, config.input_dim)
    predictions = experiment.inference(test_input)
    print(f"\nInference on 5 samples:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions.squeeze().tolist()}")

    # Print metrics summary
    metrics_summary = experiment.get_metrics_summary()
    print(f"\nMetrics summary:")
    for metric_name, stats in metrics_summary.items():
        print(f"  {metric_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Min: {stats['min']:.4f}")
        print(f"    Max: {stats['max']:.4f}")

    # Cleanup
    experiment.cleanup()


if __name__ == "__main__":
    main()
