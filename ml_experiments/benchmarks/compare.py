"""
Model comparison utilities using Rich tables for side-by-side metric analysis.

This module provides tools for comparing multiple models across various metrics:
- Side-by-side performance comparison
- Rich table visualization
- Statistical significance testing
- Export to various formats (JSON, CSV, markdown)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


class ModelMetrics(BaseModel):
    """Metrics for a single model.

    This class stores comprehensive metrics for model comparison including
    latency, throughput, memory usage, and quality metrics.
    """

    model_name: str = Field(..., description="Model name or identifier")

    # Latency metrics (milliseconds)
    latency_mean: Optional[float] = Field(default=None, description="Mean latency in ms")
    latency_std: Optional[float] = Field(default=None, description="Std dev latency in ms")
    latency_p50: Optional[float] = Field(default=None, description="P50 latency in ms")
    latency_p95: Optional[float] = Field(default=None, description="P95 latency in ms")
    latency_p99: Optional[float] = Field(default=None, description="P99 latency in ms")

    # Throughput metrics
    throughput: Optional[float] = Field(default=None, description="Samples per second")
    tokens_per_second: Optional[float] = Field(default=None, description="Tokens per second")

    # Memory metrics (MB)
    memory_allocated: Optional[float] = Field(default=None, description="Allocated memory in MB")
    memory_peak: Optional[float] = Field(default=None, description="Peak memory usage in MB")
    model_size: Optional[float] = Field(default=None, description="Model size in MB")

    # Quality metrics
    accuracy: Optional[float] = Field(default=None, description="Accuracy score")
    fid_score: Optional[float] = Field(default=None, description="FID score")
    lpips_score: Optional[float] = Field(default=None, description="LPIPS score")
    clip_score: Optional[float] = Field(default=None, description="CLIP score")

    # Additional metadata
    quantization: Optional[str] = Field(default=None, description="Quantization type")
    compiled: bool = Field(default=False, description="Whether model is compiled")
    batch_size: Optional[int] = Field(default=None, description="Batch size used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)


@dataclass
class ComparisonConfig:
    """Configuration for model comparison.

    Attributes:
        show_latency: Whether to show latency metrics
        show_throughput: Whether to show throughput metrics
        show_memory: Whether to show memory metrics
        show_quality: Whether to show quality metrics
        show_metadata: Whether to show metadata columns
        highlight_best: Whether to highlight best values
        sort_by: Column to sort by (None for no sorting)
        ascending: Sort order (True for ascending, False for descending)
    """
    show_latency: bool = True
    show_throughput: bool = True
    show_memory: bool = True
    show_quality: bool = True
    show_metadata: bool = True
    highlight_best: bool = True
    sort_by: Optional[str] = None
    ascending: bool = True


class ModelComparison:
    """Utility for comparing multiple models side-by-side.

    This class provides comprehensive model comparison capabilities with:
    - Rich table visualization
    - Flexible metric selection
    - Best value highlighting
    - Export to multiple formats
    - Statistical analysis

    Example:
        >>> comparison = ModelComparison()
        >>> comparison.add_model("Model A", metrics_a)
        >>> comparison.add_model("Model B", metrics_b)
        >>> comparison.print_table()
        >>> comparison.export_json("comparison.json")
    """

    def __init__(self, config: Optional[ComparisonConfig] = None):
        """Initialize model comparison.

        Args:
            config: Comparison configuration
        """
        self.config = config or ComparisonConfig()
        self.models: List[ModelMetrics] = []

    def add_model(
        self,
        model_name: str,
        metrics: Union[ModelMetrics, Dict[str, Any]],
    ) -> None:
        """Add a model to the comparison.

        Args:
            model_name: Name of the model
            metrics: Model metrics (ModelMetrics object or dict)
        """
        if isinstance(metrics, dict):
            metrics = ModelMetrics(model_name=model_name, **metrics)
        else:
            metrics.model_name = model_name

        self.models.append(metrics)
        logger.info(f"Added model '{model_name}' to comparison")

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the comparison.

        Args:
            model_name: Name of the model to remove

        Returns:
            True if model was removed, False if not found
        """
        for i, model in enumerate(self.models):
            if model.model_name == model_name:
                self.models.pop(i)
                logger.info(f"Removed model '{model_name}' from comparison")
                return True
        return False

    def _get_best_value(
        self,
        metric_name: str,
        values: List[Optional[float]],
        lower_is_better: bool = False,
    ) -> Optional[float]:
        """Get the best value for a metric.

        Args:
            metric_name: Name of the metric
            values: List of metric values
            lower_is_better: Whether lower values are better

        Returns:
            Best value or None if all values are None
        """
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None

        if lower_is_better:
            return min(valid_values)
        else:
            return max(valid_values)

    def _format_cell(
        self,
        value: Optional[float],
        best_value: Optional[float],
        format_str: str = ".2f",
        suffix: str = "",
        highlight: bool = True,
    ) -> Text:
        """Format a table cell with optional highlighting.

        Args:
            value: Cell value
            best_value: Best value for this metric
            format_str: Format string for the value
            suffix: Suffix to append (e.g., 'ms', 'MB')
            highlight: Whether to highlight best values

        Returns:
            Formatted Rich Text object
        """
        if value is None:
            return Text("N/A", style="dim")

        formatted = f"{value:{format_str}}{suffix}"

        # Highlight best value
        if highlight and best_value is not None and abs(value - best_value) < 1e-6:
            return Text(formatted, style="bold green")
        else:
            return Text(formatted)

    def print_table(
        self,
        title: str = "Model Comparison",
        config: Optional[ComparisonConfig] = None,
    ) -> None:
        """Print a formatted comparison table.

        Args:
            title: Table title
            config: Optional override for comparison configuration
        """
        config = config or self.config

        if not self.models:
            console.print("[yellow]No models to compare[/yellow]")
            return

        # Sort models if requested
        models = self.models
        if config.sort_by:
            models = sorted(
                models,
                key=lambda m: getattr(m, config.sort_by) or float('inf'),
                reverse=not config.ascending,
            )

        # Create table
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )

        # Add model name column
        table.add_column("Model", style="cyan", no_wrap=True)

        # Collect all metric values for finding best
        metric_values = {}

        # Add latency columns
        if config.show_latency:
            for metric in ["latency_mean", "latency_p50", "latency_p95", "latency_p99"]:
                metric_values[metric] = [getattr(m, metric) for m in models]

            table.add_column("Mean Latency\n(ms)", justify="right")
            table.add_column("P50\n(ms)", justify="right")
            table.add_column("P95\n(ms)", justify="right")
            table.add_column("P99\n(ms)", justify="right")

        # Add throughput columns
        if config.show_throughput:
            metric_values["throughput"] = [m.throughput for m in models]
            table.add_column("Throughput\n(samples/s)", justify="right")

            if any(m.tokens_per_second is not None for m in models):
                metric_values["tokens_per_second"] = [m.tokens_per_second for m in models]
                table.add_column("Tokens/s", justify="right")

        # Add memory columns
        if config.show_memory:
            for metric in ["memory_peak", "model_size"]:
                metric_values[metric] = [getattr(m, metric) for m in models]

            table.add_column("Peak Memory\n(MB)", justify="right")
            table.add_column("Model Size\n(MB)", justify="right")

        # Add quality columns
        if config.show_quality:
            for metric in ["accuracy", "fid_score", "clip_score"]:
                if any(getattr(m, metric) is not None for m in models):
                    metric_values[metric] = [getattr(m, metric) for m in models]
                    label = metric.replace("_", " ").title()
                    table.add_column(label, justify="right")

        # Add metadata columns
        if config.show_metadata:
            table.add_column("Quantization", justify="center")
            table.add_column("Compiled", justify="center")

        # Calculate best values
        best_values = {}
        if config.highlight_best:
            # Lower is better for latency, memory, FID
            lower_is_better_metrics = [
                "latency_mean", "latency_p50", "latency_p95", "latency_p99",
                "memory_peak", "model_size", "fid_score"
            ]

            for metric, values in metric_values.items():
                lower_is_better = metric in lower_is_better_metrics
                best_values[metric] = self._get_best_value(metric, values, lower_is_better)

        # Add rows
        for model in models:
            row = [model.model_name]

            # Latency metrics
            if config.show_latency:
                for metric in ["latency_mean", "latency_p50", "latency_p95", "latency_p99"]:
                    value = getattr(model, metric)
                    row.append(
                        self._format_cell(
                            value,
                            best_values.get(metric),
                            format_str=".2f",
                            highlight=config.highlight_best,
                        )
                    )

            # Throughput metrics
            if config.show_throughput:
                row.append(
                    self._format_cell(
                        model.throughput,
                        best_values.get("throughput"),
                        format_str=".2f",
                        highlight=config.highlight_best,
                    )
                )

                if any(m.tokens_per_second is not None for m in models):
                    row.append(
                        self._format_cell(
                            model.tokens_per_second,
                            best_values.get("tokens_per_second"),
                            format_str=".2f",
                            highlight=config.highlight_best,
                        )
                    )

            # Memory metrics
            if config.show_memory:
                for metric in ["memory_peak", "model_size"]:
                    value = getattr(model, metric)
                    row.append(
                        self._format_cell(
                            value,
                            best_values.get(metric),
                            format_str=".2f",
                            highlight=config.highlight_best,
                        )
                    )

            # Quality metrics
            if config.show_quality:
                for metric in ["accuracy", "fid_score", "clip_score"]:
                    if any(getattr(m, metric) is not None for m in models):
                        value = getattr(model, metric)
                        row.append(
                            self._format_cell(
                                value,
                                best_values.get(metric),
                                format_str=".4f",
                                highlight=config.highlight_best,
                            )
                        )

            # Metadata
            if config.show_metadata:
                row.append(Text(model.quantization or "none", style="blue"))
                row.append(Text("✓" if model.compiled else "✗", style="green" if model.compiled else "dim"))

            table.add_row(*row)

        console.print("\n")
        console.print(table)

    def print_summary(self) -> None:
        """Print a summary of all models in the comparison."""
        if not self.models:
            console.print("[yellow]No models to summarize[/yellow]")
            return

        console.print("\n[bold cyan]Comparison Summary[/bold cyan]")
        console.print(f"Total models: {len(self.models)}")

        # Find overall best performers
        all_latencies = [
            m.latency_mean for m in self.models if m.latency_mean is not None
        ]
        all_throughputs = [
            m.throughput for m in self.models if m.throughput is not None
        ]
        all_memory = [
            m.memory_peak for m in self.models if m.memory_peak is not None
        ]

        if all_latencies:
            best_latency_model = min(
                [m for m in self.models if m.latency_mean is not None],
                key=lambda m: m.latency_mean,
            )
            console.print(
                f"[green]Fastest model:[/green] {best_latency_model.model_name} "
                f"({best_latency_model.latency_mean:.2f} ms)"
            )

        if all_throughputs:
            best_throughput_model = max(
                [m for m in self.models if m.throughput is not None],
                key=lambda m: m.throughput,
            )
            console.print(
                f"[green]Highest throughput:[/green] {best_throughput_model.model_name} "
                f"({best_throughput_model.throughput:.2f} samples/s)"
            )

        if all_memory:
            best_memory_model = min(
                [m for m in self.models if m.memory_peak is not None],
                key=lambda m: m.memory_peak,
            )
            console.print(
                f"[green]Most memory efficient:[/green] {best_memory_model.model_name} "
                f"({best_memory_model.memory_peak:.2f} MB)"
            )

    def export_json(self, output_path: Union[str, Path]) -> None:
        """Export comparison to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)

        data = {
            "models": [model.to_dict() for model in self.models],
            "config": {
                "show_latency": self.config.show_latency,
                "show_throughput": self.config.show_throughput,
                "show_memory": self.config.show_memory,
                "show_quality": self.config.show_quality,
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"[green]Exported comparison to {output_path}[/green]")

    def export_csv(self, output_path: Union[str, Path]) -> None:
        """Export comparison to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        output_path = Path(output_path)

        if not self.models:
            console.print("[yellow]No models to export[/yellow]")
            return

        # Get all field names from the first model
        fieldnames = ["model_name"]
        sample_dict = self.models[0].to_dict()
        fieldnames.extend([k for k in sample_dict.keys() if k != "model_name"])

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for model in self.models:
                row = model.to_dict()
                # Convert metadata dict to string
                if "metadata" in row:
                    row["metadata"] = json.dumps(row["metadata"])
                writer.writerow(row)

        console.print(f"[green]Exported comparison to {output_path}[/green]")

    def export_markdown(self, output_path: Union[str, Path]) -> None:
        """Export comparison to Markdown table.

        Args:
            output_path: Path to output Markdown file
        """
        output_path = Path(output_path)

        if not self.models:
            console.print("[yellow]No models to export[/yellow]")
            return

        lines = ["# Model Comparison\n"]

        # Create table header
        headers = ["Model", "Mean Latency (ms)", "P95 (ms)", "Throughput (samples/s)",
                   "Peak Memory (MB)", "Quantization"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Add rows
        for model in self.models:
            row = [
                model.model_name,
                f"{model.latency_mean:.2f}" if model.latency_mean else "N/A",
                f"{model.latency_p95:.2f}" if model.latency_p95 else "N/A",
                f"{model.throughput:.2f}" if model.throughput else "N/A",
                f"{model.memory_peak:.2f}" if model.memory_peak else "N/A",
                model.quantization or "none",
            ]
            lines.append("| " + " | ".join(row) + " |")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        console.print(f"[green]Exported comparison to {output_path}[/green]")

    def get_winner(self, metric: str, lower_is_better: bool = False) -> Optional[ModelMetrics]:
        """Get the best model for a specific metric.

        Args:
            metric: Metric name (e.g., 'latency_mean', 'throughput')
            lower_is_better: Whether lower values are better

        Returns:
            ModelMetrics object of the best model, or None if metric not available
        """
        valid_models = [m for m in self.models if getattr(m, metric, None) is not None]

        if not valid_models:
            return None

        if lower_is_better:
            return min(valid_models, key=lambda m: getattr(m, metric))
        else:
            return max(valid_models, key=lambda m: getattr(m, metric))


def compare_models(
    models: Dict[str, Dict[str, Any]],
    title: str = "Model Comparison",
    config: Optional[ComparisonConfig] = None,
) -> ModelComparison:
    """Quick helper to compare multiple models.

    Args:
        models: Dictionary mapping model_name -> metrics dict
        title: Table title
        config: Comparison configuration

    Returns:
        ModelComparison object

    Example:
        >>> models = {
        ...     "Model A": {"latency_mean": 10.5, "throughput": 95.2},
        ...     "Model B": {"latency_mean": 8.3, "throughput": 120.4},
        ... }
        >>> comparison = compare_models(models)
        >>> comparison.print_table()
    """
    comparison = ModelComparison(config=config)

    for model_name, metrics in models.items():
        comparison.add_model(model_name, metrics)

    comparison.print_table(title=title)
    return comparison
