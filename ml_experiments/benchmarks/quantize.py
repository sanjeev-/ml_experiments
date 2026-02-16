"""
Quantization and optimization utilities using bitsandbytes and torch.compile.

This module provides wrappers for model quantization and compilation:
- bitsandbytes int8 and int4 quantization
- torch.compile optimization
- Mixed quantization strategies
- Performance comparison helpers
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from rich.console import Console

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console()


class QuantizationType(str, Enum):
    """Supported quantization types."""

    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"
    NONE = "none"


class CompileMode(str, Enum):
    """torch.compile optimization modes."""

    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        quantization_type: Type of quantization to apply
        compute_dtype: Data type for computation (fp16, bf16, fp32)
        device_map: Device mapping strategy ('auto', 'balanced', etc.)
        load_in_8bit: Load model in 8-bit mode (bitsandbytes)
        load_in_4bit: Load model in 4-bit mode (bitsandbytes)
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
        bnb_4bit_quant_type: Quantization type for 4-bit ('fp4' or 'nf4')
        bnb_4bit_use_double_quant: Use double quantization for 4-bit
    """
    quantization_type: QuantizationType = QuantizationType.NONE
    compute_dtype: Optional[torch.dtype] = None
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class CompileConfig:
    """Configuration for torch.compile optimization.

    Attributes:
        enabled: Whether to enable compilation
        mode: Compilation mode
        fullgraph: Whether to compile the full graph
        dynamic: Whether to use dynamic shapes
        backend: Backend to use ('inductor', 'aot_eager', etc.)
    """
    enabled: bool = True
    mode: CompileMode = CompileMode.DEFAULT
    fullgraph: bool = False
    dynamic: bool = False
    backend: str = "inductor"


class OptimizationResult(BaseModel):
    """Result of model optimization."""

    original_size_mb: float = Field(..., description="Original model size in MB")
    optimized_size_mb: float = Field(..., description="Optimized model size in MB")
    compression_ratio: float = Field(..., description="Compression ratio")
    quantization_type: str = Field(..., description="Type of quantization applied")
    compiled: bool = Field(default=False, description="Whether model is compiled")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def print_summary(self) -> None:
        """Print optimization summary."""
        console.print("\n[bold cyan]Optimization Summary[/bold cyan]")
        console.print(f"  Original Size: [yellow]{self.original_size_mb:.2f} MB[/yellow]")
        console.print(f"  Optimized Size: [green]{self.optimized_size_mb:.2f} MB[/green]")
        console.print(f"  Compression: [magenta]{self.compression_ratio:.2f}x[/magenta]")
        console.print(f"  Quantization: [blue]{self.quantization_type}[/blue]")
        console.print(f"  Compiled: [cyan]{self.compiled}[/cyan]")


class ModelOptimizer:
    """Unified interface for model quantization and compilation.

    This class provides a simple API for applying various optimization techniques:
    - bitsandbytes 8-bit and 4-bit quantization
    - torch.compile optimization
    - Mixed precision training
    - Memory-efficient loading

    Example:
        >>> optimizer = ModelOptimizer()
        >>> config = QuantizationConfig(quantization_type=QuantizationType.INT8)
        >>> optimized_model = optimizer.quantize(model, config)
        >>> compiled_model = optimizer.compile(optimized_model)
    """

    def __init__(self):
        """Initialize model optimizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB.

        Args:
            model: PyTorch model

        Returns:
            Model size in megabytes
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024**2)
        return total_size_mb

    def quantize_int8(
        self,
        model: nn.Module,
        device_map: str = "auto",
    ) -> nn.Module:
        """Apply 8-bit quantization using bitsandbytes.

        Args:
            model: Model to quantize
            device_map: Device mapping strategy

        Returns:
            Quantized model
        """
        if not BITSANDBYTES_AVAILABLE:
            logger.warning("bitsandbytes not available, returning original model")
            return model

        console.print("[cyan]Applying 8-bit quantization...[/cyan]")

        # Replace linear layers with 8-bit versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Create 8-bit linear layer
                int8_module = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                )

                # Copy weights
                int8_module.weight = bnb.nn.Int8Params(
                    module.weight.data,
                    requires_grad=False,
                    has_fp16_weights=False,
                )

                if module.bias is not None:
                    int8_module.bias = module.bias

                # Replace module
                setattr(parent, child_name, int8_module)

        console.print("[green]8-bit quantization applied successfully[/green]")
        return model

    def quantize_int4(
        self,
        model: nn.Module,
        compute_dtype: Optional[torch.dtype] = None,
        quant_type: str = "nf4",
        use_double_quant: bool = True,
    ) -> nn.Module:
        """Apply 4-bit quantization using bitsandbytes.

        Args:
            model: Model to quantize
            compute_dtype: Computation dtype (default: torch.float16)
            quant_type: Quantization type ('fp4' or 'nf4')
            use_double_quant: Whether to use nested quantization

        Returns:
            Quantized model
        """
        if not BITSANDBYTES_AVAILABLE:
            logger.warning("bitsandbytes not available, returning original model")
            return model

        console.print(f"[cyan]Applying 4-bit quantization (type: {quant_type})...[/cyan]")

        compute_dtype = compute_dtype or torch.float16

        # Replace linear layers with 4-bit versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Create 4-bit linear layer
                int4_module = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                    compress_statistics=use_double_quant,
                    quant_type=quant_type,
                )

                # Copy weights
                int4_module.weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    compress_statistics=use_double_quant,
                    quant_type=quant_type,
                )

                if module.bias is not None:
                    int4_module.bias = module.bias

                # Replace module
                setattr(parent, child_name, int4_module)

        console.print("[green]4-bit quantization applied successfully[/green]")
        return model

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
    ) -> tuple[nn.Module, OptimizationResult]:
        """Apply quantization based on configuration.

        Args:
            model: Model to quantize
            config: Quantization configuration

        Returns:
            Tuple of (quantized_model, optimization_result)
        """
        original_size = self._get_model_size(model)

        quantized_model = model

        if config.quantization_type == QuantizationType.INT8:
            quantized_model = self.quantize_int8(model, device_map=config.device_map)
        elif config.quantization_type == QuantizationType.INT4:
            quantized_model = self.quantize_int4(
                model,
                compute_dtype=config.bnb_4bit_compute_dtype,
                quant_type=config.bnb_4bit_quant_type,
                use_double_quant=config.bnb_4bit_use_double_quant,
            )
        elif config.quantization_type == QuantizationType.FP16:
            console.print("[cyan]Converting to FP16...[/cyan]")
            quantized_model = model.half()
        elif config.quantization_type == QuantizationType.BF16:
            console.print("[cyan]Converting to BF16...[/cyan]")
            quantized_model = model.bfloat16()

        optimized_size = self._get_model_size(quantized_model)
        compression_ratio = original_size / optimized_size if optimized_size > 0 else 1.0

        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=optimized_size,
            compression_ratio=compression_ratio,
            quantization_type=config.quantization_type.value,
            compiled=False,
        )

        return quantized_model, result

    def compile(
        self,
        model: nn.Module,
        config: Optional[CompileConfig] = None,
    ) -> nn.Module:
        """Compile model using torch.compile.

        Args:
            model: Model to compile
            config: Compilation configuration

        Returns:
            Compiled model
        """
        config = config or CompileConfig()

        if not config.enabled:
            logger.info("Compilation disabled, returning original model")
            return model

        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model

        console.print(
            f"[cyan]Compiling model with mode: {config.mode.value}, "
            f"backend: {config.backend}...[/cyan]"
        )

        try:
            compiled_model = torch.compile(
                model,
                mode=config.mode.value,
                fullgraph=config.fullgraph,
                dynamic=config.dynamic,
                backend=config.backend,
            )
            console.print("[green]Model compiled successfully[/green]")
            return compiled_model
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            console.print(f"[yellow]Warning: Compilation failed, using original model[/yellow]")
            return model

    def optimize(
        self,
        model: nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
        compile_config: Optional[CompileConfig] = None,
    ) -> tuple[nn.Module, OptimizationResult]:
        """Apply full optimization pipeline (quantization + compilation).

        Args:
            model: Model to optimize
            quant_config: Quantization configuration
            compile_config: Compilation configuration

        Returns:
            Tuple of (optimized_model, optimization_result)
        """
        console.print("\n[bold cyan]Starting Model Optimization[/bold cyan]")

        # Apply quantization
        if quant_config and quant_config.quantization_type != QuantizationType.NONE:
            model, result = self.quantize(model, quant_config)
        else:
            original_size = self._get_model_size(model)
            result = OptimizationResult(
                original_size_mb=original_size,
                optimized_size_mb=original_size,
                compression_ratio=1.0,
                quantization_type="none",
                compiled=False,
            )

        # Apply compilation
        if compile_config and compile_config.enabled:
            model = self.compile(model, compile_config)
            result.compiled = True

        result.print_summary()
        return model, result


def quantize_model(
    model: nn.Module,
    quantization_type: Union[str, QuantizationType] = "int8",
    **kwargs,
) -> nn.Module:
    """Quick helper to quantize a model.

    Args:
        model: Model to quantize
        quantization_type: Type of quantization ('int8', 'int4', 'fp16', 'bf16')
        **kwargs: Additional quantization parameters

    Returns:
        Quantized model
    """
    if isinstance(quantization_type, str):
        quantization_type = QuantizationType(quantization_type)

    config = QuantizationConfig(quantization_type=quantization_type, **kwargs)
    optimizer = ModelOptimizer()
    quantized_model, _ = optimizer.quantize(model, config)
    return quantized_model


def compile_model(
    model: nn.Module,
    mode: Union[str, CompileMode] = "default",
    **kwargs,
) -> nn.Module:
    """Quick helper to compile a model.

    Args:
        model: Model to compile
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        **kwargs: Additional compilation parameters

    Returns:
        Compiled model
    """
    if isinstance(mode, str):
        mode = CompileMode(mode)

    config = CompileConfig(mode=mode, **kwargs)
    optimizer = ModelOptimizer()
    return optimizer.compile(model, config)


def optimize_model(
    model: nn.Module,
    quantization: Optional[str] = None,
    compile_mode: Optional[str] = None,
    **kwargs,
) -> tuple[nn.Module, OptimizationResult]:
    """Quick helper to apply full optimization pipeline.

    Args:
        model: Model to optimize
        quantization: Quantization type ('int8', 'int4', 'fp16', 'bf16', None)
        compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune', None)
        **kwargs: Additional optimization parameters

    Returns:
        Tuple of (optimized_model, optimization_result)

    Example:
        >>> model = MyModel()
        >>> optimized, result = optimize_model(model, quantization='int8', compile_mode='default')
        >>> result.print_summary()
    """
    quant_config = None
    if quantization:
        quant_type = QuantizationType(quantization)
        quant_config = QuantizationConfig(quantization_type=quant_type)

    compile_config = None
    if compile_mode:
        mode = CompileMode(compile_mode)
        compile_config = CompileConfig(enabled=True, mode=mode)

    optimizer = ModelOptimizer()
    return optimizer.optimize(model, quant_config, compile_config)


class QuantizationComparison:
    """Compare different quantization strategies.

    This utility helps compare the trade-offs between different quantization
    approaches by measuring model size, memory usage, and inference latency.
    """

    def __init__(self):
        """Initialize comparison utility."""
        self.optimizer = ModelOptimizer()

    def compare_quantizations(
        self,
        model_fn: callable,
        quantization_types: list[QuantizationType],
    ) -> Dict[str, OptimizationResult]:
        """Compare different quantization types.

        Args:
            model_fn: Function that returns a fresh model instance
            quantization_types: List of quantization types to compare

        Returns:
            Dictionary mapping quantization type to optimization result
        """
        results = {}

        console.print("\n[bold cyan]Comparing Quantization Methods[/bold cyan]")

        for quant_type in quantization_types:
            console.print(f"\n[yellow]Testing {quant_type.value}...[/yellow]")

            # Get fresh model
            model = model_fn()

            # Apply quantization
            config = QuantizationConfig(quantization_type=quant_type)
            _, result = self.optimizer.quantize(model, config)

            results[quant_type.value] = result

        # Print comparison table
        self._print_comparison_table(results)

        return results

    def _print_comparison_table(self, results: Dict[str, OptimizationResult]) -> None:
        """Print comparison table.

        Args:
            results: Dictionary of quantization results
        """
        from rich.table import Table

        table = Table(
            title="Quantization Comparison",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Quantization", style="cyan")
        table.add_column("Size (MB)", justify="right", style="green")
        table.add_column("Compression", justify="right", style="yellow")

        for quant_type, result in results.items():
            table.add_row(
                quant_type,
                f"{result.optimized_size_mb:.2f}",
                f"{result.compression_ratio:.2f}x",
            )

        console.print("\n")
        console.print(table)
