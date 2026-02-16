"""
Visualization utilities for image grids and tensorboard logging.

This module provides utilities for creating image grids, plotting results,
and logging to tensorboard for ML experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def make_image_grid(
    images: Union[List[Image.Image], List[np.ndarray], torch.Tensor],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> Image.Image:
    """
    Create a grid of images for visualization.

    Args:
        images: List of PIL Images, numpy arrays, or torch tensor
        nrow: Number of images per row
        padding: Padding between images
        normalize: If True, normalize images to [0, 1]
        value_range: Tuple (min, max) for normalization
        scale_each: If True, normalize each image independently
        pad_value: Value for padding pixels

    Returns:
        PIL Image containing the grid

    Examples:
        >>> images = [Image.open(f"img_{i}.png") for i in range(16)]
        >>> grid = make_image_grid(images, nrow=4)
        >>> grid.save("grid.png")
    """
    # Convert inputs to tensor
    if isinstance(images, torch.Tensor):
        tensor = images
    elif isinstance(images, list):
        if len(images) == 0:
            raise ValueError("Empty image list")

        # Convert list to tensor
        if isinstance(images[0], Image.Image):
            # PIL Images
            arrays = [np.array(img) for img in images]
            tensor = torch.from_numpy(np.stack(arrays))
            # Convert from HWC to CHW
            if tensor.ndim == 4:  # Batch of images
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor.float() / 255.0
        elif isinstance(images[0], np.ndarray):
            # Numpy arrays
            tensor = torch.from_numpy(np.stack(images))
            if tensor.ndim == 4 and tensor.shape[-1] in [1, 3, 4]:
                # Assume HWC format
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise TypeError(f"Unsupported image type: {type(images[0])}")
    else:
        raise TypeError(f"Unsupported images type: {type(images)}")

    # Ensure tensor is float
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    # Create grid using torchvision
    grid = torchvision.utils.make_grid(
        tensor,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        pad_value=pad_value,
    )

    # Convert to PIL Image
    grid_np = grid.cpu().numpy()
    grid_np = np.transpose(grid_np, (1, 2, 0))  # CHW to HWC

    # Convert to uint8
    if grid_np.max() <= 1.0:
        grid_np = (grid_np * 255).astype(np.uint8)
    else:
        grid_np = grid_np.astype(np.uint8)

    # Handle grayscale
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)

    return Image.fromarray(grid_np)


def save_image_grid(
    images: Union[List[Image.Image], List[np.ndarray], torch.Tensor],
    save_path: Union[str, Path],
    nrow: int = 8,
    **kwargs,
) -> None:
    """
    Create and save an image grid.

    Args:
        images: List of images or tensor
        save_path: Path to save the grid
        nrow: Number of images per row
        **kwargs: Additional arguments for make_image_grid

    Examples:
        >>> images = [torch.randn(3, 64, 64) for _ in range(16)]
        >>> save_image_grid(images, "output_grid.png", nrow=4)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_image_grid(images, nrow=nrow, **kwargs)
    grid.save(save_path)

    logger.info(f"Saved image grid to {save_path}")


def plot_images_with_captions(
    images: List[Union[Image.Image, np.ndarray]],
    captions: Optional[List[str]] = None,
    nrow: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot images with optional captions using matplotlib.

    Args:
        images: List of PIL Images or numpy arrays
        captions: Optional list of captions for each image
        nrow: Number of images per row
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure

    Examples:
        >>> images = [Image.open(f"img_{i}.png") for i in range(8)]
        >>> captions = [f"Image {i}" for i in range(8)]
        >>> fig = plot_images_with_captions(images, captions, nrow=4)
        >>> plt.show()
    """
    n_images = len(images)
    ncol = min(nrow, n_images)
    nrow_actual = (n_images + ncol - 1) // ncol

    if figsize is None:
        figsize = (ncol * 3, nrow_actual * 3)

    fig, axes = plt.subplots(nrow_actual, ncol, figsize=figsize)

    # Handle single image case
    if n_images == 1:
        axes = np.array([[axes]])
    elif nrow_actual == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(nrow_actual * ncol):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        if idx < n_images:
            img = images[idx]

            # Convert to numpy if PIL Image
            if isinstance(img, Image.Image):
                img = np.array(img)

            ax.imshow(img)
            ax.axis("off")

            if captions and idx < len(captions):
                ax.set_title(captions[idx], fontsize=10)
        else:
            ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


class TensorboardLogger:
    """
    Wrapper for tensorboard logging with utilities for ML experiments.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = "",
        flush_secs: int = 120,
    ):
        """
        Initialize tensorboard logger.

        Args:
            log_dir: Directory for tensorboard logs
            comment: Comment to append to log directory name
            flush_secs: Seconds between log flushes

        Raises:
            ImportError: If tensorboard is not available

        Examples:
            >>> logger = TensorboardLogger("./runs", comment="experiment_1")
            >>> logger.log_scalar("loss", 0.5, step=100)
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "tensorboard is not installed. Install with: pip install tensorboard"
            )

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=flush_secs,
        )

        logger.info(f"Initialized tensorboard logger at {self.log_dir}")

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
        walltime: Optional[float] = None,
    ) -> None:
        """
        Log a scalar value.

        Args:
            tag: Tag name for the scalar
            value: Scalar value
            step: Global step
            walltime: Optional walltime for the value
        """
        self.writer.add_scalar(tag, value, step, walltime=walltime)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log multiple scalar values with the same main tag.

        Args:
            main_tag: Main tag grouping the scalars
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Global step

        Examples:
            >>> logger.log_scalars("metrics", {"accuracy": 0.9, "f1": 0.85}, step=100)
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """
        Log an image.

        Args:
            tag: Tag name for the image
            image: Image as tensor, numpy array, or PIL Image
            step: Global step
            dataformats: Format of the image ('CHW', 'HWC', 'HW')
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                dataformats = "HW"
            elif image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
                dataformats = "HWC"

        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_images(
        self,
        tag: str,
        images: Union[torch.Tensor, List[Image.Image]],
        step: int,
        **kwargs,
    ) -> None:
        """
        Log multiple images as a grid.

        Args:
            tag: Tag name for the images
            images: Tensor or list of images
            step: Global step
            **kwargs: Additional arguments for make_grid
        """
        if isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], Image.Image):
                # Convert PIL images to tensor
                arrays = [np.array(img) for img in images]
                tensor = torch.from_numpy(np.stack(arrays))
                if tensor.ndim == 4:
                    tensor = tensor.permute(0, 3, 1, 2)
                tensor = tensor.float() / 255.0
                images = tensor

        if isinstance(images, torch.Tensor):
            grid = torchvision.utils.make_grid(images, **kwargs)
            self.writer.add_image(tag, grid, step)

    def log_figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: int,
        close: bool = True,
    ) -> None:
        """
        Log a matplotlib figure.

        Args:
            tag: Tag name for the figure
            figure: Matplotlib figure
            step: Global step
            close: Whether to close the figure after logging
        """
        self.writer.add_figure(tag, figure, step, close=close)

    def log_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        step: int,
        bins: str = "tensorflow",
    ) -> None:
        """
        Log a histogram of values.

        Args:
            tag: Tag name
            values: Values to histogram
            step: Global step
            bins: Binning method
        """
        self.writer.add_histogram(tag, values, step, bins=bins)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        self.writer.add_text(tag, text, step)

    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log hyperparameters and optional metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional dictionary of metrics
        """
        self.writer.add_hparams(hparams, metrics or {})

    def log_graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor,
    ) -> None:
        """
        Log model graph.

        Args:
            model: PyTorch model
            input_to_model: Example input tensor
        """
        self.writer.add_graph(model, input_to_model)

    def flush(self) -> None:
        """Flush pending logs to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the tensorboard writer."""
        self.writer.close()
        logger.info("Closed tensorboard logger")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class WandBLogger:
    """
    Wrapper for Weights & Biases logging.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        dir: Optional[str] = None,
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
            tags: List of tags for the run
            notes: Notes about the run
            dir: Directory for W&B files

        Raises:
            ImportError: If wandb is not installed

        Examples:
            >>> logger = WandBLogger(
            ...     project="ml_experiments",
            ...     name="test_run",
            ...     config={"lr": 0.001, "batch_size": 32}
            ... )
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )

        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            dir=dir,
        )

        logger.info(f"Initialized W&B run: {self.run.name}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics and data.

        Args:
            data: Dictionary of data to log
            step: Optional step number
        """
        wandb.log(data, step=step)

    def log_image(
        self,
        key: str,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log an image.

        Args:
            key: Key for the image
            image: Image to log
            caption: Optional caption
            step: Optional step number
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))

        wandb_image = wandb.Image(image, caption=caption)
        wandb.log({key: wandb_image}, step=step)

    def log_images(
        self,
        key: str,
        images: List[Union[Image.Image, np.ndarray]],
        captions: Optional[List[str]] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log multiple images.

        Args:
            key: Key for the images
            images: List of images
            captions: Optional list of captions
            step: Optional step number
        """
        wandb_images = [
            wandb.Image(img, caption=cap if captions else None)
            for img, cap in zip(images, captions or [None] * len(images))
        ]
        wandb.log({key: wandb_images}, step=step)

    def finish(self) -> None:
        """Finish the W&B run."""
        wandb.finish()
        logger.info("Finished W&B run")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a torch tensor to PIL Image.

    Args:
        tensor: Tensor in CHW format, range [0, 1] or [-1, 1]

    Returns:
        PIL Image

    Examples:
        >>> tensor = torch.randn(3, 256, 256)
        >>> img = tensor_to_image(tensor)
        >>> img.save("output.png")
    """
    # Clone to avoid modifying original
    tensor = tensor.clone().detach().cpu()

    # Normalize to [0, 1] if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)

    # Convert to numpy
    np_img = tensor.numpy()

    # CHW to HWC
    if np_img.ndim == 3:
        np_img = np.transpose(np_img, (1, 2, 0))

    # Convert to uint8
    np_img = (np_img * 255).astype(np.uint8)

    # Handle grayscale
    if np_img.shape[-1] == 1:
        np_img = np_img.squeeze(-1)

    return Image.fromarray(np_img)


def create_comparison_grid(
    image_groups: Dict[str, List[Union[Image.Image, np.ndarray]]],
    max_images_per_group: int = 8,
    save_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Create a comparison grid showing multiple groups of images.

    Each group is shown as a row with a label.

    Args:
        image_groups: Dictionary mapping group names to lists of images
        max_images_per_group: Maximum images to show per group
        save_path: Optional path to save the grid

    Returns:
        PIL Image of the comparison grid

    Examples:
        >>> grid = create_comparison_grid({
        ...     "Original": original_images,
        ...     "Generated": generated_images,
        ...     "Ground Truth": gt_images,
        ... })
        >>> grid.save("comparison.png")
    """
    from matplotlib.patches import Rectangle

    n_groups = len(image_groups)
    n_cols = max_images_per_group

    # Create figure
    fig, axes = plt.subplots(
        n_groups,
        n_cols,
        figsize=(n_cols * 2, n_groups * 2),
    )

    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (group_name, images) in enumerate(image_groups.items()):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(images):
                img = images[col_idx]

                # Convert to numpy if PIL
                if isinstance(img, Image.Image):
                    img = np.array(img)

                ax.imshow(img)

                # Add group label on first image
                if col_idx == 0:
                    ax.text(
                        -0.1,
                        0.5,
                        group_name,
                        transform=ax.transAxes,
                        fontsize=12,
                        va="center",
                        rotation=90,
                    )
            ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 3)
    grid_image = Image.fromarray(buf)

    plt.close(fig)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        grid_image.save(save_path)
        logger.info(f"Saved comparison grid to {save_path}")

    return grid_image
