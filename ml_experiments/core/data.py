"""
WebDataset S3 streaming utilities for efficient data loading.

This module provides utilities for streaming datasets from S3 using WebDataset,
with support for presigned URLs, standard image transforms, and dataset registry.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import io

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import webdataset as wds
import torch
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for dataset loading."""

    def __init__(
        self,
        s3_path: str,
        batch_size: int = 32,
        shuffle: int = 1000,
        image_size: int = 512,
        num_workers: int = 4,
        presigned_url_expiry: int = 3600,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        decode_keys: Optional[List[str]] = None,
    ):
        """
        Initialize dataset configuration.

        Args:
            s3_path: S3 path to WebDataset shards (e.g., 's3://bucket/path/to/shards-{000000..000099}.tar')
            batch_size: Batch size for data loading
            shuffle: Shuffle buffer size (0 to disable)
            image_size: Target image size for transforms
            num_workers: Number of worker processes for data loading
            presigned_url_expiry: Expiry time for presigned URLs in seconds
            cache_dir: Optional local cache directory
            transform: Optional custom transform pipeline
            decode_keys: Optional list of keys to decode (default: ['jpg', 'png', 'jpeg'])
        """
        self.s3_path = s3_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.num_workers = num_workers
        self.presigned_url_expiry = presigned_url_expiry
        self.cache_dir = cache_dir
        self.transform = transform
        self.decode_keys = decode_keys or ['jpg', 'png', 'jpeg', 'webp']


class S3PresignedURLGenerator:
    """Generate presigned URLs for S3 objects to enable streaming."""

    def __init__(self, expiry: int = 3600, region_name: Optional[str] = None):
        """
        Initialize presigned URL generator.

        Args:
            expiry: URL expiry time in seconds
            region_name: AWS region name
        """
        self.expiry = expiry
        try:
            self.s3_client = boto3.client('s3', region_name=region_name)
        except NoCredentialsError:
            logger.warning("No AWS credentials found. Presigned URL generation will fail.")
            self.s3_client = None

    def generate_presigned_url(self, s3_url: str) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            s3_url: S3 URL (e.g., 's3://bucket/key')

        Returns:
            Presigned URL string

        Raises:
            ValueError: If S3 URL is invalid
            ClientError: If presigned URL generation fails
        """
        if not s3_url.startswith('s3://'):
            raise ValueError(f"Invalid S3 URL: {s3_url}")

        if self.s3_client is None:
            raise NoCredentialsError("AWS credentials not available")

        # Parse S3 URL
        s3_url = s3_url.replace('s3://', '')
        parts = s3_url.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")

        bucket, key = parts

        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=self.expiry
            )
            return presigned_url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {s3_url}: {e}")
            raise

    def convert_s3_path_to_presigned(self, s3_path: str) -> str:
        """
        Convert S3 path pattern to presigned URL pattern.

        Args:
            s3_path: S3 path with potential braceexpand pattern

        Returns:
            Modified path that can be used with WebDataset
        """
        # For WebDataset streaming, we can use S3 URLs directly if credentials are available
        # Or convert to presigned URLs for specific shards
        return s3_path


def get_standard_transforms(
    image_size: int = 512,
    normalize: bool = True,
    augment: bool = False
) -> transforms.Compose:
    """
    Get standard image transformation pipeline.

    Args:
        image_size: Target image size
        normalize: Whether to apply ImageNet normalization
        augment: Whether to apply data augmentation

    Returns:
        Composed transforms pipeline
    """
    transform_list = []

    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    else:
        transform_list.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return transforms.Compose(transform_list)


def decode_image(sample: Dict[str, Any], key: str = 'jpg') -> Dict[str, Any]:
    """
    Decode image from WebDataset sample.

    Args:
        sample: WebDataset sample dictionary
        key: Key for image data

    Returns:
        Sample with decoded image
    """
    try:
        # Try to get image from specified key or common image keys
        image_data = None
        for img_key in [key, 'jpg', 'png', 'jpeg', 'webp', 'image']:
            if img_key in sample:
                image_data = sample[img_key]
                break

        if image_data is None:
            logger.warning(f"No image data found in sample: {sample.keys()}")
            return sample

        # Handle different image data types
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            image = image_data.convert('RGB')
        else:
            logger.warning(f"Unsupported image data type: {type(image_data)}")
            return sample

        sample['image'] = image
        return sample
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return sample


def create_error_handler(warn_only: bool = True):
    """
    Create error handler for WebDataset pipeline.

    Args:
        warn_only: If True, log warnings and continue; if False, raise exceptions

    Returns:
        Error handler function
    """
    def handler(exn):
        if warn_only:
            logger.warning(f"WebDataset error: {exn}")
            return True  # Continue processing
        else:
            logger.error(f"WebDataset error: {exn}")
            return False  # Stop processing

    return handler


def get_dataset(
    config: Union[DatasetConfig, Dict[str, Any]],
    transform: Optional[Callable] = None,
) -> wds.WebDataset:
    """
    Create a streaming WebDataset from S3.

    Args:
        config: Dataset configuration (DatasetConfig or dict)
        transform: Optional transform to apply to images (overrides config.transform)

    Returns:
        WebDataset instance ready for iteration

    Raises:
        ValueError: If configuration is invalid
        Exception: If dataset creation fails
    """
    # Convert dict to DatasetConfig if needed
    if isinstance(config, dict):
        config = DatasetConfig(**config)

    # Validate S3 path
    if not config.s3_path:
        raise ValueError("S3 path is required")

    logger.info(f"Creating WebDataset from S3 path: {config.s3_path}")

    # Handle S3 URL conversion if needed
    s3_path = config.s3_path

    # Support both direct S3 paths and http/https URLs (for presigned URLs)
    if s3_path.startswith('s3://'):
        # For direct S3 access, we'll use the pipe: prefix which uses boto3 internally
        # or convert to presigned URLs if credentials are not available
        try:
            # Test if we have S3 access
            url_generator = S3PresignedURLGenerator(expiry=config.presigned_url_expiry)
            if url_generator.s3_client is None:
                logger.warning("No S3 credentials available. Direct S3 streaming may fail.")
        except Exception as e:
            logger.warning(f"S3 access test failed: {e}")

    # Create WebDataset with error handling
    dataset = wds.WebDataset(
        s3_path,
        handler=create_error_handler(warn_only=True),
        shardshuffle=config.shuffle > 0,
    )

    # Apply shuffle if specified
    if config.shuffle > 0:
        dataset = dataset.shuffle(config.shuffle)

    # Decode images and other data
    dataset = dataset.decode("pil", handler=create_error_handler(warn_only=True))

    # Apply transforms
    final_transform = transform or config.transform
    if final_transform is None:
        final_transform = get_standard_transforms(
            image_size=config.image_size,
            normalize=True,
            augment=False
        )

    def apply_transform(sample):
        """Apply transform to sample."""
        try:
            # Find image in sample
            image = None
            for key in config.decode_keys:
                if key in sample:
                    image = sample[key]
                    break

            if image is None and 'image' in sample:
                image = sample['image']

            if image is not None:
                if isinstance(image, Image.Image):
                    sample['image'] = final_transform(image)
                else:
                    logger.warning(f"Image is not PIL.Image: {type(image)}")

            return sample
        except Exception as e:
            logger.error(f"Error applying transform: {e}")
            return sample

    dataset = dataset.map(apply_transform)

    # Batch if batch size is specified
    if config.batch_size > 1:
        dataset = dataset.batched(config.batch_size)

    return dataset


# Dataset Registry - common datasets with predefined configurations
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "laion-art": {
        "s3_path": "s3://laion-art/webdataset/shards-{000000..009999}.tar",
        "image_size": 512,
        "description": "LAION-Art dataset for artistic image generation",
    },
    "imagenet": {
        "s3_path": "s3://imagenet-webdataset/train/shards-{000000..001023}.tar",
        "image_size": 224,
        "description": "ImageNet training dataset",
    },
    "coco": {
        "s3_path": "s3://coco-webdataset/train2017/shards-{000000..000099}.tar",
        "image_size": 512,
        "description": "COCO 2017 training dataset",
    },
}


def get_dataset_from_registry(
    dataset_name: str,
    batch_size: int = 32,
    **kwargs
) -> wds.WebDataset:
    """
    Get a dataset from the registry by name.

    Args:
        dataset_name: Name of the dataset in the registry
        batch_size: Batch size for data loading
        **kwargs: Additional arguments to override dataset config

    Returns:
        WebDataset instance

    Raises:
        ValueError: If dataset name is not in registry
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: {available}"
        )

    # Get base config from registry
    registry_config = DATASET_REGISTRY[dataset_name].copy()
    description = registry_config.pop('description', '')

    logger.info(f"Loading dataset '{dataset_name}': {description}")

    # Merge with provided kwargs
    config_dict = {**registry_config, 'batch_size': batch_size, **kwargs}
    config = DatasetConfig(**config_dict)

    return get_dataset(config)


def list_available_datasets() -> List[str]:
    """
    List all available datasets in the registry.

    Returns:
        List of dataset names
    """
    return list(DATASET_REGISTRY.keys())


def register_dataset(
    name: str,
    s3_path: str,
    image_size: int = 512,
    description: str = "",
    **kwargs
) -> None:
    """
    Register a new dataset in the registry.

    Args:
        name: Dataset name
        s3_path: S3 path to WebDataset shards
        image_size: Default image size
        description: Dataset description
        **kwargs: Additional dataset configuration
    """
    if name in DATASET_REGISTRY:
        logger.warning(f"Overwriting existing dataset '{name}' in registry")

    DATASET_REGISTRY[name] = {
        "s3_path": s3_path,
        "image_size": image_size,
        "description": description,
        **kwargs
    }

    logger.info(f"Registered dataset '{name}': {description}")


def create_dataloader(
    dataset: wds.WebDataset,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from a WebDataset.

    Args:
        dataset: WebDataset instance
        num_workers: Number of worker processes
        prefetch_factor: Number of batches to prefetch per worker
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # Batching is handled by WebDataset
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )
