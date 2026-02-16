"""WebDataset utilities for S3 streaming."""

import logging
from typing import Any, Callable, Dict, List, Optional

import boto3
import torch
import webdataset as wds
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field
from torchvision import transforms

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""

    s3_bucket: str = Field(..., description="S3 bucket name")
    s3_prefix: str = Field(..., description="S3 prefix/path to dataset shards")
    shard_pattern: str = Field(default="*.tar", description="Pattern for shard files")
    batch_size: int = Field(default=32, description="Batch size")
    num_workers: int = Field(default=4, description="Number of workers")
    shuffle_buffer: int = Field(default=1000, description="Shuffle buffer size")
    image_size: int = Field(default=256, description="Image size for transforms")
    presigned_url_expiry: int = Field(default=3600, description="Presigned URL expiry in seconds")


# Standard image transforms
def get_image_transforms(image_size: int = 256, normalize: bool = True) -> transforms.Compose:
    """Get standard image transforms pipeline.

    Args:
        image_size: Target image size
        normalize: Whether to normalize images

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

    return transforms.Compose(transform_list)


def list_s3_shards(bucket: str, prefix: str, pattern: str = "*.tar") -> List[str]:
    """List all shard files in S3.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix
        pattern: File pattern (e.g., '*.tar')

    Returns:
        List of S3 keys
    """
    s3_client = boto3.client("s3")
    keys = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(pattern.replace("*", "")):
                        keys.append(key)

        logger.info(f"Found {len(keys)} shards in s3://{bucket}/{prefix}")
        return keys

    except ClientError as e:
        logger.error(f"Error listing S3 shards: {e}")
        raise


def generate_presigned_urls(
    bucket: str,
    keys: List[str],
    expiry: int = 3600
) -> List[str]:
    """Generate presigned URLs for S3 objects.

    Args:
        bucket: S3 bucket name
        keys: List of S3 keys
        expiry: URL expiry time in seconds

    Returns:
        List of presigned URLs
    """
    s3_client = boto3.client("s3")
    urls = []

    for key in keys:
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expiry
            )
            urls.append(url)
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {key}: {e}")
            raise

    return urls


def get_dataset(
    config: DatasetConfig,
    transform: Optional[Callable] = None,
    decode: str = "pil"
) -> wds.WebDataset:
    """Get a streaming WebDataset from S3.

    Args:
        config: Dataset configuration
        transform: Optional transform to apply to images
        decode: Decode format ('pil', 'rgb', 'torch')

    Returns:
        WebDataset instance
    """
    # List shards
    shard_keys = list_s3_shards(
        config.s3_bucket,
        config.s3_prefix,
        config.shard_pattern
    )

    if not shard_keys:
        raise ValueError(f"No shards found in s3://{config.s3_bucket}/{config.s3_prefix}")

    # Generate presigned URLs
    shard_urls = generate_presigned_urls(
        config.s3_bucket,
        shard_keys,
        config.presigned_url_expiry
    )

    # Use default transforms if none provided
    if transform is None:
        transform = get_image_transforms(config.image_size)

    # Create WebDataset pipeline
    dataset = (
        wds.WebDataset(shard_urls)
        .shuffle(config.shuffle_buffer)
        .decode(decode)
        .to_tuple("jpg;png", "json")
        .map_tuple(transform, lambda x: x)  # Apply transform to images
    )

    return dataset


def get_dataloader(
    config: DatasetConfig,
    transform: Optional[Callable] = None,
) -> torch.utils.data.DataLoader:
    """Get a DataLoader for streaming WebDataset.

    Args:
        config: Dataset configuration
        transform: Optional transform to apply to images

    Returns:
        DataLoader instance
    """
    dataset = get_dataset(config, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return dataloader


# Dataset registry for common datasets
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "imagenet": {
        "s3_bucket": "imagenet-dataset",
        "s3_prefix": "train",
        "image_size": 256,
    },
    "coco": {
        "s3_bucket": "coco-dataset",
        "s3_prefix": "train2017",
        "image_size": 256,
    },
    # Add more datasets as needed
}


def get_dataset_by_name(
    name: str,
    batch_size: int = 32,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Get a dataset by name from registry.

    Args:
        name: Dataset name from registry
        batch_size: Batch size
        **kwargs: Additional config overrides

    Returns:
        DataLoader instance
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found in registry. Available: {list(DATASET_REGISTRY.keys())}")

    config_dict = DATASET_REGISTRY[name].copy()
    config_dict.update(kwargs)
    config_dict["batch_size"] = batch_size

    config = DatasetConfig(**config_dict)
    return get_dataloader(config)
