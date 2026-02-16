"""
S3 utilities for WebDataset shard listing and presigned URL generation.

This module provides utilities for working with S3 in the context of WebDataset
streaming, including listing shards, generating presigned URLs, and bucket operations.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError

logger = logging.getLogger(__name__)


class S3Client:
    """
    S3 client wrapper with utilities for WebDataset operations.

    Provides methods for listing shards, generating presigned URLs,
    and other S3 operations needed for ML experiments.
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        profile_name: Optional[str] = None,
    ):
        """
        Initialize S3 client.

        Args:
            region_name: AWS region name (default: None, uses default region)
            aws_access_key_id: AWS access key ID (default: None, uses env/config)
            aws_secret_access_key: AWS secret access key (default: None, uses env/config)
            profile_name: AWS profile name (default: None, uses default profile)

        Raises:
            NoCredentialsError: If AWS credentials are not available
        """
        try:
            # Build kwargs for boto3 client
            kwargs = {}
            if region_name:
                kwargs["region_name"] = region_name
            if aws_access_key_id and aws_secret_access_key:
                kwargs["aws_access_key_id"] = aws_access_key_id
                kwargs["aws_secret_access_key"] = aws_secret_access_key
            if profile_name:
                # Use session with profile
                session = boto3.Session(profile_name=profile_name)
                self.client = session.client("s3", **kwargs)
            else:
                self.client = boto3.client("s3", **kwargs)

            logger.info("S3 client initialized successfully")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    @staticmethod
    def parse_s3_url(s3_url: str) -> Tuple[str, str]:
        """
        Parse S3 URL into bucket and key.

        Args:
            s3_url: S3 URL (e.g., 's3://bucket/path/to/key' or 's3://bucket/path/')

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URL is invalid

        Examples:
            >>> S3Client.parse_s3_url("s3://my-bucket/data/shards-00001.tar")
            ('my-bucket', 'data/shards-00001.tar')
        """
        if not s3_url.startswith("s3://"):
            raise ValueError(f"Invalid S3 URL: {s3_url}. Must start with 's3://'")

        parsed = urlparse(s3_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"Invalid S3 URL: {s3_url}. Bucket name is empty")

        return bucket, key

    def list_shards(
        self,
        s3_path: str,
        pattern: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[str]:
        """
        List WebDataset shards in an S3 path.

        Args:
            s3_path: S3 path to search (e.g., 's3://bucket/path/to/shards/')
            pattern: Optional regex pattern to filter shard names (e.g., r'shard-\d+\.tar')
            max_keys: Maximum number of keys to return (default: 1000)

        Returns:
            List of S3 URLs for matching shards

        Raises:
            ValueError: If S3 path is invalid
            ClientError: If S3 operation fails

        Examples:
            >>> client = S3Client()
            >>> shards = client.list_shards("s3://my-bucket/data/", pattern=r'.*\.tar$')
        """
        try:
            bucket, prefix = self.parse_s3_url(s3_path)

            logger.info(f"Listing shards in s3://{bucket}/{prefix}")

            # List objects with pagination
            paginator = self.client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys},
            )

            shard_keys = []
            compiled_pattern = re.compile(pattern) if pattern else None

            for page in page_iterator:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]

                    # Skip directories
                    if key.endswith("/"):
                        continue

                    # Apply pattern filter if specified
                    if compiled_pattern and not compiled_pattern.search(key):
                        continue

                    shard_keys.append(f"s3://{bucket}/{key}")

            logger.info(f"Found {len(shard_keys)} shards")
            return shard_keys

        except ClientError as e:
            logger.error(f"Failed to list shards: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing shards: {e}")
            raise

    def generate_presigned_url(
        self,
        s3_url: str,
        expiration: int = 3600,
        http_method: str = "GET",
    ) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            s3_url: S3 URL (e.g., 's3://bucket/key')
            expiration: URL expiration time in seconds (default: 3600)
            http_method: HTTP method for the URL (default: 'GET')

        Returns:
            Presigned URL string

        Raises:
            ValueError: If S3 URL is invalid
            ClientError: If presigned URL generation fails

        Examples:
            >>> client = S3Client()
            >>> url = client.generate_presigned_url("s3://bucket/data.tar", expiration=7200)
        """
        try:
            bucket, key = self.parse_s3_url(s3_url)

            params = {
                "Bucket": bucket,
                "Key": key,
            }

            # Map HTTP method to boto3 operation
            operation_map = {
                "GET": "get_object",
                "PUT": "put_object",
                "HEAD": "head_object",
            }

            operation = operation_map.get(http_method.upper())
            if not operation:
                raise ValueError(f"Unsupported HTTP method: {http_method}")

            presigned_url = self.client.generate_presigned_url(
                operation,
                Params=params,
                ExpiresIn=expiration,
            )

            logger.debug(f"Generated presigned URL for {s3_url} (expires in {expiration}s)")
            return presigned_url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL: {e}")
            raise

    def generate_presigned_urls_batch(
        self,
        s3_urls: List[str],
        expiration: int = 3600,
    ) -> Dict[str, str]:
        """
        Generate presigned URLs for multiple S3 objects.

        Args:
            s3_urls: List of S3 URLs
            expiration: URL expiration time in seconds (default: 3600)

        Returns:
            Dictionary mapping S3 URLs to presigned URLs

        Examples:
            >>> client = S3Client()
            >>> urls = ["s3://bucket/shard1.tar", "s3://bucket/shard2.tar"]
            >>> presigned = client.generate_presigned_urls_batch(urls)
        """
        presigned_urls = {}
        for s3_url in s3_urls:
            try:
                presigned_urls[s3_url] = self.generate_presigned_url(
                    s3_url,
                    expiration=expiration,
                )
            except Exception as e:
                logger.warning(f"Failed to generate presigned URL for {s3_url}: {e}")
                continue

        logger.info(f"Generated {len(presigned_urls)}/{len(s3_urls)} presigned URLs")
        return presigned_urls

    def upload_file(
        self,
        local_path: str,
        s3_url: str,
        extra_args: Optional[Dict] = None,
    ) -> bool:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path
            s3_url: Destination S3 URL
            extra_args: Extra arguments for upload (e.g., {'ACL': 'public-read'})

        Returns:
            True if upload successful, False otherwise

        Examples:
            >>> client = S3Client()
            >>> client.upload_file("model.pth", "s3://bucket/checkpoints/model.pth")
        """
        try:
            bucket, key = self.parse_s3_url(s3_url)

            self.client.upload_file(
                local_path,
                bucket,
                key,
                ExtraArgs=extra_args or {},
            )

            logger.info(f"Uploaded {local_path} to {s3_url}")
            return True

        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {e}")
            return False

    def download_file(
        self,
        s3_url: str,
        local_path: str,
    ) -> bool:
        """
        Download a file from S3.

        Args:
            s3_url: Source S3 URL
            local_path: Destination local path

        Returns:
            True if download successful, False otherwise

        Examples:
            >>> client = S3Client()
            >>> client.download_file("s3://bucket/model.pth", "./model.pth")
        """
        try:
            bucket, key = self.parse_s3_url(s3_url)

            # Create parent directories if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            self.client.download_file(bucket, key, local_path)

            logger.info(f"Downloaded {s3_url} to {local_path}")
            return True

        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading file: {e}")
            return False

    def object_exists(self, s3_url: str) -> bool:
        """
        Check if an S3 object exists.

        Args:
            s3_url: S3 URL to check

        Returns:
            True if object exists, False otherwise

        Examples:
            >>> client = S3Client()
            >>> if client.object_exists("s3://bucket/file.txt"):
            ...     print("File exists")
        """
        try:
            bucket, key = self.parse_s3_url(s3_url)
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Error checking object existence: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking object existence: {e}")
            return False

    def get_object_size(self, s3_url: str) -> Optional[int]:
        """
        Get the size of an S3 object in bytes.

        Args:
            s3_url: S3 URL

        Returns:
            Size in bytes, or None if object doesn't exist

        Examples:
            >>> client = S3Client()
            >>> size = client.get_object_size("s3://bucket/file.txt")
            >>> print(f"File size: {size / 1024 / 1024:.2f} MB")
        """
        try:
            bucket, key = self.parse_s3_url(s3_url)
            response = self.client.head_object(Bucket=bucket, Key=key)
            return response["ContentLength"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"Object not found: {s3_url}")
            else:
                logger.error(f"Error getting object size: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting object size: {e}")
            return None


def expand_shard_pattern(s3_path: str) -> List[str]:
    """
    Expand WebDataset brace pattern to list of S3 URLs.

    Supports patterns like:
    - s3://bucket/shards-{000..099}.tar -> 100 URLs
    - s3://bucket/data-{0..9}-{a..c}.tar -> 30 URLs

    Args:
        s3_path: S3 path with brace pattern

    Returns:
        List of expanded S3 URLs

    Examples:
        >>> urls = expand_shard_pattern("s3://bucket/shard-{000..002}.tar")
        >>> print(urls)
        ['s3://bucket/shard-000.tar', 's3://bucket/shard-001.tar', 's3://bucket/shard-002.tar']
    """
    # Use webdataset's built-in braceexpand
    try:
        import braceexpand
        return list(braceexpand.braceexpand(s3_path))
    except ImportError:
        # Fallback: simple numeric range expansion
        logger.warning("braceexpand not available, using simple expansion")

        # Match pattern like {000..099}
        match = re.search(r'\{(\d+)\.\.(\d+)\}', s3_path)
        if not match:
            return [s3_path]

        start_str, end_str = match.groups()
        start = int(start_str)
        end = int(end_str)
        width = len(start_str)

        expanded = []
        for i in range(start, end + 1):
            url = re.sub(
                r'\{\d+\.\.\d+\}',
                str(i).zfill(width),
                s3_path,
                count=1
            )
            expanded.append(url)

        return expanded


def get_shard_count(s3_path: str) -> int:
    """
    Get the number of shards from a WebDataset path pattern.

    Args:
        s3_path: S3 path with or without brace pattern

    Returns:
        Number of shards

    Examples:
        >>> count = get_shard_count("s3://bucket/shard-{000..099}.tar")
        >>> print(count)  # 100
    """
    expanded = expand_shard_pattern(s3_path)
    return len(expanded)


def validate_s3_credentials() -> bool:
    """
    Validate that S3 credentials are available and working.

    Returns:
        True if credentials are valid, False otherwise

    Examples:
        >>> if validate_s3_credentials():
        ...     print("S3 credentials are valid")
        ... else:
        ...     print("S3 credentials are invalid or not configured")
    """
    try:
        client = boto3.client("s3")
        # Try to list buckets as a simple test
        client.list_buckets()
        logger.info("S3 credentials validated successfully")
        return True
    except NoCredentialsError:
        logger.warning("No AWS credentials found")
        return False
    except ClientError as e:
        logger.error(f"S3 credentials validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating S3 credentials: {e}")
        return False
