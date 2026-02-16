"""
HuggingFace Hub utilities for model loading with caching and error handling.

This module provides helpers for loading models from HuggingFace Hub with
proper caching, error handling, and optimization support.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    snapshot_download,
    list_repo_files,
    ModelCard,
)
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


class HFModelLoader:
    """
    HuggingFace model loader with caching and error handling.

    Provides utilities for loading models, downloading files, and managing
    the HuggingFace cache.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ):
        """
        Initialize HuggingFace model loader.

        Args:
            cache_dir: Directory to cache models (default: None, uses HF default)
            token: HuggingFace API token for private models (default: None)
            use_auth_token: Deprecated, use token instead

        Examples:
            >>> loader = HFModelLoader(cache_dir="./hf_cache")
            >>> model = loader.load_model("bert-base-uncased", model_class="AutoModel")
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Handle deprecated use_auth_token parameter
        if use_auth_token is not None:
            logger.warning(
                "use_auth_token is deprecated, use token parameter instead"
            )
            if token is None:
                token = use_auth_token if isinstance(use_auth_token, str) else None

        self.token = token or os.environ.get("HF_TOKEN")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache directory: {self.cache_dir}")

        # Initialize HF API client
        self.api = HfApi(token=self.token)

    def load_model(
        self,
        model_name: str,
        model_class: str = "AutoModel",
        revision: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict]] = None,
        low_cpu_mem_usage: bool = True,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Any:
        """
        Load a model from HuggingFace Hub.

        Args:
            model_name: Model name or path on HuggingFace Hub
            model_class: Model class name (e.g., 'AutoModel', 'AutoModelForCausalLM')
            revision: Model revision (branch, tag, or commit hash)
            torch_dtype: Target dtype (e.g., torch.float16, torch.bfloat16)
            device_map: Device map for model parallelism (e.g., 'auto', 'cuda:0')
            low_cpu_mem_usage: Use low CPU memory loading
            trust_remote_code: Trust remote code for custom models
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Loaded model

        Raises:
            ImportError: If transformers is not installed
            RepositoryNotFoundError: If model not found
            HfHubHTTPError: If download fails

        Examples:
            >>> loader = HFModelLoader()
            >>> model = loader.load_model(
            ...     "stabilityai/stable-diffusion-xl-base-1.0",
            ...     model_class="StableDiffusionXLPipeline",
            ...     torch_dtype=torch.float16,
            ...     device_map="auto"
            ... )
        """
        try:
            # Dynamic import based on model class
            if "StableDiffusion" in model_class or "Pipeline" in model_class:
                from diffusers import (
                    StableDiffusionPipeline,
                    StableDiffusionXLPipeline,
                    DiffusionPipeline,
                )
                class_map = {
                    "StableDiffusionPipeline": StableDiffusionPipeline,
                    "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
                    "DiffusionPipeline": DiffusionPipeline,
                }
                model_cls = class_map.get(model_class, DiffusionPipeline)
            else:
                from transformers import (
                    AutoModel,
                    AutoModelForCausalLM,
                    AutoModelForSeq2SeqLM,
                    AutoModelForMaskedLM,
                    AutoModelForSequenceClassification,
                )
                class_map = {
                    "AutoModel": AutoModel,
                    "AutoModelForCausalLM": AutoModelForCausalLM,
                    "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
                    "AutoModelForMaskedLM": AutoModelForMaskedLM,
                    "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
                }
                model_cls = class_map.get(model_class, AutoModel)

            logger.info(f"Loading model: {model_name} ({model_class})")

            # Build kwargs for from_pretrained
            load_kwargs = {
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "revision": revision,
                "trust_remote_code": trust_remote_code,
                **kwargs,
            }

            # Add token if available
            if self.token:
                load_kwargs["token"] = self.token

            # Add optimization kwargs
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            if low_cpu_mem_usage:
                load_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            # Load model
            model = model_cls.from_pretrained(model_name, **load_kwargs)

            logger.info(f"Successfully loaded model: {model_name}")
            return model

        except ImportError as e:
            logger.error(f"Required library not installed: {e}")
            raise
        except RepositoryNotFoundError:
            logger.error(f"Model not found: {model_name}")
            raise
        except HfHubHTTPError as e:
            logger.error(f"Failed to download model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            raise

    def load_tokenizer(
        self,
        model_name: str,
        tokenizer_class: str = "AutoTokenizer",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Any:
        """
        Load a tokenizer from HuggingFace Hub.

        Args:
            model_name: Model name or path on HuggingFace Hub
            tokenizer_class: Tokenizer class name (default: 'AutoTokenizer')
            revision: Model revision
            trust_remote_code: Trust remote code for custom tokenizers
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Loaded tokenizer

        Examples:
            >>> loader = HFModelLoader()
            >>> tokenizer = loader.load_tokenizer("bert-base-uncased")
        """
        try:
            from transformers import AutoTokenizer, CLIPTokenizer, T5Tokenizer

            class_map = {
                "AutoTokenizer": AutoTokenizer,
                "CLIPTokenizer": CLIPTokenizer,
                "T5Tokenizer": T5Tokenizer,
            }
            tokenizer_cls = class_map.get(tokenizer_class, AutoTokenizer)

            logger.info(f"Loading tokenizer: {model_name}")

            load_kwargs = {
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "revision": revision,
                "trust_remote_code": trust_remote_code,
                **kwargs,
            }

            if self.token:
                load_kwargs["token"] = self.token

            tokenizer = tokenizer_cls.from_pretrained(model_name, **load_kwargs)

            logger.info(f"Successfully loaded tokenizer: {model_name}")
            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def download_model(
        self,
        model_name: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> Path:
        """
        Download entire model repository from HuggingFace Hub.

        Args:
            model_name: Model name or path on HuggingFace Hub
            revision: Model revision to download
            allow_patterns: List of file patterns to include
            ignore_patterns: List of file patterns to exclude

        Returns:
            Path to downloaded model directory

        Examples:
            >>> loader = HFModelLoader()
            >>> model_path = loader.download_model(
            ...     "stabilityai/stable-diffusion-xl-base-1.0",
            ...     allow_patterns=["*.safetensors", "*.json"]
            ... )
        """
        try:
            logger.info(f"Downloading model repository: {model_name}")

            download_kwargs = {
                "repo_id": model_name,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "revision": revision,
                "allow_patterns": allow_patterns,
                "ignore_patterns": ignore_patterns,
            }

            if self.token:
                download_kwargs["token"] = self.token

            model_path = snapshot_download(**download_kwargs)

            logger.info(f"Successfully downloaded model to: {model_path}")
            return Path(model_path)

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def download_file(
        self,
        model_name: str,
        filename: str,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> Path:
        """
        Download a specific file from a HuggingFace model repository.

        Args:
            model_name: Model name or path on HuggingFace Hub
            filename: Name of the file to download
            revision: Model revision
            subfolder: Subfolder within the repository

        Returns:
            Path to downloaded file

        Examples:
            >>> loader = HFModelLoader()
            >>> config_path = loader.download_file(
            ...     "bert-base-uncased",
            ...     "config.json"
            ... )
        """
        try:
            logger.info(f"Downloading file: {filename} from {model_name}")

            download_kwargs = {
                "repo_id": model_name,
                "filename": filename,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "revision": revision,
                "subfolder": subfolder,
            }

            if self.token:
                download_kwargs["token"] = self.token

            file_path = hf_hub_download(**download_kwargs)

            logger.info(f"Successfully downloaded file to: {file_path}")
            return Path(file_path)

        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            raise

    def list_model_files(
        self,
        model_name: str,
        revision: Optional[str] = None,
    ) -> List[str]:
        """
        List all files in a HuggingFace model repository.

        Args:
            model_name: Model name or path on HuggingFace Hub
            revision: Model revision

        Returns:
            List of file paths in the repository

        Examples:
            >>> loader = HFModelLoader()
            >>> files = loader.list_model_files("bert-base-uncased")
            >>> print(files[:5])
        """
        try:
            files = list_repo_files(
                repo_id=model_name,
                revision=revision,
                token=self.token,
            )
            return files
        except Exception as e:
            logger.error(f"Failed to list model files: {e}")
            raise

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model from HuggingFace Hub.

        Args:
            model_name: Model name or path on HuggingFace Hub

        Returns:
            Dictionary with model information

        Examples:
            >>> loader = HFModelLoader()
            >>> info = loader.get_model_info("bert-base-uncased")
            >>> print(info['downloads'])
        """
        try:
            model_info = self.api.model_info(model_name, token=self.token)
            return {
                "id": model_info.id,
                "author": model_info.author,
                "downloads": model_info.downloads,
                "likes": model_info.likes,
                "tags": model_info.tags,
                "pipeline_tag": model_info.pipeline_tag,
                "library_name": model_info.library_name,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear the HuggingFace cache.

        Args:
            model_name: Optional model name to clear specific model cache.
                       If None, clears entire cache.

        Examples:
            >>> loader = HFModelLoader(cache_dir="./hf_cache")
            >>> loader.clear_cache("bert-base-uncased")  # Clear specific model
            >>> loader.clear_cache()  # Clear all cache
        """
        if not self.cache_dir:
            logger.warning("No cache directory specified")
            return

        try:
            if model_name:
                # Try to clear specific model cache
                # This is a best-effort operation as HF cache structure can vary
                logger.info(f"Attempting to clear cache for model: {model_name}")
                model_name_normalized = model_name.replace("/", "--")

                # Look for model directories in cache
                for item in self.cache_dir.iterdir():
                    if model_name_normalized in item.name:
                        if item.is_dir():
                            shutil.rmtree(item)
                            logger.info(f"Cleared cache directory: {item}")
                        else:
                            item.unlink()
                            logger.info(f"Cleared cache file: {item}")
            else:
                # Clear entire cache directory
                logger.info(f"Clearing entire cache directory: {self.cache_dir}")
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Cache cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise


def get_model_card(model_name: str, token: Optional[str] = None) -> Optional[str]:
    """
    Get the model card (README) for a HuggingFace model.

    Args:
        model_name: Model name or path on HuggingFace Hub
        token: Optional HuggingFace API token

    Returns:
        Model card text, or None if not available

    Examples:
        >>> card = get_model_card("bert-base-uncased")
        >>> print(card[:200])
    """
    try:
        card = ModelCard.load(model_name, token=token)
        return card.text
    except Exception as e:
        logger.error(f"Failed to get model card: {e}")
        return None


def check_model_exists(model_name: str, token: Optional[str] = None) -> bool:
    """
    Check if a model exists on HuggingFace Hub.

    Args:
        model_name: Model name or path to check
        token: Optional HuggingFace API token

    Returns:
        True if model exists, False otherwise

    Examples:
        >>> if check_model_exists("bert-base-uncased"):
        ...     print("Model exists!")
    """
    try:
        api = HfApi(token=token)
        api.model_info(model_name)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        logger.error(f"Error checking model existence: {e}")
        return False


def get_recommended_dtype(device: Optional[str] = None) -> torch.dtype:
    """
    Get recommended dtype for model loading based on device capabilities.

    Args:
        device: Target device (e.g., 'cuda', 'cpu', 'cuda:0')

    Returns:
        Recommended torch dtype

    Examples:
        >>> dtype = get_recommended_dtype("cuda")
        >>> print(dtype)  # torch.float16 or torch.bfloat16
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if "cuda" in device:
        # Check if bfloat16 is supported (Ampere or newer)
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability[0] >= 8:  # Ampere or newer
                logger.info("Using bfloat16 (recommended for Ampere+ GPUs)")
                return torch.bfloat16
        logger.info("Using float16")
        return torch.float16
    else:
        logger.info("Using float32 for CPU")
        return torch.float32


def optimize_model_loading_kwargs(
    device: Optional[str] = None,
    low_memory: bool = True,
    quantize: bool = False,
) -> Dict[str, Any]:
    """
    Get optimized kwargs for model loading based on device and preferences.

    Args:
        device: Target device
        low_memory: Use low memory loading strategies
        quantize: Use quantization (requires bitsandbytes)

    Returns:
        Dictionary of kwargs for from_pretrained

    Examples:
        >>> kwargs = optimize_model_loading_kwargs(device="cuda", quantize=True)
        >>> model = AutoModel.from_pretrained("model-name", **kwargs)
    """
    kwargs = {}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set dtype
    kwargs["torch_dtype"] = get_recommended_dtype(device)

    # Low memory options
    if low_memory and "cuda" in device:
        kwargs["low_cpu_mem_usage"] = True
        kwargs["device_map"] = "auto"

    # Quantization
    if quantize:
        try:
            import bitsandbytes  # noqa: F401
            kwargs["load_in_8bit"] = True
            logger.info("Enabling 8-bit quantization")
        except ImportError:
            logger.warning(
                "bitsandbytes not installed, quantization not available"
            )

    return kwargs
