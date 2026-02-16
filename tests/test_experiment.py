"""Tests for the base Experiment class."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from ml_experiments.core import Experiment, ExperimentConfig, ExperimentFactory


class DummyExperiment(Experiment):
    """Dummy experiment for testing."""

    def train(self):
        """Simple training loop."""
        for step in range(10):
            loss = 1.0 / (step + 1)
            self.log_metrics({"loss": loss}, step=step)

    def evaluate(self):
        """Simple evaluation."""
        return {"accuracy": 0.95, "f1": 0.93}

    def inference(self, x):
        """Simple inference."""
        return x * 2


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def basic_config(temp_dir):
    """Create a basic experiment configuration."""
    return ExperimentConfig(
        experiment_name="test_experiment",
        experiment_type="dummy",
        output_dir=temp_dir,
        seed=42,
        device="cpu",
        log_level="INFO",
    )


def test_experiment_init_with_config_object(basic_config):
    """Test initialization with ExperimentConfig object."""
    exp = DummyExperiment(basic_config)
    assert exp.config.experiment_name == "test_experiment"
    assert exp.config.experiment_type == "dummy"


def test_experiment_init_with_dict(temp_dir):
    """Test initialization with configuration dict."""
    config_dict = {
        "experiment_name": "test_dict",
        "experiment_type": "dummy",
        "output_dir": temp_dir,
        "seed": 42,
    }
    exp = DummyExperiment(config_dict)
    assert exp.config.experiment_name == "test_dict"


def test_experiment_init_with_yaml(temp_dir):
    """Test initialization with YAML configuration file."""
    config_dict = {
        "experiment_name": "test_yaml",
        "experiment_type": "dummy",
        "output_dir": temp_dir,
        "seed": 42,
    }
    config_path = Path(temp_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    exp = DummyExperiment(config_path)
    assert exp.config.experiment_name == "test_yaml"


def test_experiment_setup(basic_config):
    """Test experiment setup."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    assert exp._is_setup
    assert exp.device is not None
    assert Path(basic_config.output_dir).exists()
    assert (Path(basic_config.output_dir) / "config.yaml").exists()


def test_experiment_log_metrics(basic_config):
    """Test metric logging."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    exp.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=0)

    assert "loss" in exp._metrics_history
    assert "accuracy" in exp._metrics_history
    assert exp._metrics_history["loss"][0] == 0.5
    assert exp._metrics_history["accuracy"][0] == 0.9


def test_experiment_log_metrics_with_tensor(basic_config):
    """Test metric logging with torch tensors."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    loss_tensor = torch.tensor(0.5)
    exp.log_metrics({"loss": loss_tensor}, step=0)

    assert "loss" in exp._metrics_history
    assert exp._metrics_history["loss"][0] == 0.5


def test_experiment_save_and_load_checkpoint(basic_config, temp_dir):
    """Test checkpoint saving and loading."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    # Log some metrics
    exp.log_metrics({"loss": 0.5}, step=5)

    # Save checkpoint
    state_dict = {"model": "dummy_weights"}
    checkpoint_path = exp.save_checkpoint(state_dict=state_dict, step=5)

    assert checkpoint_path.exists()

    # Load checkpoint
    loaded = exp.load_checkpoint(checkpoint_path)

    assert loaded["step"] == 5
    assert "loss" in loaded["metrics_history"]
    assert loaded["state_dict"]["model"] == "dummy_weights"


def test_experiment_metrics_summary(basic_config):
    """Test metrics summary generation."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    # Log some metrics
    for i in range(10):
        exp.log_metrics({"loss": 1.0 / (i + 1)}, step=i)

    summary = exp.get_metrics_summary()

    assert "loss" in summary
    assert "mean" in summary["loss"]
    assert "std" in summary["loss"]
    assert "min" in summary["loss"]
    assert "max" in summary["loss"]
    assert "latest" in summary["loss"]
    assert summary["loss"]["count"] == 10


def test_experiment_train(basic_config):
    """Test training execution."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    exp.train()

    assert len(exp._metrics_history["loss"]) == 10
    # Loss should decrease
    assert exp._metrics_history["loss"][0] > exp._metrics_history["loss"][-1]


def test_experiment_evaluate(basic_config):
    """Test evaluation execution."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    metrics = exp.evaluate()

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] == 0.95


def test_experiment_inference(basic_config):
    """Test inference execution."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    result = exp.inference(5)
    assert result == 10


def test_experiment_cleanup(basic_config):
    """Test experiment cleanup."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    exp.log_metrics({"loss": 0.5}, step=0)
    exp.cleanup()

    # Check that metrics summary was saved
    summary_path = Path(basic_config.output_dir) / "metrics_summary.json"
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)
    assert "loss" in summary


def test_experiment_factory_register():
    """Test experiment factory registration."""

    @ExperimentFactory.register("test_type")
    class TestExperiment(Experiment):
        def train(self):
            pass

        def evaluate(self):
            return {}

        def inference(self):
            pass

    assert "test_type" in ExperimentFactory.list_types()


def test_experiment_factory_create(temp_dir):
    """Test experiment factory creation."""

    @ExperimentFactory.register("factory_test")
    class FactoryTestExperiment(Experiment):
        def train(self):
            pass

        def evaluate(self):
            return {}

        def inference(self):
            pass

    config = ExperimentConfig(
        experiment_name="factory_exp",
        experiment_type="factory_test",
        output_dir=temp_dir,
    )

    exp = ExperimentFactory.create(config)
    assert isinstance(exp, FactoryTestExperiment)
    assert exp.config.experiment_name == "factory_exp"


def test_config_validation_invalid_log_level():
    """Test that invalid log level raises error."""
    with pytest.raises(ValueError, match="log_level must be one of"):
        ExperimentConfig(
            experiment_name="test",
            experiment_type="dummy",
            log_level="INVALID",
        )


def test_config_validation_invalid_device():
    """Test that invalid device raises error."""
    with pytest.raises(ValueError, match="device must be one of"):
        ExperimentConfig(
            experiment_name="test",
            experiment_type="dummy",
            device="invalid_device",
        )


def test_experiment_setup_called_twice(basic_config):
    """Test that calling setup twice raises error."""
    exp = DummyExperiment(basic_config)
    exp.setup()

    with pytest.raises(RuntimeError, match="already been called"):
        exp.setup()


def test_checkpoint_before_setup(basic_config):
    """Test that saving checkpoint before setup raises error."""
    exp = DummyExperiment(basic_config)

    with pytest.raises(RuntimeError, match="Must call setup"):
        exp.save_checkpoint()
