import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    if not HAS_PIL:
        pytest.skip("PIL not available")
    return Image.new('RGB', (224, 224), color='red')


@pytest.fixture
def sample_tensor():
    """Create a sample torch tensor for testing."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array for testing."""
    if not HAS_NUMPY:
        pytest.skip("numpy not available")
    return np.random.randn(224, 224, 3)


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.batch_size = 32
    config.learning_rate = 0.001
    config.num_epochs = 10
    config.device = 'cpu'
    config.data_dir = '/tmp/data'
    config.output_dir = '/tmp/output'
    return config


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = MagicMock()
    model.eval.return_value = None
    model.train.return_value = None
    if HAS_TORCH:
        model.forward.return_value = torch.randn(1, 1000)
        model.parameters.return_value = [torch.randn(10, 10)]
    else:
        model.forward.return_value = Mock()
        model.parameters.return_value = [Mock()]
    return model


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader for testing."""
    dataloader = MagicMock()
    if HAS_TORCH:
        sample_batch = {
            'image': torch.randn(2, 3, 224, 224),
            'text': torch.randint(0, 1000, (2, 50)),
            'label': torch.randint(0, 10, (2,))
        }
    else:
        sample_batch = {
            'image': Mock(),
            'text': Mock(),
            'label': Mock()
        }
    dataloader.__iter__.return_value = iter([sample_batch])
    dataloader.__len__.return_value = 1
    return dataloader


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return [
        "A cat sitting on a mat",
        "A dog running in the park", 
        "A bird flying in the sky"
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return [0, 1, 2]


@pytest.fixture
def mock_lightning_module():
    """Create a mock Lightning module."""
    module = MagicMock()
    if HAS_TORCH:
        module.device = torch.device('cpu')
    else:
        module.device = Mock()
    module.training = False
    module.eval.return_value = None
    module.train.return_value = None
    return module


@pytest.fixture
def sample_dataset_item():
    """Sample dataset item structure."""
    if HAS_TORCH:
        return {
            'image_id': '12345',
            'image': torch.randn(3, 224, 224),
            'text': 'A sample caption',
            'text_ids': torch.randint(0, 1000, (50,)),
            'label': torch.tensor(1)
        }
    else:
        return {
            'image_id': '12345',
            'image': Mock(),
            'text': 'A sample caption',
            'text_ids': Mock(),
            'label': Mock()
        }


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 2023, 2003, 1037, 7099, 102]
    tokenizer.decode.return_value = "this is a test"
    tokenizer.vocab_size = 30522
    return tokenizer


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    if HAS_TORCH:
        torch.manual_seed(42)
    if HAS_NUMPY:
        np.random.seed(42)


@pytest.fixture
def mock_file_system(temp_dir):
    """Create a mock file system structure for testing."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    (data_dir / "train.json").write_text('{"data": "train"}')
    (data_dir / "val.json").write_text('{"data": "val"}')
    (data_dir / "test.json").write_text('{"data": "test"}')
    
    return {
        'data_dir': data_dir,
        'train_file': data_dir / "train.json",
        'val_file': data_dir / "val.json", 
        'test_file': data_dir / "test.json"
    }