import pytest
import os
import zarr
import torch
import torch.nn as nn
import json
from pathlib import Path


from drivetorch import (
    store,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 10)
        self.register_buffer('test_buffer', torch.tensor([1]))

    def forward(self, x):
        return self.net(x) + self.test_buffer


class TestStore:
    r"""
    :func:`store` unit tests.
    """
    def test_store_directory(self):
        model = SimpleModel()
        model_path = Path('model_test')
        store(model, storeinfo=model_path)
        assets = os.listdir(model_path)
        assert 'metadata.json' in assets
        with open(model_path / 'metadata.json', 'rb') as f:
            hash_map = json.load(f)['hash_map']
        for hash_value in hash_map.values():
            assert hash_value in assets

    def test_store_data(self):
        model = SimpleModel()
        model_path = Path('model_test')
        store(model, storeinfo=model_path)
        with open(model_path / 'metadata.json', 'rb') as f:
            hash_map = json.load(f)['hash_map']

        net_weight = zarr.open(model_path / hash_map['net.weight'], 'r')
        assert torch.allclose(torch.from_numpy(net_weight[:]), model.net.weight)
        net_bias = zarr.open(model_path / hash_map['net.bias'], 'r')
        assert torch.allclose(torch.from_numpy(net_bias[:]), model.net.bias)
        test_buffer = zarr.open(model_path / hash_map['test_buffer'], 'r')
        assert torch.allclose(torch.from_numpy(test_buffer[:]), model.test_buffer)
