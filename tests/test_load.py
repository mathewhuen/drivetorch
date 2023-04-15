import pytest
import torch
import torch.nn as nn


from drivetorch import (
    store,
    load,
)


from utils import temp_cwd


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 10)
        self.test_param = nn.Parameter(torch.tensor([1.]))
        self.register_buffer('test_buffer', torch.tensor([1.]))

    def forward(self, x):
        return self.net(x) + self.test_buffer


class TestLoad:
    r"""
    :func:`load` unit tests.
    """
    @staticmethod
    def reset_model_weights(model):
        for parameter in model.parameters():
            nn.init.normal_(parameter)

    @pytest.mark.parametrize(
        'storeinfo',
        [
            'model_to_load',  # str storeinfo
            {'path': 'model_to_load'},  # dict storeinfo
            None,  # default directory (based on PID) should work
        ]
    )
    @torch.no_grad()
    def test_load(self, storeinfo):
        model_1 = SimpleModel()
        model_2 = SimpleModel()
        self.reset_model_weights(model_2)

        tensor_in = torch.randn((1, 10))
        tensor_out = model_1(tensor_in)

        store(model_1, storeinfo=storeinfo)
        del model_1
        load(model_2, storeinfo=storeinfo)

        assert torch.allclose(model_2(tensor_in), tensor_out)
