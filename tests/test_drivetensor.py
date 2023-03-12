import pytest
import os
import numpy as np


from drivetorch import DriveTensor


class TestDriveTensor:
    r"""
    :class:`DriveTensor` Unit tests.
    """

    @pytest.mark.parametrize(
        'data,store_data',
        [
            (np.arange(3).astype('float'), {'store': 'data'}),
            (np.arange(9).reshape(1, 3, 1, 3).astype('float'), 'data'),
            (np.random.normal(0, 1, size=(5, 10)), {'store': 'data'}),
            (np.random.normal(0, 1, size=(5, 10)), None),
        ]
    )
    def test_init_success(self, data, store_data):
        tensor = DriveTensor(
            data=data,
            store_data=store_data,
        )
        pth = tensor._tensor.store.dir_path()
        assets = os.listdir(pth)
        assert len(assets) > 0
        assert '.zarray' in assets

    @pytest.mark.parametrize(
        'data,store_data',
        [
            (None, {'store': 'data'}),
            (None, dict()),
        ]
    )
    def test_init_fail(self, data, store_data):
        with pytest.raises(AssertionError) as e_info:
            DriveTensor(
                data=data,
                store_data=store_data,
            )
