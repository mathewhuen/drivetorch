import pytest
import os
import numpy as np


from drivetorch import DriveTensor


class TestDriveTensor:
    r"""
    :class:`DriveTensor` Unit tests.
    """
    @staticmethod
    def add_store_path(store_data, tmp_path):
        if 'store' in store_data:
            store_data['store'] = tmp_path / store_data['store']
        return store_data

    @pytest.mark.parametrize(
        'data,store_data',
        [
            (np.arange(3).astype('float'), {'store': 'data'}),
            (np.random.normal(0, 1, size=(5, 10)), {'store': 'data'}),
        ]
    )
    def test_init_success(self, data, store_data, tmp_path):
        self.add_store_path(store_data, tmp_path)
        DriveTensor(
            data=data,
            store_data=store_data,
        )
        assets = os.listdir(store_data['store'])
        assert len(assets) > 0
        assert '.zarray' in assets

    @pytest.mark.parametrize(
        'data,store_data',
        [
            (None, {'store': 'data'}),
            (np.random.normal(0, 1, size=(5, 10)), dict()),
        ]
    )
    def test_init_fail(self, data, store_data, tmp_path):
        self.add_store_path(store_data, tmp_path)
        with pytest.raises(AssertionError) as e_info:
            DriveTensor(
                data=data,
                store_data=store_data,
            )
