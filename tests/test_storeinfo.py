import pytest
import os
from multiprocessing import current_process


from drivetorch import (
    init_storeinfo,
    StoreInfo,
    StoreInfoGenerator,
)



class TestStoreInfo:
    r"""
    :class:`StoreInfo` unit tests.
    """
    @pytest.mark.parametrize(
        'store',
        [None, 'storeinfo_test']
    )
    def test_init(self, store):
        StoreInfo(store=store)

    @pytest.mark.parametrize(
        'store,pre_init',
        [
            (None, False),
            (None, True),
            ('storeinfo_test', False),
            ('storeinfo_test', True),
        ]
    )
    def test_constructor(self, store, pre_init):
        if pre_init:
            store = StoreInfo(store=store)
        store_info = init_storeinfo(store=store)
        assert isinstance(store_info, StoreInfo)

    def test_temp_storage(self):
        # get cwd, change and run, change back
        store = init_storeinfo()
        files = os.listdir('.')
        assert files == ['.drivetorch_temp']
        pid_directory = os.listdir('.drivetorch_temp/')
        assert pid_directory == [str(current_process().pid)]
        assert store['temporary']

    def test_attributes(self):
        store = 'tmp_store'
        store_info = StoreInfo(store=store)
        attributes = ['path', 'identifier', 'temporary']
        for attribute in attributes:
            assert attribute in store_info
        non_attributes = ['store']
        for attribute in non_attributes:
            assert attribute not in store_info
        assert store_info.store == store_info['path'] / store_info['identifier']

class TestStoreInfoGenerator:
    r"""
    :class:`StoreInfoGenerator` unit tests.
    """
    @pytest.mark.parametrize(
        'store',
        [None, 'storeinfo_test']
    )
    def test_init(self, store):
        StoreInfoGenerator(store=store)

    @pytest.mark.parametrize(
        'store,identifier',
        [
            (None,'param1'),
            ('storeinfo_test','param1'),
        ]
    )
    def test_get_storeinfo(self, store, identifier):
        storeinfo_generator = StoreInfoGenerator(store=store)
        storeinfo = storeinfo_generator.get_storeinfo(identifier=identifier)
        assert storeinfo['identifier'] == identifier

    @pytest.mark.parametrize(
        'store',
        [None, 'storeinfo_test']
    )
    def test_get_storeinfo_fail(self, store):
        storeinfo_generator = StoreInfoGenerator(store=store)
        with pytest.raises(TypeError):
            storeinfo = storeinfo_generator.get_storeinfo()
