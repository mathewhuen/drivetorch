from copy import deepcopy
from multiprocessing import current_process
from pathlib import Path
import json



def init_storeinfo(store=None, identifier=None, general=False, *args, **kwargs):
    r"""
    Convenience function for initializing a :class:`StoreInfo` instance
    if not already given.

    Args:
        store (path-like or StoreInfo, optional): If of type :class:`StoreInfo`\,
            then just returns store. Otherwise constructs a new instance of
            :class:`StoreInfo` from given parameters.
    """
    if isinstance(store, StoreInfo):
        return store
    if isinstance(store, dict):
        store.update(kwargs)
        return StoreInfo(identifier=identifier, *args, **kwargs)
    else:
        type_error = (
            "'store' should be of None, path-like, a dict, or StoreInfo. But "
            f"passed as type: {type(store)}."
        )
        assert store is None or isinstance(store, str) or isinstance(store, Path), type_error
        if store is None and 'path' in kwargs:
            path = kwargs.pop('path')
        else:
            path = store
        if general:
            return ModelStoreInfo(path, **kwargs)
        else:
            return StoreInfo(path, identifier=identifier, **kwargs)


def parse_store_path(path=None, identifier=None, ignore_identifier=False):
    r"""
    Returns the given store path if not None. Otherwise, creates a
    store path in .drivetorch_temp/ and creates the specified
    directories if they do not exist.
    This is a convenience function for creating temporary paths when
    none are explicitly given.

    Args:
        path (Any, optional): Given path. If not None, returns `path`\.
            Otherwise, returns a path to a folder in .drivetorch_temp/

    Returns:
        path-like: Path to the directory to which :class:`DriveTensor`\s
            should be saved.
    """
    temporary = False
    if ignore_identifier:
        identifier = None
    elif identifier is None:
        identifier = str(StoreInfo._counter)
        StoreInfo._counter += 1
    if path is None:
        path = Path('.drivetorch_temp/') / str(current_process().pid)
        temporary = True
    elif not isinstance(path, Path):
        path = Path(path)

    mkdir_path = path / identifier if identifier is not None else path
    mkdir_path.mkdir(parents=True, exist_ok=True)

    return path, identifier, temporary


class StoreInfo(dict):
    r"""
    Object for parsing and holding storage parameters for :class:`DriveTensor`\.
    The current implementation will likely change as more features are added.

    This object is used as kwargs for zarr storage.
    """

    _counter = 0

    def __init__(
            self,
            path=None,
            identifier=None,
            ignore_identifier=False,
            **kwargs,
    ):
        r"""
        Initializes :class:`StoreInfo`\.

        Args:
            path (Any, optional): str or path object pointing to
                directory in which tensors should be stored.
                If not given, writes to a directory in .drivetorch_temp/
        """
        super(StoreInfo, self).__init__()
        self.store_type = 'directory'  # currently, the only supported type
        path, identifier, temporary = parse_store_path(
            path,
            identifier,
            ignore_identifier,
        )
        self['path'] = path
        self['identifier'] = identifier
        self['temporary'] = temporary

        # compression
        self['compressor'] = kwargs.get('compressor', 'default')

        # chunks
        # assume loading full array if not specified
        self['chunks'] = kwargs.get('chunks', False)

    def _get_store(self):
        return self['path'] / self['identifier']

    def _get_hashpath(self):
        if self.store_type == 'directory':
            return self['path'] / 'hash_map.json'
        else:
            raise NotImplementedError

    def __getattr__(self, attr, *args, **kwargs):
        if attr == 'store':
            return self._get_store()
        elif attr == 'hashpath':
            return self._get_hashpath()
        return super().__getattr__(attr, *args, **kwargs)

    def __getitem__(self, item, *args, **kwargs):
        if item == 'store':
            return self._get_store()
        elif item == 'hashpath':
            return self._get_hashpath()
        return super().__getitem__(item, *args, **kwargs)

    def __setattr__(self, attr, value, *args, **kwargs):
        protected_attributes = [
            'store',
            'hashpath',
        ]
        if attr in protected_attributes:
            message = (
                f"The '{attr}' attribute of {self.__class__.__name__} cannot "
                "be set directly."
            )
            raise RuntimeError(message)
        return super().__setattr__(attr, value, *args, **kwargs)


class ModelStoreInfo(StoreInfo):
    r"""
    Helper class for generating multiple :class:`StoreInfo` instances for the same
    model.
    """

    def __init__(self, path=None, **kwargs):
        r"""
        Initializes :class:`ModelStoreInfo` class.

        This class should be used as a general storage kwargs dict for one
        model, and the :meth:`get_storeinfo` should be used to get
        parameter-specific :class:`StoreInfo` instances.
        """
        if 'identifier' in kwargs:
            del kwargs['identifier']
        if 'ignore_identifier' in kwargs:
            del kwargs['ignore_identifier']
        super().__init__(
            path=path,
            identifier=None,
            ignore_identifier=True,
            **kwargs,
        )
        #self._kwargs = deepcopy(kwargs)
        #if store is not None
        #self._kwargs.update({'store': store})

    def _get_store(self):
        raise NotImplementedError

    def get_storeinfo(self, identifier):
        r"""
        Returns an instance of :class:`StoreInfo` with the given identifier.

        Args:
            identifier (str): The identifier to use for the new instance of
                :class:`StoreInfo`\. Should be a nonempty string.
        """
        error = (
            "'identifier' should be a nonempty string but was "
            f"'{identifier}'"
        )
        assert isinstance(identifier, str) and identifier != '', error
        store = deepcopy(self)
        store['identifier'] = identifier
        return init_storeinfo(**store)
