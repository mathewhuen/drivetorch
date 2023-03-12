from copy import deepcopy
from multiprocessing import current_process
from pathlib import Path


def init_storeinfo(store=None, identifier=None, *args, **kwargs):
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
        return StoreInfo(store, identifier=identifier, **kwargs)


def parse_store_path(path=None, identifier=None):
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
    if identifier is None:
        identifier = str(current_process().pid)
    if path is None:
        path = Path('.drivetorch_temp/')
        temporary = True
    elif not isinstance(path, Path):
        path = Path(path)
    (path / identifier).mkdir(parents=True, exist_ok=True)
    return path, identifier, temporary


class StoreInfo(dict):
    r"""
    Object for parsing and holding storage parameters for :class:`DriveTensor`\.
    The current implementation will likely change as more features are added.

    This object is used as kwargs for zarr storage.
    """
    def __init__(self, store=None, identifier=None, **kwargs):
        r"""
        Initializes :class:`StoreInfo`\.

        Args:
            store (Any, optional): str or path object pointing to
                directory in which tensors should be stored.
                If not given, writes to a directory in .drivetorch_temp/
                Need to change `store` to `path`
        """
        super(StoreInfo, self).__init__()
        self.store_type = 'directory'  # currently, the only supported type
        path, identifier, temporary = parse_store_path(store, identifier)
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


class StoreInfoGenerator:
    r"""
    Helper class for generating multiple :class:`StoreInfo` instances for the same
    model.
    """

    def __init__(self, store=None, **kwargs):
        r"""
        Initializes :class:`StoreInfoGenerator`\.
        """
        self._store = init_storeinfo(store, **kwargs)

    def get_storeinfo(self, identifier):
        store = deepcopy(self._store)
        store['identifier'] = identifier
        return store
