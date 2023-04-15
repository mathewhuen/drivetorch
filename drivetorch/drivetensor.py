r"""
:class:`DriveTensor`\s are PyTorch tensor- and parameter-like zarr array
wrappers. They do not implement every functionality of PyTorch tensors, but
they should be usable as drop-in replacements for `nn.Parameter`\s for
inference.
"""


import torch
import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from warnings import warn
from typing import Optional, Union
from zarr import open as zarr_open
from pathlib import Path


# from drivetorch.handler import DriveTensorHandler  # For trace
from drivetorch.storeinfo import init_storeinfo, StoreInfo


# TODO:
#   - update how dtype is used. Don't modify the saved array, just recast when loaded?


dtype_map = {
    torch.float32: np.float32,
}


class DriveTensor:
    r"""
    Tensor wrapper for storing data on a drive. Implemented with zarr.

    :class:`DriveTensor` instances are read-only and not implemented for
    the standard arithmetic functions. They can be used, however, in
    any standard torch function or module.
    """
    def __init__(
        self,
        data: ArrayLike = None,
        store_data: Union[dict, str, StoreInfo] = None,
        from_numpy: bool = False,
        as_param: bool = False,
        # handler: Optional[DriveTensorHandler] = None,  # for trace
        from_store: bool = False,
        **kwargs,
    ):
        r"""
        Constructs :class:`DriveTensor` from array-like data.

        Args:
            data (`array_like`): Array-like data to wrap.
            store_data (dict, str, or :class:`StoreInfo`\, optional): Parameters
                for zarr storage. If a dict, it should include a `store` field
                that will be used as a storage path. If input is a str, it will
                be used as a storage path. If input is :class:`StoreInfo`, it
                will be treated as a dict.
            from_numpy (bool, optional): Data will be initialized with
                `torch.from_numpy` if True and with torch.as_tensor` if
                False.
            as_param (bool, optional): To be used as a PyTorch
                parameter.
            from_store (bool, optional): If True, does not perform any
                processing, instead only creating a persistent zarr
                array.
            **kwargs: Keyword arguments to use when initializing this
                object's tensor data.

        Example:
            >>> data = [1, 2, 3, 4]
            >>> t = DriveTensor(data, {'store': 'temp.data'})
            >>> t
            <DriveTensor(data=<zarr.core.Array (4,) int64 read-only>, store_data={'store': 'temp.data'})
        """
        if from_store:
            self._init_from_store(
                store_data=store_data,
                as_param=as_param,
                # handler=handler,
                **kwargs,
            )
        else:
            self._init(
                data=data,
                store_data=store_data,
                from_numpy=from_numpy,
                as_param=as_param,
                # handler=handler,
                **kwargs,
            )

    def _init_from_store(
            self,
            store_data: Union[dict, str, StoreInfo] = None,
            as_param: bool = False,
            # handler: Optional[DriveTensorHandler] = None,  # for trace
            **kwargs,
    ):
        self._tensor = zarr_open(  # make read-only
            store=store_data['store'],
            mode='r',
        )
        self._store_data = store_data
        self._kwargs = kwargs
        self._shape = self._tensor.shape
        self._is_param = as_param
        # self._handler = handler  # for trace

    def _init(
            self,
            data: ArrayLike,
            store_data: Union[dict, str, StoreInfo] = None,
            from_numpy: bool = False,
            as_param: bool = False,
            # handler: Optional[DriveTensorHandler] = None,  # for trace
            **kwargs,
    ):
        assert data is not None
        store_data = init_storeinfo(store_data)
        #if isinstance(store_data, str):
        #    store_data = {'store': store_data}
        #assert 'store' in store_data
        # consider not creating tensor first to reduce the number of copies
        if isinstance(data, DriveTensor):
            tensor_data = tensor = data._tensor[:]
            dtype = data._tensor.dtype
            shape = data.shape
        elif from_numpy:
            assert isinstance(data, ndarray)
            tensor = torch.from_numpy(data, **kwargs)
            dtype = data.dtype
            shape = tensor.shape
            tensor_data = tensor.detach().numpy()
        else:
            if isinstance(data, torch.Tensor):  # need to change how __class__ is used to get past torch.nn.Parameter since we need to parse DriveTensor here and all DriveTensors just look like normal tensors.
                if type(data) is torch.nn.Parameter or (hasattr(data, '_is_param') and data._is_param):
                    as_param = True
                if data.device.type != 'cpu':
                    warning = (
                        "DriveTensor input data is not on the CPU (it "
                        f'appears to be on device "{data.device.type}")'
                        ". Data will be moved to the CPU before storing"
                        "."
                    )
                    warn(warning)
                    data = data.cpu()
            tensor = torch.as_tensor(data, **kwargs)
            dtype = tensor.detach().numpy().dtype  # later use dtype map if not worrying about multiple copies.
            shape = tensor.shape
            tensor_data = tensor.detach().numpy()
        self._tensor = zarr_open(
            store=store_data['store'],
            mode='w',
            shape=shape,
            chunks=store_data.get('chunks', 1),
            dtype=store_data.get('dtype', dtype)
        )
        self._tensor[:] = tensor_data
        self._shape = shape
        self._store_data = store_data
        self._kwargs = kwargs
        self._tensor = zarr_open(  # make read-only
            store=store_data['store'],
            mode='r',
            # shape=tensor.shape,
            # chunks=store_data.get('chunks', False),
            # dtype=store_data.get('dtype', dtype)
        )
        self._is_param = as_param
        del tensor, tensor_data
        #self.non_modifiable_attributes = [
        #    '_tensor',
        #    '_shape',
        #    '_store_data',
        #    '_kwargs',
        #]
        # self._handler = handler  # for trace

    def __getattr__(self, attr):
        if attr == 'shape':
            return self._shape
        if attr == 'grad_fn':
            return None
        if attr == 'data':
            return self.get_tensor()
        # return getattr(self, attr)
        # return super().__getattr__
        raise AttributeError(f'Unknown attribute "{attr}"')

    def type(self, dtype=None, non_blocking=False, **kwargs) -> str or torch.Tensor:
        if non_blocking:
            warn(
                'non_blocking given as True but not yet implemented for '
                'class DriveTorch.'
            )
        if dtype is None:
            return self.__class__.__name__
        else:
            # this should check the dtype of DriveTensor, cast if wrong and return or just return if correct
            raise NotImplementedError

    #def __setattr__(self, attr, value):
    #    if attr in self.non_modifiable_attributes:
    #        raise AttributeError(
    #            f"attribute '{attr}' cannot be modified manually."
    #        )
    #    return setattr(self, attr, value)

    def __repr__(self):
        # make meaningful representation
        return f"DriveTensor(data={self._tensor}, store_data={self._store_data})"

    def __str__(self):
        return f"""<DriveTensor with shape {tuple(self._shape)} at "{self._store_data['store']}">"""

    def get_tensor(self):
        # if self._handler is not None:  # for trace
        #     self._handler.ping(self)  # for trace
        return torch.from_numpy(self._tensor[:], **self._kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = dict()
        args = [
            arg.get_tensor() if hasattr(arg, '_tensor') else arg
            for arg in args
        ]
        return func(*args, **kwargs)

    def detach(self):
        return self

    def requires_grad_(self, *args, **kwargs):
        return self

    @property  # hacky method to register as a tensor
    def __class__(self):
        return type('DriveTensor', (torch.Tensor, DriveTensor), {})
    #@classmethod
    #def __instancecheck__(cls, instance):
    #    breakpoint()
    #    if instance == torch.nn.Parameter and self._is_param:
    #        return True
    #    return isinstance(instance, cls)

    def __del__(self):
        #if self._store_data.store_type == 'directory':
        #    if 'drivetorch.temp' in self._store_data.store:
        #        # delete zarr object.
        #        # if final object in drive, delete directory
        pass

