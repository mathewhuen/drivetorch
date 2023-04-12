r"""
The :mod:`store_load` module contains functions for storing and loading
PyTorch models to a drive.

Currently, the only implemented storage function is :func:`store`\, but
it requires loading the entire model before it can be stored.
Therefore, models may need to be stored and then transfered to low-
memory devices.

We will release several optimizations in the future to help reduce the
memory usage in the initial store and load functions.
"""


import zarr
from pathlib import Path
import json
from copy import deepcopy
from torch.nn import Module
from typing import Optional, Union
from hashlib import sha256
from pickle import dumps


# from drivetorch.handler import DriveTensorHandler
from drivetorch.drivetensor import DriveTensor
from drivetorch.storeinfo import (
    init_storeinfo,
    ModelStoreInfo,
)
from drivetorch.utils import hash_tensor, set_nested


# TODO:
#    - Add support for saving the handler object
#    - Add TempParameter support (and as a context manager)
#    - (Later) add custom pickler


def store(
        model: Module,
        storeinfo: Optional[Union[str, dict, ModelStoreInfo]] = None,
        # handler: Optional[DriveTensorHandler] = None,
        **kwargs,
):
    r"""
    Store a PyTorch model.

    Args:
        model (`torch.nn.Module`): The model to store.
        storeinfo (str or dict or :class:`ModelStoreInfo`, optional):
            Parameters for zarr storage, to be shared for all model
            parameters and buffers.
            If a str, dict, or ModelStoreInfo, it will be used to create
            an instance of :class:`ModelStoreInfo`\.
            If None, model parameters and buffers will be saved to a
            local .drivetorch_temp directory.
            Defaults to None.
    """
    model_storeinfo = init_storeinfo(storeinfo, general=True)

    named_parameters = list(model.named_parameters()) + list(model.named_buffers())
    hash_map = dict()
    for name, parameter in named_parameters:
        param_hash = hash_tensor(parameter[:])
        hash_map[name] = param_hash
        storeinfo_instance = model_storeinfo.get_storeinfo(param_hash)
        drive_tensor = DriveTensor(
            data=parameter,
            store_data=storeinfo_instance,
            # handler=handler,
            as_param=True,
        )
        set_nested(model, name.split('.'), drive_tensor)
    metadata = {'hash_map': hash_map}
    model_storeinfo.store_metadata(metadata)
    return model


def load(
        model: Module,
        storeinfo: Union[str, dict, ModelStoreInfo],
):
    model_storeinfo = init_storeinfo(storeinfo, general=True)
    metadata = model_storeinfo.load_metadata()
    named_parameters = list(model.named_parameters()) + list(model.named_buffers())
    for name, parameter in named_parameters:
        param_hash = metadata['hash_map'][name]
        storeinfo_instance = model_storeinfo.get_storeinfo(param_hash)
        drive_tensor = DriveTensor(
            store_data=storeinfo_instance,
            from_store=True,
            as_param=True,
        )
        set_nested(model, name.split('.'), drive_tensor)
