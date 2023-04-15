r"""
General utility functions.
"""


from torch import Tensor, from_numpy
from torch.nn import Module, Parameter
from numpy import ndarray
from typing import List, Union
from hashlib import sha256
from pickle import dumps


def set_nested(
        module: Module,
        atts: List[str],
        tensor: Union[Tensor, Parameter],
):
    r"""
    Set a nested parameter or buffer with `tensor`\.

    Args:
        module (Module): Module with a descendent parameter or buffer
            to set.
        atts (list[str]): List of nested child names to specify a
            descendent parameter or buffer.
            For example, the list ['attention', 'linear_1', 'weight']
            will update the weight parameter or buffer of the
            module.attention.linear_1 submodule.
        tensor (Tensor or Parameter): The tensor with which to update
            the parameter or buffer.
    """
    placeholder = module
    for att in atts[:-1]:
        placeholder = getattr(placeholder, att)
    setattr(placeholder, atts[-1], tensor)


def hash_tensor(data: Union[ndarray, Tensor, Parameter]):
    r"""
    Hash a tensor-like object.

    First convert the data to a str with `pickle.dumps` then hash with
    `hashlib.sha256`\.

    Args:
        data (ndarray or Tensor or Parameter): The data to hash.

    Returns:
        str: Hash of the input data.
    """
    if isinstance(data, ndarray):
        data = from_numpy(data)
    return sha256(dumps(data[:])).hexdigest()
