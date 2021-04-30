import os
import sys
from enum import Enum
import torch
import numpy as np
import tqdm

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
DATA_ROOT = os.getenv('DATA_ROOT', './image_net/')
SELECTED = 'selected/'
MOBILE = 'mobile/'
WEIGHTS = 'weights/'

tqdm_settings = {
    'file': sys.stdout,
    'bar_format': '{l_bar}{bar:30}{r_bar}{bar:-10b}'
}


class Backend(Enum):
    TENSOR_FLOW = 'tf'
    PY_TORCH = 'torch'


class ModelType(Enum):
    MOBILE_NET_V2 = 'mobilenet_v2'
    RES_NET50 = 'resnet50'
    INCEPTION_V3 = 'inception_v3'


def get_image_size():
    return int(os.getenv('IMAGE_SIZE', 224))


def set_image_size(size):
    os.environ['IMAGE_SIZE'] = str(size)


def progress(*args, **kwargs):
    """
    A shortcut for tqdm(xrange(*args), **kwargs).
    On Python3+ range is used instead of xrange.
    """
    return tqdm.trange(*args, **kwargs, **tqdm_settings)


def channel_first(x: np.ndarray):
    if x.ndim == 3:
        return x.transpose((2, 0, 1))
    elif x.ndim == 4:
        return x.transpose((0, 3, 1, 2))
    else:
        raise TypeError('bad dimensions')


def channel_last(x: np.ndarray):
    if x.ndim == 3:
        return x.transpose((1, 2, 0))
    elif x.ndim == 4:
        return x.transpose((0, 2, 3, 1))
    else:
        raise TypeError('bad dimensions')


def cuda(obj):
    if torch.cuda.is_available():
        return obj.cuda()
    return obj


def cpu(obj):
    if torch.cuda.is_available():
        return obj.cpu()
    return obj
