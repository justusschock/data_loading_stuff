# basic imports
from data_loading.data_loader import DataLoader
from data_loading.dataset import AbstractDataset, IterableDataset, DictDataset, BaseCacheDataset, \
    BaseExtendCacheDataset, BaseLazyDataset, ConcatDataset, Nii3DCacheDatset, \
    Nii3DLazyDataset
from data_loading.augmenter import Augmenter
from data_loading.data_manager import DataManager
from data_loading.load_utils import  LoadSample, LoadSampleLabel

from data_loading.sampler import *


from delira import get_backends as _get_backends

# If torch backend is available: Import Torchvision dataset
if "TORCH" in _get_backends():
    from data_loading.dataset import TorchvisionClassificationDataset


# if numba is installed: Import Numba Transforms
try:
    from delira.data_loading.numba_transform import NumbaTransform, \
        NumbaTransformWrapper, NumbaCompose
except ImportError:
    pass
