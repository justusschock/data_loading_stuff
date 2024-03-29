import json
import logging
import os
from abc import abstractmethod

import SimpleITK as sitk

from delira.utils.decorators import make_deprecated

logger = logging.getLogger(__name__)


def load_nii(path):
    """
    Loads a single nii file
    Parameters
    ----------
    path: str
        path to nii file which should be loaded

    Returns
    -------
    np.ndarray
        numpy array containing the loaded data
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


class BaseLabelGenerator(object):
    """
    Base Class to load labels from json files

    .. deprecated-removed: 0.3.3 0.3.5

    """

    @make_deprecated('dict containing labels')
    def __init__(self, fpath):
        """

        Parameters
        ----------
        fpath : str
            filepath to json file

        Raises
        ------
        AssertionError
            `fpath` does not end with 'json'

        """
        assert fpath.endswith('json')
        self.fpath = fpath
        self.data = self._load()

    def _load(self):
        """
        Private Helper function to load the file

        Returns
        -------
        Any
            loaded values from file

        """
        with open(os.path.join(self.fpath), 'r') as f:
            label = json.load(f)
        return label

    @abstractmethod
    def get_labels(self):
        """
        Abstractmethod to get labels from class

        Raises
        ------
        NotImplementedError
            if not overwritten in subclass

        """
        raise NotImplementedError()
