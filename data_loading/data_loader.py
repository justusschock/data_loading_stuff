import numpy as np
from data_loading.dataset import AbstractDataset, DictDataset, IterableDataset
from collections import Iterable


class DataLoader:
    def __init__(self, data):
        self._process_id = None

        # do nothing if data
        if isinstance(data, AbstractDataset):
            dataset = data

        else:
            # wrap it into dataset depending on datatype
            if isinstance(data, dict):
                dataset = DictDataset(data)
            elif isinstance(data, Iterable):
                dataset = IterableDataset(data)
            else:
                raise TypeError("Invalid dataset type: %s"
                                % type(data).__name__)

        self.dataset = dataset

    def __call__(self, indices):
        # get data for all indices
        data = [self.dataset[idx] for idx in indices]

        data_dict = {}

        # concatenate dict entities by keys
        for _result_dict in data:
            for key, val in _result_dict.items():
                if key in data_dict.keys():
                    _result_dict[key].append(val)
                else:
                    _result_dict[key] = [val]

        # convert list to numpy arrays
        for key, val_list in data_dict.items():
            data_dict[key] = np.asarray(val_list)

        return

    @property
    def process_id(self):
        if self._process_id is None:
            return 0
        return self._process_id

    @process_id.setter
    def process_id(self, new_id):
        if self._process_id is not None:
            raise AttributeError("Attribute 'process_id' can be set only once")

        self._process_id = new_id

