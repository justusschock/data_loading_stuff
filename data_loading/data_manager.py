import inspect
import logging

from batchgenerators.transforms import AbstractTransform

from multiprocessing import Queue

from delira import get_current_debug_mode
from data_loading.data_loader import DataLoader
from data_loading.sampler import SequentialSampler, AbstractSampler
from data_loading.augmenter import Augmenter
from data_loading.dataset import DictDataset, IterableDataset, AbstractDataset
from collections import Iterable

logger = logging.getLogger(__name__)


class DataManager(object):
    """
    Class to Handle Data
    Creates Dataset (if necessary), Dataloader and Augmenter

    """

    def __init__(self, data, batch_size, n_process_augmentation,
                 transforms, sampler_cls=SequentialSampler,
                 drop_last=False, data_loader_cls=None,
                 **sampler_kwargs):
        """

        Parameters
        ----------
        data : str or Dataset
            if str: Path to data samples
            if dataset: Dataset
        batch_size : int
            Number of samples per batch
        n_process_augmentation : int
            Number of processes for augmentations
        transforms :
            Data transformations for augmentation
        sampler_cls : AbstractSampler
            class defining the sampling strategy
        drop_last : bool
            whether to drop the last (possibly smaller) batch
        data_loader_cls : subclass of SlimDataLoaderBase
            DataLoader class
        **sampler_kwargs :
            other keyword arguments (passed to sampler_cls)

        Raises
        ------
        AssertionError
            ``data_loader_cls`` is not :obj:`None` and not a subclass of
            `DataLoader`
        TypeError
            ``data`` is not a Dataset object and not of type dict or iterable

        See Also
        --------
        :class:`AbstractDataset`

        """

        # Instantiate Hidden variables for property access
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self._batch_size = None
        self._n_process_augmentation = None
        self._transforms = None
        self._data_loader_cls = None
        self._sampler = None
        self.drop_last = drop_last

        # set actual values to properties
        self.batch_size = batch_size

        self.n_process_augmentation = n_process_augmentation
        self.transforms = transforms

        if data_loader_cls is None:
            logger.info("No dataloader Class specified. Using DataLoader")
            data_loader_cls = DataLoader
        else:
            assert inspect.isclass(data_loader_cls), \
                "data_loader_cls must be class not instance of class"
            assert issubclass(data_loader_cls, DataLoader), \
                "dater_loader_cls must be subclass of DataLoader"

        self.data_loader_cls = data_loader_cls

        if not isinstance(data, AbstractDataset):
            if isinstance(data, dict):
                data = DictDataset(data)
            elif isinstance(data, Iterable):
                data = IterableDataset(data)
            else:
                raise TypeError("Invalid Data type given: %s"
                                % type(data).__name__)

        self.data = data

        assert inspect.isclass(sampler_cls) and issubclass(sampler_cls,
                                                           AbstractSampler)
        self.sampler = sampler_cls.from_dataset(self.data, **sampler_kwargs)

    def get_batchgen(self, seed=1):
        """
        Create DataLoader and Batchgenerator

        Parameters
        ----------
        seed : int
            seed for Random Number Generator

        Returns
        -------
        Augmenter
           The actual iterable batchgenerator

        Raises
        ------
        AssertionError
            :attr:`BaseDataManager.n_batches` is smaller than or equal to zero

        """
        assert self.n_batches > 0

        sampler_queues = []

        for idx in range(self.n_process_augmentation):
            sampler_queues.append(Queue())

        data_loader = self.data_loader_cls(
            self.data
        )

        return Augmenter(data_loader=data_loader,
                         sampler=self.sampler,
                         num_processes=self.n_process_augmentation,
                         transforms=self.transforms,
                         seed=seed,
                         drop_last=self.drop_last
                         )

    def get_subset(self, indices):
        """
        Returns a Subset of the current datamanager based on given indices

        Parameters
        ----------
        indices : iterable
            valid indices to extract subset from current dataset

        Returns
        -------
        :class:`BaseDataManager`
            manager containing the subset

        """

        subset_kwargs = {
            "batch_size": self.batch_size,
            "n_process_augmentation": self.n_process_augmentation,
            "transforms": self.transforms,
            "sampler_cls": self.sampler.__class__,
            "data_loader_cls": self.data_loader_cls,
        }

        return self.__class__(
            self.data.get_subset(indices),
            **subset_kwargs)

    def update_state_from_dict(self, new_state: dict):
        """
        Updates internal state and therfore the behavior from dict.
        If a key is not specified, the old attribute value will be used

        Parameters
        ----------
        new_state : dict
            The dict to update the state from.
            Valid keys are:

                * ``batch_size``
                * ``n_process_augmentation``
                * ``data_loader_cls``
                * ``sampler``
                * ``sampling_kwargs``
                * ``transforms``

            If a key is not specified, the old value of the corresponding
            attribute will be used

        Raises
        ------
        KeyError
            Invalid keys are specified

        """

        # update batch_size if specified
        self.batch_size = new_state.pop("batch_size", self.batch_size)
        # update n_process_augmentation if specified
        self.n_process_augmentation = new_state.pop(
            "n_process_augmentation", self.n_process_augmentation)
        # update data_loader_cls if specified
        self.data_loader_cls = new_state.pop("data_loader_cls",
                                             self.data_loader_cls)
        # update
        new_sampler = new_state.pop("sampler", None)
        if new_sampler is not None:
            self.sampler = new_sampler.from_dataset(
                self.data,
                **new_state.pop("sampling_kwargs", {}))
        self.transforms = new_state.pop("transforms", self.transforms)

        if new_state:
            raise KeyError("Invalid Keys in new_state given: %s"
                           % (','.join(map(str, new_state.keys()))))

    @property
    def batch_size(self):
        """
        Property to access the batchsize

        Returns
        -------
        int
            the batchsize
        """

        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        """
        Setter for current batchsize, casts to int before setting the attribute

        Parameters
        ----------
        new_batch_size : int, Any
            the new batchsize; should be int but can be of any type that can be
            casted to an int

        """

        self._batch_size = int(new_batch_size)

    @property
    def n_process_augmentation(self):
        """
        Property to access the number of augmentation processes

        Returns
        -------
        int
            number of augmentation processes
        """

        if get_current_debug_mode():
            return 1
        return self._n_process_augmentation

    @n_process_augmentation.setter
    def n_process_augmentation(self, new_process_number):
        """
        Setter for number of augmentation processes, casts to int before
        setting the attribute


        Parameters
        ----------
        new_process_number : int, Any
            new number of augmentation processes; should be int but can be of
            any type that can be casted to an int

        """

        self._n_process_augmentation = int(new_process_number)

    @property
    def transforms(self):
        """
        Property to access the current data transforms

        Returns
        -------
        None, ``AbstractTransform``
            The transformation, can either be None or an instance of
            ``AbstractTransform``
        """

        return self._transforms

    @transforms.setter
    def transforms(self, new_transforms):
        """
        Setter for data transforms, assert if transforms are of valid type
        (either None or instance of ``AbstractTransform``)

        Parameters
        ----------
        new_transforms : None, ``AbstractTransform``
            the new transforms

        """

        assert new_transforms is None or isinstance(new_transforms,
                                                    AbstractTransform)

        self._transforms = new_transforms

    @property
    def data_loader_cls(self):
        """
        Property to access the current data loader class

        Returns
        -------
        type
            Subclass of ``DataLoader``
        """

        return self._data_loader_cls

    @data_loader_cls.setter
    def data_loader_cls(self, new_loader_cls):
        """
        Setter for current data loader class, asserts if class is of valid type
        (must be a class and a subclass of ``DataLoader``)

        Parameters
        ----------
        new_loader_cls : type
            the new data loader class

        """

        assert inspect.isclass(new_loader_cls) and issubclass(
            new_loader_cls, DataLoader)

        self._data_loader_cls = new_loader_cls

    @property
    def sampler(self):
        """
        Property to access the current sampler

        Returns
        -------

        :class:`AbstractSampler`
            the current sampler
        """

        return self._sampler

    @sampler.setter
    def sampler(self, new_sampler):
        """
        Setter for current sampler.
        If a valid class instance is passed, the sampler is simply assigned, if
        a valid class type is passed, the sampler is created from the dataset

        Parameters
        ----------
        new_sampler : :class:`AbstractSampler`, type
            instance or class object of new sampler

        Raises
        ------
        ValueError
            Neither a valid class instance nor a valid class type is given

        """

        if inspect.isclass(new_sampler) and issubclass(new_sampler,
                                                       AbstractSampler):
            self._sampler = new_sampler.from_dataset(self.data)

        elif isinstance(new_sampler, AbstractSampler):
            self._sampler = new_sampler

        else:
            raise ValueError("Given Sampler is neither a subclass of \
                            AbstractSampler, nor an instance of a sampler ")

    @property
    def n_samples(self):
        """
        Number of Samples

        Returns
        -------
        int
            Number of Samples

        """
        return len(self.sampler)

    @property
    def n_batches(self):
        """
        Returns Number of Batches based on batchsize and number of samples

        Returns
        -------
        int
            Number of Batches

        Raises
        ------
        AssertionError
            :attr:`BaseDataManager.n_samples` is smaller than or equal to zero

        """
        assert self.n_samples > 0

        n_batches = self.n_samples // self.batch_size

        truncated_batch = self.n_samples % self.batch_size

        n_batches += int(bool(truncated_batch) and not self.drop_last)

        return n_batches
