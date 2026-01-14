from torch.utils.data import Dataset
import abc
import torch

class BaseVSFDataset(Dataset, abc.ABC):
    """
    Abstract Base Class for Variable Subset Forecasting Datasets.
    Enforces the standard output format:
    x, y, mask shapes: (Time, Node, Channel)
    """
    @abc.abstractmethod
    def __getitem__(self, index):
        """
        Must return a dict with keys:
        - 'x': torch.Tensor (Time, Node, Channel) - Input features (masked)
        - 'y': torch.Tensor (Time, Node, Channel) - Target (ground truth)
        - 'mask': torch.Tensor (Time, Node, Channel) - 0/1 mask (1=observed)
        - 'time_stamp': (Optional) Time encoding
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        pass
