import torch.nn as nn
import abc

class BaseVSFModel(nn.Module, abc.ABC):
    """
    Abstract Base Class for VSF Models.
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, batch):
        """
        Standard forward pass.
        args:
            batch: Dict containing 'x', 'y', 'mask', etc. from DataLoader.
                   x shape: (Batch, Time, Node, Channel)
        returns:
            Dict containing:
            - 'pred': Predictions (Batch, Time, Node, Channel)
            - 'imputed': (Optional) Imputed values
            - 'loss': (Optional) Calculated loss if permitted
        """
        pass
