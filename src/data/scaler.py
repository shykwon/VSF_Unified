import torch
import numpy as np

class StandardScaler:
    """
    Standard Scaler (Mean/Std) for PyTorch Tensors or Numpy Arrays.
    Follows sklearn API.
    """
    def __init__(self, mean=None, std=None, device='cpu'):
        self.mean = mean
        self.std = std
        self.device = device
        
    def fit(self, data):
        """
        Compute mean and std from data.
        data: (N_samples, ...)
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            
        self.mean = torch.mean(data)
        self.std = torch.std(data)
        
        if self.std == 0:
            self.std = 1.0
            
    def transform(self, data):
        """
        Apply standardization: (data - mean) / std
        """
        if isinstance(data, np.ndarray):
            result = (data - self.mean.numpy()) / self.std.numpy()
            return result
        elif isinstance(data, torch.Tensor):
            device = data.device
            return (data - self.mean.to(device)) / self.std.to(device)
        else:
            raise TypeError("Data must be numpy array or torch tensor")
            
    def inverse_transform(self, data):
        """
        Apply inverse standardization: data * std + mean
        """
        if isinstance(data, np.ndarray):
            result = (data * self.std.numpy()) + self.mean.numpy()
            return result
        elif isinstance(data, torch.Tensor):
            device = data.device
            return (data * self.std.to(device)) + self.mean.to(device)
        else:
            raise TypeError("Data must be numpy array or torch tensor")
