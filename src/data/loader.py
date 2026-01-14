import torch
import numpy as np
import pandas as pd
import os
from src.core.dataset import BaseVSFDataset

def load_dataset(name, data_dir, window_size=12, horizon=12, stride=1, mode='train', train_ratio=0.7, val_ratio=0.1, limit_samples=None, scaler=None):
    """
    Factory function to load real-world datasets.
    Args:
        name: 'METR-LA', 'PEMS-BAY', 'SOLAR', 'TRAFFIC', 'ELECTRICITY'
        data_dir: Path to data/raw or similar
        limit_samples: If set, limit total data used to this number (for debugging)
        scaler: Optional StandardScaler instance. If valid/test mode, pas current fitted scaler.
    """
    name = name.upper()
    
    # Define file paths
    if name == 'METR-LA':
        path = os.path.join(data_dir, 'metr-la.h5')
        df = pd.read_hdf(path)
        data = df.values
    elif name == 'PEMS-BAY':
        path = os.path.join(data_dir, 'pems-bay.h5')
        df = pd.read_hdf(path)
        data = df.values
    elif name == 'SOLAR':
        path = os.path.join(data_dir, 'solar.txt')
        data = np.loadtxt(path, delimiter=',')
    elif name == 'TRAFFIC':
        path = os.path.join(data_dir, 'traffic.txt')
        data = np.loadtxt(path, delimiter=',')
    elif name == 'ELECTRICITY':
        path = os.path.join(data_dir, 'electricity.txt')
        data = np.loadtxt(path, delimiter=',')
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Data is (Time, Node) usually. Expand to (Time, Node, Channel=1)
    if data.ndim == 2:
        data = data[..., np.newaxis]
        
    if limit_samples:
        data = data[:limit_samples]
        print(f"DEBUG: Limiting dataset to {limit_samples} samples.")
        
    num_samples = data.shape[0]
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    # Creating or using Scaler
    from src.data.scaler import StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        # Fit on TRAIN data always to prevent leakage
        scaler.fit(data[:train_end])
    
    
    if mode == 'train':
        data_slice = data[:train_end]
    elif mode == 'val':
        data_slice = data[train_end:val_end]
    elif mode == 'test':
        data_slice = data[val_end:]
    elif mode == 'all':
        # Apply transform to whole data, then split
        data = scaler.transform(data)
        
        train_dataset = WindowedVSFDataset(data[:train_end], window_size, horizon, stride, mode='train')
        val_dataset = WindowedVSFDataset(data[train_end:val_end], window_size, horizon, stride, mode='val')
        test_dataset = WindowedVSFDataset(data[val_end:], window_size, horizon, stride, mode='test')
        
        return train_dataset, val_dataset, test_dataset, scaler
    else:
        data_slice = data

    # Transform data_slice
    data_slice = scaler.transform(data_slice)
    
    dataset = WindowedVSFDataset(data_slice, window_size, horizon, stride, mode=mode)
    return dataset, scaler


class StandardVSFDataset(BaseVSFDataset):
    """
    Standard Dataset for VSF models.
    Loads data from numpy arrays or standard formats.
    """
    def __init__(self, data, mask=None, timestamps=None, mean=None, std=None):
        """
        data: (Time, Node, Channel) or (Time, Node)
        mask: (Time, Node, Channel)
        """
        super().__init__()
        self.data = np.array(data)
        if len(self.data.shape) == 2:
            self.data = self.data[..., np.newaxis] # (T, N, 1)
            
        self.T, self.N, self.C = self.data.shape
        
        if mask is not None:
            self.mask = np.array(mask)
            if len(self.mask.shape) == 2:
                self.mask = self.mask[..., np.newaxis]
        else:
            self.mask = np.ones_like(self.data)
            
        self.timestamps = timestamps
        self.mean = mean
        self.std = std

    def __len__(self):
        # This dataset usually represents the whole sequence.
        # But BaseVSFDataset might be used with a sliding window sampler effectively?
        # Alternatively, if we follow Pytorch Dataset convention, __len__ is number of samples.
        # But here 'data' is the full time series.
        # Usually we split this into windows. 
        # For simplicity, let's assume this class holds the data, and we might implement a WindowedDataset subclass
        # Or this class itself handles windowing if provided with 'window_size'.
        # Let's match FDW/GinAR usage often: they hold full data and index into it.
        # But for 'DataLoader', we usually iterate over windows.
        # Let's assume we pass in PRE-SLICED data (B, T, N, C) for now?
        # NO, usually we pass (Time, Node) and slice it on the fly.
        # Let's make this class simple: it holds the data.
        # But __getitem__ needs to return a sample.
        # If we use a separate Sampler, that's fine.
        # But standard Pytorch Dataset __getitem__(idx) returns the idx-th sample.
        
        # Let's assume we implement a SlidingWindow version.
        return self.T # This is ambiguous.
        pass

class WindowedVSFDataset(BaseVSFDataset):
    def __init__(self, data, window_size, horizon, stride=1, mask=None, mode='train'):
        super().__init__()
        # data: (TotalTime, Node, Channel)
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.mode = mode
        
        if mask is not None:
             self.mask = torch.FloatTensor(mask)
        else:
             self.mask = torch.ones_like(self.data)
             
        self.n_samples = (self.data.shape[0] - window_size - horizon) // stride + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        
        # x: Input window
        x = self.data[start:end] # (T_in, N, C)
        x_mask = self.mask[start:end]
        
        # y: Target window (horizon)
        y_start = end
        y_end = end + self.horizon
        y = self.data[y_start:y_end] # (T_out, N, C)
        y_mask = self.mask[y_start:y_end]
        
        # In VSF, we might need a unified "mask" for input
        # Return dict matching core/dataset
        return {
            'x': x,
            'y': y,
            'mask': x_mask, # Mask for input
            # 'y_mask': y_mask # Optional
        }

class SyntheticVSFDataset(WindowedVSFDataset):
    def __init__(self, num_samples=100, seq_len=100, num_nodes=10, num_channels=1, window_size=12, horizon=12):
        # Generate random data: (TotalTime, Node, Channel) where TotalTime allows num_samples
        # n_samples = (Total - w - h) // 1 + 1
        # Total = n_samples + w + h - 1
        total_time = num_samples + window_size + horizon
        data = np.random.randn(total_time, num_nodes, num_channels).astype(np.float32)
        super().__init__(data, window_size, horizon)
