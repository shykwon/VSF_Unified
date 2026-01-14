import torch
import torch.nn as nn
import numpy as np

class Metrics:
    """
    Standard metrics for time series forecasting.
    """
    
    @staticmethod
    def MAE(pred, true):
        return torch.mean(torch.abs(pred - true))

    @staticmethod
    def MSE(pred, true):
        return torch.mean((pred - true) ** 2)

    @staticmethod
    def RMSE(pred, true):
        return torch.sqrt(Metrics.MSE(pred, true))

    @staticmethod
    def MAPE(pred, true):
        return torch.mean(torch.abs((pred - true) / (true + 1e-8))) * 100

class MaskedMetrics:
    """
    Metrics that ignore missing values (NaN or 0.0) in the ground truth.
    Typically used for graph-based models or imputation tasks.
    """
    
    @staticmethod
    def MaskedMAE(pred, true, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(true)
        else:
            mask = (true != null_val)
        
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        
        loss = torch.abs(pred - true)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    @staticmethod
    def MaskedMSE(pred, true, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(true)
        else:
            mask = (true != null_val)
        
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        
        loss = (pred - true) ** 2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
    
    @staticmethod
    def MaskedRMSE(pred, true, null_val=np.nan):
        return torch.sqrt(MaskedMetrics.MaskedMSE(pred, true, null_val))
