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

    @staticmethod
    def MaskedMAPE(pred, true, null_val=np.nan):
        """
        Masked MAPE: 결측이 아닌 위치에서만 MAPE 계산
        """
        if np.isnan(null_val):
            mask = ~torch.isnan(true)
        else:
            mask = (true != null_val)

        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        # MAPE 계산 (0으로 나누기 방지)
        loss = torch.abs((pred - true) / (torch.abs(true) + 1e-8)) * 100
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)


class ObservedMetrics:
    """
    Metrics computed only on observed (non-missing) positions using explicit mask.
    mask=1: observed, mask=0: missing

    이 메트릭은 VSF 태스크에서 관측된 센서만 평가할 때 사용합니다.
    """

    @staticmethod
    def ObservedMAE(pred, true, mask):
        """
        Observed MAE: mask=1인 위치(관측된 곳)에서만 MAE 계산
        """
        mask = mask.float()
        num_observed = mask.sum()
        if num_observed == 0:
            return torch.tensor(0.0)

        loss = torch.abs(pred - true) * mask
        return loss.sum() / num_observed

    @staticmethod
    def ObservedMSE(pred, true, mask):
        """
        Observed MSE: mask=1인 위치(관측된 곳)에서만 MSE 계산
        """
        mask = mask.float()
        num_observed = mask.sum()
        if num_observed == 0:
            return torch.tensor(0.0)

        loss = ((pred - true) ** 2) * mask
        return loss.sum() / num_observed

    @staticmethod
    def ObservedRMSE(pred, true, mask):
        """
        Observed RMSE: mask=1인 위치(관측된 곳)에서만 RMSE 계산
        """
        return torch.sqrt(ObservedMetrics.ObservedMSE(pred, true, mask))

    @staticmethod
    def ObservedMAPE(pred, true, mask, eps=0.1):
        """
        Observed MAPE: mask=1인 위치(관측된 곳)에서만 MAPE 계산
        eps: 0으로 나누기 방지용 epsilon (기본 0.1, Solar 등 0값 많은 데이터용)
        """
        mask = mask.float()
        num_observed = mask.sum()
        if num_observed == 0:
            return torch.tensor(0.0)

        # 0에 가까운 값 제외 (MAPE 폭발 방지)
        valid_mask = mask * (torch.abs(true) > eps).float()
        num_valid = valid_mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0)

        loss = torch.abs((pred - true) / (torch.abs(true) + 1e-8)) * 100 * valid_mask
        return loss.sum() / num_valid
