import torch
import torch.nn as nn
import os
import time
import numpy as np
from src.core.metrics import Metrics, MaskedMetrics, ObservedMetrics

class BaseTrainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = config['device']
        self.model.to(self.device)
        
        # Scaler for Inverse Transform
        self.scaler = config.get('scaler', None)

        # Loss Function Configuration
        self.loss_fn_name = config.get('loss_fn', 'mse')
        if self.loss_fn_name == 'masked_mae':
            self.loss_fn = MaskedMetrics.MaskedMAE
        else:
            self.loss_fn = nn.MSELoss()
            
    def train_epoch(self, loader):
        raise NotImplementedError

    def evaluate(self, loader):
        raise NotImplementedError

    def fit(self, train_loader, val_loader, epochs):
        raise NotImplementedError
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

class UnifiedTrainer(BaseTrainer):
    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        # Gradient clipping (original FDW uses clip=5.0)
        self.clip_grad = config.get('clip_grad', 5.0)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for batch in loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(batch)
            
            # Hybrid Loss Handling
            if 'loss' in output:
                # 1. Internal Loss (CSDI, SRDI, SAITS)
                loss = output['loss']
            else:
                # 2. External Loss (FDW, GinAR)
                pred = output['pred']
                true = batch['y']
                
                # Slicing if needed (match shapes)
                if pred.shape != true.shape:
                     # Assume pred is (B, T_out, N, C) and true is (B, T, N, C) or similar
                     # It's better to ensure shapes match in Wrapper, but safety check here
                     if pred.shape[1] < true.shape[1]:
                         true = true[:, -pred.shape[1]:, ...]
                
                if self.loss_fn_name == 'masked_mae':
                    loss = self.loss_fn(pred, true, null_val=0.0)
                else:
                    loss = self.loss_fn(pred, true)
            
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_trues = []
        all_masks = []  # 관측 마스크 수집

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                output = self.model(batch)

                # Logic for Metrics
                # Even if model returns 'loss', we need 'pred' for metrics
                if 'pred' in output:
                    pred = output['pred']
                else:
                    # Should not happen if wrapper is correct
                    continue

                true = batch['y']
                # y_mask: target(y)에 대한 관측 마스크 (1=observed, 0=missing)
                # 주의: batch['mask']는 x에 대한 마스크이므로 사용하면 안됨
                mask = batch.get('y_mask', None)

                # Match shapes
                if pred.shape[1] < true.shape[1]:
                    true = true[:, -pred.shape[1]:, ...]
                    if mask is not None:
                        mask = mask[:, -pred.shape[1]:, ...]

                if self.scaler is not None:
                    # Inverse Transform
                    # data shape usually (B, T, N, C)
                    # scaler expects tensor
                    pred = self.scaler.inverse_transform(pred)
                    true = self.scaler.inverse_transform(true)

                all_preds.append(pred.cpu())
                all_trues.append(true.cpu())
                if mask is not None:
                    all_masks.append(mask.cpu())

                # Hybrid Loss for Validation Loss Tracking
                if 'loss' in output:
                    loss = output['loss']
                else:
                    if self.loss_fn_name == 'masked_mae':
                        loss = self.loss_fn(pred, true, null_val=0.0)
                    else:
                        loss = self.loss_fn(pred, true)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Calculate Aggregated Metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)

        # 기본 메트릭 (전체 데이터, imputed 포함)
        metrics = {
            'loss': avg_loss,
            'MAE': Metrics.MAE(all_preds, all_trues).item(),
            'RMSE': Metrics.RMSE(all_preds, all_trues).item(),
        }

        # 관측 마스크가 있으면 ObservedMetrics 계산
        if all_masks:
            all_masks = torch.cat(all_masks, dim=0)
            # 관측된 위치에서만 계산 (mask=1)
            metrics['ObservedMAE'] = ObservedMetrics.ObservedMAE(all_preds, all_trues, all_masks).item()
            metrics['ObservedRMSE'] = ObservedMetrics.ObservedRMSE(all_preds, all_trues, all_masks).item()
            metrics['ObservedMAPE'] = ObservedMetrics.ObservedMAPE(all_preds, all_trues, all_masks).item()
            # 관측률 (디버깅용)
            metrics['ObservedRatio'] = (all_masks.sum() / all_masks.numel()).item()
        else:
            # 마스크가 없으면 전체와 동일
            metrics['ObservedMAE'] = metrics['MAE']
            metrics['ObservedRMSE'] = metrics['RMSE']
            metrics['ObservedMAPE'] = Metrics.MAPE(all_preds, all_trues).item()
            metrics['ObservedRatio'] = 1.0

        return metrics

    def fit(self, train_loader, val_loader, epochs):
        """
        Standard training loop.
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            # Simple logging
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-start_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.config.get('log_dir', '.'), 'best_model.pth')
                self.save(save_path)

