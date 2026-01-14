import sys
import os
import torch
from src.core.model import BaseVSFModel

# Add SAITS source path
SAITS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/saits")
sys.path.insert(0, SAITS_PATH)

try:
    from modeling.saits import SAITS
except ImportError as e:
    print(f"Warning: Could not import SAITS from {SAITS_PATH}. Error: {e}")
    import traceback
    traceback.print_exc()
    SAITS = None

class SAITSWrapper(BaseVSFModel):
    def __init__(self, config):
        """
        SAITS(d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, n_groups, n_group_inner_layers, diagonal_attention_mask, MIT, param_sharing_strategy, device, input_with_mask, ORT)
        """
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Args mapping
        d_time = config.get('seq_len', 100)
        d_feature = config.get('num_nodes', 207)
        d_model = config.get('d_model', 256)
        d_inner = config.get('d_inner', 512)
        n_head = config.get('n_head', 4)
        d_k = config.get('d_k', 64)
        d_v = config.get('d_v', 64)
        dropout = config.get('dropout', 0.1)
        n_groups = config.get('n_groups', 2)
        n_group_inner_layers = config.get('n_group_inner_layers', 2)
        
        # MIT=False to avoid requiring X_holdout during training
        # Set MIT=True only when you have proper holdout data for imputation evaluation
        self.MIT = config.get('MIT', False)

        self.model = SAITS(
            d_time=d_time,
            d_feature=d_feature,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            n_groups=n_groups,
            n_group_inner_layers=n_group_inner_layers,
            diagonal_attention_mask=True,
            MIT=self.MIT,
            param_sharing_strategy='between_group',
            device=self.device,
            input_with_mask=True,
            ORT=True
        ).to(self.device)

    def forward(self, batch):
        """
        SAITS input: dict with keys "indices", "X", "missing_mask"

        For MIT=True (imputation evaluation during training):
            - batch should contain 'x_holdout' and 'indicating_mask'
            - x_holdout: ground truth for artificially masked positions
            - indicating_mask: mask indicating which positions were artificially masked
        """
        x = batch['x'].to(self.device)
        mask = batch['mask'].to(self.device)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)
            mask = mask.squeeze(-1)

        inputs = {
            "X": x,
            "missing_mask": 1 - mask, # SAITS expects 1 for missing, 0 for observed
            "indices": batch.get('indices', torch.arange(len(x)).to(self.device))
        }

        # For MIT=True, provide holdout data for imputation loss calculation
        if self.MIT and self.training:
            if 'x_holdout' in batch and 'indicating_mask' in batch:
                x_holdout = batch['x_holdout'].to(self.device)
                indicating_mask = batch['indicating_mask'].to(self.device)
                if x_holdout.shape[-1] == 1:
                    x_holdout = x_holdout.squeeze(-1)
                    indicating_mask = indicating_mask.squeeze(-1)
                inputs["X_holdout"] = x_holdout
                inputs["indicating_mask"] = indicating_mask
            else:
                # If MIT=True but no holdout data, use X itself as holdout (self-supervised)
                inputs["X_holdout"] = x
                inputs["indicating_mask"] = 1 - mask  # missing positions

        # SAITS forward(inputs, stage='train'/'val'/'test')
        # Returns dict with 'imputed_data', 'reconstruction_loss', etc.

        stage = 'train' if self.training else 'test'
        results = self.model(inputs, stage=stage)
        
        # Results handling
        if 'imputed_data' in results:
             pred = results['imputed_data'].unsqueeze(-1) # Add channel back
        else:
             pred = None
             
        out = {'pred': pred}
        if 'total_loss' in results:
             out['loss'] = results['total_loss']
             
        return out
