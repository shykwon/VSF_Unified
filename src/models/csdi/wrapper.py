import sys
import os
import torch
from src.core.model import BaseVSFModel

import importlib.util

# Add CSDI source path
CSDI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/csdi")

def _load_csdi_module():
    """Load CSDI main_model with isolated module cache to avoid SRDI conflicts."""
    # We need to make sure 'diff_models' is loaded from CSDI path
    # because CSDI's main_model imports it, and it differs from SRDI's diff_models.
    
    old_path = list(sys.path)
    sys.path.insert(0, CSDI_PATH)
    
    # Check if conflicting modules are loaded
    saved_modules = {}
    conflicting_names = ['diff_models', 'main_model']
    for name in conflicting_names:
        if name in sys.modules:
            saved_modules[name] = sys.modules.pop(name)
            
    try:
        # Load CSDI's main_model
        # We use standard import because we want it to resolve 'diff_models' using usage of sys.path
        spec = importlib.util.spec_from_file_location(
            "csdi_main_model",
            os.path.join(CSDI_PATH, "main_model.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["csdi_main_model"] = module # Register unique name
        
        spec.loader.exec_module(module)
        return module
    finally:
        # Restore sys.path
        sys.path = old_path
        # Restore old modules to sys.modules (so other parts of app see the original ones)
        # Note: csdi_main_model will strictly hold definition to the classes it loaded
        # which link to the CSDI versions of diff_models it imported.
        for name, mod in saved_modules.items():
            sys.modules[name] = mod

try:
    _csdi_module = _load_csdi_module()
    CSDI_Forecasting = _csdi_module.CSDI_Forecasting
except Exception as e:
    print(f"Warning: Could not import CSDI from {CSDI_PATH}. Error: {e}")
    import traceback
    traceback.print_exc()
    CSDI_Forecasting = None

class CSDIWrapper(BaseVSFModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        
        target_dim = config.get('num_nodes', 207)
        
        # CSDI expects a specific nested config structure
        # We assume self.config is correct or we construct a dummy one
        if 'model' not in self.config:
            # Construct default config for CSDI (based on base_forecasting.yaml)
            self.csdi_config = {
                "model": {
                    "timeemb": 128,
                    "featureemb": 16,
                    "is_unconditional": False,
                    "target_strategy": "random",
                    "num_sample_features": min(target_dim, 64),  # Required for CSDI_Forecasting
                },
                "diffusion": {
                    "layers": 4,
                    "channels": 64,
                    "nheads": 8,
                    "diffusion_embedding_dim": 128,
                    "beta_start": 0.0001,
                    "beta_end": 0.5,
                    "num_steps": 50,
                    "schedule": "quad",
                    "is_linear": True,  # Required for diff_models.py
                }
            }
        else:
            self.csdi_config = self.config
            # Ensure is_linear exists in diffusion config
            if "diffusion" in self.csdi_config and "is_linear" not in self.csdi_config["diffusion"]:
                self.csdi_config["diffusion"]["is_linear"] = True
        
        self.model = CSDI_Forecasting(self.csdi_config, self.device, target_dim).to(self.device)

    def forward(self, batch):
        x = batch['x'].to(self.device) # (B, T_hist, N, C)
        
        # Assumption: C=1 for CSDI
        if x.shape[-1] == 1:
            x = x.squeeze(-1) # (B, T_hist, N)
            
        B, T_hist, N = x.shape
        pred_len = self.config.get('pred_len', 12)
        
        # 1. Prepare Future Part
        if self.training and 'y' in batch:
            y = batch['y'].to(self.device) # (B, T_out, N, C)
            if y.shape[-1] == 1:
                y = y.squeeze(-1) # (B, T_out, N)
            if y.shape[1] > pred_len:
                y = y[:, :pred_len, :]
            future_data = y
        else:
            future_data = torch.zeros(B, pred_len, N).to(self.device)
            
        # 2. Concatenate (History + Future)
        # CSDI expects (B, T, N) and permutes appropriately
        full_x = torch.cat([x, future_data], dim=1) # (B, T_total, N)
        
        # 3. Create Masks
        mask_hist = torch.ones_like(x)
        mask_future = torch.zeros_like(future_data)
        full_observed_mask = torch.cat([mask_hist, mask_future], dim=1) # (B, T_total, N)
        
        # gt_mask: We want to mask everything as 1 generally for GT validity
        full_gt_mask = torch.ones_like(full_x)
        
        # 4. Timepoints
        T_total = full_x.shape[1]
        timepoints = torch.arange(T_total).unsqueeze(0).expand(B, -1).to(self.device)
        
        csdi_batch = {
            "observed_data": full_x,
            "observed_mask": full_observed_mask,
            "gt_mask": full_gt_mask,
            "timepoints": timepoints,
            "cut_length": torch.zeros(B).long().to(self.device)
        }
        
        if self.training:
            loss = self.model(csdi_batch, is_train=1)
            # CSDI returns scalar loss in base implementation
            return {'loss': loss}
        else:
             samples, _, _, _, _ = self.model.evaluate(csdi_batch, n_samples=1)
             # samples: (B, n_samples, K, L) -> (B, 1, N, T_total)
             
             samples = samples.squeeze(1) # (B, N, T_total)
             samples = samples.permute(0, 2, 1) # (B, T_total, N)
             
             # Slice future
             pred = samples[:, -pred_len:, :]
             
             # Unsqueeze channel
             pred = pred.unsqueeze(-1) # (B, T_out, N, 1)
             
             return {'pred': pred}
