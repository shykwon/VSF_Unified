import sys
import os
import torch
import torch.nn as nn
import importlib.util
from src.core.model import BaseVSFModel

# Add SRDI source path explicitly to avoid conflicts with CSDI's main_model
SRDI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/SRDI")

def _load_srdi_module():
    """Load SRDI main_model with isolated module cache to avoid CSDI conflicts."""
    # Save conflicting modules from cache
    saved_modules = {}
    modules_to_save = ['diff_models', 'main_model', 'Model']
    for mod_name in modules_to_save:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    # Ensure SRDI path is first in sys.path
    if SRDI_PATH in sys.path:
        sys.path.remove(SRDI_PATH)
    sys.path.insert(0, SRDI_PATH)

    try:
        # Load SRDI's main_model with fresh imports
        spec = importlib.util.spec_from_file_location(
            "srdi_main_model",
            os.path.join(SRDI_PATH, "main_model.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["srdi_main_model"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        # Restore saved modules
        for mod_name, mod in saved_modules.items():
            sys.modules[mod_name] = mod

try:
    _srdi_module = _load_srdi_module()
    CSDI_vsf = _srdi_module.CSDI_vsf
except Exception as e:
    print(f"Warning: Could not import SRDI from {SRDI_PATH}. Error: {e}")
    import traceback
    traceback.print_exc()
    CSDI_vsf = None

class SRDIWrapper(BaseVSFModel):
    def __init__(self, config):
        """
        SRDI uses CSDI_vsf for imputation/diffusion.
        config must match CSDI_vsf expectations (nested 'model' and 'diffusion' keys).
        """
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')

        target_dim = config.get('num_nodes', 207)

        # SRDI (CSDI_vsf) requires nested config structure like CSDI
        if 'model' not in self.config:
            # Construct default config for SRDI
            self.srdi_config = {
                "model": {
                    "timeemb": 128,
                    "featureemb": 16,
                    "is_unconditional": False,
                    "target_strategy": "random",
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
                    "is_linear": True,
                }
            }
        else:
            self.srdi_config = self.config
            # Ensure is_linear exists
            if "diffusion" in self.srdi_config and "is_linear" not in self.srdi_config["diffusion"]:
                self.srdi_config["diffusion"]["is_linear"] = True

        self.model = CSDI_vsf(self.srdi_config, self.device, target_dim).to(self.device)

    def forward(self, batch):
        x = batch['x'].to(self.device) # (B, T_hist, N, C)
        
        # Assumption: C=1 for SRDI
        if x.shape[-1] == 1:
            x = x.squeeze(-1) # (B, T_hist, N)
            
        B, T_hist, N = x.shape
        pred_len = self.config.get('pred_len', 12) # Default 12 or from config
        
        # 1. Prepare Future Part
        if self.training and 'y' in batch:
            y = batch['y'].to(self.device) # (B, T_out, N, C)
            if y.shape[-1] == 1:
                y = y.squeeze(-1) # (B, T_out, N)
            # Ensure y matches pred_len
            if y.shape[1] > pred_len:
                y = y[:, :pred_len, :]
            future_data = y
        else:
            future_data = torch.zeros(B, pred_len, N).to(self.device)

        # 2. Concatenate (History + Future) -> Observed Data
        # For training, observed_data contains GT in future part, but it will be masked out by observed_mask
        full_x = torch.cat([x, future_data], dim=1) # (B, T_total, N)
        
        # 3. Create Masks
        # observed_mask: 1 for history, 0 for future
        mask_hist = torch.ones_like(x)
        mask_future = torch.zeros_like(future_data)
        full_observed_mask = torch.cat([mask_hist, mask_future], dim=1) # (B, T_total, N)
        
        # gt_mask: 1 where we have ground truth. 
        # For simplicity in this forecasting task:
        full_gt_mask = torch.ones_like(full_x)

        # 4. Permute to (B, N, T) for CSDI/SRDI
        full_x = full_x.permute(0, 2, 1) # (B, N, T_total)
        full_observed_mask = full_observed_mask.permute(0, 2, 1)
        full_gt_mask = full_gt_mask.permute(0, 2, 1)
        
        # 5. Timepoints
        T_total = full_x.shape[-1]
        timepoints = torch.arange(T_total).unsqueeze(0).expand(B, -1).to(self.device)

        csdi_batch = {
            "observed_data": full_x,
            "observed_mask": full_observed_mask,
            "gt_mask": full_gt_mask,
            "timepoints": timepoints,
            "cut_length": torch.zeros(B).long().to(self.device)
        }

        if self.training:
            loss, loss_d = self.model(csdi_batch, is_train=1)
            # Normalize loss if needed, CSDI usually returns sum or mean.
            return {'loss': loss}
        else:
            # evaluate returns samples: (B, n_samples, N, T_total)
            samples = self.model.evaluate(csdi_batch, n_samples=1) 
            
            # Squeeze n_samples
            samples = samples.squeeze(1) # (B, N, T_total)
            
            # Permute back to (B, T_total, N)
            samples = samples.permute(0, 2, 1)
            
            # Slice Future Part
            pred = samples[:, -pred_len:, :]
            
            # Unsqueeze Channel
            pred = pred.unsqueeze(-1) # (B, T_out, N, 1)
            
            return {'pred': pred} 
