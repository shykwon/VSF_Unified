import sys
import os
import torch
import torch.nn as nn
import importlib.util
import numpy as np
from src.core.model import BaseVSFModel

# Add SRDI source path
SRDI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/SRDI")

def _load_srdi_modules():
    """Load SRDI modules with isolated module cache to avoid conflicts."""
    saved_modules = {}
    modules_to_save = ['diff_models', 'main_model', 'Model', 'net', 'util']
    for mod_name in modules_to_save:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    if SRDI_PATH in sys.path:
        sys.path.remove(SRDI_PATH)
    sys.path.insert(0, SRDI_PATH)

    try:
        # Load SRDI's main_model (CSDI_vsf)
        spec_main = importlib.util.spec_from_file_location(
            "srdi_main_model",
            os.path.join(SRDI_PATH, "main_model.py")
        )
        main_module = importlib.util.module_from_spec(spec_main)
        sys.modules["srdi_main_model"] = main_module
        spec_main.loader.exec_module(main_module)

        # Load SRDI's net.py (gtnet)
        spec_net = importlib.util.spec_from_file_location(
            "srdi_net",
            os.path.join(SRDI_PATH, "net.py")
        )
        net_module = importlib.util.module_from_spec(spec_net)
        sys.modules["srdi_net"] = net_module
        spec_net.loader.exec_module(net_module)

        # Load SRDI's util.py (data_processing, masked_mae)
        spec_util = importlib.util.spec_from_file_location(
            "srdi_util",
            os.path.join(SRDI_PATH, "util.py")
        )
        util_module = importlib.util.module_from_spec(spec_util)
        sys.modules["srdi_util"] = util_module
        spec_util.loader.exec_module(util_module)

        return main_module, net_module, util_module
    finally:
        for mod_name, mod in saved_modules.items():
            sys.modules[mod_name] = mod

try:
    _main_module, _net_module, _util_module = _load_srdi_modules()
    CSDI_vsf = _main_module.CSDI_vsf
    gtnet = _net_module.gtnet
    data_processing = _util_module.data_processing
    masked_mae = _util_module.masked_mae
except Exception as e:
    print(f"Warning: Could not import SRDI modules from {SRDI_PATH}. Error: {e}")
    import traceback
    traceback.print_exc()
    CSDI_vsf = None
    gtnet = None

# Try to import learn2learn for MAML
try:
    import learn2learn as l2l
    HAS_LEARN2LEARN = True
except ImportError:
    print("Warning: learn2learn not installed. SRDI will run without MAML (reduced performance).")
    print("Install with: pip install learn2learn")
    HAS_LEARN2LEARN = False
    l2l = None


class SRDIArgs:
    """Helper class to convert config dict to args object for SRDI compatibility."""
    def __init__(self, config):
        self.device = config.get('device', 'cpu')
        self.num_nodes = config.get('num_nodes', 207)
        self.seq_in_len = config.get('seq_in_len', 12)
        self.seq_out_len = config.get('seq_out_len', 12)
        self.in_dim = config.get('in_dim', 1)

        # gtnet (forecaster) parameters - matching original defaults
        self.gcn_true = config.get('gcn_true', True)
        self.buildA_true = config.get('buildA_true', True)
        self.gcn_depth = config.get('gcn_depth', 2)
        self.dropout = config.get('dropout', 0.3)
        self.subgraph_size = min(config.get('subgraph_size', 20), self.num_nodes)
        self.node_dim = config.get('node_dim', 40)
        self.dilation_exponential = config.get('dilation_exponential', 1)
        self.conv_channels = config.get('conv_channels', 32)
        self.residual_channels = config.get('residual_channels', 32)
        self.skip_channels = config.get('skip_channels', 64)
        self.end_channels = config.get('end_channels', 128)
        self.layers = config.get('layers', 3)
        self.propalpha = config.get('propalpha', 0.05)
        self.tanhalpha = config.get('tanhalpha', 3)

        # MAML parameters
        self.maml_lr = config.get('maml_lr', 0.001)
        self.adaptation_steps = config.get('adaptation_steps', 1)

        # Masking ratio for diffusion
        self.missing_ratio = config.get('missing_ratio', 0.85)

        # For gtnet forward compatibility
        self.adj_identity_train_test = config.get('adj_identity_train_test', False)
        self.mask_remaining = config.get('mask_remaining', False)


class SRDIWrapper(BaseVSFModel):
    """
    Full SRDI Implementation (Level 3)

    Architecture:
        1. CSDI_vsf: Diffusion-based imputation module
        2. gtnet: MTGNN-based forecaster
        3. MAML: Meta-learning for test-time adaptation

    Training Flow:
        Input -> MAML(CSDI_vsf) -> Imputed -> gtnet -> Forecast

    Test-time Adaptation:
        For each test sample:
            1. Clone MAML model
            2. Adapt to current sample (few gradient steps)
            3. Generate imputation
            4. Forecast with gtnet
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        self.args = SRDIArgs(config)

        target_dim = self.args.num_nodes

        # GPU profile for diffusion config
        gpu_profile = config.get('gpu_profile', '4090')

        if gpu_profile == '1080ti':
            # Reduced settings for GTX 1080 Ti (11GB)
            self.diffusion_config = {
                "model": {
                    "timeemb": 64,
                    "featureemb": 8,
                    "is_unconditional": False,
                    "target_strategy": "random",
                },
                "diffusion": {
                    "layers": 2,
                    "channels": 32,
                    "nheads": 4,
                    "diffusion_embedding_dim": 32,
                    "beta_start": 0.0001,
                    "beta_end": 0.5,
                    "num_steps": 50,
                    "schedule": "quad",
                    "is_linear": True,
                }
            }
        else:
            # Full paper settings for 4090/A100 (24GB+)
            # Based on base_forecasting.yaml
            self.diffusion_config = {
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
                    "diffusion_embedding_dim": 32,  # base_forecasting.yaml uses 32
                    "beta_start": 0.0001,
                    "beta_end": 0.5,
                    "num_steps": 50,
                    "schedule": "quad",
                    "is_linear": True,
                }
            }

        # 1. Diffusion Module (CSDI_vsf)
        self.diffusion = CSDI_vsf(
            config=self.diffusion_config,
            device=self.device,
            target_dim=target_dim
        ).to(self.device)

        # 2. Forecaster (gtnet/MTGNN)
        predefined_A = config.get('adj_mx', None)
        if predefined_A is not None:
            if isinstance(predefined_A, np.ndarray):
                adj = predefined_A.astype(np.float32)
                # Remove self-loops (original SRDI behavior)
                adj = adj - np.eye(adj.shape[0], dtype=np.float32)
                adj = np.clip(adj, 0, None)
                predefined_A = torch.from_numpy(adj).to(self.device)
            elif isinstance(predefined_A, torch.Tensor):
                predefined_A = predefined_A.float() - torch.eye(predefined_A.shape[0])
                predefined_A = torch.clamp(predefined_A, min=0).to(self.device)
            print(f"SRDI: Using predefined adjacency matrix, shape={predefined_A.shape}")
        else:
            print(f"SRDI: No adjacency matrix provided, using learned graph")

        self.forecaster = gtnet(
            gcn_true=self.args.gcn_true,
            buildA_true=self.args.buildA_true,
            gcn_depth=self.args.gcn_depth,
            num_nodes=self.args.num_nodes,
            device=self.device,
            predefined_A=predefined_A,
            dropout=self.args.dropout,
            subgraph_size=self.args.subgraph_size,
            node_dim=self.args.node_dim,
            dilation_exponential=self.args.dilation_exponential,
            conv_channels=self.args.conv_channels,
            residual_channels=self.args.residual_channels,
            skip_channels=self.args.skip_channels,
            end_channels=self.args.end_channels,
            seq_length=self.args.seq_in_len,
            in_dim=self.args.in_dim,
            out_dim=self.args.seq_out_len,
            layers=self.args.layers,
            propalpha=self.args.propalpha,
            tanhalpha=self.args.tanhalpha,
            layer_norm_affline=True
        ).to(self.device)

        # 3. MAML Wrapper (if available)
        if HAS_LEARN2LEARN:
            self.maml = l2l.algorithms.MAML(self.diffusion, lr=self.args.maml_lr).to(self.device)
            self.use_maml = True
            print("SRDI: MAML enabled for test-time adaptation")
        else:
            self.maml = None
            self.use_maml = False
            print("SRDI: Running without MAML (install learn2learn for full performance)")

        # Loss function
        self.loss_fn = masked_mae

    def _prepare_diffusion_batch(self, x, missing_ratio=0.85):
        """
        Prepare batch for diffusion model following SRDI's data_processing logic.

        Args:
            x: Input tensor (B, N, T) or (B, C, N, T)
            missing_ratio: Ratio of nodes to mask

        Returns:
            batch dict for CSDI_vsf
        """
        # Ensure x is (B, N, T) - diffusion expects this format
        if x.dim() == 4:
            x = x.squeeze(1) if x.shape[1] == 1 else x[:, 0, :, :]

        B, N, T = x.shape
        device = x.device

        # Create masks
        mask_num = int(missing_ratio * N)
        observed_mask = torch.ones_like(x)
        gt_mask = torch.ones_like(x)

        # Random masking for each batch
        for b in range(B):
            mask_indices = torch.randperm(N)[:mask_num]
            gt_mask[b, mask_indices, :] = 0.0

            # Additional 10% masking on remaining nodes (diffusion secondary mask)
            remaining_indices = torch.randperm(N)[mask_num:]
            additional_mask = remaining_indices[:int(len(remaining_indices) * 0.1)]
            gt_mask[b, additional_mask, :] = 0.0

        # Timepoints
        timepoints = torch.arange(T).unsqueeze(0).expand(B, -1).to(device)

        batch = {
            "observed_data": x,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
        }

        return batch

    def forward(self, batch):
        """
        SRDI Forward Pass

        Training:
            1. Prepare diffusion batch with masking
            2. MAML adaptation on diffusion loss
            3. Generate imputed data
            4. Forecast with gtnet
            5. Compute combined loss

        Inference:
            1. Per-sample MAML adaptation (if enabled)
            2. Imputation
            3. Forecasting
        """
        x = batch['x'].to(self.device)  # (B, T, N, C)
        y = batch['y'].to(self.device) if 'y' in batch else None

        # Convert to SRDI expected format: (B, C, N, T)
        x = x.permute(0, 3, 2, 1)  # (B, T, N, C) -> (B, C, N, T)
        if y is not None:
            y = y.permute(0, 3, 2, 1)

        B = x.shape[0]

        if self.training:
            return self._forward_train(x, y)
        else:
            return self._forward_eval(x, y)

    def _forward_train(self, x, y):
        """Training forward pass with MAML adaptation."""
        B = x.shape[0]
        device = x.device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        all_preds = []

        # Process each sample with MAML (following original SRDI trainer)
        for i in range(B):
            sample_x = x[i:i+1]  # (1, C, N, T)
            sample_y = y[i:i+1] if y is not None else None

            # Prepare diffusion batch
            # SRDI uses (N, T) format internally for single sample
            sample_for_diff = sample_x.squeeze(0).squeeze(0)  # (N, T)
            if sample_for_diff.dim() == 3:
                sample_for_diff = sample_for_diff[0]  # Take first channel

            diff_batch = self._prepare_diffusion_batch(
                sample_for_diff.unsqueeze(0),  # (1, N, T)
                missing_ratio=self.args.missing_ratio
            )

            # Move batch to device
            for k, v in diff_batch.items():
                if isinstance(v, torch.Tensor):
                    diff_batch[k] = v.to(device)

            if self.use_maml:
                # MAML: Clone, compute loss, adapt
                clone = self.maml.clone()
                diff_result = clone(diff_batch, is_train=1)
                # CSDI_vsf returns (loss, loss_d) tuple
                if isinstance(diff_result, tuple):
                    diff_loss, loss_d = diff_result
                    diff_loss = diff_loss + 0.000001 * loss_d  # Following original SRDI
                else:
                    diff_loss = diff_result

                # Adaptation step
                clone.adapt(diff_loss, allow_unused=True, allow_nograd=True)

                # Generate imputed samples
                imputed = clone.module.evaluate(diff_batch, n_samples=1)
            else:
                # Without MAML: Direct forward
                diff_result = self.diffusion(diff_batch, is_train=1)
                # CSDI_vsf returns (loss, loss_d) tuple
                if isinstance(diff_result, tuple):
                    diff_loss, loss_d = diff_result
                    diff_loss = diff_loss + 0.000001 * loss_d  # Following original SRDI
                else:
                    diff_loss = diff_result
                imputed = self.diffusion.evaluate(diff_batch, n_samples=1)

            # imputed shape: (B, n_samples, N, T) -> (1, 1, N, T)
            imputed = imputed.squeeze(1)  # (1, N, T)

            # Prepare for forecaster: (B, C, N, T)
            imputed_for_forecast = imputed.unsqueeze(1)  # (1, 1, N, T)

            # Forecaster forward
            idx = torch.arange(self.args.num_nodes).to(device)
            pred = self.forecaster(imputed_for_forecast, idx=idx, args=self.args)
            # pred shape: (B, T_out, N, 1)

            all_preds.append(pred)

            # Compute forecast loss
            if sample_y is not None:
                # sample_y: (1, C, N, T_out) -> need to match pred shape
                real = sample_y.permute(0, 3, 2, 1)  # (1, T_out, N, C)
                fc_loss, _ = self.loss_fn(pred, real, 0.0)
                sample_loss = diff_loss + fc_loss
            else:
                sample_loss = diff_loss

            total_loss = total_loss + sample_loss

        # Average loss over batch
        total_loss = total_loss / B

        # Concatenate predictions
        preds = torch.cat(all_preds, dim=0)  # (B, T_out, N, 1)

        # Convert back to standard format (B, T, N, C)
        # preds is already in (B, T_out, N, 1) which matches

        return {
            'pred': preds,
            'loss': total_loss
        }

    def _forward_eval(self, x, y):
        """Evaluation forward pass with optional MAML test-time adaptation."""
        B = x.shape[0]
        device = x.device

        all_preds = []

        with torch.no_grad():
            for i in range(B):
                sample_x = x[i:i+1]  # (1, C, N, T)

                # Prepare diffusion batch
                sample_for_diff = sample_x.squeeze(0).squeeze(0)
                if sample_for_diff.dim() == 3:
                    sample_for_diff = sample_for_diff[0]

                diff_batch = self._prepare_diffusion_batch(
                    sample_for_diff.unsqueeze(0),
                    missing_ratio=self.args.missing_ratio
                )

                for k, v in diff_batch.items():
                    if isinstance(v, torch.Tensor):
                        diff_batch[k] = v.to(device)

                if self.use_maml:
                    # Test-time adaptation with MAML
                    clone = self.maml.clone()

                    # Enable gradients temporarily for adaptation
                    with torch.enable_grad():
                        diff_result = clone(diff_batch, is_train=1)
                        # CSDI_vsf returns (loss, loss_d) tuple
                        if isinstance(diff_result, tuple):
                            diff_loss, loss_d = diff_result
                            diff_loss = diff_loss + 0.000001 * loss_d
                        else:
                            diff_loss = diff_result
                        diff_loss_tensor = diff_loss.clone().detach().requires_grad_(True)
                        clone.adapt(diff_loss_tensor, allow_unused=True, allow_nograd=True)

                    # Generate imputed samples
                    imputed = clone.module.evaluate(diff_batch, n_samples=1)
                else:
                    imputed = self.diffusion.evaluate(diff_batch, n_samples=1)

                # imputed: (1, n_samples, N, T) -> (1, N, T)
                imputed = imputed.squeeze(1)

                # Prepare for forecaster
                imputed_for_forecast = imputed.unsqueeze(1)  # (1, 1, N, T)

                # Forecaster
                idx = torch.arange(self.args.num_nodes).to(device)
                pred = self.forecaster(imputed_for_forecast, idx=idx, args=self.args)

                all_preds.append(pred)

        preds = torch.cat(all_preds, dim=0)  # (B, T_out, N, 1)

        return {'pred': preds}

    def get_optimizer_params(self):
        """Return parameters for optimizer (both diffusion and forecaster)."""
        if self.use_maml:
            return list(self.maml.parameters()) + list(self.forecaster.parameters())
        else:
            return list(self.diffusion.parameters()) + list(self.forecaster.parameters())
