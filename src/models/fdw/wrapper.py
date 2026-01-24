import sys
import os
import torch
import torch.nn as nn
from src.core.model import BaseVSFModel

# Add FDW source path to sys.path to allow imports from its modules
FDW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/google_vsf_time_series")
sys.path.insert(0, FDW_PATH)

try:
    from net import gtnet
except ImportError as e:
    # Fallback or error handling if path is incorrect
    print(f"Warning: Could not import FDW from {FDW_PATH}. Error: {e}")
    import traceback
    traceback.print_exc()
    gtnet = None

class FDWArgs:
    """Simple args object for FDW gtnet compatibility."""
    def __init__(self, device='cpu', adj_identity_train_test=False):
        self.device = device
        self.adj_identity_train_test = adj_identity_train_test


class FDWWrapper(BaseVSFModel):
    def __init__(self, config):
        """
        config needs to contain arguments required by gtnet.
        """
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        # Create args object for gtnet forward compatibility
        self.args = FDWArgs(
            device=self.device,
            adj_identity_train_test=config.get('adj_identity_train_test', False)
        )
        
        # Original paper default parameters (train_multi_step.py)
        # Note: FDW paper uses these values for all GPU configs
        # The model is relatively lightweight and doesn't need reduction
        gcn_true = config.get('gcn_true', True)
        # If predefined adjacency is provided, don't learn new one (use predefined)
        # buildA_true=True means learn adjacency, =False means use predefined_A
        has_predefined_A = config.get('adj_mx') is not None or config.get('predefined_A') is not None
        buildA_true = config.get('buildA_true', not has_predefined_A)  # Default: False if adj provided
        gcn_depth = config.get('gcn_depth', 2)
        num_nodes = config.get('num_nodes', 207)
        dropout = config.get('dropout', 0.3)
        # subgraph_size must be <= num_nodes
        subgraph_size = min(config.get('subgraph_size', 20), num_nodes)
        node_dim = config.get('node_dim', 40)
        dilation_exponential = config.get('dilation_exponential', 1)
        # Original paper defaults (train_multi_step.py)
        conv_channels = config.get('conv_channels', 32)
        residual_channels = config.get('residual_channels', 32)
        skip_channels = config.get('skip_channels', 64)
        end_channels = config.get('end_channels', 128)
        seq_length = config.get('seq_in_len', 12)
        # in_dim: METR-LA uses 2 (speed + time_of_day), others use 1
        in_dim = config.get('in_dim', 1)
        out_dim = config.get('seq_out_len', 12)
        layers = config.get('layers', 3)
        propalpha = config.get('propalpha', 0.05)
        tanhalpha = config.get('tanhalpha', 3)
        
        # Predefined Adjacency Matrix (needed for gtnet)
        # Check both 'predefined_A' and 'adj_mx' keys for compatibility
        predefined_A = config.get('predefined_A', None)
        if predefined_A is None:
            predefined_A = config.get('adj_mx', None)

        # Convert numpy array to torch tensor
        # Original FDW (train_multi_step.py:137): predefined_A - eye(N) to remove self-loops
        if predefined_A is not None:
            import numpy as np
            if isinstance(predefined_A, np.ndarray):
                adj = predefined_A.astype(np.float32)
                # Remove self-loops (original FDW behavior)
                adj = adj - np.eye(adj.shape[0], dtype=np.float32)
                # Clip negative values to 0 (in case original had no self-loops)
                adj = np.clip(adj, 0, None)
                predefined_A = torch.from_numpy(adj).to(self.device)
            elif isinstance(predefined_A, torch.Tensor):
                # Remove self-loops
                predefined_A = predefined_A.float() - torch.eye(predefined_A.shape[0])
                predefined_A = torch.clamp(predefined_A, min=0).to(self.device)
            print(f"FDW: Using predefined adjacency matrix, shape={predefined_A.shape}")
        else:
            print(f"FDW: No adjacency matrix provided, using learned graph")

        self.model = gtnet(
            gcn_true=gcn_true, 
            buildA_true=buildA_true, 
            gcn_depth=gcn_depth, 
            num_nodes=num_nodes, 
            device=self.device, 
            predefined_A=predefined_A, 
            dropout=dropout, 
            subgraph_size=subgraph_size, 
            node_dim=node_dim, 
            dilation_exponential=dilation_exponential, 
            conv_channels=conv_channels, 
            residual_channels=residual_channels, 
            skip_channels=skip_channels, 
            end_channels=end_channels, 
            seq_length=seq_length, 
            in_dim=in_dim, 
            out_dim=out_dim, 
            layers=layers, 
            propalpha=propalpha, 
            tanhalpha=tanhalpha, 
            layer_norm_affline=True
        ).to(self.device)

    def forward(self, batch):
        """
        batch: Dict from data loader
               x: (Batch, Time, Node, Channel)
        FDW expects: (Batch, Channel, Node, Time) ??? 
        Actually FDW `net.py` / `train_multi_step.py`:
        Model input shape inspection required. 
        In train_multi_step.py: 
             trainx = torch.Tensor(x).to(device)
             trainx = trainx.transpose(1, 3) 
        The loader output (Data_solve/util.py) produces (Mean process... wait)
        Let's look at FDW logic in our wrapper.
        
        Standard Input: (Batch, Time, Node, Channel)
        FDW Logic:
           In forward(): input = input.transpose(1,3)
           
        """
        x = batch['x'].to(self.device) # (B, T, N, C)
        
        # FDW uses (Batch, In_Dim, Num_Nodes, Seq_Len) usually?
        # Let's verify input shape expectation from user code analysis
        # From `train_multi_step.py`:
        # trainx = trainx.transpose(1, 3)
        # If input was (B, T, N, C), transpose(1,3) -> (B, C, N, T)
        # FDW `gtnet` forward: 
        #   input shape is passed to start_conv(in_dim, residual_channels)
        #   Conv2d(in_dim, ...) -> suggests (B, C, H, W) where C=in_dim
        #   So yes, (B, C, N, T) seems correct.
        
        x = x.permute(0, 3, 2, 1) # (B, T, N, C) -> (B, C, N, T)
        
        # Masking logic support (VSF)
        # If 'mask_remaining' is set in config (Subset forecasting)
        # Need to handle idx_subset if relevant
        
        idx = None
        if 'idx' in batch:
            idx = batch['idx']  # for subset selection

        output = self.model(x, idx=idx, args=self.args)  # pass args for gtnet compatibility 
        
        # Output handling
        # FDW gtnet returns (B, Out_Dim=Seq_Len, N, 1) based on analysis of shape (2, 12, 10, 1).
        # We want to return (B, T, N, C) which is (B, 12, 10, 1).
        # So we do NOT need to permute.
        
        # Ensure channel dimension exists
        if output.dim() == 3:
             output = output.unsqueeze(-1)
             
        # If output is (B, C, N, T) -> (B, 12, 10, 1) where C=12? No, standard is (B, T, N, C).
        # Previous permute(0, 3, 2, 1) on (2, 12, 10, 1) gave (2, 1, 10, 12).
        # So original was (2, 12, 10, 1).
        # This matches (B, T, N, C).
        
        return {'pred': output}
