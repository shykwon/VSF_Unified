import sys
import os
import torch
from src.core.model import BaseVSFModel

# Add GinAR source path (parent directory so model1 is recognized as package)
GINAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/ginar")
sys.path.insert(0, GINAR_PATH)

try:
    from model1.ginar_arch import GinAR
except ImportError as e:
    print(f"Warning: Could not import GinAR from {GINAR_PATH}. Error: {e}")
    import traceback
    traceback.print_exc()
    GinAR = None

class GinARWrapper(BaseVSFModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')

        # Args mapping - Original defaults from main_ginar.py
        input_len = config.get('seq_in_len', 12)
        num_id = config.get('num_nodes', 207)
        out_len = config.get('seq_out_len', 12)
        in_size = config.get('in_dim', 1)
        emb_size = config.get('emb_size', 16)
        grap_size = config.get('grap_size', 8)
        layer_num = config.get('layers', 2)      # Original: 2
        dropout = config.get('dropout', 0.15)    # Original: 0.15
        adj_mx = config.get('adj_mx', None)

        # GinAR requires adj_mx as list of 2 torch tensors [adj_forward, adj_backward]
        # Original uses "doubletransition": [P.T, P_reverse.T] where P = D^{-1}A
        if adj_mx is None:
            # Create identity matrices if not provided
            # Note: GinAR combines adaptive + predefined graphs, so identity is suboptimal
            adj_mx = [
                torch.eye(num_id, dtype=torch.float32),
                torch.eye(num_id, dtype=torch.float32)
            ]
            print(f"GinAR: No adjacency provided, using identity (adaptive graph only)")
        else:
            import numpy as np

            # Handle single matrix - convert to doubletransition format
            if isinstance(adj_mx, np.ndarray):
                def calculate_transition_matrix(adj):
                    """Calculate transition matrix P = D^{-1}A (original GinAR method)"""
                    adj = adj.astype(np.float32)
                    row_sum = adj.sum(axis=1).flatten()
                    d_inv = np.power(row_sum, -1)
                    d_inv[np.isinf(d_inv)] = 0.
                    # P = D^{-1} * A
                    return (d_inv[:, np.newaxis] * adj)

                # doubletransition: [P.T, P_reverse.T]
                # P = transition_matrix(adj), P_reverse = transition_matrix(adj.T)
                P_forward = calculate_transition_matrix(adj_mx)
                P_backward = calculate_transition_matrix(adj_mx.T)

                adj_mx = [
                    torch.from_numpy(P_forward.T).float(),   # P.T
                    torch.from_numpy(P_backward.T).float()   # P_reverse.T
                ]
                print(f"GinAR: Using predefined adjacency (doubletransition), shape={adj_mx[0].shape}")
            elif isinstance(adj_mx, list):
                converted = []
                for adj in adj_mx:
                    if isinstance(adj, np.ndarray):
                        converted.append(torch.from_numpy(adj).float())
                    elif isinstance(adj, torch.Tensor):
                        converted.append(adj.float())
                    else:
                        converted.append(torch.tensor(adj, dtype=torch.float32))
                adj_mx = converted
                if len(adj_mx) == 1:
                    adj_mx = [adj_mx[0], adj_mx[0].T]
                print(f"GinAR: Using predefined adjacency matrices")

        self.model = GinAR(
            input_len=input_len,
            num_id=num_id,
            out_len=out_len,
            in_size=in_size,
            emb_size=emb_size,
            grap_size=grap_size,
            layer_num=layer_num,
            dropout=dropout,
            adj_mx=adj_mx
        ).to(self.device)

    def forward(self, batch):
        """
        Input:  batch['x'] - (B, H, N, C) where H=history length
        Output: {'pred': (B, L, N, 1)} where L=prediction length

        GinAR internally:
          - Input: (B, H, N, C) → transpose to (B, C, H, N)
          - Output: (B, L, N) → we add channel dim
        """
        x = batch['x'].to(self.device)  # (B, H, N, C)

        output = self.model(x)  # (B, L, N)

        # Add channel dimension for unified interface
        if output.dim() == 3:
            output = output.unsqueeze(-1)  # (B, L, N, 1)

        return {'pred': output}
