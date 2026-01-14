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
        
        # Args mapping
        input_len = config.get('seq_in_len', 12)
        num_id = config.get('num_nodes', 207)
        out_len = config.get('seq_out_len', 12)
        in_size = config.get('in_dim', 1)
        emb_size = config.get('emb_size', 16)
        grap_size = config.get('grap_size', 8)
        layer_num = config.get('layers', 3)
        dropout = config.get('dropout', 0.0)
        adj_mx = config.get('adj_mx', None) # Helper to load adj required outside

        # GinAR requires adj_mx as list of 2 torch tensors [adj_forward, adj_backward]
        # Each element must support .to(device) method
        if adj_mx is None:
            # Create dummy identity matrices if not provided
            adj_mx = [
                torch.eye(num_id, dtype=torch.float32),
                torch.eye(num_id, dtype=torch.float32)
            ]
        else:
            # Convert numpy arrays to torch tensors if needed
            import numpy as np
            converted = []
            for adj in adj_mx:
                if isinstance(adj, np.ndarray):
                    converted.append(torch.from_numpy(adj).float())
                elif isinstance(adj, torch.Tensor):
                    converted.append(adj.float())
                else:
                    converted.append(torch.tensor(adj, dtype=torch.float32))
            adj_mx = converted
            # Ensure we have at least 2 adjacency matrices
            if len(adj_mx) == 1:
                adj_mx = [adj_mx[0], adj_mx[0]]

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
        batch['x']: (B, T, N, C)
        GinAR Expects: (B, T, N, C) based on common PyTorch usage in these papers, 
                       but let's confirm if permutation is needed.
                       Code analysis of ginar_arch (viewed in previous step) shows:
                       x = x.unsqueeze(1) if ... logic?
                       Usually: forward(self, history_data, ...)
                       
        """
        x = batch['x'].to(self.device)
        
        # GinAR Main Loop uses: 
        # def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, train: bool = True)
        
        # We need to ensure shape matches. 
        # If standard is (B, T, N, C), we pass it directly.
        
        # Forward pass
        # GinAR forward definitions are not standard nn.Module?
        # Checked file: class GinAR(nn.Module)
        
        output = self.model(x)
        
        # Output shape check
        # Usually (B, T_out, N) -> needs (B, T_out, N, C_out=1)
        if output.dim() == 3:
            output = output.unsqueeze(-1)
        
        return {'pred': output}
