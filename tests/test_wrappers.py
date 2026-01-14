import sys
import os
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

# Imports moved inside main to allow logging capture

def test_wrapper(name, wrapper_cls, config, sample_batch):
    print(f"Testing {name}...", end=" ")
    try:
        model = wrapper_cls(config)
        
        # Move to device if needed? Assumed CPU for test
        
        output = model(sample_batch)
        
        if 'pred' in output:
             print(f"Success! Output shape: {output['pred'].shape}")
        elif 'loss' in output:
             print(f"Success! Loss: {output['loss']}")
        else:
             print(f"Success! (Unknown output keys: {output.keys()})")
             
    except Exception as e:
        print(f"Failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Redirect stdout/stderr - Enabled for full log capture
    log_file = open("verification_log.txt", "w")
    sys.stdout = log_file
    sys.stderr = log_file

    print("Starting verification...")
    
    # Delayed Imports
    from src.data.loader import SyntheticVSFDataset
    from src.models.fdw import FDWWrapper
    from src.models.ginar import GinARWrapper
    from src.models.srdi import SRDIWrapper
    from src.models.csdi import CSDIWrapper
    from src.models.saits import SAITSWrapper
    from src.models.gimcc.wrapper import GIMCCWrapper

    # Common Config
    config = {
        'num_nodes': 10,
        'input_len': 12,
        'output_len': 12, # for forecasting
        'input_dim': 1,
        'output_dim': 1,
        'hidden_dim': 32,
        'device': 'cpu',
        # specific args
        'seq_len': 12, # duplicate for some models
        'pred_len': 12,
        'adj_mx': None, # Some models might need this
        'subgraph_size': 5, # Fix FDW k out of range (must be <= num_nodes)
        
        # GIMCC Specifics
        'w_fc': 1.0,
        'w_ssl': 0.8,
        'w_subgraph': 0.1,
        'w_graph': 0.1,
        'hidden': [32],
        'cl': True,
        'dropout': 0.3, 
        'num_levels': 2,
        'gcn_true': True,
        'buildA_true': True,
        'gcn_depth': 2,
        'node_dim': 40,
        'dilation_exponential': 1,
        'conv_channels': 16,
        'residual_channels': 16,
        'skip_channels': 32,
        'end_channels': 64,
        'propalpha': 0.05,
        'tanhalpha': 3,
        'imputer_name': 'FourImputer'
    }
    
    # Dataset
    dataset = SyntheticVSFDataset(num_samples=5, seq_len=24, num_nodes=10, num_channels=1, window_size=12, horizon=12)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    models = [
        ("FDW", FDWWrapper),
        ("GinAR", GinARWrapper),
        ("SRDI", SRDIWrapper),
        ("CSDI", CSDIWrapper),
        ("SAITS", SAITSWrapper),
        ("GIMCC", GIMCCWrapper)
    ]

    for name, cls in models:
        test_wrapper(name, cls, config, batch)

    print("GIMCC test usually requires valid adjacency matrix, skipping for generic test if not robust.")
    try:
         # Minimal GIMCC test
         pass
    except:
         pass
         
    log_file.close()

if __name__ == "__main__":
    main()
