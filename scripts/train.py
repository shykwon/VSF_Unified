
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_dataset
from src.core.trainer import UnifiedTrainer
from src.models.fdw.wrapper import FDWWrapper
from src.models.ginar.wrapper import GinARWrapper
from src.models.srdi.wrapper import SRDIWrapper
from src.models.csdi.wrapper import CSDIWrapper
from src.models.saits.wrapper import SAITSWrapper

def main():
    parser = argparse.ArgumentParser(description="VSF Training Script")
    parser.add_argument("--model", type=str, required=True, choices=['fdw', 'ginar', 'srdi', 'csdi', 'saits'], help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (METR-LA, SOLAR, etc.)")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--window_size", type=int, default=12, help="Input window size")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon")
    parser.add_argument("--debug", action="store_true", help="Run with small subset for debugging")
    
    args = parser.parse_args()
    
    limit_samples = 200 if args.debug else None
    
    print(f"Running Experiment: Model={args.model.upper()}, Dataset={args.dataset}, Device={args.device}, Debug={args.debug}")
    
    # 1. Load Data
    print("Loading data...")
    train_set = load_dataset(args.dataset, args.data_dir, 
                             window_size=args.window_size, horizon=args.horizon, 
                             mode='train', limit_samples=limit_samples)
    val_set = load_dataset(args.dataset, args.data_dir, 
                           window_size=args.window_size, horizon=args.horizon, 
                           mode='val', limit_samples=limit_samples)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    
    # 2. Initialize Model
    # Determine input dimensions from data
    # Standard format: (B, T, N, C)
    # Most models want num_nodes
    sample = train_set[0]
    # sample['x'] is (T_in, N, C)
    
    T, N, C = sample['x'].shape
    
    config = {
        'num_nodes': N,
        'seq_in_len': args.window_size,
        'seq_out_len': args.horizon,
        'in_dim': C,
        'input_dim': C, # Some use this name
        'hidden_dim': 32, # Default
        'device': args.device,
        'window_size': args.window_size, # Some wrappers use this
        'horizon': args.horizon, # Some wrappers use this
        # Model specific defaults
        'n_layers': 2,
        'dropout': 0.1,
        'layers': 2 # FDW specific?
    }
    
    print(f"Model Config: num_nodes={N}, seq_in={args.window_size}, seq_out={args.horizon}")
    
    if args.model == 'fdw':
        model = FDWWrapper(config)
    elif args.model == 'ginar':
        model = GinARWrapper(config)
    elif args.model == 'srdi':
        model = SRDIWrapper(config)
    elif args.model == 'csdi':
        # CSDI might need target_dim same as num_nodes
        config['target_dim'] = N
        model = CSDIWrapper(config)
    elif args.model == 'saits':
        model = SAITSWrapper(config)
        
    print(f"Model {args.model.upper()} initialized.")
    
    # 3. Setup Trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = UnifiedTrainer(model, optimizer, config, device=args.device)
    
    # 4. Run Training
    save_path = f"experiments/best_{args.model}_{args.dataset}.pt"
    os.makedirs("experiments", exist_ok=True)
    
    trainer.fit(train_loader, val_loader, epochs=args.epochs, save_path=save_path)
    
if __name__ == "__main__":
    main()
