#!/usr/bin/env python
"""
VSF Research Platform - Unified Training Script

Usage:
    # Basic usage
    python scripts/train.py --model fdw --dataset metr-la

    # With options
    python scripts/train.py --model csdi --dataset pems-bay --epochs 100 --seed 42

    # Multi-seed experiments
    python scripts/train.py --model fdw --seeds 42,123,456 --dataset metr-la

    # With YAML config
    python scripts/train.py --config configs/experiment.yaml

    # Debug mode (small dataset)
    python scripts/train.py --model fdw --dataset metr-la --debug
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import set_seed, get_device, UnifiedTrainer, Metrics, MaskedMetrics
from src.data.loader import load_dataset
from src.data.scaler import StandardScaler


# ============================================================================
# Model Registry
# ============================================================================
MODEL_REGISTRY = {}


def get_model_wrapper(name):
    """Lazy load model wrappers to avoid import errors."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        if name == 'fdw':
            from src.models.fdw.wrapper import FDWWrapper
            MODEL_REGISTRY['fdw'] = FDWWrapper
        elif name == 'ginar':
            from src.models.ginar.wrapper import GinARWrapper
            MODEL_REGISTRY['ginar'] = GinARWrapper
        elif name == 'csdi':
            from src.models.csdi.wrapper import CSDIWrapper
            MODEL_REGISTRY['csdi'] = CSDIWrapper
        elif name == 'srdi':
            from src.models.srdi.wrapper import SRDIWrapper
            MODEL_REGISTRY['srdi'] = SRDIWrapper
        elif name == 'saits':
            from src.models.saits.wrapper import SAITSWrapper
            MODEL_REGISTRY['saits'] = SAITSWrapper
        elif name == 'gimcc':
            from src.models.gimcc.wrapper import GIMCCWrapper
            MODEL_REGISTRY['gimcc'] = GIMCCWrapper
        else:
            raise ValueError(f"Unknown model: {name}. Available: fdw, ginar, csdi, srdi, saits, gimcc")
    return MODEL_REGISTRY[name]


# ============================================================================
# Early Stopping
# ============================================================================
class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ============================================================================
# Result Logger
# ============================================================================
class ResultLogger:
    """Unified logging for experiments."""

    def __init__(self, log_dir, use_tensorboard=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_log = []
        self.use_tensorboard = use_tensorboard
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
                print(f"TensorBoard enabled: tensorboard --logdir {self.log_dir / 'tensorboard'}")
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

    def log_epoch(self, epoch, train_loss, val_metrics, lr=None):
        """Log epoch results."""
        record = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_metrics.get('loss', 0)),
            'val_mae': float(val_metrics.get('MAE', 0)),
            'val_rmse': float(val_metrics.get('RMSE', 0)),
            'val_mape': float(val_metrics.get('MAPE', 0)),
            'lr': float(lr) if lr else None
        }
        self.train_log.append(record)

        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics.get('loss', 0), epoch)
            self.writer.add_scalar('Metrics/MAE', val_metrics.get('MAE', 0), epoch)
            self.writer.add_scalar('Metrics/RMSE', val_metrics.get('RMSE', 0), epoch)
            self.writer.add_scalar('Metrics/MAPE', val_metrics.get('MAPE', 0), epoch)
            if lr:
                self.writer.add_scalar('LR', lr, epoch)

    def save_config(self, config):
        """Save experiment configuration."""
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def save_results(self, test_metrics, train_time):
        """Save final results."""
        # Convert tensors to floats
        test_metrics_serializable = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                                      for k, v in test_metrics.items()}

        results = {
            'test_metrics': test_metrics_serializable,
            'train_time_seconds': train_time,
            'train_log': self.train_log
        }

        results_path = self.log_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save CSV for easy analysis
        import csv
        csv_path = self.log_dir / 'train_log.csv'
        if self.train_log:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.train_log[0].keys())
                writer.writeheader()
                writer.writerows(self.train_log)

        print(f"Results saved to: {self.log_dir}")

    def close(self):
        if self.writer:
            self.writer.close()


# ============================================================================
# Adjacency Matrix Loading
# ============================================================================
def load_adj_matrix(dataset_name, data_dir):
    """Load adjacency matrix for graph-based models."""
    import pickle

    adj_files = {
        'metr-la': 'adj_mx.pkl',
        'pems-bay': 'adj_mx_bay.pkl',
    }

    dataset_name = dataset_name.lower()
    if dataset_name not in adj_files:
        return None

    # Try multiple possible paths for adjacency matrix
    possible_paths = [
        Path(data_dir) / 'sensor_graph' / adj_files[dataset_name],
        Path(data_dir).parent / 'sensor_graph' / adj_files[dataset_name],  # data/sensor_graph
        Path('data') / 'sensor_graph' / adj_files[dataset_name],  # Absolute fallback
    ]

    adj_path = None
    for p in possible_paths:
        if p.exists():
            adj_path = p
            break

    if adj_path is None:
        print(f"Warning: Adjacency file not found. Tried: {possible_paths}")
        return None

    try:
        with open(adj_path, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
        print(f"Loaded adjacency matrix: {adj_mx.shape}")
        return adj_mx
    except Exception as e:
        print(f"Warning: Failed to load adjacency matrix: {e}")
        return None


# ============================================================================
# Data Loading
# ============================================================================
def create_dataloaders(args):
    """Create train/val/test dataloaders."""
    print(f"\n{'='*60}")
    print(f"Loading dataset: {args.dataset}")
    print(f"{'='*60}")

    limit_samples = 500 if args.debug else None
    if limit_samples:
        print(f"DEBUG MODE: Limiting to {limit_samples} samples")

    # Load data using factory function
    train_dataset, val_dataset, test_dataset, scaler = load_dataset(
        name=args.dataset,
        data_dir=args.data_dir,
        window_size=args.seq_in,
        horizon=args.seq_out,
        stride=1,
        mode='all',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        limit_samples=limit_samples
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Get num_nodes from data
    sample = train_dataset[0]
    num_nodes = sample['x'].shape[1]
    print(f"Num nodes: {num_nodes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, scaler, num_nodes


# ============================================================================
# Training Function
# ============================================================================
def train_single_experiment(args, seed):
    """Run a single training experiment with given seed."""

    # Set seed for reproducibility
    set_seed(seed)

    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.model}_{args.dataset}_seed{seed}_{timestamp}"
    log_dir = Path(args.log_dir) / exp_name

    logger = ResultLogger(log_dir, use_tensorboard=args.tensorboard)

    # Save config
    config_dict = vars(args).copy()
    config_dict['seed'] = seed
    config_dict['exp_name'] = exp_name
    logger.save_config(config_dict)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, test_loader, scaler, num_nodes = create_dataloaders(args)

    # Load adjacency matrix (for graph models)
    adj_mx = load_adj_matrix(args.dataset, args.data_dir)

    # Create model config
    device = get_device(args.device)
    # Set hidden_dim based on GPU profile if not explicitly set
    gpu_profile = getattr(args, 'gpu_profile', '4090')
    if gpu_profile == '1080ti':
        default_hidden = 32
    else:
        default_hidden = 64  # Full paper settings for 4090/A100

    # Use args.hidden_dim if explicitly set, otherwise use profile default
    hidden_dim = args.hidden_dim if args.hidden_dim != 32 else default_hidden

    # in_dim: Original FDW uses 2 for METR-LA (speed + time_of_day), 1 for others
    # However, our data loader currently only loads speed (1 channel)
    # TODO: Add time_of_day feature to data loader for full FDW reproduction
    # For now, keep in_dim=1 to match current data pipeline
    in_dim = 1

    model_config = {
        'num_nodes': num_nodes,
        'seq_in_len': args.seq_in,
        'seq_out_len': args.seq_out,
        'pred_len': args.seq_out,
        'seq_len': args.seq_in,
        'in_dim': in_dim,
        'input_dim': in_dim,
        'hidden_dim': hidden_dim,
        'device': device,
        'adj_mx': adj_mx,
        'gpu_profile': gpu_profile,  # Pass to model wrappers
        # Model-specific configs
        'd_model': hidden_dim,
        'n_head': 4,
        'd_k': 32,
        'd_v': 32,
        'd_inner': hidden_dim * 2,
        'dropout': args.dropout,
        'n_layers': 2,
        'layers': 2,
        'window_size': args.seq_in,
        'horizon': args.seq_out,
    }

    # Create model
    print(f"\nInitializing model: {args.model.upper()}")
    ModelWrapper = get_model_wrapper(args.model)
    model = ModelWrapper(model_config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Create trainer
    trainer_config = {
        'device': device,
        'loss_fn': args.loss,
        'scaler': scaler,
        'log_dir': str(log_dir)
    }
    trainer = UnifiedTrainer(model, optimizer, trainer_config)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs...")
    print(f"Loss function: {args.loss}")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics['loss']

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        logger.log_epoch(epoch, train_loss, val_metrics, current_lr)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"MAE: {val_metrics['MAE']:.4f} | "
              f"RMSE: {val_metrics['RMSE']:.4f} | "
              f"Time: {epoch_time:.1f}s", end='')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save(str(log_dir / 'best_model.pth'))
            print(" ★ Best", end='')

        print()  # New line

        # LR scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n>>> Early stopping triggered at epoch {epoch}")
            break

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f}min)")

    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating best model on test set...")
    print(f"{'='*60}")

    trainer.load(str(log_dir / 'best_model.pth'))
    test_metrics = trainer.evaluate(test_loader)

    print(f"\n>>> Test Results:")
    print(f"    MAE:         {test_metrics['MAE']:.4f}")
    print(f"    RMSE:        {test_metrics['RMSE']:.4f}")
    print(f"    MAPE:        {test_metrics['MAPE']:.2f}%")
    print(f"    MaskedMAE:   {test_metrics['MaskedMAE']:.4f}")
    print(f"    MaskedRMSE:  {test_metrics['MaskedRMSE']:.4f}")

    # Save results
    logger.save_results(test_metrics, train_time)
    logger.close()

    return test_metrics, log_dir


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='VSF Research Platform - Unified Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and Dataset
    parser.add_argument('--model', type=str, default='fdw',
                        choices=['fdw', 'ginar', 'csdi', 'srdi', 'saits', 'gimcc'],
                        help='Model to train')
    parser.add_argument('--dataset', type=str, default='metr-la',
                        choices=['metr-la', 'pems-bay', 'solar', 'traffic', 'electricity'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Data directory')

    # Sequence lengths
    parser.add_argument('--seq_in', type=int, default=12,
                        help='Input sequence length')
    parser.add_argument('--seq_out', type=int, default=12,
                        help='Output sequence length (prediction horizon)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension')

    # Loss function
    parser.add_argument('--loss', type=str, default='masked_mae',
                        choices=['mse', 'masked_mae'],
                        help='Loss function')

    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    # Data splits
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')

    # Seeds
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (single experiment)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Multiple seeds separated by comma (e.g., "42,123,456")')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file (overrides CLI args)')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small dataset, fast iteration)')

    # GPU Profile (determines model size)
    parser.add_argument('--gpu_profile', type=str, default='4090',
                        choices=['1080ti', '4090', 'a100'],
                        help='GPU profile: 1080ti (11GB, reduced), 4090 (24GB, full), a100 (40GB+, full)')

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
            print(f"Loaded config from: {args.config}")
            # Override args with yaml config
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")

    # Debug mode adjustments
    if args.debug:
        args.epochs = min(args.epochs, 5)
        args.patience = 3
        print("DEBUG MODE: epochs=5, patience=3, limited data")

    return args


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("    VSF Research Platform - Training")
    print(f"{'='*60}")
    print(f"  Model:      {args.model.upper()}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Loss:       {args.loss}")
    print(f"  Device:     {args.device}")

    # Determine seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        print(f"  Seeds:      {seeds} (multi-seed mode)")
    else:
        seeds = [args.seed]
        print(f"  Seed:       {args.seed}")

    print(f"{'='*60}\n")

    # Run experiments
    all_results = []
    for i, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n>>> Running experiment {i+1}/{len(seeds)} with seed={seed}")

        test_metrics, log_dir = train_single_experiment(args, seed)
        all_results.append({
            'seed': seed,
            'metrics': {k: float(v) for k, v in test_metrics.items()},
            'log_dir': str(log_dir)
        })

    # Summary for multi-seed experiments
    if len(seeds) > 1:
        print(f"\n{'='*60}")
        print("    Multi-Seed Experiment Summary")
        print(f"{'='*60}")

        mae_scores = [r['metrics']['MAE'] for r in all_results]
        rmse_scores = [r['metrics']['RMSE'] for r in all_results]
        mape_scores = [r['metrics']['MAPE'] for r in all_results]

        print(f"  MAE:  {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")
        print(f"  RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")
        print(f"  MAPE: {np.mean(mape_scores):.2f}% +/- {np.std(mape_scores):.2f}%")

        # Save summary
        summary_path = Path(args.log_dir) / f"{args.model}_{args.dataset}_summary.json"
        summary = {
            'model': args.model,
            'dataset': args.dataset,
            'seeds': seeds,
            'config': vars(args),
            'results': all_results,
            'summary': {
                'MAE_mean': float(np.mean(mae_scores)),
                'MAE_std': float(np.std(mae_scores)),
                'RMSE_mean': float(np.mean(rmse_scores)),
                'RMSE_std': float(np.std(rmse_scores)),
                'MAPE_mean': float(np.mean(mape_scores)),
                'MAPE_std': float(np.std(mape_scores))
            }
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved to: {summary_path}")

    print(f"\n{'='*60}")
    print("    Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
