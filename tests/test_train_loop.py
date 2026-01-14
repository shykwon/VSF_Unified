import torch
import sys
import os
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.trainer import UnifiedTrainer
from src.core.model import BaseVSFModel
from src.data.loader import SyntheticVSFDataset
from src.data.scaler import StandardScaler

class DummyModel(BaseVSFModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1) # simple mapping
    
    def forward(self, batch):
        x = batch['x'] # (B, T, N, C)
        # Just return x as pred for simplicity
        return {'pred': self.layer(x)}

def test_train_loop():
    print("Testing Train Loop and Scaler...")
    
    device = 'cpu'
    config = {
        'device': device,
        'loss_fn': 'mse', # Test with MSE
        'log_dir': './tests/logs',
        'scaler': StandardScaler() # Test with Scaler
    }
    
    if os.path.exists(config['log_dir']):
        shutil.rmtree(config['log_dir'])
    os.makedirs(config['log_dir'])
    
    # 1. Create Dataset & DataLoader
    dataset = SyntheticVSFDataset(num_samples=20, window_size=5, horizon=5)
    
    # Fit scaler manually for test
    # In real usage, loader does this.
    config['scaler'].fit(dataset.data)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    
    # 2. Create Model & Trainer
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    trainer = UnifiedTrainer(model, optimizer, config)
    
    # 3. Call fit
    try:
        trainer.fit(loader, loader, epochs=2)
        print("Trainer.fit() executed successfully.")
    except Exception as e:
        print(f"Trainer.fit() failed: {e}")
        raise e
        
    # Check if model saved
    if os.path.exists(os.path.join(config['log_dir'], 'best_model.pth')):
        print("Best model saved successfully.")
    else:
        print("Best model NOT saved.")

if __name__ == "__main__":
    test_train_loop()
