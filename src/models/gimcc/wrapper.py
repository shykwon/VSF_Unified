import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.model import BaseVSFModel

# Add GIMCC source path
GIMCC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/gimcc")
sys.path.insert(0, GIMCC_PATH)

try:
    from models.CausalDiscovery import CD_src, CD_trg, CD_subset
    from models.Imputer import Imputer
    from models.graph.subgraph_matching import OrderEmbedder
    from forecasters.MTGNN import gtnet
    from models.graph.wl_matching import weisfeiler_lehman_kernel
    from models.graph.graph_utils import add_gaussian_noise_to_nodes, create_pyg_data
    from util import masked_mae, zero_out_remaining_input
except ImportError as e:
    print(f"Warning: Could not import GIMCC modules. Error: {e}")
    import traceback
    traceback.print_exc()
    CD_src = None

class GIMCCArgs:
    """Helper to convert dictionary config to object attributes for GIMCC compatibility."""
    def __init__(self, config):
        self.device = config.get('device', 'cpu')
        
        # Default GIMCC args (from main.py parser defaults)
        self.num_nodes = config.get('num_nodes', 207)
        self.dropout = config.get('dropout', 0.3)
        self.gcn_depth = config.get('gcn_depth', 2)
        self.subgraph_size = config.get('subgraph_size', 20)
        self.node_dim = config.get('node_dim', 40)
        self.dilation_exponential = config.get('dilation_exponential', 1)
        self.conv_channels = config.get('conv_channels', 32)
        self.residual_channels = config.get('residual_channels', 32)
        self.skip_channels = config.get('skip_channels', 64)
        self.end_channels = config.get('end_channels', 128)
        self.seq_in_len = config.get('seq_in_len', 12)
        self.in_dim = config.get('in_dim', 1) 
        self.seq_out_len = config.get('seq_out_len', 12)
        self.layers = config.get('layers', 3)
        self.propalpha = config.get('propalpha', 0.05)
        self.tanhalpha = config.get('tanhalpha', 3)
        self.imputer_name = config.get('imputer_name', 'FourImputer')
        self.lag = config.get('lag', 11) # used in OrderEmbedder input_dim
        
        # For MTGNN
        self.gcn_true = config.get('gcn_true', True)
        self.buildA_true = config.get('buildA_true', True)
        self.adj_identity_train_test = config.get('adj_identity_train_test', False)
        
        # Weights
        self.w_fc = config.get('w_fc', 1.0)
        self.w_ssl = config.get('w_ssl', 0.8)
        self.w_subgraph = config.get('w_subgraph', 0.1)
        self.w_graph = config.get('w_graph', 0.1)

        # Causal Discovery specific
        self.hidden = config.get('hidden', [256]) # used in cMLP
        
        # Misc
        self.cl = config.get('cl', True) # curriculum learning
        
        # Imputer specific
        self.mid_channels = config.get('mid_channels', 128)
        self.sequence_len = self.seq_in_len
        self.fourier_modes = config.get('fourier_modes', self.sequence_len // 2)
        self.input_channels = self.num_nodes * self.in_dim

class GIMCCWrapper(BaseVSFModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        self.args = GIMCCArgs(config)
        
        # Initialize GIMCC components
        self.cd_src = CD_src(self.args).to(self.device)
        self.cd_trg = CD_trg(self.args).to(self.device)
        self.cd_subset = CD_subset(self.args).to(self.device)
        self.decoder = Imputer(self.args, imputer_name=self.args.imputer_name, activation='relu', dropout=0.3).to(self.device)
        
        # Subgraph matching model
        # Note: OrderEmbedder input_dim logic from main.py
        # subgraph_model = OrderEmbedder(input_dim=args.lag, hidden_dim=64, device=device)
        self.subgraph_model = OrderEmbedder(input_dim=self.args.lag, hidden_dim=64, device=self.device).to(self.device)
        
        # Forecaster (MTGNN / gtnet)
        # Handle predefined_A if passed (usually None for pure learning)
        predefined_A = config.get('adj_mx', None)
        
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
        
        # Helper for subgraph labels
        self.subgraph_labels = torch.tensor([1]*1 + [0]*1).to(self.device)
        self.subgraph_criterion = nn.NLLLoss()

        self.loss_fn_metric = masked_mae

    def forward(self, batch):
        """
        GIMCC Forward and Loss Calculation.
        Incorporates logic from GIMCC/trainer.py 'train' method.
        """
        # Data Prep
        if isinstance(batch, (tuple, list)):
             x, y = batch
        else:
             x = batch['x']
             y = batch['y']
        
        x = x.to(self.device) 
        y = y.to(self.device)
        
        # GIMCC expects (B, C, N, T) or similar permutations
        # From trainer.py:
        # trainx = trainx.transpose(1, 3) -> (B, C, N, T) if starting from (B, T, N, C)?
        # Let's check main.py line 340: trainx = trainx.transpose(1, 3)
        # If input is (B, T, N, C), transpose(1,3) -> (B, C, N, T).
        
        input_data = x.permute(0, 3, 2, 1) # (B, C, N, T)
        real_val = y.permute(0, 3, 2, 1)   # (B, C, N, T)
        
        # Usually GIMCC handles 1 channel for now based on args.in_dim default 1
        # If multiple channels, we might need adjustments.
        # trainer.py: tx = trainx[:, :, id, :] => GIMCC seems to slice nodes?
        
        idx = torch.arange(self.args.num_nodes).to(self.device)
        
        # Subset selection (Random or Predefined)
        # For wrapper, we simulate "Random Node Split" behavior per batch if training
        # But doing it per batch might be unstable? GIMCC does it per epoch/run.
        # For simplicity, let's use all nodes or a simple subset strategy.
        # trainer.py uses 'idx_subset' passed from main.py loop.
        
        # We will define a simplified subset strategy here or usage all nodes if not specified.
        # GIMCC core value is Causal Discovery on Subsets.
        # Let's generate a random subset on the fly for training.
        
        if self.training:
            # Random subset logic
            # lower_limit=15, upper=15 default. That's 15%?
            k = int(self.args.num_nodes * 0.15) 
            if k < 1: k = 1
            idx_subset = torch.randperm(self.args.num_nodes)[:k].to(self.device)
        else:
            # Validation/Test: usage all or subset?
            # main.py inference loop uses zero_out_remaining_input but then evaluates.
            # Usually we evaluate on the target nodes.
            # Let's assume full validation for now, or use the same subset logic.
            # For stability, let's use all nodes during eval if GIMCC allows.
            idx_subset = idx 

        # --- Forward Flow (following trainer.py logic) ---
        
        x_src = input_data.clone()
        # Zero out remaining
        # x_subset depends on implementation of zero_out_remaining_input
        # We verify if we imported it correctly.
        x_subset = zero_out_remaining_input(input_data.clone(), idx_subset, self.device)

        # 1. Causal Discovery (src)
        pred_src, dag = self.cd_src(x_src)
        last_src = x_src[:, :, :, -1]
        loss_CD_src, _ = self.loss_fn_metric(pred_src, last_src, 0.0)

        # 2. Causal Discovery (subset)
        pred_subset, graph_subset = self.cd_subset(x_subset)
        last_subset = x_subset[:, :, :, -1]
        loss_CD_subset, _ = self.loss_fn_metric(pred_subset, last_subset, 0.0)
        
        # 3. Imputation
        imputed_data = self.decoder(x_subset, dag)
        # Enforce known values
        # idx_subset are the KEPT nodes. Wait. 
        # zero_out_remaining_input keeps idx_subset and zeros others.
        # So we should enforce idx_subset values from x_subset (which are true values).
        # Imputer fills the ZEROS.
        # trainer.py line 103: imputed_data[:, :, idx_subset, :] = x_subset[:, :, idx_subset, :]
        
        # Note: In PyTorch, tensor assignment with advanced indexing can be tricky in graph.
        # But this is standard practice.
        # We need to make sure shapes match.
        imputed_data_clone = imputed_data.clone() # Avoid inplace error if needed
        imputed_data_clone[:, :, idx_subset, :] = x_subset[:, :, idx_subset, :]
        imputed_data = imputed_data_clone

        # 4. Causal Discovery (trg - on imputed)
        pred_trg, graph_trg = self.cd_trg(imputed_data)
        last_trg = imputed_data[:, :, :, -1]
        loss_CD_trg, _ = self.loss_fn_metric(pred_trg, last_trg, 0.0)
        
        # 5. Global Graph Matching Loss
        loss_global = weisfeiler_lehman_kernel(dag, graph_trg)
        
        # 6. Subgraph Encoder Loss
        # This part requires negative sampling etc.
        # Simplified:
        pos_src = x_src[:, :, :, :-1].squeeze(-1) if x_src.shape[-1]==1 else x_src[:, :, :, :-1].mean(dim=1) # Handle shape
        # Actually x_src is (B, C, N, T). :-1 is time.
        # trainer.py line 115: pos_src = x_src[:, :, :, :-1].squeeze() 
        # squeeze() is dangerous if batch=1.
        
        # Let's skip complex Subgraph Model training in the Wrapper for now IF it causes shape issues, 
        # but to be faithful we should try (or dummy it if it's too heavy).
        # Assuming we can run it:
        pos_src = x_src[:, :, :, :-1].mean(dim=1) # (B, N, T-1) approx?
        pos_trg = x_subset[:, :, :, :-1].mean(dim=1)
        
        # Negatives
        neg_src = add_gaussian_noise_to_nodes(pos_src)
        neg_trg = add_gaussian_noise_to_nodes(pos_trg)
        
        pos_a = create_pyg_data(dag, pos_src)
        pos_b = create_pyg_data(graph_subset, pos_trg)
        neg_a = create_pyg_data(dag, neg_src)
        neg_b = create_pyg_data(graph_subset, neg_trg)
        
        # Subgraph Embeddings
        emb_pos_a = self.subgraph_model.emb_model(pos_a)
        emb_pos_b = self.subgraph_model.emb_model(pos_b)
        emb_neg_a = self.subgraph_model.emb_model(neg_a)
        emb_neg_b = self.subgraph_model.emb_model(neg_b)
        
        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        
        pred_sub = self.subgraph_model(emb_as, emb_bs)
        
        # Cannot use self.subgraph_model.criterion directly if we want to combine losses nicely?
        # trainer.py calls backward() here.
        loss_subgraph_encoder = self.subgraph_model.criterion(pred_sub, None, self.subgraph_labels)
        
        # Subgraph CLF
        pred_sub_clf = self.subgraph_model.predict(pred_sub)
        pred_sub_clf = self.subgraph_model.clf_model(pred_sub_clf.unsqueeze(1))
        loss_local = self.subgraph_criterion(pred_sub_clf, self.subgraph_labels)
        
        # 7. Forecaster
        # Pass imputed data to forecaster
        # MTGNN expects (B, C, N, T)
        
        # idx passed to forecaster?
        # trainer.py line 147: pred_result = self.forecaster(imputed_data, idx=idx, args=args)
        pred_result = self.forecaster(imputed_data, idx=idx, args=self.args)
        
        # pred_result is (B, Out_Dim, N, 1)? Check MTGNN output.
        # Wrapper needs to return standard (B, T, N, C).
        # MTGNN usually returns (B, T_out, N, C_out).
        
        # trainer.py line 148: pred_result = pred_result.transpose(1,3) => (B, C, N, T) ??
        # No, MTGNN usually returns (B, 1, N, T) if C=1?
        # Let's inspect MTGNN:
        # returns x. 
        # end_conv_2 output: (B, out_dim, N, 1). 
        # So shape is (B, T_out, N, 1).
        # We want (B, T_out, N, C).
        # If we return it directly, it is (B, T, N, C).
        
        # However, for Loss Calculation we match 'real'.
        # real is (B, C, N, T) in this function logic.
        
        # Let's permute pred back to (B, C, N, T) for loss calculation inside here.
        pred_for_loss = pred_result.permute(0, 3, 2, 1) # (B, 1, N, T_out)
        
        # Slicing for real value (horizon)
        # real_val is (B, C, N, T_in+T_out)? No, 'y' from loader is usually T_out only.
        # Check loader.py: y is horizon.
        real_for_loss = real_val # (B, C, N, T_out)
        
        # 8. Forecasting Loss & SSL Loss
        # SSL: input vs imputed
        # loss_fc: pred vs real
        
        loss_fc, _ = self.loss_fn_metric(pred_result, y, 0.0) # Using standard shapes (B,T,N,C) for this metric call if possible?
        # Wait, loss_fn_metric (util.masked_mae) handles tensor shapes broadly.
        # But let's use the permuted ones if we stick to GIMCC logic.
        # Actually simplest is using original batch['y'] and pred_result.
        
        loss_fc, _ = self.loss_fn_metric(pred_result, y, 0.0)
        
        # SSL Loss: input(x) vs imputed(x part)
        # imputed_data is (B, C, N, T_in). 
        # Input x permuted is (B, C, N, T_in).
        # We need to inverse permute imputed_data to (B, T, N, C) if we want to match x?
        # Or just use the permuted versions.
        loss_ssl, _ = self.loss_fn_metric(input_data, imputed_data, 0.0)
        
        # Total Loss
        total_loss = (self.args.w_fc * loss_fc) + \
                     (self.args.w_ssl * loss_ssl) + \
                     (self.args.w_graph * loss_global) + \
                     (self.args.w_subgraph * loss_local) + \
                     loss_CD_src + loss_CD_subset + loss_CD_trg + loss_subgraph_encoder
                     
        # If evaluation (not training), we might just return pred.
        # But 'loss' in output is useful for validation tracking.
        
        return {
            'pred': pred_result, # (B, T, N, C)
            'loss': total_loss,
            'aux_losses': {
                'fc': loss_fc,
                'ssl': loss_ssl,
                'global': loss_global,
                'local': loss_local
            }
        }
