import torch
import torch.nn as nn
import torch.nn.functional as F

class GIN_Generator(nn.Module):
    def __init__(self, latent_dim, max_nodes, num_node_feats, hidden_dim=256, dropout=0.0):
        """
        Args:
            latent_dim (int): Size of the input noise vector z.
            max_nodes (int): Maximum number of nodes (N) for the graph (28 for MUTAG).
            num_node_feats (int): Number of atom types (7 for MUTAG).
            hidden_dim (int): Size of hidden layers.
            dropout (float): Dropout rate for regularization (optional).
        """
        super(GIN_Generator, self).__init__()
        self.n = max_nodes
        self.d = num_node_feats
        
        # 1. Flexible Backbone
        # We use hidden_dim passed as an argument instead of hardcoded numbers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # LeakyReLU is often better for GANs than ReLU
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # 2. Heads with dynamic input sizes
        # Head for Adjacency Matrix (N x N)
        self.adj_head = nn.Linear(hidden_dim * 2, self.n * self.n)
        
        # Head for Node Features (N x D)
        self.feat_head = nn.Linear(hidden_dim * 2, self.n * self.d)

    def forward(self, z, temp=1.0):
        """
        Args:
            z (Tensor): The noise vector [batch_size, latent_dim] created outside.
            temp (float): Gumbel-Softmax temperature.
        """
        batch_size = z.size(0) # z usually is sampled from a normal N(0, I) distrubution
        
        # Pass noise through backbone
        h = self.fc(z)
        
        # Reshape output to (Batch, N, N) for Adjacency
        raw_adj = self.adj_head(h).view(batch_size, self.n, self.n)
        
        # Symmetrize the adjacency matrix (Undirected Graph constraint)
        # Molecules are undirected; edge (i,j) must equal edge (j,i)
        raw_adj = (raw_adj + raw_adj.transpose(1, 2)) / 2
        
        # Reshape output to (Batch, N, D) for Node Features
        raw_feat = self.feat_head(h).view(batch_size, self.n, self.d)
        
        # --- Gumbel-Softmax ---
        
        # 1. Adjacency: Shape [Batch, N, N, 2] -> Sample 0 (No edge) or 1 (Edge)
        adj_logits = torch.stack([raw_adj, -raw_adj], dim=-1)
        A_tilde = F.gumbel_softmax(adj_logits, tau=temp, hard=True)[:, :, :, 0]
        
        # 2. Features: Shape [Batch, N, D] -> Sample one atom type
        X_tilde = F.gumbel_softmax(raw_feat, tau=temp, hard=True)
        
        return A_tilde, X_tilde