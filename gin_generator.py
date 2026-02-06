"""
gin_generator.py
Generator for GIN-Graph model-level explanation.

Based on: "GIN-Graph: A Generative Interpretation Network for 
Model-Level Explanation of Graph Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GINGenerator(nn.Module):
    """
    Generator network for GIN-Graph.
    
    Takes a noise vector z and outputs:
    - A_tilde: Adjacency matrix [batch, N, N]
    - X_tilde: Node feature matrix [batch, N, D]
    
    Uses Gumbel-Softmax for differentiable discrete sampling.
    """
    
    def __init__(
        self,
        latent_dim: int,
        max_nodes: int,
        num_node_feats: int,
        hidden_dim: int = 256,
        dropout: float = 0.0
    ):
        """
        Args:
            latent_dim: Size of the input noise vector z
            max_nodes: Maximum number of nodes (N) for the graph (28 for MUTAG)
            num_node_feats: Number of node feature types (7 for MUTAG)
            hidden_dim: Size of hidden layers
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.n = max_nodes
        self.d = num_node_feats
        self.latent_dim = latent_dim
        
        # Backbone MLP
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Head for Adjacency Matrix (N x N)
        self.adj_head = nn.Linear(hidden_dim * 2, self.n * self.n)

        # Head for Node Features (N x D)
        self.feat_head = nn.Linear(hidden_dim * 2, self.n * self.d)
    
    def forward(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate graphs from noise vectors.
        
        Args:
            z: Noise vector [batch_size, latent_dim]
            temperature: Gumbel-Softmax temperature (lower = more discrete)
            hard: If True, use straight-through estimator for hard samples
        
        Returns:
            A_tilde: Adjacency matrix [batch, N, N]
            X_tilde: Node features [batch, N, D]
        """
        batch_size = z.size(0)
        
        # Pass through backbone
        h = self.backbone(z)
        
        # Generate raw adjacency logits
        raw_adj = self.adj_head(h).view(batch_size, self.n, self.n)
        
        # Symmetrize (undirected graphs)
        raw_adj = (raw_adj + raw_adj.transpose(1, 2)) / 2
        
        # Generate raw node features
        raw_feat = self.feat_head(h).view(batch_size, self.n, self.d)
        
        # Apply Gumbel-Softmax for differentiable discrete sampling
        
        # Adjacency: Binary choice (edge/no-edge)
        # Stack as [batch, N, N, 2] for softmax over last dim
        adj_logits = torch.stack([raw_adj, -raw_adj], dim=-1)
        A_tilde = F.gumbel_softmax(adj_logits, tau=temperature, hard=hard)[:, :, :, 0]

        # Symmetrize: each edge decided by ONE Gumbel sample (upper triangle),
        # then mirrored. Averaging would create 0.5 values with hard=True,
        # causing threshold-based evaluation to drop most edges.
        upper = torch.triu(A_tilde, diagonal=1)
        A_tilde = upper + upper.transpose(1, 2)
        
        # Node features: One-hot categorical
        X_tilde = F.gumbel_softmax(raw_feat, tau=temperature, hard=hard, dim=-1)
        
        return A_tilde, X_tilde
    
    def sample(self, num_samples: int, device: torch.device, temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample graphs from the generator.
        
        Args:
            num_samples: Number of graph
            s to generate
            device: Device to generate on
            temperature: Gumbel-Softmax temperature
            
        Returns:
            A_tilde, X_tilde
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.forward(z, temperature=temperature, hard=True)


class GINDiscriminator(nn.Module):
    """
    Discriminator network for GIN-Graph (WGAN-GP style).
    
    Takes dense graph representations and outputs a scalar score.
    Uses GNN layers for graph-level discrimination.
    """
    
    def __init__(
        self,
        max_nodes: int,
        num_node_feats: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            max_nodes: Maximum number of nodes (N)
            num_node_feats: Number of node feature types (D)
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        # Use dense GCN-like layers that work with [batch, N, N] adjacency
        self.conv1 = DenseGCNLayer(num_node_feats, hidden_dim)
        self.conv2 = DenseGCNLayer(hidden_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [batch, N, D]
            adj: Adjacency matrix [batch, N, N]
            
        Returns:
            score: Discriminator score [batch, 1]
        """
        h = F.relu(self.conv1(x, adj))
        h = F.relu(self.conv2(h, adj))
        
        # Global mean pooling
        graph_emb = torch.mean(h, dim=1)
        
        return self.mlp(graph_emb)
    
    def compute_gradient_penalty(
        self,
        real_x: torch.Tensor,
        real_adj: torch.Tensor,
        fake_x: torch.Tensor,
        fake_adj: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_x, real_adj: Real graph data
            fake_x, fake_adj: Generated graph data
            device: Compute device
            
        Returns:
            Gradient penalty scalar
        """
        batch_size = real_x.size(0)
        
        # Random interpolation factor
        epsilon = torch.rand(batch_size, 1, 1, device=device)
        
        # Interpolated samples
        interp_x = (epsilon * real_x + (1 - epsilon) * fake_x).requires_grad_(True)
        interp_adj = (epsilon * real_adj + (1 - epsilon) * fake_adj).requires_grad_(True)
        
        # Discriminator output on interpolated samples
        interp_scores = self.forward(interp_x, interp_adj)
        
        # Compute gradients
        grad_outputs = torch.ones_like(interp_scores, device=device)
        gradients = torch.autograd.grad(
            outputs=interp_scores,
            inputs=[interp_x, interp_adj],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )
        
        # Flatten and compute norm
        grad_x = gradients[0].view(batch_size, -1)
        grad_adj = gradients[1].view(batch_size, -1)
        grad_flat = torch.cat([grad_x, grad_adj], dim=1)
        
        gradient_norm = grad_flat.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty


class DenseGCNLayer(nn.Module):
    """
    Dense GCN layer that works with batched dense adjacency matrices.
    
    H' = σ(A * H * W)
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, N, in_dim]
            adj: Adjacency matrix [batch, N, N]
            
        Returns:
            Output features [batch, N, out_dim]
        """
        # Add self-loops
        batch_size, n, _ = x.size()
        identity = torch.eye(n, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self = adj + identity
        
        # Normalize (simple symmetric normalization)
        degree = adj_with_self.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj_with_self / degree
        
        # Message passing: A * H
        support = torch.bmm(adj_norm, x)
        
        # Transform
        return self.linear(support)
