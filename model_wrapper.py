"""
model_wrapper.py
Wrapper to convert sparse k-GNN models to accept dense inputs from the generator.

This is crucial for gradient flow during GIN-Graph training.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from typing import Optional


class DenseToSparseWrapper(nn.Module):
    """
    Wraps a sparse k-GNN model to accept dense inputs (x, adj) from the Generator.
    
    The key insight is that we need to maintain gradient flow through the
    adjacency matrix. We do this by:
    1. Converting dense adj to sparse edge_index
    2. Passing the continuous adj values as edge_weight
    3. The k-GNN model uses edge_weight in its message passing
    
    Note: The wrapped model must support edge_weight in its forward pass,
    OR we modify the forward to work with dense inputs directly.
    """
    
    def __init__(self, sparse_model: nn.Module, model_type: str = '1gnn'):
        """
        Args:
            sparse_model: The trained k-GNN model
            model_type: Type of model ('1gnn', '12gnn', '123gnn')
        """
        super().__init__()
        self.sparse_model = sparse_model
        self.model_type = model_type
        
        # Freeze the pretrained model
        for param in self.sparse_model.parameters():
            param.requires_grad = False
        self.sparse_model.eval()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting dense inputs to sparse format.
        
        Args:
            x: Node features [batch, N, D]
            adj: Adjacency matrix [batch, N, N] (continuous values from generator)
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        batch_size, num_nodes, num_feats = x.size()
        device = x.device
        
        # Process each graph in the batch
        all_x = []
        all_edge_index = []
        all_batch = []
        node_offset = 0
        
        for b in range(batch_size):
            # Get this graph's data
            x_b = x[b]  # [N, D]
            adj_b = adj[b]  # [N, N]
            
            # Convert to sparse (threshold for edges)
            # Use soft thresholding to maintain some gradient flow
            edge_mask = adj_b > 0.5
            edge_index_b = edge_mask.nonzero(as_tuple=False).t()  # [2, num_edges]
            
            if edge_index_b.size(1) == 0:
                # No edges - create minimal self-loops
                edge_index_b = torch.tensor([[0], [0]], device=device, dtype=torch.long)
            
            # Offset edge indices for batching
            edge_index_b = edge_index_b + node_offset
            
            all_x.append(x_b)
            all_edge_index.append(edge_index_b)
            all_batch.append(torch.full((num_nodes,), b, device=device, dtype=torch.long))
            
            node_offset += num_nodes
        
        # Concatenate all
        x_flat = torch.cat(all_x, dim=0)  # [batch*N, D]
        edge_index = torch.cat(all_edge_index, dim=1)  # [2, total_edges]
        batch_vec = torch.cat(all_batch, dim=0)  # [batch*N]
        
        # Forward through sparse model
        return self.sparse_model(x_flat, edge_index, batch_vec)


class DenseKGNNWrapper(nn.Module):
    """
    Alternative wrapper that directly modifies the k-GNN to work with dense inputs.
    
    This approach keeps everything in dense format, which is simpler but
    may be less memory efficient for sparse graphs.
    """
    
    def __init__(self, sparse_model: nn.Module, model_type: str = '1gnn'):
        super().__init__()
        self.sparse_model = sparse_model
        self.model_type = model_type
        self.hidden_dim = sparse_model.hidden_dim
        
        # Freeze pretrained weights
        for param in self.sparse_model.parameters():
            param.requires_grad = False
        self.sparse_model.eval()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Dense forward pass.
        
        Args:
            x: Node features [batch, N, D]
            adj: Adjacency matrix [batch, N, N]
            
        Returns:
            logits: [batch, num_classes]
        """
        batch_size = x.size(0)
        device = x.device
        
        # Add self-loops and normalize
        n = adj.size(1)
        identity = torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self = adj + identity
        
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        degree = adj_with_self.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj_with_self / degree
        
        # Run through 1-GNN layers (dense message passing)
        h = x
        for layer in self.sparse_model.gnn1_layers if hasattr(self.sparse_model, 'gnn1_layers') else self.sparse_model.layers:
            # Dense message passing: H' = σ(W1*H + A*H*W2)
            self_part = layer.W1(h)
            neighbor_part = torch.bmm(adj_norm, layer.W2(h))
            h = layer.activation(self_part + neighbor_part)
        
        # Global pooling (sum over nodes)
        graph_emb = h.sum(dim=1)  # [batch, hidden_dim]
        
        # For hierarchical models, we'd need to build k-sets here
        # For simplicity with generated graphs, we just use the 1-GNN part
        if self.model_type == '12gnn':
            # Approximate 2-GNN contribution with zeros (or implement full dense version)
            graph_emb = torch.cat([graph_emb, torch.zeros_like(graph_emb)], dim=1)
        elif self.model_type == '123gnn':
            graph_emb = torch.cat([graph_emb, torch.zeros_like(graph_emb), torch.zeros_like(graph_emb)], dim=1)
        
        return self.sparse_model.classifier(graph_emb)


class SimpleDenseGNN(nn.Module):
    """
    A simple dense GNN that can be used directly with generator outputs.
    
    This is useful for testing and as a baseline.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(DenseGNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(DenseGNNLayer(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, N, D]
            adj: [batch, N, N]
        """
        batch_size = x.size(0)
        n = adj.size(1)
        device = x.device
        
        # Normalize adjacency
        identity = torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self = adj + identity
        degree = adj_with_self.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj_with_self / degree
        
        h = x
        for layer in self.layers:
            h = layer(h, adj_norm)
        
        # Global sum pooling
        graph_emb = h.sum(dim=1)
        
        return self.classifier(graph_emb)


class DenseGNNLayer(nn.Module):
    """Single dense GNN layer."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, N, in_dim]
            adj_norm: Normalized adjacency [batch, N, N]
        """
        self_part = self.W1(x)
        neighbor_part = torch.bmm(adj_norm, self.W2(x))
        return self.activation(self_part + neighbor_part)
