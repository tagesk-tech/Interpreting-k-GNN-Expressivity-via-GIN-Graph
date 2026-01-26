"""
models_kgnn.py
k-GNN implementations based on Morris et al. "Weisfeiler and Leman Go Neural"

This module implements:
- 1-GNN: Standard message passing on nodes
- 2-GNN: Message passing on 2-sets (node pairs)  
- 3-GNN: Message passing on 3-sets (node triplets)
- Hierarchical variants: 1-2-GNN, 1-3-GNN, 1-2-3-GNN

The key insight from Morris et al. is that k-GNNs operate on k-element subsets
of nodes, with the local neighborhood defined as sets that differ by exactly
one element and where the differing elements are connected in the original graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from typing import Optional, Tuple, Dict


# =============================================================================
# Basic Layer Implementations
# =============================================================================

class OneGNNLayer(MessagePassing):
    """
    1-GNN layer implementing:
    f^(t)(v) = σ( f^(t-1)(v) · W1 + Σ_{u ∈ N(v)} f^(t-1)(u) · W2 )
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr='add')
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self_part = self.W1(x)
        neighbor_part = self.propagate(edge_index, x=x)
        return self.activation(self_part + neighbor_part)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return self.W2(x_j)


class KSetLayer(nn.Module):
    """
    Generic k-set layer for 2-GNN and 3-GNN.
    Performs message passing on k-sets using manually constructed neighborhoods.
    
    f^(t)(s) = σ( f^(t-1)(s) · W1 + Σ_{t ∈ N_L(s)} f^(t-1)(t) · W2 )
    
    where N_L(s) is the local neighborhood of k-set s.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_k_sets, in_dim] features for each k-set
            edge_index: [2, num_edges] connectivity between k-sets
        """
        self_part = self.W1(x)
        
        if edge_index.numel() > 0 and edge_index.size(1) > 0:
            source_feats = self.W2(x[edge_index[0]])
            neighbor_part = torch.zeros_like(self_part)
            neighbor_part.index_add_(0, edge_index[1], source_feats)
        else:
            neighbor_part = torch.zeros_like(self_part)
        
        return self.activation(self_part + neighbor_part)


# =============================================================================
# Helper Functions for k-set Construction
# =============================================================================

def build_local_adj(
    edge_index: torch.Tensor,
    node_indices: torch.Tensor,
    n: int,
    device: torch.device
) -> torch.Tensor:
    """Build local adjacency matrix for a subgraph."""
    adj = torch.zeros((n, n), device=device, dtype=torch.float)
    
    max_idx = max(edge_index.max().item() + 1, node_indices.max().item() + 1)
    global_to_local = torch.full((max_idx,), -1, device=device, dtype=torch.long)
    global_to_local[node_indices] = torch.arange(n, device=device)
    
    mask = (global_to_local[edge_index[0]] >= 0) & (global_to_local[edge_index[1]] >= 0)
    local_src = global_to_local[edge_index[0, mask]]
    local_dst = global_to_local[edge_index[1, mask]]
    
    adj[local_src, local_dst] = 1.0
    adj[local_dst, local_src] = 1.0
    
    return adj


def build_2set_edges(
    pairs: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Build edges between 2-sets based on local neighborhood definition.
    {u, v} ~ {v, w} if (u, w) ∈ E
    """
    num_pairs = pairs.size(0)
    if num_pairs == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    pair_to_idx = {}
    for idx, (u, v) in enumerate(pairs.tolist()):
        pair_to_idx[tuple(sorted((u, v)))] = idx
    
    src_list, dst_list = [], []
    
    for i, (u, v) in enumerate(pairs.tolist()):
        for w in (adj[v] == 1).nonzero(as_tuple=True)[0].tolist():
            if w != u and adj[u, w] == 1:
                target = tuple(sorted((v, w)))
                if target in pair_to_idx:
                    src_list.append(i)
                    dst_list.append(pair_to_idx[target])
        
        for w in (adj[u] == 1).nonzero(as_tuple=True)[0].tolist():
            if w != v and adj[v, w] == 1:
                target = tuple(sorted((u, w)))
                if target in pair_to_idx:
                    src_list.append(i)
                    dst_list.append(pair_to_idx[target])
    
    if src_list:
        return torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    return torch.empty((2, 0), dtype=torch.long, device=device)


def build_3set_edges(
    triplets: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Build edges between 3-sets based on local neighborhood definition.
    {a, b, c} ~ {a, b, d} if (c, d) ∈ E
    """
    num_triplets = triplets.size(0)
    if num_triplets == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    triplet_to_idx = {}
    for idx, (a, b, c) in enumerate(triplets.tolist()):
        triplet_to_idx[tuple(sorted((a, b, c)))] = idx
    
    src_list, dst_list = [], []
    
    for i, (a, b, c) in enumerate(triplets.tolist()):
        for d in (adj[c] == 1).nonzero(as_tuple=True)[0].tolist():
            if d not in [a, b, c]:
                target = tuple(sorted((a, b, d)))
                if target in triplet_to_idx:
                    src_list.append(i)
                    dst_list.append(triplet_to_idx[target])
        
        for d in (adj[b] == 1).nonzero(as_tuple=True)[0].tolist():
            if d not in [a, b, c]:
                target = tuple(sorted((a, c, d)))
                if target in triplet_to_idx:
                    src_list.append(i)
                    dst_list.append(triplet_to_idx[target])
        
        for d in (adj[a] == 1).nonzero(as_tuple=True)[0].tolist():
            if d not in [a, b, c]:
                target = tuple(sorted((b, c, d)))
                if target in triplet_to_idx:
                    src_list.append(i)
                    dst_list.append(triplet_to_idx[target])
    
    if src_list:
        return torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    return torch.empty((2, 0), dtype=torch.long, device=device)


# =============================================================================
# Standalone Models
# =============================================================================

class OneGNN(nn.Module):
    """
    Standalone 1-GNN model.
    Standard message passing GNN operating on nodes.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(OneGNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(OneGNNLayer(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        
        graph_emb = global_add_pool(h, batch)
        return self.classifier(graph_emb)
    
    def get_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Get graph embedding before classification."""
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return global_add_pool(h, batch)


class Hierarchical12GNN(nn.Module):
    """
    Hierarchical 1-2-GNN.
    
    First runs 1-GNN to get node embeddings, then uses those embeddings
    to initialize 2-GNN features. Final representation concatenates
    pooled outputs from both levels.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers_1: int = 3,
        num_layers_2: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1-GNN layers
        self.gnn1_layers = nn.ModuleList()
        self.gnn1_layers.append(OneGNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers_1 - 1):
            self.gnn1_layers.append(OneGNNLayer(hidden_dim, hidden_dim))
        
        # 2-GNN layers (input: concatenated 1-GNN features + iso type)
        two_set_input_dim = 2 * hidden_dim + 1
        self.gnn2_layers = nn.ModuleList()
        self.gnn2_layers.append(KSetLayer(two_set_input_dim, hidden_dim))
        for _ in range(num_layers_2 - 1):
            self.gnn2_layers.append(KSetLayer(hidden_dim, hidden_dim))
        
        # Classifier (concatenates 1-GNN and 2-GNN pooled features)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def _build_2sets(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build 2-sets using 1-GNN embeddings as features."""
        device = h.device
        
        all_feats, all_batch, all_src, all_dst = [], [], [], []
        offset = 0
        
        for g in batch.unique():
            mask = (batch == g)
            nodes = mask.nonzero(as_tuple=True)[0]
            n = nodes.size(0)
            if n < 2:
                continue
            
            pairs = torch.combinations(torch.arange(n, device=device), r=2)
            adj = build_local_adj(edge_index, nodes, n, device)
            
            feat_u = h[nodes[pairs[:, 0]]]
            feat_v = h[nodes[pairs[:, 1]]]
            iso = adj[pairs[:, 0], pairs[:, 1]].unsqueeze(1)
            
            all_feats.append(torch.cat([feat_u, feat_v, iso], dim=1))
            all_batch.append(torch.full((pairs.size(0),), g.item(), device=device, dtype=torch.long))
            
            edges = build_2set_edges(pairs, adj, device)
            if edges.size(1) > 0:
                all_src.append(edges[0] + offset)
                all_dst.append(edges[1] + offset)
            offset += pairs.size(0)
        
        if not all_feats:
            return None, None, None
        
        final_x = torch.cat(all_feats)
        final_batch = torch.cat(all_batch)
        if all_src:
            final_edges = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
        else:
            final_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return final_x, final_edges, final_batch
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        # 1-GNN
        h = x
        for layer in self.gnn1_layers:
            h = layer(h, edge_index)
        graph_emb_1 = global_add_pool(h, batch)
        
        # 2-GNN using 1-GNN embeddings
        two_x, two_edges, two_batch = self._build_2sets(h, edge_index, batch)
        
        if two_x is not None:
            h2 = two_x
            for layer in self.gnn2_layers:
                h2 = layer(h2, two_edges)
            graph_emb_2 = global_add_pool(h2, two_batch)
        else:
            graph_emb_2 = torch.zeros_like(graph_emb_1)
        
        combined = torch.cat([graph_emb_1, graph_emb_2], dim=1)
        return self.classifier(combined)


class Hierarchical123GNN(nn.Module):
    """
    Hierarchical 1-2-3-GNN.
    
    Runs 1-GNN, then uses embeddings for both 2-GNN and 3-GNN.
    Final representation concatenates all three pooled outputs.
    
    This is the most expressive model, capturing structure at all three scales.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers_1: int = 3,
        num_layers_2: int = 2,
        num_layers_3: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1-GNN layers
        self.gnn1_layers = nn.ModuleList()
        self.gnn1_layers.append(OneGNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers_1 - 1):
            self.gnn1_layers.append(OneGNNLayer(hidden_dim, hidden_dim))
        
        # 2-GNN layers
        two_set_input_dim = 2 * hidden_dim + 1
        self.gnn2_layers = nn.ModuleList()
        self.gnn2_layers.append(KSetLayer(two_set_input_dim, hidden_dim))
        for _ in range(num_layers_2 - 1):
            self.gnn2_layers.append(KSetLayer(hidden_dim, hidden_dim))
        
        # 3-GNN layers
        three_set_input_dim = 3 * hidden_dim + 4  # 4 for iso type one-hot
        self.gnn3_layers = nn.ModuleList()
        self.gnn3_layers.append(KSetLayer(three_set_input_dim, hidden_dim))
        for _ in range(num_layers_3 - 1):
            self.gnn3_layers.append(KSetLayer(hidden_dim, hidden_dim))
        
        # Classifier (concatenates all three)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def _build_2sets(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = h.device
        all_feats, all_batch, all_src, all_dst = [], [], [], []
        offset = 0
        
        for g in batch.unique():
            mask = (batch == g)
            nodes = mask.nonzero(as_tuple=True)[0]
            n = nodes.size(0)
            if n < 2:
                continue
            
            pairs = torch.combinations(torch.arange(n, device=device), r=2)
            adj = build_local_adj(edge_index, nodes, n, device)
            
            feat_u = h[nodes[pairs[:, 0]]]
            feat_v = h[nodes[pairs[:, 1]]]
            iso = adj[pairs[:, 0], pairs[:, 1]].unsqueeze(1)
            
            all_feats.append(torch.cat([feat_u, feat_v, iso], dim=1))
            all_batch.append(torch.full((pairs.size(0),), g.item(), device=device, dtype=torch.long))
            
            edges = build_2set_edges(pairs, adj, device)
            if edges.size(1) > 0:
                all_src.append(edges[0] + offset)
                all_dst.append(edges[1] + offset)
            offset += pairs.size(0)
        
        if not all_feats:
            return None, None, None
        return (torch.cat(all_feats),
                torch.stack([torch.cat(all_src), torch.cat(all_dst)]) if all_src else torch.empty((2, 0), dtype=torch.long, device=device),
                torch.cat(all_batch))
    
    def _build_3sets(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = h.device
        all_feats, all_batch, all_src, all_dst = [], [], [], []
        offset = 0
        
        for g in batch.unique():
            mask = (batch == g)
            nodes = mask.nonzero(as_tuple=True)[0]
            n = nodes.size(0)
            if n < 3:
                continue
            
            trips = torch.combinations(torch.arange(n, device=device), r=3)
            adj = build_local_adj(edge_index, nodes, n, device)
            
            fa = h[nodes[trips[:, 0]]]
            fb = h[nodes[trips[:, 1]]]
            fc = h[nodes[trips[:, 2]]]
            
            # Iso type: count edges (0, 1, 2, or 3)
            ec = (adj[trips[:, 0], trips[:, 1]] + 
                  adj[trips[:, 1], trips[:, 2]] + 
                  adj[trips[:, 0], trips[:, 2]]).long()
            iso = torch.zeros((trips.size(0), 4), device=device)
            iso.scatter_(1, ec.unsqueeze(1), 1.0)
            
            all_feats.append(torch.cat([fa, fb, fc, iso], dim=1))
            all_batch.append(torch.full((trips.size(0),), g.item(), device=device, dtype=torch.long))
            
            edges = build_3set_edges(trips, adj, device)
            if edges.size(1) > 0:
                all_src.append(edges[0] + offset)
                all_dst.append(edges[1] + offset)
            offset += trips.size(0)
        
        if not all_feats:
            return None, None, None
        return (torch.cat(all_feats),
                torch.stack([torch.cat(all_src), torch.cat(all_dst)]) if all_src else torch.empty((2, 0), dtype=torch.long, device=device),
                torch.cat(all_batch))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        # 1-GNN
        h = x
        for layer in self.gnn1_layers:
            h = layer(h, edge_index)
        graph_emb_1 = global_add_pool(h, batch)
        
        # 2-GNN
        two_x, two_edges, two_batch = self._build_2sets(h, edge_index, batch)
        if two_x is not None:
            h2 = two_x
            for layer in self.gnn2_layers:
                h2 = layer(h2, two_edges)
            graph_emb_2 = global_add_pool(h2, two_batch)
        else:
            graph_emb_2 = torch.zeros_like(graph_emb_1)
        
        # 3-GNN
        three_x, three_edges, three_batch = self._build_3sets(h, edge_index, batch)
        if three_x is not None:
            h3 = three_x
            for layer in self.gnn3_layers:
                h3 = layer(h3, three_edges)
            graph_emb_3 = global_add_pool(h3, three_batch)
        else:
            graph_emb_3 = torch.zeros_like(graph_emb_1)
        
        combined = torch.cat([graph_emb_1, graph_emb_2, graph_emb_3], dim=1)
        return self.classifier(combined)


# =============================================================================
# Model Factory
# =============================================================================

def get_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create k-GNN models.
    
    Args:
        model_name: One of '1gnn', '12gnn', '123gnn'
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model
    """
    models = {
        '1gnn': lambda: OneGNN(input_dim, hidden_dim, output_dim, **kwargs),
        '12gnn': lambda: Hierarchical12GNN(input_dim, hidden_dim, output_dim, **kwargs),
        '123gnn': lambda: Hierarchical123GNN(input_dim, hidden_dim, output_dim, **kwargs),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name]()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
