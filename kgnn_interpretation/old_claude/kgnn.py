"""
Hierarchical k-GNN Implementation based on:
- "Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks" (Morris et al., AAAI 2019)
- "How Powerful Are Graph Neural Networks?" (Xu et al., ICLR 2019)

This implementation provides k-dimensional GNNs that operate on k-element subsets
of nodes, capturing higher-order graph structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_adj
from itertools import combinations
from typing import Optional, Tuple, List, Dict
import math


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network convolution layer.
    
    From "How Powerful Are Graph Neural Networks?" (Xu et al., ICLR 2019):
    h_v^(k) = MLP^(k)((1 + eps^(k)) * h_v^(k-1) + sum_{u in N(v)} h_u^(k-1))
    
    This is the most expressive aggregation under the 1-WL framework.
    """
    
    def __init__(self, in_channels: int, out_channels: int, eps: float = 0.0, 
                 train_eps: bool = True, mlp_layers: int = 2):
        super().__init__(aggr='add')
        
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        # Build MLP
        layers = []
        layers.append(nn.Linear(in_channels, out_channels))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        
        for _ in range(mlp_layers - 1):
            layers.append(nn.Linear(out_channels, out_channels))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Aggregate neighbors
        out = self.propagate(edge_index, x=x)
        # Add self-loop with learnable epsilon
        out = (1 + self.eps) * x + out
        return self.mlp(out)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class OneGNN(nn.Module):
    """
    Standard 1-dimensional GNN (operates on nodes).
    Uses GIN convolution for maximum expressiveness within 1-WL bounds.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, dropout: float = 0.5, eps: float = 0.0,
                 train_eps: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GINConv(in_channels, hidden_channels, eps, train_eps))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(hidden_channels, hidden_channels, eps, train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        self.convs.append(GINConv(hidden_channels, out_channels, eps, train_eps))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        # Store intermediate representations for analysis
        self.layer_representations = []
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None,
                return_all_layers: bool = False) -> torch.Tensor:
        
        self.layer_representations = [x.clone()]
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            self.layer_representations.append(x.clone())
        
        if return_all_layers:
            return x, self.layer_representations
        return x
    
    def get_layer_embeddings(self) -> List[torch.Tensor]:
        """Return embeddings from all layers for interpretability analysis."""
        return self.layer_representations


class KSetNeighborhood:
    """
    Computes neighborhoods for k-element subsets as defined in Morris et al.
    
    For a k-set s = {s1, ..., sk}, the neighborhood N(s) consists of all k-sets t
    where |s ∩ t| = k-1 (they share k-1 elements).
    
    Local neighborhood N_L(s): subsets where the differing elements are connected by an edge
    Global neighborhood N_G(s): N(s) \ N_L(s)
    """
    
    @staticmethod
    def compute_k_sets(num_nodes: int, k: int) -> torch.Tensor:
        """Generate all k-element subsets of nodes."""
        if k > num_nodes:
            return torch.tensor([], dtype=torch.long)
        
        k_sets = list(combinations(range(num_nodes), k))
        return torch.tensor(k_sets, dtype=torch.long)
    
    @staticmethod
    def compute_neighborhoods(k_sets: torch.Tensor, edge_index: torch.Tensor,
                             num_nodes: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute local and global neighborhoods for all k-sets.
        
        Returns:
            local_edge_index: edges between k-sets in local neighborhood
            global_edge_index: edges between k-sets in global neighborhood
        """
        if len(k_sets) == 0:
            return torch.tensor([[], []], dtype=torch.long), torch.tensor([[], []], dtype=torch.long)
        
        # Create adjacency set for fast lookup
        adj_set = set()
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            adj_set.add((u, v))
            adj_set.add((v, u))
        
        # Create k-set to index mapping
        k_set_to_idx = {tuple(sorted(k_sets[i].tolist())): i for i in range(len(k_sets))}
        
        local_edges = [[], []]
        global_edges = [[], []]
        
        for idx, k_set in enumerate(k_sets):
            k_set_list = k_set.tolist()
            k_set_sorted = tuple(sorted(k_set_list))
            
            # For each position, try replacing with each node
            for pos in range(k):
                old_node = k_set_list[pos]
                for new_node in range(num_nodes):
                    if new_node in k_set_list:
                        continue
                    
                    # Create new k-set
                    new_k_set = k_set_list.copy()
                    new_k_set[pos] = new_node
                    new_k_set_sorted = tuple(sorted(new_k_set))
                    
                    if new_k_set_sorted in k_set_to_idx:
                        neighbor_idx = k_set_to_idx[new_k_set_sorted]
                        
                        # Check if it's local (edge exists between differing nodes)
                        if (old_node, new_node) in adj_set:
                            local_edges[0].append(idx)
                            local_edges[1].append(neighbor_idx)
                        else:
                            global_edges[0].append(idx)
                            global_edges[1].append(neighbor_idx)
        
        local_edge_index = torch.tensor(local_edges, dtype=torch.long)
        global_edge_index = torch.tensor(global_edges, dtype=torch.long)
        
        return local_edge_index, global_edge_index


class KGNNConv(nn.Module):
    """
    k-dimensional GNN convolution layer.
    
    Operates on k-element subsets and performs message passing between
    subsets that differ by one element.
    
    f_k^(t)(s) = σ(f_k^(t-1)(s) · W1 + Σ_{u∈N_L(s)∪N_G(s)} f_k^(t-1)(u) · W2)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_global: bool = True, separate_weights: bool = True):
        super().__init__()
        
        self.use_global = use_global
        self.separate_weights = separate_weights
        
        self.W1 = nn.Linear(in_channels, out_channels, bias=False)
        self.W2_local = nn.Linear(in_channels, out_channels, bias=False)
        
        if use_global and separate_weights:
            self.W2_global = nn.Linear(in_channels, out_channels, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor, local_edge_index: torch.Tensor,
                global_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: k-set features [num_k_sets, in_channels]
            local_edge_index: local neighborhood edges
            global_edge_index: global neighborhood edges (optional)
        """
        # Self transformation
        out = self.W1(x)
        
        # Local neighborhood aggregation
        if local_edge_index.numel() > 0:
            local_agg = self._aggregate(x, local_edge_index, self.W2_local)
            out = out + local_agg
        
        # Global neighborhood aggregation
        if self.use_global and global_edge_index is not None and global_edge_index.numel() > 0:
            W_global = self.W2_global if self.separate_weights else self.W2_local
            global_agg = self._aggregate(x, global_edge_index, W_global)
            out = out + global_agg
        
        out = self.bn(out)
        out = F.relu(out)
        
        return out
    
    def _aggregate(self, x: torch.Tensor, edge_index: torch.Tensor, 
                   weight: nn.Linear) -> torch.Tensor:
        """Aggregate neighbor features."""
        row, col = edge_index
        # Transform neighbor features
        neighbor_features = weight(x[col])
        # Sum aggregation
        out = torch.zeros(x.size(0), weight.out_features, device=x.device)
        out.index_add_(0, row, neighbor_features)
        return out


class KGNN(nn.Module):
    """
    k-dimensional GNN that operates on k-element subsets.
    
    This captures higher-order graph structures that 1-GNNs cannot distinguish.
    """
    
    def __init__(self, k: int, in_channels: int, hidden_channels: int, 
                 out_channels: int, num_layers: int = 2,
                 use_global: bool = False, dropout: float = 0.5):
        super().__init__()
        
        self.k = k
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_global = use_global
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(KGNNConv(in_channels, hidden_channels, use_global))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(KGNNConv(hidden_channels, hidden_channels, use_global))
        
        # Last layer
        if num_layers > 1:
            self.convs.append(KGNNConv(hidden_channels, out_channels, use_global))
        
        self.layer_representations = []
        
    def forward(self, x: torch.Tensor, local_edge_index: torch.Tensor,
                global_edge_index: Optional[torch.Tensor] = None,
                return_all_layers: bool = False) -> torch.Tensor:
        """
        Args:
            x: k-set features [num_k_sets, in_channels]
            local_edge_index: local neighborhood edges
            global_edge_index: global neighborhood edges
        """
        self.layer_representations = [x.clone()]
        
        for i, conv in enumerate(self.convs):
            x = conv(x, local_edge_index, global_edge_index)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            self.layer_representations.append(x.clone())
        
        if return_all_layers:
            return x, self.layer_representations
        return x
    
    def get_layer_embeddings(self) -> List[torch.Tensor]:
        return self.layer_representations


class HierarchicalKGNN(nn.Module):
    """
    Hierarchical k-GNN (1-2-3-GNN) from Morris et al.
    
    Key insight: Initialize k-GNN features using learned features from (k-1)-GNN.
    This creates a hierarchical representation that captures graph structures
    at multiple granularities.
    
    f_k^(0)(s) = σ([f^iso(s), Σ_{u⊂s} f_{k-1}^(T_{k-1})(u)] · W_{k-1})
    
    where f^iso(s) is the isomorphism type encoding of the induced subgraph G[s].
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 64,
                 num_classes: int = 2, max_k: int = 3,
                 num_layers_1gnn: int = 3, num_layers_kgnn: int = 2,
                 use_global: bool = False, dropout: float = 0.5):
        super().__init__()
        
        self.max_k = max_k
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # 1-GNN (operates on nodes)
        self.gnn_1 = OneGNN(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers_1gnn,
            dropout=dropout
        )
        
        # Higher-order k-GNNs
        self.k_gnns = nn.ModuleDict()
        self.k_init_mlps = nn.ModuleDict()
        
        for k in range(2, max_k + 1):
            # MLP to combine isomorphism type with lower-order features
            # Input: isomorphism type features + pooled (k-1)-features
            iso_dim = self._get_iso_type_dim(k)
            self.k_init_mlps[str(k)] = nn.Sequential(
                nn.Linear(iso_dim + hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            )
            
            self.k_gnns[str(k)] = KGNN(
                k=k,
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers_kgnn,
                use_global=use_global,
                dropout=dropout
            )
        
        # Final MLP classifier
        # Concatenate features from all k levels
        total_features = hidden_channels * max_k
        self.classifier = nn.Sequential(
            nn.Linear(total_features, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # Store layer-wise representations for interpretability
        self.all_layer_representations = {}
        
    def _get_iso_type_dim(self, k: int) -> int:
        """
        Get dimension for isomorphism type encoding.
        For simplicity, we use a fixed encoding based on edge counts
        and degree sequences within the k-subset.
        """
        # Number of possible edges in k-subset: k*(k-1)/2
        # Plus degree features for each position
        return k * (k - 1) // 2 + k
    
    def _compute_iso_type_features(self, k_sets: torch.Tensor, 
                                   edge_index: torch.Tensor,
                                   num_nodes: int) -> torch.Tensor:
        """
        Compute isomorphism type features for k-sets.
        
        Two k-tuples get the same feature if the induced subgraphs are isomorphic.
        We encode this using edge presence and sorted degree sequence.
        """
        if len(k_sets) == 0:
            return torch.tensor([], dtype=torch.float32)
        
        k = k_sets.size(1)
        num_k_sets = k_sets.size(0)
        
        # Create adjacency matrix for fast lookup
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = True
        
        # Features: edge presence (upper triangular) + sorted degrees
        num_edges = k * (k - 1) // 2
        features = torch.zeros(num_k_sets, num_edges + k, device=edge_index.device)
        
        for idx, k_set in enumerate(k_sets):
            nodes = k_set.tolist()
            
            # Edge presence features
            edge_idx = 0
            degrees = torch.zeros(k)
            for i in range(k):
                for j in range(i + 1, k):
                    if adj[nodes[i], nodes[j]]:
                        features[idx, edge_idx] = 1.0
                        degrees[i] += 1
                        degrees[j] += 1
                    edge_idx += 1
            
            # Sorted degree sequence
            sorted_degrees, _ = torch.sort(degrees)
            features[idx, num_edges:] = sorted_degrees
        
        return features
    
    def _pool_lower_order_features(self, k_sets: torch.Tensor,
                                   lower_features: torch.Tensor,
                                   lower_k_sets: torch.Tensor) -> torch.Tensor:
        """
        Pool features from (k-1)-sets that are subsets of each k-set.
        
        f_init(s) = Σ_{u⊂s, |u|=k-1} f_{k-1}(u)
        """
        if len(k_sets) == 0:
            return torch.tensor([], dtype=torch.float32)
        
        k = k_sets.size(1)
        num_k_sets = k_sets.size(0)
        feature_dim = lower_features.size(1)
        
        # Create mapping from (k-1)-set to index
        lower_k_set_to_idx = {
            tuple(sorted(lower_k_sets[i].tolist())): i 
            for i in range(len(lower_k_sets))
        }
        
        pooled_features = torch.zeros(num_k_sets, feature_dim, device=lower_features.device)
        
        for idx, k_set in enumerate(k_sets):
            k_set_list = k_set.tolist()
            # Get all (k-1)-subsets
            for subset in combinations(k_set_list, k - 1):
                subset_sorted = tuple(sorted(subset))
                if subset_sorted in lower_k_set_to_idx:
                    subset_idx = lower_k_set_to_idx[subset_sorted]
                    pooled_features[idx] += lower_features[subset_idx]
        
        return pooled_features
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor, return_layer_info: bool = False) -> torch.Tensor:
        """
        Forward pass through hierarchical k-GNN.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            return_layer_info: Whether to return layer-wise representations
        """
        self.all_layer_representations = {}
        
        # Get unique graphs in batch
        batch_size = batch.max().item() + 1
        
        all_graph_features = []
        
        for graph_idx in range(batch_size):
            # Get nodes for this graph
            node_mask = batch == graph_idx
            graph_x = x[node_mask]
            num_nodes = graph_x.size(0)
            
            # Get edges for this graph
            node_indices = torch.where(node_mask)[0]
            node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
            
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            graph_edge_index = edge_index[:, edge_mask]
            
            # Remap edge indices to local graph
            graph_edge_index_local = torch.zeros_like(graph_edge_index)
            for i in range(graph_edge_index.size(1)):
                graph_edge_index_local[0, i] = node_map[graph_edge_index[0, i].item()]
                graph_edge_index_local[1, i] = node_map[graph_edge_index[1, i].item()]
            
            # 1-GNN forward
            node_features, layer_reps_1 = self.gnn_1(
                graph_x, graph_edge_index_local, return_all_layers=True
            )
            
            if return_layer_info:
                if '1-gnn' not in self.all_layer_representations:
                    self.all_layer_representations['1-gnn'] = []
                self.all_layer_representations['1-gnn'].append(layer_reps_1)
            
            # Pool 1-GNN features for graph representation
            graph_rep_1 = node_features.mean(dim=0, keepdim=True)  # [1, hidden]
            
            k_features = {1: node_features}
            k_sets_dict = {1: torch.arange(num_nodes).unsqueeze(1)}
            graph_reps = [graph_rep_1]
            
            # Higher-order k-GNNs
            for k in range(2, self.max_k + 1):
                if num_nodes < k:
                    # Not enough nodes for k-sets
                    graph_reps.append(torch.zeros(1, self.hidden_channels, device=x.device))
                    continue
                
                # Compute k-sets
                k_sets = KSetNeighborhood.compute_k_sets(num_nodes, k)
                k_sets = k_sets.to(x.device)
                k_sets_dict[k] = k_sets
                
                if len(k_sets) == 0:
                    graph_reps.append(torch.zeros(1, self.hidden_channels, device=x.device))
                    continue
                
                # Compute neighborhoods
                local_edges, global_edges = KSetNeighborhood.compute_neighborhoods(
                    k_sets, graph_edge_index_local, num_nodes, k
                )
                local_edges = local_edges.to(x.device)
                global_edges = global_edges.to(x.device)
                
                # Compute initial features
                iso_features = self._compute_iso_type_features(
                    k_sets, graph_edge_index_local, num_nodes
                )
                
                pooled_features = self._pool_lower_order_features(
                    k_sets, k_features[k-1], k_sets_dict[k-1]
                )
                
                # Combine and transform
                combined = torch.cat([iso_features, pooled_features], dim=1)
                k_init_features = self.k_init_mlps[str(k)](combined)
                
                # k-GNN forward
                k_out, layer_reps_k = self.k_gnns[str(k)](
                    k_init_features, local_edges, global_edges, return_all_layers=True
                )
                
                if return_layer_info:
                    key = f'{k}-gnn'
                    if key not in self.all_layer_representations:
                        self.all_layer_representations[key] = []
                    self.all_layer_representations[key].append(layer_reps_k)
                
                k_features[k] = k_out
                
                # Pool k-set features for graph representation
                graph_rep_k = k_out.mean(dim=0, keepdim=True)
                graph_reps.append(graph_rep_k)
            
            # Concatenate all k-level representations
            graph_features = torch.cat(graph_reps, dim=1)
            all_graph_features.append(graph_features)
        
        # Stack all graph features
        batch_features = torch.cat(all_graph_features, dim=0)
        
        # Classification
        out = self.classifier(batch_features)
        
        if return_layer_info:
            return out, self.all_layer_representations
        return out
    
    def get_all_layer_representations(self) -> Dict[str, List]:
        """Return all stored layer representations for interpretability analysis."""
        return self.all_layer_representations


class Hierarchical123GNN(HierarchicalKGNN):
    """
    Convenience class for 1-2-3-GNN architecture from Morris et al.
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 64,
                 num_classes: int = 2, **kwargs):
        super().__init__(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            max_k=3,
            **kwargs
        )
