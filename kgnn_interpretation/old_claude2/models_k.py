"""
models_k.py
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
from itertools import combinations


# =============================================================================
# Basic Layer Implementations
# =============================================================================

class OneGNNLayer(MessagePassing):
    """
    1-GNN layer implementing:
    f^(t)(v) = σ( f^(t-1)(v) · W1 + Σ_{u ∈ N(v)} f^(t-1)(u) · W2 )
    """
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index):
        self_part = self.W1(x)
        neighbor_part = self.propagate(edge_index, x=x)
        return self.activation(self_part + neighbor_part)
    
    def message(self, x_j):
        return self.W2(x_j)


class KSetLayer(nn.Module):
    """
    Generic k-set layer for 2-GNN and 3-GNN.
    Performs message passing on k-sets using manually constructed neighborhoods.
    
    f^(t)(s) = σ( f^(t-1)(s) · W1 + Σ_{t ∈ N_L(s)} f^(t-1)(t) · W2 )
    
    where N_L(s) is the local neighborhood of k-set s.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index):
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
# Standalone Models
# =============================================================================

class OneGNN(nn.Module):
    """
    Standalone 1-GNN model.
    Standard message passing GNN operating on nodes.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(OneGNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(OneGNNLayer(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        
        graph_emb = global_add_pool(h, batch)
        return self.classifier(graph_emb)


class TwoGNN(nn.Module):
    """
    Standalone 2-GNN model.
    Message passing on 2-sets (pairs of nodes).
    
    Initial features are based on the isomorphism type of the induced subgraph,
    which for pairs is simply whether an edge exists between the two nodes.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial embedding: node features concatenated + edge indicator
        # For 2-sets: [feat_i || feat_j || edge_exists] 
        two_set_input_dim = 2 * input_dim + 1
        
        self.layers = nn.ModuleList()
        self.layers.append(KSetLayer(two_set_input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(KSetLayer(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def build_2sets(self, x, edge_index, batch):
        """
        Build 2-sets (pairs) and their local neighborhood connectivity.
        
        For 2-sets {u, v} and {v, w}, they are connected in the local neighborhood
        if and only if (u, w) is an edge in the original graph.
        """
        device = x.device
        
        all_feats = []
        all_batch = []
        all_edges_src = []
        all_edges_dst = []
        offset = 0
        
        for g in batch.unique():
            mask = (batch == g)
            node_indices = mask.nonzero(as_tuple=True)[0]
            n = node_indices.size(0)
            
            if n < 2:
                continue
            
            # Create pairs
            local_indices = torch.arange(n, device=device)
            pairs = torch.combinations(local_indices, r=2)  # [num_pairs, 2]
            
            # Get global indices
            global_u = node_indices[pairs[:, 0]]
            global_v = node_indices[pairs[:, 1]]
            
            # Get features
            feat_u = x[global_u]
            feat_v = x[global_v]
            
            # Build adjacency for this subgraph
            adj = self._build_local_adj(edge_index, node_indices, n, device)
            
            # Isomorphism type: edge exists or not
            iso_type = adj[pairs[:, 0], pairs[:, 1]].unsqueeze(1)
            
            # Concatenate features
            pair_feats = torch.cat([feat_u, feat_v, iso_type], dim=1)
            all_feats.append(pair_feats)
            all_batch.append(torch.full((pairs.size(0),), g.item(), device=device, dtype=torch.long))
            
            # Build local neighborhood edges
            pair_edges = self._build_2set_edges(pairs, adj, device)
            if pair_edges.size(1) > 0:
                all_edges_src.append(pair_edges[0] + offset)
                all_edges_dst.append(pair_edges[1] + offset)
            
            offset += pairs.size(0)
        
        if not all_feats:
            return None, None, None
        
        final_x = torch.cat(all_feats, dim=0)
        final_batch = torch.cat(all_batch, dim=0)
        
        if all_edges_src:
            final_edges = torch.stack([torch.cat(all_edges_src), torch.cat(all_edges_dst)])
        else:
            final_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return final_x, final_edges, final_batch
    
    def _build_local_adj(self, edge_index, node_indices, n, device):
        """Build local adjacency matrix for a subgraph."""
        adj = torch.zeros((n, n), device=device, dtype=torch.float)
        
        # Create mapping from global to local indices
        global_to_local = torch.full((edge_index.max().item() + 1,), -1, device=device, dtype=torch.long)
        global_to_local[node_indices] = torch.arange(n, device=device)
        
        # Filter edges within this subgraph
        mask = (global_to_local[edge_index[0]] >= 0) & (global_to_local[edge_index[1]] >= 0)
        local_src = global_to_local[edge_index[0, mask]]
        local_dst = global_to_local[edge_index[1, mask]]
        
        adj[local_src, local_dst] = 1.0
        adj[local_dst, local_src] = 1.0
        
        return adj
    
    def _build_2set_edges(self, pairs, adj, device):
        """
        Build edges between 2-sets based on local neighborhood definition.
        {u, v} ~ {v, w} if (u, w) ∈ E
        """
        num_pairs = pairs.size(0)
        if num_pairs == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Create pair lookup
        pair_to_idx = {}
        for idx, (u, v) in enumerate(pairs.tolist()):
            pair_to_idx[tuple(sorted((u, v)))] = idx
        
        src_list = []
        dst_list = []
        
        for i, (u, v) in enumerate(pairs.tolist()):
            # For each neighbor w of v (where w != u)
            neighbors_v = (adj[v] == 1).nonzero(as_tuple=True)[0]
            for w in neighbors_v.tolist():
                if w == u:
                    continue
                # Check if (u, w) is an edge
                if adj[u, w] == 1:
                    target = tuple(sorted((v, w)))
                    if target in pair_to_idx:
                        src_list.append(i)
                        dst_list.append(pair_to_idx[target])
            
            # For each neighbor w of u (where w != v)
            neighbors_u = (adj[u] == 1).nonzero(as_tuple=True)[0]
            for w in neighbors_u.tolist():
                if w == v:
                    continue
                if adj[v, w] == 1:
                    target = tuple(sorted((u, w)))
                    if target in pair_to_idx:
                        src_list.append(i)
                        dst_list.append(pair_to_idx[target])
        
        if src_list:
            return torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    def forward(self, x, edge_index, batch):
        # Build 2-sets from input
        two_x, two_edges, two_batch = self.build_2sets(x, edge_index, batch)
        
        if two_x is None:
            # Fallback for very small graphs
            return torch.zeros((batch.max().item() + 1, self.classifier[-1].out_features), device=x.device)
        
        h = two_x
        for layer in self.layers:
            h = layer(h, two_edges)
        
        graph_emb = global_add_pool(h, two_batch)
        return self.classifier(graph_emb)


class ThreeGNN(nn.Module):
    """
    Standalone 3-GNN model.
    Message passing on 3-sets (triplets of nodes).
    
    Initial features encode the isomorphism type of the induced subgraph
    on 3 nodes (no edges, 1 edge, 2 edges, or triangle).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial embedding: concatenated node features + isomorphism type (4 possibilities)
        # Iso types for 3 nodes: 0 edges, 1 edge, 2 edges (path), 3 edges (triangle)
        three_set_input_dim = 3 * input_dim + 4  # 4-dim one-hot for iso type
        
        self.layers = nn.ModuleList()
        self.layers.append(KSetLayer(three_set_input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(KSetLayer(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def build_3sets(self, x, edge_index, batch):
        """
        Build 3-sets (triplets) and their local neighborhood connectivity.
        
        For 3-sets {u, v, w} and {u, v, z}, they are connected if (w, z) ∈ E.
        """
        device = x.device
        
        all_feats = []
        all_batch = []
        all_edges_src = []
        all_edges_dst = []
        offset = 0
        
        for g in batch.unique():
            mask = (batch == g)
            node_indices = mask.nonzero(as_tuple=True)[0]
            n = node_indices.size(0)
            
            if n < 3:
                continue
            
            # Create triplets
            local_indices = torch.arange(n, device=device)
            triplets = torch.combinations(local_indices, r=3)  # [num_triplets, 3]
            
            # Get global indices
            global_a = node_indices[triplets[:, 0]]
            global_b = node_indices[triplets[:, 1]]
            global_c = node_indices[triplets[:, 2]]
            
            # Get features
            feat_a = x[global_a]
            feat_b = x[global_b]
            feat_c = x[global_c]
            
            # Build adjacency
            adj = self._build_local_adj(edge_index, node_indices, n, device)
            
            # Compute isomorphism type for each triplet
            iso_type = self._compute_3set_iso_type(triplets, adj, device)
            
            # Concatenate features
            triplet_feats = torch.cat([feat_a, feat_b, feat_c, iso_type], dim=1)
            all_feats.append(triplet_feats)
            all_batch.append(torch.full((triplets.size(0),), g.item(), device=device, dtype=torch.long))
            
            # Build local neighborhood edges
            triplet_edges = self._build_3set_edges(triplets, adj, device)
            if triplet_edges.size(1) > 0:
                all_edges_src.append(triplet_edges[0] + offset)
                all_edges_dst.append(triplet_edges[1] + offset)
            
            offset += triplets.size(0)
        
        if not all_feats:
            return None, None, None
        
        final_x = torch.cat(all_feats, dim=0)
        final_batch = torch.cat(all_batch, dim=0)
        
        if all_edges_src:
            final_edges = torch.stack([torch.cat(all_edges_src), torch.cat(all_edges_dst)])
        else:
            final_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return final_x, final_edges, final_batch
    
    def _build_local_adj(self, edge_index, node_indices, n, device):
        """Build local adjacency matrix for a subgraph."""
        adj = torch.zeros((n, n), device=device, dtype=torch.float)
        
        global_to_local = torch.full((edge_index.max().item() + 1,), -1, device=device, dtype=torch.long)
        global_to_local[node_indices] = torch.arange(n, device=device)
        
        mask = (global_to_local[edge_index[0]] >= 0) & (global_to_local[edge_index[1]] >= 0)
        local_src = global_to_local[edge_index[0, mask]]
        local_dst = global_to_local[edge_index[1, mask]]
        
        adj[local_src, local_dst] = 1.0
        adj[local_dst, local_src] = 1.0
        
        return adj
    
    def _compute_3set_iso_type(self, triplets, adj, device):
        """
        Compute isomorphism type for 3-sets.
        Types: 0 edges, 1 edge, 2 edges (path), 3 edges (triangle)
        Returns one-hot encoding of size 4.
        """
        a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        
        # Count edges in each triplet
        edge_ab = adj[a, b]
        edge_bc = adj[b, c]
        edge_ac = adj[a, c]
        edge_count = (edge_ab + edge_bc + edge_ac).long()
        
        # One-hot encode (0, 1, 2, or 3 edges)
        iso_type = torch.zeros((triplets.size(0), 4), device=device)
        iso_type.scatter_(1, edge_count.unsqueeze(1), 1.0)
        
        return iso_type
    
    def _build_3set_edges(self, triplets, adj, device):
        """
        Build edges between 3-sets based on local neighborhood definition.
        {a, b, c} ~ {a, b, d} if (c, d) ∈ E
        """
        num_triplets = triplets.size(0)
        if num_triplets == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Create triplet lookup
        triplet_to_idx = {}
        for idx, (a, b, c) in enumerate(triplets.tolist()):
            triplet_to_idx[tuple(sorted((a, b, c)))] = idx
        
        src_list = []
        dst_list = []
        
        n = adj.size(0)
        
        for i, (a, b, c) in enumerate(triplets.tolist()):
            nodes = [a, b, c]
            
            # For each pair of nodes in the triplet, try replacing the third
            for j, (n1, n2) in enumerate([(a, b), (a, c), (b, c)]):
                # n3 is the node to be replaced
                n3 = nodes[2 - j] if j == 0 else (nodes[0] if j == 2 else nodes[1])
                if j == 0:
                    n3 = c
                elif j == 1:
                    n3 = b
                else:
                    n3 = a
                
                # Find neighbors of n3 that could form a new triplet
                neighbors = (adj[n3] == 1).nonzero(as_tuple=True)[0]
                for d in neighbors.tolist():
                    if d in nodes:
                        continue
                    
                    target = tuple(sorted((n1, n2, d)))
                    if target in triplet_to_idx:
                        src_list.append(i)
                        dst_list.append(triplet_to_idx[target])
        
        if src_list:
            return torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    def forward(self, x, edge_index, batch):
        three_x, three_edges, three_batch = self.build_3sets(x, edge_index, batch)
        
        if three_x is None:
            return torch.zeros((batch.max().item() + 1, self.classifier[-1].out_features), device=x.device)
        
        h = three_x
        for layer in self.layers:
            h = layer(h, three_edges)
        
        graph_emb = global_add_pool(h, three_batch)
        return self.classifier(graph_emb)


# =============================================================================
# Hierarchical Models
# =============================================================================

class Hierarchical12GNN(nn.Module):
    """
    Hierarchical 1-2-GNN.
    
    First runs 1-GNN to get node embeddings, then uses those embeddings
    to initialize 2-GNN features. Final representation concatenates
    pooled outputs from both levels.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers_1=3, num_layers_2=2):
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
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def build_2sets_from_embeddings(self, h, edge_index, batch):
        """Build 2-sets using 1-GNN embeddings as features."""
        device = h.device
        
        all_feats = []
        all_batch = []
        all_edges_src = []
        all_edges_dst = []
        offset = 0
        
        for g in batch.unique():
            mask = (batch == g)
            node_indices = mask.nonzero(as_tuple=True)[0]
            n = node_indices.size(0)
            
            if n < 2:
                continue
            
            local_indices = torch.arange(n, device=device)
            pairs = torch.combinations(local_indices, r=2)
            
            global_u = node_indices[pairs[:, 0]]
            global_v = node_indices[pairs[:, 1]]
            
            feat_u = h[global_u]
            feat_v = h[global_v]
            
            adj = self._build_local_adj(edge_index, node_indices, n, device)
            iso_type = adj[pairs[:, 0], pairs[:, 1]].unsqueeze(1)
            
            pair_feats = torch.cat([feat_u, feat_v, iso_type], dim=1)
            all_feats.append(pair_feats)
            all_batch.append(torch.full((pairs.size(0),), g.item(), device=device, dtype=torch.long))
            
            pair_edges = self._build_2set_edges(pairs, adj, device)
            if pair_edges.size(1) > 0:
                all_edges_src.append(pair_edges[0] + offset)
                all_edges_dst.append(pair_edges[1] + offset)
            
            offset += pairs.size(0)
        
        if not all_feats:
            return None, None, None
        
        final_x = torch.cat(all_feats, dim=0)
        final_batch = torch.cat(all_batch, dim=0)
        
        if all_edges_src:
            final_edges = torch.stack([torch.cat(all_edges_src), torch.cat(all_edges_dst)])
        else:
            final_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return final_x, final_edges, final_batch
    
    def _build_local_adj(self, edge_index, node_indices, n, device):
        adj = torch.zeros((n, n), device=device, dtype=torch.float)
        global_to_local = torch.full((edge_index.max().item() + 1,), -1, device=device, dtype=torch.long)
        global_to_local[node_indices] = torch.arange(n, device=device)
        mask = (global_to_local[edge_index[0]] >= 0) & (global_to_local[edge_index[1]] >= 0)
        local_src = global_to_local[edge_index[0, mask]]
        local_dst = global_to_local[edge_index[1, mask]]
        adj[local_src, local_dst] = 1.0
        adj[local_dst, local_src] = 1.0
        return adj
    
    def _build_2set_edges(self, pairs, adj, device):
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
    
    def forward(self, x, edge_index, batch):
        # 1-GNN
        h = x
        for layer in self.gnn1_layers:
            h = layer(h, edge_index)
        graph_emb_1 = global_add_pool(h, batch)
        
        # 2-GNN using 1-GNN embeddings
        two_x, two_edges, two_batch = self.build_2sets_from_embeddings(h, edge_index, batch)
        
        if two_x is not None:
            h2 = two_x
            for layer in self.gnn2_layers:
                h2 = layer(h2, two_edges)
            graph_emb_2 = global_add_pool(h2, two_batch)
        else:
            graph_emb_2 = torch.zeros_like(graph_emb_1)
        
        combined = torch.cat([graph_emb_1, graph_emb_2], dim=1)
        return self.classifier(combined)


class Hierarchical13GNN(nn.Module):
    """
    Hierarchical 1-3-GNN.
    
    First runs 1-GNN, then uses embeddings to initialize 3-GNN.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers_1=3, num_layers_3=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1-GNN layers
        self.gnn1_layers = nn.ModuleList()
        self.gnn1_layers.append(OneGNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers_1 - 1):
            self.gnn1_layers.append(OneGNNLayer(hidden_dim, hidden_dim))
        
        # 3-GNN layers
        three_set_input_dim = 3 * hidden_dim + 4  # iso type has 4 classes
        self.gnn3_layers = nn.ModuleList()
        self.gnn3_layers.append(KSetLayer(three_set_input_dim, hidden_dim))
        for _ in range(num_layers_3 - 1):
            self.gnn3_layers.append(KSetLayer(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def build_3sets_from_embeddings(self, h, edge_index, batch):
        """Build 3-sets using 1-GNN embeddings."""
        device = h.device
        
        all_feats = []
        all_batch = []
        all_edges_src = []
        all_edges_dst = []
        offset = 0
        
        for g in batch.unique():
            mask = (batch == g)
            node_indices = mask.nonzero(as_tuple=True)[0]
            n = node_indices.size(0)
            
            if n < 3:
                continue
            
            local_indices = torch.arange(n, device=device)
            triplets = torch.combinations(local_indices, r=3)
            
            global_a = node_indices[triplets[:, 0]]
            global_b = node_indices[triplets[:, 1]]
            global_c = node_indices[triplets[:, 2]]
            
            feat_a = h[global_a]
            feat_b = h[global_b]
            feat_c = h[global_c]
            
            adj = self._build_local_adj(edge_index, node_indices, n, device)
            iso_type = self._compute_3set_iso_type(triplets, adj, device)
            
            triplet_feats = torch.cat([feat_a, feat_b, feat_c, iso_type], dim=1)
            all_feats.append(triplet_feats)
            all_batch.append(torch.full((triplets.size(0),), g.item(), device=device, dtype=torch.long))
            
            triplet_edges = self._build_3set_edges(triplets, adj, device)
            if triplet_edges.size(1) > 0:
                all_edges_src.append(triplet_edges[0] + offset)
                all_edges_dst.append(triplet_edges[1] + offset)
            
            offset += triplets.size(0)
        
        if not all_feats:
            return None, None, None
        
        final_x = torch.cat(all_feats, dim=0)
        final_batch = torch.cat(all_batch, dim=0)
        
        if all_edges_src:
            final_edges = torch.stack([torch.cat(all_edges_src), torch.cat(all_edges_dst)])
        else:
            final_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return final_x, final_edges, final_batch
    
    def _build_local_adj(self, edge_index, node_indices, n, device):
        adj = torch.zeros((n, n), device=device, dtype=torch.float)
        global_to_local = torch.full((edge_index.max().item() + 1,), -1, device=device, dtype=torch.long)
        global_to_local[node_indices] = torch.arange(n, device=device)
        mask = (global_to_local[edge_index[0]] >= 0) & (global_to_local[edge_index[1]] >= 0)
        local_src = global_to_local[edge_index[0, mask]]
        local_dst = global_to_local[edge_index[1, mask]]
        adj[local_src, local_dst] = 1.0
        adj[local_dst, local_src] = 1.0
        return adj
    
    def _compute_3set_iso_type(self, triplets, adj, device):
        a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        edge_count = (adj[a, b] + adj[b, c] + adj[a, c]).long()
        iso_type = torch.zeros((triplets.size(0), 4), device=device)
        iso_type.scatter_(1, edge_count.unsqueeze(1), 1.0)
        return iso_type
    
    def _build_3set_edges(self, triplets, adj, device):
        num_triplets = triplets.size(0)
        if num_triplets == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        triplet_to_idx = {}
        for idx, (a, b, c) in enumerate(triplets.tolist()):
            triplet_to_idx[tuple(sorted((a, b, c)))] = idx
        
        src_list, dst_list = [], []
        n = adj.size(0)
        
        for i, (a, b, c) in enumerate(triplets.tolist()):
            # Replace c with d where (c, d) is edge
            for d in (adj[c] == 1).nonzero(as_tuple=True)[0].tolist():
                if d not in [a, b, c]:
                    target = tuple(sorted((a, b, d)))
                    if target in triplet_to_idx:
                        src_list.append(i)
                        dst_list.append(triplet_to_idx[target])
            
            # Replace b with d where (b, d) is edge
            for d in (adj[b] == 1).nonzero(as_tuple=True)[0].tolist():
                if d not in [a, b, c]:
                    target = tuple(sorted((a, c, d)))
                    if target in triplet_to_idx:
                        src_list.append(i)
                        dst_list.append(triplet_to_idx[target])
            
            # Replace a with d where (a, d) is edge
            for d in (adj[a] == 1).nonzero(as_tuple=True)[0].tolist():
                if d not in [a, b, c]:
                    target = tuple(sorted((b, c, d)))
                    if target in triplet_to_idx:
                        src_list.append(i)
                        dst_list.append(triplet_to_idx[target])
        
        if src_list:
            return torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    def forward(self, x, edge_index, batch):
        # 1-GNN
        h = x
        for layer in self.gnn1_layers:
            h = layer(h, edge_index)
        graph_emb_1 = global_add_pool(h, batch)
        
        # 3-GNN
        three_x, three_edges, three_batch = self.build_3sets_from_embeddings(h, edge_index, batch)
        
        if three_x is not None:
            h3 = three_x
            for layer in self.gnn3_layers:
                h3 = layer(h3, three_edges)
            graph_emb_3 = global_add_pool(h3, three_batch)
        else:
            graph_emb_3 = torch.zeros_like(graph_emb_1)
        
        combined = torch.cat([graph_emb_1, graph_emb_3], dim=1)
        return self.classifier(combined)


class Hierarchical123GNN(nn.Module):
    """
    Hierarchical 1-2-3-GNN.
    
    Runs 1-GNN, then uses embeddings for both 2-GNN and 3-GNN.
    Final representation concatenates all three pooled outputs.
    
    This is the most expressive model, capturing structure at all three scales.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers_1=3, num_layers_2=2, num_layers_3=2):
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
        three_set_input_dim = 3 * hidden_dim + 4
        self.gnn3_layers = nn.ModuleList()
        self.gnn3_layers.append(KSetLayer(three_set_input_dim, hidden_dim))
        for _ in range(num_layers_3 - 1):
            self.gnn3_layers.append(KSetLayer(hidden_dim, hidden_dim))
        
        # Classifier (concatenates all three)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch):
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
    
    # Reuse building methods (simplified versions)
    def _build_2sets(self, h, edge_index, batch):
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
            adj = self._build_adj(edge_index, nodes, n, device)
            
            feat_u = h[nodes[pairs[:, 0]]]
            feat_v = h[nodes[pairs[:, 1]]]
            iso = adj[pairs[:, 0], pairs[:, 1]].unsqueeze(1)
            
            all_feats.append(torch.cat([feat_u, feat_v, iso], dim=1))
            all_batch.append(torch.full((pairs.size(0),), g.item(), device=device, dtype=torch.long))
            
            edges = self._2set_edges(pairs, adj, device)
            if edges.size(1) > 0:
                all_src.append(edges[0] + offset)
                all_dst.append(edges[1] + offset)
            offset += pairs.size(0)
        
        if not all_feats:
            return None, None, None
        return (torch.cat(all_feats), 
                torch.stack([torch.cat(all_src), torch.cat(all_dst)]) if all_src else torch.empty((2,0), dtype=torch.long, device=device),
                torch.cat(all_batch))
    
    def _build_3sets(self, h, edge_index, batch):
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
            adj = self._build_adj(edge_index, nodes, n, device)
            
            fa = h[nodes[trips[:, 0]]]
            fb = h[nodes[trips[:, 1]]]
            fc = h[nodes[trips[:, 2]]]
            
            # Iso type
            ec = (adj[trips[:,0], trips[:,1]] + adj[trips[:,1], trips[:,2]] + adj[trips[:,0], trips[:,2]]).long()
            iso = torch.zeros((trips.size(0), 4), device=device)
            iso.scatter_(1, ec.unsqueeze(1), 1.0)
            
            all_feats.append(torch.cat([fa, fb, fc, iso], dim=1))
            all_batch.append(torch.full((trips.size(0),), g.item(), device=device, dtype=torch.long))
            
            edges = self._3set_edges(trips, adj, device)
            if edges.size(1) > 0:
                all_src.append(edges[0] + offset)
                all_dst.append(edges[1] + offset)
            offset += trips.size(0)
        
        if not all_feats:
            return None, None, None
        return (torch.cat(all_feats),
                torch.stack([torch.cat(all_src), torch.cat(all_dst)]) if all_src else torch.empty((2,0), dtype=torch.long, device=device),
                torch.cat(all_batch))
    
    def _build_adj(self, edge_index, nodes, n, device):
        adj = torch.zeros((n, n), device=device)
        g2l = torch.full((edge_index.max().item()+1,), -1, device=device, dtype=torch.long)
        g2l[nodes] = torch.arange(n, device=device)
        m = (g2l[edge_index[0]] >= 0) & (g2l[edge_index[1]] >= 0)
        adj[g2l[edge_index[0,m]], g2l[edge_index[1,m]]] = 1
        adj[g2l[edge_index[1,m]], g2l[edge_index[0,m]]] = 1
        return adj
    
    def _2set_edges(self, pairs, adj, device):
        p2i = {tuple(sorted((u,v))): i for i,(u,v) in enumerate(pairs.tolist())}
        src, dst = [], []
        for i,(u,v) in enumerate(pairs.tolist()):
            for w in (adj[v]==1).nonzero(as_tuple=True)[0].tolist():
                if w != u and adj[u,w]==1:
                    t = tuple(sorted((v,w)))
                    if t in p2i: src.append(i); dst.append(p2i[t])
            for w in (adj[u]==1).nonzero(as_tuple=True)[0].tolist():
                if w != v and adj[v,w]==1:
                    t = tuple(sorted((u,w)))
                    if t in p2i: src.append(i); dst.append(p2i[t])
        return torch.tensor([src,dst], dtype=torch.long, device=device) if src else torch.empty((2,0), dtype=torch.long, device=device)
    
    def _3set_edges(self, trips, adj, device):
        t2i = {tuple(sorted((a,b,c))): i for i,(a,b,c) in enumerate(trips.tolist())}
        src, dst = [], []
        for i,(a,b,c) in enumerate(trips.tolist()):
            for d in (adj[c]==1).nonzero(as_tuple=True)[0].tolist():
                if d not in [a,b,c]:
                    t = tuple(sorted((a,b,d)))
                    if t in t2i: src.append(i); dst.append(t2i[t])
            for d in (adj[b]==1).nonzero(as_tuple=True)[0].tolist():
                if d not in [a,b,c]:
                    t = tuple(sorted((a,c,d)))
                    if t in t2i: src.append(i); dst.append(t2i[t])
            for d in (adj[a]==1).nonzero(as_tuple=True)[0].tolist():
                if d not in [a,b,c]:
                    t = tuple(sorted((b,c,d)))
                    if t in t2i: src.append(i); dst.append(t2i[t])
        return torch.tensor([src,dst], dtype=torch.long, device=device) if src else torch.empty((2,0), dtype=torch.long, device=device)


# =============================================================================
# Model Factory
# =============================================================================

def get_model(model_name, input_dim, hidden_dim, output_dim):
    """
    Factory function to create k-GNN models.
    
    Args:
        model_name: One of '1gnn', '2gnn', '3gnn', '12gnn', '13gnn', '123gnn'
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Number of output classes
    
    Returns:
        Initialized model
    """
    models = {
        '1gnn': lambda: OneGNN(input_dim, hidden_dim, output_dim, num_layers=3),
        '2gnn': lambda: TwoGNN(input_dim, hidden_dim, output_dim, num_layers=2),
        '3gnn': lambda: ThreeGNN(input_dim, hidden_dim, output_dim, num_layers=2),
        '12gnn': lambda: Hierarchical12GNN(input_dim, hidden_dim, output_dim),
        '13gnn': lambda: Hierarchical13GNN(input_dim, hidden_dim, output_dim),
        '123gnn': lambda: Hierarchical123GNN(input_dim, hidden_dim, output_dim),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name]()