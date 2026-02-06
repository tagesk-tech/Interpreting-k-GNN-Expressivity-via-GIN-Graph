"""
model_wrapper.py
Wrapper to convert sparse k-GNN models to accept dense inputs from the generator.

Provides fully differentiable dense forward passes for hierarchical models
(1gnn, 12gnn, 123gnn), with gradient flow through both node features AND
adjacency matrices. This is critical for GIN-Graph training.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from models_kgnn import _sample_ksets, build_3set_edges


class DenseToSparseWrapper(nn.Module):
    """
    Wraps a sparse k-GNN model to accept dense inputs (x, adj) from the Generator.

    Uses fully differentiable dense forward passes that maintain gradient flow
    through the adjacency matrix. Supports hierarchical models: 1gnn, 12gnn, 123gnn.
    """

    def __init__(self, sparse_model: nn.Module, model_type: str = '1gnn'):
        super().__init__()
        self.sparse_model = sparse_model
        self.model_type = model_type

        # Freeze the pretrained model
        for param in self.sparse_model.parameters():
            param.requires_grad = False
        self.sparse_model.eval()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting dense inputs through the pretrained k-GNN.

        Args:
            x: Node features [batch, N, D]
            adj: Adjacency matrix [batch, N, N] (continuous values from generator)

        Returns:
            logits: Classification logits [batch, num_classes]
        """
        if self.model_type == '1gnn':
            return self._forward_dense_1gnn(x, adj)
        elif self.model_type == '12gnn':
            return self._forward_dense_12gnn(x, adj)
        elif self.model_type == '123gnn':
            return self._forward_dense_123gnn(x, adj)
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Use '1gnn', '12gnn', or '123gnn'."
            )

    # ==================== Dense Building Blocks ====================

    def _dense_1gnn(self, x, adj):
        """
        Dense 1-GNN message passing using pretrained OneGNNLayer weights.
        Fully differentiable through adj.

        Equivalent to: f^(t)(v) = sigma(f^(t-1)(v)*W1 + A*f^(t-1)*W2)
        """
        layers = getattr(self.sparse_model, 'gnn1_layers',
                         getattr(self.sparse_model, 'layers', []))
        h = x
        for layer in layers:
            h = layer.activation(layer.W1(h) + torch.bmm(adj, layer.W2(h)))
        return h

    def _dense_2gnn(self, h, adj):
        """
        Dense 2-set message passing using pretrained KSetLayer weights.
        Fully differentiable through adj.

        For each pair (i,j), canonical feature: [h_min(i,j) || h_max(i,j) || adj[i,j]]
        Neighbor def per Morris et al.: {u,v}~{v,w} iff (u,w)∈E
        (replaced element must be adjacent to new element).
        """
        B, N, D = h.shape
        device = h.device

        # Remove self-loops for correct k-set neighborhoods
        eye = torch.eye(N, device=device).unsqueeze(0)
        adj_clean = adj * (1.0 - eye)

        # Build canonical pair features: [h_min(i,j) || h_max(i,j) || adj[i,j]]
        # Upper triangle (i<j) already canonical; lower triangle swaps.
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        iso = adj_clean.unsqueeze(-1)                 # [B, N, N, 1]

        # Canonical: first = h_min_idx, second = h_max_idx
        upper = torch.triu(torch.ones(N, N, device=device), diagonal=1)  # i<j
        lower = upper.t()                                                 # i>j
        m = upper.unsqueeze(0).unsqueeze(-1)  # [1,N,N,1] — 1 where i<j
        l = lower.unsqueeze(0).unsqueeze(-1)  # [1,N,N,1] — 1 where i>j
        # Where i<j: [h_i, h_j]; where i>j: [h_j, h_i]; diagonal: doesn't matter
        first  = h_i * m + h_j * l + h_i * (1 - m - l)
        second = h_j * m + h_i * l + h_j * (1 - m - l)
        pair_feat = torch.cat([first, second, iso], dim=-1)  # [B, N, N, 2D+1]

        # Apply pretrained 2-GNN layers
        for layer in self.sparse_model.gnn2_layers:
            g = layer.W2(pair_feat)        # [B, N, N, hidden]
            self_part = layer.W1(pair_feat)

            # Morris et al. neighbor definition for 2-sets:
            # {u,v}~{v,w} iff (u,w)∈E (only the replaced element must be adjacent)
            # Case 1 (replace u with w in {u,v}→{v,w}): need adj[u,w]=1
            #   For pair (i,j), neighbor pair (j,w): need adj[i,w] (replacing i)
            #   agg1[i,j] = sum_w adj[i,w] * g[j,w]
            # Case 2 (replace v with w in {u,v}→{u,w}): need adj[v,w]=1
            #   For pair (i,j), neighbor pair (i,w): need adj[j,w] (replacing j)
            #   agg2[i,j] = sum_w adj[j,w] * g[i,w]
            agg1 = torch.einsum('biw,bjwd->bijd', adj_clean, g)
            agg2 = torch.einsum('bjw,biwd->bijd', adj_clean, g)

            pair_feat = layer.activation(self_part + agg1 + agg2)

        # Pool unique pairs (upper triangle, matching pretrained global_add_pool)
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
        return (pair_feat * mask.unsqueeze(0).unsqueeze(-1)).sum(dim=(1, 2))

    def _sparse_3gnn(self, h, adj):
        """
        3-GNN using optimized sparse approach with soft iso-types.

        Full dense 3-set tensors [B,N,N,N,D] would be too large for most graphs.
        Instead we use sparse k-set construction with:
        - torch.no_grad() for structure computation (speed)
        - Soft iso-types from continuous adj values (gradient flows)
        - Node features from dense 1-GNN (gradient flows)
        """
        B, N, D = h.shape
        device = h.device

        eye = torch.eye(N, device=device).unsqueeze(0)
        adj_clean = adj * (1.0 - eye)

        all_feats, all_batch, all_src, all_dst = [], [], [], []
        offset = 0

        for b in range(B):
            h_b = h[b]            # [N, D] — gradient to generator
            adj_b = adj_clean[b]  # [N, N] — gradient to generator

            # Structure computation (no gradient needed)
            with torch.no_grad():
                binary_adj = (adj_b > 0.5).float()
                trips = _sample_ksets(N, 3, 3000, device)
                edges = build_3set_edges(trips, binary_adj, device)

            # Features with soft adjacency (differentiable!)
            fa = h_b[trips[:, 0]]
            fb = h_b[trips[:, 1]]
            fc = h_b[trips[:, 2]]

            # Soft iso-type: continuous edge count (gradient flows through adj!)
            ec_soft = (adj_b[trips[:, 0], trips[:, 1]] +
                       adj_b[trips[:, 1], trips[:, 2]] +
                       adj_b[trips[:, 0], trips[:, 2]])
            # Soft one-hot via triangular basis at {0, 1, 2, 3}
            k_vals = torch.arange(4, device=device, dtype=torch.float)
            iso = torch.clamp(1.0 - (ec_soft.unsqueeze(1) - k_vals).abs(), min=0)

            all_feats.append(torch.cat([fa, fb, fc, iso], dim=1))
            all_batch.append(torch.full((trips.size(0),), b, device=device, dtype=torch.long))

            if edges.size(1) > 0:
                all_src.append(edges[0] + offset)
                all_dst.append(edges[1] + offset)
            offset += trips.size(0)

        if not all_feats:
            return torch.zeros(B, self.sparse_model.hidden_dim, device=device)

        trip_feat = torch.cat(all_feats)
        batch_vec = torch.cat(all_batch)
        if all_src:
            trip_edges = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
        else:
            trip_edges = torch.empty((2, 0), dtype=torch.long, device=device)

        # Run all triplets through pretrained 3-GNN layers (batched)
        h3 = trip_feat
        for layer in self.sparse_model.gnn3_layers:
            self_part = layer.W1(h3)
            if trip_edges.size(1) > 0:
                src_feat = layer.W2(h3[trip_edges[0]])
                nbr_part = torch.zeros_like(self_part)
                nbr_part.index_add_(0, trip_edges[1], src_feat)
            else:
                nbr_part = torch.zeros_like(self_part)
            h3 = layer.activation(self_part + nbr_part)

        return global_add_pool(h3, batch_vec)

    # ==================== Dense Forward Passes ====================

    def _forward_dense_1gnn(self, x, adj):
        """1-GNN: fully differentiable dense forward."""
        h = self._dense_1gnn(x, adj)
        graph_emb = h.sum(dim=1)
        return self.sparse_model.classifier(graph_emb)

    def _forward_dense_12gnn(self, x, adj):
        """12-GNN: fully differentiable dense 1-GNN + 2-GNN."""
        h = self._dense_1gnn(x, adj)
        emb1 = h.sum(dim=1)
        emb2 = self._dense_2gnn(h, adj)
        return self.sparse_model.classifier(torch.cat([emb1, emb2], dim=1))

    def _forward_dense_123gnn(self, x, adj):
        """123-GNN: dense 1+2-GNN, optimized sparse 3-GNN."""
        h = self._dense_1gnn(x, adj)
        emb1 = h.sum(dim=1)
        emb2 = self._dense_2gnn(h, adj)
        emb3 = self._sparse_3gnn(h, adj)
        return self.sparse_model.classifier(torch.cat([emb1, emb2, emb3], dim=1))

    def get_embedding(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Get graph embedding before classification (dense inputs)."""
        if self.model_type == '1gnn':
            h = self._dense_1gnn(x, adj)
            return h.sum(dim=1)
        elif self.model_type == '12gnn':
            h = self._dense_1gnn(x, adj)
            return torch.cat([h.sum(dim=1), self._dense_2gnn(h, adj)], dim=1)
        elif self.model_type == '123gnn':
            h = self._dense_1gnn(x, adj)
            return torch.cat([h.sum(dim=1), self._dense_2gnn(h, adj),
                              self._sparse_3gnn(h, adj)], dim=1)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


# ==================== Utility Models ====================

class SimpleDenseGNN(nn.Module):
    """
    A simple dense GNN that can be used directly with generator outputs.
    Useful for testing and as a baseline.
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
        batch_size = x.size(0)
        n = adj.size(1)
        device = x.device

        identity = torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self = adj + identity
        degree = adj_with_self.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj_with_self / degree

        h = x
        for layer in self.layers:
            h = layer(h, adj_norm)

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
        self_part = self.W1(x)
        neighbor_part = torch.bmm(adj_norm, self.W2(x))
        return self.activation(self_part + neighbor_part)
