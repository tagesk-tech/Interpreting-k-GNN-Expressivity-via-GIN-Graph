"""
GIN-Graph: Generative Interpretation Network for Model-Level Explanation of GNNs

This implementation analyzes trained 1-GNN models to discover what structural
patterns they have learned, based on the GIN-Graph paper (Yue et al., 2025).

Usage:
    python gin_graph.py --model_path trained_1gnn.pt --target_class 0
    python gin_graph.py --model_path trained_1gnn.pt --target_class 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, Optional, Dict, List
import argparse
import os


# =============================================================================
# 1-GNN Model (must match your training architecture)
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


class OneGNN(nn.Module):
    """Complete 1-GNN model for graph classification."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(OneGNNLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(OneGNNLayer(hidden_dim, hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return self.classifier(x)
    
    def get_embedding(self, x, edge_index, batch):
        """Get graph embedding before classification."""
        for layer in self.layers:
            x = layer(x, edge_index)
        return global_mean_pool(x, batch)


# =============================================================================
# GIN-Graph Generator
# =============================================================================

class GraphGenerator(nn.Module):
    """
    Generator network that produces adjacency matrix and node features
    from a latent vector z.
    """
    def __init__(self, latent_dim: int, num_nodes: int, node_feature_dim: int, 
                 hidden_dim: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        
        # MLP for generating node features (num_nodes x node_feature_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * node_feature_dim)
        )
        
        # MLP for generating adjacency matrix (num_nodes x num_nodes)
        self.adj_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * num_nodes)
        )
    
    def forward(self, z: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate graph from latent vector.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            temperature: Gumbel-Softmax temperature
            
        Returns:
            X: Node features [batch_size, num_nodes, node_feature_dim]
            A: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size = z.size(0)
        
        # Generate node features with Gumbel-Softmax for categorical features
        node_logits = self.node_mlp(z).view(batch_size, self.num_nodes, self.node_feature_dim)
        X = F.gumbel_softmax(node_logits, tau=temperature, hard=False, dim=-1)
        
        # Generate adjacency matrix
        adj_logits = self.adj_mlp(z).view(batch_size, self.num_nodes, self.num_nodes)
        
        # Make symmetric (undirected graph)
        adj_logits = (adj_logits + adj_logits.transpose(-1, -2)) / 2
        
        # Apply Gumbel-Softmax for edge existence (binary)
        # Stack with zeros to create 2-class (no edge, edge) logits
        adj_logits_binary = torch.stack([torch.zeros_like(adj_logits), adj_logits], dim=-1)
        A_soft = F.gumbel_softmax(adj_logits_binary, tau=temperature, hard=False, dim=-1)[..., 1]
        
        # Zero out diagonal (no self-loops)
        mask = 1 - torch.eye(self.num_nodes, device=z.device)
        A = A_soft * mask
        
        return X, A


class GraphDiscriminator(nn.Module):
    """
    Discriminator network using a GNN to distinguish real from generated graphs.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = OneGNNLayer(node_feature_dim, hidden_dim)
        self.conv2 = OneGNNLayer(hidden_dim, hidden_dim)
        self.conv3 = OneGNNLayer(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


# =============================================================================
# GIN-Graph Interpreter
# =============================================================================

class GINGraph:
    """
    GIN-Graph: Generative Interpretation Network for Model-Level Explanation.
    
    Generates explanation graphs that maximize prediction probability for a 
    target class while maintaining similarity to real graphs.
    """
    
    def __init__(self, 
                 gnn_model: nn.Module,
                 dataset,
                 num_nodes: int = 10,
                 latent_dim: int = 32,
                 hidden_dim: int = 128,
                 device: str = 'cpu'):
        """
        Args:
            gnn_model: The trained GNN model to explain
            dataset: The dataset the GNN was trained on
            num_nodes: Number of nodes in generated explanation graphs
            latent_dim: Dimension of latent vector z
            hidden_dim: Hidden dimension for generator/discriminator
            device: Device to run on
        """
        self.device = torch.device(device)
        self.gnn_model = gnn_model.to(self.device)
        self.gnn_model.eval()
        
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.node_feature_dim = dataset.num_node_features
        self.num_classes = dataset.num_classes
        
        # Initialize generator and discriminator
        self.generator = GraphGenerator(
            latent_dim=latent_dim,
            num_nodes=num_nodes,
            node_feature_dim=self.node_feature_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.discriminator = GraphDiscriminator(
            node_feature_dim=self.node_feature_dim,
            hidden_dim=hidden_dim // 2
        ).to(self.device)
        
        self.latent_dim = latent_dim
        
        # Compute dataset statistics for validation
        self._compute_dataset_statistics()
    
    def _compute_dataset_statistics(self):
        """Compute mean and std of average degree per class."""
        self.class_degree_stats = {}
        self.class_embeddings = {}
        
        for class_idx in range(self.num_classes):
            degrees = []
            embeddings = []
            
            for data in self.dataset:
                if data.y.item() == class_idx:
                    # Compute average degree
                    num_nodes = data.x.size(0)
                    num_edges = data.edge_index.size(1) // 2  # Undirected
                    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
                    degrees.append(avg_degree)
                    
                    # Get embedding
                    with torch.no_grad():
                        data = data.to(self.device)
                        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                        emb = self.gnn_model.get_embedding(data.x, data.edge_index, batch)
                        embeddings.append(emb.cpu())
            
            degrees = np.array(degrees)
            self.class_degree_stats[class_idx] = {
                'mean': degrees.mean(),
                'std': degrees.std()
            }
            
            # Mean embedding for the class
            self.class_embeddings[class_idx] = torch.cat(embeddings, dim=0).mean(dim=0)
    
    def _adj_to_edge_index(self, A: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert adjacency matrix to edge_index format."""
        # Threshold to get discrete adjacency
        A_discrete = (A > threshold).float()
        
        # Get edge indices
        edge_index = (A_discrete > 0).nonzero(as_tuple=False).t()
        return edge_index
    
    def _matrix_to_pyg_data(self, X: torch.Tensor, A: torch.Tensor, 
                            threshold: float = 0.5) -> Data:
        """Convert generated matrices to PyG Data object."""
        # Get discrete node features (argmax for one-hot)
        x = F.one_hot(X.argmax(dim=-1), num_classes=self.node_feature_dim).float()
        
        # Get edge index
        edge_index = self._adj_to_edge_index(A, threshold)
        
        return Data(x=x, edge_index=edge_index)
    
    def _compute_validation_score(self, X: torch.Tensor, A: torch.Tensor, 
                                   target_class: int) -> Dict[str, float]:
        """
        Compute validation score for generated graph.
        v = (s * p * d)^(1/3)
        """
        # Convert to PyG data
        data = self._matrix_to_pyg_data(X[0], A[0])
        data = data.to(self.device)
        
        if data.edge_index.size(1) == 0:
            return {'score': 0, 'probability': 0, 'similarity': 0, 'degree_score': 0}
        
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Prediction probability
            logits = self.gnn_model(data.x, data.edge_index, batch)
            probs = F.softmax(logits, dim=-1)
            p = probs[0, target_class].item()
            
            # Embedding similarity
            emb = self.gnn_model.get_embedding(data.x, data.edge_index, batch)
            target_emb = self.class_embeddings[target_class].to(self.device)
            s = F.cosine_similarity(emb, target_emb.unsqueeze(0)).item()
            s = max(0, s)  # Clamp negative similarities
        
        # Degree score
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1) // 2
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        
        mu = self.class_degree_stats[target_class]['mean']
        sigma = self.class_degree_stats[target_class]['std']
        if sigma > 0:
            d = np.exp(-((avg_degree - mu) ** 2) / (2 * sigma ** 2))
        else:
            d = 1.0 if abs(avg_degree - mu) < 0.1 else 0.0
        
        # Combined validation score
        score = (s * p * d) ** (1/3)
        
        return {
            'score': score,
            'probability': p,
            'similarity': s,
            'degree_score': d,
            'avg_degree': avg_degree
        }
    
    def _dynamic_lambda(self, t: int, T: int, lambda_min: float = 0.0, 
                        lambda_max: float = 0.5, p: float = 0.3, k: float = 10.0) -> float:
        """
        Dynamic loss weight as per Equation 3 in GIN-Graph paper.
        λ(t) = λ_min + (λ_max - λ_min) * σ(k * (2 * t/T - p) / (1 - p) - 1)
        """
        progress = (2 * t / T - p) / (1 - p) - 1
        weight = torch.sigmoid(torch.tensor(k * progress)).item()
        return lambda_min + (lambda_max - lambda_min) * weight
    
    def _get_real_batch(self, target_class: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of real graphs from the target class."""
        class_data = [d for d in self.dataset if d.y.item() == target_class]
        
        # Sample random graphs
        indices = np.random.choice(len(class_data), min(batch_size, len(class_data)), replace=True)
        batch_list = [class_data[i] for i in indices]
        
        # Pad to same size
        max_nodes = self.num_nodes
        X_list = []
        A_list = []
        
        for data in batch_list:
            num_nodes = min(data.x.size(0), max_nodes)
            
            # Pad node features
            X = torch.zeros(max_nodes, self.node_feature_dim)
            X[:num_nodes] = data.x[:num_nodes]
            X_list.append(X)
            
            # Create adjacency matrix
            A = torch.zeros(max_nodes, max_nodes)
            edge_index = data.edge_index
            valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_edges]
            A[edge_index[0], edge_index[1]] = 1
            A_list.append(A)
        
        X = torch.stack(X_list).to(self.device)
        A = torch.stack(A_list).to(self.device)
        
        # Create batch tensor for discriminator
        batch = torch.arange(len(batch_list)).repeat_interleave(max_nodes).to(self.device)
        
        return X, A, batch
    
    def train(self, target_class: int, epochs: int = 500, batch_size: int = 32,
              lr_g: float = 1e-4, lr_d: float = 1e-4, n_critic: int = 5,
              gp_weight: float = 10.0, temperature_start: float = 1.0,
              temperature_end: float = 0.5, verbose: bool = True) -> List[Dict]:
        """
        Train GIN-Graph to generate explanation graphs for target class.
        
        Args:
            target_class: Class to generate explanations for
            epochs: Number of training epochs
            batch_size: Batch size
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
            n_critic: Discriminator updates per generator update
            gp_weight: Gradient penalty weight
            temperature_start: Initial Gumbel-Softmax temperature
            temperature_end: Final Gumbel-Softmax temperature
            verbose: Print progress
            
        Returns:
            List of generated explanation graphs with scores
        """
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        explanations = []
        best_score = 0
        best_explanation = None
        
        for epoch in range(epochs):
            # Anneal temperature
            temperature = temperature_start - (temperature_start - temperature_end) * (epoch / epochs)
            
            # Dynamic lambda
            lambda_weight = self._dynamic_lambda(epoch, epochs)
            
            # ==================
            # Train Discriminator
            # ==================
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                
                # Real graphs
                X_real, A_real, batch_real = self._get_real_batch(target_class, batch_size)
                
                # Convert real to discriminator format
                x_real_flat = X_real.view(-1, self.node_feature_dim)
                edge_indices_real = []
                for i in range(X_real.size(0)):
                    edges = (A_real[i] > 0.5).nonzero(as_tuple=False).t()
                    edges = edges + i * self.num_nodes
                    edge_indices_real.append(edges)
                edge_index_real = torch.cat(edge_indices_real, dim=1) if edge_indices_real else torch.zeros(2, 0, dtype=torch.long, device=self.device)
                
                d_real = self.discriminator(x_real_flat, edge_index_real, batch_real)
                
                # Fake graphs
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                X_fake, A_fake = self.generator(z, temperature)
                
                x_fake_flat = X_fake.view(-1, self.node_feature_dim)
                batch_fake = torch.arange(batch_size).repeat_interleave(self.num_nodes).to(self.device)
                
                edge_indices_fake = []
                for i in range(batch_size):
                    # Soft edges for differentiability during training
                    A_i = A_fake[i]
                    edges = (A_i > 0.3).nonzero(as_tuple=False).t()
                    edges = edges + i * self.num_nodes
                    edge_indices_fake.append(edges)
                edge_index_fake = torch.cat(edge_indices_fake, dim=1) if edge_indices_fake else torch.zeros(2, 0, dtype=torch.long, device=self.device)
                
                d_fake = self.discriminator(x_fake_flat.detach(), edge_index_fake, batch_fake)
                
                # Wasserstein loss
                d_loss = d_fake.mean() - d_real.mean()
                
                # Gradient penalty
                alpha = torch.rand(batch_size, 1, 1, device=self.device)
                X_interp = (alpha * X_real + (1 - alpha) * X_fake.detach()).requires_grad_(True)
                A_interp = (alpha * A_real + (1 - alpha) * A_fake.detach()).requires_grad_(True)
                
                x_interp_flat = X_interp.view(-1, self.node_feature_dim)
                edge_indices_interp = []
                for i in range(batch_size):
                    edges = (A_interp[i] > 0.3).nonzero(as_tuple=False).t()
                    edges = edges + i * self.num_nodes
                    edge_indices_interp.append(edges)
                edge_index_interp = torch.cat(edge_indices_interp, dim=1) if edge_indices_interp else torch.zeros(2, 0, dtype=torch.long, device=self.device)
                
                d_interp = self.discriminator(x_interp_flat, edge_index_interp, batch_fake)
                
                gradients = torch.autograd.grad(
                    outputs=d_interp,
                    inputs=[X_interp, A_interp],
                    grad_outputs=torch.ones_like(d_interp),
                    create_graph=True,
                    retain_graph=True
                )
                grad_norm = sum(g.norm(2) for g in gradients if g is not None)
                gp = gp_weight * ((grad_norm - 1) ** 2)
                
                d_loss_total = d_loss + gp
                d_loss_total.backward()
                optimizer_D.step()
            
            # ==================
            # Train Generator
            # ==================
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            X_fake, A_fake = self.generator(z, temperature)
            
            x_fake_flat = X_fake.view(-1, self.node_feature_dim)
            batch_fake = torch.arange(batch_size).repeat_interleave(self.num_nodes).to(self.device)
            
            edge_indices_fake = []
            for i in range(batch_size):
                A_i = A_fake[i]
                edges = (A_i > 0.3).nonzero(as_tuple=False).t()
                edges = edges + i * self.num_nodes
                edge_indices_fake.append(edges)
            edge_index_fake = torch.cat(edge_indices_fake, dim=1) if edge_indices_fake else torch.zeros(2, 0, dtype=torch.long, device=self.device)
            
            # GAN loss
            d_fake = self.discriminator(x_fake_flat, edge_index_fake, batch_fake)
            g_loss_gan = -d_fake.mean()
            
            # GNN prediction loss (cross-entropy for target class)
            g_loss_gnn = 0
            for i in range(min(batch_size, 8)):  # Subsample for efficiency
                data = self._matrix_to_pyg_data(X_fake[i], A_fake[i], threshold=0.3)
                data = data.to(self.device)
                
                if data.edge_index.size(1) > 0:
                    batch_single = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    logits = self.gnn_model(data.x, data.edge_index, batch_single)
                    target = torch.tensor([target_class], device=self.device)
                    g_loss_gnn += F.cross_entropy(logits, target)
            
            g_loss_gnn = g_loss_gnn / min(batch_size, 8)
            
            # Combined loss with dynamic weighting
            g_loss = (1 - lambda_weight) * g_loss_gan + lambda_weight * g_loss_gnn
            g_loss.backward()
            optimizer_G.step()
            
            # ==================
            # Evaluation
            # ==================
            if (epoch + 1) % 50 == 0 or epoch == 0:
                self.generator.eval()
                with torch.no_grad():
                    z = torch.randn(1, self.latent_dim, device=self.device)
                    X_gen, A_gen = self.generator(z, temperature=0.5)
                    
                    scores = self._compute_validation_score(X_gen, A_gen, target_class)
                    
                    if scores['score'] > best_score:
                        best_score = scores['score']
                        best_explanation = {
                            'X': X_gen.cpu(),
                            'A': A_gen.cpu(),
                            'scores': scores,
                            'epoch': epoch + 1
                        }
                    
                    explanations.append({
                        'epoch': epoch + 1,
                        'scores': scores,
                        'X': X_gen.cpu(),
                        'A': A_gen.cpu()
                    })
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} | λ={lambda_weight:.3f} | "
                          f"Score={scores['score']:.4f} | Prob={scores['probability']:.4f} | "
                          f"Sim={scores['similarity']:.4f} | Deg={scores['degree_score']:.4f}")
                
                self.generator.train()
        
        # Store best explanation
        self.best_explanation = best_explanation
        return explanations
    
    def visualize_explanation(self, explanation: Optional[Dict] = None, 
                              title: str = "Explanation Graph",
                              save_path: Optional[str] = None) -> plt.Figure:
        """Visualize an explanation graph."""
        if explanation is None:
            explanation = self.best_explanation
        
        if explanation is None:
            raise ValueError("No explanation available. Run train() first.")
        
        X = explanation['X'][0]
        A = explanation['A'][0]
        scores = explanation['scores']
        
        # Convert to discrete graph
        X_discrete = X.argmax(dim=-1).numpy()
        A_discrete = (A > 0.5).numpy().astype(int)
        
        # Create NetworkX graph
        G = nx.Graph()
        num_nodes = X_discrete.shape[0]
        
        # Add nodes with features
        for i in range(num_nodes):
            G.add_node(i, feature=X_discrete[i])
        
        # Add edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if A_discrete[i, j] > 0:
                    G.add_edge(i, j)
        
        # Remove isolated nodes for cleaner visualization
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)
        
        if G.number_of_nodes() == 0:
            print("Warning: Generated graph has no connected nodes")
            return None
        
        # MUTAG atom colors (C, N, O, F, I, Cl, Br)
        atom_colors = ['gray', 'blue', 'red', 'green', 'purple', 'orange', 'brown']
        atom_labels = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Get node colors based on features
        node_colors = []
        node_labels = {}
        for node in G.nodes():
            feat = G.nodes[node]['feature']
            if feat < len(atom_colors):
                node_colors.append(atom_colors[feat])
                node_labels[node] = atom_labels[feat]
            else:
                node_colors.append('gray')
                node_labels[node] = '?'
        
        # Draw graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=500,
                edge_color='black', width=2, with_labels=False)
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=12, font_weight='bold')
        
        # Add title with scores
        ax.set_title(f"{title}\n"
                     f"Validation Score: {scores['score']:.4f} | "
                     f"Probability: {scores['probability']:.4f} | "
                     f"Similarity: {scores['similarity']:.4f}",
                     fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig


# =============================================================================
# Main Script
# =============================================================================

def load_model(model_path: str, device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    """Load a trained 1-GNN model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = OneGNN(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='GIN-Graph: Model-level GNN Explanation')
    parser.add_argument('--model_path', type=str, default='trained_1gnn.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class to explain (0=Non-Mutagen, 1=Mutagen)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs')
    parser.add_argument('--num_nodes', type=int, default=12,
                        help='Number of nodes in generated graphs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='explanations',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("GIN-Graph: Model-Level GNN Explanation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Target class: {args.target_class}")
    print()
    
    # Load dataset
    print("Loading MUTAG dataset...")
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    print(f"  Graphs: {len(dataset)}")
    print(f"  Node features: {dataset.num_node_features}")
    print(f"  Classes: {dataset.num_classes}")
    print()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, checkpoint = load_model(args.model_path, device)
    print(f"  Hidden dim: {checkpoint['hidden_dim']}")
    print(f"  Test accuracy: {checkpoint.get('best_test_acc', 'N/A')}")
    print()
    
    # Initialize GIN-Graph
    print("Initializing GIN-Graph interpreter...")
    interpreter = GINGraph(
        gnn_model=model,
        dataset=dataset,
        num_nodes=args.num_nodes,
        latent_dim=32,
        hidden_dim=128,
        device=device
    )
    print()
    
    # Train
    class_names = ['Non-Mutagen', 'Mutagen']
    print(f"Generating explanation for class: {class_names[args.target_class]}")
    print("-" * 60)
    
    explanations = interpreter.train(
        target_class=args.target_class,
        epochs=args.epochs,
        batch_size=32,
        verbose=True
    )
    
    print("-" * 60)
    print()
    
    # Results
    best = interpreter.best_explanation
    if best:
        print("=" * 60)
        print("BEST EXPLANATION FOUND")
        print("=" * 60)
        print(f"  Epoch: {best['epoch']}")
        print(f"  Validation Score: {best['scores']['score']:.4f}")
        print(f"  Prediction Probability: {best['scores']['probability']:.4f}")
        print(f"  Embedding Similarity: {best['scores']['similarity']:.4f}")
        print(f"  Degree Score: {best['scores']['degree_score']:.4f}")
        print(f"  Average Degree: {best['scores']['avg_degree']:.2f}")
        print()
        
        # Save visualization
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"explanation_class{args.target_class}.png")
        interpreter.visualize_explanation(
            best,
            title=f"Explanation for {class_names[args.target_class]}",
            save_path=save_path
        )
        plt.show()
    else:
        print("No valid explanation found. Try increasing epochs or adjusting parameters.")


if __name__ == "__main__":
    main()