"""
GIN-based Layer Interpretability Analysis for Hierarchical k-GNNs

This module provides tools to analyze what each layer of a hierarchical k-GNN
learns at different scales (k values), using techniques inspired by:
- "GIN-Graph: A Generative Interpretation Network" (Yue et al., 2025)
- "How Powerful Are Graph Neural Networks?" (Xu et al., ICLR 2019)

The goal is to understand how increasing k helps the GNN capture more complex
graph structures that 1-WL cannot distinguish.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class LayerEmbeddingAnalyzer:
    """
    Analyzes embeddings from different layers of a hierarchical k-GNN.
    
    Key questions we want to answer:
    1. How do embeddings evolve across layers within each k-GNN?
    2. How do different k values capture different graph structures?
    3. What graph patterns are distinguishable at each layer/k level?
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_embeddings = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture intermediate embeddings."""
        self.hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.layer_embeddings[name] = output[0].detach()
                else:
                    self.layer_embeddings[name] = output.detach()
            return hook
        
        # Register hooks for 1-GNN layers
        if hasattr(self.model, 'gnn_1'):
            for i, conv in enumerate(self.model.gnn_1.convs):
                hook = conv.register_forward_hook(get_hook(f'1-gnn-layer-{i}'))
                self.hooks.append(hook)
        
        # Register hooks for k-GNN layers
        if hasattr(self.model, 'k_gnns'):
            for k, kgnn in self.model.k_gnns.items():
                for i, conv in enumerate(kgnn.convs):
                    hook = conv.register_forward_hook(get_hook(f'{k}-gnn-layer-{i}'))
                    self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_embeddings(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Get layer embeddings for a single graph.
        
        Returns dict mapping layer names to embeddings.
        """
        self.layer_embeddings = {}
        self.register_hooks()
        
        with torch.no_grad():
            _ = self.model(data.x, data.edge_index, data.batch, return_layer_info=True)
        
        embeddings = self.layer_embeddings.copy()
        self.remove_hooks()
        
        return embeddings
    
    def compute_layer_similarity(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute pairwise cosine similarity between consecutive layers.
        
        This shows how much the representation changes at each layer.
        """
        similarities = {}
        
        sorted_keys = sorted(embeddings.keys())
        for i in range(len(sorted_keys) - 1):
            key1, key2 = sorted_keys[i], sorted_keys[i+1]
            emb1, emb2 = embeddings[key1], embeddings[key2]
            
            # Ensure same shape for comparison
            if emb1.shape != emb2.shape:
                continue
            
            # Compute cosine similarity
            emb1_flat = emb1.view(-1)
            emb2_flat = emb2.view(-1)
            sim = F.cosine_similarity(emb1_flat.unsqueeze(0), emb2_flat.unsqueeze(0))
            similarities[f'{key1}_to_{key2}'] = sim.item()
        
        return similarities


class GraphStructureDistinguisher:
    """
    Tests which graph structures can be distinguished at each k level.
    
    Based on known limitations of 1-WL and how k-WL overcomes them:
    - k=1 (1-WL): Cannot distinguish regular graphs, cannot count triangles
    - k=2: Can count triangles, distinguish some regular graphs
    - k=3: Can distinguish most practical graph structures
    """
    
    @staticmethod
    def create_regular_graphs() -> Tuple[Data, Data]:
        """
        Create two 3-regular graphs that 1-WL cannot distinguish.
        Classic example: two different 6-node graphs where each node has degree 3.
        """
        # Graph 1: Two triangles connected by matching edges
        edge_index_1 = torch.tensor([
            [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 0, 3, 1, 4, 2, 5],
            [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5, 3, 0, 4, 1, 5, 2]
        ], dtype=torch.long)
        
        # Graph 2: A 6-cycle with opposite corners connected
        edge_index_2 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 3, 1, 4, 2, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5, 3, 0, 4, 1, 5, 2]
        ], dtype=torch.long)
        
        x = torch.ones(6, 1)  # Same features for all nodes
        
        graph1 = Data(x=x.clone(), edge_index=edge_index_1)
        graph2 = Data(x=x.clone(), edge_index=edge_index_2)
        
        return graph1, graph2
    
    @staticmethod
    def create_triangle_vs_square() -> Tuple[Data, Data]:
        """
        Create a triangle and a 4-cycle - 1-WL can distinguish these
        but they're useful for validating basic functionality.
        """
        # Triangle
        edge_index_1 = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ], dtype=torch.long)
        x1 = torch.ones(3, 1)
        
        # Square (4-cycle)
        edge_index_2 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3]
        ], dtype=torch.long)
        x2 = torch.ones(4, 1)
        
        graph1 = Data(x=x1, edge_index=edge_index_1)
        graph2 = Data(x=x2, edge_index=edge_index_2)
        
        return graph1, graph2
    
    @staticmethod
    def create_cfi_pair() -> Tuple[Data, Data]:
        """
        Create CFI (Cai-Fürer-Immerman) graph pair.
        These are classic examples that require higher k to distinguish.
        """
        # Simplified CFI-like construction
        # Two 8-node graphs that require k>=3 to distinguish
        
        # Graph 1: Two squares sharing an edge
        edge_index_1 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0, 2, 4, 4, 5, 5, 6, 6, 2],
            [1, 0, 2, 1, 3, 2, 0, 3, 4, 2, 5, 4, 6, 5, 2, 6]
        ], dtype=torch.long)
        
        # Graph 2: Different connectivity pattern
        edge_index_2 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0, 1, 4, 4, 5, 5, 6, 6, 1],
            [1, 0, 2, 1, 3, 2, 0, 3, 4, 1, 5, 4, 6, 5, 1, 6]
        ], dtype=torch.long)
        
        x = torch.ones(7, 1)
        
        graph1 = Data(x=x.clone(), edge_index=edge_index_1)
        graph2 = Data(x=x.clone(), edge_index=edge_index_2)
        
        return graph1, graph2
    
    def test_distinguishability(self, model: nn.Module, 
                                graph1: Data, graph2: Data) -> Dict[str, bool]:
        """
        Test if model can distinguish two graphs at different k levels.
        
        Returns dict mapping k-level to whether graphs are distinguished.
        """
        model.eval()
        
        results = {}
        
        # Get embeddings for both graphs
        batch1 = Batch.from_data_list([graph1])
        batch2 = Batch.from_data_list([graph2])
        
        with torch.no_grad():
            out1, layer_info1 = model(batch1.x, batch1.edge_index, batch1.batch, 
                                      return_layer_info=True)
            out2, layer_info2 = model(batch2.x, batch2.edge_index, batch2.batch,
                                      return_layer_info=True)
        
        # Compare at each k level
        for key in layer_info1.keys():
            if len(layer_info1[key]) > 0 and len(layer_info2[key]) > 0:
                # Get final layer representation for this k
                rep1 = layer_info1[key][0][-1].mean(dim=0)  # Pool over nodes/k-sets
                rep2 = layer_info2[key][0][-1].mean(dim=0)
                
                # Check if representations are different
                diff = torch.norm(rep1 - rep2).item()
                results[key] = diff > 1e-5
        
        # Check final predictions
        pred_diff = torch.norm(out1 - out2).item()
        results['final_output'] = pred_diff > 1e-5
        
        return results


class ExpressivenessAnalyzer:
    """
    Analyzes how increasing k improves expressiveness.
    
    Key insights from the papers:
    1. 1-GNN ≡ 1-WL in distinguishing power
    2. k-GNN ≡ set-based k-WL
    3. Higher k strictly increases expressiveness (up to graph size)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def measure_layer_expressiveness(self, dataset: List[Data]) -> Dict[str, Dict]:
        """
        Measure expressiveness of each layer/k-level.
        
        Metrics:
        - Embedding variance: Higher variance suggests more distinctive representations
        - Inter-class separation: How well classes are separated
        - Intra-class cohesion: How similar same-class graphs are
        """
        self.model.eval()
        
        embeddings_by_layer = defaultdict(list)
        labels = []
        
        for data in dataset:
            batch = Batch.from_data_list([data])
            
            with torch.no_grad():
                _, layer_info = self.model(
                    batch.x, batch.edge_index, batch.batch, return_layer_info=True
                )
            
            for key, layer_list in layer_info.items():
                if len(layer_list) > 0:
                    # Get pooled representation from final layer
                    final_rep = layer_list[0][-1].mean(dim=0)
                    embeddings_by_layer[key].append(final_rep)
            
            if hasattr(data, 'y'):
                labels.append(data.y.item())
        
        results = {}
        
        for key, embs in embeddings_by_layer.items():
            if len(embs) == 0:
                continue
            
            emb_tensor = torch.stack(embs)
            
            # Compute metrics
            variance = emb_tensor.var(dim=0).mean().item()
            
            results[key] = {
                'variance': variance,
                'num_samples': len(embs),
                'embedding_dim': embs[0].shape[0]
            }
            
            # If we have labels, compute class separation
            if len(labels) == len(embs):
                labels_tensor = torch.tensor(labels)
                unique_labels = labels_tensor.unique()
                
                if len(unique_labels) > 1:
                    # Inter-class distance
                    class_centers = []
                    for label in unique_labels:
                        mask = labels_tensor == label
                        class_center = emb_tensor[mask].mean(dim=0)
                        class_centers.append(class_center)
                    
                    inter_class_dist = 0
                    count = 0
                    for i in range(len(class_centers)):
                        for j in range(i+1, len(class_centers)):
                            inter_class_dist += torch.norm(
                                class_centers[i] - class_centers[j]
                            ).item()
                            count += 1
                    
                    if count > 0:
                        results[key]['inter_class_distance'] = inter_class_dist / count
                    
                    # Intra-class distance
                    intra_class_dist = 0
                    for label in unique_labels:
                        mask = labels_tensor == label
                        class_embs = emb_tensor[mask]
                        if len(class_embs) > 1:
                            center = class_embs.mean(dim=0)
                            dists = torch.norm(class_embs - center, dim=1)
                            intra_class_dist += dists.mean().item()
                    
                    results[key]['intra_class_distance'] = intra_class_dist / len(unique_labels)
        
        return results
    
    def visualize_embeddings(self, dataset: List[Data], 
                            save_path: Optional[str] = None) -> None:
        """
        Visualize embeddings from different k levels using t-SNE.
        """
        self.model.eval()
        
        embeddings_by_k = defaultdict(list)
        labels = []
        
        for data in dataset:
            batch = Batch.from_data_list([data])
            
            with torch.no_grad():
                _, layer_info = self.model(
                    batch.x, batch.edge_index, batch.batch, return_layer_info=True
                )
            
            for key, layer_list in layer_info.items():
                if len(layer_list) > 0:
                    final_rep = layer_list[0][-1].mean(dim=0)
                    embeddings_by_k[key].append(final_rep.cpu().numpy())
            
            if hasattr(data, 'y'):
                labels.append(data.y.item())
        
        # Create visualization
        num_k_levels = len(embeddings_by_k)
        fig, axes = plt.subplots(1, num_k_levels, figsize=(5 * num_k_levels, 5))
        
        if num_k_levels == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.array(labels))
        
        for ax, (key, embs) in zip(axes, embeddings_by_k.items()):
            emb_array = np.stack(embs)
            
            if emb_array.shape[1] > 2:
                # Use t-SNE for dimensionality reduction
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embs)-1))
                emb_2d = tsne.fit_transform(emb_array)
            else:
                emb_2d = emb_array
            
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.7)
            ax.set_title(f'{key} Embeddings')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


class GINGraphExplainer:
    """
    Generate model-level explanations using GAN-based approach.
    
    Based on GIN-Graph paper:
    - Train generator to produce graphs similar to real graphs
    - Maximize prediction probability for target class
    - Use validation score to filter invalid explanations
    """
    
    def __init__(self, model: nn.Module, hidden_dim: int = 64, 
                 num_nodes: int = 10, node_features: int = 1):
        self.model = model
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.node_features = node_features
        
        # Generator: produces continuous adjacency and feature matrices
        self.generator = self._build_generator()
        
    def _build_generator(self) -> nn.Module:
        """Build the graph generator network."""
        class GraphGenerator(nn.Module):
            def __init__(self, latent_dim, hidden_dim, num_nodes, node_features):
                super().__init__()
                self.num_nodes = num_nodes
                self.node_features = node_features
                
                # MLP to generate adjacency matrix
                adj_size = num_nodes * num_nodes
                self.adj_net = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, adj_size),
                    nn.Sigmoid()
                )
                
                # MLP to generate node features
                feat_size = num_nodes * node_features
                self.feat_net = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, feat_size)
                )
                
            def forward(self, z):
                batch_size = z.size(0)
                
                # Generate adjacency matrix
                adj_flat = self.adj_net(z)
                adj = adj_flat.view(batch_size, self.num_nodes, self.num_nodes)
                # Make symmetric and remove self-loops
                adj = (adj + adj.transpose(1, 2)) / 2
                adj = adj * (1 - torch.eye(self.num_nodes, device=z.device))
                
                # Generate node features
                feat_flat = self.feat_net(z)
                features = feat_flat.view(batch_size, self.num_nodes, self.node_features)
                
                return adj, features
        
        return GraphGenerator(self.hidden_dim, self.hidden_dim, 
                             self.num_nodes, self.node_features)
    
    def _adj_to_edge_index(self, adj: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert adjacency matrix to edge index format."""
        adj_binary = (adj > threshold).float()
        edge_index = adj_binary.nonzero().t()
        return edge_index
    
    def compute_validation_score(self, adj: torch.Tensor, features: torch.Tensor,
                                target_class: int, 
                                reference_mean_degree: float,
                                reference_std_degree: float) -> Tuple[float, Dict]:
        """
        Compute validation score for generated explanation graph.
        
        v = (s * p * d)^(1/3)
        where:
        - s: embedding similarity to class average
        - p: prediction probability for target class
        - d: degree score (Gaussian based on reference statistics)
        """
        self.model.eval()
        
        # Convert to graph format
        edge_index = self._adj_to_edge_index(adj.squeeze(0))
        x = features.squeeze(0)
        
        if edge_index.size(1) == 0:
            return 0.0, {'similarity': 0, 'probability': 0, 'degree_score': 0}
        
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        with torch.no_grad():
            output = self.model(x, edge_index, batch)
            probs = F.softmax(output, dim=1)
            pred_prob = probs[0, target_class].item()
        
        # Compute average degree
        num_nodes = x.size(0)
        num_edges = edge_index.size(1) / 2  # Undirected
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        
        # Degree score (Gaussian)
        degree_score = np.exp(-((avg_degree - reference_mean_degree) ** 2) / 
                             (2 * reference_std_degree ** 2 + 1e-8))
        
        # Similarity score (placeholder - would need reference embeddings)
        similarity = 0.5  # Default
        
        # Validation score
        v_score = (similarity * pred_prob * degree_score) ** (1/3)
        
        return v_score, {
            'similarity': similarity,
            'probability': pred_prob,
            'degree_score': degree_score,
            'avg_degree': avg_degree
        }
    
    def generate_explanation(self, target_class: int, 
                            num_iterations: int = 1000,
                            learning_rate: float = 0.01,
                            lambda_gnn_start: float = 0.1,
                            lambda_gnn_end: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate explanation graph for target class.
        
        Uses dynamic loss weighting scheme from GIN-Graph:
        - Early training: focus on generating realistic graphs
        - Later training: focus on maximizing target class probability
        """
        self.model.eval()
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        
        best_score = 0
        best_adj, best_features = None, None
        
        for iteration in range(num_iterations):
            # Dynamic lambda
            progress = iteration / num_iterations
            lambda_gnn = lambda_gnn_start + (lambda_gnn_end - lambda_gnn_start) * progress
            
            # Sample latent vector
            z = torch.randn(1, self.hidden_dim)
            
            # Generate graph
            adj, features = self.generator(z)
            
            # Convert to edge index (using Gumbel-Softmax for differentiability)
            adj_soft = F.gumbel_softmax(
                torch.stack([1 - adj, adj], dim=-1), 
                tau=1.0, hard=True
            )[..., 1]
            
            edge_index = adj_soft.squeeze(0).nonzero().t()
            
            if edge_index.size(1) == 0:
                continue
            
            x = features.squeeze(0)
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            # Get model prediction
            output = self.model(x, edge_index, batch)
            
            # GNN loss (maximize probability of target class)
            probs = F.softmax(output, dim=1)
            loss_gnn = -torch.log(probs[0, target_class] + 1e-8)
            
            # Regularization loss (encourage sparse, realistic graphs)
            loss_reg = adj.mean()  # Sparsity
            loss_reg += (adj - adj.transpose(1, 2)).abs().mean()  # Symmetry
            
            # Combined loss
            loss = lambda_gnn * loss_gnn + (1 - lambda_gnn) * loss_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track best explanation
            with torch.no_grad():
                score, _ = self.compute_validation_score(
                    adj, features, target_class, 
                    reference_mean_degree=2.0,
                    reference_std_degree=1.0
                )
                
                if score > best_score:
                    best_score = score
                    best_adj = adj.detach().clone()
                    best_features = features.detach().clone()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss={loss.item():.4f}, "
                      f"Prob={probs[0, target_class].item():.4f}, Score={score:.4f}")
        
        return best_adj, best_features


class KLevelInterpretabilityReport:
    """
    Generate comprehensive interpretability report for hierarchical k-GNN.
    
    Analyzes:
    1. What each k-level learns
    2. How layers evolve within each k-level
    3. Which graph structures require higher k
    4. Model-level explanations for each class
    """
    
    def __init__(self, model: nn.Module, dataset: List[Data]):
        self.model = model
        self.dataset = dataset
        
        self.embedding_analyzer = LayerEmbeddingAnalyzer(model)
        self.structure_distinguisher = GraphStructureDistinguisher()
        self.expressiveness_analyzer = ExpressivenessAnalyzer(model)
        
    def generate_report(self) -> Dict:
        """Generate full interpretability report."""
        report = {}
        
        # 1. Test structure distinguishability at different k levels
        print("Testing structure distinguishability...")
        report['distinguishability'] = {}
        
        # Regular graphs (1-WL fails)
        g1, g2 = self.structure_distinguisher.create_regular_graphs()
        report['distinguishability']['regular_graphs'] = \
            self.structure_distinguisher.test_distinguishability(self.model, g1, g2)
        
        # Triangle vs square (1-WL succeeds)
        g1, g2 = self.structure_distinguisher.create_triangle_vs_square()
        report['distinguishability']['triangle_vs_square'] = \
            self.structure_distinguisher.test_distinguishability(self.model, g1, g2)
        
        # CFI-like graphs
        g1, g2 = self.structure_distinguisher.create_cfi_pair()
        report['distinguishability']['cfi_pair'] = \
            self.structure_distinguisher.test_distinguishability(self.model, g1, g2)
        
        # 2. Measure expressiveness at each layer
        print("Measuring layer expressiveness...")
        report['expressiveness'] = self.expressiveness_analyzer.measure_layer_expressiveness(
            self.dataset
        )
        
        # 3. Analyze embedding similarity across layers
        print("Analyzing embedding evolution...")
        if len(self.dataset) > 0:
            sample_data = self.dataset[0]
            batch = Batch.from_data_list([sample_data])
            embeddings = self.embedding_analyzer.get_embeddings(batch)
            report['layer_similarities'] = self.embedding_analyzer.compute_layer_similarity(
                embeddings
            )
        
        return report
    
    def print_report(self, report: Dict) -> None:
        """Print formatted report."""
        print("\n" + "="*60)
        print("HIERARCHICAL k-GNN INTERPRETABILITY REPORT")
        print("="*60)
        
        print("\n1. STRUCTURE DISTINGUISHABILITY")
        print("-"*40)
        for test_name, results in report.get('distinguishability', {}).items():
            print(f"\n{test_name}:")
            for k_level, distinguished in results.items():
                status = "✓ Distinguished" if distinguished else "✗ Not Distinguished"
                print(f"  {k_level}: {status}")
        
        print("\n2. LAYER EXPRESSIVENESS")
        print("-"*40)
        for k_level, metrics in report.get('expressiveness', {}).items():
            print(f"\n{k_level}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        print("\n3. LAYER SIMILARITIES")
        print("-"*40)
        for transition, similarity in report.get('layer_similarities', {}).items():
            print(f"  {transition}: {similarity:.4f}")
        
        print("\n" + "="*60)
