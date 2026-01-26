"""
metrics.py
Metrics for evaluating model-level explanation graphs.

Based on the GIN-Graph paper's validation criteria:
- Prediction probability (p): How confident the GNN is about the target class
- Embedding similarity (s): How similar the explanation embedding is to real graphs
- Degree score (d): How realistic the graph structure is

Validation score: v = (s * p * d)^(1/3)
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ExplanationMetrics:
    """Metrics for a single explanation graph."""
    prediction_probability: float
    embedding_similarity: float
    degree_score: float
    validation_score: float
    average_degree: float
    num_nodes: int
    num_edges: int
    is_valid: bool
    granularity: float  # 1 - min(1, num_nodes / avg_nodes_in_class)


def compute_degree_score(
    avg_degree: float,
    class_mean_degree: float,
    class_std_degree: float
) -> float:
    """
    Compute degree-based validity score.
    
    Uses a Gaussian kernel centered at the class mean degree.
    
    Args:
        avg_degree: Average degree of the explanation graph
        class_mean_degree: Mean average degree for the target class
        class_std_degree: Standard deviation of average degrees
        
    Returns:
        Degree score in [0, 1]
    """
    if class_std_degree == 0:
        return 1.0 if avg_degree == class_mean_degree else 0.0
    
    # Gaussian kernel: exp(-(x - μ)² / (2σ²))
    score = np.exp(-((avg_degree - class_mean_degree) ** 2) / (2 * class_std_degree ** 2))
    return float(score)


def is_valid_explanation(
    avg_degree: float,
    class_mean_degree: float,
    class_std_degree: float,
    threshold: float = 3.0
) -> bool:
    """
    Check if an explanation graph is valid based on degree statistics.
    
    A graph is invalid if its average degree is more than `threshold` 
    standard deviations from the class mean.
    
    Args:
        avg_degree: Average degree of the explanation graph
        class_mean_degree: Mean average degree for the target class
        class_std_degree: Standard deviation of average degrees
        threshold: Number of standard deviations for validity
        
    Returns:
        True if valid, False otherwise
    """
    lower = class_mean_degree - threshold * class_std_degree
    upper = class_mean_degree + threshold * class_std_degree
    return lower <= avg_degree <= upper


def compute_validation_score(
    prediction_prob: float,
    embedding_sim: float,
    degree_score: float
) -> float:
    """
    Compute the combined validation score.
    
    v = (s * p * d)^(1/3)
    
    This is sensitive to low values - a graph cannot achieve a high score
    if it performs poorly in any one aspect.
    
    Args:
        prediction_prob: Prediction probability for target class
        embedding_sim: Cosine similarity of embeddings
        degree_score: Degree-based validity score
        
    Returns:
        Validation score in [0, 1]
    """
    # Clamp to valid range
    p = max(0.0, min(1.0, prediction_prob))
    s = max(0.0, min(1.0, embedding_sim))
    d = max(0.0, min(1.0, degree_score))
    
    return (s * p * d) ** (1/3)


def compute_granularity(
    num_nodes: int,
    class_avg_nodes: float
) -> float:
    """
    Compute granularity metric.
    
    k = 1 - min(1, num_nodes / avg_nodes)
    
    - k = 0: Coarse-grained (explanation is as large as typical graphs)
    - k → 1: Fine-grained (explanation highlights small substructures)
    
    Args:
        num_nodes: Number of nodes in explanation graph
        class_avg_nodes: Average number of nodes in the target class
        
    Returns:
        Granularity score in [0, 1)
    """
    if class_avg_nodes == 0:
        return 0.0
    return 1.0 - min(1.0, num_nodes / class_avg_nodes)


class ExplanationEvaluator:
    """
    Evaluator for model-level explanation graphs.
    
    Computes validation scores, filters invalid explanations,
    and tracks statistics across multiple generations.
    """
    
    def __init__(
        self,
        class_stats: Dict[int, Dict[str, float]],
        validity_threshold: float = 3.0,
        min_validation_score: float = 0.5
    ):
        """
        Args:
            class_stats: Per-class statistics {class: {'mean_degree': x, 'std_degree': y, 'avg_nodes': z}}
            validity_threshold: Standard deviations for degree-based validity
            min_validation_score: Minimum score to consider an explanation valid
        """
        self.class_stats = class_stats
        self.validity_threshold = validity_threshold
        self.min_validation_score = min_validation_score
    
    def evaluate_single(
        self,
        adj: np.ndarray,
        x: np.ndarray,
        target_class: int,
        prediction_prob: float,
        embedding_sim: float,
        edge_threshold: float = 0.5
    ) -> ExplanationMetrics:
        """
        Evaluate a single explanation graph.
        
        Args:
            adj: Adjacency matrix [N, N]
            x: Node features [N, D]
            target_class: Target class for explanation
            prediction_prob: Model's prediction probability for target class
            embedding_sim: Cosine similarity with class embedding centroid
            edge_threshold: Threshold for counting edges
            
        Returns:
            ExplanationMetrics dataclass
        """
        # Count edges and nodes (removing isolated nodes)
        edges = (adj > edge_threshold).astype(np.float32)
        np.fill_diagonal(edges, 0)  # Remove self-loops
        
        # Make symmetric if not already
        edges = np.maximum(edges, edges.T)
        
        # Count actual nodes (non-isolated)
        degree_per_node = edges.sum(axis=1)
        active_nodes = (degree_per_node > 0).sum()
        num_edges = int(edges.sum() / 2)  # Divide by 2 for undirected
        
        # Handle empty graphs
        if active_nodes == 0:
            return ExplanationMetrics(
                prediction_probability=prediction_prob,
                embedding_similarity=embedding_sim,
                degree_score=0.0,
                validation_score=0.0,
                average_degree=0.0,
                num_nodes=0,
                num_edges=0,
                is_valid=False,
                granularity=1.0
            )
        
        avg_degree = num_edges / active_nodes
        
        # Get class statistics
        stats = self.class_stats.get(target_class, {})
        class_mean_degree = stats.get('mean_degree', 1.0)
        class_std_degree = stats.get('std_degree', 1.0)
        class_avg_nodes = stats.get('avg_nodes', 10.0)
        
        # Compute metrics
        degree_score = compute_degree_score(avg_degree, class_mean_degree, class_std_degree)
        is_valid = is_valid_explanation(avg_degree, class_mean_degree, class_std_degree, self.validity_threshold)
        validation_score = compute_validation_score(prediction_prob, embedding_sim, degree_score)
        granularity = compute_granularity(active_nodes, class_avg_nodes)
        
        # Additional validity check based on overall score
        is_valid = is_valid and (validation_score >= self.min_validation_score)
        
        return ExplanationMetrics(
            prediction_probability=prediction_prob,
            embedding_similarity=embedding_sim,
            degree_score=degree_score,
            validation_score=validation_score,
            average_degree=avg_degree,
            num_nodes=int(active_nodes),
            num_edges=num_edges,
            is_valid=is_valid,
            granularity=granularity
        )
    
    def evaluate_batch(
        self,
        adjs: np.ndarray,
        xs: np.ndarray,
        target_class: int,
        prediction_probs: np.ndarray,
        embedding_sims: np.ndarray
    ) -> List[ExplanationMetrics]:
        """
        Evaluate a batch of explanation graphs.
        
        Args:
            adjs: Adjacency matrices [batch, N, N]
            xs: Node features [batch, N, D]
            target_class: Target class
            prediction_probs: Prediction probabilities [batch]
            embedding_sims: Embedding similarities [batch]
            
        Returns:
            List of ExplanationMetrics
        """
        results = []
        batch_size = adjs.shape[0]
        
        for i in range(batch_size):
            metrics = self.evaluate_single(
                adjs[i], xs[i], target_class,
                float(prediction_probs[i]),
                float(embedding_sims[i])
            )
            results.append(metrics)
        
        return results
    
    def get_best_explanations(
        self,
        metrics_list: List[ExplanationMetrics],
        top_k: int = 10,
        valid_only: bool = True
    ) -> List[Tuple[int, ExplanationMetrics]]:
        """
        Get the top-k best explanation graphs.
        
        Args:
            metrics_list: List of metrics for all generated graphs
            top_k: Number of top graphs to return
            valid_only: If True, only consider valid explanations
            
        Returns:
            List of (index, metrics) tuples, sorted by validation score
        """
        indexed_metrics = list(enumerate(metrics_list))
        
        if valid_only:
            indexed_metrics = [(i, m) for i, m in indexed_metrics if m.is_valid]
        
        # Sort by validation score (descending)
        indexed_metrics.sort(key=lambda x: x[1].validation_score, reverse=True)
        
        return indexed_metrics[:top_k]
    
    def compute_summary_stats(
        self,
        metrics_list: List[ExplanationMetrics]
    ) -> Dict[str, float]:
        """
        Compute summary statistics across all explanations.
        
        Returns:
            Dictionary with mean/std of all metrics
        """
        if not metrics_list:
            return {}
        
        valid_metrics = [m for m in metrics_list if m.is_valid]
        
        return {
            'total_generated': len(metrics_list),
            'num_valid': len(valid_metrics),
            'validity_rate': len(valid_metrics) / len(metrics_list),
            'mean_validation_score': np.mean([m.validation_score for m in metrics_list]),
            'mean_valid_score': np.mean([m.validation_score for m in valid_metrics]) if valid_metrics else 0,
            'mean_prediction_prob': np.mean([m.prediction_probability for m in metrics_list]),
            'mean_embedding_sim': np.mean([m.embedding_similarity for m in metrics_list]),
            'mean_degree_score': np.mean([m.degree_score for m in metrics_list]),
            'mean_granularity': np.mean([m.granularity for m in valid_metrics]) if valid_metrics else 0,
        }
