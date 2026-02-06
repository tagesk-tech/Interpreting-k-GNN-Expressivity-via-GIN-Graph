"""
train_gin_graph.py
Training script for GIN-Graph model-level explanation generator.

This trains a GAN to generate explanation graphs for a pre-trained k-GNN model.

Usage:
    python train_gin_graph.py --model 1gnn --target_class 0
    python train_gin_graph.py --model 12gnn --target_class 0
    python train_gin_graph.py --model 123gnn --target_class 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import to_dense_adj
import numpy as np
import argparse
import os
import time
from typing import Optional, Tuple, Dict, List

from data_loader import load_dataset, get_class_subset, get_class_statistics, get_dataset_statistics, AVAILABLE_DATASETS
from models_kgnn import get_model
from gin_generator import GINGenerator, GINDiscriminator
from model_wrapper import DenseToSparseWrapper, SimpleDenseGNN
from dynamic_weighting import DynamicWeighting
from metrics import ExplanationEvaluator, ExplanationMetrics
from config import ExperimentConfig, GINGraphConfig, DataConfig, get_class_name


class GINGraphTrainer:
    """
    Trainer for GIN-Graph model-level explanation generator.
    
    Combines:
    - WGAN-GP for realistic graph generation
    - Pre-trained k-GNN for class-specific optimization
    - Dynamic loss weighting for stable training
    """
    
    def __init__(
        self,
        pretrained_gnn: nn.Module,
        model_type: str,
        target_class: int,
        config: GINGraphConfig,
        data_config: DataConfig,
        device: torch.device,
        class_stats: Dict,
        dataset_name: str = 'mutag'
    ):
        self.device = device
        self.config = config
        self.target_class = target_class
        self.model_type = model_type
        self.max_nodes = data_config.gin_max_nodes  # Use smaller size for generation
        self.num_node_feats = data_config.num_node_features
        self.dataset_name = dataset_name.lower()

        # Log if using reduced size
        if data_config.gin_max_nodes < data_config.max_nodes:
            print(f"  Using reduced generation size: {data_config.gin_max_nodes} nodes "
                  f"(dataset max: {data_config.max_nodes})")
        
        # Wrap the pretrained GNN for dense inputs
        self.pretrained_gnn = DenseToSparseWrapper(pretrained_gnn, model_type).to(device)
        self.pretrained_gnn.eval()
        
        # Initialize generator and discriminator
        self.generator = GINGenerator(
            latent_dim=config.latent_dim,
            max_nodes=self.max_nodes,
            num_node_feats=self.num_node_feats,
            hidden_dim=config.hidden_dim,
            dropout=config.generator_dropout
        ).to(device)
        
        self.discriminator = GINDiscriminator(
            max_nodes=self.max_nodes,
            num_node_feats=self.num_node_feats,
            hidden_dim=config.hidden_dim
        ).to(device)
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999)
        )
        
        # Evaluation
        self.evaluator = ExplanationEvaluator(class_stats)
        self.class_stats = class_stats
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'gan_loss': [],
            'gnn_loss': [],
            'degree_loss': [],
            'lambda': [],
            'pred_prob': [],
        }
    
    def prepare_real_batch(
        self,
        batch_list: List,
        max_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of real graphs in dense format.

        Args:
            batch_list: List of PyG Data objects
            max_nodes: Maximum nodes for padding/truncation

        Returns:
            x: [batch, N, D]
            adj: [batch, N, N]
        """
        real_x_list = []
        real_adj_list = []

        for data in batch_list:
            num_nodes = data.num_nodes
            x = data.x.float()

            if num_nodes > max_nodes:
                # Truncate large graphs to max_nodes
                x = x[:max_nodes]
                # Filter edges to only include nodes < max_nodes
                edge_index = data.edge_index
                mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
                edge_index = edge_index[:, mask]
                adj = to_dense_adj(edge_index, max_num_nodes=max_nodes)[0]
                x_padded = x  # Already at max_nodes
            elif num_nodes < max_nodes:
                # Pad small graphs
                x_padded = F.pad(x, (0, 0, 0, max_nodes - num_nodes))
                adj = to_dense_adj(data.edge_index, max_num_nodes=max_nodes)[0]
            else:
                # Exact size
                x_padded = x
                adj = to_dense_adj(data.edge_index, max_num_nodes=max_nodes)[0]

            real_x_list.append(x_padded)
            real_adj_list.append(adj)

        real_x = torch.stack(real_x_list).to(self.device)
        real_adj = torch.stack(real_adj_list).to(self.device)

        return real_x, real_adj

    def _compute_degree_loss(self, fake_adj: torch.Tensor) -> torch.Tensor:
        """
        Compute degree regularization loss.

        Penalizes generated graphs whose average degree deviates from
        the target class's expected degree distribution.

        Args:
            fake_adj: Generated adjacency matrices [batch, N, N]

        Returns:
            Scalar degree loss tensor
        """
        # Get target statistics
        stats = self.class_stats.get(self.target_class, {})
        target_mean = stats.get('mean_degree', 2.0)
        target_std = stats.get('std_degree', 1.0)

        # Compute degree per node
        degrees = fake_adj.sum(dim=-1)

        # Count active nodes per graph (nodes with at least one edge)
        active_mask = degrees > 0.5
        num_active = active_mask.float().sum(dim=-1).clamp(min=1)

        # Total edges per graph
        total_edges = fake_adj.sum(dim=(-1, -2)) / 2

        # Average degree per graph (2*|E|/|V| to match data_loader stats)
        avg_degrees = 2 * total_edges / num_active

        # Normalized squared error (auto-scales by variance)
        normalized_error = (avg_degrees - target_mean) / target_std
        degree_loss = (normalized_error ** 2).mean()

        return self.config.degree_lambda * degree_loss

    def train_discriminator_step(
        self,
        real_x: torch.Tensor,
        real_adj: torch.Tensor,
        batch_size: int
    ) -> float:
        """Single discriminator training step."""
        self.optimizer_D.zero_grad()
        
        # Score real graphs
        real_scores = self.discriminator(real_x, real_adj)
        
        # Generate fake graphs
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_adj, fake_x = self.generator(z, temperature=self.config.temperature)

        # Score fake graphs (detached)
        fake_scores = self.discriminator(fake_x.detach(), fake_adj.detach())
        
        # Gradient penalty
        gp = self.discriminator.compute_gradient_penalty(
            real_x, real_adj,
            fake_x.detach(), fake_adj.detach(),
            self.device
        )
        
        # WGAN loss: E[D(fake)] - E[D(real)] + λ * GP
        d_loss = fake_scores.mean() - real_scores.mean() + self.config.gp_lambda * gp
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item()
    
    def train_generator_step(
        self,
        batch_size: int,
        current_lambda: float
    ) -> Tuple[float, float, float, float, float]:
        """Single generator training step."""
        self.optimizer_G.zero_grad()

        # Generate fake graphs
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_adj, fake_x = self.generator(z, temperature=self.config.temperature)

        # GAN loss: -E[D(fake)]
        gan_scores = self.discriminator(fake_x, fake_adj)
        l_gan = -gan_scores.mean()

        # GNN loss: Cross-entropy for target class
        gnn_logits = self.pretrained_gnn(fake_x, fake_adj)
        target_labels = torch.full(
            (batch_size,), self.target_class,
            device=self.device, dtype=torch.long
        )
        l_gnn = F.cross_entropy(gnn_logits, target_labels)

        # Degree regularization loss (adaptive based on dataset variance)
        l_degree = self._compute_degree_loss(fake_adj)

        # Combined loss with dynamic weighting
        # Degree loss is always active to enforce structure
        total_loss = (1 - current_lambda) * l_gan + current_lambda * l_gnn + l_degree

        total_loss.backward()
        self.optimizer_G.step()

        # Compute prediction probability for logging
        with torch.no_grad():
            probs = F.softmax(gnn_logits, dim=1)
            pred_prob = probs[:, self.target_class].mean().item()

        return total_loss.item(), l_gan.item(), l_gnn.item(), pred_prob, l_degree.item()
    
    def train(
        self,
        dataset,
        epochs: int,
        verbose: bool = True,
        log_interval: int = 10,
        checkpoint_interval: int = 20,
        output_dir: Optional[str] = None
    ):
        """
        Full training loop.

        Args:
            dataset: PyG dataset filtered to target class
            epochs: Number of training epochs
            verbose: Print progress
            log_interval: Epochs between log messages
            checkpoint_interval: Save checkpoint + samples every N epochs (0 to disable)
            output_dir: Directory for intermediate checkpoints/samples (required if checkpoint_interval > 0)
        """
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x  # Keep as list for custom processing
        )
        
        total_iters = epochs * len(train_loader)
        weight_scheduler = DynamicWeighting(
            total_iters,
            min_lambda=self.config.lambda_min,
            max_lambda=self.config.lambda_max,
            p=self.config.lambda_p,
            k=self.config.lambda_k
        )
        
        if verbose:
            print(f"Starting GIN-Graph training")
            print(f"  Target class: {self.target_class} ({get_class_name(self.target_class, self.dataset_name)})")
            print(f"  Dataset size: {len(dataset)}")
            print(f"  Epochs: {epochs}")
            print(f"  Total iterations: {total_iters}")
            print()
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_pred_prob = 0
            num_batches = 0
            
            for batch_list in train_loader:
                batch_size = len(batch_list)
                
                # Prepare real data
                real_x, real_adj = self.prepare_real_batch(batch_list, self.max_nodes)
                
                # Train discriminator
                for _ in range(self.config.n_critic):
                    d_loss = self.train_discriminator_step(real_x, real_adj, batch_size)
                
                # Train generator
                current_lambda = weight_scheduler.get_lambda()
                g_loss, gan_loss, gnn_loss, pred_prob, degree_loss = self.train_generator_step(
                    batch_size, current_lambda
                )

                # Track metrics
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                epoch_pred_prob += pred_prob
                num_batches += 1
                self.global_step += 1

                # Store history
                self.history['d_loss'].append(d_loss)
                self.history['g_loss'].append(g_loss)
                self.history['gan_loss'].append(gan_loss)
                self.history['gnn_loss'].append(gnn_loss)
                self.history['degree_loss'].append(degree_loss)
                self.history['lambda'].append(current_lambda)
                self.history['pred_prob'].append(pred_prob)
            
            # Log progress
            if verbose and (epoch % log_interval == 0 or epoch == epochs - 1):
                avg_d = epoch_d_loss / num_batches
                avg_g = epoch_g_loss / num_batches
                avg_prob = epoch_pred_prob / num_batches
                print(f"Epoch {epoch:4d} | D Loss: {avg_d:7.4f} | G Loss: {avg_g:7.4f} | "
                      f"Pred Prob: {avg_prob:.4f} | λ: {current_lambda:.3f}")

            # Intermediate checkpointing
            if (output_dir and checkpoint_interval > 0
                    and (epoch % checkpoint_interval == 0 or epoch == epochs - 1)):
                os.makedirs(output_dir, exist_ok=True)
                # Save model checkpoint
                ckpt_path = os.path.join(
                    output_dir,
                    f'ckpt_{self.dataset_name}_{self.model_type}_epoch{epoch}.pt'
                )
                self.save_checkpoint(ckpt_path)
                # Generate a small batch of samples
                sample_adjs, sample_xs, sample_metrics = self.generate_explanations(num_samples=16)
                predictions = np.array([m.prediction_probability for m in sample_metrics])
                sample_path = os.path.join(
                    output_dir,
                    f'samples_{self.dataset_name}_{self.model_type}_epoch{epoch}.npz'
                )
                np.savez(sample_path, adjs=sample_adjs, xs=sample_xs,
                         predictions=predictions, epoch=epoch)
                if verbose:
                    print(f"         → Checkpoint + 16 samples saved to {output_dir}")
    
    def generate_explanations(
        self,
        num_samples: int = 100,
        temperature: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, List[ExplanationMetrics]]:
        """
        Generate and evaluate explanation graphs.
        
        Args:
            num_samples: Number of explanations to generate
            temperature: Gumbel-Softmax temperature (lower = more discrete)
            
        Returns:
            adjs: Adjacency matrices [num_samples, N, N]
            xs: Node features [num_samples, N, D]
            metrics: List of ExplanationMetrics
        """
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            fake_adj, fake_x = self.generator(z, temperature=temperature, hard=True)
            
            # Get predictions from the GNN
            logits = self.pretrained_gnn(fake_x, fake_adj)
            probs = F.softmax(logits, dim=1)
            pred_probs = probs[:, self.target_class].cpu().numpy()
            
            # For embedding similarity, we'd need to compute embeddings
            # For now, use prediction probability as a proxy
            embedding_sims = pred_probs  # Simplified
        
        adjs = fake_adj.cpu().numpy()
        xs = fake_x.cpu().numpy()
        
        # Evaluate each explanation
        metrics = self.evaluator.evaluate_batch(
            adjs, xs, self.target_class,
            pred_probs, embedding_sims
        )
        
        self.generator.train()
        
        return adjs, xs, metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config,
            'target_class': self.target_class,
            'model_type': self.model_type,
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']


def load_pretrained_kgnn(
    model_name: str,
    checkpoint_dir: str,
    device: torch.device,
    dataset_name: str = 'mutag'
) -> nn.Module:
    """Load a pretrained k-GNN model."""
    checkpoint_path = os.path.join(checkpoint_dir, f'{dataset_name}_{model_name}.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first: python train_kgnn.py --dataset {dataset_name} --model {model_name}"
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = get_model(
        checkpoint['model_name'],
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_name.upper()} (Test Acc: {checkpoint['best_test_acc']:.4f})")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train GIN-Graph explanation generator')
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=AVAILABLE_DATASETS,
                        help=f'Dataset to use ({", ".join(AVAILABLE_DATASETS)})')
    parser.add_argument('--model', type=str, default='1gnn',
                        choices=['1gnn', '12gnn', '123gnn'],
                        help='k-GNN model to explain')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class for explanation generation')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension for generator')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory with k-GNN checkpoints')
    parser.add_argument('--gin_checkpoint_dir', type=str, default='./gin_checkpoints',
                        help='Directory for GIN-Graph model checkpoints and training samples')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for final analysis results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of explanation samples to generate')
    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    dataset_name = args.dataset.lower()

    print("=" * 60)
    print("GIN-Graph Training")
    print("=" * 60)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Model to explain: {args.model.upper()}")
    print(f"Target class: {args.target_class}")
    print(f"Device: {device}")
    print()

    # Load dataset
    print(f"Loading {dataset_name.upper()} dataset...")
    dataset = load_dataset(dataset_name)
    
    # Get class statistics for validation
    class_stats = get_class_statistics(dataset)
    
    # Add average nodes per class
    for label in [0, 1]:
        nodes = [d.num_nodes for d in dataset if d.y.item() == label]
        class_stats[label]['avg_nodes'] = np.mean(nodes)
    
    # Filter to target class
    target_dataset = get_class_subset(dataset, args.target_class)
    print(f"Target class has {len(target_dataset)} graphs")
    print()
    
    # Load pretrained k-GNN
    print("Loading pretrained k-GNN...")
    pretrained_gnn = load_pretrained_kgnn(args.model, args.checkpoint_dir, device, dataset_name)
    print()

    # Create configs with dataset-specific settings
    gin_config = GINGraphConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    data_config = DataConfig.from_dataset(dataset_name)
    
    # Create trainer
    trainer = GINGraphTrainer(
        pretrained_gnn=pretrained_gnn,
        model_type=args.model,
        target_class=args.target_class,
        config=gin_config,
        data_config=data_config,
        device=device,
        class_stats=class_stats,
        dataset_name=dataset_name
    )
    
    # Train (intermediate checkpoints + samples go to gin_checkpoint_dir)
    print("=" * 60)
    print("Training GIN-Graph Generator")
    print("=" * 60)
    trainer.train(target_dataset, epochs=args.epochs, log_interval=30,
                  output_dir=os.path.join(args.gin_checkpoint_dir, dataset_name, 'training'))
    print()

    # Save final GIN-Graph model to gin_checkpoint_dir/<dataset>/
    gin_dataset_dir = os.path.join(args.gin_checkpoint_dir, dataset_name)
    os.makedirs(gin_dataset_dir, exist_ok=True)
    gin_save_path = os.path.join(
        gin_dataset_dir,
        f'{args.model}_class{args.target_class}.pt'
    )
    trainer.save_checkpoint(gin_save_path)
    print(f"GIN-Graph model saved to: {gin_save_path}")

    # Generate and evaluate explanations
    print()
    print("=" * 60)
    print("Generating Explanation Graphs")
    print("=" * 60)
    adjs, xs, metrics = trainer.generate_explanations(num_samples=args.num_samples)

    # Print summary
    summary = trainer.evaluator.compute_summary_stats(metrics)
    print(f"\nGeneration Summary:")
    print(f"  Total generated: {summary['total_generated']}")
    print(f"  Valid explanations: {summary['num_valid']} ({summary['validity_rate']*100:.1f}%)")
    print(f"  Mean validation score: {summary['mean_validation_score']:.4f}")
    print(f"  Mean prediction prob: {summary['mean_prediction_prob']:.4f}")

    # Get best explanations
    best = trainer.evaluator.get_best_explanations(metrics, top_k=5)
    print(f"\nTop 5 Explanations:")
    for rank, (idx, m) in enumerate(best, 1):
        print(f"  {rank}. Score: {m.validation_score:.4f}, Nodes: {m.num_nodes}, "
              f"Edges: {m.num_edges}, Pred: {m.prediction_probability:.4f}")

    # Save final analysis results to output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    if best:
        best_indices = [idx for idx, _ in best]
        results_path = os.path.join(
            args.output_dir,
            f'explanations_{dataset_name}_{args.model}_class{args.target_class}.npz'
        )
        np.savez(results_path, adjs=adjs[best_indices], xs=xs[best_indices],
                 indices=best_indices)
        print(f"\nBest explanations saved to: {results_path}")


if __name__ == "__main__":
    main()
