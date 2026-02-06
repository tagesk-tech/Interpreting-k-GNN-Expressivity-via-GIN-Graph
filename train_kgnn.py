"""
train_kgnn.py
Training script for k-GNN models on graph classification datasets.

Usage:
    python train_kgnn.py --model 1gnn --epochs 100
    python train_kgnn.py --model 12gnn --epochs 100  # Hierarchical 1-2-GNN
    python train_kgnn.py --model 123gnn --epochs 100 # Hierarchical 1-2-3-GNN
    python train_kgnn.py --all  # Train all hierarchical models

    # Train on different datasets:
    python train_kgnn.py --dataset mutag --model 1gnn
    python train_kgnn.py --dataset proteins --model 12gnn
"""

import torch
import torch.nn as nn
import argparse
import time
import os
from pathlib import Path

from data_loader import load_dataset, create_data_loaders, get_dataset_statistics, print_dataset_statistics, AVAILABLE_DATASETS
from models_kgnn import get_model, count_parameters
from config import ExperimentConfig, KGNNConfig, DataConfig


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    return total_loss / len(loader), correct / total


def train_single_model(
    model_name: str,
    config: ExperimentConfig,
    device: torch.device,
    dataset,
    train_loader,
    test_loader,
    verbose: bool = True
):
    """Train a single k-GNN model."""

    if verbose:
        print("=" * 60)
        print(f"Training {model_name.upper()} on {config.data.name}")
        print("=" * 60)
    
    # Create model
    model = get_model(
        model_name,
        input_dim=dataset.num_node_features,
        hidden_dim=config.kgnn.hidden_dim,
        output_dim=dataset.num_classes,
        dropout=config.kgnn.dropout
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    if verbose:
        print(f"Model: {model_name.upper()}")
        print(f"Parameters: {num_params:,}")
        print(f"Hidden dim: {config.kgnn.hidden_dim}")
        print(f"Learning rate: {config.kgnn.learning_rate}")
        print()
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.kgnn.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    if verbose:
        print("Training...")
        print("-" * 70)
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Test Loss':>10} | {'Test Acc':>10} | {'Time':>8}")
        print("-" * 70)
    
    best_test_acc = 0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(config.kgnn.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>10.4f} | {test_loss:>10.4f} | {test_acc:>10.4f} | {epoch_time:>7.2f}s")
    
    if verbose:
        print("-" * 70)
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Best Test Accuracy:  {best_test_acc:.4f} (epoch {best_epoch})")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Save model (include dataset name in filename)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    dataset_name = config.data.name.lower()
    save_path = os.path.join(config.checkpoint_dir, f'{dataset_name}_{model_name}.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'dataset_name': dataset_name,
        'hidden_dim': config.kgnn.hidden_dim,
        'input_dim': dataset.num_node_features,
        'output_dim': dataset.num_classes,
        'final_test_acc': test_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'num_params': num_params,
    }, save_path)
    
    if verbose:
        print(f"\nModel saved to '{save_path}'")
    
    return model, best_test_acc


def main():
    parser = argparse.ArgumentParser(description='Train k-GNN models on graph classification datasets')
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=AVAILABLE_DATASETS,
                        help=f'Dataset to train on ({", ".join(AVAILABLE_DATASETS)})')
    parser.add_argument('--model', type=str, default='1gnn',
                        choices=['1gnn', '12gnn', '123gnn'],
                        help='Model architecture to train')
    parser.add_argument('--all', action='store_true',
                        help='Train all models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    args = parser.parse_args()

    # Create config with dataset-specific settings
    data_config = DataConfig.from_dataset(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed
    )

    config = ExperimentConfig(
        data=data_config,
        kgnn=KGNNConfig(
            hidden_dim=args.hidden,
            learning_rate=args.lr,
            epochs=args.epochs
        ),
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )

    device = config.get_device()

    # Set seeds
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print(f"k-GNN Training for {config.data.name} Dataset")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    # Load data
    print(f"Loading {config.data.name} dataset...")
    dataset = load_dataset(args.dataset, config.data.root)
    stats = get_dataset_statistics(dataset)
    print_dataset_statistics(stats)
    
    train_loader, test_loader, _, _ = create_data_loaders(
        dataset,
        train_ratio=config.data.train_ratio,
        batch_size=config.data.batch_size,
        seed=config.data.seed
    )
    print()
    
    # Determine which models to train
    if args.all:
        models_to_train = ['1gnn', '12gnn', '123gnn']
    else:
        models_to_train = [args.model]
    
    # Train models
    results = {}
    for model_name in models_to_train:
        print("\n" + "=" * 60)
        model, acc = train_single_model(
            model_name, config, device,
            dataset, train_loader, test_loader
        )
        results[model_name] = acc
        print()
    
    # Summary
    if len(models_to_train) > 1:
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for model_name, acc in results.items():
            print(f"  {model_name.upper():>8}: {acc:.4f}")


if __name__ == "__main__":
    main()
