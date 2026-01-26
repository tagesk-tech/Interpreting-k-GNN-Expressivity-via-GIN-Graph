"""
train.py
Training script for k-GNN experiments.

Usage:
    python train.py --model 1gnn --epochs 100
    python train.py --model 2gnn --epochs 100
    python train.py --model 3gnn --epochs 100
    python train.py --model 12gnn --epochs 100
    python train.py --model 13gnn --epochs 100
    python train.py --model 123gnn --epochs 100
"""

import torch
import torch.nn as nn
import argparse
import time
from data_loader import load_mutag, create_data_loaders
from models_k import get_model


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


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Train k-GNN models on MUTAG')
    parser.add_argument('--model', type=str, default='1gnn',
                        choices=['1gnn', '2gnn', '3gnn', '12gnn', '13gnn', '123gnn'],
                        help='Model architecture to train')
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
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Print header
    print("=" * 60)
    print(f"Training {args.model.upper()} on MUTAG")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Hidden dim: {args.hidden}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Load data
    print("Loading MUTAG dataset...")
    dataset = load_mutag()
    
    train_loader, test_loader, _, _ = create_data_loaders(
        dataset, 
        train_ratio=0.8, 
        batch_size=args.batch_size,
        seed=args.seed
    )
    print()
    
    # Create model
    print("Creating model...")
    model = get_model(
        args.model,
        input_dim=dataset.num_node_features,
        hidden_dim=args.hidden,
        output_dim=dataset.num_classes
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model: {args.model.upper()}")
    print(f"Parameters: {num_params:,}")
    print()
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Training...")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Test Loss':>10} | {'Test Acc':>10} | {'Time':>8}")
    print("-" * 60)
    
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>10.4f} | {test_loss:>10.4f} | {test_acc:>10.4f} | {epoch_time:>7.2f}s")
    
    print("-" * 60)
    print()
    
    # Final results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Best Test Accuracy:  {best_test_acc:.4f} (epoch {best_epoch})")
    print()
    
    # Save model
    save_path = f'trained_{args.model}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': args.model,
        'hidden_dim': args.hidden,
        'input_dim': dataset.num_node_features,
        'output_dim': dataset.num_classes,
        'final_test_acc': test_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
    }, save_path)
    print(f"Model saved to '{save_path}'")


if __name__ == "__main__":
    main()