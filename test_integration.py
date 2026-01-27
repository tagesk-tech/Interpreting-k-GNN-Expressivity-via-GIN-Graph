"""
test_integration.py
Comprehensive fast integration tests for the k-GNN + GIN-Graph pipeline.

This test validates that ALL components work correctly end-to-end without
running expensive full training. It tests:
1. All k-GNN models (1gnn, 12gnn, 123gnn) forward/backward pass
2. Short training runs (1-2 epochs) for all models
3. GIN-Graph trainer initialization and short runs
4. Model wrappers (dense-to-sparse conversion)
5. Full pipeline: train → generate → evaluate

Run: python test_integration.py
"""

import sys
import time
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


# ============================================================================
# Test utilities
# ============================================================================

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration


def run_test(test_func, name: str) -> TestResult:
    """Run a test function and capture results."""
    start = time.time()
    try:
        result = test_func()
        duration = time.time() - start
        if result is True or result is None:
            return TestResult(name, True, "OK", duration)
        elif isinstance(result, str):
            return TestResult(name, True, result, duration)
        else:
            return TestResult(name, False, str(result), duration)
    except Exception as e:
        duration = time.time() - start
        tb = traceback.format_exc()
        return TestResult(name, False, f"{e}\n{tb}", duration)


# ============================================================================
# Test 1: Data Loading
# ============================================================================

def test_data_loading():
    """Test MUTAG dataset loading and statistics."""
    from data_loader import load_mutag, get_dataset_statistics, create_data_loaders, get_class_statistics
    
    dataset = load_mutag('./data')
    stats = get_dataset_statistics(dataset)
    
    # Verify expected dataset properties
    assert stats['num_graphs'] == 188, f"Expected 188 graphs, got {stats['num_graphs']}"
    assert stats['num_node_features'] == 7, f"Expected 7 node features, got {stats['num_node_features']}"
    assert stats['num_classes'] == 2, f"Expected 2 classes, got {stats['num_classes']}"
    
    # Test data loaders
    train_loader, test_loader, train_data, test_data = create_data_loaders(
        dataset, train_ratio=0.8, batch_size=32, seed=42
    )
    
    assert len(train_data) + len(test_data) == 188
    
    # Test class statistics
    class_stats = get_class_statistics(dataset)
    assert 0 in class_stats and 1 in class_stats
    
    return f"188 graphs, 7 features, 2 classes"


# ============================================================================
# Test 2: k-GNN Model Forward Pass
# ============================================================================

def test_kgnn_forward_pass():
    """Test forward pass for all k-GNN models with real data."""
    from data_loader import load_mutag, create_data_loaders
    from models_kgnn import get_model, count_parameters
    
    dataset = load_mutag('./data')
    train_loader, _, _, _ = create_data_loaders(dataset, batch_size=8, seed=42)
    
    # Get a batch
    batch = next(iter(train_loader))
    device = torch.device('cpu')
    
    results = []
    for model_name in ['1gnn', '12gnn', '123gnn']:
        model = get_model(
            model_name,
            input_dim=dataset.num_node_features,
            hidden_dim=32,  # Small for speed
            output_dim=dataset.num_classes,
            dropout=0.0
        ).to(device)
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Check output shape
        assert out.shape == (batch.y.size(0), 2), f"{model_name}: wrong output shape {out.shape}"
        
        # Check gradients flow
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        
        # Verify gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, f"{model_name}: no gradients computed"
        
        params = count_parameters(model)
        results.append(f"{model_name}:{params}")
    
    return ", ".join(results)


# ============================================================================
# Test 3: k-GNN Short Training
# ============================================================================

def test_kgnn_short_training():
    """Test that k-GNN models can be trained for a few epochs."""
    from data_loader import load_mutag, create_data_loaders
    from models_kgnn import get_model
    
    dataset = load_mutag('./data')
    train_loader, test_loader, _, _ = create_data_loaders(dataset, batch_size=32, seed=42)
    device = torch.device('cpu')
    
    results = []
    
    for model_name in ['1gnn', '12gnn', '123gnn']:
        model = get_model(
            model_name,
            input_dim=dataset.num_node_features,
            hidden_dim=32,
            output_dim=dataset.num_classes,
            dropout=0.0
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Train for 2 epochs
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(2):
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            if initial_loss is None:
                initial_loss = avg_loss
            final_loss = avg_loss
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        acc = correct / total
        results.append(f"{model_name}:{acc:.2f}")
    
    return ", ".join(results)


# ============================================================================
# Test 4: GIN Generator/Discriminator
# ============================================================================

def test_gin_generator_discriminator():
    """Test GIN-Graph generator and discriminator."""
    from gin_generator import GINGenerator, GINDiscriminator
    
    batch_size = 4
    latent_dim = 32
    max_nodes = 28
    num_node_feats = 7
    
    generator = GINGenerator(
        latent_dim=latent_dim,
        max_nodes=max_nodes,
        num_node_feats=num_node_feats,
        hidden_dim=64,
        dropout=0.0
    )
    
    discriminator = GINDiscriminator(
        max_nodes=max_nodes,
        num_node_feats=num_node_feats,
        hidden_dim=64
    )
    
    # Test generator forward
    z = torch.randn(batch_size, latent_dim)
    adj, x = generator(z, temperature=1.0, hard=True)
    
    assert adj.shape == (batch_size, max_nodes, max_nodes), f"Wrong adj shape: {adj.shape}"
    assert x.shape == (batch_size, max_nodes, num_node_feats), f"Wrong x shape: {x.shape}"
    
    # Test discriminator forward
    scores = discriminator(x, adj)
    assert scores.shape == (batch_size, 1), f"Wrong scores shape: {scores.shape}"
    
    # Test gradient flow through discriminator
    loss = scores.mean()
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in discriminator.parameters())
    assert has_grad, "Discriminator has no gradients"
    
    # Test gradient penalty
    discriminator.zero_grad()
    real_x = torch.randn(batch_size, max_nodes, num_node_feats)
    real_adj = torch.rand(batch_size, max_nodes, max_nodes)
    
    z = torch.randn(batch_size, latent_dim)
    fake_adj, fake_x = generator(z, temperature=1.0, hard=True)
    
    gp = discriminator.compute_gradient_penalty(
        real_x, real_adj, fake_x.detach(), fake_adj.detach(), torch.device('cpu')
    )
    
    assert gp.shape == (), "Gradient penalty should be scalar"
    assert gp.item() >= 0, "Gradient penalty should be non-negative"
    
    return f"adj:{adj.shape}, x:{x.shape}, gp:{gp.item():.4f}"


# ============================================================================
# Test 5: Model Wrapper (Dense to Sparse)
# ============================================================================

def test_model_wrapper():
    """Test DenseToSparseWrapper with all model types."""
    from models_kgnn import get_model
    from model_wrapper import DenseToSparseWrapper, SimpleDenseGNN
    
    batch_size = 4
    max_nodes = 15
    num_features = 7
    num_classes = 2
    
    # Create dense inputs
    x = torch.randn(batch_size, max_nodes, num_features)
    adj = torch.rand(batch_size, max_nodes, max_nodes)
    adj = (adj + adj.transpose(1, 2)) / 2  # Symmetrize
    adj = (adj > 0.5).float()  # Binarize
    
    results = []
    
    for model_name in ['1gnn', '12gnn', '123gnn']:
        # Create base model
        base_model = get_model(
            model_name,
            input_dim=num_features,
            hidden_dim=32,
            output_dim=num_classes,
            dropout=0.0
        )
        
        # Wrap it
        wrapper = DenseToSparseWrapper(base_model, model_name)
        
        # Forward pass with dense inputs
        out = wrapper(x, adj)
        
        assert out.shape == (batch_size, num_classes), f"{model_name} wrapper: wrong shape {out.shape}"
        results.append(f"{model_name}:OK")
    
    # Also test SimpleDenseGNN
    simple = SimpleDenseGNN(num_features, 32, num_classes, num_layers=2)
    out = simple(x, adj)
    assert out.shape == (batch_size, num_classes), f"SimpleDenseGNN: wrong shape {out.shape}"
    results.append("SimpleDense:OK")
    
    return ", ".join(results)


# ============================================================================
# Test 6: Dynamic Weighting
# ============================================================================

def test_dynamic_weighting():
    """Test dynamic loss weighting scheme."""
    from dynamic_weighting import DynamicWeighting, ConstantWeighting, LinearWeighting
    
    total_iters = 1000
    
    # Test DynamicWeighting
    dw = DynamicWeighting(total_iters, min_lambda=0.0, max_lambda=1.0, p=0.4, k=10.0)
    
    lambdas = [dw.get_lambda(i) for i in range(total_iters)]
    
    # Check bounds
    assert all(0 <= l <= 1 for l in lambdas), "Lambda out of bounds"
    
    # Check monotonic increase (approximately)
    early = np.mean(lambdas[:100])
    late = np.mean(lambdas[-100:])
    assert late > early, f"Lambda should increase: early={early:.3f}, late={late:.3f}"
    
    # Test ConstantWeighting
    cw = ConstantWeighting(0.5)
    assert cw.get_lambda() == 0.5
    
    # Test LinearWeighting
    lw = LinearWeighting(total_iters, min_lambda=0.0, max_lambda=1.0, warmup_fraction=0.2)
    l_early = lw.get_lambda(0)
    l_late = lw.get_lambda(999)
    assert l_late > l_early
    
    return f"early:{early:.3f}, late:{late:.3f}"


# ============================================================================
# Test 7: Metrics and Evaluation
# ============================================================================

def test_metrics():
    """Test explanation metrics computation."""
    from metrics import (
        ExplanationEvaluator, 
        compute_degree_score,
        compute_validation_score,
        compute_granularity,
        is_valid_explanation
    )
    
    # Test individual functions
    d = compute_degree_score(1.0, 1.0, 0.5)
    assert 0 <= d <= 1, f"Degree score out of bounds: {d}"
    
    v = compute_validation_score(0.9, 0.8, 0.7)
    assert 0 <= v <= 1, f"Validation score out of bounds: {v}"
    
    k = compute_granularity(5, 20)
    assert k == 0.75, f"Granularity wrong: {k}"
    
    valid = is_valid_explanation(1.0, 1.0, 0.5, threshold=3.0)
    assert valid == True, "Should be valid"
    
    # Test evaluator
    class_stats = {
        0: {'mean_degree': 1.0, 'std_degree': 0.5, 'avg_nodes': 15.0},
        1: {'mean_degree': 1.2, 'std_degree': 0.4, 'avg_nodes': 17.0}
    }
    
    evaluator = ExplanationEvaluator(class_stats)
    
    # Create test graph
    adj = np.random.rand(15, 15)
    adj = (adj + adj.T) / 2
    adj = (adj > 0.7).astype(np.float32)
    
    x = np.zeros((15, 7))
    x[np.arange(15), np.random.randint(0, 7, 15)] = 1
    
    metrics = evaluator.evaluate_single(adj, x, 0, 0.8, 0.7)
    
    assert 0 <= metrics.validation_score <= 1
    assert metrics.num_nodes >= 0
    assert metrics.num_edges >= 0
    
    # Test batch evaluation
    adjs = np.stack([adj, adj])
    xs = np.stack([x, x])
    probs = np.array([0.8, 0.9])
    sims = np.array([0.7, 0.8])
    
    batch_metrics = evaluator.evaluate_batch(adjs, xs, 0, probs, sims)
    assert len(batch_metrics) == 2
    
    # Test best explanations
    best = evaluator.get_best_explanations(batch_metrics, top_k=1, valid_only=False)
    assert len(best) <= 1
    
    # Test summary stats
    summary = evaluator.compute_summary_stats(batch_metrics)
    assert 'total_generated' in summary
    assert summary['total_generated'] == 2
    
    return f"v={metrics.validation_score:.3f}, nodes={metrics.num_nodes}"


# ============================================================================
# Test 8: GIN-Graph Trainer Short Run
# ============================================================================

def test_gin_graph_trainer():
    """Test GIN-Graph trainer initialization and short training run."""
    from data_loader import load_mutag, get_class_statistics, get_class_subset
    from models_kgnn import get_model
    from train_gin_graph import GINGraphTrainer
    from config import GINGraphConfig, DataConfig
    
    device = torch.device('cpu')
    
    # Load data
    dataset = load_mutag('./data')
    class_stats = get_class_statistics(dataset)
    for label in [0, 1]:
        nodes = [d.num_nodes for d in dataset if d.y.item() == label]
        class_stats[label]['avg_nodes'] = np.mean(nodes)
    
    # Create a simple pre-trained model (just initialized, not actually trained)
    pretrained_gnn = get_model(
        '1gnn',
        input_dim=7,
        hidden_dim=32,
        output_dim=2,
        dropout=0.0
    ).to(device)
    
    # Create trainer config with minimal settings
    gin_config = GINGraphConfig(
        latent_dim=16,
        hidden_dim=32,
        learning_rate=0.001,
        epochs=2,
        batch_size=16,
        n_critic=1,
        gp_lambda=10.0
    )
    
    data_config = DataConfig(max_nodes=28, num_node_features=7)
    
    # Create trainer
    trainer = GINGraphTrainer(
        pretrained_gnn=pretrained_gnn,
        model_type='1gnn',
        target_class=0,
        config=gin_config,
        data_config=data_config,
        device=device,
        class_stats=class_stats
    )
    
    # Get target class subset
    target_dataset = get_class_subset(dataset, 0)
    
    # Train for just 2 epochs (very short)
    trainer.train(target_dataset, epochs=2, verbose=False, log_interval=1)
    
    # Verify history was recorded
    assert len(trainer.history['d_loss']) > 0, "No discriminator loss recorded"
    assert len(trainer.history['g_loss']) > 0, "No generator loss recorded"
    assert len(trainer.history['lambda']) > 0, "No lambda recorded"
    
    # Generate explanations
    adjs, xs, metrics = trainer.generate_explanations(num_samples=5, temperature=0.1)
    
    assert adjs.shape[0] == 5, f"Wrong number of explanations: {adjs.shape[0]}"
    assert xs.shape[0] == 5
    assert len(metrics) == 5
    
    return f"generated 5 explanations, d_loss={trainer.history['d_loss'][-1]:.4f}"


# ============================================================================
# Test 9: Full Mini Pipeline
# ============================================================================

def test_full_mini_pipeline():
    """Test the complete pipeline: load → train kGNN → train GIN-Graph → evaluate."""
    from data_loader import load_mutag, create_data_loaders, get_class_statistics, get_class_subset
    from models_kgnn import get_model
    from train_gin_graph import GINGraphTrainer
    from config import GINGraphConfig, DataConfig
    from metrics import ExplanationEvaluator
    
    device = torch.device('cpu')
    
    # Step 1: Load data
    dataset = load_mutag('./data')
    train_loader, test_loader, _, _ = create_data_loaders(dataset, batch_size=32, seed=42)
    
    class_stats = get_class_statistics(dataset)
    for label in [0, 1]:
        nodes = [d.num_nodes for d in dataset if d.y.item() == label]
        class_stats[label]['avg_nodes'] = np.mean(nodes)
    
    # Step 2: Train k-GNN (just 1 epoch)
    model = get_model('1gnn', input_dim=7, hidden_dim=32, output_dim=2, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
    
    # Evaluate k-GNN
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    kgnn_acc = correct / total
    
    # Step 3: Train GIN-Graph (just 1 epoch)
    gin_config = GINGraphConfig(
        latent_dim=16,
        hidden_dim=32,
        epochs=1,
        batch_size=16
    )
    data_config = DataConfig(max_nodes=28, num_node_features=7)
    
    trainer = GINGraphTrainer(
        pretrained_gnn=model,
        model_type='1gnn',
        target_class=0,
        config=gin_config,
        data_config=data_config,
        device=device,
        class_stats=class_stats
    )
    
    target_dataset = get_class_subset(dataset, 0)
    trainer.train(target_dataset, epochs=1, verbose=False)
    
    # Step 4: Generate and evaluate explanations
    adjs, xs, metrics = trainer.generate_explanations(num_samples=10)
    
    summary = trainer.evaluator.compute_summary_stats(metrics)
    
    # Verify everything worked
    assert summary['total_generated'] == 10
    assert 'mean_validation_score' in summary
    
    return f"kGNN acc={kgnn_acc:.2f}, gen={summary['total_generated']}, val={summary['mean_validation_score']:.3f}"


# ============================================================================
# Test 10: Config Module
# ============================================================================

def test_config():
    """Test configuration classes."""
    from config import (
        DataConfig, 
        KGNNConfig, 
        GINGraphConfig, 
        ExperimentConfig,
        ATOM_LABELS,
        ATOM_COLORS,
        CLASS_NAMES
    )
    
    # Test DataConfig
    dc = DataConfig()
    assert dc.name == "MUTAG"
    assert dc.max_nodes == 28
    assert dc.num_node_features == 7
    
    # Test KGNNConfig
    kc = KGNNConfig()
    assert kc.hidden_dim == 64
    assert kc.epochs == 100
    
    # Test GINGraphConfig
    gc = GINGraphConfig()
    assert gc.latent_dim == 32
    assert gc.gp_lambda == 10.0
    
    # Test ExperimentConfig
    ec = ExperimentConfig()
    assert '1gnn' in ec.models
    device = ec.get_device()
    assert device.type in ['cpu', 'cuda']
    
    # Test constants
    assert len(ATOM_LABELS) == 7
    assert len(ATOM_COLORS) >= 7
    assert 0 in CLASS_NAMES and 1 in CLASS_NAMES
    
    return f"configs OK, device={device.type}"


# ============================================================================
# Test 11: Visualization (no display, just check it doesn't crash)
# ============================================================================

def test_visualization():
    """Test visualization utilities don't crash."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    from visualize import (
        adj_to_networkx,
        plot_explanation_graph,
        plot_explanation_grid,
        plot_training_history,
        plot_validation_scores_distribution
    )
    from metrics import ExplanationMetrics
    import matplotlib.pyplot as plt
    
    # Create test data
    num_samples = 4
    max_nodes = 10
    num_features = 7
    
    adjs = np.random.rand(num_samples, max_nodes, max_nodes)
    adjs = (adjs + adjs.transpose(0, 2, 1)) / 2
    adjs = (adjs > 0.7).astype(np.float32)
    
    xs = np.zeros((num_samples, max_nodes, num_features))
    for i in range(num_samples):
        for j in range(max_nodes):
            xs[i, j, np.random.randint(num_features)] = 1
    
    # Test adj_to_networkx
    G, labels, colors = adj_to_networkx(adjs[0], xs[0])
    assert G is not None
    
    # Test plot_explanation_graph
    fig, ax = plt.subplots()
    plot_explanation_graph(adjs[0], xs[0], ax=ax, title="Test")
    plt.close(fig)
    
    # Test plot_explanation_grid
    fig = plot_explanation_grid(adjs, xs, num_cols=2)
    plt.close(fig)
    
    # Test plot_training_history
    history = {
        'd_loss': [1.0, 0.9, 0.8, 0.7],
        'g_loss': [0.5, 0.6, 0.7, 0.8],
        'gan_loss': [0.3, 0.4, 0.5, 0.6],
        'gnn_loss': [0.2, 0.3, 0.4, 0.5],
        'lambda': [0.1, 0.3, 0.5, 0.7],
        'pred_prob': [0.5, 0.6, 0.7, 0.8]
    }
    fig = plot_training_history(history)
    plt.close(fig)
    
    # Test plot_validation_scores_distribution
    metrics = [
        ExplanationMetrics(
            prediction_probability=0.8,
            embedding_similarity=0.7,
            degree_score=0.9,
            validation_score=0.8,
            average_degree=1.0,
            num_nodes=10,
            num_edges=15,
            is_valid=True,
            granularity=0.5
        )
        for _ in range(10)
    ]
    fig = plot_validation_scores_distribution(metrics)
    plt.close(fig)
    
    return "all visualizations OK"


# ============================================================================
# Test 12: k-set Building Functions
# ============================================================================

def test_kset_building():
    """Test the k-set building functions used in 2-GNN and 3-GNN."""
    from models_kgnn import build_local_adj, build_2set_edges, build_3set_edges
    
    device = torch.device('cpu')
    
    # Create a simple graph: triangle (0-1-2) with extra node (3) connected to 0
    # Edges: (0,1), (1,2), (0,2), (0,3)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 2, 0, 3],
        [1, 0, 2, 1, 2, 0, 3, 0]
    ], dtype=torch.long, device=device)
    
    node_indices = torch.tensor([0, 1, 2, 3], device=device)
    
    # Test build_local_adj
    adj = build_local_adj(edge_index, node_indices, 4, device)
    
    assert adj.shape == (4, 4)
    assert adj[0, 1] == 1 and adj[1, 0] == 1  # Edge 0-1
    assert adj[0, 2] == 1 and adj[2, 0] == 1  # Edge 0-2
    assert adj[1, 2] == 1 and adj[2, 1] == 1  # Edge 1-2
    assert adj[0, 3] == 1 and adj[3, 0] == 1  # Edge 0-3
    assert adj[1, 3] == 0  # No edge 1-3
    
    # Test build_2set_edges
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], device=device)
    edges_2set = build_2set_edges(pairs, adj, device)
    
    # Should have some edges between 2-sets
    assert edges_2set.shape[0] == 2
    
    # Test build_3set_edges
    triplets = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], device=device)
    edges_3set = build_3set_edges(triplets, adj, device)
    
    assert edges_3set.shape[0] == 2
    
    return f"2set edges: {edges_2set.shape[1]}, 3set edges: {edges_3set.shape[1]}"


# ============================================================================
# Test 13: Gradient Flow Through Full GIN-Graph Pipeline
# ============================================================================

def test_gradient_flow():
    """Test that gradients flow correctly through the entire GIN-Graph pipeline."""
    from gin_generator import GINGenerator, GINDiscriminator
    from models_kgnn import get_model
    from model_wrapper import DenseToSparseWrapper
    
    device = torch.device('cpu')
    batch_size = 4
    latent_dim = 16
    max_nodes = 15
    num_features = 7
    
    # Create components
    generator = GINGenerator(latent_dim, max_nodes, num_features, hidden_dim=32)
    discriminator = GINDiscriminator(max_nodes, num_features, hidden_dim=32)
    
    base_gnn = get_model('1gnn', input_dim=num_features, hidden_dim=32, output_dim=2)
    wrapped_gnn = DenseToSparseWrapper(base_gnn, '1gnn')
    
    # Create optimizers
    opt_G = torch.optim.Adam(generator.parameters(), lr=0.001)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Simulate one training step
    
    # 1. Train discriminator
    opt_D.zero_grad()
    
    # Real data (fake for testing)
    real_x = torch.randn(batch_size, max_nodes, num_features)
    real_adj = torch.rand(batch_size, max_nodes, max_nodes)
    real_adj = (real_adj > 0.5).float()
    
    real_scores = discriminator(real_x, real_adj)
    
    # Fake data
    z = torch.randn(batch_size, latent_dim)
    fake_adj, fake_x = generator(z, temperature=1.0)
    fake_scores = discriminator(fake_x.detach(), fake_adj.detach())
    
    d_loss = fake_scores.mean() - real_scores.mean()
    d_loss.backward()
    
    # Check discriminator gradients
    d_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                     for p in discriminator.parameters())
    assert d_has_grad, "Discriminator has no gradients"
    
    opt_D.step()
    
    # 2. Train generator
    opt_G.zero_grad()
    
    z = torch.randn(batch_size, latent_dim)
    fake_adj, fake_x = generator(z, temperature=1.0)
    
    # GAN loss
    gan_scores = discriminator(fake_x, fake_adj)
    gan_loss = -gan_scores.mean()
    
    # GNN loss
    gnn_logits = wrapped_gnn(fake_x, fake_adj)
    target_labels = torch.zeros(batch_size, dtype=torch.long)
    gnn_loss = F.cross_entropy(gnn_logits, target_labels)
    
    # Combined loss
    total_loss = 0.5 * gan_loss + 0.5 * gnn_loss
    total_loss.backward()
    
    # Check generator gradients
    g_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                     for p in generator.parameters())
    assert g_has_grad, "Generator has no gradients"
    
    opt_G.step()
    
    return f"d_loss={d_loss.item():.4f}, g_loss={total_loss.item():.4f}"


# ============================================================================
# Test 14: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    from metrics import ExplanationEvaluator, compute_granularity
    from gin_generator import GINGenerator
    
    # Test empty graph evaluation
    class_stats = {
        0: {'mean_degree': 1.0, 'std_degree': 0.5, 'avg_nodes': 15.0}
    }
    evaluator = ExplanationEvaluator(class_stats)
    
    # Empty adjacency matrix (no edges)
    empty_adj = np.zeros((10, 10), dtype=np.float32)
    x = np.zeros((10, 7), dtype=np.float32)
    x[:, 0] = 1  # All same atom type
    
    metrics = evaluator.evaluate_single(empty_adj, x, 0, 0.5, 0.5)
    assert metrics.num_edges == 0
    assert metrics.is_valid == False  # Empty graph should be invalid
    
    # Test granularity edge cases
    assert compute_granularity(0, 10) == 1.0  # No nodes
    assert compute_granularity(10, 10) == 0.0  # Same as average
    assert compute_granularity(20, 10) == 0.0  # Larger than average (clamped)
    assert compute_granularity(5, 0) == 0.0  # Zero average (edge case)
    
    # Test generator with batch size 1
    gen = GINGenerator(latent_dim=16, max_nodes=10, num_node_feats=7, hidden_dim=32)
    z = torch.randn(1, 16)
    adj, x = gen(z)
    assert adj.shape == (1, 10, 10)
    assert x.shape == (1, 10, 7)
    
    return "all edge cases handled"


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all integration tests."""
    print("=" * 70)
    print("k-GNN + GIN-Graph Integration Tests")
    print("=" * 70)
    print()
    
    tests = [
        (test_data_loading, "Data Loading"),
        (test_kgnn_forward_pass, "k-GNN Forward Pass"),
        (test_kgnn_short_training, "k-GNN Short Training"),
        (test_gin_generator_discriminator, "GIN Generator/Discriminator"),
        (test_model_wrapper, "Model Wrapper"),
        (test_dynamic_weighting, "Dynamic Weighting"),
        (test_metrics, "Metrics & Evaluation"),
        (test_gin_graph_trainer, "GIN-Graph Trainer"),
        (test_full_mini_pipeline, "Full Mini Pipeline"),
        (test_config, "Config Module"),
        (test_visualization, "Visualization"),
        (test_kset_building, "k-Set Building"),
        (test_gradient_flow, "Gradient Flow"),
        (test_edge_cases, "Edge Cases"),
    ]
    
    results = []
    total_time = 0
    
    for test_func, name in tests:
        print(f"Running: {name}...", end=" ", flush=True)
        result = run_test(test_func, name)
        results.append(result)
        total_time += result.duration
        
        if result.passed:
            print(f"✓ PASS ({result.duration:.2f}s) - {result.message}")
        else:
            print(f"✗ FAIL ({result.duration:.2f}s)")
            print(f"    Error: {result.message[:200]}...")
    
    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {r.name:30} {status:8} ({r.duration:.2f}s)")
    
    print()
    print(f"Total: {passed} passed, {failed} failed ({total_time:.2f}s)")
    print()
    
    if failed == 0:
        print("🎉 All tests passed! Your code is ready for full training.")
        print()
        print("Next steps:")
        print("  1. Train k-GNN models: python train_kgnn.py --all --epochs 100")
        print("  2. Run full experiment: python run_experiment.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the errors above before training.")
        # Print details of failed tests
        print()
        print("Failed test details:")
        for r in results:
            if not r.passed:
                print(f"\n--- {r.name} ---")
                print(r.message)
        return 1


if __name__ == "__main__":
    sys.exit(main())