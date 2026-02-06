"""
test_setup.py
Quick test to verify the project setup is working correctly.

Run: python test_setup.py
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"  ✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch Geometric: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
        return False
    
    try:
        import networkx
        print(f"  ✓ NetworkX {networkx.__version__}")
    except ImportError as e:
        print(f"  ✗ NetworkX: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    return True


def test_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    modules = [
        'config',
        'data_loader',
        'models_kgnn',
        'gin_generator',
        'model_wrapper',
        'dynamic_weighting',
        'metrics',
        'visualize',
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            all_ok = False
    
    return all_ok


def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    
    try:
        from data_loader import load_mutag, get_dataset_statistics
        
        dataset = load_mutag()
        stats = get_dataset_statistics(dataset)
        
        print(f"  ✓ Loaded MUTAG: {stats['num_graphs']} graphs")
        print(f"  ✓ Node features: {stats['num_node_features']}")
        print(f"  ✓ Classes: {stats['num_classes']}")
        return True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from models_kgnn import get_model, count_parameters
        
        for model_name in ['1gnn', '12gnn', '123gnn']:
            model = get_model(model_name, input_dim=7, hidden_dim=64, output_dim=2)
            params = count_parameters(model)
            print(f"  ✓ {model_name.upper()}: {params:,} parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_generator():
    """Test GIN-Graph generator."""
    print("\nTesting GIN-Graph generator...")
    
    try:
        import torch
        from gin_generator import GINGenerator, GINDiscriminator
        
        generator = GINGenerator(latent_dim=32, max_nodes=28, num_node_feats=7)
        discriminator = GINDiscriminator(max_nodes=28, num_node_feats=7)
        
        # Test forward pass
        z = torch.randn(4, 32)
        adj, x = generator(z)
        
        print(f"  ✓ Generator output: adj {adj.shape}, x {x.shape}")
        
        score = discriminator(x, adj)
        print(f"  ✓ Discriminator output: {score.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Generator test failed: {e}")
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")
    
    try:
        import numpy as np
        from metrics import ExplanationEvaluator
        
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
        
        print(f"  ✓ Validation score: {metrics.validation_score:.4f}")
        print(f"  ✓ Is valid: {metrics.is_valid}")
        
        return True
    except Exception as e:
        print(f"  ✗ Metrics test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("k-GNN Interpretation Project - Setup Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Dependencies", test_imports()))
    results.append(("Project Modules", test_modules()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Generator", test_generator()))
    results.append(("Metrics", test_metrics()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("  1. Train k-GNN models: python train_kgnn.py --all")
        print("  2. Train GIN-Graph: python train_gin_graph.py --model 1gnn --target_class 0")
        print("  3. Analyze results: python research.py gin --model 1gnn --target_class 0")
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
