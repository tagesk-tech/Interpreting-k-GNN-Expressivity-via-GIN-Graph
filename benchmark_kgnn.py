"""
benchmark_kgnn.py
Benchmark sparse and dense 2-GNN paths with and without optimizations.

Compares:
  1. Sparse edge building: original (sampled) vs original (all-pairs) vs numba (all-pairs)
  2. Dense forward pass: plain vs torch.compile

Usage:
    python benchmark_kgnn.py --dataset mutag
    python benchmark_kgnn.py --dataset proteins
    python benchmark_kgnn.py --all
"""

import torch
import time
import argparse
import numpy as np

from models_kgnn import (
    build_2set_edges, build_2set_edges_fast, NUMBA_AVAILABLE,
    build_local_adj, _sample_ksets, get_model
)
from model_wrapper import (
    DenseToSparseWrapper, _dense_2gnn_layer_fn,
    _TORCH_COMPILE_AVAILABLE
)
from data_loader import load_dataset, create_data_loaders, AVAILABLE_DATASETS
from config import DataConfig


def benchmark_sparse_edge_building(dataset_name, num_graphs=20, num_warmup=3):
    """Benchmark build_2set_edges vs build_2set_edges_fast on real graphs."""
    print(f"\n{'='*60}")
    print(f"Sparse 2-Set Edge Building — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  Numba available: {NUMBA_AVAILABLE}")

    dataset = load_dataset(dataset_name)
    _, _, train_ds, _ = create_data_loaders(dataset, batch_size=1, seed=42)

    device = torch.device('cpu')

    results = {
        'original_sampled': [],
        'original_all': [],
        'numba_all': [],
    }
    graph_info = []

    count = 0
    for data in train_ds:
        if count >= num_graphs + num_warmup:
            break

        n = data.num_nodes
        if n < 2:
            continue

        # Build local adj (single graph, batch=0)
        adj = build_local_adj(
            data.edge_index,
            torch.arange(n, device=device),
            n, device
        )

        # All pairs
        pairs_all = torch.combinations(torch.arange(n, device=device), r=2)
        # Sampled pairs (legacy limit)
        pairs_sampled = _sample_ksets(n, 2, 5000, device)

        is_warmup = count < num_warmup
        count += 1

        if is_warmup:
            # Warm up numba JIT + torch ops
            build_2set_edges(pairs_sampled, adj, device)
            if NUMBA_AVAILABLE:
                build_2set_edges_fast(pairs_all, adj, device)
            continue

        # Benchmark original with sampling (legacy)
        t0 = time.perf_counter()
        e1 = build_2set_edges(pairs_sampled, adj, device)
        t_sampled = time.perf_counter() - t0

        # Benchmark original with all pairs
        t0 = time.perf_counter()
        e2 = build_2set_edges(pairs_all, adj, device)
        t_all = time.perf_counter() - t0

        # Benchmark numba with all pairs
        if NUMBA_AVAILABLE:
            t0 = time.perf_counter()
            e3 = build_2set_edges_fast(pairs_all, adj, device)
            t_numba = time.perf_counter() - t0
        else:
            t_numba = float('nan')
            e3 = e2

        results['original_sampled'].append(t_sampled)
        results['original_all'].append(t_all)
        results['numba_all'].append(t_numba)
        graph_info.append({
            'n': n,
            'pairs_sampled': pairs_sampled.size(0),
            'pairs_all': pairs_all.size(0),
            'edges_sampled': e1.size(1),
            'edges_all': e2.size(1),
            'edges_numba': e3.size(1),
        })

        print(f"  Graph n={n:4d} | sampled={pairs_sampled.size(0):6d} pairs "
              f"({t_sampled:.4f}s, {e1.size(1):6d} edges) | "
              f"all={pairs_all.size(0):6d} pairs "
              f"(orig {t_all:.4f}s, numba {t_numba:.4f}s, {e2.size(1):6d} edges)")

    # Verify correctness: numba should produce same edges as original
    if NUMBA_AVAILABLE and graph_info:
        edges_match = all(
            g['edges_all'] == g['edges_numba'] for g in graph_info
        )
        print(f"\n  Correctness check: numba edges match original = {edges_match}")

    _print_timing_summary(results)
    return results


def benchmark_dense_2gnn(dataset_name, batch_size=64, num_warmup=5, num_runs=30):
    """Benchmark _dense_2gnn with and without torch.compile."""
    print(f"\n{'='*60}")
    print(f"Dense 2-GNN Forward Pass — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  torch.compile available: {_TORCH_COMPILE_AVAILABLE}")

    data_config = DataConfig.from_dataset(dataset_name)
    N = data_config.gin_max_nodes
    D = data_config.num_node_features
    hidden_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  N={N}, D={D}, B={batch_size}, hidden={hidden_dim}")

    # Create base model
    base = get_model('12gnn', input_dim=D, hidden_dim=hidden_dim, output_dim=2)

    # Plain wrapper (no compile)
    wrapper_plain = DenseToSparseWrapper(base, '12gnn', use_compile=False).to(device)

    # Compiled wrapper
    wrapper_compiled = DenseToSparseWrapper(base, '12gnn', use_compile=True).to(device)

    # Random input
    x = torch.randn(batch_size, N, D, device=device)
    adj = (torch.rand(batch_size, N, N, device=device) > 0.7).float()
    adj = (adj + adj.transpose(1, 2)).clamp(max=1.0)

    results = {'plain': [], 'compiled': []}

    # Warmup (especially important for torch.compile)
    print(f"\n  Warming up ({num_warmup} iterations)...")
    compile_works = False
    for i in range(num_warmup):
        with torch.no_grad():
            wrapper_plain(x, adj)
            if _TORCH_COMPILE_AVAILABLE and not compile_works and i == 0:
                try:
                    wrapper_compiled(x, adj)
                    compile_works = True
                except Exception as e:
                    print(f"  torch.compile failed (likely missing build deps): {type(e).__name__}")
                    print(f"  Continuing with plain-only benchmark")
            elif compile_works:
                wrapper_compiled(x, adj)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark plain
    print(f"  Benchmarking plain ({num_runs} runs)...")
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            wrapper_plain(x, adj)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        results['plain'].append(time.perf_counter() - t0)

    # Benchmark compiled
    if compile_works:
        print(f"  Benchmarking compiled ({num_runs} runs)...")
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                wrapper_compiled(x, adj)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            results['compiled'].append(time.perf_counter() - t0)

        # Correctness check
        with torch.no_grad():
            out_plain = wrapper_plain(x, adj)
            out_compiled = wrapper_compiled(x, adj)
        max_diff = (out_plain - out_compiled).abs().max().item()
        print(f"\n  Correctness check: max diff = {max_diff:.2e} "
              f"({'OK' if max_diff < 1e-4 else 'MISMATCH'})")
    else:
        print("  Skipping compiled benchmark (torch.compile not available or failed)")

    _print_timing_summary(results)
    return results


def _print_timing_summary(results):
    """Print summary table for benchmark results."""
    print(f"\n  {'Method':<25s} {'Mean':>10s} {'Std':>10s} {'Median':>10s} {'Min':>10s}")
    print(f"  {'-'*65}")
    for key, times in results.items():
        if times and not any(np.isnan(t) for t in times):
            print(f"  {key:<25s} {np.mean(times):>10.5f}s {np.std(times):>10.5f}s "
                  f"{np.median(times):>10.5f}s {np.min(times):>10.5f}s")
        elif times:
            print(f"  {key:<25s} {'N/A (numba not available)':>40s}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark k-GNN optimizations')
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=AVAILABLE_DATASETS,
                        help='Dataset to benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Benchmark all datasets')
    parser.add_argument('--num_graphs', type=int, default=20,
                        help='Number of graphs for sparse benchmark')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for dense benchmark')
    parser.add_argument('--num_runs', type=int, default=30,
                        help='Number of timed runs for dense benchmark')
    args = parser.parse_args()

    datasets = AVAILABLE_DATASETS if args.all else [args.dataset]

    for ds in datasets:
        print(f"\n{'#'*60}")
        print(f"# Benchmarking {ds.upper()}")
        print(f"{'#'*60}")

        benchmark_sparse_edge_building(ds, num_graphs=args.num_graphs)
        benchmark_dense_2gnn(ds, batch_size=args.batch_size, num_runs=args.num_runs)

    print(f"\n{'='*60}")
    print("Benchmark complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
