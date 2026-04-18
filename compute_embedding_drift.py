#!/usr/bin/env python3
"""Compute normalized embedding drift from saved generated datasets.

This is a post-hoc analysis script: it reuses the saved generated `.npz`
datasets under `results/{dataset}/comparison/` and existing checkpoints, and
does not retrain or regenerate any graphs.
"""

import argparse
import json
import os

import numpy as np
import torch

from compare_datasets import (
    compute_generated_embeddings,
    compute_real_embeddings,
    load_all_models,
    resolve_device,
)
from config import get_class_name
from model_wrapper import DenseToSparseWrapper


def compute_dataset_drift(
    dataset_name: str,
    checkpoint_dir: str,
    gin_checkpoint_dir: str,
    results_dir: str,
    device: torch.device,
):
    """Compute normalized centroid drift for one dataset."""
    kgnns, _, train_dataset, _, _ = load_all_models(
        dataset_name, checkpoint_dir, gin_checkpoint_dir, device
    )

    dataset_results = {
        "dataset": dataset_name,
        "models": {},
    }

    for model_name, kgnn in kgnns.items():
        real_emb, real_labels = compute_real_embeddings(kgnn, train_dataset, device)

        real_centroids = {}
        for class_idx in [0, 1]:
            class_mask = real_labels.flatten() == class_idx
            real_centroids[class_idx] = real_emb[class_mask].mean(axis=0)

        between_class_distance = float(
            np.linalg.norm(real_centroids[0] - real_centroids[1])
        )

        wrapper = DenseToSparseWrapper(kgnn, model_name).to(device)
        wrapper.eval()

        generated_path = os.path.join(
            results_dir, dataset_name, "comparison", f"{model_name}_generated.npz"
        )
        generated = np.load(generated_path)

        model_results = {
            "between_class_distance": between_class_distance,
            "classes": {},
        }

        for class_idx in [0, 1]:
            gen_emb = compute_generated_embeddings(
                wrapper,
                generated[f"adjs_class{class_idx}"],
                generated[f"xs_class{class_idx}"],
                device,
            )
            gen_centroid = gen_emb.mean(axis=0)
            drift = float(np.linalg.norm(gen_centroid - real_centroids[class_idx]))
            normalized_drift = (
                drift / between_class_distance if between_class_distance > 0 else None
            )

            model_results["classes"][str(class_idx)] = {
                "class_name": get_class_name(class_idx, dataset_name),
                "drift": drift,
                "normalized_drift": normalized_drift,
            }

        dataset_results["models"][model_name] = model_results

    return dataset_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalized embedding drift from saved generated datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["mutag", "proteins", "all"],
        default="all",
        help="Dataset to analyze",
    )
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--gin_checkpoint_dir", default="./gin_checkpoints")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Use stdout when omitted.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    datasets = ["mutag", "proteins"] if args.dataset == "all" else [args.dataset]

    output = {
        dataset_name: compute_dataset_drift(
            dataset_name,
            args.checkpoint_dir,
            args.gin_checkpoint_dir,
            args.results_dir,
            device,
        )
        for dataset_name in datasets
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
