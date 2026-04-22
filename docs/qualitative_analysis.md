# Qualitative Analysis Guide — Four Dataset/Class Combinations

This guide walks through the qualitative analysis across the four combinations in the study (MUTAG × {class 0, class 1} and PROTEINS × {class 0, class 1}), for both the standard **1-GNN** and the hierarchical **1-2-GNN**. Each section points to the exact figures in `results/` that support the observation.

All figures referenced here are already generated and committed under `results/`. Numeric values come from `report.json` files in the same directories.

---

## How to use this guide

For each of the four combinations, four lenses are used:

1. **Generated-graph visualisations** — `explanations.png` (top-10 graphs with per-graph validation score `v` and node count `n`)
2. **Score distributions** — `metrics.png` (validation score, prediction probability, degree score, validity)
3. **Structural fidelity** — `structural_*.png` (generated vs real: degree histogram, atom/secondary-structure proportions, graph-size histogram, average-degree boxplot)
4. **Cross-model behaviour** — `cross_classification.png` and `embedding_t-sne_*.png` (does the *other* GNN agree, and where do the generated graphs sit in embedding space)

The qualitative claim each figure supports is stated up front; the supporting numbers follow.

---

## MUTAG — Class 0 (Mutagen)

**Real dataset:** 125 graphs, mean ~14 atoms, mean degree 2.09, atom composition ≈ 64% C / 19% O / 13% N / 3% Cl. Real mutagens are small aromatic molecules dominated by C/N/O rings with occasional halogens.

### 1-GNN generator

**Images:**
- [`results/mutag/1gnn_class0/figures/explanations.png`](../results/mutag/1gnn_class0/figures/explanations.png) — top 10 generated mutagens
- [`results/mutag/1gnn_class0/figures/metrics.png`](../results/mutag/1gnn_class0/figures/metrics.png) — score distributions
- [`results/mutag/comparison/figures/structural_1gnn_class0.png`](../results/mutag/comparison/figures/structural_1gnn_class0.png) — gen vs real distributions

**What the figures show:**
- Generated graphs have ~24 nodes (the size regulariser pushes toward `gen_size=28`), with cluttered C-N-O clusters and scattered F/Cl.
- Atom distribution tracks the real data but over-represents halogens (F at 6.2% vs 0.9% real; Cl at 8.6% vs 3.0% real).
- Validity 36%, mean validation score 0.37, prediction probability 1.0, degree score 0.26 — the model learns "mutagen-like" atom clusters but the graph topology is fragmented.

### 1-2-GNN generator

**Images:**
- [`results/mutag/12gnn_class0/figures/explanations.png`](../results/mutag/12gnn_class0/figures/explanations.png)
- [`results/mutag/12gnn_class0/figures/metrics.png`](../results/mutag/12gnn_class0/figures/metrics.png)
- [`results/mutag/comparison/figures/structural_12gnn_class0.png`](../results/mutag/comparison/figures/structural_12gnn_class0.png)

**What the figures show:**
- Similar size (~23 nodes), but atom distribution is heavily skewed: only 46% C, with a huge 27% F and 11% Br (effectively absent in real data).
- Validity 41%, mean score 0.43 — slightly better than 1-GNN; graphs look less "molecular" but cluster halogens.

### Cross-model check

**Image:** [`results/mutag/comparison/figures/cross_classification.png`](../results/mutag/comparison/figures/cross_classification.png)

**Observation:** Both generators achieve **≥90% cross-model agreement** on class 0.
- 1-GNN graphs → 90% classed as mutagen by 1-2-GNN
- 1-2-GNN graphs → 100% classed as mutagen by 1-GNN

**Defensible claim:** The two architectures agree on what makes a mutagen. The halogenated-aromatic core signal is visible to both.

---

## MUTAG — Class 1 (Non-Mutagen)

**Real dataset:** 63 graphs, ~20 atoms, mean degree 2.24, 73% C with near-zero halogens. Real non-mutagens are carbon-rich chains/rings without reactive halogens.

### 1-GNN generator

**Images:**
- [`results/mutag/1gnn_class1/figures/explanations.png`](../results/mutag/1gnn_class1/figures/explanations.png)
- [`results/mutag/1gnn_class1/figures/metrics.png`](../results/mutag/1gnn_class1/figures/metrics.png)
- [`results/mutag/comparison/figures/structural_1gnn_class1.png`](../results/mutag/comparison/figures/structural_1gnn_class1.png)

**What the figures show:**
- ~27-node graphs, 63% C, 19% O, but **7.7% Iodine** — entirely absent in the real dataset.
- Validity 78%, score 0.68, embedding similarity 0.94 — strong class match to centroid.

### 1-2-GNN generator

**Images:**
- [`results/mutag/12gnn_class1/figures/explanations.png`](../results/mutag/12gnn_class1/figures/explanations.png)
- [`results/mutag/12gnn_class1/figures/metrics.png`](../results/mutag/12gnn_class1/figures/metrics.png)
- [`results/mutag/comparison/figures/structural_12gnn_class1.png`](../results/mutag/comparison/figures/structural_12gnn_class1.png)

**What the figures show:**
- Nearly pure C-dominated graph: **80% C, 17% O, zero halogens** — the closest match to real class 1 composition in the whole MUTAG experiment.
- Validity 82%, score 0.74 (highest mean validation score on MUTAG).
- Size-distribution KS statistic 0.59 — markedly better than 1-GNN's 0.89.

### Cross-model check — the key divergence

**Image:** [`results/mutag/comparison/figures/cross_classification.png`](../results/mutag/comparison/figures/cross_classification.png)

**Observation:** 1-2-GNN's non-mutagens collapse to **12% cross-model agreement** — the 1-GNN classifies 88% of them as mutagens.

**Defensible claim:** This is the central result of the MUTAG experiment. The 1-2-GNN has learned a decision surface for "non-mutagen" that depends on **pairwise substructure** invisible to the 1-GNN's node-averaging. The generated graphs are structurally *more authentic* (80% C matches real 73% better than anything 1-GNN produces) yet fall on the wrong side of 1-GNN's boundary.

---

## PROTEINS — Class 0 (Non-Enzyme)

**Real dataset:** 663 graphs, highly variable size (median ~49, tail up to 620 nodes), mean degree 3.80, ~50% Helix / 46% Sheet / 3% Coil.

### 1-GNN generator

**Images:**
- [`results/proteins/1gnn_class0/figures/explanations.png`](../results/proteins/1gnn_class0/figures/explanations.png)
- [`results/proteins/1gnn_class0/figures/metrics.png`](../results/proteins/1gnn_class0/figures/metrics.png)
- [`results/proteins/comparison/figures/structural_1gnn_class0.png`](../results/proteins/comparison/figures/structural_1gnn_class0.png)

**What the figures show:**
- **100% validity**, mean validation score **0.99** — the strongest result in the whole study.
- Size ~47 nodes, atom distribution 54% Helix / 46% Sheet (no Coil), degree 3.81 — matches class 0 almost exactly.
- Visualisations show clear Helix-rich modules with tight clusters.

### 1-2-GNN generator

**Images:**
- [`results/proteins/12gnn_class0/figures/explanations.png`](../results/proteins/12gnn_class0/figures/explanations.png)
- [`results/proteins/12gnn_class0/figures/metrics.png`](../results/proteins/12gnn_class0/figures/metrics.png)
- [`results/proteins/comparison/figures/structural_12gnn_class0.png`](../results/proteins/comparison/figures/structural_12gnn_class0.png)

**What the figures show:**
- 50-node graphs pinned against `gen_size=50`, composition 51% H / 47% S / 2% C — actually closer to real than 1-GNN (includes the small Coil fraction).
- Validity 100%, mean score 0.76, prediction probability 0.56 — the model is *less* confident than the stronger atom match would suggest.

### Cross-model check — the mirror divergence

**Image:** [`results/proteins/comparison/figures/cross_classification.png`](../results/proteins/comparison/figures/cross_classification.png)

**Observation:** 1-GNN non-enzyme graphs collapse to **9% cross-model agreement** — the 1-2-GNN rejects 91% of them as enzymes.

**Defensible claim:** The mirror image of MUTAG class 1. 1-GNN produces graphs whose *marginal* statistics match real class 0 (H/S ratio, degree, size), yet the 1-2-GNN sees something wrong — presumably a pairwise Helix/Sheet neighbourhood distribution that the 1-GNN has no way to control during generation.

---

## PROTEINS — Class 1 (Enzyme)

**Real dataset:** 450 graphs, ~24 atoms, mean degree 3.64, **56% Sheet / 41% Helix / 3% Coil** — the secondary-structure ratio is inverted relative to class 0 (Sheets dominate).

### 1-GNN generator

**Images:**
- [`results/proteins/1gnn_class1/figures/explanations.png`](../results/proteins/1gnn_class1/figures/explanations.png)
- [`results/proteins/1gnn_class1/figures/metrics.png`](../results/proteins/1gnn_class1/figures/metrics.png)
- [`results/proteins/comparison/figures/structural_1gnn_class1.png`](../results/proteins/comparison/figures/structural_1gnn_class1.png)

**What the figures show:**
- ~24-node graphs, **75% Sheet / 24% Helix** — over-commits to the Sheet signal vs real 56/41.
- Validity 37% (lowest on PROTEINS), mean score 0.33, but mean-*valid* score 0.89 — large quality spread.
- Dense Sheet networks with scattered Helix nodes at the periphery.

### 1-2-GNN generator

**Images:**
- [`results/proteins/12gnn_class1/figures/explanations.png`](../results/proteins/12gnn_class1/figures/explanations.png)
- [`results/proteins/12gnn_class1/figures/metrics.png`](../results/proteins/12gnn_class1/figures/metrics.png)
- [`results/proteins/comparison/figures/structural_12gnn_class1.png`](../results/proteins/comparison/figures/structural_12gnn_class1.png)

**What the figures show:**
- ~30-node graphs, **89% Sheet / 11% Helix** — even more extreme Sheet dominance.
- Validity 94%, mean score 0.52, embedding similarity only 0.15 (class-confident but far from class centroid).

### Cross-model check

**Image:** [`results/proteins/comparison/figures/cross_classification.png`](../results/proteins/comparison/figures/cross_classification.png)

**Observation:** **≥98% cross-model agreement** — both architectures agree.

**Defensible claim:** For enzymes both architectures have learned "many Sheets → enzyme". The difference is how extreme each generator gets; the agreement is genuine.

---

## Embedding space (t-SNE)

**Images:**
- [`results/mutag/comparison/figures/embedding_t-sne_1gnn.png`](../results/mutag/comparison/figures/embedding_t-sne_1gnn.png)
- [`results/mutag/comparison/figures/embedding_t-sne_12gnn.png`](../results/mutag/comparison/figures/embedding_t-sne_12gnn.png)
- [`results/proteins/comparison/figures/embedding_t-sne_1gnn.png`](../results/proteins/comparison/figures/embedding_t-sne_1gnn.png)
- [`results/proteins/comparison/figures/embedding_t-sne_12gnn.png`](../results/proteins/comparison/figures/embedding_t-sne_12gnn.png)

**What the figures show:**
- In every panel, generated class-0 and class-1 point clouds form **two distinct clusters**, which is expected because the generator is trained to maximise class separation.
- In every panel, the real-data points form a separate cluster away from both generated clouds — i.e. the generator finds a "shortcut" to high classifier confidence rather than reproducing the real embedding distribution.
- This explains why embedding similarity `s` is the dominant drag on mean validation score: the generator sits at the right side of the decision boundary but not near the class centroid.

---

## Synthesis table

| Combination | Val. score (1g / 12g) | Validity (1g / 12g) | Size KS (1g / 12g) | Cross-agree (1g / 12g) | Core claim |
|---|---|---|---|---|---|
| MUTAG c0 (Mutagen) | 0.37 / 0.43 | 36% / 41% | 0.89 / 0.88 | **90% / 100%** | Both agree — shared signal |
| MUTAG c1 (Non-Mutagen) | 0.68 / 0.74 | 78% / 82% | 0.89 / **0.59** | **92% / 12%** | 1-2-GNN finds decision surface invisible to 1-GNN |
| PROTEINS c0 (Non-Enzyme) | **0.99** / 0.76 | 100% / 100% | **0.66** / 0.70 | **9% / 100%** | Mirror asymmetry — 1-GNN graphs rejected by 1-2-GNN |
| PROTEINS c1 (Enzyme) | 0.33 / 0.52 | 37% / **94%** | **0.54** / 0.74 | 99% / 98% | Both agree — Sheet-dominant signal |

### Dataset character

- **MUTAG** is small (188 graphs) with a clear chemical signal: halogenated aromatics → mutagen; clean hydrocarbons → non-mutagen. Both generators exploit atom-type distributions effectively.
- **PROTEINS** is larger (1113 graphs) and structurally noisier; real graph sizes span 4–620 nodes but the generator's `gen_size=50` cap collapses the size distribution (KS statistics near 1 for size on class 0). Despite this, the hierarchical architecture's sensitivity to pairwise neighbourhood statistics produces the most interesting divergence.

### Where the expressivity gap shows up

Two combinations out of four expose the hierarchical model's extra expressivity:

1. **MUTAG class 1:** 1-2-GNN's non-mutagen (80% C) is structurally *more authentic* than 1-GNN's but falls outside 1-GNN's decision boundary (12% cross-agreement). The 1-2-GNN has learned that *absence of halogen-halogen pair patterns* is sufficient evidence — something 1-GNN cannot encode.
2. **PROTEINS class 0:** 1-GNN produces an H/S mix with correct marginals (KS=0.66, the best on PROTEINS), but 1-2-GNN reclassifies 91% as enzymes. The 1-2-GNN's decision depends on *Helix/Sheet pairwise neighbourhood distribution*, which the 1-GNN cannot control during generation.

The other two combinations (MUTAG class 0, PROTEINS class 1) show ≥90% cross-agreement and thus do **not** separate the two architectures — useful as controls that confirm the framework is not just noisy.

---

## Reading order for a defence

If you need to walk someone through this in ten minutes, open figures in this order:

1. [`results/mutag/comparison/figures/cross_classification.png`](../results/mutag/comparison/figures/cross_classification.png) — point at the 12% cell. That is the central empirical finding.
2. [`results/proteins/comparison/figures/cross_classification.png`](../results/proteins/comparison/figures/cross_classification.png) — point at the 9% cell. Same phenomenon, mirrored.
3. [`results/mutag/12gnn_class1/figures/explanations.png`](../results/mutag/12gnn_class1/figures/explanations.png) vs [`results/mutag/1gnn_class1/figures/explanations.png`](../results/mutag/1gnn_class1/figures/explanations.png) — show that 1-2-GNN's non-mutagens are visibly cleaner C-chains.
4. [`results/mutag/comparison/figures/structural_12gnn_class1.png`](../results/mutag/comparison/figures/structural_12gnn_class1.png) — atom-distribution bar chart proves the "cleaner" observation quantitatively (80% C, matching real 73%).
5. [`results/mutag/comparison/figures/embedding_t-sne_12gnn.png`](../results/mutag/comparison/figures/embedding_t-sne_12gnn.png) — shows the clean class separation in 1-2-GNN's embedding space, while real data sits in its own cluster.
6. The other six per-combination `explanations.png` and `structural_*.png` files for completeness.
