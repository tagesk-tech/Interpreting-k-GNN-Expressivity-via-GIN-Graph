# Kvalitativ analys — guide till bilder och tabeller (§5.B)

Allt här bygger på 1000 genererade grafer per (modell, klass), inte topp-N.
Generatorn tar in en r.v. och spottar ut ett exempel åt gången, så topp-N
är en stickprov från svansen av slumpen, inte en karakterisering. Vi
mäter därför hela populationen.

## Aggregerade fingeravtryck per (dataset, klass)

För varje `{dataset}` (mutag, proteins):

| Vad det säger | Bild |
|---|---|
| Cykellängdsfördelning (3-, 4-, 5-, 6-cykler), exakta räkningar | `results/{dataset}/comparison/figures/fingerprint_cycles.png` |
| Atom/struktur-typsfördelning per kant — 1-2-GNN | `results/{dataset}/comparison/figures/fingerprint_edge_pairs_12gnn.png` |
| Atom/struktur-typsfördelning per kant — 1-GNN | `results/{dataset}/comparison/figures/fingerprint_edge_pairs_1gnn.png` |
| Medelgrad per nodtyp | `results/{dataset}/comparison/figures/fingerprint_degree_by_type.png` |

Strukturella fördelningar per (dataset, modell, klass) — KS-test mot
verklig data:

| Vad det säger | Bild |
|---|---|
| Grad/typ/storleksfördelning, gen vs real | `results/{dataset}/comparison/figures/structural_{model}_class{N}.png` |

Cross-model-classification (vem håller med vem):

| Vad det säger | Bild |
|---|---|
| Heatmap + bars över alla 8 cellerna | `results/{dataset}/comparison/figures/cross_classification.png` |

## Illustrativa exempel (inte källa till påståenden)

För visuell intuition — slumpmässigt valda exempel (inte topp-rankade):

| Vad det säger | Bild |
|---|---|
| 5 riktiga grafer per klass | `results/{dataset}/comparison/figures/real_samples.png` |
| 5 random gen per klass (per modell) | `results/{dataset}/comparison/figures/random_samples_{model}.png` |
| 5 random gen vs 5 real, per (modell, klass) | `results/{dataset}/comparison/figures/real_vs_random_{model}_class{N}.png` |

## Cykelräkningar — siffrorna bakom `fingerprint_cycles.png`

Genomsnittligt antal exakta enkla cykler per graf, från `fingerprints.json`:

### MUTAG

| Class | Source | 3-cyc | 4-cyc | 5-cyc | 6-cyc |
|---|---|---|---|---|---|
| 0 (Non-Mut.) | real        | 0.00 | 0.00 | 0.21 | 1.48 |
| 0 (Non-Mut.) | 1-GNN gen   | 0.82 | 0.96 | 1.22 | 1.46 |
| 0 (Non-Mut.) | 1-2-GNN gen | 0.50 | 1.42 | 0.97 | 1.63 |
| 1 (Mutagen)  | real        | 0.00 | 0.00 | 0.44 | 3.02 |
| 1 (Mutagen)  | 1-GNN gen   | 1.01 | 1.19 | 3.31 | 3.47 |
| 1 (Mutagen)  | 1-2-GNN gen | 0.15 | 3.88 | 3.21 | 2.29 |

Real MUTAG har **noll trianglar och fyrkanter**. Båda generatorerna inför
dem; men 1-2-GNN har en betydligt lägre 3-cykel-rate (0.50 vs 0.82 i klass
0, 0.15 vs 1.01 i klass 1) — vilket är förenligt med att 2-WL kan
distinguera trianglar som 1-WL inte kan.

### PROTEINS

| Class | Source | 3-cyc | 4-cyc | 5-cyc | 6-cyc |
|---|---|---|---|---|---|
| 0 (Enzyme)     | real        | 24.79 | 34.58 |  41.02 |  65.97 |
| 0 (Enzyme)     | 1-GNN gen   | 15.93 | 45.19 |  95.59 | 204.31 |
| 0 (Enzyme)     | 1-2-GNN gen |  9.07 | 25.21 |  86.52 | 272.20 |
| 1 (Non-Enzyme) | real        | 15.61 | 21.55 |  21.94 |  31.45 |
| 1 (Non-Enzyme) | 1-GNN gen   | 11.08 | 33.56 |  78.70 | 189.72 |
| 1 (Non-Enzyme) | 1-2-GNN gen | 11.62 | 16.45 |  49.96 | 131.38 |

## Population-statistik per (modell, klass)

Från `results/{dataset}/comparison/report.json`, beräknat på 1000 grafer
per cell. Här är de viktiga aggregaten:

### MUTAG

| Class | Source | $\bar{n}$ | $\bar{d}$ | %C | %N | %O | %halogen |
|---|---|---|---|---|---|---|---|
| 0 (Non-Mut.) | real        | 13.9 | 2.09 | 64.2 | 13.0 | 18.8 |  4.1 |
| 0 (Non-Mut.) | 1-GNN gen   | 23.8 | 2.07 | 57.1 | 11.1 | 17.0 | 14.9 |
| 0 (Non-Mut.) | 1-2-GNN gen | 23.2 | 2.07 | 45.7 |  0.1 | 12.5 | 41.7 |
| 1 (Mutagen)  | real        | 19.8 | 2.24 | 73.2 |  9.3 | 17.1 |  0.4 |
| 1 (Mutagen)  | 1-GNN gen   | 27.3 | 2.23 | 63.4 |  7.5 | 19.1 | 10.0 |
| 1 (Mutagen)  | 1-2-GNN gen | 24.3 | 2.25 | 79.8 |  2.9 | 17.2 |  0.0 |

### PROTEINS

| Class | Source | $\bar{n}$ | $\bar{d}$ | %Helix | %Sheet | %Coil/Turn |
|---|---|---|---|---|---|---|
| 0 (Enzyme)     | real        | 48.7 | 3.80 | 50.4 | 46.5 | 3.1 |
| 0 (Enzyme)     | 1-GNN gen   | 46.7 | 3.81 | 54.2 | 45.8 | 0.0 |
| 0 (Enzyme)     | 1-2-GNN gen | 50.0 | 3.80 | 50.9 | 47.1 | 2.0 |
| 1 (Non-Enzyme) | real        | 23.7 | 3.64 | 40.7 | 56.2 | 3.1 |
| 1 (Non-Enzyme) | 1-GNN gen   | 24.1 | 3.72 | 24.8 | 75.2 | 0.0 |
| 1 (Non-Enzyme) | 1-2-GNN gen | 29.6 | 3.64 | 11.3 | 88.7 | 0.0 |

## Cross-model agreement (1000 grafer per cell)

Från `results/{dataset}/comparison/report.json` (`cross_agreement`):

| Cell | Same-model | Cross-model |
|---|---|---|
| MUTAG c0 (1-GNN)    | 100.0% |  89.8% |
| MUTAG c0 (1-2-GNN)  | 100.0% |  99.9% |
| MUTAG c1 (1-GNN)    |  99.9% |  90.5% |
| MUTAG c1 (1-2-GNN)  | 100.0% | **12.3%** |
| PROTEINS c0 (1-GNN)   | 100.0% | **14.0%** |
| PROTEINS c0 (1-2-GNN) |  97.6% | 100.0% |
| PROTEINS c1 (1-GNN)   |  99.9% |  98.8% |
| PROTEINS c1 (1-2-GNN) | 100.0% |  98.5% |

De två cellerna som sticker ut: MUTAG c1 (1-2-GNN) på 12.3% och PROTEINS
c0 (1-GNN) på 14.0%. Det är där arkitekturerna faktiskt skiljer sig.

## t-SNE — varför vi inte använder det

Kort version: cross-classification och embedding-similarity (s-värdet i
Tabell `tab:sanity` i §5.A) ger redan kvantitativa svar i det riktiga
64-dim-rummet. En 2D t-SNE skulle bara återupprepa svaret visuellt, och
projektionen förvränger avstånd på sätt som vi inte kontrollerar
(perplexity-val, slumpfrö, etc.). Tidigare i projektet såg vi att tighta
kluster i embedding-rummet såg ut som dramatiska gap till real data i
t-SNE — mer dramatiskt än cross-classification och s-värdet stödde. Vi
litar på de aggregerade fingeravtrycken i original-rummet.

## Reproduktion

```bash
# Steg 1: Generera 1000 grafer per (model, class) per dataset.
python compare_datasets.py --dataset mutag --num_samples 1000
python compare_datasets.py --dataset proteins --num_samples 1000

# Steg 2: Beräkna fingeravtryck (cykler, kanttyper, grad-per-typ).
python aggregate_fingerprints.py --dataset mutag
python aggregate_fingerprints.py --dataset proteins

# Steg 3 (valfritt, illustrativa figurer): random-sample exempel.
python plot_qualitative_overview.py
```

## Läsordning för möte / försvar

1. `cross_classification.png` — peka på 12%-cellen (MUTAG c1) och
   14%-cellen (PROTEINS c0). Det är huvudfynden.
2. `fingerprint_cycles.png` för MUTAG — visar att 1-2-GNN trycker ned
   3-cyklar (0.15 vs 1.01 i klass 1) men producerar fler 4-cyklar
   (3.88 vs 1.19). Förenligt med 2-WL-teorin.
3. `structural_*.png` för siffer-stöd (atomfördelning, grad, storlek).
4. `fingerprint_edge_pairs_{model}.png` för kantkomposition — visar
   t.ex. att 1-2-GNN i MUTAG c1 koncentrerar bonds i C-C/C-N/C-O.
5. `random_samples_{model}.png` om du vill visa "vad spottar generatorn
   ut typiskt".
