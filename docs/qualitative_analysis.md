# Kvalitativ analys — fyra kombinationer

Guide till bilderna för de fyra kombinationerna: MUTAG × {klass 0, klass 1} och PROTEINS × {klass 0, klass 1}, för både 1-GNN och 1-2-GNN.

Klass-etiketter läses från `report.json` (`class_label`). Prediction scores och validation scores kommer från samma filer.

## Bilder per kombination — 3 per dataset×klass

För varje `{dataset}/{model}_class{N}`:

| Syfte | Bild |
|---|---|
| Topp-10 genererade grafer med score | `results/{dataset}/{model}_class{N}/figures/explanations.png` |
| Score-fördelning (validation / prediction / validity) | `results/{dataset}/{model}_class{N}/figures/metrics.png` |
| Genererad vs riktig — distributioner | `results/{dataset}/comparison/figures/structural_{model}_class{N}.png` |

Nya bilder i denna PR (riktiga grafer + jämförelser):

| Syfte | Bild |
|---|---|
| 5 riktiga grafer per klass | `results/{dataset}/comparison/figures/real_samples.png` |
| Topp-5 genererade (rad per klass) | `results/{dataset}/comparison/figures/overview_{model}.png` |
| Topp-5 gen (topp) vs 5 riktiga (botten) | `results/{dataset}/comparison/figures/real_vs_gen_{model}_class{N}.png` |

Globala bilder per dataset:

| Syfte | Bild |
|---|---|
| Cross-model agreement (vem håller med vem) | `results/{dataset}/comparison/figures/cross_classification.png` |
| t-SNE embedding-space | `results/{dataset}/comparison/figures/embedding_t-sne_{model}.png` |

## Prediction/validation scores per kombination

Från `results/{dataset}/{model}_class{N}/report.json`:

| Kombination | Validity | Mean val. score | Mean pred. prob | Mean embedding sim |
|---|---|---|---|---|
| MUTAG c0 / 1-GNN   | 36%  | 0.37 | 1.00 | 0.70 |
| MUTAG c0 / 1-2-GNN | 41%  | 0.43 | 1.00 | 0.77 |
| MUTAG c1 / 1-GNN   | 78%  | 0.68 | 1.00 | 0.94 |
| MUTAG c1 / 1-2-GNN | 82%  | 0.74 | 1.00 | 0.93 |
| PROTEINS c0 / 1-GNN   | 100% | 0.99 | 0.99 | 0.99 |
| PROTEINS c0 / 1-2-GNN | 100% | 0.76 | 0.56 | 0.78 |
| PROTEINS c1 / 1-GNN   | 37%  | 0.33 | 0.92 | 0.36 |
| PROTEINS c1 / 1-2-GNN | 94%  | 0.52 | 1.00 | 0.15 |

## Cross-model agreement

Från `results/{dataset}/comparison/report.json` (`cross_agreement`):

| Kombination | Same-model | Cross-model |
|---|---|---|
| MUTAG c0 (1-GNN)    | 100% | 90%  |
| MUTAG c0 (1-2-GNN)  | 100% | 100% |
| MUTAG c1 (1-GNN)    | 100% | 92%  |
| MUTAG c1 (1-2-GNN)  | 100% | **12%** |
| PROTEINS c0 (1-GNN)   | 100% | **9%** |
| PROTEINS c0 (1-2-GNN) | 98%  | 100% |
| PROTEINS c1 (1-GNN)   | 100% | 100% |
| PROTEINS c1 (1-2-GNN) | 100% | 98%  |

Två celler som sticker ut (12% och 9%) — där arkitekturerna faktiskt skiljer sig.

## Läsordning för möte / försvar

1. Öppna `cross_classification.png` för båda dataset — peka på 12%-cellen (MUTAG c1) och 9%-cellen (PROTEINS c0). Det är huvudfynden.
2. Öppna `real_vs_gen_1gnn_class1.png` och `real_vs_gen_12gnn_class1.png` för MUTAG — se att 1-2-GNN:s non-mutagener är mycket renare C-kedjor (~80% C) jämfört med 1-GNN:s (~63% C + Iodine).
3. Samma för PROTEINS c0 — se att 1-GNN:s non-enzymer har H/S-marginal som ser rätt ut men förkastas av 1-2-GNN.
4. `structural_*.png` för siffer-stöd (atomfördelning, degree, storlek).
5. `embedding_t-sne_*.png` för att visa var genererade grafer sitter jämfört med riktig data.
