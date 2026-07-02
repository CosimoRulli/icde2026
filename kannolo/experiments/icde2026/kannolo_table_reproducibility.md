# kANNolo → ICDE 2026 main-results table (`table:results1`) — reproducibility map


**How to reproduce any cell** (the configs build the index and the EGB permutation from the downloaded
collection, `build=true` / `permutation=true`):

```bash
python3 scripts/run_experiments.py --exp <best-config TOML>
```
Each run writes a fresh `runs/<name>_<timestamp>/report.tsv`; the `recall_<Acc>` row of the report
reproduces the corresponding table row below (µs has minor run-to-run variance). The exact query
parameters for each row are in the `[query.recall_<Acc>]` subsection of the config.

---

## Splade Cocondenser  —  Mem. budget 1.0 = 4.0 GiB

best-config dir: `experiments/best_configs/msmarco-v1/splade-cocondenser/svb_grid/`

### kannolo

| Budget | k | Acc. | best-config TOML | µs / GiB |
|---|---|---|---|---|
| >=1.0 | 10 | 90% | `cocondenser_svb_grid_f16_m16_k10_best.toml` | 299 / 4.6 |
| >=1.0 | 10 | 95% | `cocondenser_svb_grid_f16_m16_k10_best.toml` | 431 / 4.6 |
| >=1.0 | 10 | 99% | `cocondenser_svb_grid_f16_m32_k10_best.toml` | 999 / 4.9 |
| >=1.0 | 100 | 90% | `cocondenser_svb_grid_f16_m16_k100_best.toml` | 823 / 4.6 |
| >=1.0 | 100 | 95% | `cocondenser_svb_grid_f16_m16_k100_best.toml` | 942 / 4.6 |
| >=1.0 | 100 | 99% | `cocondenser_svb_grid_f16_m32_k100_best.toml` | 2512 / 4.9 |

### kannolo [new]

| Budget | k | Acc. | best-config TOML | µs / GiB |
|---|---|---|---|---|
| <1.0 | 10 | 90% | `cocondenser_svb_grid_dp8_m16_k10_best.toml` | 301 / 2.7 |
| <1.0 | 10 | 95% | `cocondenser_svb_grid_dp8_m16_k10_best.toml` | 434 / 2.7 |
| <1.0 | 10 | 99% | — (target not reached) | — |
| <1.0 | 100 | 90% | `cocondenser_svb_grid_dp8_m16_k100_best.toml` | 776 / 2.7 |
| <1.0 | 100 | 95% | `cocondenser_svb_grid_dp8_m16_k100_best.toml` | 900 / 2.7 |
| <1.0 | 100 | 99% | `cocondenser_svb_grid_dp8_m32_k100_best.toml` | 4350 / 3.1 |

---

## LiLSR  =  lilsr-big  —  Mem. budget 1.0 = 12.8 GiB

best-config dir: `experiments/best_configs/msmarco-v1/splade-lilsr-big/svb_grid/`

### kannolo

| Budget | k | Acc. | best-config TOML | µs / GiB |
|---|---|---|---|---|
| >=1.0 | 10 | 90% | `lilsr-big_svb_grid_f16_m32_k10_best.toml` | 5779 / 13.3 |
| >=1.0 | 10 | 95% | `lilsr-big_svb_grid_f16_m32_k10_best.toml` | 27642 / 13.3 |
| >=1.0 | 10 | 99% | — (target not reached) | — |
| >=1.0 | 100 | 90% | `lilsr-big_svb_grid_f16_m32_k100_best.toml` | 8069 / 13.3 |
| >=1.0 | 100 | 95% | `lilsr-big_svb_grid_f16_m32_k100_best.toml` | 43173 / 13.3 |
| >=1.0 | 100 | 99% | — (target not reached) | — |

### kannolo [new]

| Budget | k | Acc. | best-config TOML | µs / GiB |
|---|---|---|---|---|
| >=1.0 | 10 | 90% | `lilsr-big_svb_grid_blocksparse_m32_k10_best.toml` | 3312 / 13.5 |
| >=1.0 | 10 | 95% | `lilsr-big_svb_grid_blocksparse_m32_k10_best.toml` | 17161 / 13.5 |
| >=1.0 | 10 | 99% | — (target not reached) | — |
| >=1.0 | 100 | 90% | `lilsr-big_svb_grid_blocksparse_m32_k100_best.toml` | 4500 / 13.5 |
| >=1.0 | 100 | 95% | `lilsr-big_svb_grid_blocksparse_m32_k100_best.toml` | 26623 / 13.5 |
| >=1.0 | 100 | 99% | — (target not reached) | — |
| <1.0 | 10 | 90% | `lilsr-big_svb_grid_sparseq_m32_k10_best.toml` | 3616 / 6.9 |
| <1.0 | 10 | 95% | `lilsr-big_svb_grid_sparseq_m32_k10_best.toml` | 39749 / 6.9 |
| <1.0 | 10 | 99% | — (target not reached) | — |
| <1.0 | 100 | 90% | `lilsr-big_svb_grid_sparseq_m32_k100_best.toml` | 5175 / 6.9 |
| <1.0 | 100 | 95% | `lilsr-big_svb_grid_sparseq_m32_k100_best.toml` | 25110 / 6.9 |
| <1.0 | 100 | 99% | — (target not reached) | — |

---

## Splade-v3  =  splade_v3  —  Mem. budget 1.0 = 6.0 GiB

best-config dir: `experiments/best_configs/msmarco-v1/splade-v3/svb_grid/`

### kannolo

| Budget | k | Acc. | best-config TOML | µs / GiB |
|---|---|---|---|---|
| >=1.0 | 10 | 90% | `splade_v3_svb_grid_f16_m32_k10_best.toml` | 445 / 6.4 |
| >=1.0 | 10 | 95% | `splade_v3_svb_grid_f16_m32_k10_best.toml` | 636 / 6.4 |
| >=1.0 | 10 | 99% | `splade_v3_svb_grid_f16_m32_k10_best.toml` | 2327 / 6.4 |
| >=1.0 | 100 | 90% | `splade_v3_svb_grid_f16_m32_k100_best.toml` | 1425 / 6.4 |
| >=1.0 | 100 | 95% | `splade_v3_svb_grid_f16_m32_k100_best.toml` | 1657 / 6.4 |
| >=1.0 | 100 | 99% | `splade_v3_svb_grid_f16_m32_k100_best.toml` | 7729 / 6.4 |

### kannolo [new]

| Budget | k | Acc. | best-config TOML | µs / GiB |
|---|---|---|---|---|
| <1.0 | 10 | 90% | `splade_v3_svb_grid_dp8_m16_k10_best.toml` | 417 / 3.4 |
| <1.0 | 10 | 95% | `splade_v3_svb_grid_dp8_m16_k10_best.toml` | 782 / 3.4 |
| <1.0 | 10 | 99% | — (target not reached) | — |
| <1.0 | 100 | 90% | `splade_v3_svb_grid_dp8_m16_k100_best.toml` | 924 / 3.4 |
| <1.0 | 100 | 95% | `splade_v3_svb_grid_dp8_m16_k100_best.toml` | 1402 / 3.4 |
| <1.0 | 100 | 99% | `splade_v3_svb_grid_dp8_m32_k100_best.toml` | 20531 / 3.7 |

---
