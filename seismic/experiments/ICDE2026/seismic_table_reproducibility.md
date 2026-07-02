# Seismic → ICDE 2026 main-results table (`table:results1`) — reproducibility map

Every cell of the Seismic block of the results table maps to **one config TOML** below. After placing the
datasets (see the top-level [`README.md`](../../../README.md)) and creating the index directories, run any
config with the shared harness:

```bash
python3 scripts/run_experiments.py --exp <config TOML>
```

Each run builds the index, sweeps the query parameters, and writes a `report.tsv` with query time (µs),
recall, and index size (GiB); the `recall_<Acc>` row of the report reproduces the corresponding table row
below (µs has minor run-to-run variance — set the `NUMA` line in each config's `[settings]` to match your
host for representative timings).

Each table states its config dir(s), relative to `seismic/experiments/ICDE2026/`; the row's TOML file
lives in the dir matching its `k` / `Acc.` (`k_<k>` stands for `k_10` or `k_100`). The exact build and
query parameters for each cell are in the config TOML (`[indexing_parameters]` and `[query.recall_<Acc>]`).

The paper compares three method variants per dataset:

| Variant | Meaning |
|---|---|
| **Seismic** | Classic Seismic baseline (`f16` / `dotvbyte` value types). |
| **Seismic [new] ≥1.0** | New quantized encoders, memory budget **≥ 1.0×** the `f16` forward index. |
| **Seismic [new] <1.0** | New quantized encoders **below 1.0×** (medoid summaries + sparse encoders). |

---

## Splade Cocondenser  =  cocondenser

### Seismic (baseline)

config dirs:
- k=10: `msmarco-v1-cocondenser/k_10/_legacy/config_1_seismic/best_configs/`
- k=100: `msmarco-v1-cocondenser/k_100/summary_f16/best_configs/mem_budget_2.0/`

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 197 / 4.1 |
| 10  | 95% | `recall_95.toml` | 275 / 5.8 |
| 10  | 99% | `recall_99.toml` | 621 / 7.8 |
| 100 | 90% | `recall_90.toml` | 739 / 7.3 |
| 100 | 95% | `recall_95.toml` | 1129 / 8.4 |
| 100 | 99% | `recall_99.toml` | 2862 / 8.4 |

### Seismic [new] — ≥ 1.0 budget  (`dotpacking8-scalar`; R99 → `dotpacking8-f16`)

config dirs:
- R90 / R95: `msmarco-v1-cocondenser/k_<k>/summary_dotpacking8/best_configs/mem_budget_1.0/`
- R99: `msmarco-v1-cocondenser/k_<k>/summary_dotpacking8_f16/best_configs/mem_budget_1.5/`

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 162 / 4.0 |
| 10  | 95% | `recall_95.toml` | 332 / 4.1 |
| 10  | 99% | `recall_99.toml` | 640 / 6.8 |
| 100 | 90% | `recall_90.toml` | 764 / 4.2 |
| 100 | 95% | `recall_95.toml` | 1849 / 4.2 |
| 100 | 99% | `recall_99.toml` | 2943 / 7.4 |

### Seismic [new] — < 1.0 budget  (`dotpacking8-scalar`, medoid summaries; R99 → `f16`)

config dirs:
- R90 / R95: `msmarco-v1-cocondenser/k_<k>/medoid_dotpacking8/best_configs/mem_budget_1.0/`
- R99 (k=10): `msmarco-v1-cocondenser/k_10/medoid_f16/best_configs/mem_budget_1.5/`

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 438 / 2.6 |
| 10  | 95% | `recall_95.toml` | 794 / 2.8 |
| 10  | 99% | `recall_99.toml` | 2337 / 3.8 |
| 100 | 90% | `recall_90.toml` | 1178 / 2.9 |
| 100 | 95% | `recall_95.toml` | 2116 / 3.0 |
| 100 | 99% | — (target not reached at this budget) | — |

---

## LiLSR  =  lilsr (LiLSR big)

### Seismic (baseline)

config dir: `msmarco-v1-lilsr/k_<k>/summary_f16/best_configs/mem_budget_2.0/`

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 858 / 15.7 |
| 10  | 95% | `recall_95.toml` | 1253 / 16.6 |
| 10  | 99% | `recall_99.toml` | 3068 / 23.1 |
| 100 | 90% | `recall_90.toml` | 2996 / 20.1 |
| 100 | 95% | `recall_95.toml` | 3284 / 20.1 |
| 100 | 99% | `recall_99.toml` | 9258 / 26.1 |

### Seismic [new] — ≥ 1.0 budget  (`dotpacking8-sparse-scalar`; k=10 R99 → `blocked-sparse`)

config dirs:
- R90 / R95: `msmarco-v1-lilsr/k_<k>/summary_dotpacking8_sparse/best_configs/mem_budget_1.0/`
- R99 (k=10): `msmarco-v1-lilsr/k_10/summary_blocked_sparse/best_configs/mem_budget_2.0/`
- R99 (k=100): `msmarco-v1-lilsr/k_100/summary_dotpacking8_sparse/best_configs/mem_budget_2.0/`

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 463 / 8.9 |
| 10  | 95% | `recall_95.toml` | 690 / 10.9 |
| 10  | 99% | `recall_99.toml` | 1475 / 23.4 |
| 100 | 90% | `recall_90.toml` | 1447 / 13.8 |
| 100 | 95% | `recall_95.toml` | 1657 / 13.8 |
| 100 | 99% | `recall_99.toml` | 8792 / 21.6 |

### Seismic [new] — < 1.0 budget  (`dotpacking8-sparse-scalar`, medoid; k=10 R99 → `dotpackingsparse-f16`)

config dirs:
- R90 / R95: `msmarco-v1-lilsr/k_<k>/medoid_dotpacking8_sparse/best_configs/mem_budget_1.0/`
- R99 (k=10): `msmarco-v1-lilsr/` (dataset root)

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 982 / 7.5 |
| 10  | 95% | `recall_95.toml` | 1986 / 8.0 |
| 10  | 99% | `best_99_k10.toml` | 11832 / 11.2 |
| 100 | 90% | `recall_90.toml` | 2206 / 7.8 |
| 100 | 95% | `recall_95.toml` | 4238 / 8.0 |
| 100 | 99% | — (target not reached at this budget) | — |

---

## Splade-v3  =  splade-v3

### Seismic (baseline)

config dir: `msmarco-v1-splade-v3/k_<k>/summary_f16/best_configs/mem_budget_2.0/`

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 158 / 8.6 |
| 10  | 95% | `recall_95.toml` | 199 / 9.0 |
| 10  | 99% | `recall_99.toml` | 462 / 11.1 |
| 100 | 90% | `recall_90.toml` | 981 / 9.2 |
| 100 | 95% | `recall_95.toml` | 1281 / 10.6 |
| 100 | 99% | `recall_99.toml` | 3114 / 11.1 |

### Seismic [new] — ≥ 1.0 budget  (`dotpacking8-scalar`)

config dirs:
- R90 / R95: `msmarco-v1-splade-v3/k_<k>/summary_dotpacking8/best_configs/mem_budget_1.25/`
- R99: `msmarco-v1-splade-v3/k_<k>/summary_dotpacking8/best_configs/mem_budget_2.0/`

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 162 / 5.6 |
| 10  | 95% | `recall_95.toml` | 211 / 6.9 |
| 10  | 99% | `recall_99.toml` | 949 / 9.1 |
| 100 | 90% | `recall_90.toml` | 993 / 6.3 |
| 100 | 95% | `recall_95.toml` | 1289 / 6.6 |
| 100 | 99% | `recall_99.toml` | 3679 / 9.4 |

### Seismic [new] — < 1.0 budget  (`dotpacking8-scalar`, medoid; R99 → `dotpacking8-f16`)

config dirs:
- R90 / R95: `msmarco-v1-splade-v3/k_<k>/medoid_dotpacking8/best_configs/mem_budget_1.0/`
- R99: `msmarco-v1-splade-v3/` (dataset root)

| k | Acc. | config TOML | µs / GiB |
|---|---|---|---|
| 10  | 90% | `recall_90.toml` | 594 / 3.7 |
| 10  | 95% | `recall_95.toml` | 978 / 3.7 |
| 10  | 99% | `best_99_k10.toml` | 3324 / 5.4 |
| 100 | 90% | `recall_90.toml` | 1517 / 3.7 |
| 100 | 95% | `recall_95.toml` | 2441 / 4.1 |
| 100 | 99% | `best_99_k100.toml` | 12676 / 5.4 |
