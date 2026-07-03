# ICDE 2026 - Compact and Efficient Indexes for Learned Sparse Retrieval 

This repository allows to reproduce the main results table of our ICDE 2026 submission. 
The repo is organized in two sub-repo.

- **`kannolo/`** — *kANNolo*. 
- **`seismic/`** — *Seismic*.

Every subo repo contains a .md files with a results table and every cell of the results table maps to one configuration (TOML) file. The cell → config ledgers are:

- kANNolo: [`kannolo/experiments/icde2026/kannolo_table_reproducibility.md`](kannolo/experiments/icde2026/kannolo_table_reproducibility.md)
- Seismic: [`seismic/experiments/ICDE2026/seismic_table_reproducibility.md`](seismic/experiments/ICDE2026/seismic_table_reproducibility.md)

Both engines run the same harness: `python3 scripts/run_experiments.py --exp <config.toml>`, which builds
the index, sweeps the query parameters, and writes a `report.tsv` with query time (µs), recall, and index
size.

The three datasets are shared by both engines:

| Paper name | Dataset folder | Collection |
|---|---|---|
| Splade Cocondenser | `cocondenser` | MS MARCO v1 passage — SPLADE CoCondenser |
| LiLSR              | `lilsr-big`   | MS MARCO v1 passage — LiLSR (big) |
| Splade-v3          | `splade_v3`   | MS MARCO v1 passage — SPLADE-v3 |

---

## 1. Get the data

All datasets are released **already pre-processed in our binary format** working for both engines. They are hosted on HuggingFace at
[`tuskanny/datasets_for_icde`](https://huggingface.co/datasets/tuskanny/datasets_for_icde).

The shipped configs read each dataset from `~/data/<dataset>/data/`. On the Hub the files sit directly
under `<dataset>/` (no `data/` level), so download to a staging folder and move each dataset into place:

```bash
pip install -U huggingface_hub
huggingface-cli download tuskanny/datasets_for_icde --repo-type dataset --local-dir ~/hf_staging
# or one dataset only:  add  --include "cocondenser/*"

for d in cocondenser lilsr-big splade_v3; do
    [ -d ~/hf_staging/$d ] && mkdir -p ~/data/$d && mv ~/hf_staging/$d ~/data/$d/data
done
rm -rf ~/hf_staging   # leftover HF metadata
```

This gives you, for each `<dataset>` in `{cocondenser, lilsr-big, splade_v3}`:

```
~/data/<dataset>/data/
    documents.bin             # collection (binary format)
    queries.bin               # queries   (binary format)
    doc_ids.npy               # original document ids
    queries_ids.npy           # original query ids
    groundtruth.tsv           # exact top-10  ground truth  (used by the k=10 configs)
    groundtruth_100.tsv       # exact top-100 ground truth  (used by the k=100 configs)
    qrels.dev.small.tsv       
    token_to_id_mapping.json  # token → id map 
```

That's all the setup needed: the configs read from `~/data/<dataset>/data` and write the built index
next to it (`~/data/<dataset>/indexes` for Seismic, `~/data/<dataset>/kannolo` for kANNolo — both created
automatically on first run). To use a different location, just edit the `[folder]` section (`data` /
`index`) of the config you run.

---

## 2. Build prerequisites

**Rust** (both engines compile with `cargo`):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Python** (python >= 3.11, one environment per engine, or a shared one):

```bash
pip install -r seismic/scripts/requirements.txt
pip install -r kannolo/scripts/requirements.txt
```

> **NUMA.** On multi-socket machines, uncomment/adjust the `NUMA` line in the `[settings]` section of a
> config (e.g. `numactl --physcpubind='0-15' --localalloc`) to match your hardware for representative
> timings.

---

## 3. Reproduce the Seismic results

```bash
cd seismic
RUSTFLAGS="-C target-cpu=native" cargo build --release
python3 scripts/run_experiments.py --exp experiments/ICDE2026/<dataset>/<...>/recall_90.toml
```

Pick the exact config for any table cell from
[`seismic/experiments/ICDE2026/seismic_table_reproducibility.md`](seismic/experiments/ICDE2026/seismic_table_reproducibility.md).
Each run creates `report.tsv` reporting query time, recall, and index size.

## 4. Reproduce the kANNolo results

```bash
cd kannolo
RUSTFLAGS="-C target-cpu=native" cargo build --release
python3 scripts/run_experiments.py --exp experiments/best_configs/msmarco-v1/<dataset>/svb_grid/<config>_best.toml
```

Pick the exact config for any table cell from
[`kannolo/experiments/icde2026/kannolo_table_reproducibility.md`](kannolo/experiments/icde2026/kannolo_table_reproducibility.md).
Each config's `[query]` subsections (`recall_90` / `recall_95` / `recall_99`) reproduce the corresponding
rows of the table.

---

## Repository layout

```
.
├── README.md                # this file
├── kannolo/                 # kANNolo engine + ICDE2026 best configs
│   ├── scripts/run_experiments.py
│   └── experiments/
│       ├── icde2026/kannolo_table_reproducibility.md
│       └── best_configs/msmarco-v1/{splade-cocondenser,splade-lilsr-big,splade-v3}/svb_grid/
└── seismic/                 # Seismic engine + ICDE2026 best configs
    ├── scripts/run_experiments.py
    └── experiments/ICDE2026/
        ├── seismic_table_reproducibility.md
        └── msmarco-v1-{cocondenser,lilsr,splade-v3}/...
```
