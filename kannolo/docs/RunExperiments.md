## Replicate Results

We provide a quick way to replicate the results of our paper. 

Use the [`scripts/run_experiments.py`](scripts/run_experiments.py) script to quickly reproduce a result from the paper. 
This script is configurable via TOML files, which specify the parameters to build the index and execute queries on it.  
The script measures average query time (in microseconds), recall with respect to the true closest vectors of the query (accuracy@k), MRR or other metrics with respect to judged qrels if specified, and index space usage (bytes).
Reranking and multivector experiments are no longer supported; the runner will error if a config includes them.

TOML files to reproduce the experiments of our ICDE 2026 paper are in
[`experiments/best_configs/msmarco-v1`](experiments/best_configs/msmarco-v1). The mapping from each cell of
the results table to its exact config is in
[`experiments/icde2026/kannolo_table_reproducibility.md`](experiments/icde2026/kannolo_table_reproducibility.md).

The pre-processed datasets are released on Hugging Face at
[`tuskanny/datasets_for_icde`](https://huggingface.co/datasets/tuskanny/datasets_for_icde). See the
top-level [`README.md`](../../README.md) for the download command, the expected on-disk layout, and how to
create the index output directories.

### Running the Experiment

After downloading the data and creating the index directories (see the top-level README), build kANNolo
and install the Python requirements:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
pip install -r scripts/requirements.txt
```

If needed, install Rust first:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

As an example, reproduce the Splade Cocondenser (`cocondenser`) cell at value-type `f16`, m=16, k=10:

```bash
python3 scripts/run_experiments.py \
    --exp experiments/best_configs/msmarco-v1/splade-cocondenser/svb_grid/cocondenser_svb_grid_f16_m16_k10_best.toml
```

The script will build an index using the unified binary parameters specified at the top level of the TOML file (`build-command`, `dataset-type`, `value-type`, optional sparse `component-type`, `encoder`, `graph-type`) and the traditional indexing parameters in the `[indexing_parameters]` section (`m`, `ef-construction`, `metric`).  
The index is saved under the `[folder] index` directory set in the TOML (for the example above,
`~/data/cocondenser/kannolo`).  
You can change directory names by modifying the `[folder]` section in the TOML file.

Next, the script will query the same index with different parameters, as specified in the `[query]` section.  
These parameters provide different trade-offs between query time and accuracy.

**Important**: if your machine is NUMA, the NUMA setting in the TOML file should be UNcommented and should be configured according to your hardware for better performance. 

## TOML Configuration Structure

The TOML configuration files have been updated to work with the unified binaries. Here's the structure:

### Top-level Parameters
- `build-command`: Path to the unified build binary (e.g., `"./target/release/hnsw_build"`)
- `query-command`: Path to the unified search binary (e.g., `"./target/release/hnsw_search"`)
- `dataset-type`: Type of vectors - `"dense"` or `"sparse"` (multivector is no longer supported)
- `value-type`: Value type - `"f32"`, `"f16"`, `"fixedu8"`, or `"fixedu16"` (for `encoder = "pq"` and `encoder = "dotvbyte"`, this is ignored)
- `component-type`: Sparse-only component type - `"u16"` or `"u32"` (DotVByte requires `"u16"`)
- `encoder`: Encoder type - `"plain"`, `"pq"`, or `"dotvbyte"` (`pq` is dense-only, `dotvbyte` is sparse-only)
- `graph-type`: Graph type - `"standard"` or `"fixed-degree"`

### Sections
- `[indexing_parameters]`: Traditional HNSW parameters (`m`, `ef-construction`, `metric`)
- `[pq_parameters]`: PQ-specific parameters (`pq-subspaces`, `nbits`, `sample-size`) when using PQ encoder. Supported `pq-subspaces` values are `4, 8, 16, 32, 64, 96, 128`. `nbits` and `sample-size` are accepted for compatibility but ignored by the current vectorium PQ implementation.
- `[folder]`: Directory paths for data, indexes, and experiments
- `[filename]`: Filenames for dataset, queries, groundtruth, etc.
- `[settings]`: Runtime settings (k, NUMA, build flag, evaluation metric)
- `[query]`: Different ef-search values for query experiments

### Example TOML Structure

Here's an example of the complete TOML structure for a dense PQ experiment:

```toml
name = "example_hnsw_pq"
title = "Example HNSW PQ Experiment"
description = "Example experiment with Product Quantization"
dataset = "Example Dataset"
build-command = "./target/release/hnsw_build"
query-command = "./target/release/hnsw_search"
dataset-type = "dense"
value-type = "f32"
encoder = "pq"
graph-type = "standard"

[settings]
k = 10
num-runs = 1
NUMA = "numactl --physcpubind='0-15' --localalloc"
build = true
metric = ""

[folder]
data = "~/knn_datasets/dense_datasets/example"
index = "~/knn_indexes/dense_datasets/example"
experiment = "."

[filename]
dataset = "dataset.npy"
queries = "queries.npy"
groundtruth = "groundtruth.npy"
index = "example_index"

[indexing_parameters]
m = 16
ef-construction = 150
metric = "dotproduct"

[pq_parameters]  # Only needed when encoder = "pq"
pq-subspaces = 64
nbits = 8
sample-size = 100000

[query]
    [query.efs_40]
    ef-search = 40
    [query.efs_80]
    ef-search = 80
``` 

### Getting the Results
The script creates a folder named `sift_hnsw_XXX`, where `XXX` encodes the datetime at which the script was executed. This ensures that each run creates a unique directory.

Inside the folder, you can find the data collected during the experiment.

The most important file is `report.tsv`, which reports *query time* and *accuracy*.
