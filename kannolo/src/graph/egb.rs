use half::f16;
use rgb::forward::Doc;
use rgb::recursive_graph_bisection;

use crate::graph::graph::GraphTrait;
use crate::hnsw_utils::invert_mapping;
use rayon::prelude::*;
use std::time::Instant;
use vectorium::encoders::blocked_sparse::BlockedSparseEncoder;
use vectorium::encoders::dotpacking8_scalaru8::DotPacking8ScalarU8Encoder;
use vectorium::encoders::dotpackingsparse_scalaru8::DotPackingSparseScalarU8Encoder;
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::distances::DotProduct;
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::vector_encoder::SparseDataEncoder;
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, DenseVectorEncoder, FixedU8Q, FromF32,
    PackedSparseDataset, PackedSparseDatasetGrowable, PlainSparseDataset,
    PlainSparseDatasetGrowable, PlainSparseQuantizer, ScalarSparseDataset, SparseVectorView,
};

// EGB parameters
const ITERATIONS: usize = 10; // number of iterations for EGB
const MIN_PARTITION_SIZE: usize = 64; // minimum partition size
const MAX_DEPTH: usize = 100; // maximum recursion depth
const PARALLEL_SWITCH: usize = 100; // parallel switch threshold
const DEPTH_LIMIT: usize = 1; // depth limit for parallelism
const SORT_LEAF: bool = true; // whether to sort leaf nodes
const ID: usize = 1; // identifier for logging

/// Compute an EGB permutation (old_id → new_id) using the (single-level) graph.
pub fn permute_graph_bisection_from_graph<G>(graph: &G) -> Vec<usize>
where
    G: GraphTrait + Sync,
{
    let total_start = Instant::now();
    let n = graph.n_nodes();

    // 1) Collect used neighbor IDs (as "terms").
    let start = Instant::now();
    let mut all_terms: Vec<u32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|u| graph.neighbors(u).map(|v| v as u32))
        .collect();
    println!(
        "[######] EGB collect terms: {:.2}s ({} raw terms)",
        start.elapsed().as_secs_f32(),
        all_terms.len()
    );

    // 2) Compact term ID space while avoiding a hash lookup per edge.
    let start = Instant::now();
    all_terms.par_sort_unstable();
    all_terms.dedup();
    let num_terms = all_terms.len();
    println!(
        "[######] EGB sort/dedup terms: {:.2}s ({} unique terms)",
        start.elapsed().as_secs_f32(),
        num_terms
    );

    let start = Instant::now();
    let mut term_map = vec![u32::MAX; n];
    for (new_id, &old_id) in all_terms.iter().enumerate() {
        term_map[old_id as usize] = new_id as u32;
    }
    println!(
        "[######] EGB build dense term map: {:.2}s",
        start.elapsed().as_secs_f32()
    );

    // 3) Build Docs for RGB (neighbors → compact term IDs).
    let start = Instant::now();
    let mut docs: Vec<Doc> = (0..n)
        .into_par_iter()
        .map(|u| {
            let mut terms: Vec<u32> = graph.neighbors(u).map(|v| term_map[v]).collect();
            terms.sort_unstable();
            terms.dedup();

            Doc {
                terms,
                org_id: u as u32,
                gain: 0.0,
                leaf_id: -1,
            }
        })
        .collect();
    println!(
        "[######] EGB build docs: {:.2}s ({} docs)",
        start.elapsed().as_secs_f32(),
        docs.len()
    );

    // 4) Run recursive bipartite partitioning.
    let start = Instant::now();
    recursive_graph_bisection(
        &mut docs,
        num_terms,
        ITERATIONS,
        MIN_PARTITION_SIZE,
        MAX_DEPTH,
        PARALLEL_SWITCH,
        DEPTH_LIMIT,
        SORT_LEAF,
        ID,
    );
    println!(
        "[######] EGB recursive bisection: {:.2}s",
        start.elapsed().as_secs_f32()
    );

    // 5) Build permutation (old_id → new_id)
    let start = Instant::now();
    let mut perm = vec![0usize; n];
    for (new_id, doc) in docs.iter().enumerate() {
        perm[doc.org_id as usize] = new_id;
    }
    println!(
        "[######] EGB build permutation: {:.2}s",
        start.elapsed().as_secs_f32()
    );
    println!(
        "[######] EGB total: {:.2}s",
        total_start.elapsed().as_secs_f32()
    );
    perm
}

pub fn permute_dense_dataset<E>(
    dataset: &DenseDataset<E>,
    permutation: &[usize],
) -> DenseDataset<E>
where
    E: DenseVectorEncoder + Clone,
    E::OutputValueType: Copy + Default,
    DenseDataset<E>: Dataset<Encoder = E>,
{
    let n = dataset.len();
    let dim = dataset.output_dim();
    assert_eq!(permutation.len(), n);

    let values = dataset.values();
    let mut permuted = vec![E::OutputValueType::default(); n * dim];
    for (old_id, &new_id) in permutation.iter().enumerate() {
        let old_start = old_id * dim;
        let new_start = new_id * dim;
        permuted[new_start..new_start + dim]
            .copy_from_slice(&values[old_start..old_start + dim]);
    }

    DenseDataset::from_raw(permuted.into_boxed_slice(), n, dataset.encoder().clone())
}

pub fn permute_plain_sparse_dataset<C, V, D>(
    dataset: &PlainSparseDataset<C, V, D>,
    permutation: &[usize],
) -> PlainSparseDataset<C, V, D>
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + vectorium::SpaceUsage,
    V: vectorium::ValueType + vectorium::Float + FromF32 + vectorium::SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    let n = dataset.len();
    assert_eq!(permutation.len(), n);
    let old_id_by_new_id = invert_mapping(permutation);
    let mut permuted = PlainSparseDatasetGrowable::<C, V, D>::new(dataset.encoder().clone());

    for old_id in old_id_by_new_id {
        permuted.push(dataset.get(old_id as vectorium::VectorId));
    }

    permuted.into()
}

pub fn permute_scalar_sparse_dataset<C, V, D>(
    dataset: &ScalarSparseDataset<C, f32, V, D>,
    permutation: &[usize],
) -> ScalarSparseDataset<C, f32, V, D>
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + vectorium::SpaceUsage,
    V: vectorium::ValueType + vectorium::Float + FromF32 + vectorium::SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    let n = dataset.len();
    assert_eq!(permutation.len(), n);
    let old_id_by_new_id = invert_mapping(permutation);
    let dim = dataset.output_dim();
    let plain_encoder = PlainSparseQuantizer::<C, V, D>::new(dim, dim);
    let mut plain = PlainSparseDatasetGrowable::<C, V, D>::new(plain_encoder);

    for old_id in old_id_by_new_id {
        plain.push(dataset.get(old_id as vectorium::VectorId));
    }

    let plain: PlainSparseDataset<C, V, D> = plain.into();
    plain.relabel_as_scalar::<f32>()
}

pub fn permute_dotvbyte_dataset(
    dataset: &PackedSparseDataset<DotVByteFixedU8Encoder>,
    permutation: &[usize],
) -> PackedSparseDataset<DotVByteFixedU8Encoder> {
    let n = dataset.len();
    assert_eq!(permutation.len(), n);
    let old_id_by_new_id = invert_mapping(permutation);
    let mut permuted =
        PackedSparseDatasetGrowable::<DotVByteFixedU8Encoder>::new(dataset.encoder().clone());

    for old_id in old_id_by_new_id {
        let decoded = dataset
            .encoder()
            .decode_vector(dataset.get(old_id as vectorium::VectorId));
        let values: Vec<FixedU8Q> = decoded
            .values()
            .iter()
            .map(|&value| FixedU8Q::from_f32_saturating(value))
            .collect();
        let view = SparseVectorView::new(decoded.components(), values.as_slice());
        permuted.push(view);
    }

    permuted.into()
}

pub fn permute_dotpacking8_dataset(
    dataset: &PackedSparseDataset<DotPacking8ScalarU8Encoder>,
    permutation: &[usize],
) -> PackedSparseDataset<DotPacking8ScalarU8Encoder> {
    let n = dataset.len();
    assert_eq!(permutation.len(), n);
    let old_id_by_new_id = invert_mapping(permutation);
    // Reuse the already-trained encoder (same per-component scales), so re-encoding the decoded
    // f32 values reproduces the original u8 codes exactly -- a lossless reorder.
    let mut permuted =
        PackedSparseDatasetGrowable::<DotPacking8ScalarU8Encoder>::new(dataset.encoder().clone());

    for old_id in old_id_by_new_id {
        let decoded = dataset
            .encoder()
            .decode_vector(dataset.get(old_id as vectorium::VectorId));
        // DotPacking8 input values are f32, so push the decoded view directly (no remap).
        let view = SparseVectorView::new(decoded.components(), decoded.values());
        permuted.push(view);
    }

    permuted.into()
}

/// Decode a dp8 (u8-quantized) packed dataset into a plain f32 sparse dataset, preserving order.
///
/// Bridges the `dotpacking8-scalaru8` (dp8) on-disk format to the f32 source that the
/// `dotpacking8-sparse-scalar` (sparse-q) `ConvertFrom` expects: decoding u8 -> f32 and then
/// re-encoding with the same scalar-u8 quantizer reproduces the original u8 codes, so the
/// resulting sparse-q dataset is numerically identical to dp8 (only the in-memory layout
/// changes for faster queries).
pub fn decode_dotpacking8_to_plain_f32(
    dataset: PackedSparseDataset<DotPacking8ScalarU8Encoder>,
) -> PlainSparseDataset<u16, f32, DotProduct> {
    let n = dataset.len();
    let dim = dataset.output_dim();
    let mut plain = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(
        PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim),
    );

    for id in 0..n {
        let decoded = dataset
            .encoder()
            .decode_vector(dataset.get(id as vectorium::VectorId));
        let view = SparseVectorView::new(decoded.components(), decoded.values());
        plain.push(view);
    }

    plain.into()
}

pub fn permute_blocked_sparse_dataset(
    dataset: &PackedSparseDataset<BlockedSparseEncoder>,
    permutation: &[usize],
) -> PackedSparseDataset<BlockedSparseEncoder> {
    let n = dataset.len();
    assert_eq!(permutation.len(), n);
    let old_id_by_new_id = invert_mapping(permutation);
    // Reuse the already-built encoder, so re-encoding the decoded f32 values reproduces the
    // original f16 codes exactly -- a lossless reorder.
    let mut permuted =
        PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(dataset.encoder().clone());

    for old_id in old_id_by_new_id {
        let decoded = dataset
            .encoder()
            .decode_vector(dataset.get(old_id as vectorium::VectorId));
        // BlockedSparse stores f16 values, so narrow the decoded f32 back to f16 before pushing.
        let values: Vec<f16> = decoded
            .values()
            .iter()
            .map(|&value| f16::from_f32_saturating(value))
            .collect();
        let view = SparseVectorView::new(decoded.components(), values.as_slice());
        permuted.push(view);
    }

    permuted.into()
}

pub fn permute_dotpacking_sparse_dataset(
    dataset: &PackedSparseDataset<DotPackingSparseScalarU8Encoder>,
    permutation: &[usize],
) -> PackedSparseDataset<DotPackingSparseScalarU8Encoder> {
    let n = dataset.len();
    assert_eq!(permutation.len(), n);
    let old_id_by_new_id = invert_mapping(permutation);
    // Reuse the already-trained encoder (same per-component scales), so re-encoding the decoded
    // f32 values reproduces the original u8 codes exactly -- a lossless reorder.
    let mut permuted =
        PackedSparseDatasetGrowable::<DotPackingSparseScalarU8Encoder>::new(
            dataset.encoder().clone(),
        );

    for old_id in old_id_by_new_id {
        let decoded = dataset
            .encoder()
            .decode_vector(dataset.get(old_id as vectorium::VectorId));
        // DotPackingSparse input values are f32, so push the decoded view directly (no remap).
        let view = SparseVectorView::new(decoded.components(), decoded.values());
        permuted.push(view);
    }

    permuted.into()
}
