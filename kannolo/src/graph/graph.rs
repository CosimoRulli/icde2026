use optional::Optioned;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vectorium::core::dataset::ScoredItemGeneric;
use vectorium::distances::Distance;
use vectorium::vector_encoder::{QueryEvaluator, VectorEncoder};
use vectorium::{Dataset, VectorId};

use crate::graph::neighbors::{NeighborData, Neighbors};
use crate::hnsw_utils::{
    PackedArray, from_max_heap_to_min_heap, invert_mapping, validate_permutation,
};
use crate::visited_set::{VisitedSet, create_visited_set};

/// A trait that defines the common interface for different graph implementations.
///
/// This allows graph indexes to be generic over the specific graph storage strategy.
/// Graph construction is handled through concrete type constructors and `Default`.
pub trait GraphTrait {
    /// Returns an iterator over the local IDs of the neighbors of node `u`.
    fn neighbors<'a>(&'a self, u: usize) -> impl Iterator<Item = usize> + 'a;

    /// Returns the number of nodes in the graph.
    #[must_use]
    fn n_nodes(&self) -> usize;

    /// Returns true if the graph is empty, false otherwise.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.n_nodes() == 0
    }

    /// Returns the number of edges in the graph.
    #[must_use]
    fn n_edges(&self) -> usize;

    /// Returns the maximum degree of any node in the graph.
    #[must_use]
    fn max_degree(&self) -> usize;

    /// Returns the external (original dataset) ID of a node given its local graph ID.
    /// If the graph has no external ID mapping, this function returns the local ID itself.
    #[must_use]
    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        id
    }

    /// Returns the memory space used by the graph structure in bytes.
    #[must_use]
    fn get_space_usage_bytes(&self) -> usize;

    /// Returns the memory space used by the neighbors storage in bytes.
    #[must_use]
    fn neighbors_size_bytes(&self) -> usize;

    /// Greedily searches for the single nearest neighbor to a query, starting from an `entry_point`.
    ///
    /// # Arguments
    /// * `dataset`: The dataset containing the vectors.
    /// * `query_evaluator`: An evaluator that can compute the distance from the query to any vector in the dataset.
    /// * `entry_point`: The candidate (`distance`, `id`) from which the search begins.
    ///
    /// # Returns
    /// The best `ScoredItemGeneric` found during the search.
    #[must_use]
    fn greedy_search_nearest<'e, D>(
        &self,
        dataset: &D,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        entry_point: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
    ) -> ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>
    where
        D: Dataset,
    {
        let mut nearest_id = entry_point.vector;
        let mut nearest_distance = entry_point.distance;
        let mut updated = true;

        while updated {
            updated = false;

            for neighbor in self.neighbors(nearest_id) {
                let vector_id = self.get_external_id(neighbor);
                let distance_neighbor =
                    query_evaluator.compute_distance(dataset.get(vector_id as VectorId));

                if distance_neighbor < nearest_distance {
                    nearest_distance = distance_neighbor;
                    nearest_id = neighbor;
                    updated = true;
                }
            }
        }

        ScoredItemGeneric {
            distance: nearest_distance,
            vector: nearest_id,
        }
    }

    /// Performs a greedy search on the graph to find the top `k` nearest neighbors.
    /// It uses a beam search-like approach, maintaining a list of candidates to visit (`ef`)
    /// and returning the `k` best results found.
    ///
    /// # Arguments
    /// * `dataset`: The dataset containing the vectors.
    /// * `starting_node`: The candidate from which the search begins.
    /// * `query_evaluator`: An evaluator that can compute distances to the query.
    /// * `k`: The number of nearest neighbors to return.
    /// * `ef`: The size of the dynamic candidate list during the search.
    /// * `lambda`: Relaxation parameter used for adaptive early stopping/admission.
    ///
    /// # Returns
    /// A `Vec` containing tuples of `(distance, id)` for the `k` nearest neighbors.
    #[must_use]
    fn greedy_search_topk<'e, D>(
        &self,
        dataset: &'e D,
        starting_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        k: usize,
        ef: usize,
        lambda: f32,
    ) -> Vec<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        let top_candidates = self.search_candidates_for_query(
            dataset,
            starting_node,
            query_evaluator,
            ef,
            k,
            lambda,
        );

        let mut top_k = top_candidates.into_sorted_vec();
        top_k.truncate(k);
        top_k
    }

    /// Search candidates for a query (uses efSearch and top-k pruning).
    #[must_use]
    fn search_candidates_for_query<'e, D>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef_search: usize,
        k: usize,
        lambda: f32,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        self.search_candidates_impl::<D>(
            dataset,
            entry_node,
            query_evaluator,
            ef_search,
            Some(k),
            lambda,
        )
    }

    /// Search candidates for insertion (uses efConstruction, no top-k pruning).
    #[must_use]
    fn search_candidates_for_insert<'e, D>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef_construction: usize,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        self.search_candidates_impl::<D>(
            dataset,
            entry_node,
            query_evaluator,
            ef_construction,
            None,
            0.0,
        )
    }

    /// Shared implementation for candidate search.
    #[must_use]
    fn search_candidates_impl<'e, D>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef: usize,
        k: Option<usize>,
        lambda: f32,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        // max-heap: We want to substitute worst result with a better one
        let mut top_candidates: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        // min-heap: We want to extract best candidate first to visit it
        let mut candidates: BinaryHeap<
            Reverse<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>,
        > = BinaryHeap::with_capacity(ef);

        let mut visited_table = create_visited_set(dataset.len(), ef);

        top_candidates.push(entry_node);
        candidates.push(Reverse(entry_node));

        visited_table.insert(entry_node.vector);

        while let Some(Reverse(node)) = candidates.pop() {
            let id_candidate = node.vector;
            let distance_candidate = node.distance;

            if top_candidates.len() >= ef.max(k.unwrap_or(0)) {
                // Standard HNSW termination: stop when the best remaining candidate
                // is worse than the worst result collected so far.
                let worst_top = &top_candidates.peek().unwrap().distance;
                if !distance_candidate.is_within_relaxation(worst_top, lambda) {
                    break;
                }
            }

            // TODO: prefetching dataset vectors for sparse neighbors is suboptimal because
            // dataset.range_from_id() does a random access into the offsets array. Storing
            // per-neighbor vector offsets directly in the graph would allow earlier prefetches.

            // Accumulate up to 6 unvisited neighbors for SIMD batch distance computation.
            let mut buf_local: [usize; 6] = [0; 6];
            let mut buf_ext: [VectorId; 6] = [0; 6];
            let mut count = 0usize;

            let mut admit = |local_id: usize, distance: <D::Encoder as VectorEncoder>::Distance| {
                let candidate = ScoredItemGeneric {
                    distance,
                    vector: local_id,
                };
                let should_add = if top_candidates.len() < ef {
                    true
                } else if let Some(top_node) = top_candidates.peek() {
                    candidate
                        .distance
                        .is_within_relaxation(&top_node.distance, lambda)
                } else {
                    false
                };
                if should_add {
                    candidates.push(Reverse(candidate));
                    top_candidates.push(candidate);
                }
                if top_candidates.len() > ef {
                    top_candidates.pop();
                }
            };

            // Collect neighbors into a stack buffer before prefetching dataset vectors.
            //
            // Without this buffering, the loop would interleave SVB decompression with
            // dataset vector accesses, causing the hardware prefetcher to thrash between
            // the two memory regions. By collecting all neighbor IDs first, we can issue
            // all vector prefetches in one batch, giving them maximum lead time before
            // the distance computation pass.
            const CHUNK_SIZE: usize = 64;
            let mut neighbors_buf: [usize; CHUNK_SIZE] = [0; CHUNK_SIZE];
            let mut chunk_idx = 0;

            for neighbor_local_id in self.neighbors(id_candidate) {
                neighbors_buf[chunk_idx] = neighbor_local_id;
                chunk_idx += 1;

                if chunk_idx == CHUNK_SIZE {
                    // Pass 1: prefetch all dataset vectors for this chunk.
                    for i in 0..CHUNK_SIZE {
                        let range = dataset
                            .range_from_id(self.get_external_id(neighbors_buf[i]) as VectorId);
                        dataset.prefetch_with_range(range);
                    }
                    // Pass 2: visited check, batch distance computation, admit.
                    for i in 0..CHUNK_SIZE {
                        let neighbor_local_id = neighbors_buf[i];
                        if !visited_table.contains(neighbor_local_id) {
                            visited_table.insert(neighbor_local_id);
                            buf_local[count] = neighbor_local_id;
                            buf_ext[count] = self.get_external_id(neighbor_local_id) as VectorId;
                            count += 1;
                            if count == 6 {
                                let dists = query_evaluator.compute_distances_batch6([
                                    dataset.get(buf_ext[0]),
                                    dataset.get(buf_ext[1]),
                                    dataset.get(buf_ext[2]),
                                    dataset.get(buf_ext[3]),
                                    dataset.get(buf_ext[4]),
                                    dataset.get(buf_ext[5]),
                                ]);
                                for j in 0..6 {
                                    admit(buf_local[j], dists[j]);
                                }
                                count = 0;
                            }
                        }
                    }
                    chunk_idx = 0;
                }
            }

            // Same two-pass pattern for the final partial chunk (common case for degree < 64).
            if chunk_idx > 0 {
                for i in 0..chunk_idx {
                    let range =
                        dataset.range_from_id(self.get_external_id(neighbors_buf[i]) as VectorId);
                    dataset.prefetch_with_range(range);
                }

                for i in 0..chunk_idx {
                    let neighbor_local_id = neighbors_buf[i];
                    if !visited_table.contains(neighbor_local_id) {
                        visited_table.insert(neighbor_local_id);
                        buf_local[count] = neighbor_local_id;
                        buf_ext[count] = self.get_external_id(neighbor_local_id) as VectorId;
                        count += 1;
                        if count == 6 {
                            let dists = query_evaluator.compute_distances_batch6([
                                dataset.get(buf_ext[0]),
                                dataset.get(buf_ext[1]),
                                dataset.get(buf_ext[2]),
                                dataset.get(buf_ext[3]),
                                dataset.get(buf_ext[4]),
                                dataset.get(buf_ext[5]),
                            ]);
                            for j in 0..6 {
                                admit(buf_local[j], dists[j]);
                            }
                            count = 0;
                        }
                    }
                }
            }
            // Flush remaining neighbours (fewer than 6).
            for i in 0..count {
                let d = query_evaluator.compute_distance(dataset.get(buf_ext[i]));
                admit(buf_local[i], d);
            }
        }
        top_candidates
    }

    /// Performs ACORN-1 filtered approximate nearest-neighbor search.
    ///
    /// Only vectors satisfying `predicate(external_id) == true` are returned.
    /// To maintain connectivity in sparse predicate sub-graphs, the search performs a
    /// two-hop neighbor expansion: when a direct neighbor does not satisfy the predicate,
    /// its own neighbors are also inspected ("jumping over" non-matching nodes).
    ///
    /// # Arguments
    /// * `dataset` – Dataset containing the raw vectors.
    /// * `entry_node` – `(distance, local_id)` starting point for the search.
    /// * `query_evaluator` – Computes distances from the query to any vector.
    /// * `ef` – Dynamic candidate list size (controls recall vs. speed).
    /// * `k` – Number of results requested (used for early-termination threshold).
    /// * `lambda` – Relaxation parameter for adaptive early stopping (`0.0` = standard HNSW).
    /// * `predicate` – Called with the **external** vector ID; returns `true` for eligible vectors.
    #[must_use]
    fn acorn_search_candidates_filtered<'e, D, F>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef: usize,
        k: usize,
        lambda: f32,
        predicate: &F,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
        F: Fn(usize) -> bool,
    {
        let mut top_candidates: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        let mut candidates: BinaryHeap<
            Reverse<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>,
        > = BinaryHeap::with_capacity(ef);

        let mut visited = create_visited_set(dataset.len(), ef);

        visited.insert(entry_node.vector);
        candidates.push(Reverse(entry_node));

        if predicate(self.get_external_id(entry_node.vector)) {
            top_candidates.push(entry_node);
        }

        let mut non_pred_direct: Vec<usize> = Vec::new();

        while let Some(Reverse(node)) = candidates.pop() {
            if top_candidates.len() >= ef.max(k) {
                let worst_top = top_candidates.peek().unwrap().distance;
                if !node.distance.is_within_relaxation(&worst_top, lambda) {
                    break;
                }
            }

            non_pred_direct.clear();

            for neighbor_local in self.neighbors(node.vector) {
                if visited.contains(neighbor_local) {
                    continue;
                }
                visited.insert(neighbor_local);

                let ext = self.get_external_id(neighbor_local);
                if predicate(ext) {
                    let d = query_evaluator.compute_distance(dataset.get(ext as VectorId));
                    let cand = ScoredItemGeneric {
                        distance: d,
                        vector: neighbor_local,
                    };
                    let should_add = if top_candidates.len() < ef {
                        true
                    } else if let Some(top) = top_candidates.peek() {
                        cand.distance.is_within_relaxation(&top.distance, lambda)
                    } else {
                        false
                    };
                    if should_add {
                        candidates.push(Reverse(cand));
                        top_candidates.push(cand);
                    }
                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }
                } else {
                    non_pred_direct.push(neighbor_local);
                }
            }

            // --- Phase 2: two-hop expansion (ACORN-1 core) ---
            for &mid_local in &non_pred_direct {
                for neighbor_local in self.neighbors(mid_local) {
                    if visited.contains(neighbor_local) {
                        continue;
                    }
                    visited.insert(neighbor_local);

                    let ext = self.get_external_id(neighbor_local);
                    if predicate(ext) {
                        let d = query_evaluator.compute_distance(dataset.get(ext as VectorId));
                        let cand = ScoredItemGeneric {
                            distance: d,
                            vector: neighbor_local,
                        };
                        let should_add = if top_candidates.len() < ef {
                            true
                        } else if let Some(top) = top_candidates.peek() {
                            cand.distance.is_within_relaxation(&top.distance, lambda)
                        } else {
                            false
                        };
                        if should_add {
                            candidates.push(Reverse(cand));
                            top_candidates.push(cand);
                        }
                        if top_candidates.len() > ef {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        top_candidates
    }

    /// ACORN-γ filtered search on a pre-expanded neighbor graph.
    ///
    /// Unlike [`acorn_search_candidates_filtered`], this method performs **no two-hop
    /// expansion** at query time. Designed for use with pre-expanded neighbor lists
    /// (e.g. `AcornGammaNeighbors`) where connectivity is already embedded.
    #[must_use]
    fn acorn_gamma_search_filtered<'e, D, F>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef: usize,
        k: usize,
        lambda: f32,
        predicate: &F,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
        F: Fn(usize) -> bool,
    {
        let mut top_candidates: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        let mut candidates: BinaryHeap<
            Reverse<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>,
        > = BinaryHeap::with_capacity(ef);

        let mut visited = create_visited_set(dataset.len(), ef);

        visited.insert(entry_node.vector);
        candidates.push(Reverse(entry_node));
        if predicate(self.get_external_id(entry_node.vector)) {
            top_candidates.push(entry_node);
        }

        while let Some(Reverse(node)) = candidates.pop() {
            if top_candidates.len() >= ef.max(k) {
                let worst_top = top_candidates.peek().unwrap().distance;
                if !node.distance.is_within_relaxation(&worst_top, lambda) {
                    break;
                }
            }

            for neighbor_local in self.neighbors(node.vector) {
                if visited.contains(neighbor_local) {
                    continue;
                }
                visited.insert(neighbor_local);

                let ext = self.get_external_id(neighbor_local);
                let d = query_evaluator.compute_distance(dataset.get(ext as VectorId));
                let cand = ScoredItemGeneric {
                    distance: d,
                    vector: neighbor_local,
                };

                let should_add = if top_candidates.len() < ef {
                    true
                } else if let Some(top) = top_candidates.peek() {
                    cand.distance.is_within_relaxation(&top.distance, lambda)
                } else {
                    false
                };

                if should_add {
                    candidates.push(Reverse(cand));
                    if predicate(ext) {
                        top_candidates.push(cand);
                        if top_candidates.len() > ef {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        top_candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorium::DenseDataset;
    use vectorium::core::dataset::ScoredItemGeneric;
    use vectorium::core::vector::DenseVectorView;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;

    /// Build a line graph of `n` nodes (each connected to i-1 and i+1).
    fn build_line_graph(n: usize, max_degree: usize) -> GrowableGraph {
        let mut g = GrowableGraph::with_max_degree(max_degree);
        g.reserve(n);
        g.advance_inserted_nodes(n);
        for i in 0..n {
            let mut nbrs: Vec<usize> = Vec::new();
            if i > 0 {
                nbrs.push(i - 1);
            }
            if i + 1 < n {
                nbrs.push(i + 1);
            }
            g.push_with_precomputed_reverse_links(None, &nbrs, i, &[]);
        }
        g
    }

    /// Regression test for the 2026-03-06 early-termination bug.
    ///
    /// `search_candidates_for_insert` passes `k=None` to `search_candidates_impl`.
    /// Before the fix, `if let Some(k_limit) = k && ...` never fired for k=None,
    /// so the loop drained the entire candidate queue. The fix uses `k.unwrap_or(0)`,
    /// making termination fire as soon as `top_candidates.len() >= ef`.
    ///
    /// This test verifies:
    /// 1. The returned heap is bounded by ef.
    /// 2. The true nearest neighbour is found (search is still correct after the fix).
    #[test]
    fn search_candidates_for_insert_bounded_by_ef_and_finds_nearest() {
        let n = 20usize;
        let ef = 5usize;

        // 1-D dataset: vectors [0.0], [1.0], ..., [19.0].
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        // Query at 10.0 — true nearest neighbour is node 10 (distance 0).
        let query_val = [10.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query);
        let entry_dist = evaluator.compute_distance(dataset.get(0));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 0usize,
        };

        let top = graph.search_candidates_for_insert(&dataset, entry, &evaluator, ef);

        assert!(top.len() <= ef, "heap size {} exceeds ef={}", top.len(), ef);

        let best = top.into_sorted_vec().into_iter().next().unwrap();
        assert_eq!(
            best.vector, 10,
            "expected nearest node 10, got {}",
            best.vector
        );
        assert_eq!(best.distance, SquaredEuclideanDistance::from(0.0));
    }

    /// Verify the `k=Some` path (`search_candidates_for_query`) is also bounded and correct.
    #[test]
    fn search_candidates_for_query_bounded_and_correct() {
        let n = 20usize;
        let ef = 6usize;
        let k = 3usize;

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        let query_val = [5.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query);
        let entry_dist = evaluator.compute_distance(dataset.get(0));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 0usize,
        };

        let top_heap = graph.search_candidates_for_query(&dataset, entry, &evaluator, ef, k, 0.0);

        assert!(top_heap.len() <= ef);
        let mut results = top_heap.into_sorted_vec();
        results.truncate(k);

        assert_eq!(
            results[0].vector, 5,
            "expected nearest node 5, got {}",
            results[0].vector
        );
        assert_eq!(results[0].distance, SquaredEuclideanDistance::from(0.0));
    }
}

/// A graph representation where the adjacency lists are owned by the `Neighbors`
/// implementation.
///
/// Each compression backend decides internally how to locate one node's list:
/// plain and bit-packed variants keep offsets, Elias-Fano keeps offsets, and
/// block-based encodings can use their own block index.
/// Node ids are represented as `u32` but they are returned as usize ones.
///
/// # Fields
/// - `neighbors`: The compressed or uncompressed neighbor storage.
/// - `ids_mapping`: Optional internal-to-external id mapping, used after graph remapping.
///
#[derive(Serialize, Deserialize)]
pub struct Graph<N: Neighbors> {
    neighbors: N,
    ids_mapping: Option<PackedArray>, // This is used to map the internal IDs to external IDs
    max_degree: usize,
    n_nodes: usize,
}

impl<N: Neighbors + Default> Default for Graph<N> {
    fn default() -> Self {
        Self {
            neighbors: N::default(),
            ids_mapping: None,
            max_degree: 0,
            n_nodes: 0,
        }
    }
}

impl<N> GraphTrait for Graph<N>
where
    N: Neighbors,
{
    #[inline]
    fn neighbors<'a>(&'a self, id: usize) -> impl Iterator<Item = usize> + 'a {
        self.neighbors.iter_node(id).map(|u| u as usize)
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.neighbors.len()
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            mapping.get(id)
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.byte_size();
        let ids_mapping_size = self.ids_mapping.as_ref().map_or(0, PackedArray::byte_size);

        neighbors_size + ids_mapping_size
    }

    fn neighbors_size_bytes(&self) -> usize {
        self.neighbors.byte_size()
    }
}

impl<N> From<GrowableGraph> for Graph<N>
where
    N: Neighbors + From<NeighborData>,
{
    fn from(growable_graph: GrowableGraph) -> Self {
        let n_nodes = growable_graph.n_nodes();
        let max_degree = growable_graph.max_degree();

        let mut neighbors = Vec::with_capacity(growable_graph.neighbors.len());
        let mut offsets = Vec::with_capacity(n_nodes + 1);

        offsets.push(0);
        for v in 0..n_nodes {
            let start = v * max_degree;
            let end = start + max_degree;
            // Collect only the non-None neighbors
            let mut cur_neighbors: Vec<u32> = growable_graph.neighbors[start..end]
                .iter()
                .filter_map(|&opt| {
                    if opt.is_some() {
                        Some(opt.unwrap())
                    } else {
                        None
                    }
                })
                .collect();

            cur_neighbors.sort_unstable();
            cur_neighbors.dedup();

            neighbors.extend(cur_neighbors);
            offsets.push(neighbors.len());
        }

        let final_mapping = growable_graph.ids_mapping.map(PackedArray::from);

        let neighbor_data = NeighborData {
            data: neighbors.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
        };

        Self {
            neighbors: N::from(neighbor_data),
            ids_mapping: final_mapping,
            max_degree,
            n_nodes,
        }
    }
}

impl<Nsrc, Ndst> From<&Graph<Nsrc>> for Graph<Ndst>
where
    Nsrc: Neighbors,
    Ndst: Neighbors + From<NeighborData>,
{
    fn from(src: &Graph<Nsrc>) -> Self {
        let n = src.n_nodes;

        let mut data = Vec::with_capacity(src.n_edges());
        let mut off = Vec::with_capacity(n + 1);
        off.push(0);

        for id in 0..n {
            let mut cur_neighbors: Vec<u32> = src.neighbors(id).map(|x| x as u32).collect();
            cur_neighbors.sort_unstable();
            cur_neighbors.dedup();

            data.extend(cur_neighbors);
            off.push(data.len());
        }

        let offsets = off.into_boxed_slice();

        Self {
            neighbors: Ndst::from(NeighborData {
                data: data.into_boxed_slice(),
                offsets,
            }),
            ids_mapping: src.ids_mapping.clone(),
            max_degree: src.max_degree,
            n_nodes: src.n_nodes,
        }
    }
}

#[cfg(test)]
mod graph_from_tests {
    use super::*;
    use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};

    fn make_plain_graph() -> Graph<PlainNeighbors> {
        // 5 nodi, liste variabili, includiamo anche liste vuote
        let data: Vec<u32> = vec![
            1, 3, // node 0
            0, 2, 4, // node 1
            // node 2 (empty)
            1, // node 3
            0, 2, 3, // node 4
        ];

        let offsets: Vec<usize> = vec![
            0, // start node 0
            2, // end node 0
            5, // end node 1
            5, // node 2 empty
            6, // node 3
            9, // node 4
        ];

        Graph {
            neighbors: PlainNeighbors::from(NeighborData {
                data: data.into_boxed_slice(),
                offsets: offsets.clone().into_boxed_slice(),
            }),
            ids_mapping: Some(PackedArray::from(vec![10, 11, 12, 13, 14])),
            max_degree: 3,
            n_nodes: 5,
        }
    }

    fn assert_same_graph<G1: GraphTrait, G2: GraphTrait>(g1: &G1, g2: &G2) {
        assert_eq!(g1.n_nodes(), g2.n_nodes());
        assert_eq!(g1.max_degree(), g2.max_degree());

        for id in 0..g1.n_nodes() {
            let a: Vec<_> = g1.neighbors(id).collect();
            let b: Vec<_> = g2.neighbors(id).collect();
            assert_eq!(a, b, "neighbors mismatch for node {id}");
        }
    }

    #[test]
    fn from_plain_to_plain() {
        let src = make_plain_graph();
        let dst: Graph<PlainNeighbors> = Graph::from(&src);

        assert_same_graph(&src, &dst);

        for i in 0..src.n_nodes() {
            assert_eq!(src.get_external_id(i), dst.get_external_id(i));
        }
    }

    #[test]
    fn from_plain_to_stream_vbyte() {
        let src = make_plain_graph();
        let dst: Graph<StreamVByteNeighbors> = Graph::from(&src);

        assert_same_graph(&src, &dst);
    }

    #[test]
    fn from_stream_vbyte_to_plain_roundtrip() {
        let src_plain = make_plain_graph();
        let src_svb: Graph<StreamVByteNeighbors> = Graph::from(&src_plain);

        let dst: Graph<PlainNeighbors> = Graph::from(&src_svb);

        assert_same_graph(&src_plain, &dst);
    }

    #[test]
    fn from_preserves_empty_lists() {
        let data: Vec<u32> = vec![
            1, 2, // node 0
            // node 1 empty
            3, // node 2
        ];

        let offsets: Vec<usize> = vec![0, 2, 2, 3];

        let src = Graph {
            neighbors: PlainNeighbors::from(NeighborData {
                data: data.into_boxed_slice(),
                offsets: offsets.clone().into_boxed_slice(),
            }),
            ids_mapping: None,
            max_degree: 2,
            n_nodes: 3,
        };

        let dst: Graph<StreamVByteNeighbors> = Graph::from(&src);
        let roundtrip: Graph<PlainNeighbors> = Graph::from(&dst);

        assert_same_graph(&src, &roundtrip);
    }
}

impl<Nsrc> Graph<Nsrc>
where
    Nsrc: Neighbors,
{
    pub fn remap_ground_and_compress_with_permutation<Ndst>(
        &self,
        mapping: &[usize],
    ) -> (Graph<Ndst>, Vec<usize>)
    where
        Ndst: Neighbors + From<NeighborData>,
    {
        let n = self.n_nodes;
        assert_eq!(
            mapping.len(),
            n,
            "invalid graph permutation length: expected {}, got {}",
            n,
            mapping.len()
        );
        validate_permutation(mapping).unwrap_or_else(|e| panic!("{e}"));

        let inv = invert_mapping(mapping);
        let mut all_neighbors = Vec::with_capacity(self.n_edges());
        let mut new_offsets = Vec::with_capacity(n + 1);
        let mut neigh = Vec::new();
        new_offsets.push(0);

        for new_id in 0..n {
            let old_id = inv[new_id];
            neigh.clear();

            neigh.extend(self.neighbors(old_id).map(|v| mapping[v] as u32));
            neigh.sort_unstable();
            neigh.dedup();

            all_neighbors.extend_from_slice(&neigh);
            new_offsets.push(all_neighbors.len());
        }

        let offsets = new_offsets.into_boxed_slice();

        let graph = Graph {
            neighbors: Ndst::from(NeighborData {
                data: all_neighbors.into_boxed_slice(),
                offsets,
            }),
            ids_mapping: None,
            max_degree: self.max_degree,
            n_nodes: n,
        };
        (graph, inv)
    }

    pub fn remap_level_and_compress_with_old_local_order<Ndst>(
        &self,
        old_locals_by_new_local: &[usize],
        global_mapping: &[usize],
    ) -> (Graph<Ndst>, Vec<usize>)
    where
        Ndst: Neighbors + From<NeighborData>,
    {
        let n = self.n_nodes;
        assert_eq!(
            old_locals_by_new_local.len(),
            n,
            "invalid upper-level local order length: expected {}, got {}",
            n,
            old_locals_by_new_local.len()
        );

        let mut seen = vec![false; n];
        for &old_local in old_locals_by_new_local {
            assert!(
                old_local < n,
                "invalid upper-level local id {} for level size {}",
                old_local,
                n
            );
            assert!(
                !seen[old_local],
                "duplicate upper-level local id {}",
                old_local
            );
            seen[old_local] = true;
        }

        let mut local_mapping = vec![0usize; n];
        for (new_local, &old_local) in old_locals_by_new_local.iter().enumerate() {
            local_mapping[old_local] = new_local;
        }

        let mut all_neighbors = Vec::with_capacity(self.n_edges());
        let mut new_offsets = Vec::with_capacity(n + 1);
        let mut new_external_ids = Vec::with_capacity(n);
        let mut neigh = Vec::new();
        new_offsets.push(0);

        for &old_local in old_locals_by_new_local {
            neigh.clear();
            neigh.extend(self.neighbors(old_local).map(|v| local_mapping[v] as u32));
            neigh.sort_unstable();
            neigh.dedup();

            all_neighbors.extend_from_slice(&neigh);
            new_offsets.push(all_neighbors.len());

            let old_global = self.get_external_id(old_local);
            assert!(
                old_global < global_mapping.len(),
                "invalid upper-level external id {} for global permutation length {}",
                old_global,
                global_mapping.len()
            );
            new_external_ids.push(global_mapping[old_global]);
        }

        let offsets = new_offsets.into_boxed_slice();

        (
            Graph {
                neighbors: Ndst::from(NeighborData {
                    data: all_neighbors.into_boxed_slice(),
                    offsets,
                }),
                ids_mapping: Some(PackedArray::from(new_external_ids)),
                max_degree: self.max_degree,
                n_nodes: n,
            },
            local_mapping,
        )
    }
}

/// A representation of a graph where the adjacency lists of the nodes are stored in a fixed degree format.
/// If a node's degree is less than the maximum degree, it is padded with `None` values.
/// None values are represented as `usize::MAX`. The nodes ids are in the range `[0, len)`
/// Node ids are represented as `u32` but they are returned as usize ones.
/// Moreover, the largest value is reserved. This means that we allow a
/// maximum of `u32::MAX - 1` nodes.
///
/// # Fields
/// - `neighbors`: A list of all neighbors for vectors in the graph. The neighbors for each vector
///   are stored in a contiguous block.
/// - `max_degree`: The maximum degree of any node in the graph.
/// - `n_edges`: The number of edges in the graph.
/// - `n_nodes`: The number of nodes in the graph.
///
#[derive(Serialize, Deserialize)]
pub struct GraphFixedDegree {
    neighbors: Box<[Optioned<u32>]>, // Using Optioned<u32> to represent neighbors, where None is represented by u32::MAX
    ids_mapping: Option<Box<[usize]>>, // This is used to map the internal IDs to external IDs
    max_degree: usize,
    n_edges: usize,
    n_nodes: usize,
}

impl Default for GraphFixedDegree {
    fn default() -> Self {
        GraphFixedDegree {
            neighbors: Box::new([]),
            ids_mapping: None, // No mapping by default
            max_degree: 0,
            n_edges: 0,
            n_nodes: 0,
        }
    }
}

impl GraphTrait for GraphFixedDegree {
    #[inline]
    fn neighbors<'a>(&'a self, u: usize) -> impl Iterator<Item = usize> + 'a {
        let start = u * self.max_degree;
        let end = start + self.max_degree;
        self.neighbors[start..end]
            .iter()
            .take_while(|&opt| opt.is_some())
            .map(|opt| opt.unwrap() as usize)
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            if id >= mapping.len() {
                panic!("ID out of bounds: {}", id);
            }
            mapping[id]
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<Optioned<u32>>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        neighbors_size + ids_mapping_size
    }

    fn neighbors_size_bytes(&self) -> usize {
        self.neighbors.len() * std::mem::size_of::<Optioned<u32>>()
    }
}

impl From<GrowableGraph> for GraphFixedDegree {
    /// Converts a `GrowableGraph` into a fixed-degree `GraphFixedDegree` (preserves padding).
    fn from(growable_graph: GrowableGraph) -> Self {
        let ids_mapping = growable_graph
            .ids_mapping
            .map(|mapping| mapping.into_boxed_slice());

        GraphFixedDegree {
            neighbors: growable_graph.neighbors.into_boxed_slice(),
            ids_mapping,
            max_degree: growable_graph.max_degree,
            n_edges: growable_graph.n_edges,
            n_nodes: growable_graph.n_nodes,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct GrowableGraph {
    neighbors: Vec<Optioned<u32>>, // Using Optioned<u32> to represent neighbors, where None is represented by u32::MAX
    ids_mapping: Option<Vec<usize>>, // This is used to map the internal IDs to external IDs
    max_degree: usize,
    n_edges: usize,
    n_nodes: usize,
    inserted_nodes: usize, // Number of nodes that have been actually inserted
}

impl GraphTrait for GrowableGraph {
    #[inline]
    fn neighbors<'a>(&'a self, u: usize) -> impl Iterator<Item = usize> + 'a {
        let start = u * self.max_degree;
        let end = start + self.max_degree;
        self.neighbors[start..end]
            .iter()
            .take_while(|&opt| opt.is_some())
            .map(|opt| opt.unwrap() as usize)
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            if id >= mapping.len() {
                panic!("ID out of bounds: {}", id);
            }
            mapping[id]
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<Optioned<u32>>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        neighbors_size + ids_mapping_size
    }

    fn neighbors_size_bytes(&self) -> usize {
        self.neighbors.len() * std::mem::size_of::<Optioned<u32>>()
    }
}

impl<N> From<Graph<N>> for GrowableGraph
where
    N: Neighbors,
{
    fn from(graph: Graph<N>) -> Self {
        let max_degree = graph.max_degree;
        let n_nodes = graph.n_nodes;
        let mut neighbors = Vec::with_capacity(n_nodes * max_degree);

        for v in 0..n_nodes {
            let mut degree = 0usize;
            for nbr in graph.neighbors.iter_node(v) {
                neighbors.push(Optioned::some(nbr));
                degree += 1;
            }

            let pad = max_degree.saturating_sub(degree);
            neighbors.extend((0..pad).map(|_| Optioned::none()));
        }

        let ids_mapping = graph.ids_mapping.map(|mapping| mapping.to_vec());

        GrowableGraph {
            neighbors,
            ids_mapping,
            max_degree,
            n_edges: graph.neighbors.len(),
            n_nodes,
            inserted_nodes: n_nodes,
        }
    }
}

impl From<GraphFixedDegree> for GrowableGraph {
    fn from(graph: GraphFixedDegree) -> Self {
        let ids_mapping = graph.ids_mapping.map(|mapping| mapping.into_vec());

        GrowableGraph {
            neighbors: graph.neighbors.into_vec(),
            ids_mapping,
            max_degree: graph.max_degree,
            n_edges: graph.n_edges,
            n_nodes: graph.n_nodes,
            inserted_nodes: graph.n_nodes,
        }
    }
}

impl GrowableGraph {
    /// Creates a new `GrowableGraph` with the specified maximum degree.
    #[must_use]
    pub fn with_max_degree(max_degree: usize) -> Self {
        GrowableGraph {
            neighbors: Vec::new(),
            ids_mapping: None, // No mapping by default
            max_degree,
            n_edges: 0,
            n_nodes: 0,
            inserted_nodes: 0, // No nodes inserted yet
        }
    }

    /// Returns the number of nodes that have been inserted into the graph.
    #[must_use]
    #[inline]
    pub fn inserted_nodes(&self) -> usize {
        self.inserted_nodes
    }

    /// Advances the count of inserted nodes by a given amount.
    /// This is used by the parallel builder to update the state after a batch is processed.
    pub fn advance_inserted_nodes(&mut self, count: usize) {
        self.inserted_nodes += count;
    }

    /// Pre-allocates space for a fixed number of nodes.
    pub fn reserve(&mut self, n_expected_nodes: usize) {
        self.neighbors = vec![Optioned::none(); n_expected_nodes * self.max_degree];
        self.n_nodes = n_expected_nodes; // The graph now has a fixed capacity
        self.ids_mapping = None; // No mapping by default
    }

    /// Sets the ID mapping for the graph, converting local IDs to external/original IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the mapping length does not match the number of nodes in the graph.
    pub fn set_mapping(&mut self, mapping: Vec<usize>) -> Result<(), String> {
        if mapping.len() != self.n_nodes {
            return Err(format!(
                "Mapping length mismatch: got {}, expected {}",
                mapping.len(),
                self.n_nodes
            ));
        }
        self.ids_mapping = Some(mapping);
        Ok(())
    }

    /// A version of push for the parallel builder that accepts pre-computed reverse links.
    pub fn push_with_precomputed_reverse_links(
        &mut self,
        external_id: Option<usize>,
        neighbors: &[usize],
        local_id: usize,
        reverse_links: &[(usize, Vec<usize>)], // (neighbor_id, new_neighbor_list_for_it)
    ) {
        // Add forward links
        let start = local_id * self.max_degree;
        for (i, &neighbor) in neighbors.iter().enumerate() {
            self.neighbors[start + i] = Optioned::some(neighbor as u32);
        }
        self.n_edges += neighbors.len();

        if let Some(vec_id) = external_id {
            if let Some(mapping) = self.ids_mapping.as_mut() {
                if local_id >= mapping.len() {
                    panic!(
                        "Attempted to write to local_id {} but ids_mapping len is {}",
                        local_id,
                        mapping.len()
                    );
                }
                mapping[local_id] = vec_id;
            } else {
                panic!("Attempted to set external ID for a graph without an ID mapping.");
            }
        } else {
            // If no external ID is provided, we assume the local ID is the external ID
            if let Some(mapping) = self.ids_mapping.as_mut() {
                if local_id >= mapping.len() {
                    panic!(
                        "Attempted to write to local_id {} but ids_mapping len is {}",
                        local_id,
                        mapping.len()
                    );
                }
                mapping[local_id] = local_id;
            }
        }

        // Add pre-computed reverse links
        for (neighbor_id, new_neighbor_list) in reverse_links {
            let start = *neighbor_id * self.max_degree;
            for (i, &n) in new_neighbor_list.iter().enumerate() {
                self.neighbors[start + i] = Optioned::some(n as u32);
            }
            // Pad with None
            for i in new_neighbor_list.len()..self.max_degree {
                self.neighbors[start + i] = Optioned::none();
            }
        }
    }

    pub fn precompute_reverse_links<D>(
        &self,
        dataset: &D,
        node_to_insert_local_id: usize,
        forward_neighbors: &[usize],
    ) -> Vec<(usize, Vec<usize>)>
    // (neighbor_local_id, new_neighbor_list_for_it)
    where
        D: Dataset + Sync,
    {
        let mut reverse_links_data = Vec::with_capacity(forward_neighbors.len());

        for &neighbor_local_id in forward_neighbors {
            let neighbor_vector_id = self.get_external_id(neighbor_local_id) as VectorId;

            // 1. Build a max-heap containing the neighbor's current neighbors and the new node.
            //    The distances are all relative to `neighbor_vector_id`.
            let mut closest_vectors = BinaryHeap::<
                ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
            >::new();

            // Add its current neighbors
            for local_id in self.neighbors(neighbor_local_id) {
                let vector_id = self.get_external_id(local_id) as VectorId;
                let dist = dataset.encoder().compute_distance_between(
                    dataset.get(neighbor_vector_id),
                    dataset.get(vector_id),
                );
                closest_vectors.push(ScoredItemGeneric {
                    distance: dist,
                    vector: local_id,
                });
            }

            // Add the new reverse link (the node we are inserting)
            let node_to_insert_vector_id =
                self.get_external_id(node_to_insert_local_id) as VectorId;
            let dist_to_inserted_node = dataset.encoder().compute_distance_between(
                dataset.get(neighbor_vector_id),
                dataset.get(node_to_insert_vector_id),
            );
            closest_vectors.push(ScoredItemGeneric {
                distance: dist_to_inserted_node,
                vector: node_to_insert_local_id,
            });

            // 2. Use the robust `shrink_neighbor_list` heuristic to prune the list.
            let new_neighbor_list =
                self.shrink_neighbor_list(dataset, &mut closest_vectors, self.max_degree);

            reverse_links_data.push((neighbor_local_id, new_neighbor_list));
        }
        reverse_links_data
    }

    pub fn shrink_neighbor_list<D>(
        &self,
        dataset: &D,
        closest_vectors: &mut BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        >,
        max_size: usize,
    ) -> Vec<usize>
    where
        D: Dataset + Sync,
    {
        if closest_vectors.len() <= max_size {
            return closest_vectors
                .iter()
                .map(|candidate| candidate.vector)
                .collect();
        }

        let mut min_heap = from_max_heap_to_min_heap(closest_vectors);
        let mut new_closest_vectors: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        while let Some(node) = min_heap.pop() {
            let node1 = node.0;
            let mut keep_node_1 = true;

            // The robust pruning heuristic from the paper.
            // For each candidate, check if it is closer to the query than it is to any
            // other candidate already in the result set.
            for node2 in new_closest_vectors.iter() {
                let node1_vector_id = self.get_external_id(node1.vector) as VectorId;
                let node2_vector_id = self.get_external_id(node2.vector) as VectorId;
                let dist_node_1_node2 = dataset.encoder().compute_distance_between(
                    dataset.get(node1_vector_id),
                    dataset.get(node2_vector_id),
                );
                if dist_node_1_node2 < node1.distance {
                    keep_node_1 = false;
                    break;
                }
            }

            if keep_node_1 {
                new_closest_vectors.push(node1);
                if new_closest_vectors.len() >= max_size {
                    return new_closest_vectors.iter().map(|c| c.vector).collect();
                }
            }
        }

        // Return the IDs of the closest vectors
        new_closest_vectors
            .iter()
            .map(|candidate| candidate.vector)
            .collect()
    }

    /// Finds and prunes neighbors for a new node and computes the necessary reverse links.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `Vec<usize>`: The pruned forward neighbors for the new node.
    /// - `Vec<(usize, Vec<usize>)>`: The pre-computed reverse links for existing neighbors.
    /// - `ScoredItemGeneric`: The best candidate found, to be used as the entry point for the next lower level.
    #[must_use]
    pub fn find_and_prune_neighbors<'e, D>(
        &self,
        dataset: &'e D,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        ef_construction: usize,
        m: usize,
        future_local_id: usize,
    ) -> (
        Vec<usize>,
        Vec<(usize, Vec<usize>)>,
        ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
    )
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        // 1. Get candidate neighbors
        let mut neighbors_nodes = self.search_candidates_for_insert(
            dataset,
            entry_node,
            query_evaluator,
            ef_construction,
        );

        // The new entry point for the next level is the best candidate we found.
        let new_entry_node = *neighbors_nodes.peek().unwrap();

        // 2. Prune with heuristic
        let forward_neighbors = self.shrink_neighbor_list(dataset, &mut neighbors_nodes, m);

        // 3. Compute reverse links with the PRUNED list
        let reverse_links =
            self.precompute_reverse_links(dataset, future_local_id, &forward_neighbors);

        (forward_neighbors, reverse_links, new_entry_node)
    }
}
