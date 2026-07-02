use dsi_bitstream::prelude::{
    BitReader, BitSeek, BitWrite, BufBitWriter, LE, MemWordReader, MemWordWriterVec, ZetaRead,
    ZetaWrite,
};
use postcard;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use tantivy_bitpacker::{BitPacker, BitUnpacker};
use toolkit::elias_fano::EliasFano;
use toolkit::stream_vbyte::StreamVByteBlocks;

/// Stores raw neighbor data and offsets
#[derive(Clone)]
pub struct NeighborData {
    pub data: Box<[u32]>,      // neighbor IDs
    pub offsets: Box<[usize]>, // segment offsets
}

/// Trait for neighbor arrays (plain or compressed)
pub trait Neighbors {
    type Iter<'a>: Iterator<Item = u32> + 'a
    where
        Self: 'a;

    fn len(&self) -> usize;
    fn n_nodes(&self) -> usize;

    fn iter_node<'a>(&'a self, node_id: usize) -> Self::Iter<'a>;

    fn byte_size(&self) -> usize;
}

/// Plain neighbors stored as Box<[u32]>
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct PlainNeighbors {
    data: Box<[u32]>,
    offsets: Box<[usize]>,
}

impl From<NeighborData> for PlainNeighbors {
    fn from(nd: NeighborData) -> Self {
        PlainNeighbors {
            data: nd.data,
            offsets: nd.offsets,
        }
    }
}

pub struct PlainNeighborsIter<'a> {
    slice: std::slice::Iter<'a, u32>,
}

impl<'a> Iterator for PlainNeighborsIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        self.slice.next().copied()
    }
}

impl Neighbors for PlainNeighbors {
    type Iter<'a> = PlainNeighborsIter<'a>;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn n_nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn iter_node<'a>(&'a self, node_id: usize) -> Self::Iter<'a> {
        let start = self.offsets[node_id];
        let end = self.offsets[node_id + 1];
        PlainNeighborsIter {
            slice: self.data[start..end].iter(),
        }
    }

    fn byte_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<u32>()
            + self.offsets.len() * std::mem::size_of::<usize>()
    }
}

/// Elias-Fano encoded offsets (compressed variable-length encoding)
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct EliasFanoOffsets {
    ef: EliasFano, // Elias-Fano structure
}

impl From<Box<[usize]>> for EliasFanoOffsets {
    fn from(data: Box<[usize]>) -> Self {
        Self::from_slice(&data)
    }
}

impl EliasFanoOffsets {
    pub fn from_slice(offsets: &[usize]) -> Self {
        let mut vec = Vec::with_capacity(offsets.len());
        for (i, &v) in offsets.iter().enumerate() {
            vec.push(v + i);
        }
        EliasFanoOffsets {
            ef: EliasFano::from(&vec),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ef.len()
    }

    #[inline]
    pub fn get(&self, i: usize) -> usize {
        self.ef.select(i).unwrap() - i
    }

    #[inline]
    pub fn try_get(&self, i: usize) -> Option<usize> {
        self.ef.select(i).map(|v| v - i)
    }

    pub fn byte_size(&self) -> usize {
        postcard::to_allocvec(&self.ef).unwrap().len()
    }
}

/// Stream VByte compressed independently per adjacency list.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StreamVByteNeighbors {
    blocks: StreamVByteBlocks<u32>,
    empty_nodes: Box<[u64]>,
    logical_len: usize,
}

const STREAMVBYTE_BLOCKS_SMALL_BUF: usize = 64;
const STREAMVBYTE_BLOCKS_MAX_BUF: usize = 256;

pub enum StreamVByteIter {
    Empty,
    Small {
        buf: [u32; STREAMVBYTE_BLOCKS_SMALL_BUF],
        pos: usize,
        len: usize,
        acc: u32,
    },
    Large {
        buf: [u32; STREAMVBYTE_BLOCKS_MAX_BUF],
        pos: usize,
        len: usize,
        acc: u32,
    },
}

impl StreamVByteIter {
    #[inline]
    fn empty() -> Self {
        Self::Empty
    }

    #[inline]
    fn small() -> Self {
        Self::Small {
            buf: [0; STREAMVBYTE_BLOCKS_SMALL_BUF],
            pos: 0,
            len: 0,
            acc: 0,
        }
    }

    #[inline]
    fn large() -> Self {
        Self::Large {
            buf: [0; STREAMVBYTE_BLOCKS_MAX_BUF],
            pos: 0,
            len: 0,
            acc: 0,
        }
    }
}

impl Iterator for StreamVByteIter {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            StreamVByteIter::Empty => None,
            StreamVByteIter::Small { buf, pos, len, acc } => {
                if *pos >= *len {
                    return None;
                }

                *acc = acc.wrapping_add(buf[*pos]);
                *pos += 1;
                Some(*acc)
            }
            StreamVByteIter::Large { buf, pos, len, acc } => {
                if *pos >= *len {
                    return None;
                }

                *acc = acc.wrapping_add(buf[*pos]);
                *pos += 1;
                Some(*acc)
            }
        }
    }
}

impl From<NeighborData> for StreamVByteNeighbors {
    fn from(nd: NeighborData) -> Self {
        let mut deltas = Vec::with_capacity(nd.data.len().max(nd.offsets.len().saturating_sub(1)));
        let mut offsets = Vec::with_capacity(nd.offsets.len());
        let n_nodes = nd.offsets.len().saturating_sub(1);
        let mut empty_nodes = vec![0u64; n_nodes.div_ceil(64)];
        let mut has_empty_nodes = false;

        offsets.push(0);
        for node_id in 0..n_nodes {
            let start = nd.offsets[node_id];
            let end = nd.offsets[node_id + 1];
            if start == end {
                has_empty_nodes = true;
                empty_nodes[node_id / 64] |= 1u64 << (node_id % 64);
                deltas.push(0);
                offsets.push(deltas.len());
                continue;
            }

            assert!(
                end - start <= 256,
                "StreamVByteBlocks supports adjacency lists up to 256 neighbors"
            );

            let mut prev = 0u32;
            for &value in &nd.data[start..end] {
                debug_assert!(
                    value >= prev,
                    "Neighbor list must be sorted in ascending order"
                );
                deltas.push(value - prev);
                prev = value;
            }
            offsets.push(deltas.len());
        }

        StreamVByteNeighbors {
            blocks: StreamVByteBlocks::new(&deltas, &offsets),
            empty_nodes: if has_empty_nodes {
                empty_nodes.into_boxed_slice()
            } else {
                Box::new([])
            },
            logical_len: nd.data.len(),
        }
    }
}

impl StreamVByteNeighbors {
    #[inline]
    fn is_empty_node(&self, node_id: usize) -> bool {
        !self.empty_nodes.is_empty()
            && self
                .empty_nodes
                .get(node_id / 64)
                .is_some_and(|word| (word & (1u64 << (node_id % 64))) != 0)
    }
}

impl Neighbors for StreamVByteNeighbors {
    type Iter<'a> = StreamVByteIter;

    fn len(&self) -> usize {
        self.logical_len
    }

    fn n_nodes(&self) -> usize {
        self.blocks.num_blocks()
    }

    fn iter_node<'a>(&'a self, node_id: usize) -> Self::Iter<'a> {
        if self.is_empty_node(node_id) {
            return StreamVByteIter::empty();
        }

        if self.blocks.block_len(node_id) <= STREAMVBYTE_BLOCKS_SMALL_BUF {
            let mut iter = StreamVByteIter::small();
            if let StreamVByteIter::Small { buf, len, .. } = &mut iter {
                *len = self.blocks.get_block(node_id, buf);
            }
            iter
        } else {
            let mut iter = StreamVByteIter::large();
            if let StreamVByteIter::Large { buf, len, .. } = &mut iter {
                *len = self.blocks.get_block(node_id, buf);
            }
            iter
        }
    }

    fn byte_size(&self) -> usize {
        postcard::to_allocvec(&self.blocks).unwrap().len()
            + self.empty_nodes.len() * std::mem::size_of::<u64>()
    }
}

fn write_zeta_segment(
    writer: &mut impl ZetaWrite<LE>,
    node_id: usize,
    neighbors: &[u32],
    codec_k: usize,
) -> usize {
    if neighbors.is_empty() {
        return 0;
    }

    let first_delta = neighbors[0] as i64 - node_id as i64;
    let mut written_bits = writer.write_zeta(i64_to_nat(first_delta), codec_k).unwrap();

    let mut prev = neighbors[0];
    for &value in &neighbors[1..] {
        debug_assert!(
            value >= prev,
            "Neighbor list must be sorted in ascending order"
        );
        written_bits += writer
            .write_zeta((value - prev - 1) as u64, codec_k)
            .unwrap();
        prev = value;
    }

    written_bits
}

#[inline]
fn i64_to_nat(value: i64) -> u64 {
    if value >= 0 {
        (value as u64) << 1
    } else {
        ((-value) as u64) * 2 - 1
    }
}

#[inline]
fn nat_to_i64(value: u64) -> i64 {
    if value & 1 == 0 {
        (value >> 1) as i64
    } else {
        -((value >> 1) as i64) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nd(data: Vec<u32>, offsets: Vec<usize>) -> NeighborData {
        NeighborData {
            data: data.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
        }
    }

    fn collect_all<N: Neighbors>(n: &N) -> Vec<u32> {
        let mut out = Vec::new();
        for node_id in 0..n.n_nodes() {
            out.extend(n.iter_node(node_id));
        }
        out
    }

    fn collect_node<N: Neighbors>(n: &N, node_id: usize) -> Vec<u32> {
        n.iter_node(node_id).collect()
    }

    fn expected_concat_segments(data: &[u32], offsets: &[usize]) -> Vec<u32> {
        let mut out = Vec::new();
        if offsets.len() < 2 {
            return out;
        }
        for w in offsets.windows(2) {
            out.extend_from_slice(&data[w[0]..w[1]]);
        }
        out
    }

    fn assert_neighbor_impl_matches_plain_flat<N>(neighbors: N, data: &[u32], offsets: &[usize])
    where
        N: Neighbors,
    {
        let expected = expected_concat_segments(data, offsets);

        assert_eq!(neighbors.len(), expected.len());
        assert_eq!(collect_all(&neighbors), expected);
        assert_eq!(neighbors.n_nodes(), offsets.len().saturating_sub(1));
    }

    fn assert_neighbor_impl_matches_plain_by_segments<N>(
        neighbors: N,
        data: &[u32],
        offsets: &[usize],
    ) where
        N: Neighbors,
    {
        let expected = expected_concat_segments(data, offsets);
        assert_eq!(neighbors.len(), expected.len());

        if offsets.len() < 2 {
            assert!(collect_all(&neighbors).is_empty());
            return;
        }

        for (node_id, w) in offsets.windows(2).enumerate() {
            let start = w[0];
            let end = w[1];
            assert_eq!(
                collect_node(&neighbors, node_id),
                data[start..end].to_vec(),
                "segment mismatch for node {node_id}"
            );
        }
    }

    #[test]
    fn plain_neighbors_full_and_ranges() {
        let raw = nd(vec![1, 3, 7, 2, 4, 9], vec![0, 3, 6]);
        let plain = PlainNeighbors::from(raw.clone());

        assert_eq!(plain.len(), 6);
        assert_eq!(collect_all(&plain), vec![1, 3, 7, 2, 4, 9]);
        assert_eq!(collect_node(&plain, 0), vec![1, 3, 7]);
        assert_eq!(collect_node(&plain, 1), vec![2, 4, 9]);
        assert_eq!(
            plain.byte_size(),
            6 * std::mem::size_of::<u32>() + 3 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn elias_fano_offsets_roundtrip() {
        let raw = vec![0usize, 3, 7, 7, 10, 15].into_boxed_slice();
        let ef = EliasFanoOffsets::from(raw.clone());

        assert_eq!(ef.len(), raw.len());
        for i in 0..raw.len() {
            assert_eq!(ef.get(i), raw[i], "offset mismatch at index {i}");
        }
        assert!(ef.byte_size() > 0 || ef.len() == 0);
    }

    #[test]
    fn stream_vbyte_neighbors_match_plain_single_segment() {
        let raw = nd(vec![1, 3, 7, 15, 31], vec![0, 5]);
        let svb = StreamVByteNeighbors::from(raw.clone());

        assert_neighbor_impl_matches_plain_by_segments(svb, &raw.data, &raw.offsets);
    }

    #[test]
    fn stream_vbyte_neighbors_match_plain_multiple_segments() {
        let raw = nd(vec![2, 4, 8, 9, 3, 5, 11, 100, 101], vec![0, 4, 7, 9]);
        let svb = StreamVByteNeighbors::from(raw.clone());

        assert_neighbor_impl_matches_plain_by_segments(svb, &raw.data, &raw.offsets);
    }

    #[test]
    fn stream_vbyte_neighbors_with_empty_segments() {
        let raw = nd(vec![1, 5, 9, 20], vec![0, 0, 2, 2, 4]);
        let svb = StreamVByteNeighbors::from(raw.clone());

        assert_neighbor_impl_matches_plain_by_segments(svb, &raw.data, &raw.offsets);
    }

    #[test]
    fn empty_neighbors_are_supported() {
        let raw = nd(vec![], vec![0]);

        let plain = PlainNeighbors::from(raw.clone());
        //let bp = BitPackedNeighbors::from(raw.clone());
        let svb = StreamVByteNeighbors::from(raw.clone());

        assert_eq!(plain.len(), 0);
        assert_eq!(svb.len(), 0);

        assert!(collect_all(&plain).is_empty());
        assert!(collect_all(&svb).is_empty());
    }

    #[test]
    fn default_stream_vbyte_is_empty() {
        let svb = StreamVByteNeighbors::default();
        assert_eq!(svb.len(), 0);
        assert!(collect_all(&svb).is_empty());
    }

    #[test]
    fn byte_size_is_nonzero_for_nonempty_structures() {
        let raw = nd(vec![1, 2, 3, 8, 13, 21], vec![0, 3, 6]);

        let plain = PlainNeighbors::from(raw.clone());
        let svb = StreamVByteNeighbors::from(raw.clone());

        assert!(plain.byte_size() > 0);
        assert!(svb.byte_size() > 0);
    }
}
