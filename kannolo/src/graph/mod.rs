pub mod egb;
pub mod graph;
pub mod neighbors;

pub use graph::{GraphFixedDegree, GraphTrait, GrowableGraph};
pub use neighbors::{NeighborData, Neighbors, PlainNeighbors};

/// Default graph type using plain (uncompressed) neighbor storage.
/// Used by existing binaries and the Python API for backward compatibility.
pub type Graph = graph::Graph<neighbors::PlainNeighbors>;
