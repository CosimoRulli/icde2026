use clap::{Parser, ValueEnum};
use half::f16;
use num_traits::FromPrimitive;
use serde::Deserialize;

use kannolo::graph::egb::permute_graph_bisection_from_graph;
use kannolo::graph::graph::{Graph, GraphTrait};
use kannolo::graph::neighbors::PlainNeighbors;
use kannolo::hnsw::HNSW;
use kannolo::hnsw_utils::validate_permutation;

use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::blocked_sparse::BlockedSparseEncoder;
use vectorium::encoders::dotpacking8_scalaru8::DotPacking8ScalarU8Encoder;
use vectorium::encoders::dotpackingsparse_scalaru8::DotPackingSparseScalarU8Encoder;
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::{
    ComponentType, DenseDataset, FixedU8Q, FixedU16Q, Float, FromF32, IndexSerializer,
    PackedSparseDataset, PlainSparseDataset, ScalarSparseDataset, SpaceUsage, ValueType,
};

#[derive(Debug, Clone, ValueEnum)]
enum DatasetType {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum ValueTypeArg {
    F16,
    #[default]
    F32,
    Fixedu8,
    Fixedu16,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum EncoderType {
    #[default]
    Plain,
    Pq,
    Dotvbyte,
    #[value(name = "dotpacking8-scalaru8")]
    Dotpacking8Scalaru8,
    #[value(name = "blocked-sparse")]
    BlockedSparse,
    #[value(name = "dotpacking8-sparse-scalar")]
    DotpackingSparseScalaru8,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum ComponentTypeArg {
    #[default]
    U16,
    U32,
}

#[derive(Clone, Copy, Debug)]
enum DistanceKind {
    Euclidean,
    DotProduct,
}

fn parse_metric(metric: &str) -> DistanceKind {
    match metric {
        "euclidean" | "l2" => DistanceKind::Euclidean,
        "dotproduct" | "ip" => DistanceKind::DotProduct,
        _ => {
            eprintln!(
                "Error: Invalid distance type. Choose between 'euclidean' and 'dotproduct'."
            );
            std::process::exit(1);
        }
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the index.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// Output .npy file for the EGB permutation (old_id -> new_id).
    #[clap(short, long, value_parser)]
    output_file: String,

    /// The type of vectors (dense or sparse).
    #[clap(long, value_enum)]
    dataset_type: DatasetType,

    /// Value type for stored values. Dense plain: f32, f16. Sparse plain: f32, f16, fixedu8, fixedu16.
    /// Ignored for dotvbyte and pq.
    #[clap(long = "value-type", value_enum)]
    #[arg(default_value_t = ValueTypeArg::F32)]
    value_type: ValueTypeArg,

    /// Component type for sparse datasets (`u16` or `u32`).
    /// DotVByte currently supports only `u16`.
    #[clap(long = "component-type", value_enum)]
    #[arg(default_value_t = ComponentTypeArg::U16)]
    component_type: ComponentTypeArg,

    /// Encoder type. Dense: plain, pq. Sparse: plain, dotvbyte.
    #[clap(long, value_enum)]
    #[arg(default_value_t = EncoderType::Plain)]
    encoder: EncoderType,

    /// The distance metric ("euclidean" or "dotproduct").
    #[clap(long, value_parser)]
    distance: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    pq_subspaces: usize,
}

fn main() {
    let args = Args::parse();

    match (&args.dataset_type, &args.encoder) {
        (DatasetType::Sparse, EncoderType::Pq) => {
            eprintln!("Error: PQ encoder is only available for dense vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Dotvbyte) => {
            eprintln!("Error: DotVByte encoder is only available for sparse vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Dotpacking8Scalaru8) => {
            eprintln!("Error: DotPacking8 encoder is only available for sparse vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::BlockedSparse) => {
            eprintln!("Error: BlockedSparse encoder is only available for sparse vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::DotpackingSparseScalaru8) => {
            eprintln!("Error: DotPackingSparse encoder is only available for sparse vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Plain)
            if matches!(
                args.value_type,
                ValueTypeArg::Fixedu8 | ValueTypeArg::Fixedu16
            ) =>
        {
            eprintln!("Error: fixedu8/fixedu16 value types are only available for sparse vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, _) if !matches!(args.component_type, ComponentTypeArg::U16) => {
            eprintln!("Error: component-type is only applicable to sparse datasets.");
            std::process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte)
            if !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!("Error: DotVByte encoder supports only component-type u16.");
            std::process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::Dotpacking8Scalaru8)
            if !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!("Error: DotPacking8 encoder supports only component-type u16.");
            std::process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::BlockedSparse)
            if !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!("Error: BlockedSparse encoder supports only component-type u16.");
            std::process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::DotpackingSparseScalaru8)
            if !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!("Error: DotPackingSparse encoder supports only component-type u16.");
            std::process::exit(1);
        }
        _ => {}
    }

    let metric = parse_metric(&args.distance);

    match (&args.dataset_type, &args.encoder, &args.value_type) {
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32) => {
            dense_plain_f32(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16) => {
            dense_plain_f16(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Pq, _) => dense_pq(&args, metric),
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32) => {
            sparse_plain_f32(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16) => {
            sparse_plain_f16(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu8) => {
            sparse_scalar::<FixedU8Q>(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu16) => {
            sparse_scalar::<FixedU16Q>(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte, _) => sparse_dotvbyte(&args, metric),
        (DatasetType::Sparse, EncoderType::Dotpacking8Scalaru8, _) => {
            sparse_dotpacking8(&args, metric)
        }
        (DatasetType::Sparse, EncoderType::BlockedSparse, _) => {
            sparse_blocked_sparse(&args, metric)
        }
        (DatasetType::Sparse, EncoderType::DotpackingSparseScalaru8, _) => {
            sparse_dotpacking_sparse(&args, metric)
        }
        (DatasetType::Dense, EncoderType::Dotvbyte, _)
        | (DatasetType::Dense, EncoderType::Dotpacking8Scalaru8, _)
        | (DatasetType::Dense, EncoderType::BlockedSparse, _)
        | (DatasetType::Dense, EncoderType::DotpackingSparseScalaru8, _)
        | (DatasetType::Sparse, EncoderType::Pq, _)
        | (
            DatasetType::Dense,
            EncoderType::Plain,
            ValueTypeArg::Fixedu8 | ValueTypeArg::Fixedu16,
        ) => unreachable!(),
    }
}

fn dense_plain_f32(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => dense_plain_f32_with_distance::<SquaredEuclideanDistance>(args),
        DistanceKind::DotProduct => dense_plain_f32_with_distance::<DotProduct>(args),
    }
}

fn dense_plain_f32_with_distance<D>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
{
    let index: HNSW<DenseDataset<PlainDenseQuantizer<f32, D>>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn dense_plain_f16(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => dense_plain_f16_with_distance::<SquaredEuclideanDistance>(args),
        DistanceKind::DotProduct => dense_plain_f16_with_distance::<DotProduct>(args),
    }
}

fn dense_plain_f16_with_distance<D>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
{
    let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, D>>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn dense_pq(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => dense_pq_with_distance::<SquaredEuclideanDistance>(args),
        DistanceKind::DotProduct => dense_pq_with_distance::<DotProduct>(args),
    }
}

fn dense_pq_with_distance<D>(args: &Args)
where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance,
{
    match args.pq_subspaces {
        4 => dense_pq_with_subspaces::<4, D>(args),
        8 => dense_pq_with_subspaces::<8, D>(args),
        16 => dense_pq_with_subspaces::<16, D>(args),
        32 => dense_pq_with_subspaces::<32, D>(args),
        48 => dense_pq_with_subspaces::<48, D>(args),
        64 => dense_pq_with_subspaces::<64, D>(args),
        96 => dense_pq_with_subspaces::<96, D>(args),
        128 => dense_pq_with_subspaces::<128, D>(args),
        192 => dense_pq_with_subspaces::<192, D>(args),
        256 => dense_pq_with_subspaces::<256, D>(args),
        384 => dense_pq_with_subspaces::<384, D>(args),
        _ => {
            eprintln!(
                "Error: Invalid pq-subspaces value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384."
            );
            std::process::exit(1);
        }
    }
}

fn dense_pq_with_subspaces<const M: usize, D>(args: &Args)
where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance,
{
    let index: HNSW<DenseDataset<ProductQuantizer<M, D>>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn sparse_plain_f32(args: &Args, metric: DistanceKind) {
    match args.component_type {
        ComponentTypeArg::U16 => sparse_plain_f32_with_component::<u16>(args, metric),
        ComponentTypeArg::U32 => sparse_plain_f32_with_component::<u32>(args, metric),
    }
}

fn sparse_plain_f32_with_component<C>(args: &Args, metric: DistanceKind)
where
    C: ComponentType + FromPrimitive + SpaceUsage + for<'de> Deserialize<'de>,
{
    match metric {
        DistanceKind::Euclidean => {
            sparse_plain_f32_with_distance::<C, SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => sparse_plain_f32_with_distance::<C, DotProduct>(args),
    }
}

fn sparse_plain_f32_with_distance<C, D>(args: &Args)
where
    C: ComponentType + FromPrimitive + SpaceUsage + for<'de> Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
{
    let index: HNSW<PlainSparseDataset<C, f32, D>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn sparse_plain_f16(args: &Args, metric: DistanceKind) {
    match args.component_type {
        ComponentTypeArg::U16 => sparse_plain_f16_with_component::<u16>(args, metric),
        ComponentTypeArg::U32 => sparse_plain_f16_with_component::<u32>(args, metric),
    }
}

fn sparse_plain_f16_with_component<C>(args: &Args, metric: DistanceKind)
where
    C: ComponentType + FromPrimitive + SpaceUsage + for<'de> Deserialize<'de>,
{
    match metric {
        DistanceKind::Euclidean => {
            sparse_plain_f16_with_distance::<C, SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => sparse_plain_f16_with_distance::<C, DotProduct>(args),
    }
}

fn sparse_plain_f16_with_distance<C, D>(args: &Args)
where
    C: ComponentType + FromPrimitive + SpaceUsage + for<'de> Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
{
    let index: HNSW<PlainSparseDataset<C, f16, D>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn sparse_scalar<V>(args: &Args, metric: DistanceKind)
where
    V: ValueType + Float + FromF32 + SpaceUsage + for<'de> Deserialize<'de>,
{
    match args.component_type {
        ComponentTypeArg::U16 => sparse_scalar_with_component::<u16, V>(args, metric),
        ComponentTypeArg::U32 => sparse_scalar_with_component::<u32, V>(args, metric),
    }
}

fn sparse_scalar_with_component<C, V>(args: &Args, metric: DistanceKind)
where
    C: ComponentType + FromPrimitive + SpaceUsage + for<'de> Deserialize<'de>,
    V: ValueType + Float + FromF32 + SpaceUsage + for<'de> Deserialize<'de>,
{
    match metric {
        DistanceKind::Euclidean => {
            sparse_scalar_with_distance::<C, V, SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => sparse_scalar_with_distance::<C, V, DotProduct>(args),
    }
}

fn sparse_scalar_with_distance<C, V, D>(args: &Args)
where
    C: ComponentType + FromPrimitive + SpaceUsage + for<'de> Deserialize<'de>,
    V: ValueType + Float + FromF32 + SpaceUsage + for<'de> Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
{
    let index: HNSW<ScalarSparseDataset<C, f32, V, D>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn sparse_dotvbyte(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: DotVByte encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => sparse_dotvbyte_dp(args),
    }
}

fn sparse_dotvbyte_dp(args: &Args) {
    let index: HNSW<
        PackedSparseDataset<DotVByteFixedU8Encoder>,
        Graph<PlainNeighbors>,
    > = load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn sparse_dotpacking8(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: DotPacking8 encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => sparse_dotpacking8_dp(args),
    }
}

fn sparse_dotpacking8_dp(args: &Args) {
    let index: HNSW<
        PackedSparseDataset<DotPacking8ScalarU8Encoder>,
        Graph<PlainNeighbors>,
    > = load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn sparse_blocked_sparse(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: BlockedSparse encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => sparse_blocked_sparse_dp(args),
    }
}

fn sparse_blocked_sparse_dp(args: &Args) {
    let index: HNSW<PackedSparseDataset<BlockedSparseEncoder>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn sparse_dotpacking_sparse(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: DotPackingSparse encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => sparse_dotpacking_sparse_dp(args),
    }
}

fn sparse_dotpacking_sparse_dp(args: &Args) {
    let index: HNSW<PackedSparseDataset<DotPackingSparseScalarU8Encoder>, Graph<PlainNeighbors>> =
        load_index(args);
    write_level0_permutation(index.get_level0(), args);
}

fn load_index<T>(args: &Args) -> T
where
    T: IndexSerializer,
    T: for<'de> Deserialize<'de>,
{
    println!("[######] Loading index {}...", args.index_file);
    <T as IndexSerializer>::load_index(&args.index_file).unwrap_or_else(|e| {
        eprintln!("Error loading index {}: {e:?}", args.index_file);
        std::process::exit(1);
    })
}

fn write_level0_permutation<G>(graph: &G, args: &Args)
where
    G: GraphTrait + Sync,
{
    let n = graph.n_nodes();
    println!("[######] Ground nodes: {n}");
    println!("[######] Ground edges: {}", graph.n_edges());
    println!("[######] Computing EGB permutation...");

    let permutation = permute_graph_bisection_from_graph(graph);
    validate_permutation(&permutation).unwrap_or_else(|e| {
        eprintln!("Error validating EGB permutation: {e:?}");
        std::process::exit(1);
    });

    let permutation_u32 = permutation
        .iter()
        .map(|&x| {
            u32::try_from(x).unwrap_or_else(|_| {
                eprintln!("Error: permutation value {x} does not fit in u32.");
                std::process::exit(1);
            })
        })
        .collect::<Vec<_>>();

    println!("[######] Writing permutation to {}...", args.output_file);
    write_npy_u32_1d(&args.output_file, &permutation_u32).unwrap_or_else(|e| {
        eprintln!("Error writing permutation .npy {}: {e:?}", args.output_file);
        std::process::exit(1);
    });
}

fn write_npy_u32_1d(path: &str, data: &[u32]) -> std::io::Result<()> {
    write_npy_header_and_data(path, data.len(), "<u4", |w| {
        use std::io::Write;
        for &x in data {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    })
}

fn write_npy_header_and_data<F>(path: &str, len: usize, descr: &str, write_body: F) -> std::io::Result<()>
where
    F: FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>,
{
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1, 0])?;

    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({},), }}",
        descr, len
    );

    let preamble_len = 6 + 2 + 2;
    let total_no_pad = preamble_len + dict.len() + 1;
    let pad_len = (16 - (total_no_pad % 16)) % 16;

    let mut header = String::with_capacity(dict.len() + pad_len + 1);
    header.push_str(&dict);
    for _ in 0..pad_len {
        header.push(' ');
    }
    header.push('\n');

    let header_len = header.len() as u16;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(header.as_bytes())?;

    write_body(&mut w)?;
    w.flush()?;
    Ok(())
}
