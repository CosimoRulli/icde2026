use std::io::Write;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;
use vectorium::core::index::Index;
use std::fs::File;

use kannolo::graph::egb::{
    decode_dotpacking8_to_plain_f32, permute_blocked_sparse_dataset, permute_dense_dataset,
    permute_dotpacking8_dataset, permute_dotpacking_sparse_dataset, permute_dotvbyte_dataset,
    permute_plain_sparse_dataset, permute_scalar_sparse_dataset,
};
use kannolo::graph::graph::{Graph, GraphTrait, GrowableGraph};
use kannolo::graph::neighbors::{
    BitPackedNeighbors, EliasFanoNeighbors, NeighborData, Neighbors,
    PlainNeighbors, StreamVByteNeighbors, ZetaNeighbors,
    set_zeta_codec_k,
};
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use kannolo::hnsw_utils::validate_permutation;
use vectorium::IndexSerializer;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::blocked_sparse::BlockedSparseEncoder;
use vectorium::encoders::dotpacking8_scalaru8::DotPacking8ScalarU8Encoder;
use vectorium::encoders::dotpackingsparse_scalaru8::DotPackingSparseScalarU8Encoder;
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::{
    Dataset, DenseDataset, DenseVectorEncoder, FixedU8Q, FixedU16Q,
    PackedSparseDataset, PlainDenseDataset, PlainSparseDataset, ScalarSparseDataset, VectorEncoder,
};

#[derive(Debug, Clone, ValueEnum)]
enum DatasetType {
    Dense,
    Sparse,
}

/// Value type for stored values.
/// Dense plain: `f32`, `f16`.
/// Sparse plain: `f32`, `f16`, `fixedu8`, `fixedu16`.
/// Ignored for `dotvbyte` and `pq`.
#[derive(Debug, Clone, ValueEnum, Default)]
enum ValueTypeArg {
    F16,
    #[default]
    F32,
    Fixedu8,
    Fixedu16,
}

impl std::fmt::Display for ValueTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueTypeArg::F16 => write!(f, "f16"),
            ValueTypeArg::F32 => write!(f, "f32"),
            ValueTypeArg::Fixedu8 => write!(f, "fixedu8"),
            ValueTypeArg::Fixedu16 => write!(f, "fixedu16"),
        }
    }
}

/// Encoder type.
/// Dense: `plain`, `pq`.
/// Sparse: `plain`, `dotvbyte`, `dotpacking8-scalaru8`.
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

impl std::fmt::Display for EncoderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderType::Plain => write!(f, "plain"),
            EncoderType::Pq => write!(f, "pq"),
            EncoderType::Dotvbyte => write!(f, "dotvbyte"),
            EncoderType::Dotpacking8Scalaru8 => write!(f, "dotpacking8-scalaru8"),
            EncoderType::BlockedSparse => write!(f, "blocked-sparse"),
            EncoderType::DotpackingSparseScalaru8 => write!(f, "dotpacking8-sparse-scalar"),
        }
    }
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum ComponentTypeArg {
    #[default]
    U16,
    U32,
}

#[derive(Debug, Clone, ValueEnum)]
enum CompressionType {
    None,
    BitPacked,
    EliasFano,
    Zeta,
    StreamVByte,
}

#[derive(Debug, Clone, ValueEnum)]
enum EarlyTerminationMethod {
    None,
    DistanceAdaptive,
}

#[derive(Clone, Copy, Debug)]
enum DistanceKind {
    Euclidean,
    DotProduct,
}

trait GraphBound: GraphTrait + for<'de> serde::Deserialize<'de> + From<GrowableGraph> {}
impl<T> GraphBound for T where T: GraphTrait + for<'de> serde::Deserialize<'de> + From<GrowableGraph>
{}

fn parse_metric(metric: &str) -> DistanceKind {
    match metric {
        "euclidean" | "l2" => DistanceKind::Euclidean,
        "dotproduct" | "ip" => DistanceKind::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'euclidean' and 'dotproduct'.");
            std::process::exit(1);
        }
    }
}

fn read_npy_queries<D>(path: &str) -> PlainDenseDataset<f32, D>
where
    D: ScalarDenseSupportedDistance,
{
    read_npy_f32::<D>(path).unwrap_or_else(|e| {
        eprintln!("Error reading .npy file: {e:?}");
        std::process::exit(1);
    })
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the index.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

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

    /// The type of graph compression to apply in memory after loading the plain index.
    #[clap(long, value_enum)]
    #[arg(default_value_t = CompressionType::None)]
    compression: CompressionType,

    /// Zeta codec k parameter used when --compression zeta is selected.
    #[clap(long = "zeta-k", value_parser)]
    #[arg(default_value_t = 3)]
    zeta_k: usize,

    /// Optional .npy file containing an EGB permutation (old_id -> new_id).
    /// When provided, the graph and dataset are remapped consistently before search.
    #[clap(long, value_parser)]
    permutation_file: Option<String>,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The ef_search parameter.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    ef_search: usize,

    /// Early termination strategy for search.
    #[clap(long, value_enum)]
    #[arg(default_value_t = EarlyTerminationMethod::None)]
    early_termination: EarlyTerminationMethod,

    /// Lambda parameter for DistanceAdaptive strategy.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1.0)]
    lambda: f32,

    /// Number of runs for timing.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    num_runs: usize,

    /// Optional full-precision f32 dataset for reranking PQ results.
    /// Only applicable for PQ encoder. If provided, the search will retrieve k_candidates
    /// results from PQ and rerank them using the full-precision dataset, returning top k.
    #[clap(long, value_parser)]
    rerank_dataset: Option<String>,

    /// Number of candidates to retrieve before reranking (only used with rerank_dataset).
    /// If not specified, defaults to k * 10.
    #[clap(long, value_parser)]
    k_candidates: Option<usize>,

    /// Brute in-memory convert. Load the on-disk source index (plain-f16 for
    /// `--encoder blocked-sparse`, dp8 for `--encoder dotpacking8-sparse-scalar`) and convert
    /// its dataset to the target encoder before compress + search. Reuses the source graph and
    /// its EGB permutation, so no rebuild is needed (recall matches the source index exactly).
    #[clap(long = "convert-in-memory")]
    convert_in_memory: bool,
}

fn main() {
    let args: Args = Args::parse();

    // Cross-validation of encoder / dataset-type combinations
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

    if args.zeta_k == 0 {
        eprintln!("Error: --zeta-k must be greater than 0.");
        std::process::exit(1);
    }
    set_zeta_codec_k(args.zeta_k);
    if matches!(args.compression, CompressionType::Zeta) {
        println!("[######] Zeta codec k: {}", args.zeta_k);
    }

    // Brute in-memory convert path: load an existing plain-f16 / dp8 index and convert it to the
    // target encoder in memory (no rebuild). Reuses the source graph + EGB permutation.
    if args.convert_in_memory {
        match (&args.dataset_type, &args.encoder) {
            (DatasetType::Sparse, EncoderType::BlockedSparse) => {
                search_sparse_blocked_sparse_convert(&args);
            }
            (DatasetType::Sparse, EncoderType::DotpackingSparseScalaru8) => {
                search_sparse_dotpacking_sparse_convert(&args);
            }
            _ => {
                eprintln!(
                    "Error: --convert-in-memory only supports sparse blocked-sparse / dotpacking8-sparse-scalar encoders."
                );
                std::process::exit(1);
            }
        }
        return;
    }

    match (&args.dataset_type, &args.encoder, &args.value_type) {
        // Dense plain f32
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32) => {
            search_dense_plain_f32(&args, metric);
        }
        // Dense plain f16
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16) => {
            search_dense_plain_f16(&args, metric);
        }
        // Dense PQ (value-type ignored)
        (DatasetType::Dense, EncoderType::Pq, _) => {
            search_dense_pq(&args, metric);
        }
        // Sparse plain f16
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16) => {
            search_sparse_plain_f16(&args, metric);
        }
        // Sparse plain f32
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32) => {
            search_sparse_plain_f32(&args, metric);
        }
        // Sparse plain fixedu8
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu8) => {
            search_sparse_scalar::<FixedU8Q>(&args, metric);
        }
        // Sparse plain fixedu16
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu16) => {
            search_sparse_scalar::<FixedU16Q>(&args, metric);
        }
        // Sparse dotvbyte (value-type ignored)
        (DatasetType::Sparse, EncoderType::Dotvbyte, _) => {
            search_sparse_dotvbyte(&args, metric);
        }
        // Sparse dotpacking8 scalar-u8 (value-type ignored)
        (DatasetType::Sparse, EncoderType::Dotpacking8Scalaru8, _) => {
            search_sparse_dotpacking8(&args, metric);
        }
        // Sparse blocked-sparse (f16 blocked dot-product; value-type ignored)
        (DatasetType::Sparse, EncoderType::BlockedSparse, _) => {
            search_sparse_blocked_sparse(&args, metric);
        }
        // Sparse dotpacking8 sparse-scalar-u8 (value-type ignored)
        (DatasetType::Sparse, EncoderType::DotpackingSparseScalaru8, _) => {
            search_sparse_dotpacking_sparse(&args, metric);
        }
        // Unreachable: caught by earlier validation
        (DatasetType::Dense, EncoderType::Dotvbyte, _)
        | (DatasetType::Dense, EncoderType::Dotpacking8Scalaru8, _)
        | (DatasetType::Dense, EncoderType::BlockedSparse, _)
        | (DatasetType::Dense, EncoderType::DotpackingSparseScalaru8, _)
        | (DatasetType::Sparse, EncoderType::Pq, _)
        | (
            DatasetType::Dense,
            EncoderType::Plain,
            ValueTypeArg::Fixedu8 | ValueTypeArg::Fixedu16,
        ) => {
            unreachable!()
        }
    }
}

fn create_search_config(args: &Args) -> HNSWSearchConfiguration {
    let early_termination = match args.early_termination {
        EarlyTerminationMethod::None => EarlyTerminationStrategy::None,
        EarlyTerminationMethod::DistanceAdaptive => EarlyTerminationStrategy::DistanceAdaptive {
            lambda: args.lambda,
        },
    };

    HNSWSearchConfiguration::default()
        .with_ef_search(args.ef_search)
        .with_early_termination(early_termination)
}

fn load_validated_permutation(path: &str, n: usize) -> Vec<usize> {
    println!("[######] Loading EGB permutation from {path}...");
    let permutation = read_npy_usize_1d(path).unwrap_or_else(|e| {
        eprintln!("Error reading permutation .npy {path}: {e:?}");
        std::process::exit(1);
    });
    if permutation.len() != n {
        eprintln!(
            "Error: permutation length mismatch. Expected {}, got {}.",
            n,
            permutation.len()
        );
        std::process::exit(1);
    }
    validate_permutation(&permutation).unwrap_or_else(|e| {
        eprintln!("Error validating permutation {path}: {e:?}");
        std::process::exit(1);
    });
    permutation
}

fn compress_index<D, Nsrc, Ndst>(
    index: &HNSW<D, Graph<Nsrc>>,
) -> HNSW<D, Graph<Ndst>>
where
    D: Dataset + Clone,
    Nsrc: Neighbors + Sync + From<NeighborData>,
    Ndst: Neighbors + From<NeighborData>,
{
    println!("[######] Compressing graph...");
    let start_time = Instant::now();
    let compressed_index = index.into();
    println!(
        "[######] Graph compressed in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    compressed_index
}

fn compress_index_with_permuted_dataset<Dsrc, Ddst, Nsrc, Ndst, F>(
    args: &Args,
    index: &HNSW<Dsrc, Graph<Nsrc>>,
    make_dataset: F,
) -> HNSW<Ddst, Graph<Ndst>>
where
    Dsrc: Dataset + Clone,
    Ddst: Dataset,
    F: FnOnce(&Dsrc, &[usize]) -> Ddst,
    Nsrc: Neighbors + Sync + From<NeighborData>,
    Ndst: Neighbors + From<NeighborData>,
{
    println!("[######] Compressing graph and permuting dataset...");
    let start_time = Instant::now();
    let path = args
        .permutation_file
        .as_ref()
        .expect("--permutation-file is required for permuted compression");
    let n = index.get_level0().n_nodes();
    let permutation = load_validated_permutation(path, n);
    let compressed_index =
        index.remap_compress_and_replace_dataset_with_permutation::<Ndst, _, _>(
            &permutation,
            make_dataset,
        );
    println!(
        "[######] Graph compressed and dataset permuted in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    compressed_index
}

fn compress_index_with_optional_dataset_permutation<D, Nsrc, Ndst, F>(
    args: &Args,
    index: &HNSW<D, Graph<Nsrc>>,
    make_dataset: F,
) -> HNSW<D, Graph<Ndst>>
where
    D: Dataset + Clone,
    F: FnOnce(&D, &[usize]) -> D,
    Nsrc: Neighbors + Sync + From<NeighborData>,
    Ndst: Neighbors + From<NeighborData>,
{
    if args.permutation_file.is_some() {
        compress_index_with_permuted_dataset(args, index, make_dataset)
    } else {
        compress_index(index)
    }
}

fn load_and_compress_index<D, Ndst, F>(args: &Args, permute_fn: F) -> HNSW<D, Graph<Ndst>>
where
    D: Dataset + Clone + for<'de> serde::Deserialize<'de>,
    HNSW<D, Graph<PlainNeighbors>>: IndexSerializer,
    F: FnOnce(&D, &[usize]) -> D,
    PlainNeighbors: Neighbors + Sync + From<NeighborData>,
    Ndst: Neighbors + From<NeighborData>,
{
    let src: HNSW<D, Graph<PlainNeighbors>> =
        <HNSW<D, Graph<PlainNeighbors>> as IndexSerializer>::load_index(&args.index_file)
            .unwrap_or_else(|e| {
                eprintln!("Error loading index {}: {e:?}", args.index_file);
                std::process::exit(1);
            });
    compress_index_with_optional_dataset_permutation(args, &src, permute_fn)
}

macro_rules! dispatch_compression {
    ($args:expr, $dataset_ty:ty, $permute_fn:expr; $index:ident $body:block) => {
        match $args.compression {
            CompressionType::None => {
                let $index =
                    load_and_compress_index::<$dataset_ty, PlainNeighbors, _>($args, $permute_fn);
                $body
            }
            CompressionType::BitPacked => {
                let $index = load_and_compress_index::<$dataset_ty, BitPackedNeighbors, _>(
                    $args, $permute_fn,
                );
                $body
            }
            CompressionType::EliasFano => {
                let $index = load_and_compress_index::<$dataset_ty, EliasFanoNeighbors, _>(
                    $args, $permute_fn,
                );
                $body
            }
            CompressionType::Zeta => {
                let $index =
                    load_and_compress_index::<$dataset_ty, ZetaNeighbors, _>($args, $permute_fn);
                $body
            }
            CompressionType::StreamVByte => {
                let $index = load_and_compress_index::<$dataset_ty, StreamVByteNeighbors, _>(
                    $args, $permute_fn,
                );
                $body
            }
        }
    };
}

fn print_search_start(args: &Args, num_queries: usize) {
    println!(
        "[######] Starting search: {num_queries} queries, {} run(s), k={}, ef_search={}",
        args.num_runs, args.k, args.ef_search
    );
}

fn search_dense_plain_f32(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            search_dense_plain_f32_with_distance::<SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => search_dense_plain_f32_with_distance::<DotProduct>(args),
    }
}

fn search_dense_plain_f32_with_distance<D>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    dispatch_compression!(args, DenseDataset<PlainDenseQuantizer<f32, D>>, permute_dense_dataset; index {
        search_dense_plain_f32_inner(args, &index, &queries);
    });
}

fn search_dense_plain_f32_inner<D, G>(
    args: &Args,
    index: &HNSW<DenseDataset<PlainDenseQuantizer<f32, D>>, G>,
    queries: &DenseDataset<PlainDenseQuantizer<f32, D>>,
) where
    D: ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let num_queries = queries.len();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);

            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );

            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_plain_f16(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            search_dense_plain_f16_with_distance::<SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => search_dense_plain_f16_with_distance::<DotProduct>(args),
    }
}

fn search_dense_plain_f16_with_distance<D>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    dispatch_compression!(args, DenseDataset<PlainDenseQuantizer<f16, D>>, permute_dense_dataset; index {
        search_dense_plain_f16_inner(args, &index, &queries);
    });
}

fn search_dense_plain_f16_inner<D, G>(
    args: &Args,
    index: &HNSW<DenseDataset<PlainDenseQuantizer<f16, D>>, G>,
    queries: &DenseDataset<PlainDenseQuantizer<f32, D>>,
) where
    D: ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let num_queries = queries.len();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);

            results.extend(res.into_iter().map(|scored| {
                (
                    scored.distance.distance(),
                    scored.vector as usize,
                )
            }));

            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_pq(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => search_dense_pq_with_distance::<SquaredEuclideanDistance>(args),
        DistanceKind::DotProduct => search_dense_pq_with_distance::<DotProduct>(args),
    }
}

fn search_dense_pq_with_distance<D>(args: &Args)
where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance,
{
    let queries = read_npy_queries::<D>(&args.query_file);

    let rerank_dataset = if let Some(path) = &args.rerank_dataset {
        Some(read_npy_queries::<D>(path))
    } else {
        None
    };

    match args.pq_subspaces {
        4 => search_dense_pq_with_m::<4, D>(args, &queries, rerank_dataset.as_ref()),
        8 => search_dense_pq_with_m::<8, D>(args, &queries, rerank_dataset.as_ref()),
        16 => search_dense_pq_with_m::<16, D>(args, &queries, rerank_dataset.as_ref()),
        32 => search_dense_pq_with_m::<32, D>(args, &queries, rerank_dataset.as_ref()),
        48 => search_dense_pq_with_m::<48, D>(args, &queries, rerank_dataset.as_ref()),
        64 => search_dense_pq_with_m::<64, D>(args, &queries, rerank_dataset.as_ref()),
        96 => search_dense_pq_with_m::<96, D>(args, &queries, rerank_dataset.as_ref()),
        128 => search_dense_pq_with_m::<128, D>(args, &queries, rerank_dataset.as_ref()),
        192 => search_dense_pq_with_m::<192, D>(args, &queries, rerank_dataset.as_ref()),
        256 => search_dense_pq_with_m::<256, D>(args, &queries, rerank_dataset.as_ref()),
        384 => search_dense_pq_with_m::<384, D>(args, &queries, rerank_dataset.as_ref()),
        _ => {
            eprintln!(
                "Error: Invalid pq-subspaces value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384."
            );
            std::process::exit(1);
        }
    }
}

fn search_dense_pq_with_m<const M: usize, D>(
    args: &Args,
    queries: &DenseDataset<PlainDenseQuantizer<f32, D>>,
    rerank_dataset: Option<&DenseDataset<PlainDenseQuantizer<f32, D>>>,
) where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance,
    ProductQuantizer<M, D>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, D>: VectorEncoder<Distance = D>,
    DenseDataset<ProductQuantizer<M, D>>: Dataset<Encoder = ProductQuantizer<M, D>>,
{
    dispatch_compression!(args, DenseDataset<ProductQuantizer<M, D>>, permute_dense_dataset; index {
        search_dense_pq_inner(args, index, queries, rerank_dataset);
    });
}

fn search_dense_pq_inner<D, G, const M: usize>(
    args: &Args,
    index: HNSW<DenseDataset<ProductQuantizer<M, D>>, G>,
    queries: &DenseDataset<PlainDenseQuantizer<f32, D>>,
    rerank_dataset: Option<&DenseDataset<PlainDenseQuantizer<f32, D>>>,
) where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance + Copy,
    G: GraphBound,
    ProductQuantizer<M, D>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, D>: VectorEncoder<Distance = D>,
    DenseDataset<ProductQuantizer<M, D>>: Dataset<Encoder = ProductQuantizer<M, D>>,
{
    let num_queries = queries.len();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);

    index.print_space_usage_bytes();

    if let Some(rerank_data) = rerank_dataset {
        let rerank_index = vectorium::core::rerank_index::RerankIndex::new(index, rerank_data);
        let k_candidates = args.k_candidates.unwrap_or(args.k * 10);

        for _ in 0..args.num_runs {
            for query in queries.iter() {
                let start_time = Instant::now();

                let reranked_results = rerank_index.search(
                    query,
                    query,
                    k_candidates,
                    args.k,
                    &config,
                    None,
                    None,
                );

                total_time_search += start_time.elapsed().as_micros();

                for scored in reranked_results {
                    results.push((
                        scored.distance.distance(),
                        scored.vector as usize,
                    ));
                }
            }
        }
    } else {
        for _ in 0..args.num_runs {
            for query in queries.iter() {
                let start_time = Instant::now();
                let res = index.search(query, args.k, &config);

                results.extend(res.into_iter().map(|scored| {
                    (
                        scored.distance.distance(),
                        scored.vector as usize,
                    )
                }));

                total_time_search += start_time.elapsed().as_micros();
            }
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f16(args: &Args, metric: DistanceKind) {
    match args.component_type {
        ComponentTypeArg::U16 => search_sparse_plain_f16_with_component::<u16>(args, metric),
        ComponentTypeArg::U32 => search_sparse_plain_f16_with_component::<u32>(args, metric),
    }
}

fn search_sparse_plain_f16_with_component<C>(args: &Args, metric: DistanceKind)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
{
    match metric {
        DistanceKind::Euclidean => {
            search_sparse_plain_f16_with_distance::<C, SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => search_sparse_plain_f16_with_distance::<C, DotProduct>(args),
    }
}

fn search_sparse_plain_f16_with_distance<C, D>(args: &Args)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
{
    let queries: PlainSparseDataset<C, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression!(args, PlainSparseDataset<C, f16, D>, permute_plain_sparse_dataset; index {
        search_sparse_plain_f16_inner(args, &index, &queries);
    });
}

fn search_sparse_plain_f16_inner<C, D, G>(
    args: &Args,
    index: &HNSW<PlainSparseDataset<C, f16, D>, G>,
    queries: &PlainSparseDataset<C, f32, D>,
) where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = create_search_config(args);
    let num_queries = queries.len();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f32(args: &Args, metric: DistanceKind) {
    match args.component_type {
        ComponentTypeArg::U16 => search_sparse_plain_f32_with_component::<u16>(args, metric),
        ComponentTypeArg::U32 => search_sparse_plain_f32_with_component::<u32>(args, metric),
    }
}

fn search_sparse_plain_f32_with_component<C>(args: &Args, metric: DistanceKind)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
{
    match metric {
        DistanceKind::Euclidean => {
            search_sparse_plain_f32_with_distance::<C, SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => search_sparse_plain_f32_with_distance::<C, DotProduct>(args),
    }
}

fn search_sparse_plain_f32_with_distance<C, D>(args: &Args)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
{
    let queries: PlainSparseDataset<C, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression!(args, PlainSparseDataset<C, f32, D>, permute_plain_sparse_dataset; index {
        search_sparse_plain_f32_inner(args, &index, &queries);
    });
}

fn search_sparse_plain_f32_inner<C, D, G>(
    args: &Args,
    index: &HNSW<PlainSparseDataset<C, f32, D>, G>,
    queries: &PlainSparseDataset<C, f32, D>,
) where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = create_search_config(args);
    let num_queries = queries.len();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_scalar<V>(args: &Args, metric: DistanceKind)
where
    V: vectorium::ValueType
        + vectorium::Float
        + vectorium::FromF32
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
{
    match args.component_type {
        ComponentTypeArg::U16 => search_sparse_scalar_with_component::<u16, V>(args, metric),
        ComponentTypeArg::U32 => search_sparse_scalar_with_component::<u32, V>(args, metric),
    }
}

fn search_sparse_scalar_with_component<C, V>(args: &Args, metric: DistanceKind)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    V: vectorium::ValueType
        + vectorium::Float
        + vectorium::FromF32
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
{
    match metric {
        DistanceKind::Euclidean => {
            search_sparse_scalar_with_distance::<C, V, SquaredEuclideanDistance>(args)
        }
        DistanceKind::DotProduct => search_sparse_scalar_with_distance::<C, V, DotProduct>(args),
    }
}

fn search_sparse_scalar_with_distance<C, V, D>(args: &Args)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    V: vectorium::ValueType
        + vectorium::Float
        + vectorium::FromF32
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
{
    let queries: PlainSparseDataset<C, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression!(args, ScalarSparseDataset<C, f32, V, D>, permute_scalar_sparse_dataset; index {
        search_sparse_scalar_inner(args, &index, &queries);
    });
}

fn search_sparse_scalar_inner<C, V, D, G>(
    args: &Args,
    index: &HNSW<ScalarSparseDataset<C, f32, V, D>, G>,
    queries: &PlainSparseDataset<C, f32, D>,
) where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    V: vectorium::ValueType
        + vectorium::Float
        + vectorium::FromF32
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = create_search_config(args);
    let num_queries = queries.len();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

// --- DotVByte (encoder = dotvbyte, DotProduct only) ---

fn search_sparse_dotvbyte(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: DotVByte encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => search_sparse_dotvbyte_dp(args),
    }
}

fn search_sparse_dotvbyte_dp(args: &Args) {
    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression!(args, PackedSparseDataset<DotVByteFixedU8Encoder>, permute_dotvbyte_dataset; index {
        search_sparse_dotvbyte_inner(args, &index, &queries);
    });
}

fn search_sparse_dotvbyte_inner<G>(
    args: &Args,
    index: &HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, G>,
    queries: &PlainSparseDataset<u16, f32, DotProduct>,
) where
    G: GraphBound,
{
    let config = create_search_config(args);
    let num_queries = queries.len();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_dotpacking8(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: DotPacking8 encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => search_sparse_dotpacking8_dp(args),
    }
}

fn search_sparse_dotpacking8_dp(args: &Args) {
    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression!(args, PackedSparseDataset<DotPacking8ScalarU8Encoder>, permute_dotpacking8_dataset; index {
        search_sparse_dotpacking8_inner(args, &index, &queries);
    });
}

fn search_sparse_dotpacking8_inner<G>(
    args: &Args,
    index: &HNSW<PackedSparseDataset<DotPacking8ScalarU8Encoder>, G>,
    queries: &PlainSparseDataset<u16, f32, DotProduct>,
) where
    G: GraphBound,
{
    let config = create_search_config(args);
    let num_queries = queries.len();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

// --- BlockedSparse (encoder = blocked-sparse, DotProduct only) ---

fn search_sparse_blocked_sparse(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: BlockedSparse encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => search_sparse_blocked_sparse_dp(args),
    }
}

fn search_sparse_blocked_sparse_dp(args: &Args) {
    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression!(args, PackedSparseDataset<BlockedSparseEncoder>, permute_blocked_sparse_dataset; index {
        search_sparse_blocked_sparse_inner(args, &index, &queries);
    });
}

fn search_sparse_blocked_sparse_inner<G>(
    args: &Args,
    index: &HNSW<PackedSparseDataset<BlockedSparseEncoder>, G>,
    queries: &PlainSparseDataset<u16, f32, DotProduct>,
) where
    G: GraphBound,
{
    let config = create_search_config(args);
    let num_queries = queries.len();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

// --- DotPackingSparse scalar-u8 (encoder = dotpacking8-sparse-scalar, DotProduct only) ---

fn search_sparse_dotpacking_sparse(args: &Args, metric: DistanceKind) {
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: DotPackingSparse encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => search_sparse_dotpacking_sparse_dp(args),
    }
}

fn search_sparse_dotpacking_sparse_dp(args: &Args) {
    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression!(args, PackedSparseDataset<DotPackingSparseScalarU8Encoder>, permute_dotpacking_sparse_dataset; index {
        search_sparse_dotpacking_sparse_inner(args, &index, &queries);
    });
}

fn search_sparse_dotpacking_sparse_inner<G>(
    args: &Args,
    index: &HNSW<PackedSparseDataset<DotPackingSparseScalarU8Encoder>, G>,
    queries: &PlainSparseDataset<u16, f32, DotProduct>,
) where
    G: GraphBound,
{
    let config = create_search_config(args);
    let num_queries = queries.len();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    print_search_start(args, num_queries);
    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| {
                        (
                            scored.distance.distance(),
                            scored.vector as usize,
                        )
                    }),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

// --- In-memory convert path (--convert-in-memory) -----------------------------------------
//
// Load an existing plain index, convert its dataset to the target encoder in memory (keeping
// the source graph), then compress + permute + search exactly like the on-disk path. Recall is
// identical to the source index; only the (faster) vector kernel changes.

/// Convert a loaded plain-f16 index into a blocked-sparse one (f16 values -> blocked f16).
fn convert_to_blocked_sparse(
    src: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph<PlainNeighbors>>,
) -> HNSW<PackedSparseDataset<BlockedSparseEncoder>, Graph<PlainNeighbors>> {
    src.convert_dataset_into()
}

/// Convert a loaded dp8 index into a dotpacking8-sparse-scalar (sparse-q) one. dp8 is decoded to
/// plain f32 and re-encoded with the same scalar-u8 quantizer, reproducing the original u8 codes.
fn convert_to_dotpacking_sparse(
    src: HNSW<PackedSparseDataset<DotPacking8ScalarU8Encoder>, Graph<PlainNeighbors>>,
) -> HNSW<PackedSparseDataset<DotPackingSparseScalarU8Encoder>, Graph<PlainNeighbors>> {
    src.map_dataset(decode_dotpacking8_to_plain_f32)
        .convert_dataset_into()
}

/// Load the source plain index, convert it to the target encoder in memory, then compress
/// (and optionally permute) the graph. Mirrors `load_and_compress_index`, but converts first.
fn load_convert_compress<Dsrc, Ddst, Ndst, C, P>(
    args: &Args,
    to_target: C,
    permute_fn: P,
) -> HNSW<Ddst, Graph<Ndst>>
where
    Dsrc: Dataset + for<'de> serde::Deserialize<'de>,
    HNSW<Dsrc, Graph<PlainNeighbors>>: IndexSerializer,
    Ddst: Dataset + Clone,
    C: FnOnce(HNSW<Dsrc, Graph<PlainNeighbors>>) -> HNSW<Ddst, Graph<PlainNeighbors>>,
    P: FnOnce(&Ddst, &[usize]) -> Ddst,
    PlainNeighbors: Neighbors + Sync + From<NeighborData>,
    Ndst: Neighbors + From<NeighborData>,
{
    println!(
        "[######] Loading source index for in-memory conversion: {}",
        args.index_file
    );
    let src: HNSW<Dsrc, Graph<PlainNeighbors>> =
        <HNSW<Dsrc, Graph<PlainNeighbors>> as IndexSerializer>::load_index(&args.index_file)
            .unwrap_or_else(|e| {
                eprintln!("Error loading index {}: {e:?}", args.index_file);
                std::process::exit(1);
            });
    println!("[######] Converting dataset in memory to target encoder...");
    let start_time = Instant::now();
    let converted: HNSW<Ddst, Graph<PlainNeighbors>> = to_target(src);
    println!(
        "[######] In-memory dataset conversion done in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    compress_index_with_optional_dataset_permutation(args, &converted, permute_fn)
}

macro_rules! dispatch_compression_convert {
    ($args:expr, $src_ty:ty, $dst_ty:ty, $to_target:expr, $permute_fn:expr; $index:ident $body:block) => {
        match $args.compression {
            CompressionType::None => {
                let $index = load_convert_compress::<$src_ty, $dst_ty, PlainNeighbors, _, _>(
                    $args, $to_target, $permute_fn,
                );
                $body
            }
            CompressionType::BitPacked => {
                let $index = load_convert_compress::<$src_ty, $dst_ty, BitPackedNeighbors, _, _>(
                    $args, $to_target, $permute_fn,
                );
                $body
            }
            CompressionType::EliasFano => {
                let $index = load_convert_compress::<$src_ty, $dst_ty, EliasFanoNeighbors, _, _>(
                    $args, $to_target, $permute_fn,
                );
                $body
            }
            CompressionType::Zeta => {
                let $index = load_convert_compress::<$src_ty, $dst_ty, ZetaNeighbors, _, _>(
                    $args, $to_target, $permute_fn,
                );
                $body
            }
            CompressionType::StreamVByte => {
                let $index = load_convert_compress::<$src_ty, $dst_ty, StreamVByteNeighbors, _, _>(
                    $args, $to_target, $permute_fn,
                );
                $body
            }
        }
    };
}

fn search_sparse_blocked_sparse_convert(args: &Args) {
    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression_convert!(args,
        PlainSparseDataset<u16, f16, DotProduct>,
        PackedSparseDataset<BlockedSparseEncoder>,
        convert_to_blocked_sparse,
        permute_blocked_sparse_dataset; index {
        search_sparse_blocked_sparse_inner(args, &index, &queries);
    });
}

fn search_sparse_dotpacking_sparse_convert(args: &Args) {
    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });

    dispatch_compression_convert!(args,
        PackedSparseDataset<DotPacking8ScalarU8Encoder>,
        PackedSparseDataset<DotPackingSparseScalarU8Encoder>,
        convert_to_dotpacking_sparse,
        permute_dotpacking_sparse_dataset; index {
        search_sparse_dotpacking_sparse_inner(args, &index, &queries);
    });
}

fn write_results_to_file(output_path: &str, results: &[(f32, usize)], k: usize) {
    let mut file = File::create(output_path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}

fn read_npy_usize_1d(path: &str) -> anyhow::Result<Vec<usize>> {
    use std::io::{BufReader, Read};
    use anyhow::{bail, Context};

    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open {}", path))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        bail!("{} is not a .npy file", path);
    }

    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    let header_len = match version {
        [1, 0] => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as usize
        }
        [2, 0] | [3, 0] => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        }
        _ => bail!("unsupported .npy version {}.{}", version[0], version[1]),
    };

    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header).context("invalid UTF-8 in .npy header")?;

    let descr = npy_parse_header_string(header, "descr")?;
    let fortran_order = npy_parse_header_bool(header, "fortran_order")?;
    if fortran_order {
        bail!("Fortran-order .npy arrays are not supported");
    }

    let len = npy_parse_header_1d_len(header)?;
    let item_size = npy_dtype_item_size(descr)?;
    let expected_bytes = len
        .checked_mul(item_size)
        .context("overflow while computing .npy data size")?;

    let mut body = Vec::new();
    reader.read_to_end(&mut body)?;
    if body.len() != expected_bytes {
        bail!(
            "invalid .npy data size: expected {} bytes, found {}",
            expected_bytes,
            body.len()
        );
    }

    npy_decode_unsigned_usize(descr, &body)
}

fn npy_parse_header_string<'a>(header: &'a str, key: &str) -> anyhow::Result<&'a str> {
    use anyhow::{bail, Context};
    let key_pos = header
        .find(&format!("'{}'", key))
        .or_else(|| header.find(&format!("\"{}\"", key)))
        .with_context(|| format!("missing '{}' in .npy header", key))?;
    let after_key = &header[key_pos..];
    let colon = after_key
        .find(':')
        .with_context(|| format!("missing ':' after '{}' in .npy header", key))?;
    let value = after_key[colon + 1..].trim_start();
    let quote = value
        .chars()
        .next()
        .with_context(|| format!("missing value for '{}' in .npy header", key))?;
    if quote != '\'' && quote != '"' {
        bail!("expected string value for '{}' in .npy header", key);
    }
    let rest = &value[quote.len_utf8()..];
    let end = rest
        .find(quote)
        .with_context(|| format!("unterminated string for '{}' in .npy header", key))?;
    Ok(&rest[..end])
}

fn npy_parse_header_bool(header: &str, key: &str) -> anyhow::Result<bool> {
    use anyhow::{bail, Context};
    let key_pos = header
        .find(&format!("'{}'", key))
        .or_else(|| header.find(&format!("\"{}\"", key)))
        .with_context(|| format!("missing '{}' in .npy header", key))?;
    let after_key = &header[key_pos..];
    let colon = after_key
        .find(':')
        .with_context(|| format!("missing ':' after '{}' in .npy header", key))?;
    let value = after_key[colon + 1..].trim_start();
    if value.starts_with("False") {
        Ok(false)
    } else if value.starts_with("True") {
        Ok(true)
    } else {
        bail!("expected boolean value for '{}' in .npy header", key);
    }
}

fn npy_parse_header_1d_len(header: &str) -> anyhow::Result<usize> {
    use anyhow::Context;
    let key_pos = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .context("missing 'shape' in .npy header")?;
    let after_key = &header[key_pos..];
    let open = after_key
        .find('(')
        .context("missing shape tuple in .npy header")?;
    let after_open = &after_key[open + 1..];
    let close = after_open
        .find(')')
        .context("unterminated shape tuple in .npy header")?;
    let shape = &after_open[..close];
    let dims = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if dims.len() != 1 {
        anyhow::bail!("expected a 1D .npy array, found shape ({})", shape);
    }
    dims[0]
        .parse::<usize>()
        .with_context(|| format!("invalid 1D shape length '{}'", dims[0]))
}

fn npy_dtype_item_size(descr: &str) -> anyhow::Result<usize> {
    match descr {
        "|u1" | "<u1" | ">u1" => Ok(1),
        "<u2" | ">u2" => Ok(2),
        "<u4" | ">u4" => Ok(4),
        "<u8" | ">u8" => Ok(8),
        _ => anyhow::bail!("unsupported .npy dtype '{}'; expected an unsigned integer array", descr),
    }
}

fn npy_decode_unsigned_usize(descr: &str, body: &[u8]) -> anyhow::Result<Vec<usize>> {
    use anyhow::Context;
    let mut out = Vec::with_capacity(body.len() / npy_dtype_item_size(descr)?);

    match descr {
        "|u1" | "<u1" | ">u1" => out.extend(body.iter().map(|&x| x as usize)),
        "<u2" => {
            for chunk in body.chunks_exact(2) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as usize);
            }
        }
        ">u2" => {
            for chunk in body.chunks_exact(2) {
                out.push(u16::from_be_bytes([chunk[0], chunk[1]]) as usize);
            }
        }
        "<u4" => {
            for chunk in body.chunks_exact(4) {
                out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize);
            }
        }
        ">u4" => {
            for chunk in body.chunks_exact(4) {
                out.push(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize);
            }
        }
        "<u8" => {
            for chunk in body.chunks_exact(8) {
                let x = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(usize::try_from(x).context("u64 value does not fit in usize")?);
            }
        }
        ">u8" => {
            for chunk in body.chunks_exact(8) {
                let x = u64::from_be_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(usize::try_from(x).context("u64 value does not fit in usize")?);
            }
        }
        _ => unreachable!(),
    }

    Ok(out)
}
