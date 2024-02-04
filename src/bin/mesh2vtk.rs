//! Command line conversion of meshes to vtk formats
//!
//! Converts a mesh tally of any type to visual toolkit formats with a depth of
//! configurable parameters.
//!
//! # Usage
//!
//! ```text
//! Usage: mesh2vtk <meshtal> <number> [options]
//! ```
//!
//! Help is printed with the `-h` flag, and `--help` will show examples, default
//! values, examples, and any important behaviour.
//!
//! ## Mesh options
//!
//! By default, the results of every time/energy bin are extracted
//!
//! ### > How to include errors
//!
//! Corresponding uncertainty meshes are optional in case of large meshtal files.
//!
//! ```bash
//! # Extract every energy and time group, with corresponding error meshes
//! mesh2vtk /path/to/meshtal.msht 104 --errors
//! ```
//!
//! ### > How to only use the 'Total' energy/time group?
//!
//! Often only the `Total` energy/time bins are of interest, and a quick way of
//! only converting this subset is provided.
//!
//! ```bash
//! # Extract only the 'Total' energy and time groups
//! mesh2vtk /path/to/meshtal.msht 104 --total
//! ```
//!
//! ### > How to choose specific energy/time groups
//!
//! If specific energy or time groups are required,
//!
//! ```bash
//! # Extract specific energy and time groups
//! mesh2vtk /path/to/meshtal.msht 104    \
//!         --energy 1.0 1e5              \
//!         --time 1e12 total
//! ```
//!
//! For intuitive use, the groups correspond with values defined on the `EMESH`
//! and `TMESH` cards.
//!
//! *Note: mesh tallies label groups by the upper bounds defined on EMESH/TMESH
//! cards. i.e the energy group `1.0` corresponds to the `0.0>=E<1.0` bin,
//! though in reality 1 MeV particle would end up in the next group up.*
//!
//! ### > How to rescale all values
//!
//! Mcnp normalises everything so it is often the case that the results must be
//! rescaled to provide physical values.
//!
//! ```bash
//! # Rescale the result by a constant multiplier, e.g 6.50E+18 neutrons/s
//! mesh2vtk /path/to/meshtal.msht 104 --scale 6.50E+18
//! ```
//!
//! Mesh rotations are a work in progress.
//!
//! ## Cylindrical meshes
//!
//! There is no VTK representation of cylindrical meshes, so an unstructured
//! mesh is generated from verticies based on the RZT bounds.
//!
//! Unfortunately, this can result in "low-resolution" plots for meshes with
//! few theta bins. The number of theta bins can be increased to round off these
//! edges. This simply subdivides the voxels by an integer number of theta bins.
//!
//! ![Cylindrical mesh resolution option](https://github.com/repositony/meshtal/assets/63453741/9d77492f-0cea-42a5-81f4-36349fc20ec6)
//!
//! ```bash
//! # Subdivide voxels into 3 to round off cylindrical unstructured mesh
//! mesh2vtk /path/to/meshtal.msht 104 --resolution 3
//! ```
//!
//! Note that this will increase the file size and memory usage significantly
//! in some cases.
//!
//! ## VTK options
//!
//! ### > How to change the output file name
//!
//! By default the file prefix is `fmesh`, so the output files will be
//! `fmesh_<number>.vtk`. This may be changed as needed.
//!
//! ```bash
//! # Change the output name to `myvtk_104.vtr`
//! mesh2vtk /path/to/meshtal.msht 104 --output myvtk
//! ```
//!
//! ### > How to choose a Vtk format
//!
//! Most useful may be the ability to decide on output formats. XML and legacy
//! formats are supported, with both ascii and binary variants.
//!
//! ```bash
//! # Output as a binary vtk with legacy formatting
//! mesh2vtk /path/to/meshtal.msht 104 --format legacy-binary
//! ```
//!
//! ### > How to specify compression and byte order
//!
//! For more advanced usage, things like byte ordering and xml compression
//! methods are also configurable.
//!
//! ```bash
//! # Output as an xml using lzma and setting the byte order to big endian
//! mesh2vtk /path/to/meshtal.msht 104      \
//!         --format xml                    \
//!         --compresser lzma               \
//!         --endian big-endian
//! ```
//!
//! *Note - [VisIt](https://visit-dav.github.io/visit-website/index.html) only
//! reads big-endian, but most sytems are natively little-endian. For personal
//! convenience the default is big, but I am open to arguments for little endian
//! as the default.*
#![doc(hidden)]

// standard libraries
use std::path::Path;

// crate modules
use meshtal::mesh::{Geometry, Group, Mesh};
use meshtal::readers::MeshtalReader;
use meshtal::utils::*;
use meshtal::vtk::{write_vtk, MeshToVtk, MeshToVtkBuilder, VtkFormat};

// external crates
use anyhow::{anyhow, Result};
use clap::{arg, Parser, ValueEnum};
use log::*;
use vtkio::model::ByteOrder;
use vtkio::xml::Compressor;

fn main() -> Result<()> {
    // set up the command line interface and match arguments
    let cli: Cli = Cli::parse();

    // set up logging (+2 to make Info the default)
    let verbosity = cli.verbose as usize + 2;
    logging_init(verbosity, cli.quiet);

    // Get the mesh tally
    info!("Reading {}", &cli.meshtal);
    let mut mesh = try_meshtal_read(&cli)?;

    debug!("{mesh}");

    if let Some(scale) = cli.scale {
        info!("Scaling results by {:.5e}", scale);
        mesh.scale(scale);
    }

    // Generate the vtk and write to file
    info!("Converting mesh to VTK object");
    let convertor = converter_init(&mesh, &cli);
    let vtk = convertor.convert(&mesh)?;

    // decide the extension for them because people won't read the --help
    let path = get_output_path(&mesh, &cli);

    // Write the vtk
    info!("Writing VTK to file");
    write_vtk(vtk, path, cli.format)?;

    Ok(())
}

#[allow(rustdoc::invalid_rust_codeblocks)]
/// Generalised conversion of meshtal files to visual toolkit formats
///
/// Examples
/// --------
///
///  Typical use:
///     $ mesh2vtk run0.msht 104 -o my_output
///
///  Extract only the 'Total' energy and time groups:
///     $ mesh2vtk /path/to/meshtal.msht 104 --total
///
///  Include voxel errors in the output:
///     $ mesh2vtk /path/to/meshtal.msht 104 --errors
///
///  Extract specific energy and time groups:
///     $ mesh2vtk /path/to/meshtal.msht 104  \
///               --energy 1.0 1e5            \
///               --time 1e12 total
///
///  Output legacy in ascii format, with error meshes:
///     $ mesh2vtk /path/to/meshtal.msht 104  \
///               --format legacy-ascii       \
///               --errors
///
///  Alter basic mesh properties:
///     $ mesh2vtk /path/to/meshtal.msht 104  \
///               --scale 1.0                 \
///               --translate 0.5 -5.0 20.0
///
/// Notes
/// -----
///
/// CuV results are weighted by volume of each contributing cell, and
/// VoidRecord::Off will fill in missing void voxels with 0.0 flux.
///
/// Run-on numbers without whitespace and broken exponential formatting
/// are handled.
///   - e.g. "1.00E+00-2.00E+00" => 1.00E+00 -2.00E+00
///   - e.g. "1.00+002" => 1.00E+002
///
/// Meshes with a single bin for EMESH/TMESH are labelled the 'Total'
/// group for consistency.

#[derive(Parser)]
#[command(
    verbatim_doc_comment,
    arg_required_else_help(true),
    before_help(banner()),
    after_help("Typical use: mesh2vtk run0.msht 104 -o my_output \n\nNOTE: --help shows more detail and examples"),
    term_width(70),
    hide_possible_values(true),
    override_usage("mesh2vtk <meshtal> <number> [options]")
)]
struct Cli {
    // * Positional
    /// Path to input meshtal file
    #[arg(name = "meshtal")]
    meshtal: String,
    /// Mesh tally identifier
    ///
    /// e.g. 104 for FMESH104:n
    #[arg(name = "number")]
    number: u32,

    // * Optional
    /// Only extract 'Total' energy/time groups
    ///
    /// By default all energy groups are included in the vtk. This equivalent to
    /// passing '--energy total --time total' as arguments.
    #[arg(help_heading("Mesh options"))]
    #[arg(short, long)]
    total: bool,

    /// Include errors mesh in output files
    ///
    /// Error meshes omitted by default. If enabled, every mesh will have a
    /// corresponding relative uncertainty dataset (~doubles file size).
    #[arg(help_heading("Mesh options"))]
    #[arg(short, long)]
    errors: bool,

    /// Target energy group(s)
    ///
    /// By default all energy groups are included in the vtk. Specific energy
    /// groups can be provided to reduce file sizes. Values may be any
    /// combination of positive numbers and the word 'total'.
    #[arg(help_heading("Mesh options"))]
    #[arg(long)]
    #[arg(value_parser, num_args = 1.., value_delimiter = ' ')]
    #[clap(required = false)]
    #[arg(conflicts_with = "total")]
    #[arg(value_name = "list")]
    energy: Vec<String>,

    /// Target time group(s)
    ///
    /// By default all time groups are included in the vtk. Specific time
    /// groups can be provided to reduce file sizes. Values may be any
    /// combination of numbers and the word 'total'.
    #[arg(help_heading("Mesh options"))]
    #[arg(long)]
    #[arg(value_parser, num_args = 1.., value_delimiter = ' ')]
    #[clap(required = false)]
    #[arg(conflicts_with = "total")]
    #[arg(value_name = "list")]
    #[arg(allow_negative_numbers(true))]
    time: Vec<String>,

    /// Multiply all results by a constant
    ///
    /// All results in the mesh are rescaled by the value provided. e.g. --scale
    /// 10 will multiply the result of every voxel by 10.0. Errors are relative
    /// and are therefore unchanged.
    #[arg(help_heading("Mesh options"))]
    #[arg(short, long)]
    #[arg(value_name = "cst")]
    scale: Option<f64>,

    /// Name of output file (excl. extension)
    ///
    /// Defaults to `fmesh_<NUMBER>`, and will automatically append the mesh id
    /// and correct extension.
    #[arg(help_heading("Vtk options"))]
    #[arg(short, long)]
    #[arg(value_name = "path")]
    output: Option<String>,

    /// VTK output format
    ///
    /// Available visual toolkit file formats:
    ///     > xml (default)
    ///     > legacy-ascii
    ///     > legacy-binary
    #[arg(help_heading("Vtk options"))]
    #[arg(short, long, value_enum)]
    #[arg(hide_default_value(true))]
    #[arg(default_value_t = VtkFormat::Xml)]
    #[arg(verbatim_doc_comment)]
    #[arg(value_name = "format")]
    format: VtkFormat,

    /// Cylindrical mesh resolution
    ///
    /// !! WARNING !!: Every vertex is defined explicitly, so large values will
    /// significantly increase memory usage and file size.
    ///
    /// Integer value for increasing angular resolution of cylindrical meshes.
    /// Cylinders are approximated to straight edge segments so it can be useful
    /// to round this off by splitting voxels into multiple smaller segments.
    ///
    /// e.g. 4 theta bins gives 4 edges and therefore looks square. Using
    /// `--resolution 3` generates 12 edges instead and looks more rounded.
    #[arg(help_heading("Vtk options"))]
    #[arg(long)]
    #[arg(value_name = "res")]
    resolution: Option<u8>,

    /// Byte ordering
    ///
    /// Visit only reads big endian, most sytems are little endian.
    /// Defaults to big endian for convenience over performance.
    ///     > big-endian (default)
    ///     > little-endian
    #[arg(help_heading("Vtk options"))]
    #[arg(long, value_enum)]
    #[arg(hide_default_value(true))]
    #[arg(default_value_t = CliByteOrder::BigEndian)]
    #[arg(verbatim_doc_comment)]
    #[arg(value_name = "endian")]
    endian: CliByteOrder,

    /// Comression method for xml
    ///
    /// Generally just use LZMA but other options are available.
    ///     > lzma (default)
    ///     > lz4
    ///     > zlib
    ///     > none
    #[arg(long, value_enum)]
    #[arg(help_heading("Vtk options"))]
    #[arg(hide_default_value(true))]
    #[arg(default_value_t = CliCompressor::LZMA)]
    #[arg(verbatim_doc_comment)]
    #[arg(value_name = "compressor")]
    compressor: CliCompressor,

    // * Flags
    /// Verbose logging (-v, -vv)
    ///
    /// If specified, the default log level of INFO is increased to DEBUG (-v)
    /// or TRACE (-vv). Errors and Warnings are always logged unless in quiet
    /// (-q) mode.
    #[arg(short, long)]
    #[arg(action = clap::ArgAction::Count)]
    verbose: u8,

    /// Supress all log output (overrules --verbose)
    #[arg(short, long)]
    quiet: bool,
}

// Wrapper for byte order used by vtkio

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CliByteOrder {
    BigEndian,
    LittleEndian,
}

// Wrapper for compression strategy used by vtkio

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CliCompressor {
    LZ4,
    ZLib,
    LZMA,
    None,
}

fn banner() -> String {
    let mut s = f!("{:-<1$}\n", "", 70);
    s += &f!("{:^70}\n", "Meshtal :: MeshToVtk");
    s += &f!("{:-<1$}", "", 70);
    s
}

fn logging_init(verbosity: usize, quiet: bool) {
    stderrlog::new()
        .modules(vec![
            module_path!(),
            "meshtal::readers",
            "meshtal::mesh",
            "meshtal::vtk",
        ])
        .quiet(quiet)
        .verbosity(verbosity)
        .show_level(false)
        .color(stderrlog::ColorChoice::Never)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();
}

fn try_meshtal_read(cli: &Cli) -> Result<Mesh> {
    let path: &Path = Path::new(&cli.meshtal);

    let mut reader = MeshtalReader::new();
    reader.set_target_id(cli.number);
    if cli.quiet || cli.verbose > 1 {
        reader.disable_progress();
    }

    let mut mesh = reader.parse(path)?;
    Ok(std::mem::take(&mut mesh[0]))
}

fn converter_init(mesh: &Mesh, cli: &Cli) -> MeshToVtk {
    let mut builder = MeshToVtkBuilder::new().include_errors(cli.errors);

    if cli.total {
        debug!("Set all groups to 'Total' only");
        builder = builder.energy_groups(vec![Group::Total]);
        builder = builder.time_groups(vec![Group::Total]);
    } else {
        // Could easily all be nonsense, so default to getting all groups if parse fails
        let energy_groups = parse_groups(&cli.energy).unwrap_or(mesh.energy_groups());
        builder = builder.energy_groups(energy_groups);

        let time_groups = parse_groups(&cli.time).unwrap_or(mesh.time_groups());
        builder = builder.time_groups(time_groups);
    }

    if let Some(resolution) = cli.resolution {
        builder = builder.resolution(resolution);
    }

    builder = builder.byte_order(match cli.endian {
        CliByteOrder::LittleEndian => ByteOrder::LittleEndian,
        CliByteOrder::BigEndian => ByteOrder::BigEndian,
    });

    builder = builder.compressor(match cli.compressor {
        CliCompressor::LZMA => Compressor::LZMA,
        CliCompressor::LZ4 => Compressor::LZ4,
        CliCompressor::ZLib => Compressor::ZLib,
        CliCompressor::None => Compressor::None,
    });

    builder.build()
}

#[allow(clippy::redundant_closure)]
// Get any entries that parse into a number or 'total'
fn parse_groups(targets: &[String]) -> Result<Vec<Group>> {
    let mut values = targets
        .iter()
        .filter_map(|group| group.parse::<f64>().ok())
        .map(|v| Group::Value(v))
        .collect::<Vec<Group>>();

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values.dedup();

    if targets.iter().any(|t| t.to_lowercase() == "total") {
        values.push(Group::Total)
    };

    // Allow the caller to decide what to do with a failed parse of inputs
    if values.is_empty() {
        Err(anyhow!("Unable to parse the groups provided"))
    } else {
        Ok(values)
    }
}

fn get_output_path(mesh: &Mesh, cli: &Cli) -> String {
    let name = cli.output.clone().unwrap_or(String::from("fmesh"));

    let extension = match cli.format {
        VtkFormat::Xml => match mesh.geometry {
            Geometry::Rectangular => "vtr",
            Geometry::Cylindrical => "vtu",
        },
        _ => "vtk",
    };

    let path = f!("{}_{}.{}", name, cli.number, extension);
    debug!("Set output path to {path}");
    path
}
