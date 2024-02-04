//! Command line generation of weight windows
//!
//! Converts a mesh tally of any type to a weight window using the magic method
//! with configurable de-tuning options.
//!
//! # Usage
//!
//! ```text
//! # Single particle type weight window
//! Usage: mesh2ww <meshtal> <number> [options]
//!
//! # Multiple particle types
//! Usage: mesh2ww <meshtal> <number> [options] + <meshtal> <number> [options]
//! ```
//!
//! Help is printed with the `-h` flag, and `--help` will show examples, default
//! values, examples, and any important behaviour.
//!
//! ### Tuning weights
//!
//! Typical usage will generally define a de-tuning factor (`-p`/`--power`) and
//! possibly a relative error cutoff (`-e`/`--error`) for generating weights.  
//!
//! ```bash
//! mesh2ww run0.msht 104 --power 0.70 --error 0.5
//! ```
//!
//! The `--power` value modifies calculated weights by `w => w^(power)`, which
//! helps with softening extreme values. Any voxels with errors above `--error`
//! (10% in this case) continue to use analogue transport until the uncertainty
//! imporves.
//!
//! ### Renaming output files
//!
//! Of course, the weight window file may be renamed as needed:
//!
//! ```bash
//! mesh2ww run0.msht 104 --output mywwmesh.wwinp
//! ```
//!
//! ### Simplified weight window
//!
//! It is often desirable to simply generate a global weight window mesh using
//! only the 'Total' group rather than every explicit energy/time group.
//!
//! ```bash
//! mesh2ww run0.msht 104 --total
//! ```
//!
//! This is probably the recommended use case for any finely binned groups, as
//! nobody should really be trying to optimise for every energy in a 175-group
//! mesh anyway.
//!
//! ### Re-scale weights
//!
//! Generated weights are typically normalised to the total flux for each group.
//! These may be rescaled by a constant multiplier.
//!
//! ```bash
//! # Multiply all normalised weights by x2.5
//! mesh2ww run0.msht 104 --scale 2.5
//! ```
//!
//! ### Advanced de-tuning
//!
//! For fine control, the `--power` and `--error` parameters may be set
//! explicitly for every unique group.
//!
//! For example, if a mesh had 3 energy groups at 1.0 MeV, 10.0 MeV, and
//! 100.0 MeV, the power factor for each may be set to 0.8, 0.7, and 0.65
//! respectively.
//!
//! ```bash
//! # Set energy group power factors individually
//! mesh2ww run0.msht 104 --power 0.8 0.7 0.65
//! ```
//!
//! Of course this applies to time bins also. To set values for all unique
//! groups, the values must be given in order.
//!
//! For example, a mesh with 3x energy groups and 2x time groups:
//!
//! ```text
//! Energy 1.0        Power     
//!     Time 1e10      0.9     
//!     Time 1e20      0.7
//! Energy 10.0     
//!     Time 1e10      0.8     
//!     Time 1e20      0.8
//! Energy 100.0        
//!     Time 1e10      0.6     
//!     Time 1e20      0.5
//! ```
//!
//! ```bash
//! # Set energy group power factors individually
//! mesh2ww run0.msht 104 --power 0.9 0.7 0.8 0.8 0.6 0.5
//! ```
//!
//! ### Multi-particle weight windows
//!
//! Multiple tallies may be combined for weight windows covering multiple
//! particle types. This may be achieved using the `+` operator.
//!
//! The usage is as simple as combining multiple argument sets with `+` as the
//! delimiter.
//!
//! ```bash
//! mesh2ww <meshtal> <number> [options] +      \\
//!         <meshtal> <number> [options] +      \\
//!         <meshtal> <number> [options]
//! ```
//!
//! For example: 'NP_tallies.msht' contains neutron (FMESH14:n) and photon
//! (FMESH24:p) tallies, and 'E_tallies.msht' contains an electron (FMESH34:e)
//! tally.
//!
//! If all of these are the same geometry, they may be combined with all the
//! usual optionas applied to each weight set individually:
//!
//! ```bash
//! mesh2ww NP_tallies.msht 14                   +      \\
//!         NP_tallies.msht 24 -p 0.8 -e 0.15    +      \\
//!         E_tallies.msht  34 --total                  \\
//! ```
//!
//! ### Writing weights to VTK
//!
//! A Visual Toolkit file can be generated for every weight window set using the
//! `--vtk` flag.
//!
//! ```bash
//! mesh2ww file.msht 14 --vtk
//! ```
//!
//! Of course, all the usual options are available, such as increasing the
//! resolution of cylindrical meshes with few theta bins.
//!
//! ```bash
//! mesh2ww file.msht 14 --vtk --resolution 2
//! ```
//!
//! Advanced options include changing the file format, byte ordering of binary
//! outputs, and which compressor to use for XML.
//!
//! ```bash
//! mesh2ww file.msht 14 --vtk          \\
//!             --format legacy-ascii   \\
//!             --compressor lzma       \\
//!             --endian big-endian  
//! ```
//!
#![doc(hidden)]

// standard library
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::iter::zip;
use std::path::Path;

// crate modules
use meshtal::mesh::{Geometry, Mesh, Particle};
use meshtal::readers::MeshtalReader;
use meshtal::utils::*;
use meshtal::vtk::{self, VtkFormat, WeightsToVtk, WeightsToVtkBuilder};
use meshtal::weights::{self, WeightWindow};

// external crates
use anyhow::{anyhow, Result};
use bincode::serialize_into;
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command, ValueEnum};
use log::*;
use serde::{Deserialize, Serialize};
use vtkio::model::ByteOrder;
use vtkio::xml::Compressor;

// Convenience types
type ArgSet = Vec<String>;

fn main() -> Result<()> {
    // short circuit for help messages
    if help_flags() {
        return Ok(());
    }

    // set up logging (Info is the default)
    logging_init();

    // split up the command line args by the '+' delimeter and parse each one
    // through Clap to verify the arguments
    let ww_config_sets = parse_ww_config();

    // collect up all weight windows, just exclude any missing and warn the user
    info!("Generating weight windows");
    let particle_weights = collect_weight_windows(ww_config_sets)?;

    let file_config = parse_file_config();

    // Write the weight window file
    info!("Writing to {}", file_config.output);
    weights::write_multi_particle(&particle_weights, &file_config.output, !file_config.trim);

    info!("Conversion complete");
    Ok(())
}

// !    ------------------------------
// !    Weight window generation logic
// !    ------------------------------

fn collect_weight_windows(ww_config_sets: Vec<WWConfig>) -> Result<Vec<WeightWindow>> {
    // prepare for writing to VTK files if needed
    let vtk_config = parse_vtk_config();

    // prepare the ultimate return value
    let mut weight_windows: Vec<WeightWindow> = Vec::with_capacity(ww_config_sets.len());

    // Process each weight window set
    for cli in &ww_config_sets {
        // read mesh data from the meshtal file
        info!("Reading mesh {} from {}", &cli.number, &cli.meshtal);
        let mesh = try_meshtal_read(cli)?;

        // make sure the particle type is not a duplicate
        if weight_windows.iter().any(|ww| ww.particle == mesh.particle) {
            info!("{:?} already included, skipping...", mesh.particle);
            continue;
        }

        // convert mesh into WWMesh object for writing/further manipulation
        info!("Calculating {:?} weights", &mesh.particle);
        let mut ww = generate_weight_window(&mesh, cli);

        // Multiply weights by a constant factor if one is provided
        if cli.scale != 1.0 {
            info!("Scaling results by {}", cli.scale);
            ww.scale(cli.scale);
        }

        info!(
            "Weighted {:?} voxels: {:.2}%",
            ww.particle,
            ww.non_analogue_percentage()
        );

        // Write this out to a VTK for plotting is needed
        if vtk_config.vtk {
            info!("Writing {:?} VTK file", ww.particle);
            generate_vtk(&ww, &vtk_config)?;
        }

        weight_windows.push(ww);
    }

    if weight_windows.is_empty() {
        Err(anyhow!("No valid weight window sets"))
    } else {
        Ok(weight_windows)
    }
}

fn try_meshtal_read(cli: &WWConfig) -> Result<Mesh> {
    let path: &Path = Path::new(&cli.meshtal);

    let mut reader = MeshtalReader::new();
    reader.set_target_id(cli.number);
    if is_flag_present(&["-q", "--quiet"]) || collect_verbosity() > 1 {
        reader.disable_progress();
    }

    let mut mesh = reader.parse(path)?;
    Ok(std::mem::take(&mut mesh[0]))
}

fn generate_weight_window(mesh: &Mesh, cli: &WWConfig) -> WeightWindow {
    if cli.power.len() > 1 || cli.error.len() > 1 {
        if cli.total {
            warn!("Warning: Conflicting options");
            warn!(" - Multiple --power/--error values used with --total");
            warn!(" - Falling back to default values");
            weights::mesh_to_ww(mesh, 0.7, 1.0, cli.total)
        } else {
            weights::mesh_to_ww_advanced(mesh, &cli.power, &cli.error)
        }
    } else {
        weights::mesh_to_ww(mesh, cli.power[0], cli.error[0], cli.total)
    }
}

fn generate_vtk(weight_window: &WeightWindow, cli: &VtkConfig) -> Result<()> {
    // todo: skip cylindrical meshes while vtk conversion is being implemented
    if weight_window.nwg == Geometry::Cylindrical {
        warn!("Warning: Cylindrical conversion to VTK not yet implemented");
        return Ok(());
    }

    // Set up the conversion
    let convertor = build_converter(cli);
    let vtk = convertor.convert(weight_window)?;
    let extension = match cli.format {
        VtkFormat::Xml => match weight_window.nwg {
            Geometry::Rectangular => "vtr",
            Geometry::Cylindrical => "vtu",
        },
        _ => "vtk",
    };

    // Write to disk, using the paticle type as a simple file name
    vtk::write_vtk(
        vtk,
        f!("ww_{:?}.{extension}", weight_window.particle).to_lowercase(),
        VtkFormat::Xml,
    )
}

fn build_converter(cli: &VtkConfig) -> WeightsToVtk {
    WeightsToVtkBuilder::default()
        .resolution(cli.resolution)
        .byte_order(match cli.endian {
            CliByteOrder::BigEndian => ByteOrder::BigEndian,
            CliByteOrder::LittleEndian => ByteOrder::LittleEndian,
        })
        .compressor(match cli.compressor {
            CliCompressor::LZMA => Compressor::LZMA,
            CliCompressor::LZ4 => Compressor::LZ4,
            CliCompressor::ZLib => Compressor::ZLib,
            CliCompressor::None => Compressor::None,
        })
        .build()
}

// !    ---------------------------
// !    Cached weights for updating
// !    ---------------------------

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct CachedWeights {
    particle: Particle,
    weight_error: Vec<(f64, f64)>,
}

/// Generate a binary file for the cached weights
pub fn generate_binary(weight_window: &WeightWindow, mesh: &Mesh) {
    let mut f = BufWriter::new(File::create("./test.bin").unwrap());

    let weight_error = zip(&weight_window.weights, &mesh.voxels)
        .map(|(w, v)| (*w, v.error))
        .collect();

    serialize_into(
        &mut f,
        &CachedWeights {
            particle: weight_window.particle,
            weight_error,
        },
    )
    .expect("Failed to serialize cached weights to binary");
}

// !    ------------------------------
// !    Command line argument handling
// !    ------------------------------

/// Initialises the Clap CLI command and sets up arguments
fn cli_init() -> Command {
    Command::new("mesh2ww")
        .about("Conversion of meshtal file meshes to MCNP weight windows")
        .arg_required_else_help(true)
        .disable_help_flag(true)
        // .override_help(help)
        .before_help(banner())
        .after_help(after_help_message())
        .long_about(cli_long_help())
        .term_width(70)
        .hide_possible_values(true)
        .override_usage(usage_message())
        .args(positional_args())
        .args(optional_args())
        .args(debug_args())
}

/// Creates tonybox banner
fn banner() -> String {
    let mut s = f!("{:-<1$}\n", "", 70);
    s += &f!("{:^70}\n", "Meshtal :: MeshToWW");
    s += &f!("{:-<1$}", "", 70);
    s
}

/// Initialise logging for all relevant modules with variable verbosity
fn logging_init() {
    stderrlog::new()
        .modules(vec![
            module_path!(),
            "meshtal::readers",
            "meshtal::mesh",
            "meshtal::weights",
        ])
        .quiet(is_flag_present(&["-q", "--quiet"]))
        .verbosity(collect_verbosity() + 2) // +2 to make the default "Info"
        .show_level(false)
        .color(stderrlog::ColorChoice::Never)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();
}

/// Wrapper for byte order used by vtkio
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CliByteOrder {
    BigEndian,
    LittleEndian,
}

/// Wrapper for compression strategy used by vtkio
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CliCompressor {
    LZ4,
    ZLib,
    LZMA,
    None,
}

#[derive(Debug)]
struct WWConfig {
    meshtal: String,
    number: u32,
    power: Vec<f64>,
    error: Vec<f64>,
    total: bool,
    scale: f64,
}

#[derive(Debug)]
struct VtkConfig {
    vtk: bool,
    format: VtkFormat,
    compressor: CliCompressor,
    endian: CliByteOrder,
    resolution: u8,
}

#[derive(Debug)]
struct FileConfig {
    trim: bool,
    output: String,
}

fn all_argument_matches() -> Vec<ArgMatches> {
    split_argument_sets()
        .iter()
        .map(|set| cli_init().get_matches_from(set))
        .collect()
}

fn split_argument_sets() -> Vec<ArgSet> {
    let name = env::args().next().unwrap();
    let raw_args = env::args().skip(1).collect::<Vec<String>>();
    let mut tallies = Vec::with_capacity(37);

    for s in raw_args.split(|p| p == "+") {
        tallies.push(std::iter::once(name.clone()).chain(s.to_owned()).collect());
        trace!("{:?}", tallies.last());
    }

    tallies
}

fn parse_ww_config() -> Vec<WWConfig> {
    split_argument_sets()
        .iter()
        .filter_map(|arg_set| {
            let cli = parse_ww_set(arg_set.to_owned());
            if let Err(e) = &cli {
                warn!("{:?}, skipping", e);
            }
            cli.ok()
        })
        .collect::<Vec<WWConfig>>()
}

fn parse_vtk_config() -> VtkConfig {
    let matches = all_argument_matches();

    // fine to unwrap these matches because a default has been set
    VtkConfig {
        vtk: is_flag_present(&["--vtk"]),
        format: matches
            .iter()
            .find_map(|m| m.get_one::<VtkFormat>("format").cloned())
            .unwrap_or(VtkFormat::Xml),
        compressor: matches
            .iter()
            .find_map(|m| m.get_one::<CliCompressor>("compressor").cloned())
            .unwrap_or(CliCompressor::LZMA),
        endian: matches
            .iter()
            .find_map(|m| m.get_one::<CliByteOrder>("endian").cloned())
            .unwrap_or(CliByteOrder::BigEndian),
        resolution: matches
            .iter()
            .find_map(|m| m.get_one::<u8>("resolution").cloned())
            .unwrap_or(1),
    }
}

fn parse_file_config() -> FileConfig {
    let matches = all_argument_matches();

    // fine to unwrap these matches because a default has been set
    FileConfig {
        trim: is_flag_present(&["--trim"]),
        output: matches
            .iter()
            .find_map(|m| m.get_one::<String>("output").cloned())
            .unwrap_or("wwinp".to_string()),
    }
}

fn parse_ww_set(arguments: Vec<String>) -> Result<WWConfig> {
    let mut matches = cli_init().get_matches_from(arguments);

    let meshtal: Option<String> = matches.try_remove_one("meshtal")?;
    let number: Option<u32> = matches.try_remove_one("number")?;

    match meshtal {
        Some(_) => {
            // quickly check if all the files even exist
            if !Path::new(&meshtal.clone().unwrap()).exists() {
                return Err(anyhow!("Unable to find file \"{}\"", &meshtal.unwrap()));
            }
        }
        None => return Err(anyhow!("Empty <meshtal> positional argument in set")),
    }

    // fine to unwrap these matches because a default has been set
    Ok(WWConfig {
        meshtal: meshtal.unwrap(),
        number: number.unwrap(),
        power: powers_vector(&mut matches),
        error: errors_vector(&mut matches),
        total: matches.remove_one("total").unwrap(),
        scale: matches.remove_one("scale").unwrap(),
    })
}

fn help_flags() -> bool {
    let help = env::args()
        .filter(|p| (p.as_str() == "-h" || p.as_str() == "--help"))
        .collect::<Vec<String>>();

    if !help.is_empty() {
        let mut command = cli_init();
        if help.contains(&"--help".to_string()) {
            command
                .print_long_help()
                .expect("Could not print help message");
        } else {
            command.print_help().expect("Could not print help message");
        }
        true
    } else {
        false
    }
}

fn is_flag_present(names: &[&str]) -> bool {
    env::args().any(|a| names.contains(&a.as_str()))
}

fn collect_verbosity() -> usize {
    env::args()
        .filter(|p| (p.starts_with("-v") || p.as_str() == "--verbose"))
        .fold(0, |total, arg| match arg.as_str() {
            "--verbose" => total + 1,
            _ => total + arg.matches('v').count(),
        })
}

/// Full help message
fn cli_long_help() -> &'static str {
    "Conversion of meshtal file meshes to MCNP weight windows
    
For multiple particle types, use the '+' operator to combine multiple tallies that have the same dimensions.

Use the --vtk flag to generate Visual Toolkit files for plotting.

For advanced users, the --power and --error de-tuning factors may be set for individual energy/time groups. All groups must be explicitly provided.

Supports all mesh output formats for rectangular and cylindrical geometries. 

Typical examples 
----------------

    Convert single tally with defaults  
        $ mesh2ww file.msht 14

    Change the softening/de-tuning factor  
        $ mesh2ww file.msht 14 --power 0.8 

    Only generate weights for voxels with <10% error
        $ mesh2ww file.msht 14 --error 0.1

    Only use the 'Total' energy/time groups 
        $ mesh2ww file.msht 14 --total

    Multiply all weights by a constant factor
        $ mesh2ww file.msht 14 --scale 2.0


Mutli-particle examples 
-----------------------

    Use the '+' operator to combine meshes (same dimensions):
        $ mesh2ww file.msht 14 + run0.msht 24

    All options can be applied individually:
        $ mesh2ww fileA 14 -p 0.8 --scale 10    \\
                + fileB 24 -p 0.5 -e 0.15       \\
                + fileC 14 --total 

VTK plotting outputs 
--------------------

    Output a vtk for all weight window sets:
        $ mesh2ww file.msht 14 --vtk

    Make cylindrical meshes look rounder:
        $ mesh2ww file.msht 14 --vtk --resolution 2

    Change other advanced fromatting options:
        $ mesh2ww file.msht 14 --vtk    \\
                --format legacy-ascii   \\
                --compressor lzma       \\ 
                --endian big-endian     

Advanced de-tuning
------------------
    
    Set power factors individually for a 3x erg group mesh
        $ mesh2ww file.msht 104 --power 0.8 0.7 0.65

    Set both factors individually for a 3x erg group mesh
        $ mesh2ww file.msht 104         \\
                  --power 0.8 0.7 0.65  \\
                  --error 1.0 0.9 1.0

    Set factors individually for 3x erg group, 2x time groups
        $ mesh2ww file.msht 104    \\
                  --power 0.8 0.7  \\   => (e0,t0) (e0,t1)
                          0.9 0.8  \\   => (e1,t0) (e1,t1)
                          0.7 0.6  \\   => (e2,t0) (e2,t1)
                  
Notes
-----

The MAGIC method is used to convert tallies to mesh-based global weight windows. Weights are calculated as (0.5 * norm_flux)^power. Any voxels with errors larger than --error are set to analogue. Flux data are normalised by the maximum flux of each energy/time group.

CuV voidoff=yes will not output results for void cells. These will therefore always be analogue. CuV also has a habit of including -ve results, which are unphysical and considered to be 0.0 in this implementation."
}

fn after_help_message() -> &'static str {
    "Typical use: mesh2ww file.msht 104 -p 0.70 -o wwinp\n\nSee --help for detail and examples"
}

fn usage_message() -> &'static str {
    "mesh2ww <meshtal> <number> [options] [+]"
}

fn positional_args() -> [Arg; 2] {
    [arg_meshtal(), arg_number()]
}

fn debug_args() -> [Arg; 3] {
    [arg_verbosity(), arg_quiet(), arg_help()]
}

fn optional_args() -> [Arg; 11] {
    [
        arg_power(),
        arg_error(),
        arg_total(),
        arg_scale(),
        arg_output(),
        arg_padding(),
        arg_vtk(),
        arg_format(),
        arg_resolution(),
        arg_endian(),
        arg_compressor(),
    ]
}

fn arg_meshtal() -> Arg {
    Arg::new("meshtal")
        .help_heading("Arguments")
        .help("Path to meshtal file")
        .action(ArgAction::Set)
        .value_parser(value_parser!(String))
}

fn arg_number() -> Arg {
    Arg::new("number")
        .help_heading("Arguments")
        .help("Mesh tally identifier")
        .long_help("e.g. 104 for the FMESH104:n card")
        .value_parser(value_parser!(u32))
        .action(ArgAction::Set)
}

fn arg_verbosity() -> Arg {
    Arg::new("verbose")
        .short('v')
        .long("verbose")
        .help_heading("Flags")
        .help("Verbose logging (-v, -vv)")
        .long_help(
            "If specified, the default log level of INFO is increased to DEBUG (-v) or TRACE (-vv). Errors and Warnings are always logged unless quiet (-q) is used.",
        )
        .required(false)
        .action(ArgAction::Count)
}

fn arg_quiet() -> Arg {
    Arg::new("quiet")
        .short('q')
        .long("quiet")
        .help_heading("Flags")
        .help("Supress all log output (overrules --verbose)")
        .required(false)
        .action(ArgAction::SetTrue)
}

fn arg_help() -> Arg {
    Arg::new("help")
        .long("help")
        .help_heading("Flags")
        .help("Print help info (see more with '--help')")
        .required(false)
        .action(ArgAction::HelpShort)
}

fn arg_power() -> Arg {
    Arg::new("power")
            .short('p')
            .long("power")
            .help_heading("Weight options")
            .help("Set the softening/de-tuning factor")
            .long_help(
                "Set the softening/de-tuning factor\n\nDefault 0.70. The softening/de-tuning factor is applied to the weights as ww => ww^(<num>).\n\nFor advanced use, multiple values are provided. These will apply to each energy/time group individually (see examples above).",
            )
            .required(false)
            .action(ArgAction::Set)
            .value_delimiter(' ')
            .num_args(1..)
            .value_parser(value_parser!(f64))
            .default_value("0.7")
            .value_name("num")
            .hide_default_value(true)
}

fn powers_vector(matches: &mut ArgMatches) -> Vec<f64> {
    matches
        .remove_many::<f64>("power")
        .unwrap_or_default()
        .collect()
}

fn arg_error() -> Arg {
    Arg::new("error")
            .short('e')
            .long("error")
            .help_heading("Weight options")
            .help("Maximum rel. error, use analogue above")
            .long_help(
                "Maximum rel. error, use analogue above\n\nDefault 1.0 (100%). Relative errors above the provided value are set to zero, and will continue to use analogue transport until better statistics are available.\n\nFor advanced use, multiple values are provided. These will apply to each energy/time group individually (see examples above).",
            )
            .required(false)
            .action(ArgAction::Set)
            .value_delimiter(' ')
            .num_args(1..)
            .value_parser(value_parser!(f64))
            .default_value("1.0")
            .value_name("num")
            .hide_default_value(true)
}

fn errors_vector(matches: &mut ArgMatches) -> Vec<f64> {
    matches
        .remove_many::<f64>("error")
        .unwrap_or_default()
        .collect()
}

fn arg_total() -> Arg {
    Arg::new("total")
            .short('t')
            .long("total")
            .help_heading("Weight options")
            .help("Weights from 'Total' groups only")
            .long_help(
                "Weights from 'Total' groups only\n\nOften it can be desirable to simply generate the weight window mesh from the 'Total' groups rather than every explicit energy/time group.\n\nThis probably the recommended use case for any finely binned groups, as nobody should really be trying to optimise for every energy in a 175-group mesh anyway.",
            )
            .required(false)
            .action(ArgAction::SetTrue)
}

fn arg_scale() -> Arg {
    Arg::new("scale")
            .short('s')
            .long("scale")
            .help_heading("Weight options")
            .help("Multiply all weights by a constant")
            .long_help(
                "Multiply all weights by a constant\n\nAll weights calculated from the mesh are typically normalised to the total flux. These may be rescaled by the value provided. e.g. --scale 10 will multiply every weight by 10.0",
            )
            .required(false)
            .action(ArgAction::Set)
            .value_parser(value_parser!(f64))
            .default_value("1.0")
            .value_name("num")
            .hide_default_value(true)
}

fn arg_output() -> Arg {
    Arg::new("output")
        .short('o')
        .long("output")
        .help_heading("Global file options")
        .help("Name of output file ('wwinp' default)")
        .long_help(
            "Defaults to \"wwinp\". Ouptut formatted to WWOUT file specification from Appendix B of the MCNP6.2 user manual.",
        )
        .required(false)
        .action(ArgAction::Set)
        .value_parser(value_parser!(String))
        .value_name("path")
        .hide_default_value(true)
}

fn arg_padding() -> Arg {
    Arg::new("trim")
        .long("trim")
        .help_heading("Global file options")
        .help("Exclude unused particles from wwinp header")
        .long_help("For multiple particle types, it is unclear (without MCNP source access) how the header is read. Experience says you need to pad the header with zeros for all the unused particle types, ordered by particle id. If this is not the case, then --trim exists to get rid of the padding.")
        .required(false)
        .action(ArgAction::SetTrue)
}

fn arg_vtk() -> Arg {
    Arg::new("vtk")
        .long("vtk")
        .help_heading("Global VTK options")
        .help("Write VTK files for plotting")
        .long_help("Flag to specify that visual toolkit plot formats should be generated for each weight window set.")
        .required(false)
        .action(ArgAction::SetTrue)
}

fn arg_resolution() -> Arg {
    Arg::new("resolution")
        .short('r')
        .long("resolution")
        .help_heading("Global VTK options")
        .help("Cylindrical mesh resolution")
        .long_help(
            "WARNING: Every vertex is defined explicitly, so large values will significantly increase memory usage and file size.\n\nInteger value for increasing angular resolution of cylindrical meshes. Cylinders are approximated to straight edge segments so it can be useful to round this off by splitting voxels into multiple smaller segments.\n\ne.g. 4 theta bins gives 4 edges and therefore looks square. Using `--resolution 3` generates 12 edges instead and looks more rounded.",
        )
        .required(false)
        .action(ArgAction::Set)
        .value_parser(value_parser!(u8))
        .value_name("cst")
        .hide_default_value(true)
}

fn arg_format() -> Arg {
    Arg::new("format")
        .short('f')
        .long("format")
        .help_heading("Global VTK options")
        .help("Set the VTK file format")
        .long_help(
            "Available visual toolkit file formats:
    > xml (default)
    > legacy-ascii
    > legacy-binary",
        )
        .required(false)
        .action(ArgAction::Set)
        .value_parser(value_parser!(VtkFormat))
        .value_name("fmt")
        .hide_default_value(true)
}

fn arg_endian() -> Arg {
    Arg::new("endian")
        .long("endian")
        .help_heading("Global VTK options")
        .help("Byte ordering/endian")
        .long_help(
            "Visit only reads big endian, most sytems are little endian. Defaults to big endian for convenience.
    > big-endian (default)
    > little-endian",
        )
        .required(false)
        .action(ArgAction::Set)
        .value_parser(value_parser!(CliByteOrder))
        .value_name("end")
        .hide_default_value(true)
}

fn arg_compressor() -> Arg {
    Arg::new("compressor")
        .long("compressor")
        .help_heading("Global VTK options")
        .help("Compression method for XML")
        .long_help(
            "Generally just use LZMA but other options are available.
    > lzma (default)
    > lz4
    > zlib
    > none",
        )
        .required(false)
        .action(ArgAction::Set)
        .value_parser(value_parser!(CliCompressor))
        .value_name("cmp")
        .hide_default_value(true)
}
