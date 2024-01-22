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
//! the relative error cutoff (`-e`/`--error`) for generating weights.  
//!
//! ```bash
//! mesh2ww run0.msht 104 --power 0.70 --error 0.1
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
#![doc(hidden)]

// standard library
use std::env;
use std::path::Path;

// crate modules
use meshtal::mesh::Mesh;
use meshtal::readers::MeshtalReader;
use meshtal::utils::*;
use meshtal::weights::{self, WeightWindow};

// external crates
use anyhow::{anyhow, Result};
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use log::*;

// Convenience items
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
    let arg_sets = process_argument_sets();

    // collect up all weight windows, just exclude any missing and warn the user
    info!("Getting weights");
    let particle_weights = collect_weight_windows(arg_sets)?;

    // // Write the weight window file
    // let output = output_args().unwrap_or("wwinp".to_string());
    // debug!("Output = \"{output}\"");

    // info!("Writing to {output}");
    // weights::write_multi_particle(&particle_weights, &output, is_padded());

    // for ww in particle_weights {
    //     info!(
    //         "Weighted voxels {:.2}% ({:?})",
    //         ww.non_analogue_percentage(),
    //         ww.particle
    //     );
    // }

    // info!("Conversion complete");
    Ok(())
}

#[derive(Debug)]
struct Cli {
    meshtal: String,
    number: u32,
    power: Vec<f64>,
    error: Vec<f64>,
    total: bool,
    scale: f64,
}

fn collect_weight_windows(arg_sets: Vec<Cli>) -> Result<Vec<WeightWindow>> {
    let mut particle_weights: Vec<WeightWindow> = Vec::with_capacity(arg_sets.len());
    for cli in &arg_sets {
        // read mesh data from the meshtal file
        info!("Reading {}", &cli.meshtal);
        let mesh = try_meshtal_read(cli)?;

        // todo: ok so annoyingly the particle checking has to be done here
        // todo: getting more and more worth having a preprocessing step
        // todo: probably needs a new reader or short circuit in existing
        if particle_weights
            .iter()
            .any(|ww| ww.particle == mesh.particle)
        {
            info!("{:?} type already included, skipping...", mesh.particle);
            continue;
        }

        // convert mesh into WWMesh object for writing/further manipulation
        info!("Generating voxel weights");
        let mut ww = generate_weight_window(&mesh, cli);

        // Multiply weights by a constant factor if one is provided
        if cli.scale != 1.0 {
            info!("Scaling results by {}", cli.scale);
            ww.scale(cli.scale);
        }

        particle_weights.push(ww);
    }

    Ok(particle_weights)
}

fn generate_weight_window(mesh: &Mesh, cli: &Cli) -> WeightWindow {
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

fn process_argument_sets() -> Vec<Cli> {
    split_argument_sets()
        .iter()
        .filter_map(|arg_set| {
            let cli = to_cli_struct(arg_set.to_owned());
            if let Err(e) = &cli {
                warn!("Skipping {:?}", arg_set);
                warn!("Reason: {:?}", e);
            }
            cli.ok()
        })
        .collect::<Vec<Cli>>()
}

fn to_cli_struct(arguments: Vec<String>) -> Result<Cli> {
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

    Ok(Cli {
        meshtal: meshtal.unwrap(), // fine to unwrap if a default has been set
        number: number.unwrap(),
        power: powers_vector(&mut matches),
        error: errors_vector(&mut matches),
        total: matches.remove_one("total").unwrap(),
        scale: matches.remove_one("scale").unwrap(),
    })
}

fn try_meshtal_read(cli: &Cli) -> Result<Mesh> {
    let path: &Path = Path::new(&cli.meshtal);

    let mut reader = MeshtalReader::new();
    reader.set_target_id(cli.number);
    if is_quiet() || collect_verbosity() > 0 {
        reader.disable_progress();
    }

    let mut mesh = reader.parse(path)?;
    Ok(std::mem::take(&mut mesh[0]))
}

fn output_args() -> Option<String> {
    let args_iter = env::args();
    let args_iter2 = env::args().skip(1);

    args_iter.zip(args_iter2).find_map(|(a, b)| {
        if a.as_str() == "-o" || a.as_str() == "--output" {
            Some(b)
        } else {
            None
        }
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

fn is_quiet() -> bool {
    // see if the quiet flags are anywhere
    env::args().any(|a| a.as_str() == "--quiet" || a.as_str() == "-q")
}

fn is_padded() -> bool {
    // if the user sets --trim, remove the padding
    !env::args().any(|a| a.as_str() == "--trim")
}

fn collect_verbosity() -> usize {
    env::args()
        .filter(|p| (p.starts_with("-v") || p.as_str() == "--verbose"))
        .fold(0, |total, arg| match arg.as_str() {
            "--verbose" => total + 1,
            _ => total + arg.matches('v').count(),
        })
}

fn split_argument_sets() -> Vec<ArgSet> {
    let name = env::args().next().unwrap();
    let raw_args = env::args().skip(1).collect::<Vec<String>>();
    let mut tallies = Vec::with_capacity(5);

    // todo fix this clusterfuck
    for s in raw_args.split(|p| p == "+") {
        let mut v = vec![name.clone()];
        let a = s.to_owned().clone();
        v.extend(a);
        tallies.push(v);
        trace!("{:?}", tallies.last());
    }

    tallies
}

/// Creates tonybox banner

fn banner() -> String {
    let mut s = f!("{:-<1$}\n", "", 70);
    s += &f!("{:^70}\n", "Meshtal :: MeshToWW");
    s += &f!("{:-<1$}", "", 70);
    s
}

// initialise logging for all relevant modules with variable verbosity

fn logging_init() {
    stderrlog::new()
        .modules(vec![
            module_path!(),
            "meshtal::readers",
            "meshtal::mesh",
            "meshtal::weights",
        ])
        .quiet(is_quiet())
        .verbosity(collect_verbosity() + 2) // +2 to make the default "Info"
        .show_level(false)
        .color(stderrlog::ColorChoice::Never)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();
}

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

fn cli_long_help() -> &'static str {
    "Conversion of meshtal file meshes to MCNP weight windows
    
For multiple particle types, use the '+' operator to combine multiple tallies that have the same dimensions.

The MAGIC method is used to convert tallies to mesh-based global weight windows. Weights are calculated as (0.5 * (flux / reference_flux)).powf(power), with any voxels with errors larger than --error set to analogue. The reference flux is the maximum seen within each energy/time group voxel set.

Supports all mesh output formats for rectangular and cylindrical geometries. 

Typical examples 
----------------

    Convert single tally with defaults  
        $ mesh2ww run0.msht 14

    Change the softening factor and error cut  
        $ mesh2ww run0.msht 14 -p 0.8 -e 0.1

    Only use the 'Total' energy/time groups 
        $ mesh2ww run0.msht 14 --total

    Multiply all weights by a constant factor
        $ mesh2ww run0.msht 14 --scale 2.0


Mutli-particle examples 
-----------------------

    Use the '+' operator to combine meshes (same dimensions):
        $ mesh2ww run0.msht 14 + run0.msht 24

    All options can be applied individually:
        $ mesh2ww fileA 14 -p 0.8 --scale 10    \\
                + fileB 24 -p 0.5 -e 0.15       \\
                + fileC 14 --total 

Notes
-----

CuV voidoff=yes will not output results for void cells which will therefore always be analogue. CuV also has a habit of including -ve results, which are unphysical and considered to be 0.0 in this implementation."
}

fn after_help_message() -> &'static str {
    "Typical use: mesh2ww run0.msht 104 -p 0.70 -o wwinp\n\nSee --help for detail and examples"
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

fn optional_args() -> [Arg; 6] {
    [
        arg_power(),
        arg_error(),
        arg_total(),
        arg_scale(),
        arg_output(),
        arg_padding(),
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

fn arg_output() -> Arg {
    Arg::new("output")
        .short('o')
        .long("output")
        .help_heading("Global options")
        .help("Name of output file ('wwinp' default)")
        .long_help(
            "Defaults to \"wwinp\". Ouptut formatted to WWOUT file specification from Appendix B of the MCNP6.2 user manual.",
        )
        .required(false)
        .action(ArgAction::Set)
        .value_parser(value_parser!(String))
        .default_value("wwinp")
        .value_name("path")
        .hide_default_value(true)
}

fn arg_power() -> Arg {
    Arg::new("power")
            .short('p')
            .long("power")
            .help_heading("Weight options")
            .help("Set the softening/de-tuning factor")
            .long_help(
                "Set the softening/de-tuning factor\n\nDefault 0.70. The softening/de-tuning factor is applied to the weights as ww => ww^(<value>).",
            )
            .required(false)
            .action(ArgAction::Set)
            .value_delimiter(' ')
            .num_args(1..)
            .value_parser(value_parser!(f64))
            .default_value("0.7")
            .value_name("values")
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
                "Maximum rel. error, use analogue above\n\nDefault 1.0 (100%). Relative errors above the provided value are set to zero, and will continue to use analogue transport until better statistics are available.",
            )
            .required(false)
            .action(ArgAction::Set)
            .value_delimiter(' ')
            .num_args(1..)
            .value_parser(value_parser!(f64))
            .default_value("1.0") // horrible I know
            .value_name("value")
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
            .value_name("cst")
            .hide_default_value(true)
}

fn arg_padding() -> Arg {
    Arg::new("trim")
        .long("trim")
        .help_heading("Global options")
        .help("Exclude unused particle types from header")
        .required(false)
        .action(ArgAction::SetTrue)
}
