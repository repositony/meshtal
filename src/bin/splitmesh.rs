//! Command line tool to split up meshtal files
#![doc(hidden)]

// standard libraries
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

// crate modules
use meshtal::utils::f;

// external crates
use anyhow::{anyhow, Result};
use clap::{arg, Parser};
use log::*;

// nom parser combinators, though a tad overkill
use nom::bytes::complete::tag;
use nom::character::complete::{digit1, space1};
use nom::sequence::{preceded, tuple};
use nom::{self, IResult};

fn main() -> Result<()> {
    // set up the command line interface and match arguments
    let cli: Cli = Cli::parse();

    // set up logging (+2 to make 'Info' the default)
    let verbosity = cli.verbose as usize + 2;
    logging_init(verbosity, cli.quiet);

    info!("Splitting \"{}\"", cli.meshtal);

    // Take a second to check the file for validity
    debug!("Checking at least one tally exists in file");
    let id = check_for_tally(&cli)?;
    debug!("  - first mesh found: fmesh {id}");

    // Only then move on to split up the file
    debug!("Writing new files");
    let writer = get_writer(&f!("{}_{id}.msht", cli.output))?;
    split_meshtal_files(&cli, writer)
}

/// Split tallies of a meshtal into individual files
///
/// By default all meshes found in the meshtal file are copied into
/// individual files.
///
/// Files are split on lines starting with a "Mesh Tally Number" tag.
///
/// Use the --tallies option to list specific tallies of interest, and
/// --output to change the output file name prefix.
///
/// Examples
/// --------
///
///  Typical use
///     $ splitmesh run0.msht
///
///  Only split off specific tallies
///     $ splitmesh run0.msht --tallies 104 204 504
///
///  Change output file names to "mymesh_104.msht"
///     $ splitmesh run0.msht --output mymesh
///
#[derive(Parser)]
#[command(
    verbatim_doc_comment,
    arg_required_else_help(true),
    before_help(banner()),
    after_help("Typical use: splitmesh run0.msht\n\nNOTE: --help shows more detail and examples"),
    term_width(70),
    hide_possible_values(true),
    override_usage("splitmesh <meshtal> [options]")
)]
struct Cli {
    // * Positional
    /// Path to input meshtal file
    #[arg(name = "meshtal")]
    meshtal: String,

    /// Only extract specific mesh tallies
    ///
    /// By default all meshes ate extracted. Use this option to specify one or
    /// more tallies to be separated out. Invalid entries are ignored.
    #[arg(help_heading("Split options"))]
    #[arg(short, long)]
    #[arg(value_parser, num_args = 1.., value_delimiter = ' ')]
    #[clap(required = false)]
    #[arg(value_name = "id")]
    tallies: Vec<u32>,

    /// Prefix for output files ('fmesh' default)
    ///
    /// Defaults to `fmesh`. File names provided are appended with the tally
    /// number as `<output>_<id>.msht`.
    #[arg(help_heading("Split options"))]
    #[arg(short, long)]
    #[arg(value_name = "path")]
    #[arg(default_value = "fmesh")]
    output: String,

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

/// Sets up logging at runtime to allow for multiple verbosity levels
fn logging_init(verbosity: usize, quiet: bool) {
    stderrlog::new()
        .modules(vec![module_path!()])
        .quiet(quiet)
        .verbosity(verbosity)
        .show_level(false)
        .color(stderrlog::ColorChoice::Never)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();
}

/// Creates a banner fot the command line
fn banner() -> String {
    let mut s = f!("{:-<1$}\n", "", 70);
    s += &f!("{:^70}\n", "Meshtal :: SplitMesh");
    s += &f!("{:-<1$}", "", 70);
    s
}

/// Helper function for cleaning up file IO boilerplate
fn get_reader(path: &str) -> Result<BufReader<File>> {
    let file: File = File::open(path)?;
    trace!("New bufreader for {path}");
    Ok(BufReader::new(file))
}

/// Helper function for cleaning up file IO boilerplate
fn get_writer(path: &str) -> Result<BufWriter<File>> {
    let file: File = File::create(path)?;
    trace!("New bufwriter for {path}");
    Ok(BufWriter::new(file))
}

/// Make sure the file contains at least one mesh before doing anything
fn check_for_tally(cli: &Cli) -> Result<u32> {
    let reader = get_reader(&cli.meshtal)?;

    for line in reader.lines() {
        let line = line.unwrap();
        if !is_new_mesh(line.trim_start()) {
            continue;
        }

        let (_, id) = mesh_id(line.trim_start())
            .map_err(|_| anyhow!("Failed to parse id from:\n \"{line}\""))?;

        // If at least one is relevent break early and carry on with the file splitting
        if cli.tallies.is_empty() || cli.tallies.contains(&id) {
            trace!("First relevant mesh tally found: fmesh {id}");
            return Ok(id);
        }
    }

    Err(anyhow!("No relevant meshes found in file"))
}

/// Copies the relevant content to appropriate files
fn split_meshtal_files(cli: &Cli, mut writer: BufWriter<File>) -> Result<()> {
    let mut is_relevant_mesh = false;

    let reader = get_reader(&cli.meshtal)?;
    for line in reader.lines() {
        let line = line.unwrap();

        // decide what to do whenever a new mesh is found
        if is_new_mesh(line.trim_start()) {
            let (_, id) = mesh_id(line.trim_start())
                .map_err(|_| anyhow!("Failed to parse id from:\n \"{line}\""))?;

            if cli.tallies.is_empty() || cli.tallies.contains(&id) {
                let output = f!("{}_{id}.msht", cli.output);
                info!("  - {output}");
                writer = BufWriter::new(File::create(&output)?);
                is_relevant_mesh = true;
            } else {
                is_relevant_mesh = false;
            }
        }

        if is_relevant_mesh {
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
        }
    }

    Ok(())
}

/// Quick check for the new tally tag
fn is_new_mesh(i: &str) -> bool {
    i.starts_with("Mesh Tally Number")
}

/// Parse the number following a `Mesh Tally Number` tag to a u32
pub fn mesh_id(i: &str) -> IResult<&str, u32> {
    let (_, tally_id) = preceded(tuple((tag("Mesh Tally Number"), space1)), digit1)(i)?;
    nom::character::complete::u32(tally_id)
}
