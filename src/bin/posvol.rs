//! Command line tool to inspect and convert posvol files
#![doc(hidden)]

// standard libraries
use std::fs::File;
use std::io::{BufWriter, Write};

// crate modules
use meshtal::posvol::Posvol;
use meshtal::readers::read_posvol_file;
use meshtal::utils::f;

// external crates
use anyhow::Result;
use clap::{arg, Parser};
use log::*;

fn main() -> Result<()> {
    // set up the command line interface and match arguments
    let cli: Cli = Cli::parse();

    // set up logging (+2 to make 'Info' the default)
    let verbosity = cli.verbose as usize + 2;
    logging_init(verbosity, cli.quiet);

    // Try to read the posvol binary
    info!("Reading {}", &cli.file);
    let posvol = read_posvol_file(&cli.file)?;

    // Log a summary of the file parameters to the terminal for reference
    if !cli.quiet {
        print_summary(&posvol);
    }

    // Log a summary of the file parameters to the terminal for reference
    if cli.ascii {
        match &cli.pretty {
            true => write_pretty_ascii(&posvol, &cli)?,
            false => write_raw_ascii(&posvol, &cli)?,
        };
    }

    if cli.json {
        write_json(&posvol, &cli)?;
    }

    Ok(())
}

/// Inspect UKAEA CuV posvol binaries
///
/// Very simple reader for posvol binaries, skipping the need to open a
/// special viewer or sort it out manually just to check simple properties.
///
/// Allows for 1:1 conversion to ASCII, but also JSON and a more readable
/// text file.
///
/// Examples
/// --------
///
///  Print a summary of dimension properties
///     $ posvol plot_fmesh_104.bin
///
///  Convert to ASCII
///     $ posvol plot_fmesh_104.bin --ascii
///     $ posvol plot_fmesh_104.bin --ascii --pretty
///
///  Convert to JSON
///     $ posvol plot_fmesh_104.bin --json
///
///  Change file names to "myfile"
///     $ posvol plot_fmesh_104.bin --json --ascii --output myfile
///
/// Notes
/// -----
///
/// The endian is assumed to be the same as the native type of the system
/// this tool is run on. If needed, an option can be provided in future
/// updates.
#[derive(Parser)]
#[command(
    verbatim_doc_comment,
    arg_required_else_help(true),
    before_help(banner()),
    after_help(
        "Typical use: posvol plot_fmesh_104.bin\n\nNOTE: --help shows more detail and examples"
    ),
    term_width(70),
    hide_possible_values(true),
    override_usage("posvol <file> [options]")
)]
struct Cli {
    // * Positional
    /// Path to posvol binary file
    #[arg(name = "file")]
    file: String,

    /// Generate a JSON file ('posvol.json' default)
    ///
    /// Defaults to `posvol.json` if no name provided.
    #[arg(help_heading("Posvol options"))]
    #[arg(short, long)]
    json: bool,

    /// Generate an ASCII file ('posvol.txt' default)
    ///
    /// Defaults to `posvol.txt` if no name provided.
    #[arg(help_heading("Posvol options"))]
    #[arg(short, long)]
    ascii: bool,

    /// Format the ASCII file for readability
    #[arg(help_heading("Posvol options"))]
    #[arg(short, long)]
    pretty: bool,

    /// Name of output file (excl. extension)
    ///
    /// Defaults to `posvol.<ext>`, and will automatically set the relevant
    /// extension.
    #[arg(help_heading("Vtk options"))]
    #[arg(short, long)]
    #[arg(value_name = "path")]
    output: Option<String>,

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
    s += &f!("{:^70}\n", "Meshtal :: Posvol Inspector");
    s += &f!("{:-<1$}", "", 70);
    s
}

/// Helper function for cleaning up file IO boilerplate
fn get_writer(path: &str) -> Result<BufWriter<File>> {
    let file: File = File::create(path)?;
    debug!("New bufwriter for {path}");
    Ok(BufWriter::new(file))
}

/// Write posvol file to ascii
fn write_pretty_ascii(posvol: &Posvol, cli: &Cli) -> Result<()> {
    let output = match &cli.output {
        Some(o) => f!("{o}.ascii"),
        None => "posvol.ascii".to_string(),
    };

    debug!("Writing pretty ASCII format to {}", output);
    let mut writer = get_writer(&output)?;

    writeln!(writer, "Converted from {}\n", cli.file)?;

    // write the block 1 information
    writeln!(writer, "Total cells: {}", posvol.number_of_cells())?;
    writeln!(writer, "Mesh bounds in i: {}", posvol.dimensions.n_x)?;
    writeln!(writer, "Mesh bounds in j: {}", posvol.dimensions.n_y)?;
    writeln!(writer, "Mesh bounds in k: {}", posvol.dimensions.n_z)?;
    writeln!(writer, "Sample resolution i: {}", posvol.dimensions.res_x)?;
    writeln!(writer, "Sample resolution j: {}", posvol.dimensions.res_y)?;
    writeln!(writer, "Sample resolution k: {}", posvol.dimensions.res_z)?;

    writeln!(writer, "\nList of cells, grouped by voxel:\n")?;
    // write the block 2 information
    for subset in posvol.subvoxels() {
        let s = subset
            .iter()
            .map(|cell| f!("{cell}"))
            .collect::<Vec<String>>()
            .join(" ");

        writeln!(writer, "{}\n", textwrap::fill(&s, 80))?;
    }
    Ok(())
}

fn write_raw_ascii(posvol: &Posvol, cli: &Cli) -> Result<()> {
    let output = match &cli.output {
        Some(o) => f!("{o}.ascii"),
        None => "posvol.ascii".to_string(),
    };

    debug!("Writing raw ASCII format to {}", output);
    let mut writer = get_writer(&output)?;

    // write the block 1 information
    write!(writer, "24 ")?;
    write!(writer, "{} ", posvol.dimensions.res_x)?;
    write!(writer, "{} ", posvol.dimensions.res_y)?;
    write!(writer, "{} ", posvol.dimensions.res_z)?;
    write!(writer, "{} ", posvol.dimensions.n_x)?;
    write!(writer, "{} ", posvol.dimensions.n_y)?;
    write!(writer, "{} ", posvol.dimensions.n_z)?;
    write!(writer, "24 ")?;

    // write the block 2 information
    write!(writer, "{} ", posvol.number_of_cells())?;

    for cell in &posvol.cells {
        write!(writer, "{cell} ")?;
    }
    write!(writer, "{}", posvol.number_of_cells())?;
    Ok(())
}

/// Write posvol file to json
fn write_json(posvol: &Posvol, cli: &Cli) -> Result<()> {
    let output = match &cli.output {
        Some(o) => f!("{o}.json"),
        None => "posvol.json".to_string(),
    };

    debug!("Writing JSON format to {}", output);
    let writer = get_writer(&output)?;
    Ok(serde_json::to_writer_pretty(writer, posvol)?)
}

/// Write summary to the terminal
fn print_summary(posvol: &Posvol) {
    let mut s = "Summary of binary file\n".to_string();
    s += &f!(
        "voxels  : {} ({}x{}x{})\n",
        posvol.number_of_voxels(),
        posvol.dimensions.n_x - 1,
        posvol.dimensions.n_y - 1,
        posvol.dimensions.n_z - 1
    );
    s += &f!(
        "samples : {} ({}x{}x{})\n",
        posvol.number_of_subvoxels(),
        posvol.dimensions.res_x,
        posvol.dimensions.res_y,
        posvol.dimensions.res_z
    );
    s += &f!(
        "cells   : {} ({}x{})",
        posvol.number_of_cells(),
        posvol.dimensions.number_of_voxels(),
        posvol.dimensions.number_of_subvoxels(),
    );
    println!("{s}")
}
