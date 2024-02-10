//! Command line extraction of point data from meshes
#![doc(hidden)]

// standard library
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// Crate modules
use meshtal::mesh::{Geometry, Mesh, Voxel};
use meshtal::point::{self, Point};
use meshtal::readers::{parsers, MeshtalReader};
use meshtal::utils::*;

// External crates
use anyhow::{anyhow, Ok, Result};
use clap::{arg, Parser};
use log::*;

fn main() -> Result<()> {
    // set up the command line interface and match arguments
    let cli: Cli = Cli::parse();

    // set up logging (+2 to make Info the default)
    let verbosity = cli.verbose + 2;
    logging_init(verbosity, cli.quiet);

    // check points from user before bothering with the mesh
    debug!("Collecting requested points");
    let mut points: Vec<Point> = Vec::new();
    match cli.point.is_empty() {
        false => points.push(parse_cli_point(&cli)?),
        true => {
            let path = cli.file.clone().unwrap_or("points.txt".to_string());
            points.extend(point::read_points_file(path)?)
        }
    }
    check_points(&points)?;
    info!("Point conversion successful");

    // Get the mesh tally
    info!("Reading {}", &cli.meshtal);
    let mesh = try_meshtal_read(&cli)?;
    info!("Mesh read successful");

    // find the voxels
    let voxels = point::find_voxels(&mesh, &points);
    if voxels.is_empty() {
        error!("No valid points found in mesh");
        return Err(anyhow!("No valid points found in Fmesh{}", mesh.id));
    }

    // output to a file and log to console if requested
    let description = f!("fmesh:{}, {:?}", mesh.id, mesh.geometry);
    match cli.output {
        Some(path) => results_to_file(&path, &mesh, &points, &voxels, &description)?,
        None => results_to_file("./results.dat", &mesh, &points, &voxels, &description)?,
    }

    if cli.dump {
        results_to_console(&mesh, &points, &voxels, &description)
    }

    Ok(())
}

/// Extract point data from any mesh in a meshtal file
///
/// All meshtal formats are supported for rectangular and cylindrical meshes.
///
/// If no particular energy/time groups are specified the 'Total' will be used.
///
/// If void record is 'off' for UKAEA CuV tallies, void voxels will have a flux
/// of zero.
///
/// Important: Points exactly on a voxel boundary are assumed to be in the
/// lowest bounds. Averaging of voxels in this case is a work in progress.
///
/// Examples
/// --------
///
///  Typical use:
///     $ pointextract run0.msht 104 -f points.txt -o results.dat
///
///  Point from (x,y,z) coordiantes:
///     $ pointextract run0.msht 104 -p 1.0 2.0 3.0
///
///  Point from (r,z,t) coordiantes:
///     $ pointextract run0.msht 104 -p 1.0 2.0 3.0 -t rzt
///
///  Point with specific energy/time groups:
///     $ pointextract run0.msht 104 -p 1e2 total 1.0 2.0 3.0
///
///  Multiple points from input file:
///     $ pointextract run0.msht 104 -f points.txt
///   
///     points.txt interpretation:
///         '#' or non-keyword   = ignored
///         0.0 0.2 50           = i, j, k
///         1e2 15 2.0 3.1       = energy, i, j, k
///         1e2 total 15 2.0 3.1 = energy, time, i, j, k
///         rzt, cyl, xyz, rec   = keywords
///   
///     All points following a keyword are interpreted as the specified
///     geometry. Otherwise coordiantes assumed to match the mesh type.
///
///     Coordinates are converted into the correct coordinate system
///     in the background during the search.
///
#[allow(rustdoc::invalid_rust_codeblocks)]
#[derive(Parser, Debug)]
#[command(
    verbatim_doc_comment,
    arg_required_else_help(true),
    before_help(banner()),
    after_help("Typical use: pointextract run0.msht 104 -f points.txt -o results.dat\n\nNOTE: --help shows more detail and examples"),
    term_width(70),
    hide_possible_values(true),
    override_usage("pointextract <meshtal> <number> [options]")
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
    /// Quickly find a single point
    ///
    /// Expressed as (e,t,i,j,k), but only the i, j, k coordinates are required.
    ///     
    /// E.g. -p 1.0 2.0 3.0 as a minimum input.
    ///
    /// Energy and time targets may be ommited for just 'total' bins, or
    /// specified to be explicit.
    ///     
    /// E.g. -p total 1e22 1.0 2.0 3.0
    ///
    /// Assumes coordiantes are appropriate for the mesh type provided by
    /// --type, e.g. '--type rzt' for cylindrical coordinates
    #[arg(help_heading("Point options"))]
    #[arg(short, long)]
    #[arg(allow_negative_numbers(true),
    num_args(1..=5))]
    #[clap(required = false)]
    #[arg(value_name = "coords")]
    point: Vec<String>,

    /// Coordinate system for --point (e.g. xyz)
    ///
    /// Specified the coordinate system to interpret the --point argument by.
    ///
    /// Systems available:
    ///     > xyz = Cartesian coordinate system (default)
    ///     > rzt = Cylindrical coordinate system
    #[arg(help_heading("Point options"))]
    #[arg(short, long, value_enum)]
    #[arg(hide_default_value(true))]
    #[arg(default_value_t = Geometry::Rectangular)]
    #[arg(verbatim_doc_comment)]
    #[arg(id = "type")]
    type_: Geometry,

    /// File containing multiple points
    ///
    /// If no points explicitly given using --point, the tool will
    /// automatically search for a file to read points from instead.
    /// Defaults to "points.txt".
    ///
    ///     points.txt interpretation:
    ///         '#' or non-keyword   = ignored
    ///         0.0 0.2 50           = i, j, k
    ///         1e2 15 2.0 3.1       = energy, i, j, k
    ///         1e2 total 15 2.0 3.1 = energy, time, i, j, k
    ///         rzt, cyl, xyz, rec   = keywords
    ///  
    /// Points following a keyword read as the specified geometry.
    /// Otherwise coordiantes assumed to match the mesh type.
    #[arg(help_heading("Point options"))]
    #[arg(short, long)]
    #[arg(value_name = "path")]
    #[arg(verbatim_doc_comment)]
    file: Option<String>,

    /// Write results to a text file
    ///
    /// If provided, the results of all points are written to this file.
    /// Includes any failed searches for user reference.
    #[arg(help_heading("Output options"))]
    #[arg(short, long)]
    #[clap(required = false)]
    #[arg(allow_negative_numbers(true),
    num_args(0..2))]
    #[arg(value_name = "path")]
    output: Option<String>,

    /// Print results to the terminal
    ///
    /// Tables are broken up into search points and found voxels for
    /// readability. Row indexing is provided for convenience.
    #[arg(help_heading("Output options"))]
    #[arg(short, long)]
    #[clap(required = false)]
    dump: bool,

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

fn try_meshtal_read(cli: &Cli) -> Result<Mesh> {
    let path: &Path = Path::new(&cli.meshtal);

    let mut reader = MeshtalReader::new();
    println!("verbosity {}", cli.verbose);
    if cli.quiet || cli.verbose > 1 {
        reader.disable_progress();
    }
    reader.set_target_id(cli.number);

    let mut mesh = reader.parse(path)?;
    Ok(std::mem::take(&mut mesh[0]))
}

fn check_points(points: &[Point]) -> Result<()> {
    if points.is_empty() {
        Err(anyhow!("No valid point input found"))
    } else {
        Ok(())
    }
}

fn parse_cli_point(cli: &Cli) -> Result<Point> {
    match &mut parsers::points_file_point(&cli.point.join(" ")) {
        nom::IResult::Ok(data) => {
            let (_, point) = data;
            point.coordinate_type = cli.type_;
            Ok(point.to_owned())
        }
        _ => Err(anyhow!(
            "Failed to parse \"{}\" to a Point",
            &cli.point.join(" ")
        )),
    }
}

/// Write all results to a file
///
/// This will include those that failed for the user's reference.
fn results_to_file(
    path: &str,
    mesh: &Mesh,
    points: &[Point],
    voxels: &[Option<Voxel>],
    description: &str,
) -> Result<()> {
    info!("Writing results to {}", path);
    let f = File::create(path)?;
    let mut f = BufWriter::new(f);
    f.write_all(points_table(points).as_bytes())?;
    f.write_all(b"\n\n\n")?;
    f.write_all(voxels_table(mesh, voxels, description).as_bytes())?;
    Ok(())
}

/// Log the results to console
///
/// Will be exactly the same as the table that gets dumped to file
fn results_to_console(mesh: &Mesh, points: &[Point], voxels: &[Option<Voxel>], description: &str) {
    println!("\n{}", points_table(points));
    println!("\n\n{}", voxels_table(mesh, voxels, description));
}

/// generates a banner for cli tool consistency
fn banner() -> String {
    let mut s = f!("{:-<1$}\n", "", 70);
    s += &f!("{:^70}\n", "Meshtal :: PointExtract");
    s += &f!("{:-<1$}", "", 70);
    s
}

fn logging_init(verbosity: u8, quiet: bool) {
    stderrlog::new()
        .modules(vec![
            module_path!(),
            "meshtal::point",
            "meshtal::mesh",
            "meshtal::readers::meshtal_file",
        ])
        .quiet(quiet)
        .verbosity(verbosity as usize)
        .show_level(false)
        .color(stderrlog::ColorChoice::Never)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();
}

fn points_table(points: &[Point]) -> String {
    let mut s = target_heading();
    s += &f!("\n{}\n", target_columns());
    s += &"-".repeat(78);
    points
        .iter()
        .enumerate()
        .for_each(|(i, p)| s += &f!("\n{i:^6}{}", target_row(p)));
    s
}

fn voxels_table(mesh: &Mesh, voxels: &[Option<Voxel>], description: &str) -> String {
    let mut s = voxel_heading(description);
    s += &f!("\n{}\n", voxel_columns());
    s += &"-".repeat(92);
    voxels
        .iter()
        .enumerate()
        .for_each(|(i, v)| s += &f!("\n{i:^6}{}", voxel_row(mesh, v)));
    s
}

fn target_heading() -> String {
    f!("{:^72}", "Points to search")
}

fn target_columns() -> String {
    let mut s = f!("{:^6}", "id");
    s += &f!("{:^13}", "energy");
    s += &f!("{:^13}", "time");
    s += &f!("{:^13}", "i_coord");
    s += &f!("{:^13}", "j_coord");
    s += &f!("{:^13}", "k_coord");
    s += &f!("{:^6}", "system");
    s
}

fn target_row(point: &Point) -> String {
    let mut s = f!("{:^13}", f!("{}", point.e));
    s += &f!("{:^13}", f!("{}", point.t));
    s += &f!("{:^13}", point.i.sci(5, 2));
    s += &f!("{:^13}", point.j.sci(5, 2));
    s += &f!("{:^13}", point.k.sci(5, 2));
    s += &f!(" {:^7}", f!("{}", point.coordinate_type));
    s
}

fn voxel_heading(description: &str) -> String {
    let s = f!("Voxels found ({description})");
    f!("{:^92}", s)
}

fn voxel_columns() -> String {
    let mut s = f!("{:^6}", "id");
    s += &f!("{:^13}", "energy");
    s += &f!("{:^13}", "time");
    s += &f!("{:^13}", "i_coord");
    s += &f!("{:^13}", "j_coord");
    s += &f!("{:^13}", "k_coord");
    s += &f!("{:^13}", "result");
    s += &f!("{:^8}", "error");
    s
}

fn voxel_row(mesh: &Mesh, voxel: &Option<Voxel>) -> String {
    match voxel {
        None => f!("{:^92}", "Not found in mesh"),
        Some(v) => match mesh.voxel_coordinates(v.index) {
            Result::Ok(c) => {
                let mut s = f!("{:^13}", f!("{}", c.energy));
                s += &f!("{:^13}", f!("{}", c.time));
                s += &f!("{:^13}", c.i.sci(5, 2));
                s += &f!("{:^13}", c.j.sci(5, 2));
                s += &f!("{:^13}", c.k.sci(5, 2));
                s += &f!("{:^13}", v.result.sci(5, 2));
                s += &f!("{:^8.4}", v.error);
                s
            }
            Result::Err(_) => {
                let mut s = f!("{:^63}", "Unable to infer voxel coordiantes");
                s += &f!("{:>13}", v.result.sci(5, 2));
                s += &f!("{:>8.4}", v.error);
                s
            }
        },
    }
}
