//! # The Meshtal crate
//!
//! A collection of analysis tools for interacting with MCNP meshtal files
//!
//! For command line tool usage please see the
//! [README](https://github.com/repositony/meshtal).
//!
//! ## Installation
//!
//! Direct install from github:
//!
//! ```shell
//! cargo install --git https://github.com/repositony/meshtal.git
//! ```
//!
//! ## Overview
//!
//! The crate contains several command line tools for quickly performing common
//! tasks relating to MCNP meshes. More may be added based on time/usefulness.
//!
//! | Command line   | Description                                             |
//! | -------------- | ------------------------------------------------------- |
//! | `mesh2vtk`     | Convert any meshtal tally to various VTK formats        |
//! | `mesh2ww`      | Convert any meshtal tally to a mesh-based weight window |
//! | `pointextract` | Extract voxel results for any point(s) in a mesh        |
//! | `splitmesh`    | Split meshtal tallies into individual files             |
//! | `posvol`       | Inspect and convert binary UKAEA CuV posvol files       |
//!
//! All tools are fully documented with detailed `--help` messages, including
//! examples for common use cases.
//!
//! ### Supported output formats
//!
//! For more detail, see the `OUT` keyword for the `FMESH` card definition in
//! the [MCNPv6.2](https://mcnp.lanl.gov/pdf_files/TechReport_2017_LANL_LA-UR-17-29981_WernerArmstrongEtAl.pdf)
//! or [MCNPv6.3](https://mcnpx.lanl.gov/pdf_files/TechReport_2022_LANL_LA-UR-22-30006Rev.1_KuleszaAdamsEtAl.pdf)
//! user manuals.
//!
//! | Output format                       | Description                                         |
//! | ----------------------------------- | --------------------------------------------------- |
//! | [Format::COL](crate::mesh::Format)  | Column data (MCNP default)                          |
//! | [Format::CF](crate::mesh::Format)   | Column data including voxel volume                  |
//! | [Format::CUV](crate::mesh::Format)  | UKAEA Cell-under-Voxel column data                  |
//! | [Format::IJ](crate::mesh::Format)   | 2D matrix of I (col) and J (row) data, grouped by K |
//! | [Format::IK](crate::mesh::Format)   | 2D matrix of I (col) and K (row) data, grouped by J |
//! | [Format::JK](crate::mesh::Format)   | 2D matrix of J (col) and K (row) data, grouped by I |
//! | [Format::NONE](crate::mesh::Format) | `NONE` or unknown output format                     |
//!
//! Once I get my paws on MCNPv6.3 this will be extended to include the new
//! COLSCI, CFSCI, and XDMF/HDF5 formats.
//!
//! ### Supported mesh geometries
//!
//! Currently spherical meshes are not supported because barely anyone knows
//! about them, let alone use them. They are currently a low priority.
//!
//! | Mesh geometry                                  | MCNP designators |
//! | ---------------------------------------------- | ---------------- |
//! | [Geometry::Rectangular](crate::mesh::Geometry) | rec, xyz         |
//! | [Geometry::Cylindrical](crate::mesh::Geometry) | cyl, rzt         |
//!
//! ## Advanced use
//!
//! Anyone reading these docs is likely familiar with Rust, so between us the
//! command line tools are purely for colleagues and convenience. The crate
//! itself is particularly useful, since the challenge with meshtal files is
//! always trying to parse the old MCNP outputs into something useful.
//!
//! This crate allows you to read any format into a struct with a one-liner, and
//! from there you can do whatever you want with the mesh data. All mesh formats
//! are coerced into the same core [Mesh](crate::mesh::Mesh) struct.
//!
//! ```rust
//! // import the crate
//! use meshtal::{read_meshtal, read_meshtal_target};
//!
//! // read a mesh from any of the meshtal formats
//! let mesh = read_meshtal_target("./data/meshes/fmesh_114.msht", 114).unwrap();
//!
//! // read ALL meshes in a meshtal file, in any combination of formats
//! let all_meshes = read_meshtal("./data/meshes/fmesh_114.msht").unwrap();
//!
//! // now do whatever you want with them:
//! //  - mess with the voxel data,
//! //  - extract data at various points,
//! //  - turn it into a vtk,
//! //  - turn it into a global weight window,
//! //  - etc...
//! ```
//!
//! As an overview:
//! - The [mesh] module contains all of the relevant structures and
//! functionality needed for most things.
//! - The [weights] module turns meshes into global mesh-based weight windows
//! with and writes to MCNP weight WWINP files.
//! - The [vtk] module allows for writing several of the data structures to VTK
//! formats for plotting.
//! - The [point] module provides ways of extracting specific data of interest
//! from various locations in a mesh.
//! - The [posvol] module deserialises UKAEA CuV posvol files, which contain
//! very useful cell data.
//!
//! In the background, the `nom` parser combinator library allows for some
//! extremely fast parsing, `clap` is used for command line interface, and
//! `vtkio` allows conversions to various plot formats.
//!
//! The memory usage is minimised as much as possible so that extremely large
//! meshes can be handled without using obscene amounts of RAM. MCNP uses the
//! equavalent of f64 internally, so a large [Mesh](crate::mesh::Mesh) will be
//! ~24 bytes per voxel as a rough guide.
//!
//! All of the useful functionality from the file readers and core data
//! structures are re-expoerted for convenience.

// Public facing modules
pub mod mesh;
pub mod point;
pub mod posvol;
pub mod utils;
pub mod vtk;
pub mod weights;

// note that docs are hidden to prevent confusing the current simple API
pub mod readers;

// Re-exports of useful data structures
#[doc(inline)]
pub use crate::readers::{read_meshtal, read_meshtal_target, read_posvol_file};
