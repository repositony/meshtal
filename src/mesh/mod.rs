//! Core mesh library
//!
//! # Overview
//!
//! Module for storing and using mesh data from all output formats and geometry
//! types. All of the parsing and reader logic is re-exported to make reading
//! files very simple, regardless of format.
//!
//! ```rust
//! // Extract all meshes from a file into a Vec<Mesh>
//! let mesh_list = meshtal::read_meshtal("./data/meshes/fmesh_114.msht").unwrap();
//!
//! // Extract just one target mesh from a file into a Mesh
//! let mesh = meshtal::read_meshtal_target("./data/meshes/fmesh_114.msht", 114).unwrap();
//! ```
//!
//! Mesh tally data are stored in the common [Mesh] type. All [Mesh] methods are
//! therefore available for any format and any geometry type.
//!
//! # Quickstart
//!
//! Include the [meshtal](crate) crate in the `Cargo.toml` dependencies
//!
//! ```toml
//! [dependencies]
//! meshtal = "0.13.1"
//! ```
//!
//! Example to read in a mesh tally and perform varius operations.
//!
//! ```ignore
//! use meshtal::mesh;
//! fn main() {
//!     // Read mesh 104 from the meshtal file
//!     let mesh = mesh::read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
//!     // print a summary of the mesh (Display trait impemented)
//!     println!("{mesh}");
//!
//!     // points of interest
//!     let points = vec![
//!         Point::from_vec(20.5, 3.0, 52.7),
//!         Point::from_vec(20.5, 4.0, 10.0),
//!         Point::from_vec(-63.1, 15.0, 10.2),
//!     ]
//!     // find the corresponding voxels and output results
//!     let voxels = point::get_voxels(&mesh, &points);
//!     let description = f!("fmesh{}, {:?}", mesh.id, mesh.geometry);
//!     point::results_to_console(&points, &voxels, &description);
//! }
//! ```

// Split into subfiles for development, but anything important is re-exported
mod core;
mod particle;
mod voxel;

// inline important the mesh-related modules for a nice public API
#[doc(inline)]
pub use crate::mesh::core::{Format, Geometry, Mesh};

#[doc(inline)]
pub use crate::mesh::particle::Particle;

#[doc(inline)]
pub use crate::mesh::voxel::{Group, Voxel, VoxelCoordinate};

#[doc(inline)]
pub use crate::readers::{read_meshtal, read_meshtal_target};
