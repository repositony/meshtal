//! Utilities for extracting points from meshes
//!
//! The main entry points of the module are [find_voxel()] and [find_voxels()]
//! for returning the [Voxel]s found at each location.
//!
//! It is not assumed that the [Point] provided is in the format appropriate for
//! the mesh geometry, and will be quitely converted to the correct geometry
//! type in the background. i.e. [Geometry::Rectangular] for XYZ meshes and
//! [Geometry::Cylindrical] for RZT.
//!
//! ## Defining points directly
//!
//! This can of course be done thgough the struct explicitly, but for
//! convenience there are several methods for initialising points without
//! having to define common defaults.
//!
//! ```rust
//! # use meshtal::point::Point;
//! # use meshtal::mesh::{Geometry, Group};
//! // Explcict definition
//! Point {
//!     e: Group::Total,
//!     t: Group::Total,
//!     i: 1.0,
//!     j: 2.0,
//!     k: 3.0,
//!     coordinate_type: Geometry::Rectangular
//! };
//!
//! // Convenience methods
//! Point::from_xyz(1.0, 2.0, 3.0);
//! Point::from_rzt(1.0, 2.0, 0.5);
//! Point::from_xyz_vec(&vec![1.0, 2.0, 3.0]);
//! Point::from_rzt_vec(&vec![1.0, 2.0, 0.5]);
//! ```
//!
//! Another useful method set converted between geometry types
//!
//! ```rust
//! # use meshtal::point::Point;
//! # use meshtal::mesh::{Geometry, Group};
//! // Default rectangular point
//! let mut point = Point::new();
//!
//! // Convert to cylindrical equivalent
//! point = point.as_rzt();
//!
//! // Convert back to original cartesian
//! point = point.as_xyz();
//! ```
//!
//! ## Points from an input file
//!  
//! Using [read_points_file()] reads an input file that will be interpreted with
//! the following rules for each line:
//!
//! | Example line               | Interpretation           |
//! | -------------------------- | ------------------------ |
//! | Starts with `#`            | comment                  |
//! | `rzt`, `cyl`, `xyz`, `rec` | geometry keyword         |
//! | 1.0 2.0 3.0                | i, j, k                  |
//! | 1e2  1.0 2.0 3.0           | energy, i, j, k          |
//! | 1e2 total 1.0 2.0 3.0      | energy, time, i, j, k    |
//!
//! Anything else is ignored. For examples see `data/points.txt`.
//!
//! Example
//! ```rust
//! # use meshtal::point::{read_points_file, Point};
//! // Read any valid user points from a text file
//! let points: Vec<Point> = read_points_file("./data/points.txt").unwrap();
//! ```
//!
//! Any location specified in the input file will also be converted to the
//! correct mesh geometry type automatically for convenience.

// internal modules
use crate::mesh::{Geometry, Group, Mesh, Voxel};
use crate::utils::*;

#[doc(inline)]
pub use crate::readers::read_points_file;

// external crates
use anyhow::{anyhow, Result};
use log::trace;

/// Find the voxel corresponding to a specific [Point] in the mesh
///
/// This implementation is very efficient but relies on the standardised
/// order of MCNP output data. This is ensured by sorting during parsing for
/// the matrix data formats, but will prodice incorrect results for unordered
/// voxels.
///
/// Will return `None` for any point that is outside the mesh bounds.
pub fn find_voxel(mesh: &Mesh, point: &Point) -> Option<Voxel> {
    let point = &convert_point(mesh.geometry, point);

    let idx = point_to_voxel_index(mesh, point);
    match idx {
        Ok(i) => Some(mesh.voxels[i]),
        Err(_) => None,
    }
}

/// Find the voxels corresponding to multiple [Point]s in the mesh
///
/// Just loops over the vector and calls `get_voxel()` for you, lazy bastards.
/// Will return None for any point that is outside the mesh bounds.
pub fn find_voxels(mesh: &Mesh, points: &[Point]) -> Vec<Option<Voxel>> {
    let mut voxels: Vec<Option<Voxel>> = Vec::with_capacity(points.len());
    for p in points {
        voxels.push(find_voxel(mesh, p));
    }
    voxels
}

/// Convert a point to the specified geometry
///
/// If already the correct type this changes nothing.
fn convert_point(mesh_type: Geometry, point: &Point) -> Point {
    if mesh_type == Geometry::Rectangular && point.coordinate_type == Geometry::Cylindrical {
        trace!("Converting {point} to xyz");
        point.clone().as_xyz()
    } else if mesh_type == Geometry::Cylindrical && point.coordinate_type == Geometry::Rectangular {
        trace!("Converting {point} to rzt");
        point.clone().as_rzt()
    } else {
        point.clone()
    }
}

/// Checks if [Point] coordinate and groups are all within the mesh bounds
///
/// Points exactly on the boundaries are considered within the mesh. It is
/// assumed that the point has the correct coordinate system.
pub fn is_point_valid(mesh: &Mesh, point: &Point) -> bool {
    // for coordinate types need to check all the mesh boundaries
    if !is_coordinate_valid(mesh, point) {
        return false;
    }

    // Make sure the energy group is valid
    if let Group::Value(e) = point.e {
        if &e < vec_f64_min(&mesh.emesh) || &e > vec_f64_max(&mesh.emesh) {
            return false;
        }
    }

    // Make sure the energy group is valid
    if let Group::Value(t) = point.t {
        if &t < vec_f64_min(&mesh.tmesh) || &t > vec_f64_max(&mesh.tmesh) {
            return false;
        }
    }

    // should be good
    true
}

/// Checks if (i,j,k) coordinate is within the mesh bounds
///
/// Points exactly on the boundaries are considered within the mesh. It is
/// assumed that the point has the correct coordinate system.
pub fn is_coordinate_valid(mesh: &Mesh, point: &Point) -> bool {
    // for coordinate types need to check all the mesh boundaries
    &point.i >= vec_f64_min(&mesh.imesh)
        && &point.i <= vec_f64_max(&mesh.imesh)
        && &point.j >= vec_f64_min(&mesh.jmesh)
        && &point.j <= vec_f64_max(&mesh.jmesh)
        && &point.k >= vec_f64_min(&mesh.kmesh)
        && &point.k <= vec_f64_max(&mesh.kmesh)
}

/// Find the corresponding global voxel index for a [Point]
///
/// The [Point] `coordinate_type` should match the [Mesh] geometry type.
///
/// This could easily be converted in the background, but may not be the
/// best approach as behaviour will be hidden. It therefore currently throws an
/// error for differing types to be transparent.
pub fn point_to_voxel_index(mesh: &Mesh, point: &Point) -> Result<usize> {
    // check to see if the coordinate types match
    if mesh.geometry != point.coordinate_type {
        return Err(anyhow!("Point and Mesh have different coordinate systems"));
    }

    // check to see if the coordinates are even inside the mesh
    if !is_point_valid(mesh, point) {
        return Err(anyhow!("Point {point} is not valid"));
    }

    // find indicies of groups bins
    let e_idx: usize = mesh.find_energy_group_index(point.e)?;
    let t_idx: usize = mesh.find_time_group_index(point.t)?;

    // find the index using the gemetry bounds
    let i_idx: usize = Mesh::find_bin_exclusive(point.i, &mesh.imesh)?;
    let j_idx: usize = Mesh::find_bin_exclusive(point.j, &mesh.jmesh)?;
    let k_idx: usize = Mesh::find_bin_exclusive(point.k, &mesh.kmesh)?;

    Ok(mesh.etijk_to_voxel_index(e_idx, t_idx, i_idx, j_idx, k_idx))
}

/// Convert tuple of (r,z,t) to cartesian (x,y,z)
pub fn convert_rzt_to_xyz(r: f64, z: f64, t: f64) -> (f64, f64, f64) {
    (r * t.cos(), r * t.sin(), z)
}

/// Convert tuple of (x,y,z) to cylindrical (r,z,t)
pub fn convert_xyz_to_rzt(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // convert to 0-360 range, TAU = 2*PI
    let mut t = y.atan2(x);
    t = if t.is_sign_negative() {
        std::f64::consts::TAU + t
    } else {
        t
    };

    (x.hypot(y), z, t)
}

/// Representation of a point in the context of a [Mesh]
///
/// Often the energy/time groups can be left as `Total` and operating on
/// rectangular meshes.
///
/// For convenience, there are several methods for initialising points without
/// having to define these defaults. For example:
///
/// ```rust
/// # use meshtal::point::Point;
/// # use meshtal::mesh::{Geometry, Group};
/// assert_eq!(
///         Point::from_xyz(1.0, 2.0, 3.0),
///         Point {
///             e: Group::Total,
///             t: Group::Total,
///             i: 1.0,
///             j: 2.0,
///             k: 3.0,
///             coordinate_type: Geometry::Rectangular
///         }
///     );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Energy group
    pub e: Group,
    /// Time group
    pub t: Group,
    /// i coordinate
    pub i: f64,
    /// j coordinate
    pub j: f64,
    /// k coordinate
    pub k: f64,
    /// Coordiante system
    pub coordinate_type: Geometry,
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = match self.e {
            Group::Value(e) => f!("({:>13}, ", e.sci(5, 2)),
            Group::Total => f!("({:>13}, ", "Total"),
        };

        s += &match self.t {
            Group::Value(t) => f!("{:>13}, ", t.sci(5, 2)),
            Group::Total => f!("{:>13}, ", "Total"),
        };

        s += &f!(
            "{:>13},{:>13},{:>13})",
            self.i.sci(5, 2),
            self.j.sci(5, 2),
            self.k.sci(5, 2)
        );

        write!(f, "{}", s)
    }
}

impl Default for Point {
    fn default() -> Self {
        Self {
            e: Group::Total,
            t: Group::Total,
            i: 0.0,
            j: 0.0,
            k: 0.0,
            coordinate_type: Geometry::Rectangular,
        }
    }
}

impl Point {
    /// Make new point with default values
    ///
    /// Same as default(), with Total energy/time groups for a point at (0,0,0)
    /// in a recatangualr mesh.
    pub fn new() -> Self {
        Default::default()
    }

    /// Make new point from x, y, z coordinates
    ///
    /// Defaults to the Total energy/time groups for a rectangular coordiante
    /// system.
    ///
    /// To make a new point with specific energy or time groups just create an
    /// instance of the struct directly.
    pub fn from_xyz(x: f64, y: f64, z: f64) -> Self {
        Self {
            i: x,
            j: y,
            k: z,
            ..Default::default()
        }
    }

    /// Make new point from r, z, t coordinates
    ///
    /// Defaults to the Total energy/time groups for a cylindrical coordiante
    /// system.
    ///
    /// To make a new point with specific energy or time groups just create an
    /// instance of the struct directly.
    pub fn from_rzt(r: f64, z: f64, t: f64) -> Self {
        Self {
            i: r,
            j: z,
            k: t,
            coordinate_type: Geometry::Cylindrical,
            ..Default::default()
        }
    }

    /// Initialise new Point from a vector of x, y, z coordinates
    ///
    /// In the order of (i, j, k). Energy/time groups and coordinate type are
    /// set to the 'Total', and 'Geometry::Rectangular'.
    pub fn from_xyz_vec(values: &[f64]) -> Result<Self> {
        match values.len() {
            3 => Ok(Point {
                i: values[0],
                j: values[1],
                k: values[2],
                ..Default::default()
            }),
            _ => Err(anyhow!("Vector must contain 3 values as vec[i, j, k]")),
        }
    }

    /// Initialise new Point from a vector of r, z, t coordinates
    ///
    /// In the order of (i, j, k). Energy/time groups and coordinate type are
    /// set to the 'Total', and 'Geometry::Cylindrical'.
    pub fn from_rzt_vec(values: &[f64]) -> Result<Self> {
        match values.len() {
            3 => Ok(Point {
                i: values[0],
                j: values[1],
                k: values[2],
                coordinate_type: Geometry::Cylindrical,
                ..Default::default()
            }),
            _ => Err(anyhow!("Vector must contain 3 values as vec[i, j, k]")),
        }
    }

    /// Convert the point values to xyz coordinates
    ///
    /// If already the correct coordinate system, nothing will change. Anything
    /// else is converted as appropriate, modifying coordiantes and updating
    /// `coordinate_type` to [Geometry::Rectangular].
    pub fn as_xyz(self) -> Self {
        match self.coordinate_type {
            Geometry::Rectangular => self,
            Geometry::Cylindrical => {
                let (x, y, z) = convert_rzt_to_xyz(self.i, self.j, self.k);
                Self {
                    e: self.e,
                    t: self.t,
                    i: x,
                    j: y,
                    k: z,
                    coordinate_type: Geometry::Rectangular,
                }
            }
        }
    }

    /// Convert the point values to rzt coordinates
    ///
    /// If already the correct coordinate system, nothing will change. Anything
    /// else is converted as appropriate, modifying coordiantes and updating
    /// `coordinate_type` to [Geometry::Cylindrical].
    pub fn as_rzt(self) -> Self {
        match self.coordinate_type {
            Geometry::Cylindrical => self,
            Geometry::Rectangular => {
                let (r, z, theta) = convert_xyz_to_rzt(self.i, self.j, self.k);
                Self {
                    e: self.e,
                    t: self.t,
                    i: r,
                    j: z,
                    k: theta,
                    coordinate_type: Geometry::Cylindrical,
                }
            }
        }
    }
}
