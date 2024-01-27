//! Weight window mesh tools
//!
//! Implementation of global weight window meshes. The module contains tools to
//! convert from any mesh to the standardised WWINP/WWOUT/WWONE file for direct
//! use in MCNP.
//!
//! # Details
//!
//! The generation of weights implemented using the MAGIC method for global
//! mesh-based weight windows. Weights are calculated as
//! `(0.5 * (flux / reference_flux)).powf(power)`.
//!
//! The only input required from the user is:
//! - `power` for the de-tuning factor(s)
//!
//! In the current version, the flux ref is taken as the maximum flux in the
//! group. Allowing the user to specify a location or user-defined reference
//! for scaling is on the todo list.
//!
//! Of course, more advanced usage allows for rescaling weights, only using the
//! 'total' energy/time group of multi-group meshes, and specifying individual
//! `power` and `max_error` factors on a group-by-group basis.
//!
//! # Example
//!
//! Simple case of reading in a single mesh tally and writing to a WWOUT file.
//!
//! ```rust
//! # use meshtal::mesh::read_meshtal_target;
//! # use meshtal::weights::mesh_to_ww;
//! // Read tally 104 from a meshtal file
//! let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
//!
//! // Convert the mesh into a mesh-based weight window input file
//! let weight_window = mesh_to_ww(&mesh, 0.7, 0.10, false);
//!
//! // Modify the weights, write the file, etc...
//! // weight_window.scale(1.20);
//! // weight_window.write("wwout");
//! ```
//!
//! For multi-particle weight windows, the weight windows may be combined into a
//! single file. The only limitation is that all meshes must have the same
//! geometry.
//!
//! ```rust
//! # use meshtal::mesh::read_meshtal_target;
//! # use meshtal::weights::{mesh_to_ww, write_multi_particle, write_single_particle};
//! // Read multiple mesh tallies
//! let neutron_mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
//! let photon_mesh = read_meshtal_target("./data/meshes/fmesh_204.msht", 204).unwrap();
//!
//! // Convert the meshes into weight windows
//! let ww_set = [mesh_to_ww(&photon_mesh, 0.5, 1.00, false),
//!               mesh_to_ww(&neutron_mesh, 0.6, 0.98, false)];
//!
//! // Write individual wwout files
//! // write_single_particle(&ww_set[0], "photon.ww");
//! // write_single_particle(&ww_set[1], "neutron.ww");
//!
//! // Write the combined NP wwout file
//! // write_multi_particle(&ww_set, "test_multi.ww");
//! ```
//!

// standard library
use std::fs::File;
use std::io::{BufWriter, Write};

// internal modules
use crate::mesh::{Geometry, Group, Mesh, Particle, Voxel};
// use crate::utils::FloatFmt;
use crate::utils::*;

// external crates
use log::{debug, trace, warn};

/// Mesh tally to global weight windows with simple parameters
///
/// A constant power factor and error tolerance are applied to all energy/time
/// groups.
///
/// - `powers` - Softening factor used as ww=>ww^power
/// - `max_errors` - Errors above this are set to 0/analogue
/// - `total_only` - Only generate weights from [Group::Total](crate::mesh::Group)
///
/// Weights are calculated as `(0.5 * (v.result / flux_ref)).powf(power)`. For
/// example, applying a 0.7 de-tuning factor and setting voxels with errors
/// below 10% to analogue:
///
/// ```rust
/// # use meshtal::mesh::read_meshtal_target;
/// # use meshtal::weights::mesh_to_ww;
/// // Read tally 104 from a meshtal file
/// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
/// // Convert the mesh into a weight window set
/// let weight_window = mesh_to_ww(&mesh, 0.7, 0.10, false);
/// ```
///
/// By default, this generates weight windows for all time and energy groups.
/// To generate a simpler set of weight windows based only on the
/// [Group::Total](crate::mesh::Group), set the `total_only` boolean to `true`.
pub fn mesh_to_ww(mesh: &Mesh, power: f64, max_error: f64, total_only: bool) -> WeightWindow {
    let mut ww: WeightWindow = initialise_ww_from_mesh(mesh, total_only);
    ww.weights = compute_weights(mesh, &[power], &[max_error], total_only);
    ww
}

/// Mesh tally to global weight windows with fine de-tuning and errors
///
/// Same as [mesh_to_ww] but allows for individual de-tuning factors and error
/// tolerances for each group. If `powers` or `max_errors` have a single entry
/// this will be applied to all groups.
///
/// - `powers` - Softening factor used as ww=>ww^power
/// - `max_errors` - Errors above this are set to 0/analogue
///
/// A call to this may look like this, applying separate powers and errors to
/// a mesh with 3 energy groups:
///
/// ```rust
/// # use meshtal::mesh::read_meshtal_target;
/// # use meshtal::weights::mesh_to_ww_advanced;
/// // Read tally 104 from a meshtal file
/// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
/// // Convert the mesh into a set of weight windows, using different parameters per set
/// let ww = mesh_to_ww_advanced(&mesh,
///                              &[0.7, 0.5, 0.85],
///                              &[0.1, 0.1, 0.15]);
/// ```
/// The lists should be ordered such that they match the following nested order:
/// ```ignore
/// for energy in energy_groups {
///     for time in time_groups {
///         calculate weights...
///     }
/// }
/// ```
/// For example, the following energy and time groups are related to the groups
/// shown explicitly below.
/// ```text
/// Energy bin boundaries: 0.0 10.0 200.0
/// Time bin boundaries  : -1E+36 0.0 1E+16 1E+99
/// ```
///
/// ```text      
/// 0 -> Energy(10.0)   Time(0.0)       powers[0]   max_errors[0]
/// 1 -> Energy(10.0)   Time(1E+16)     powers[1]   max_errors[1]
/// 2 -> Energy(10.0)   Time(1E+99)     powers[2]   max_errors[2]
/// 3 -> Energy(200.0)  Time(0.0)       powers[3]   max_errors[3]
/// 4 -> Energy(200.0)  Time(1E+16)     powers[4]   max_errors[4]
/// 5 -> Energy(200.0)  Time(1E+99)     powers[5]   max_errors[5]
/// ```
pub fn mesh_to_ww_advanced(mesh: &Mesh, powers: &[f64], max_errors: &[f64]) -> WeightWindow {
    let mut ww: WeightWindow = initialise_ww_from_mesh(mesh, false);
    ww.weights = compute_weights(mesh, powers, max_errors, false);
    ww
}

/// Convenience function for writing a [WeightWindow] into a single wwout file
///
/// A [WeightWindow] instance corresponds to a full set of weight windows for a
/// single particle type. These can be combined into a multi-particle wwout
/// using the [write_multi_particle()] function.
///
/// ```rust, ignore
/// # use meshtal::mesh::read_meshtal_target;
/// # use meshtal::weights::{write_single_particle, mesh_to_ww};
/// // Read tally 104 from a meshtal file
/// let photon_mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
/// // Convert the mesh into a weight window set
/// let ww = mesh_to_ww(&photon_mesh, 0.5, 1.0, false);
/// write_single_particle(ww, "test.ww");
/// ```
pub fn write_single_particle(weight_window: &WeightWindow, output: &str) {
    weight_window.write(output);
}

/// Combine multiple weight window sets into a single wwout file
///
/// A [WeightWindow] instance corresponds to a full set of weight windows for a
/// single particle type. This can be written simply using the
/// [write_single_particle()] function or calling
/// [write()](crate::weights::WeightWindow::write) directly.
///
/// This function attempts to combine a list of weight windows into a single
/// wwout file for multiple particles.
///
/// Note that:
/// - Weight window sets are re-sorted by particle type
/// - Duplicate particle types are removed
/// - Sets with inconsistent geometries are removed
///
/// The remaining weight window sets that can be combined will be written to
/// the path provided as `output`.
///
/// ```rust, ignore
/// # use meshtal::mesh::read_meshtal_target;
/// # use meshtal::weights::{write_multi_particle, mesh_to_ww};
/// // Read tally 104 from a meshtal file
/// let photon_mesh = read_meshtal_target("./data/meshes/fmesh_204.msht", 204).unwrap();
/// let neutron_mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
/// // Convert the mesh into a weight window set
/// let ww_sets = [mesh_to_ww(&photon_mesh, 0.5, 1.0, false),
///                mesh_to_ww(&neutron_mesh, 0.5, 1.0, false)];
/// let weight_window = write_multi_particle(&ww_sets, "test_multi.ww", false);
/// ```
pub fn write_multi_particle(weight_windows: &[WeightWindow], output: &str, padded: bool) {
    let ww_list = preprocess_set(weight_windows);

    // assume fine >2 meshes for now
    let f = File::create(output).expect("Unable to create file");
    let mut f = BufWriter::new(f);

    debug!(" - writing block 1");
    f.write_all(combined_header(&ww_list, padded).as_bytes())
        .unwrap();
    f.write_all(ww_list[0].block_1().as_bytes()).unwrap();

    debug!(" - writing block 2");
    f.write_all(ww_list[0].block_2().as_bytes()).unwrap();

    debug!(" - writing block 3");
    for ww in ww_list {
        f.write_all(ww.block_3().as_bytes()).unwrap();
    }
}

/// Mesh-based global weight window data for WWINP/WWOUT/WWONE
///
/// The [WeightWindow] data structure represents a set of weight windows for a
/// single mesh, and therefore a single particle type. Tools to convert from
/// meshtal tallies and then combine several sets for any combination of
/// particle types are provided.
///
/// Writers are implemented to generate the standardised ASCII file format for
/// WWINP, WWOUT, and WWONE files for direct use in MCNP simulations.
///
/// Large weight window meshes are approximately ~8 bytes per unique voxel.
///
/// Note: while unreadable, the [WeightWindow] fields correspond closely to
/// the internal FORTRAN MCNP variables outlined in the Appendicies of the
/// [MCNPv6.2](https://mcnp.lanl.gov/pdf_files/TechReport_2017_LANL_LA-UR-17-29981_WernerArmstrongEtAl.pdf)
/// and [MCNPv6.3](https://mcnpx.lanl.gov/pdf_files/TechReport_2022_LANL_LA-UR-22-30006Rev.1_KuleszaAdamsEtAl.pdf)
/// user manuals. This helps maintain consistency with the underlying data for
/// the developer.
#[derive(Debug, Clone)]
pub struct WeightWindow {
    // Basic header info
    /// File type, manual states unused, so always 1.
    pub f: u8,
    /// Time-dependent windows flag, 1=no 2=yes.
    pub iv: u8,
    /// Number of particle types
    pub ni: u8,
    /// Number of energy bins for each particle type
    pub ne: usize,
    /// Number of time bins for each particle type
    pub nt: usize,
    /// Number of 'words' to describe mesh. rec=10, cyl/sph=16
    pub nr: u8,
    /// Mesh type 1=rec, 2=cyl, 3=sph
    pub nwg: Geometry,
    /// Problem description, 19-char string, can be blank.
    pub probid: String,

    // Fine mesh
    /// Total number of fine mesh points in i
    pub nfx: usize,
    /// Total number of fine mesh points in j
    pub nfy: usize,
    /// Total number of fine mesh points in k
    pub nfz: usize,

    // Coarse mesh
    /// Total number of coarse mesh points in i
    pub ncx: usize,
    /// Total number of coarse mesh points in j
    pub ncy: usize,
    /// Total number of coarse mesh points in k
    pub ncz: usize,

    // [ORIGIN] Corner of (x,y,z) rec, bottom center of (r,z,t) cyl, or center of (r,p,t) sph
    /// Origin i coordinate
    pub x0: f64,
    /// Origin j coordinate
    pub y0: f64,
    /// Origin k coordinate
    pub z0: f64,

    // [AXS] Vector from x0 y0 z0 to x1 y1 z1 defines (r,z,t) cylinder, or (r,p,t) polar axis
    /// Axis i coordinate
    pub x1: f64,
    /// Axis j coordinate
    pub y1: f64,
    /// Axis k coordinate
    pub z1: f64,

    // [VEC] Vector from x0 y0 z0 to x2 y2 z2 defines (r,z,t) cylinder, or (r,p,t) azimuthal axis
    /// Vec i coordinate
    pub x2: f64,
    /// Vec j coordinate
    pub y2: f64,
    /// Vec k coordinate
    pub z2: f64,

    // Energy and time bin boundaries
    /// Upper energy bounds for each particle type
    pub e: Vec<f64>,
    /// Upper time bounds for each particle type if nt(i)>1
    pub t: Vec<f64>,

    // Block 2 nonsense:
    // q = Fine mesh ratio (1 always) in each coarse mesh
    // p = Coarse mesh coordinates for (x,y,z), (r,z,t), or (r,p,t)
    // s = Number of fine meshes in each coarse mesh for (x,y,z), (r,z,t), or (r,p,t)
    /// List of (qx(i), px(i), sx(i)) tuples for i=1,ncx
    pub qps_x: Vec<[f64; 3]>,
    /// List of (qy(i), py(i), sy(i)) tuples for i=1,ncy
    pub qps_y: Vec<[f64; 3]>,
    /// List of (qz(i), pz(i), sz(i)) tuples for i=1,ncz
    pub qps_z: Vec<[f64; 3]>,

    // Actual weights
    /// Flattened vector of lower weights for each voxel, for each energy bin,
    /// for each time bin
    pub weights: Vec<f64>,

    /// Retain particle type for use in multi-particle sets
    pub particle: Particle,
}

// Public API
impl WeightWindow {
    /// Output the weight window to the standard ASCII file
    ///
    /// Generates the standard ASCII WWINP/WWOUT/WWONE formatted files for
    /// direct input to MCNP simulations.
    ///
    /// Tools to combine weight window sets for multiple particles are provided
    /// as part of the module (See [write_multi_particle()]).
    pub fn write(&self, path: &str) {
        // assume fine >2 meshes for now
        let f = File::create(path).expect("Unable to create file");
        let mut f = BufWriter::new(f);
        f.write_all(self.block_1_header().as_bytes()).unwrap();
        f.write_all(self.block_1().as_bytes()).unwrap();
        f.write_all(self.block_2().as_bytes()).unwrap();
        f.write_all(self.block_3().as_bytes()).unwrap();
    }

    /// Multiply all weights by a constant factor
    ///
    /// This is just a blanket multiplication that applies to every weight
    /// across all energy/time groups and particle types.
    ///
    /// ```rust
    /// # use meshtal::weights::WeightWindow;
    /// let mut wwout = WeightWindow {
    ///     weights: vec![0.2, 0.15, 0.4],
    ///     ..Default::default()
    /// };
    ///
    /// // Double every weight
    /// wwout.scale(2.0);
    ///
    /// assert_eq!(wwout.weights, vec![0.4, 0.3, 0.8])
    /// ```
    pub fn scale(&mut self, factor: f64) {
        self.weights = self
            .weights
            .iter_mut()
            .map(|w| *w * factor)
            .collect::<Vec<f64>>();
    }

    /// Calculate number of non-zero weights
    ///
    /// Useful common sense value for checking the conversion to weights and
    /// that successive weight windows are imporving. Do not expect it to
    /// reach 100% if the mesh geometry covers any areas of zero importance for
    /// a given particle type.
    ///
    /// ```rust
    /// # use meshtal::weights::WeightWindow;
    /// let wwout = WeightWindow {
    ///     weights: vec![0.2, 0.15, 0.0, 0.0],
    ///     ..Default::default()
    /// };
    ///
    /// assert_eq!(wwout.non_analogue_percentage(), 50.0)
    /// ```
    pub fn non_analogue_percentage(&self) -> f64 {
        let non_zero = self.weights.iter().filter(|&v| *v != 0.0).count();
        debug!("Non-zero weights: {}", non_zero);
        100.0 * (non_zero as f64) / (self.weights.len() as f64)
    }

    /// Generate file content as a string (not for large files)
    ///
    /// Build a string for the full wwout file. Can be useful for small files
    /// and quick checks. However, this can end up duplicating a lot of data and
    /// the memory usage could be large for a significant number of weights.
    pub fn file_content(&self) -> String {
        let mut s = self.block_1_header();
        s += &self.block_1();
        s += &self.block_2();
        s += &self.block_3();
        s
    }
}

/// ASCII file formatting in blocks for consistency with the specifications
///
/// All formats will match fortran formats in the table below, using the
/// scientific number representation for any 'g' format.
///
/// | Block | Format         | Variable List                                    |
/// | ----- | -------------- | ------------------------------------------------ |
/// | 1     | 4i10, 20x, a19 | if iv ni nr probid                               |
/// | 1     | 7i10           | nt(1) ... nt(ni) [if iv=2]                       |
/// | 1     | 7i10           | ne(1) ... ne(ni)                                 |
/// | 1     | 6g13.5         | nfx nfy nfz x0 y0 z0                             |
/// | 1     | 6g13.5         | ncx ncy ncz nwg [if nr=10]                       |
/// | 1     | 6g13.5         | ncx ncy ncz x1 y1 z1 [if nr=16]                  |
/// | 1     | 6g13.5         | x2 y2 z2 nwg [if nr=16]                          |
/// | 2     | 6g13.5         | x0 (qx(i), px(i), sx(i), i=1,ncx)                |
/// | 2     | 6g13.5         | y0 (qy(i), py(i), sy(i), i=1,ncy)                |
/// | 2     | 6g13.5         | z0 (qz(i), pz(i), sz(i), i=1,ncz)                |
/// | 3     | 6g13.5         | t(i,1) ... t(i,nt(i)) [if nt(i)>1]               |
/// | 3     | 6g13.5         | e(i,1) ... e(i,ne(i))                            |
/// | 3     | 6g13.5         | (((w(i,j,k,l,1) j=1,nft), k=1,ne(i)), l=1,nt(i)) |
///
/// More information can be found in the Appedicies of the MCNP user manuals.
impl WeightWindow {
    /// Only the header, which needs to be seperate combining particles
    pub fn block_1_header(&self) -> String {
        // if iv ni nr probid
        let mut s = f!(
            "{:>10}{:>10}{:>10}{:>10}",
            self.f,
            self.iv,
            self.ni,
            self.nr,
        );

        // a19 so no longer than 19 characters
        let mut comment = self.probid.clone();
        comment.truncate(19);
        s += &f!("{comment}\n");

        // nt(1) ... nt(ni) [if iv=2]
        if self.iv == 2 {
            s += &f!("{:>10}\n", &self.nt);
        }

        // ne(1) ... ne(ni)
        s += &f!("{:>10}\n", &self.ne);
        s
    }

    /// Block 1 data that are common to all combined window sets
    pub fn block_1(&self) -> String {
        // nfx nfy nfz x0 y0 z0
        let mut s = f!(
            "{:>13}{:>13}{:>13}",
            self.nfx.sci(5, 2),
            self.nfy.sci(5, 2),
            self.nfz.sci(5, 2)
        );
        s += &f!(
            "{:>13}{:>13}{:>13}\n",
            self.x0.sci(5, 2),
            self.y0.sci(5, 2),
            self.z0.sci(5, 2)
        );

        // ncx ncy ncz nwg          [if nr=10]
        // --------------- OR ----------------
        // ncx ncy ncz x1 y1 z1     [if nr=16]
        // x2 y2 z2 nwg             [if nr=16]
        s += &f!(
            "{:>13}{:>13}{:>13}",
            self.ncx.sci(5, 2),
            self.ncy.sci(5, 2),
            self.ncz.sci(5, 2)
        );

        match self.nwg {
            Geometry::Rectangular => s += &f!("{:>13}", (self.nwg as u8).sci(5, 2)),
            _ => {
                s += &f!(
                    "{:>13}{:>13}{:>13}\n",
                    self.x1.sci(5, 2),
                    self.y1.sci(5, 2),
                    self.z1.sci(5, 2)
                );
                s += &f!(
                    "{:>13}{:>13}{:>13}{:>13}",
                    self.x2.sci(5, 2),
                    self.y2.sci(5, 2),
                    self.z2.sci(5, 2),
                    (self.nwg as u8).sci(5, 2)
                )
            }
        }

        s += "\n";
        s
    }

    /// Block 2 containing all the coarse mesh bounds
    pub fn block_2(&self) -> String {
        // x0 ( qx(i), px(i), sx(i) ) from i=1,ncx
        let mut s = f!("{:>13}", self.x0.sci(5, 2));
        let mut count: u8 = 1;
        for x in &self.qps_x {
            s += &f!(
                "{:>13}{:>13}{}{:>13}",
                x[0].sci(5, 2),
                x[1].sci(5, 2),
                track_newlines(&mut count, 2),
                x[2].sci(5, 2)
            );
        }

        // y0 ( qy(i), py(i), sy(i) ) from i=1,ncy
        count = 1;
        s += &f!("\n{:>13}", self.y0.sci(5, 2));
        for y in &self.qps_y {
            s += &f!(
                "{:>13}{:>13}{}{:>13}",
                y[0].sci(5, 2),
                y[1].sci(5, 2),
                track_newlines(&mut count, 2),
                y[2].sci(5, 2)
            );
        }

        // z0 ( qz(i), pz(i), sz(i) ) from i=1,ncz
        count = 1;
        s += &f!("\n{:>13}", self.z0.sci(5, 2));
        for z in &self.qps_z {
            s += &f!(
                "{:>13}{:>13}{}{:>13}",
                z[0].sci(5, 2),
                z[1].sci(5, 2),
                track_newlines(&mut count, 2),
                z[2].sci(5, 2)
            );
        }

        // don't add a newline if it happens to end in one already
        if !s.ends_with('\n') {
            s += "\n";
        }
        s
    }

    /// Block 3 data are all the weight window sets for each voxel group
    pub fn block_3(&self) -> String {
        let mut s = String::new();
        let mut count: u8 = 1;

        // t(i,1) ... t(i,nt(i)) [if nt(i)>1]
        if self.t.len() > 1 {
            for t in &self.t {
                s += &f!("{:>13}", t.sci(5, 2));
                s += &track_newlines(&mut count, 6);
            }
            s += "\n";
        }

        // e(i,1) ... e(i,ne(i))
        count = 1;
        for e in &self.e {
            s += &f!("{:>13}", e.sci(5, 2));
            s += &track_newlines(&mut count, 6);
        }
        s += "\n";

        // w(i,j,k,l,1) where j=1,nft k=1,ne(i) l=1,nt(i)
        // textwrap extremely slow on large lines, just split as we go
        count = 1;
        for w in &self.weights {
            s += &f!("{:>13}", w.sci(5, 2));
            s += &track_newlines(&mut count, 6);
        }

        // don't add a newline if it happens to end in one already
        if !s.ends_with('\n') {
            s += "\n";
        }
        s
    }
}

impl Default for WeightWindow {
    fn default() -> Self {
        Self {
            f: 1,
            iv: 1,
            ni: 1,
            ne: 1,
            nt: 1,
            nr: 10,
            nwg: Geometry::Rectangular,
            probid: String::new(),
            nfx: 0,
            nfy: 0,
            nfz: 0,
            ncx: 0,
            ncy: 0,
            ncz: 0,
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            x1: 0.0,
            y1: 0.0,
            z1: 1.0,
            x2: 1.0,
            y2: 0.0,
            z2: 0.0,
            e: Vec::new(),
            t: Vec::new(),
            qps_x: Vec::new(),
            qps_y: Vec::new(),
            qps_z: Vec::new(),
            weights: Vec::new(),
            particle: Particle::Unknown,
        }
    }
}

/// Core function for setting up the weight mesh geometry
///
/// This initialises everything but the weights themselves, setting up all
/// geometry bounds and required parameters taken or inferred from the provided
/// [Mesh](crate::mesh::Mesh).
///
/// This is decoupled from the weights as it can be useful to just be able to
/// do the setup and weight calculations separately. However, the public API
/// brings these together to ensure they are used correctly.
fn initialise_ww_from_mesh(mesh: &Mesh, total_only: bool) -> WeightWindow {
    trace!("Initialising Wwinp from mesh");
    // for what this shit means look up appendix B of the mcnp6 manual
    let mut ww = WeightWindow {
        nr: match mesh.geometry {
            Geometry::Rectangular => 10,
            Geometry::Cylindrical => 16,
        },
        nwg: mesh.geometry,
        nfx: mesh.iints,
        nfy: mesh.jints,
        nfz: mesh.kints,
        ncx: mesh.iints,
        ncy: mesh.jints,
        ncz: mesh.kints,
        x0: mesh.origin[0],
        y0: mesh.origin[1],
        z0: mesh.origin[2],
        x1: mesh.axs[0],
        y1: mesh.axs[1],
        z1: mesh.axs[2],
        x2: mesh.vec[0],
        y2: mesh.vec[1],
        z2: mesh.vec[2],
        // total only will just be the maximum energy recorded I suppose
        e: if total_only {
            vec![*mesh.emesh.last().unwrap()]
        } else {
            mesh.emesh[1..].to_vec()
        },
        qps_x: qps_tuples(&mesh.imesh),
        qps_y: qps_tuples(&mesh.jmesh),
        qps_z: qps_tuples(&mesh.kmesh),
        particle: mesh.particle,
        ..Default::default()
    };

    // number of energy bins
    ww.ne = ww.e.len();

    // only bother including time info if relevant
    if mesh.tbins() > 1 && !total_only {
        ww.iv = 2;
        ww.nt = mesh.tbins();
        ww.t = mesh.tmesh[1..].to_vec();
    }

    ww
}

/// Core function for turning a flux mesh into weights
///
/// Processes each group in order because each has to be normalised to itself.
/// Because of this, it is easy to apply different power factors and error
/// tolerences for each group. The signature therefore takes lists for both
/// parameters.
///
/// For the typical functionality the `powers` and `max_errors` list may be just
/// one value long, which will be applied to every group.
fn compute_weights(mesh: &Mesh, powers: &[f64], max_errors: &[f64], total_only: bool) -> Vec<f64> {
    let (energy_groups, time_groups) = relevant_groups(mesh, total_only);

    // set up the weights vector
    let n_groups = energy_groups.len() * time_groups.len();
    let n_voxels = n_groups * mesh.iints * mesh.jints * mesh.kints;
    let mut weights: Vec<f64> = Vec::with_capacity(n_voxels);

    // collect the powers to be used for each group
    let powers = collect_power_values(powers, n_groups);
    let mut powers_iter = powers.iter();

    // collect the error tolerance to be used for each group
    let errors = collect_error_values(max_errors, n_groups);
    let mut errors_iter = errors.iter();

    // Loop over all the requested groups and apply the appropriate factors
    for energy in &energy_groups {
        for time in &time_groups {
            let voxels = mesh.slice_voxels_by_group(*energy, *time).unwrap();
            weights.extend(weight_from_voxels(
                mesh,
                voxels,
                *powers_iter.next().unwrap(),
                *errors_iter.next().unwrap(),
            ));
        }
    }

    weights
}

/// Calculates the weights for a set of voxels
///
/// This is done in groups so that each energy/time group can be normalised
/// properly. As a nice side effect this makes it very easy to have different
/// power factors and error tolerances for each group.
///
/// Weights are calculated as `(0.5 * (v.result / flux_ref)).powf(power)`
fn weight_from_voxels(mesh: &Mesh, voxels: &[Voxel], power: f64, max_error: f64) -> Vec<f64> {
    // find maximum of the energy/time group set
    let flux_ref = voxels
        .iter()
        .map(|v| v.result)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

    debug!("Reference flux: {}", flux_ref.sci(5, 2));

    // guard against groups with no results
    if flux_ref == 0.0 {
        return vec![0.0; voxels.len()];
    }

    // Main calculation, very simple
    let mut wgt: Vec<(usize, f64)> = Vec::with_capacity(voxels.len());
    for (i, v) in voxels.iter().enumerate() {
        let mut w = if v.error <= max_error {
            (0.5 * (v.result / flux_ref)).powf(power)
        } else {
            0.0
        };

        // ensure the value is reasonable (looking at you CuV)
        w = constrain_weights(w);
        wgt.push((mesh.voxel_index_to_cell_index(i), w));
    }

    wgt.sort_by(|a, b| a.0.cmp(&b.0));
    wgt.into_iter().map(|r| r.1).collect()
}

/// Generate a list of error cuts for every group in the weight window mesh
///
/// The power valuse are assumed to be in the correct order corresponding to the
/// following processing loops:
///
/// ```ignore
/// for energy in energy_groups
///     for time in time_groups
///         calculate weights...
/// ```
///
/// There are several cases that this handles.
///
/// - Multiple factors - apply each power to its respective group
/// - Single factor - apply the same power factor to evey group
/// - Empty list - default to 0.7
///
/// If the length of the multiple powers list is not valid, the first factor is
/// applied to all groups and a warning raised.
fn collect_power_values(powers: &[f64], n_groups: usize) -> Vec<f64> {
    let n_powers = powers.len();

    match n_powers {
        0 => {
            warn!("Warning: No power factor provided, defaulting to 0.7");
            [0.7].repeat(n_groups)
        }
        1 => powers.repeat(n_groups),
        _ => {
            if n_powers == n_groups {
                powers.to_vec()
            } else {
                warn!("Warning: Power factors != number of groups");
                warn!("  - Expected {}, found {}", n_groups, n_powers);
                warn!("  - Setting all groups to 0.7");
                [0.7].repeat(n_groups)
            }
        }
    }
}

/// Generate a list of power factors for every group in the weight window mesh
///
/// Like the powers, the energy tolerances are also assumed to be in the correct
/// order corresponding to the following processing loops:
///
/// ```ignore
/// for energy in energy_groups
///     for time in time_groups
///         calculate weights...
/// ```
///
/// There are several cases that this handles.
///
/// - Multiple errors - apply each error tolerance to its respective group
/// - Single error - apply the same error tolerance to evey group
/// - Empty list - default to 1.0 (100%)
///
/// If the length of the multiple errors list is not valid, the first tolerance
/// is applied to all groups and a warning raised.
fn collect_error_values(errors: &[f64], n_groups: usize) -> Vec<f64> {
    let n_errors = errors.len();

    match n_errors {
        0 => {
            warn!("Warning: No error tolerance provided, defaulting to 1.0");
            [1.0].repeat(n_groups)
        }
        1 => errors.repeat(n_groups),
        _ => {
            if n_errors == n_groups {
                errors.to_vec()
            } else {
                warn!("Warning: Error tolerences != number of groups");
                warn!("  - Expected {}, found {}", n_groups, n_errors);
                warn!("  - Setting all error cuts to 1.0");
                [1.0].repeat(n_groups)
            }
        }
    }
}

/// Builds required tuples for coarse mesh bounds
///
/// The "qps" variables are for the coarse mesh bounds:
/// - q = Fine mesh ratio (1 always) in each coarse mesh
/// - p = Coarse mesh coordinates for (x,y,z), (r,z,t), or (r,p,t)
/// - s = Number of fine meshes in each coarse mesh for (x,y,z), (r,z,t), or (r,p,t)
fn qps_tuples(mesh_bounds: &[f64]) -> Vec<[f64; 3]> {
    mesh_bounds[1..]
        .iter()
        .map(|bound| [1.0, *bound, 1.0])
        .collect()
}

/// Collect up the relevant energy and time groups to use for weights
///
/// Either just returns the `Total` for total-only selections, or will every
/// valued group if there are multiple.
fn relevant_groups(mesh: &Mesh, total_only: bool) -> (Vec<Group>, Vec<Group>) {
    match total_only {
        true => (vec![Group::Total], vec![Group::Total]),
        false => {
            // either total or the valued groups of emesh
            let energies = if mesh.ebins() > 1 {
                let groups = mesh.energy_groups();
                let (_, g) = groups.split_last().unwrap();
                g.to_vec()
            } else {
                vec![Group::Total]
            };

            // either total or the valued groups of tmesh
            let times = if mesh.tbins() > 1 {
                let groups = mesh.time_groups();
                let (_, g) = groups.split_last().unwrap();
                g.to_vec()
            } else {
                vec![Group::Total]
            };

            (energies, times)
        }
    }
}

/// Fix ridiculous values that may happen for CuV
fn constrain_weights(weight: f64) -> f64 {
    if weight < 1.0e-99 {
        // trace!("Weight {weight:.2e} < 1e-99, setting to analogue");
        0.0
    } else if weight >= 1.0e+100 {
        // trace!("Weight {weight:.2e} >= 1e+100, setting to 9.999E+99 to preserve formatting");
        9.999e99
    } else {
        weight
    }
}

/// Return a newline character once the counter reaches a target
///
/// Unfortunately textwrap is extremely slow on large lines, so more efficient
/// to just split as we go manually.
fn track_newlines(count: &mut u8, target: u8) -> &str {
    if *count == target {
        *count = 1;
        "\n"
    } else {
        *count += 1;
        ""
    }
}

/// Sort by particle type, remove duplicates, and ensure geometry match
fn preprocess_set(weight_windows: &[WeightWindow]) -> Vec<&WeightWindow> {
    let mut ww_list = weight_windows.iter().collect::<Vec<&WeightWindow>>();

    // Sort by particle type and get rid of any duplicates
    ww_list.sort_by_key(|&k| k.particle);
    ww_list.dedup_by_key(|k| k.particle);

    // Get rid of any that do not mach the mesh geometry
    let target = ww_list[0];
    ww_list.retain(|&ww| is_geometry_match(ww, target));
    ww_list.retain(|&ww| ww.particle != Particle::Unknown);
    ww_list
}

/// The wwout file forces the same geometry for every weight set
fn is_geometry_match(a: &WeightWindow, b: &WeightWindow) -> bool {
    if a.nr  != b.nr  // words (meshtype)
        || a.nfx != b.nfx
        || a.nfy != b.nfy
        || a.nfz != b.nfz
        || a.x0  != b.x0
        || a.y0  != b.y0
        || a.z0  != b.z0
        || a.x1  != b.x1
        || a.y1  != b.y1
        || a.z1  != b.z1
        || a.qps_x != b.qps_x
        || a.qps_y != b.qps_y
        || a.qps_z != b.qps_z
    {
        return false;
    }

    true
}

/// Generate the appropriate header values for a multi-particle file
///
/// Header fromat is outlined below. The number or particle types `ni`, flag for
/// time bins `iv`, and number of time `nt` and energy `ne` bins lists need to
/// be updated from the single particle case.
///
/// | Format            | Variables                 |
/// | ----------------- | ------------------------- |
/// | 4i10, 20x, a19    |  if iv ni nr probid       |
/// | 7i10              |  nt(1)...nt(ni) if iv=2   |
/// | 7i10              |  ne(1)...ne(ni)           |
///
/// The 7i suggests a maximum of 7 particle types per line, and this will need
/// zero padding to ensure all the correct particle types are used.
fn combined_header(weight_windows: &[&WeightWindow], padded: bool) -> String {
    let base = weight_windows[0];
    let iv = if weight_windows.iter().any(|ww| ww.iv == 2) {
        2
    } else {
        1
    };

    let (nt, ne) = match padded {
        true => particle_lists_padded(weight_windows),
        false => particle_lists_unpadded(weight_windows),
    };

    // if iv ni nr probid
    let mut s = f!("{:>10}{:>10}{:>10}{:>10}\n", base.f, iv, ne.len(), base.nr,);

    // nt(1) ... nt(ni) [if iv=2]
    let mut count: u8 = 1;
    if iv == 2 {
        for n_times in nt {
            s += &f!("{:>10}", n_times);
            s += &track_newlines(&mut count, 7); // 7i10 => split on 7
        }
        if !s.ends_with('\n') {
            s += "\n";
        }
    }

    // ne(1) ... ne(ni)
    count = 1;
    for n_energies in ne {
        s += &f!("{:>10}", n_energies);
        s += &track_newlines(&mut count, 7); // 7i10 => split on 7
    }

    if !s.ends_with('\n') {
        s += "\n";
    }

    s
}

/// List of number of particle time/energy groups
fn particle_lists_unpadded(weight_windows: &[&WeightWindow]) -> (Vec<usize>, Vec<usize>) {
    (
        weight_windows.iter().map(|ww| ww.nt).collect(),
        weight_windows.iter().map(|ww| ww.ne).collect(),
    )
}

/// List of number of particle time/energy groups, with 0 for missing types
fn particle_lists_padded(weight_windows: &[&WeightWindow]) -> (Vec<usize>, Vec<usize>) {
    let max = weight_windows
        .iter()
        .max_by_key(|ww| ww.particle)
        .unwrap()
        .particle
        .id() as usize;

    let mut nt = vec![0_usize; max];
    let mut ne = vec![0_usize; max];

    for ww in weight_windows {
        let idx = (ww.particle.id() - 1) as usize;
        nt[idx] = ww.nt;
        ne[idx] = ww.ne;
    }

    (nt, ne)
}
