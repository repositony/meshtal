// internal modules
use crate::mesh::particle::Particle;
use crate::mesh::voxel::{Group, Voxel, VoxelCoordinate};
use crate::utils::*;

// standard library
use core::fmt;

// external crates
use anyhow::{anyhow, Result};
use clap::ValueEnum;

/// Common struct representing a mesh tally
///
/// [Mesh] attributes correspond closely to MCNP input cards for consistency
/// and intuitive use. Units are unchanged from the MCNP defualts.
///
/// All output formats are stored as a [Mesh] to provide a common interface for
/// all post-processing operations, conversion to VTK formats, weight window
/// generation, etc...
///
/// ### Terminology
///
/// I, J, K generics
///
/// - Coordinate systems use different names i.e. (X,Y,Z) and (R,Z,Theta). The
/// generic (I, J, K) are used in keeping with MCNP user manuals.
///
/// Groups
///
/// - A full set of (I, J, K) voxels are repeated for every time and energy bin in
/// the mesh. It is possible to have additional 'Total' groups if the `EMESH` or
/// `TMESH` cards contain multiple bins. Time and energy bins are therefore
/// often mapped to [Group::Value] and [Group::Total] variants to be explicit
/// (See [Group] for details).
///
/// ### Examples
///
/// Basic reading of files is very simple, regardless of format.
///
/// ```rust
/// use meshtal::mesh::{read_meshtal, read_meshtal_target, Mesh};
///
/// // Extract all meshes from a file into a Vec<Mesh>
/// let mesh_list = read_meshtal("./data/meshes/fmesh_114.msht").unwrap();
///
/// // Extract just one target mesh from a file into a Mesh
/// let mesh = read_meshtal_target("./data/meshes/fmesh_114.msht", 114).unwrap();
/// ```
///
/// All the parsing and interpretation are done for you, and the data are in a
/// common [Mesh] type. This means ant any [Mesh] methods are available for any
/// format mesh of any geometry type.  
///
/// #### Using sets of Voxels
///
/// Useful operations are generally implemented. Those that do not require
/// knowledge of the mesh are associated functions. For example, finding the
/// maximum, minimum, and average of sets of voxels.
///
/// ```rust
/// # use meshtal::mesh::{Mesh, Voxel};
///  let voxels = vec![
///     Voxel {index: 0, result: 12.0, error: 0.1},
///     Voxel {index: 1, result: 18.0, error: 0.1},
///     Voxel {index: 2, result:  5.0, error: 0.1},
///     Voxel {index: 3, result: 23.0, error: 0.1},
/// ];
///
/// // Find the maximum of all voxels
/// assert_eq!(Mesh::maximum(&voxels), 23.0);
/// // Find the maximum of only the first three voxels
/// assert_eq!(Mesh::maximum(&voxels[0..3]), 18.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    /// Mesh tally number e.g fmesh104 => id = 104
    pub id: u32,
    /// Mesh geometry type, usually MeshType::RECTANGULAR for MCNP default
    pub geometry: Geometry,
    /// Name of the particle type
    pub particle: Particle,
    /// i mesh boundaries
    pub imesh: Vec<f64>,
    /// Number of voxels in i
    pub iints: usize,
    /// j mesh boundaries
    pub jmesh: Vec<f64>,
    /// Number of voxels in j
    pub jints: usize,
    /// k mesh boundaries
    pub kmesh: Vec<f64>,
    /// Number of voxels in j
    pub kints: usize,
    /// Energy bins
    pub emesh: Vec<f64>,
    /// Number of energy bins, EXCLUDING 'total' group
    pub eints: usize,
    /// Time bins \[shakes\]
    pub tmesh: Vec<f64>,
    /// Number of time bins, EXCLUDING 'total' group
    pub tints: usize,
    /// ORIGIN card, [0.0, 0.0, 0.0] for MCNP default
    pub origin: [f64; 3],
    /// AXS card, [0.0, 0.0, 1.0] for MCNP default
    pub axs: [f64; 3],
    /// VEC card, [1.0, 0.0, 0.0] for MCNP default
    pub vec: [f64; 3],
    ///  List of every `Voxel` in the mesh
    pub voxels: Vec<Voxel>,
    /// Detected output format in MESHTAL file
    pub format: Format,
}

impl Mesh {
    /// Multiply all voxel results by a constant factor
    ///
    /// Uncertanties are relative and are therfore unaffected.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh,Voxel};
    /// // Simple test mesh with three voxels
    /// let mut mesh = Mesh {
    ///     voxels: vec![
    ///         Voxel {result: 12.0, ..Default::default()},
    ///         Voxel {result: 18.0, ..Default::default()},
    ///         Voxel {result:  4.0, ..Default::default()},
    ///     ],
    ///     ..Default::default()
    /// };
    ///
    /// // Scale results by a factor of x0.5 (errors relative, thus unchanged)
    /// mesh.scale(0.5);
    /// assert_eq!(mesh.voxels[0].result, 6.0);
    /// assert_eq!(mesh.voxels[1].result, 9.0);
    /// assert_eq!(mesh.voxels[2].result, 2.0);
    /// ```
    pub fn scale(&mut self, factor: f64) {
        self.voxels.iter_mut().for_each(|v| v.result *= factor);
    }

    /// Translate all coordinates by (x, y, z)
    ///
    /// Simply updates the relevant origin coordiantes and mesh geometry bounds
    /// using the cartesian values provided. For cylindrical meshes the voxel
    /// bounds will be unaffected.
    pub fn translate(&mut self, x: f64, y: f64, z: f64) {
        // origin always moves for rec, cyl
        self.origin[0] += x;
        self.origin[1] += y;
        self.origin[2] += z;

        if self.geometry == Geometry::Rectangular {
            // all corrdinates and boundaries change for rectangular
            self.imesh = self.imesh.iter().map(|c| c + x).collect();
            self.jmesh = self.jmesh.iter().map(|c| c + y).collect();
            self.kmesh = self.kmesh.iter().map(|c| c + z).collect();
        }
    }

    /// Returns the number of energy bins
    ///
    /// This will include the `Total` bin in the count for tallies with
    /// multiple energy bins defined on the EMESH card.
    ///
    /// | Card values   | #Groups               |
    /// | ------------- | --------------------- |
    /// | None          | 1                     |
    /// | 0.0 100       | 1                     |
    /// | 0.0 20 100    | 3 (2 + 'Total' bin)   |
    /// | 0.0 20 100 T  | 3 (2 + 'Total' bin)   |
    pub fn ebins(&self) -> usize {
        if self.eints > 1 {
            self.eints + 1
        } else {
            1
        }
    }

    /// Returns slice of `emesh` for upper energy bin edges
    ///
    /// ```rust
    /// # use meshtal::mesh::Mesh;
    /// let mesh = Mesh {
    ///     emesh: vec![0.0, 1.0, 2.0, 3.0, 4.0],
    ///     ..Default::default()
    /// };
    /// // Get a slice of the upper edges of energy bins
    /// assert_eq!(mesh.energy_bins_upper(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn energy_bins_upper(&self) -> &[f64] {
        &self.emesh[1..]
    }

    /// Returns slice of `emesh` for lower energy bin edges
    ///
    /// ```rust
    /// # use meshtal::mesh::Mesh;
    /// let mesh = Mesh {
    ///     emesh: vec![0.0, 1.0, 2.0, 3.0, 4.0],
    ///     ..Default::default()
    /// };
    /// // Get a slice of the lower edges of energy bins
    /// assert_eq!(mesh.energy_bins_lower(), vec![0.0, 1.0, 2.0, 3.0]);
    /// ```
    pub fn energy_bins_lower(&self) -> &[f64] {
        let end = self.emesh.len() - 1;
        &self.emesh[0..end]
    }

    /// Returns a collection of all energy groups, including total
    ///
    /// Builds a full list of the energy groups in the mesh, which can include
    /// both [Group](crate::mesh::Group) variants.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh,Group};
    /// let mesh = Mesh {
    ///     eints: 2,
    ///     emesh: vec![0.0, 1.0, 2.0],
    ///     ..Default::default()
    /// };
    /// // See what energy groups the voxels may be split into
    /// assert_eq!(mesh.energy_groups(), vec![Group::Value(1.0), // bin 0.0 - 1.0
    ///                                       Group::Value(2.0), // bin 1.0 - 2.0
    ///                                       Group::Total]);
    /// ```
    pub fn energy_groups(&self) -> Vec<Group> {
        if self.ebins() > 1 {
            let mut groups = self
                .energy_bins_upper()
                .iter()
                .map(|energy| Group::Value(*energy))
                .collect::<Vec<Group>>();
            groups.push(Group::Total);
            groups
        } else {
            vec![Group::Total]
        }
    }

    /// Returns a collection of `emesh` Value() groups, ignoring any 'Total'
    ///
    /// Builds a list of only the energy groups with a value from `emesh`, and
    /// will only include the [Group::Value](crate::mesh::Group) variant.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh,Group};
    /// let mesh = Mesh {
    ///     eints: 2,
    ///     emesh: vec![0.0, 1.0, 2.0],
    ///     ..Default::default()
    /// };
    /// // List of emesh groups
    /// assert_eq!(mesh.emesh_groups(), vec![Group::Value(1.0),     // bin 0.0 - 1.0
    ///                                      Group::Value(2.0)]);   // bin 1.0 - 2.0
    /// ```
    pub fn emesh_groups(&self) -> Vec<Group> {
        self.energy_bins_upper()
            .iter()
            .map(|energy| Group::Value(*energy))
            .collect::<Vec<Group>>()
    }

    /// Returns the number of time bins
    ///
    /// This will include the 'Total' bin in the count for tallies with
    /// multiple time bins defined on the TMESH card. Defaults to 1 `Total` bin
    /// if none are explicitly defined.
    ///
    /// | Card values       | #Groups               |
    /// | ----------------- | --------------------- |
    /// | None              | 1                     |
    /// | 0.0 1e16          | 1                     |
    /// | 0.0 1e16 1e36     | 3 (2 + 'Total' bin)   |
    /// | 0.0 1e16 1e36 T   | 3 (2 + 'Total' bin)   |
    pub fn tbins(&self) -> usize {
        if self.tints > 1 {
            self.tints + 1
        } else {
            1
        }
    }

    /// Returns slice of `tmesh` for upper time bin edges
    ///
    /// ```rust
    /// # use meshtal::mesh::Mesh;
    /// let mesh = Mesh {
    ///     tmesh: vec![0.0, 1e12, 1e16, 1e20],
    ///     ..Default::default()
    /// };
    /// // Get a slice of the upper edges of time bins
    /// assert_eq!(mesh.time_bins_upper(), vec![1e12, 1e16, 1e20]);
    /// ```
    pub fn time_bins_upper(&self) -> &[f64] {
        if let Some((_, elements)) = self.tmesh.split_first() {
            elements
        } else {
            &self.tmesh
        }
    }

    /// Returns slice of `emesh` for lower time bin edges
    ///
    /// ```rust
    /// # use meshtal::mesh::Mesh;
    /// let mesh = Mesh {
    ///     tmesh: vec![0.0, 1e12, 1e16, 1e20],
    ///     ..Default::default()
    /// };
    /// // Get a slice of the lower edges of time bins
    /// assert_eq!(mesh.time_bins_lower(), vec![0.0, 1e12, 1e16]);
    /// ```
    pub fn time_bins_lower(&self) -> &[f64] {
        if let Some((_, elements)) = self.tmesh.split_last() {
            elements
        } else {
            &self.tmesh
        }
    }

    /// Returns a collection of all time groups, including total
    ///
    /// Builds a full list of the time groups in the mesh, which can include
    /// both [Group](crate::mesh::Group) variants.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh,Group};
    /// let mesh = Mesh {
    ///     tints: 2,
    ///     tmesh: vec![0.0, 1e12, 1e16],
    ///     ..Default::default()
    /// };
    /// // See what time groups the voxels may be split into
    /// assert_eq!(mesh.time_groups(), vec![Group::Value(1e12), // bin 0.0  - 1e12
    ///                                     Group::Value(1e16), // bin 1e12 - 1e16
    ///                                     Group::Total]);     // 'total' group
    /// ```
    pub fn time_groups(&self) -> Vec<Group> {
        if self.tbins() > 1 {
            let mut groups = self
                .time_bins_upper()
                .iter()
                .map(|time| Group::Value(*time))
                .collect::<Vec<Group>>();
            groups.push(Group::Total);
            groups
        } else {
            vec![Group::Total]
        }
    }

    /// Returns a collection of `tmesh` Value() groups, ignoring any 'Total'
    ///
    /// Builds a list of only the time groups with a value from `tmesh`, and
    /// will only include the [Group::Value](crate::mesh::Group) variant.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh,Group};
    /// let mesh = Mesh {
    ///     tints: 2,
    ///     tmesh: vec![0.0, 1e12, 1e16],
    ///     ..Default::default()
    /// };
    /// // List of tmesh groups
    /// assert_eq!(mesh.tmesh_groups(), vec![Group::Value(1e12),    // bin 0.0  - 1e12
    ///                                      Group::Value(1e16)]);  // bin 1e12 - 1e16
    /// ```
    pub fn tmesh_groups(&self) -> Vec<Group> {
        self.time_bins_upper()
            .iter()
            .map(|time| Group::Value(*time))
            .collect::<Vec<Group>>()
    }
}

impl Mesh {
    /// Slice the full list of mesh Voxels by both energy/time groups
    ///
    /// Very fast, but operates on indicies and therefore relies on the voxels
    /// being sorted. For an arbitrary list of [Voxel](crate::mesh::Voxel)s,
    /// use the filter methods which explicitly check the groups of every voxel
    /// provided.
    pub fn slice_voxels_by_group(
        &self,
        energy_group: Group,
        time_group: Group,
    ) -> Result<&[Voxel]> {
        let e_idx = self.find_energy_group_index(energy_group)?;
        let t_idx = self.find_time_group_index(time_group)?;

        let group_size = self.tbins() * self.iints * self.jints * self.kints;
        let start = e_idx * group_size;
        let end = start + group_size;
        let voxels = &self.voxels[start..end];

        let group_size = self.iints * self.jints * self.kints;
        let start = t_idx * group_size;
        let end = start + group_size;
        Ok(&voxels[start..end])
    }

    /// Slice the full list of mesh Voxels by energy/time index
    ///
    /// Very fast, but operates on indicies and therefore relies on the voxels
    /// being sorted. For an arbitrary list of [Voxel](crate::mesh::Voxel)s,
    /// use the filter methods which explicitly check the groups of every voxel
    /// provided.
    pub fn slice_voxels_by_idx(&self, e_idx: usize, t_idx: usize) -> Result<&[Voxel]> {
        // Just quickly make sure the values given are reasonable
        if e_idx > self.ebins() {
            return Err(anyhow!(
                "{e_idx} invalid for mesh with {} energy bins",
                self.ebins()
            ));
        } else if t_idx > self.tbins() {
            return Err(anyhow!(
                "{t_idx} invalid for mesh with {} time bins",
                self.tbins()
            ));
        };

        // slice voxels down to the full energy group
        let group_size = self.tbins() * self.iints * self.jints * self.kints;
        let start = e_idx * group_size;
        let end = start + group_size;
        let voxels = &self.voxels[start..end];

        // slice again down the right time group
        let group_size = self.iints * self.jints * self.kints;
        let start = t_idx * group_size;
        let end = start + group_size;

        // is there a more concise way? probably, but it's late and this makes sense
        Ok(&voxels[start..end])
    }

    /// Get a full set of coordinates for a voxel
    ///
    ///
    /// ```rust
    /// # use meshtal::mesh::{read_meshtal_target, Mesh, VoxelCoordinate, Group};
    /// // Get some voxel from an example file
    /// let mesh = read_meshtal_target("./data/meshes/fmesh_114.msht", 114).unwrap();
    /// // Calculate the coordinates for the voxel
    /// assert_eq!( mesh.voxel_coordinates(63).unwrap(),
    ///             VoxelCoordinate {
    ///                 energy: Group::Value(1.0),
    ///                 time: Group::Value(1.0E+30),
    ///                 i: 9.375,
    ///                 j: 4.500,
    ///                 k: 2.500} );
    /// ```
    pub fn voxel_coordinates(&self, index: usize) -> Result<VoxelCoordinate> {
        if index > self.voxels.len() {
            return Err(anyhow!(
                "Index ({}) larger than total number of voxels in mesh ({})",
                index,
                self.voxels.len()
            ));
        }

        // get the indicies
        let (e_idx, t_idx, i_idx, j_idx, k_idx) = self.voxel_index_to_etijk(index);

        Ok(VoxelCoordinate {
            energy: self.energy_group_from_index(e_idx)?,
            time: self.time_group_from_index(t_idx)?,
            i: (self.imesh[i_idx + 1] + self.imesh[i_idx]) / 2.0,
            j: (self.jmesh[j_idx + 1] + self.jmesh[j_idx]) / 2.0,
            k: (self.kmesh[k_idx + 1] + self.kmesh[k_idx]) / 2.0,
        })
    }

    /// Expected number of voxels per group set
    ///
    /// Returns the product of the number of fine mesh bins in each dimension.
    /// This is equivalent to the total number of voxels in the mesh geometry
    /// itself. For example, a 5x5x5 rectangular mesh should have 100 voxels
    /// for every unique energy/time group set.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh};
    /// // Generate a simple test mesh
    /// let mesh = Mesh {
    ///     iints: 2,
    ///     jints: 3,
    ///     kints: 4,
    ///     ..Default::default()
    /// };
    /// // Calculate the coordinates for the voxel
    /// assert_eq!(mesh.n_voxels_per_group(), 24);
    /// ```
    pub fn n_voxels_per_group(&self) -> usize {
        self.iints * self.jints * self.kints
    }

    /// Calculate how many voxels there should be
    ///
    /// Useful for common sense checking against the value returned by a
    /// `mesh.voxels.len()` call. This is the product of the number of energy
    /// bins, time bins, and fine mesh bins in each dimension.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh};
    /// // Generate a simple test mesh
    /// let mesh = Mesh {
    ///     eints: 1,   // 1 energy group (total only)
    ///     tints: 2,   // 3 time groups (2+total)
    ///     iints: 2,
    ///     jints: 3,
    ///     kints: 4,
    ///     ..Default::default()
    /// };
    /// // Calculate the coordinates for the voxel
    /// assert_eq!(mesh.n_voxels_expected(), 72);
    pub fn n_voxels_expected(&self) -> usize {
        self.ebins() * self.tbins() * self.iints * self.jints * self.kints
    }
}

/// Indexing helpers
impl Mesh {
    /// Find the global voxel index from (e,t,i,j,k) indicies
    ///
    /// The voxel index corresponds to the order seen in the column format
    /// output files. All formats are coerced into this sorting for consistency.
    /// MCNP writes voxels as (k,j,i,t,e) loops, i.e:
    ///
    /// ```rust, ignore, no_run
    /// for e in energy_groups
    ///     for t in time_groups
    ///         for i in imesh
    ///             for j in jmesh
    ///                 for k in kmesh
    ///                     // ... write voxel
    /// ```
    ///
    /// Frustratingly, MCNP internally uses the "cell index" ordering which
    /// loops in the reverse for (i,j,k). See
    /// [etijk_to_cell_index()](Mesh::etijk_to_cell_index) for details.
    pub fn etijk_to_voxel_index(
        &self,
        e_idx: usize,
        t_idx: usize,
        i_idx: usize,
        j_idx: usize,
        k_idx: usize,
    ) -> usize {
        let mut idx: usize = e_idx * (self.tbins() * self.iints * self.jints * self.kints);
        idx += t_idx * (self.iints * self.jints * self.kints);
        idx += i_idx * (self.jints * self.kints);
        idx += j_idx * (self.kints);
        idx += k_idx;
        idx
    }

    /// Find the global cell index from (e,t,i,j,k) indicies
    ///
    /// The cell index corresponds to the internal ordering used by MCNP, not
    /// what is seen in output files. The cell index number loops energy and
    /// time groups as expected, but within thses the (i,j,k) indicies are
    /// calculated as:
    ///
    /// ```text
    ///     cell index number = i + (j − 1) x I + (k − 1) x I x J
    /// ```
    ///
    /// where `I` and `J` are the total number of `imesh` and `jmesh` bins. In
    /// other words, this loops as:
    ///
    /// ```rust, ignore, no_run
    /// for e in energy_groups
    ///     for t in time_groups
    ///         for k in kmesh
    ///             for j in jmesh
    ///                 for i in imesh
    ///                     // ...some voxel
    /// ```
    ///
    /// This is the order needed by VTK formats and weight window files.
    pub fn etijk_to_cell_index(
        &self,
        e_idx: usize,
        t_idx: usize,
        i_idx: usize,
        j_idx: usize,
        k_idx: usize,
    ) -> usize {
        let mut idx: usize = e_idx * self.tbins() * self.iints * self.jints * self.kints;
        idx += t_idx * (self.iints * self.jints * self.kints);
        idx += k_idx * (self.iints * self.jints);
        idx += j_idx * self.iints;
        idx += i_idx;
        idx
    }

    // // todo implement this
    // pub fn cell_index_to_etijk(&self, _idx: usize) {}

    /// Find the (e,t,i,j,k) indicies for a given voxel index
    ///
    /// The reverse of [etijk_to_voxel_index()](Mesh::etijk_to_voxel_index).
    pub fn voxel_index_to_etijk(&self, idx: usize) -> (usize, usize, usize, usize, usize) {
        // convenient values for readability
        let a: usize = self.tbins() * self.kints * self.jints * self.iints;
        let b: usize = self.kints * self.jints * self.iints;
        let c: usize = self.kints * self.jints;
        let d: usize = self.kints;

        // find indicies in reverse (integer division floors in Rust)
        let e: usize = idx / a;
        let t: usize = (idx - e * a) / b;
        let i: usize = (idx - e * a - t * b) / c;
        let j: usize = (idx - e * a - t * b - i * c) / d;
        let k: usize = idx - e * a - t * b - i * c - j * d;

        (e, t, i, j, k)
    }

    /// Convert voxel index to a cell index
    ///
    /// Besides the obvious convenience, this is needed for UKAEA CuV formats.
    /// For some reason it was thought hilarious to order the
    /// `Number_of_material_cells_per_voxel` array by cell index while ordering
    /// the output column data by voxel index.
    pub fn voxel_index_to_cell_index(&self, idx: usize) -> usize {
        let (e, t, i, j, k) = self.voxel_index_to_etijk(idx);
        self.etijk_to_cell_index(e, t, i, j, k)
    }
}

/// Mesh associated functions
///
/// i.e. called as Mesh::function()
impl Mesh {
    /// Initialise new mesh with known id
    ///
    /// The `id` is the tally number used on the `FMESH` card in the input deck.
    /// For example, Fmesh204:n => id=204. This will initialise a [Mesh] with
    /// all of the default values.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Find the maximum result in a set of voxels
    ///
    /// If the list of voxels is empty, return 0.0 by default. To handle this
    /// or any other failing case explicitly use [Mesh::try_maximum()] instead.
    ///
    /// ```rust
    /// # use meshtal::mesh::{read_meshtal_target, Mesh};
    /// // Get a mesh from file
    /// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
    ///
    /// // Find the maximum of all voxels
    /// assert_eq!(Mesh::maximum(&mesh.voxels), 2.40000e+01);
    /// // Find the maximum of a subset of voxels
    /// assert_eq!(Mesh::maximum(&mesh.voxels[0..12]), 1.20000e+01);
    /// // Failing case of an empty list
    /// assert_eq!(Mesh::maximum([]), 0.0);
    /// ```
    pub fn maximum<V: AsRef<[Voxel]>>(voxels: V) -> f64 {
        Mesh::try_maximum(voxels).unwrap_or(0.0)
    }

    /// Find the maximum result in a set of voxels
    ///
    /// Returns a result so that the failing cases can be handled explicitly.
    /// [Mesh::maximum()] may be used if 0.0 is an acceptable default.
    ///
    /// ```rust
    /// # use meshtal::mesh::{read_meshtal_target, Mesh};
    /// // Get a mesh from file
    /// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
    ///
    /// // Passing case of a slice of Voxels
    /// assert!(Mesh::try_maximum(&mesh.voxels).is_ok());
    /// // Failing case of an empty list
    /// assert!(Mesh::try_maximum([]).is_err());
    /// ```
    pub fn try_maximum<V: AsRef<[Voxel]>>(voxels: V) -> Result<f64> {
        let max_voxel = voxels
            .as_ref()
            .iter()
            .max_by(|a, b| a.result.partial_cmp(&b.result).unwrap())
            .ok_or(anyhow!("List of voxels is empty"))?;

        Ok(max_voxel.result)
    }

    /// Find the minimum result in a set of voxels
    ///
    /// If the list of voxels is empty, return 0.0 by default. To handle this
    /// or any other failing case explicitly use [Mesh::try_minimum()] instead.
    ///
    /// ```rust
    /// # use meshtal::mesh::{read_meshtal_target, Mesh};
    /// // Get a mesh from file
    /// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
    ///
    /// // Find the maximum of all voxels
    /// assert_eq!(Mesh::minimum(&mesh.voxels), 1.00000e+00);
    /// // Find the maximum of a subset of voxels
    /// assert_eq!(Mesh::minimum(&mesh.voxels[6..12]), 7.00000e+00);
    /// // Failing case of an empty list
    /// assert_eq!(Mesh::minimum([]), 0.0);
    /// ```
    pub fn minimum<V: AsRef<[Voxel]>>(voxels: V) -> f64 {
        Mesh::try_minimum(voxels).unwrap_or(0.0)
    }

    // todo this should be an option as it will only fail if empty I think
    /// Find the minimum result in a set of voxels
    ///
    /// Returns a result so that the failing cases can be handled explicitly.
    /// [Mesh::minimum()] may be used if 0.0 is an acceptable default.
    ///
    /// ```rust
    /// # use meshtal::mesh::{read_meshtal_target, Mesh};
    /// // Get a mesh from file
    /// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
    ///
    /// // Passing case of a slice of Voxels
    /// assert!(Mesh::try_minimum(&mesh.voxels).is_ok());
    /// // Failing case of an empty list
    /// assert!(Mesh::try_minimum([]).is_err());
    /// ```
    pub fn try_minimum<V: AsRef<[Voxel]>>(voxels: V) -> Result<f64> {
        let min_voxel = voxels
            .as_ref()
            .iter()
            .min_by(|a, b| a.result.partial_cmp(&b.result).unwrap())
            .ok_or(anyhow!("List of voxels is empty"))?;

        Ok(min_voxel.result)
    }

    /// Find the average (mean) result
    ///
    /// If the list of voxels is empty, return 0.0 by default. To handle this
    /// case explicitly use [Mesh::try_average()] instead.
    ///
    /// ```rust
    /// # use meshtal::mesh::{read_meshtal_target, Mesh};
    /// // Get a mesh from file
    /// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
    ///
    /// // Find the maximum of all voxels
    /// assert_eq!(Mesh::average(&mesh.voxels), 1.25000e+01);
    /// // Find the maximum of a subset of voxels
    /// assert_eq!(Mesh::average(&mesh.voxels[6..12]), 9.50000e+00);
    /// // Failing case of an empty list
    /// assert_eq!(Mesh::average([]), 0.0);
    /// ```
    pub fn average<V: AsRef<[Voxel]>>(voxels: V) -> f64 {
        Mesh::try_average(voxels).unwrap_or(0.0)
    }

    /// Find the average (mean) result
    ///
    /// Returns a result so that an empty list can be handled explicitly, but
    /// [Mesh::average()] may be used if 0.0 is an acceptable default.
    ///
    /// ```rust
    /// # use meshtal::mesh::{read_meshtal_target, Mesh};
    /// // Get a mesh from file
    /// let mesh = read_meshtal_target("./data/meshes/fmesh_104.msht", 104).unwrap();
    ///
    /// // Find the maximum of all voxels
    /// assert!(Mesh::try_average(&mesh.voxels).is_ok());
    /// // Failing case of an empty list
    /// assert!(Mesh::try_average([]).is_err());
    /// ```
    pub fn try_average<V: AsRef<[Voxel]>>(voxels: V) -> Result<f64> {
        if voxels.as_ref().is_empty() {
            Err(anyhow!("Cannot average empty list"))
        } else {
            let total: f64 = voxels.as_ref().iter().map(|s| s.result).sum();
            Ok(total / voxels.as_ref().len() as f64)
        }
    }

    /// Get a copy of all results and errors from a collection of voxels
    ///
    /// Returns a vector of (`result`, `error`) tuples from the provided
    /// collection of voxels. Useful for quickly getting the final results.
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh,Voxel};
    /// // Simple test mesh with three voxels
    /// let voxels = vec![
    ///     Voxel {index: 0, result: 12.0, error: 0.1},
    ///     Voxel {index: 1, result: 18.0, error: 0.1},
    ///     Voxel {index: 2, result: 15.0, error: 0.1},
    /// ];
    ///
    /// assert_eq!(Mesh::voxel_data(voxels), vec![(12.0, 0.1), (18.0, 0.1), (15.0, 0.1)]);
    /// ```
    ///
    pub fn voxel_data<V: AsRef<[Voxel]>>(voxels: V) -> Vec<(f64, f64)> {
        voxels
            .as_ref()
            .iter()
            .map(|v| (v.result, v.error))
            .collect()
    }
}

/// Voxel indexing, filters, and useful searches
impl Mesh {
    /// Find index bin containing 'value', where bins are low < value <= high
    ///
    /// A value on a bin edge returns the bin below. Values equal to the lowest
    /// bound are considered part of the first bin.
    ///
    /// # Example
    /// ```text
    ///     MCNP   : emesh = 0.1 1.0 20.0
    ///     Meshtal: 0.00E+00 1.00E-01 1.00E+00 2.00E+01
    /// ```
    ///
    /// view of the requested energy groups
    /// ```text
    ///     0.0 <= bin 0 <= 0.1
    ///     0.1 < bin 1 <= 1.0
    ///     1.0 < bin 2 <= 20.0
    /// ```
    ///
    /// ```rust
    /// # use meshtal::mesh::Mesh;
    /// let bin_edges = vec![0.0, 0.1, 1.0, 20.0];
    /// // Find values in the array
    /// assert_eq!(Mesh::find_bin_inclusive(0.0, &bin_edges).unwrap(), 0);
    /// assert_eq!(Mesh::find_bin_inclusive(0.5, &bin_edges).unwrap(), 1);
    /// assert_eq!(Mesh::find_bin_inclusive(1.0, &bin_edges).unwrap(), 1);
    /// assert_eq!(Mesh::find_bin_inclusive(20.0, &bin_edges).unwrap(), 2);
    /// // Values outside the bin bounds are an error case
    /// assert!(Mesh::find_bin_inclusive(-1.0, &bin_edges).is_err());
    /// assert!(Mesh::find_bin_inclusive(21.0, &bin_edges).is_err());
    /// ```
    pub fn find_bin_inclusive(value: f64, bin_edges: &[f64]) -> Result<usize> {
        // make sure there are bin edges to check against
        if bin_edges.is_empty() {
            return Err(anyhow!("Bin edges array is empty"));
        }

        // should be fine to unwrap as not empty
        let lower_bound = bin_edges.first().unwrap();
        let upper_bound = bin_edges.last().unwrap();

        // is the value relevant?
        if &value < lower_bound || &value > upper_bound {
            return Err(anyhow!("Value {value:?} outside of bin edges"));
        }

        // special case for being on the lowest edge
        if &value == lower_bound {
            return Ok(0);
        }

        let (_, lower_edges) = bin_edges.split_last().unwrap();
        let (_, upper_edges) = bin_edges.split_first().unwrap();

        // try to find the bin index, range INCLUSIVE of upper edge
        for (idx, (low, high)) in lower_edges.iter().zip(upper_edges.iter()).enumerate() {
            // println!("     is {} < {} <= {}")
            if low < &value && &value <= high {
                return Ok(idx);
            }
        }
        Err(anyhow!("Value {value} was not found in bin edges"))
    }

    /// Find index bin containing 'value', where bins are low <= value < high
    ///
    /// A value on a bin edge returns the bin above. Values equal to the highest
    /// bound are considered part of the last bin.
    ///
    /// This mirrors the actual MCNP binning behaviour. `EMESH` card entries are
    /// upper edges, so in general values on a boundary will be recorded in the
    /// bin above. A special case is made for energies exactly on the last upper
    /// edge, since this is actually included in the tallied results.
    ///
    ///  # Example
    /// ```text
    ///     MCNP card : EMESH = 0.1 1.0 20.0
    ///     Mesh.emesh: 0.00E+00 1.00E-01 1.00E+00 2.00E+01
    /// ```
    ///
    /// MCNP view of these bins:
    /// ```text
    ///     0.0 <= bin 0 < 0.1
    ///     0.1 <= bin 1 < 1.0
    ///     1.0 <= bin 2 <= 20.0
    /// ```
    ///
    /// ```rust
    /// # use meshtal::mesh::Mesh;
    /// let bin_edges = vec![0.0, 0.1, 1.0, 20.0];
    /// // Find values in the array
    /// assert_eq!(Mesh::find_bin_exclusive(0.0, &bin_edges).unwrap(), 0);
    /// assert_eq!(Mesh::find_bin_exclusive(0.5, &bin_edges).unwrap(), 1);
    /// assert_eq!(Mesh::find_bin_exclusive(1.0, &bin_edges).unwrap(), 2);
    /// assert_eq!(Mesh::find_bin_inclusive(20.0, &bin_edges).unwrap(), 2);
    /// // Values outside the bin bounds are an error case
    /// assert!(Mesh::find_bin_exclusive(-1.0, &bin_edges).is_err());
    /// assert!(Mesh::find_bin_exclusive(21.0, &bin_edges).is_err());
    /// ```
    pub fn find_bin_exclusive(value: f64, bin_edges: &[f64]) -> Result<usize> {
        // make sure there are bin edges to check against
        if bin_edges.is_empty() {
            return Err(anyhow!("Bin edges array is empty"));
        }

        // should be fine to unwrap as not empty
        let lower_bound = bin_edges.first().unwrap();
        let upper_bound = bin_edges.last().unwrap();

        // is the value relevant?
        if &value < lower_bound || &value > upper_bound {
            return Err(anyhow!("Value {value:?} outside of bin edges"));
        }

        // special case for being on the upper edge
        if &value == upper_bound {
            return Ok(bin_edges.len() - 1);
        }

        let (_, lower_edges) = bin_edges.split_last().unwrap();
        let (_, upper_edges) = bin_edges.split_first().unwrap();

        // try to find the bin index, range EXCLUSIVE of upper edge
        for (idx, (low, high)) in lower_edges.iter().zip(upper_edges.iter()).enumerate() {
            if low <= &value && &value < high {
                return Ok(idx);
            }
        }
        Err(anyhow!("Value {value} was not found in bin edges"))
    }

    /// For a given energy, find what group the results are under
    ///
    /// Error returned for handling if the energy is outside of the emesh bounds
    /// and should be handled by the caller.
    ///
    /// Important: the group an energy is under is determined by
    /// [find_bin_inclusive()](Mesh::find_bin_inclusive) rules for MCNP output,
    /// even though internally it follows the
    /// [find_bin_exclusive()](Mesh::find_bin_exclusive) rules.
    pub fn find_energy_group(&self, energy: f64) -> Result<Group> {
        if self.ebins() == 1 {
            Ok(Group::Total)
        } else {
            let idx = Self::find_bin_inclusive(energy, &self.emesh)?;
            Ok(Group::Value(self.emesh[idx]))
        }
    }

    /// For a given energy group, find the corresponding `emesh` bin index
    ///
    /// Important: the group an energy is under is determined by
    /// [find_bin_inclusive()](Mesh::find_bin_inclusive) rules for MCNP output,
    /// even though internally it follows the
    /// [find_bin_exclusive()](Mesh::find_bin_exclusive) rules.
    pub fn find_energy_group_index(&self, energy_group: Group) -> Result<usize> {
        match energy_group {
            Group::Total => Ok(self.ebins() - 1),
            Group::Value(energy) => Self::find_bin_inclusive(energy, &self.emesh),
        }
    }

    /// For a given time, find what group the results are under
    ///
    /// Error returned for handling if the time is outside of the emesh bounds
    /// and should be handled by the caller.
    ///
    /// Important: the group a time is under is determined by
    /// [find_bin_inclusive()](Mesh::find_bin_inclusive) rules for MCNP output,
    /// even though internally it follows the
    /// [find_bin_exclusive()](Mesh::find_bin_exclusive) rules.
    pub fn find_time_group(&self, time: f64) -> Result<Group> {
        // tmesh can be empty, have just total, or values + total
        if self.tbins() == 1 {
            Ok(Group::Total)
        } else {
            let idx = Self::find_bin_inclusive(time, &self.tmesh)?;
            Ok(Group::Value(self.tmesh[idx]))
        }
    }

    /// For a given time group, find the corresponding `tmesh` bin index
    ///
    /// Important: the group a time is under is determined by
    /// [find_bin_inclusive()](Mesh::find_bin_inclusive) rules for MCNP output,
    /// even though internally it follows the
    /// [find_bin_exclusive()](Mesh::find_bin_exclusive) rules.
    pub fn find_time_group_index(&self, time_group: Group) -> Result<usize> {
        match time_group {
            Group::Total => Ok(self.tbins() - 1),
            Group::Value(time) => Self::find_bin_inclusive(time, &self.tmesh),
        }
    }

    /// Get the energy [Group](crate::mesh::Group) for the energy index `e_idx`
    ///
    /// For example:
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh, Group};
    /// // Generate a simple test mesh
    /// let mesh = Mesh {
    ///     eints: 2,
    ///     emesh: vec![0.0, 1.0, 20.0],
    ///     ..Default::default()
    /// };
    /// // Find the group for an energy bin index
    /// assert_eq!(mesh.energy_group_from_index(0).unwrap(), Group::Value(1.0));
    /// assert_eq!(mesh.energy_group_from_index(1).unwrap(), Group::Value(20.0));
    /// assert_eq!(mesh.energy_group_from_index(2).unwrap(), Group::Total);
    /// assert!(mesh.energy_group_from_index(3).is_err());
    /// ```
    pub fn energy_group_from_index(&self, e_idx: usize) -> Result<Group> {
        if e_idx < self.ebins() {
            Ok(self.energy_groups()[e_idx])
        } else {
            Err(anyhow!("Index {e_idx} outside range of energy groups"))
        }
    }

    /// Get the time [Group](crate::mesh::Group) for the time index `t_idx`
    ///
    /// For example:
    ///
    /// ```rust
    /// # use meshtal::mesh::{Mesh, Group};
    /// // Generate a simple test mesh
    /// let mesh = Mesh {
    ///     tints: 2,
    ///     tmesh: vec![0.0, 1e16, 1e30],
    ///     ..Default::default()
    /// };
    /// // Find the group for an energy bin index
    /// assert_eq!(mesh.time_group_from_index(0).unwrap(), Group::Value(1e16));
    /// assert_eq!(mesh.time_group_from_index(1).unwrap(), Group::Value(1e30));
    /// assert_eq!(mesh.time_group_from_index(2).unwrap(), Group::Total);
    /// assert!(mesh.time_group_from_index(3).is_err());
    /// ```
    pub fn time_group_from_index(&self, t_idx: usize) -> Result<Group> {
        if t_idx < self.tbins() {
            Ok(self.time_groups()[t_idx])
        } else {
            Err(anyhow!("Index {t_idx} outside range of time groups"))
        }
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            id: 0,
            geometry: Geometry::Rectangular,
            particle: Particle::Unknown,
            imesh: Vec::new(),
            iints: 0,
            jmesh: Vec::new(),
            jints: 0,
            kmesh: Vec::new(),
            kints: 0,
            emesh: Vec::new(),
            eints: 0,
            tmesh: Vec::new(),
            tints: 0,
            origin: [0.0, 0.0, 0.0],
            axs: [0.0, 0.0, 1.0],
            vec: [1.0, 0.0, 0.0],
            voxels: Vec::new(),
            format: Format::NONE,
        }
    }
}

impl fmt::Display for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let head: String = f!(
            " > Mesh {} [{:?}, {:?}]",
            self.id,
            self.particle,
            self.geometry
        );

        let mut s = f!("{}\n{}\n{}\n", "-".repeat(40), head, "-".repeat(40));

        s += &f!("origin: {:?}\n", self.origin);
        s += &f!("axs   : {:?}\n", self.axs);
        s += &f!("vec   : {:?}\n", self.vec);

        s += &f!(
            "imesh : {:>10} - {:>8} cm ({} bins)\n",
            self.imesh[0].sci(2, 2),
            self.imesh.last().unwrap().sci(2, 2),
            self.iints
        );
        s += &f!(
            "jmesh : {:>10} - {:>8} cm ({} bins)\n",
            self.jmesh[0].sci(2, 2),
            self.jmesh.last().unwrap().sci(2, 2),
            self.jints
        );
        s += &f!(
            "kmesh : {:>10} - {:>8} cm ({} bins)\n",
            self.kmesh[0].sci(2, 2),
            self.kmesh.last().unwrap().sci(2, 2),
            self.kints
        );
        s += &f!(
            "emesh : {:>10} - {:>8} MeV ({} bins)\n",
            self.emesh[0].sci(2, 2),
            self.emesh.last().unwrap().sci(2, 2),
            self.ebins()
        );
        if self.tints > 1 {
            s += &f!(
                "tmesh : {:>10} - {:>8} shakes ({} bins)\n",
                self.tmesh[0].sci(2, 2),
                self.tmesh.last().unwrap().sci(2, 2),
                self.tbins()
            );
        }
        write!(f, "{}", s)
    }
}

/// Mesh geometry types, i.e. `Rectangular`, `Cylindrical`
#[derive(Debug, PartialEq, Clone, Copy, ValueEnum)]
pub enum Geometry {
    /// Cartesian (rec, xyz) mesh type
    Rectangular = 1,
    /// Cylindrical (cyl, rzt) mesh type
    Cylindrical = 2,
    // todo add spherical mesh type and implement into parsers etc...
    // Spherical (sph, rpt) mesh type
    // Spherical = 3
}

impl Geometry {
    /// Full name i.e. 'Rectangular', 'Cylindrical'
    pub fn long_name(&self) -> &str {
        match self {
            Geometry::Rectangular => "Rectangular",
            Geometry::Cylindrical => "Cylindrical",
        }
    }

    /// Shortened name i.e. 'Rec', 'Cyl'
    pub fn short_name(&self) -> &str {
        match self {
            Geometry::Rectangular => "Rec",
            Geometry::Cylindrical => "Cyl",
        }
    }

    /// Coordinate system based name i.e. 'XYZ', 'RZT'
    pub fn geometry_name(&self) -> &str {
        match self {
            Geometry::Rectangular => "XYZ",
            Geometry::Cylindrical => "RZT",
        }
    }
}

impl fmt::Display for Geometry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.geometry_name())
    }
}

/// Meshtal output formats, e.g. `COL`, `JK`, `CUV`...
#[derive(Debug, Clone, PartialEq)]
pub enum Format {
    /// Column data (MCNP default)
    ///
    /// The default MCNPv6.2 output format.
    ///
    /// Example:
    /// ```text
    ///  Energy      X      Y      Z     Result     Rel Error
    /// 1.111E+00 -0.500 -0.733 -2.625 7.25325E-03 1.20187E-02
    /// 1.111E+00 -0.500 -0.733 -0.875 3.43507E-02 4.71983E-03
    /// etc ...
    /// ```
    COL,
    /// Column data including voxel volume
    ///
    /// Same as column, but with extra information on voxel volumes. Since these
    /// data are derivable, volume information is descarded during parsing.
    ///
    /// Example:
    /// ```text
    ///  Energy      X      Y      Z     Result     Rel Error     Volume    Rslt * Vol
    /// 1.111E+00 -0.500 -0.733 -2.625 7.25325E-03 1.20187E-02 1.28333E+00 9.30834E-03
    /// 1.111E+00 -0.500 -0.733 -0.875 3.43507E-02 4.71983E-03 1.28333E+00 4.40834E-02
    /// etc ...
    /// ```
    CF,
    /// Cell-under-Voxel column data
    ///
    /// The UKAEA Cell under Voxel patch coerces all meshes, regradless of input
    /// deck specifications, to its own format. Multiple entries can correspond
    /// to the same voxel and voxels with only void cells can be omitted
    /// entirely. This is all handled in the background by the parsers.
    ///
    /// The cell number, matrial, and volume are parsed but not currently used
    /// while the main functionality is implemented.
    ///
    /// Example:
    /// ```text
    ///  Energy   Cell Mat  Density     Volume      X     Y       Z      Result   Rel Error
    /// 1.000E+35  76  6  8.00000E+00 4.47858E+02 0.697 9.000 -16.000 1.23957E-04 2.97900E-02
    /// 1.000E+35  84  6  8.00000E+00 5.06160E+00 0.697 9.000 -16.000 2.36108E-04 1.14448E-01
    /// etc ...
    /// ```
    CUV,
    /// 2D matrix of I (col) and J (row) data, grouped by K
    ///
    /// Matrix outputs separated into tables for two dimensions, and grouped by
    /// the third. For example, the IJ tables for
    /// [Geomtery::Rectangular](crate::mesh::Geometry) are X by Y, grouped by Z
    /// bins.
    ///
    /// ```text
    /// Energy Bin: 0.00E+00 - 1.11E+00 MeV
    /// Time Bin: -1.00E+36 - 0.00E+00 shakes
    ///   Z bin: -3.50  -  -1.75
    ///     Tally Results:  X (across) by Y (down)
    ///                   -0.50        0.50
    ///         -0.73   0.00000E+00 0.00000E+00
    ///          0.00   0.00000E+00 0.00000E+00
    ///          0.73   0.00000E+00 0.00000E+00
    ///     Relative Errors
    ///                   -0.50        0.50
    ///         -0.73   0.00000     0.00000
    ///          0.00   0.00000     0.00000
    ///          0.73   0.00000     0.00000
    /// ```
    IJ,
    /// 2D matrix of I (col) and K (row) data, grouped by J
    ///
    /// Matrix outputs separated into tables for two dimensions, and grouped by
    /// the third. For example, the IK tables for
    /// [Geomtery::Rectangular](crate::mesh::Geometry) are X by Z, grouped by Y
    /// bins.
    ///
    /// ```text
    /// Energy Bin: 0.00E+00 - 1.11E+00 MeV
    /// Time Bin: -1.00E+36 - 0.00E+00 shakes
    ///   Y bin: -3.50  -  -1.75
    ///     Tally Results:  X (across) by Z (down)
    ///                   -0.50        0.50
    ///         -0.73   0.00000E+00 0.00000E+00
    ///          0.00   0.00000E+00 0.00000E+00
    ///          0.73   0.00000E+00 0.00000E+00
    ///     Relative Errors
    ///                   -0.50        0.50
    ///         -0.73   0.00000     0.00000
    ///          0.00   0.00000     0.00000
    ///          0.73   0.00000     0.00000
    /// ```
    IK,
    /// 2D matrix of J (col) and K (row) data, grouped by I
    ///
    /// Matrix outputs separated into tables for two dimensions, and grouped by
    /// the third. For example, the JK tables for
    /// [Geomtery::Rectangular](crate::mesh::Geometry) are Z by Y, grouped by X
    /// bins.
    ///
    /// ```text
    /// Energy Bin: 0.00E+00 - 1.11E+00 MeV
    /// Time Bin: -1.00E+36 - 0.00E+00 shakes
    ///   X bin: -3.50  -  -1.75
    ///     Tally Results:  Y (across) by Z (down)
    ///                   -0.50        0.50
    ///         -0.73   0.00000E+00 0.00000E+00
    ///          0.00   0.00000E+00 0.00000E+00
    ///          0.73   0.00000E+00 0.00000E+00
    ///     Relative Errors
    ///                   -0.50        0.50
    ///         -0.73   0.00000     0.00000
    ///          0.00   0.00000     0.00000
    ///          0.73   0.00000     0.00000
    /// ```
    JK,
    /// Special case for unknown format or meshes with no output
    NONE,
}
