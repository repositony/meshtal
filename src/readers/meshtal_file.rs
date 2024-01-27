// internal modules
use crate::mesh::{Format, Geometry, Group, Mesh, Particle, Voxel};
use crate::readers::parsers;
use crate::utils::*;

// standard library
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// external crates
use anyhow::{anyhow, bail, Context, Result};
use kdam::{Bar, BarBuilder, BarExt};
use log::{debug, trace};
use nom::IResult;

/// A generalised reader for legacy meshtal files of any type
///
/// Supports COL, CF, UKAEA Cell-under-Voxel, IJ, IK, and JK output formats for
/// both rectangular and cylindrical meshes.
///
/// The reader operates in two stages to minimise time wasted on erroneous
/// inputs:
///     - Quickly check mesh tally IDs and auto-detect their formatting
///     - Extract the data using the appropriate parsing strategy
///
/// Notes:
///     - CuV results are weighted by volume of each contributing cell
///     - VoidRecord::Off will fill in missing void voxels with 0.0 flux
///     - Run-on numbers without whitespace are handled e.g. 1.00E+00-2.00E+00
///     - Broken exponential formatting is handled e.g. 1.00+002 => 1.00E+002
///     - Rectangualr mesh origin set to match FMESH input cards
///
/// Example:
/// ```ignore
///     let path = Path::new(path);
///     let mut reader = MeshtalReader::new();
///     let mesh_list = reader.parse(path).unwrap();
/// ```
#[derive(Debug)]
pub struct MeshtalReader {
    /// List of extracted [Mesh] tallies
    mesh_list: Vec<Mesh>,
    /// Optionally extract only a specific mesh
    target_id: Option<u32>,
    /// Flag for an early return if the target mesh has already been extracted
    is_target_extracted: bool,
    /// Tracking required for reading the 2D matrix data line-by-line
    tracked: Tracked,
    /// CuV flag for recording of void cells
    void_record: VoidRecord,
    /// Material cells per voxel array
    mcpv: Vec<u32>,
    /// Disable progress bar?
    disable_progress: bool,
    /// Last known voxel cell data for CuV parsing
    previous_cell: Option<CellData>,
}

impl Default for MeshtalReader {
    fn default() -> Self {
        Self {
            mesh_list: Vec::new(),
            target_id: None,
            is_target_extracted: false,
            tracked: Tracked::default(),
            void_record: VoidRecord::Off,
            mcpv: Vec::new(),
            disable_progress: false,
            previous_cell: None,
        }
    }
}

/// High level methods
impl MeshtalReader {
    /// Just calls Default::default(), nothing special to be initialised
    pub fn new() -> Self {
        Default::default()
    }

    /// Parses all mesh data from a mcnp meshtal file
    ///
    /// May need to implement something to ensure precision consistency for the
    /// energy and time group values used
    pub fn parse(&mut self, path: &Path) -> Result<Vec<Mesh>> {
        // check the tally formats
        let format: FormatMap = self.check_tally_formats(path)?;

        // just make sure the requested id is in the data somewhere
        self.ensure_format_contains_target(&format)?;

        // extract all the relevant data from the file
        self.extract_meshtal_data(path, &format)?;

        // quick common sense check
        self.check_voxel_lengths()?;

        // add trailing voxels for void_record=off, which will not have been
        // included yet, and fix the uncertainties
        self.complete_cuv_voxels();

        // fix the origin to be bottom corner for rectangular meshes, not center
        // to mirror the FMESH cards
        self.apply_origin_fix();

        // sort the list for consistency between output formats
        self.sort_voxels();

        // do not care about the reader, so give the meshes to the caller
        // this saves cloning the data which is a massive win
        Ok(std::mem::take(&mut self.mesh_list))
    }

    /// Setter for specifying which mesh to target
    pub fn set_target_id(&mut self, target_id: u32) {
        self.target_id = Some(target_id);
    }

    /// Do not print the tqdm progress indicators
    pub fn disable_progress(&mut self) {
        debug!("Progress bar disabled");
        self.disable_progress = true;
    }
}

/// Primary run loop and fixes
impl MeshtalReader {
    /// Main entry point to the parsers, extracting the data records of each mesh
    fn extract_meshtal_data(&mut self, path: &Path, format: &FormatMap) -> Result<()> {
        // parse the data depending on Format type
        let file = File::open(path).with_context(|| f!("Could not open {}", path.display()))?;
        let reader = BufReader::new(file);

        // Set up all the general use stuff
        let column_hints = Self::init_column_hints();
        let matrix_hints = Self::init_matrix_hints();
        let cuv_hints = Self::init_cuv_hints();
        let mut progress_bar = self.init_progress_bar();

        debug!("Parsing mesh data");

        if !self.disable_progress {
            progress_bar.refresh()?;
        };

        for line in reader.lines() {
            progress_bar.update(1).unwrap();
            let line = line?;
            let line = line.trim_start(); // shadow needed to get around borrow

            // either the current mesh, or skip to the next loop if none exist or not targeted
            let mesh: &mut Mesh = match self.get_current_mesh(line) {
                None => continue,
                Some(m) => m,
            };

            // Set the formating if this is a new mesh
            if mesh.format == Format::NONE {
                mesh.format = format
                    .get(&mesh.id)
                    .unwrap_or_else(|| {
                        panic!(
                            "Format of Fmesh {} was not found during pre-processing",
                            mesh.id
                        )
                    })
                    .0
                    .clone();
                mesh.geometry = format.get(&mesh.id).unwrap().1; // .1 for MeshType
            }

            // Choose the appropriate parser for the format of the current mesh
            match mesh.format {
                Format::COL => self.parse_column_format(line, &column_hints)?,
                Format::CF => self.parse_column_format(line, &column_hints)?, // basically col
                Format::IJ => self.parse_matrix_format(line, &matrix_hints)?,
                Format::IK => self.parse_matrix_format(line, &matrix_hints)?,
                Format::JK => self.parse_matrix_format(line, &matrix_hints)?,
                Format::CUV => self.parse_cuv_format(line, &cuv_hints)?,
                Format::NONE => bail!("Format unknown for mesh {}", mesh.id),
            }
        }

        // need an extra line for clean spacing if the progress bar is pritned
        if !self.disable_progress {
            eprintln!()
        };

        Ok(())
    }

    /// Initialises the hints used for column data
    fn init_column_hints() -> [&'static dyn Fn(&str) -> bool; 4] {
        [
            &parsers::is_origin_axs_vec,  // 0 origin/axis/vector for cylindrical
            &parsers::is_particle_type,   // 1 particle type
            &parsers::is_geometry_bounds, // 2 imesh, jmesh, kmesh bounds
            &parsers::is_group_bounds,    // 3 energy/time bounds
        ]
    }

    /// Initialises the hints used for matrix data
    fn init_matrix_hints() -> [&'static dyn Fn(&str) -> bool; 7] {
        [
            &parsers::is_origin_axs_vec,  // 0 origin/axis/vector for cylindrical
            &parsers::is_particle_type,   // 1 particle typre
            &parsers::is_geometry_bounds, // 2 imesh, jmesh, kmesh bounds
            &parsers::is_group_bounds,    // 3 energy/time bounds
            &parsers::is_matrix_group,    // 4 energy/time group tables marker
            &parsers::is_new_table,       // 5 new pair result/error tables to follow
            &parsers::is_double_list,     // 6 any string of whitespace separated numbers
        ]
    }

    /// Initialises the hints used for cell-under-voxel data
    fn init_cuv_hints() -> [&'static dyn Fn(&str) -> bool; 6] {
        [
            &parsers::is_origin_axs_vec,  // 0 origin/axis/vector for cylindrical
            &parsers::is_particle_type,   // 1 particle typre
            &parsers::is_geometry_bounds, // 2 imesh, jmesh, kmesh bounds
            &parsers::is_group_bounds,    // 3 energy/time bounds
            &parsers::is_voidoff_status,  // 4 energy/time group tables marker
            &parsers::is_material_array,  // 5 new pair result/error tables to follow
        ]
    }

    /// Initialise the progress bar, if wanted
    fn init_progress_bar(&self) -> Bar {
        BarBuilder::default()
            .delay(0.0)
            .unit(" lines")
            .unit_scale(true)
            .disable(self.disable_progress)
            .build()
            .unwrap()
    }

    /// Finds the last relevant mesh, and makes a new one if non-existant
    fn get_current_mesh(&mut self, line: &str) -> Option<&mut Mesh> {
        // Mesh already extracted, just return early
        if self.is_target_extracted {
            return None;
        }

        // Try to find the existing mesh or make a new one
        if parsers::is_new_mesh(line) {
            let (_, id) = parsers::mesh_id(line)
                .map_err(|_| anyhow!("Failed to parse id from:\n \"{line}\""))
                .unwrap();

            // For targeted parsing, check against the target mesh id
            if let Some(target) = self.target_id {
                if target != id {
                    // Special case: infer target already extracted
                    if !self.mesh_list.is_empty() {
                        self.is_target_extracted = true;
                    }
                    return None;
                }
            }

            // add new mesh to the overall list
            self.mesh_list.push(Mesh::new(id));

            // Reset all tracked indices for matrix-type data
            self.tracked.reset();

            // Reset last known cell data and mcpv array for CuV-type data
            self.previous_cell = None;
            self.mcpv.clear();
        }

        // No meshes found yet -> not needed since .last() is an Option
        // Assume still adding to previous mesh
        self.mesh_list.last_mut()
    }

    /// Rectangular should be bottom corner rather than center of whole mesh if
    /// it is to mirror the values on the ORIGIN card
    fn apply_origin_fix(&mut self) {
        for m in &mut self.mesh_list {
            if m.geometry == Geometry::Rectangular {
                m.origin = [m.imesh[0], m.jmesh[0], m.kmesh[0]];
                trace!("Changed rectangular mesh origin to rear lower left");
            }
        }
    }

    /// Make sure that the number of voxels is as expected
    fn check_voxel_lengths(&self) -> Result<()> {
        for m in &self.mesh_list {
            if m.voxels.len() != m.n_voxels_expected() {
                return Err(anyhow!(
                    "Expected {} voxels in mesh {}, found {}",
                    m.n_voxels_expected(),
                    m.id,
                    m.voxels.len()
                ));
            }
        }
        Ok(())
    }

    /// Sort out the voxel order by consistent (e,t,i,j,k) index, as matrix can
    /// be all over the place
    fn sort_voxels(&mut self) {
        trace!("Voxels sorted by consistent (e,t,i,j,k) index");
        // matrix will be all over the place so sort for consistency
        for m in &mut self.mesh_list {
            // can skip over for column types as they are already sorted
            match m.format {
                Format::CF | Format::COL => (),
                _ => m.voxels.sort_by(|a, b| a.index.cmp(&b.index)),
            }
        }
    }
}

/// Preprocessing of tally formats
impl MeshtalReader {
    /// Quickly run through the file and find the mesh tally ids and formats
    fn check_tally_formats(&self, path: &Path) -> Result<FormatMap> {
        let file: File =
            File::open(path).with_context(|| f!("Could not open {}", path.display()))?;
        let reader: BufReader<File> = BufReader::new(file);

        let mut format_map: FormatMap = HashMap::new();
        let mut id: u32 = 0;
        let mut is_format_found: bool = false;
        let mut is_geometry_found: bool = false;
        let mut mesh_type: Geometry = Geometry::Rectangular;
        let hints = Self::init_format_hints();

        debug!("Checking tally formats");
        for line in reader.lines() {
            let line = line.unwrap();
            let line: &str = line.trim_start(); // shadow needed to get around borrow

            if parsers::is_new_mesh(line) {
                is_format_found = false;
                is_geometry_found = false;
                (_, id) = parsers::mesh_id(line)
                    .map_err(|_| anyhow!("Failed to parse id from:\n \"{line}\""))?;
                continue;
            }

            if !is_geometry_found && parsers::is_meshtype_hint(line) {
                mesh_type = Self::get_geometry_type(line);
                is_geometry_found = true;
            }

            if !is_format_found {
                if let Some(format) = Self::get_format_hint(line, hints, &mesh_type) {
                    is_format_found = true;
                    format_map.insert(id, (format, mesh_type));
                }

                // break read of file early if the target mesh format is already found
                if let Some(target) = self.target_id {
                    if format_map.contains_key(&target) {
                        break;
                    }
                };
            }

            // break read of file early if the target mesh format is already found
            if let Some(target) = self.target_id {
                if format_map.contains_key(&target) {
                    break;
                }
            }
        }

        debug!("{}", Self::formats_detected(&format_map));

        Ok(format_map)
    }

    /// Initialises the hints used to determing formatting types
    fn init_format_hints() -> [&'static dyn Fn(&str) -> bool; 3] {
        [
            &parsers::is_cuv_hint,
            &parsers::is_col_hint,
            &parsers::is_matrix_hint,
        ]
    }

    /// If a target is defined, make sure it is at least in the file
    fn ensure_format_contains_target(&self, format_map: &FormatMap) -> Result<()> {
        match self.target_id {
            None => Ok(()),
            Some(id) => {
                if format_map.contains_key(&id) {
                    Ok(())
                } else {
                    let mut msg = f!("FMESH {id} not found in meshtal file. ");
                    msg += &f!("Available tallies: {:?}", format_map.keys());
                    Err(anyhow!("{}", msg))
                }
            }
        }
    }

    /// Try to find a hint about what format the mesh could be
    fn get_format_hint(
        line: &str,
        hints: [&dyn Fn(&str) -> bool; 3],
        mesh_type: &Geometry,
    ) -> Option<Format> {
        if let Some(position) = hints.iter().position(|f| f(line)) {
            match position {
                // set the flag for format found in thes some variants here
                0 => Some(Format::CUV),
                1 => Some(Self::get_column_type(line)),
                2 => Some(Self::get_matrix_type(line, mesh_type)),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Checks the coordinate tag on the column hint
    fn get_column_type(line: &str) -> Format {
        if line.contains("Volume") {
            Format::CF
        } else {
            Format::COL
        }
    }

    /// Checks the coordinate tag on the matrix hint
    fn get_matrix_type(i: &str, geom: &Geometry) -> Format {
        // getting through the hint means starts with any of the coordinate tags
        // next() faster than nth(0)
        match i.chars().next().unwrap() {
            'X' => Format::JK,
            'R' => Format::JK,
            'T' => Format::IJ,
            'Y' => Format::IK,
            // 'Z' is ambiguous so need to check the geometry
            'Z' => match geom {
                Geometry::Rectangular => Format::IJ,
                Geometry::Cylindrical => Format::IK,
            },
            _ => unreachable!(),
        }
    }

    /// Checks the coordinate tag for cartesian or cylindrical geometry type
    fn get_geometry_type(line: &str) -> Geometry {
        match line.chars().next().unwrap() {
            'R' => Geometry::Cylindrical,
            'X' => Geometry::Rectangular,
            _ => panic!("Unable to find mesh type from {line}"),
        }
    }

    /// Constructs string of all found mesh formats
    fn formats_detected(format_map: &FormatMap) -> String {
        let mut s = "Formats found:".to_string();
        for (k, (format, mesh_type)) in format_map {
            s += &f!("\n  > Fmesh {k:<4}: {:?}, {:?}", mesh_type, format);
        }
        s
    }

    /// Parse the particle type line into the appropriate enum variant
    fn read_particle(mesh: &mut Mesh, line: &str) -> Result<()> {
        trace!("[Particle] {line}");
        let (_, particle) = parsers::first_word(line)
            .map_err(|_| anyhow!("Failed to parse {line} for particle type"))?;
        mesh.particle = Particle::try_from(particle)?;
        Ok(())
    }

    /// Parse the cylinder origin/axis/vec onto coordinate arrays
    fn read_origin_axs_vec(mesh: &mut Mesh, line: &str) -> Result<()> {
        trace!("[ori, axs, vec] {}...", &line[0..30]);
        let (i, origin) = parsers::origin(line)
            .map_err(|_| anyhow!("Failed to parse {line} for origin coordinates"))?;
        let (i, axis) =
            parsers::axis(i).map_err(|_| anyhow!("Failed to parse {line} for axis coordinates"))?;
        let (_, vec) = parsers::vec(i)
            .map_err(|_| anyhow!("Failed to parse {line} for vector coordinates"))?;

        mesh.origin = origin;
        mesh.axs = axis;
        mesh.vec = vec;
        Ok(())
    }

    /// Parse ijk bounds to f64 lists
    fn read_geometry_bounds(mesh: &mut Mesh, line: &str) -> Result<()> {
        trace!("[Geometry bounds] {}...", &line[0..30]);
        let (_, values) = parsers::geometry_bounds(line)
            .map_err(|_| anyhow!("Failed to parse i/j/k mesh bounds from\n{line}"))?;
        let n_bins: usize = values.len() - 1;

        // assign to the relevant mesh fields
        match line.chars().next().unwrap() {
            'X' | 'R' => {
                mesh.imesh = values;
                mesh.iints = n_bins;
            }
            'Y' => {
                mesh.jmesh = values;
                mesh.jints = n_bins;
            }
            'T' => {
                mesh.kmesh = values;
                mesh.kints = n_bins;
            }
            'Z' => match mesh.geometry {
                Geometry::Rectangular => {
                    mesh.kmesh = values;
                    mesh.kints = n_bins;
                }
                Geometry::Cylindrical => {
                    mesh.jmesh = values;
                    mesh.jints = n_bins;
                }
            },
            _ => bail!("No match for {line}"),
        }
        Ok(())
    }

    /// Parse energy/times to Group lists
    fn read_group_bounds(mesh: &mut Mesh, line: &str) -> Result<()> {
        trace!("[Group bounds] {}...", &line[0..30]);
        let (_, values) =
            parsers::group_bounds(line).map_err(|_| anyhow!("Problem parsing group bounds"))?;

        if line.starts_with("Energy") {
            mesh.emesh = values;
            mesh.eints = mesh.emesh.len() - 1;
        } else if line.starts_with("Time") {
            mesh.tmesh = values;
            mesh.tints = mesh.tmesh.len() - 1;
        } else {
            bail!(
                "Expected line {}... to start with \"Energy\" or \"Time\"",
                &line[0..20]
            )
        }
        Ok(())
    }
}

/// COL, sparse COL, and CF formats
impl MeshtalReader {
    /// parse column mesh tallies
    fn parse_column_format(
        &mut self,
        line: &str,
        header: &[&dyn Fn(&str) -> bool; 4],
    ) -> Result<()> {
        let mesh = self.mesh_list.last_mut().unwrap();

        // more efficient to focus on this very likely path from the full set
        match parsers::column_type_voxel(line) {
            IResult::Ok(v) => {
                let mut voxel = v.1;
                voxel.index = mesh.voxels.len();
                mesh.voxels.push(voxel);
                Ok(())
            }
            _ => Self::read_header_information(mesh, line, header),
        }
    }

    /// parse column header info such as particle, geometry, groups, etc...
    fn read_header_information(
        mesh: &mut Mesh,
        line: &str,
        header: &[&dyn Fn(&str) -> bool; 4],
    ) -> Result<()> {
        if let Some(position) = header.iter().position(|f| f(line)) {
            match position {
                // set the flag for format found in thes some variants here
                0 => Self::read_origin_axs_vec(mesh, line), // origin_axis_vec
                1 => Self::read_particle(mesh, line),       // particle
                2 => Self::read_geometry_bounds(mesh, line), // geometry bounds
                3 => Self::read_group_bounds(mesh, line),   // group bounds
                _ => unreachable!(),
            }
        } else {
            Ok(())
        }
    }
}

/// Matrix IJ, JK, IK formats
impl MeshtalReader {
    /// parse marix mesh tallies
    fn parse_matrix_format(
        &mut self,
        line: &str,
        matrix_hints: &[&dyn Fn(&str) -> bool; 7],
    ) -> Result<()> {
        let mesh = self.mesh_list.last_mut().unwrap();

        if let Some(position) = matrix_hints.iter().position(|f| f(line)) {
            match position {
                0 => Self::read_origin_axs_vec(mesh, line), // origin_axis_vec
                1 => Self::read_particle(mesh, line),       // particle
                2 => Self::read_geometry_bounds(mesh, line), // geometry bounds
                3 => Self::read_group_bounds(mesh, line),   // group bounds
                4 => {
                    trace!("[Matrix group] {}", line);
                    self.update_current_group(line)
                } // energy/time group tables marker
                5 => {
                    trace!("[Matrix table] {}", line);
                    self.update_current_table();
                    Ok(())
                } // new pair result/error tables to follow
                6 => Self::read_matrix_data(mesh, &mut self.tracked, line), // any string of whitespace separated numbers
                _ => unreachable!(),
            }
        } else {
            Ok(())
        }
    }

    /// Extract the data from rows/columns of the result/error tables
    fn read_matrix_data(mesh: &mut Mesh, tracked: &mut Tracked, line: &str) -> Result<()> {
        // first will be the k-ordinate before the values we care about
        let result: Vec<f64> = line
            .split_whitespace()
            .map(|s| {
                s.parse::<f64>()
                    .with_context(|| f!("Could not parse {s} to f64"))
                    .unwrap()
            })
            .skip(1) // ignore first entry
            .collect();

        // ignore matrix heading (just j-ordinate voxel centers)
        if Self::is_table_header(mesh, result.len()) {
            trace!("[Table header] {line}");
            tracked.row = 0; // reset the tracked row and return early
            return Ok(());
        }

        // single time bin only if no TMESH defined for the mesh
        if mesh.tmesh.is_empty() {
            tracked.time = 1;
        }
        // update last known row
        tracked.row += 1;

        // if scientific then this is the result table, so make a new voxel
        if line.contains('E') | line.contains('e') {
            trace!("[Result] {line}");
            Self::new_matrix_voxels(mesh, tracked, &result);
        }
        // otherwise this is the error table, so find it and update the voxel
        else {
            trace!("[Errors] {line}");
            // find voxel manually by index because looping over an ever-growing
            // list of voxels is a really terrible idea
            let current_row_index = Self::get_offset(mesh, tracked);

            for (col_idx, error) in result.into_iter().enumerate() {
                mesh.voxels[current_row_index + col_idx].error = error;
            }
        }

        Ok(())
    }

    /// Generate voxels for all results/errors in a table pair
    fn new_matrix_voxels(mesh: &mut Mesh, tracked: &mut Tracked, result: &[f64]) {
        for (column_idx, value) in result.iter().enumerate() {
            tracked.col = column_idx;

            // get the right global index for a given matrix type
            mesh.voxels.push(Voxel {
                index: Self::get_matrix_index(mesh, tracked),
                result: *value,
                error: 0.0, // default to zero until the errors table is parsed
            });
        }
    }

    /// Check number of values in a line against expected number of data columns
    ///
    /// note n_results discards the first match so should < columns if header
    ///     1.17        3.50        5.83
    /// |___ results=2 jints=3 => matrix header found
    /// 0.12 2.03496E-02 2.37144E-01 2.04930E-02
    /// |___ results=3 jints=3 => data to be recorded
    fn is_table_header(mesh: &Mesh, n_results: usize) -> bool {
        match mesh.format {
            Format::IJ => n_results < mesh.iints,
            Format::IK => n_results < mesh.iints,
            Format::JK => n_results < mesh.jints,
            _ => false,
        }
    }

    /// Offset to apply when finding the voxel index of error table values
    fn get_offset(mesh: &Mesh, tracked: &Tracked) -> usize {
        match mesh.format {
            Format::JK => {
                let base = mesh.voxels.len() - (mesh.jints * mesh.kints);
                base + (tracked.row - 1) * mesh.jints
            }
            Format::IK => {
                let base = mesh.voxels.len() - (mesh.iints * mesh.kints);
                base + (tracked.row - 1) * mesh.iints
            }
            Format::IJ => {
                let base = mesh.voxels.len() - (mesh.iints * mesh.jints);
                base + (tracked.row - 1) * mesh.iints
            }
            _ => panic!(),
        }
    }

    /// Find the index of a voxel based on tracked values
    fn get_matrix_index(mesh: &Mesh, tracked: &Tracked) -> usize {
        // number of columns will change depending on format
        // tracked columns are 0 indexed so no need to subtract 1
        let (i_idx, j_idx, k_idx) = match mesh.format {
            Format::IJ => (tracked.col, tracked.row - 1, tracked.table - 1),
            Format::IK => (tracked.col, tracked.table - 1, tracked.row - 1),
            Format::JK => (tracked.table - 1, tracked.col, tracked.row - 1),
            _ => panic!(),
        };

        let e_idx = tracked.erg - 1;
        let t_idx = tracked.time - 1;

        mesh.etijk_to_voxel_index(e_idx, t_idx, i_idx, j_idx, k_idx)
    }

    /// Groups are either energy or time, so deal with them as appropriate
    fn update_current_group(&mut self, line: &str) -> Result<()> {
        if line.starts_with("Energy") | line.starts_with("Total Energy") {
            self.update_current_energy();
            Ok(())
        } else if line.starts_with("Time") | line.starts_with("Total Time") {
            self.update_current_time();
            Ok(())
        } else {
            Err(anyhow!("Could not find group in {line}"))
        }
    }

    /// New energy group means increment it and reset the rest to 0
    fn update_current_energy(&mut self) {
        // increment energy
        self.tracked.erg += 1;
        // reset the time and table bins
        self.tracked.time = 0;
        self.tracked.table = 0;
        self.tracked.row = 0;
    }

    /// New time group means increment it and reset the tabel and row to 0
    fn update_current_time(&mut self) {
        // increment time bin;
        self.tracked.time += 1;
        // reset table index
        self.tracked.table = 0;
        self.tracked.row = 0;
    }

    /// New table dataset means increment it and reset the row to 0
    fn update_current_table(&mut self) {
        // increment table
        self.tracked.table += 1;
        // reset the row
        self.tracked.row = 0;
    }
}

/// Cell-under-Voxel formats
impl MeshtalReader {
    /// parse cell under voxel mesh tallies
    fn parse_cuv_format(
        &mut self,
        line: &str,
        cuv_hints: &[&dyn Fn(&str) -> bool; 6],
    ) -> Result<()> {
        // more efficient to focus on this very likely path from the full set
        match parsers::cuv_type_voxel(line) {
            IResult::Ok((_, (voxel, cell_data))) => self.process_cuv_data(voxel, cell_data),
            _ => self.read_cuv_header(line, cuv_hints),
        }
    }

    /// Parse the geometry/group bounds, patricle type, voidoff status, etc...
    fn read_cuv_header(
        &mut self,
        line: &str,
        cuv_hints: &[&dyn Fn(&str) -> bool; 6],
    ) -> Result<()> {
        let mesh = self.mesh_list.last_mut().unwrap();

        if let Some(position) = cuv_hints.iter().position(|f| f(line)) {
            match position {
                // set the flag for format found in thes some variants here
                0 => Self::read_origin_axs_vec(mesh, line), // origin_axis_vec
                1 => Self::read_particle(mesh, line),       // particle
                2 => Self::read_geometry_bounds(mesh, line), // geometry bounds
                3 => Self::read_group_bounds(mesh, line),   // group bounds
                4 => self.read_voidoff_status(line),        // energy/time group tables marker
                5 => self.read_material_array(line), // new pair result/error tables to follow
                _ => unreachable!(),
            }
        } else {
            Ok(())
        }
    }

    /// Parse voidoff status into an explicit enum variant
    fn read_voidoff_status(&mut self, line: &str) -> Result<()> {
        trace!("[Voidoff] {line}");
        let (_, status) = parsers::void_record_status(line)
            .map_err(|_| anyhow!("Failed to parse status from:\n \"{line}\""))?;
        trace!("  |_ {:?}", status);
        self.void_record = status;
        Ok(())
    }

    /// Parse the "material cells per voxel" array into a vector
    fn read_material_array(&mut self, line: &str) -> Result<()> {
        match self.void_record {
            VoidRecord::On => (),
            VoidRecord::Off => {
                if !parsers::contains_alphabetic(line) {
                    let (_, mut values) = parsers::vector_of_u32(line).map_err(|_| {
                        anyhow!("Failed to parse material array from:\n \"{line}\"")
                    })?;
                    self.mcpv.append(&mut values);
                }
            }
        }
        Ok(())
    }

    /// Check if the current cell data are part of the previous voxel
    fn is_same_coordinates(old: &Option<CellData>, new: &CellData) -> bool {
        match old {
            None => false,
            Some(old) => {
                old.energy == new.energy
                    && old.time == new.time
                    && old.i_coord == new.i_coord
                    && old.j_coord == new.j_coord
                    && old.k_coord == new.k_coord
            }
        }
    }

    /// Main logic for processing the cuv data, making new voxels and filling in
    /// the gaps as needed
    fn process_cuv_data(&mut self, voxel: Voxel, cell_data: CellData) -> Result<()> {
        // first time this runs we need to sort Number_of_material_cells_per_voxel
        // by the cell index
        self.sort_mcpv_array()?;

        let mesh: &mut Mesh = self.mesh_list.last_mut().unwrap();

        trace!(" -- {voxel}");

        match mesh.voxels.last() {
            // compare to previous coordinates for multiple cells under same voxel
            // voxel is the new one, mesh.voxels.last() is the previous one read
            Some(_) if Self::is_same_coordinates(&self.previous_cell, &cell_data) => {
                // trace!("Voxel same as previous");
                let weight = cell_data.volume / Self::total_voxel_volume(mesh, mesh.voxels.len());
                let current_voxel = mesh.voxels.last_mut().unwrap();

                // need to check for -ve results, which can happen for CuV for some reason
                let (result, error) = match voxel.result.is_sign_negative() {
                    false => (voxel.result, voxel.error),
                    true => (0.0, 0.0),
                };

                current_voxel.result += weight * result;
                current_voxel.error += (weight * error).powi(2);
            }
            // otherwise None and we need a new voxel
            _ => {
                // trace!("New voxel");
                // if the void_record=off then we need to fill in the missing voxels
                // now before adding the one that was read
                if self.void_record == VoidRecord::Off {
                    let idx = mesh.voxels.len() % self.mcpv.len();
                    if self.mcpv[idx] == 0 {
                        let distance = Self::next_nonzero_element(mesh, &self.mcpv);
                        for _ in 0..distance {
                            mesh.voxels.push(Voxel {
                                index: mesh.voxels.len(),
                                result: 0.0,
                                error: 0.0,
                            });
                        }
                    }
                }

                // need to check for -ve results, which can happen for CuV for some reason
                let (result, error) = match voxel.result.is_sign_negative() {
                    false => (voxel.result, voxel.error),
                    true => (0.0, 0.0),
                };

                // then in all cases add the parsed data to a new voxel
                let weight = cell_data.volume / Self::total_voxel_volume(mesh, mesh.voxels.len());

                mesh.voxels.push(Voxel {
                    index: mesh.voxels.len(),
                    result: weight * result,
                    error: (weight * error).powi(2),
                })
            }
        }

        trace!("{}", mesh.voxels.last().unwrap());
        self.previous_cell = Some(cell_data.clone());

        Ok(())
    }

    /// Need to reorder the mcpv array as it is annoyingly written x,y,z in
    /// contrast to the actual data
    fn sort_mcpv_array(&mut self) -> Result<()> {
        let mesh: &mut Mesh = self.mesh_list.last_mut().unwrap();
        if self.void_record == VoidRecord::Off && mesh.voxels.is_empty() {
            // check and make sure the number matches just to be sure
            if mesh.iints * mesh.jints * mesh.kints != self.mcpv.len() {
                return Err(anyhow!(
                            "Expected voxel group size {} != length of material_cells_per_voxel {}\n {:?}\n...\n{:?}",
                            mesh.iints * mesh.jints * mesh.kints,
                            self.mcpv.len(),
                            &self.mcpv[..20],
                            &self.mcpv[self.mcpv.len() - 20..]
                        ));
            }
            // reorder the mcpv array as it is annoyingly written x,y,z
            self.mcpv = self
                .mcpv
                .iter()
                .enumerate()
                .map(|(i, _)| self.mcpv[mesh.voxel_index_to_cell_index(i)])
                .collect();

            trace!("mcpv array reordered to:\n {:?}", self.mcpv);
        }
        Ok(())
    }

    /// Find the volume of a voxel for appropriate scaling of flux data
    fn total_voxel_volume(mesh: &Mesh, index: usize) -> f64 {
        let (_, _, i, j, k) = mesh.voxel_index_to_etijk(index);

        // find the expected volume [cm2]
        match mesh.geometry {
            Geometry::Rectangular => {
                (mesh.imesh[i + 1] - mesh.imesh[i])
                    * (mesh.jmesh[j + 1] - mesh.jmesh[j])
                    * (mesh.kmesh[k + 1] - mesh.kmesh[k])
            }
            Geometry::Cylindrical => {
                let dr = mesh.imesh[i + 1] - mesh.imesh[i];
                let dz = mesh.jmesh[j + 1] - mesh.jmesh[j];
                let dt = mesh.kmesh[k + 1] - mesh.kmesh[k];
                dz * ((std::f64::consts::PI * dr * dr) / dt)
            }
        }
    }

    /// Look ahead to see how many filler void voxels are needed for the
    /// VoidRecord::Off status, where these are otherwise left out of the data
    fn next_nonzero_element(mesh: &Mesh, vector: &[u32]) -> usize {
        // remainder of current index / results per_group
        let skip_value = mesh.voxels.len() % vector.len();

        // some if fine, none if zeros at the end
        let distance = vector.iter().skip(skip_value).position(|&x| x > 0);

        // todo check for if some but last in group

        // either include any zero elements from the start of the next group or
        // detect if reached the end, where you only add the remaining voxels
        distance.unwrap_or({
            let mut base_addon = vector.len() - skip_value;
            let max_voxels = mesh.ebins() * mesh.tbins() * mesh.iints * mesh.jints * mesh.kints;
            // add any zeros from the next set if not at the end of the mesh
            if mesh.voxels.len() + base_addon != max_voxels {
                base_addon += vector.iter().position(|&x| x > 0).unwrap();
            }
            base_addon
        })
    }

    /// For VoidRecord::Off there may be void voxels after the last data output
    /// so this will fill those in to complete the full mesh
    fn complete_cuv_voxels(&mut self) {
        self.mesh_list.iter_mut().for_each(|m| {
            if m.format == Format::CUV {
                debug!("Cleaning up CuV data");
                // fix existing voxels
                for v in &mut m.voxels {
                    v.error = if v.error > 0.0 { v.error.sqrt() } else { 0.0 };
                }

                // add any trailing empty voxels
                let n_actual = m.voxels.len();
                let n_target = m.n_voxels_expected();

                if n_actual != n_target {
                    trace!("Generating trailing void voxels");
                    // just add a bunch of zero result voxels on the end
                    for _ in 0..(n_target - n_actual) {
                        m.voxels.push(Voxel {
                            index: m.voxels.len(),
                            result: 0.0,
                            error: 0.0,
                        });
                    }
                }
            }
        })
    }
}

/// Convenience type for keeping track of mesh formatting
///
/// Key is the mesh tally number e.g. Fmesh204 => 204
/// Value is tuple of formatting and mesh geometry e.g. (COL, Rectangular)
type FormatMap = HashMap<u32, (Format, Geometry)>;

/// Explicit states for the CuV 'Voidoff=' card
///
/// The CuV patch contains an option to omit any flux results for void areas,
/// since these voxels will not contribute to activation. The state must be
/// known in order to fill in any missing voxels for the VoioRecord::Off case.
#[derive(Debug, PartialEq)]
pub enum VoidRecord {
    /// Void cells are included in output data
    On,
    /// Void cells are excluded in output data
    Off,
}

/// Additional information for CuV data parsing
///
/// CuV contains a lot of additional information. Keeping track of the volume
/// and coordiante information is necessary for the reader, though only two
/// instances are around at any given time.
///
/// For completeness and future features, all data are kept though some are not
/// currently used. For example, the cell number will be relevant for sub-voxel
/// resolution when plotting.
#[derive(Debug, Clone)]
pub struct CellData {
    /// Eneergy group
    pub energy: Group,
    /// Time group
    pub time: Group,
    /// i coordinate at centre of voxel
    pub i_coord: f64,
    /// j coordinate at centre of voxel
    pub j_coord: f64,
    /// k coordinate at centre of voxel
    pub k_coord: f64,
    /// cell number  
    pub cell: u32,
    /// material number  
    pub material: u32,
    /// material density  
    pub density: f64,
    /// cell volume  
    pub volume: f64,
}

/// Tracked values for matrix-format tables
///
/// For scalability the parser must read line by line. This is an issue for
/// matrix format outputs as we need to know what time, energy, etc.. a given
/// table corresponds to.
///
/// This struct keeps track of the last known group so that any given set of
/// results or errors can be assigned to the correct voxels.
#[derive(Debug, Default)]
struct Tracked {
    /// tracks current energy bin
    erg: usize,
    /// tracks current time bin
    time: usize,
    /// tracks current primary group for OUT= card
    table: usize,
    /// tracks current rows for OUT= card
    row: usize,
    /// tracks current columns for OUT= card
    col: usize,
}

impl Tracked {
    /// Set all tracked values back to 0
    fn reset(&mut self) {
        self.erg = 0;
        self.time = 0;
        self.table = 0;
        self.row = 0;
        self.col = 0;
    }
}
