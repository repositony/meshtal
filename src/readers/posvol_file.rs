//! Reader for plot_fmesh_xxx.bin binary files
//!
//! For generating fine cell-under-voxel plot meshes from the coarse mesh data.
//! The file is binary with a sequence of u32/i32 integers (whatever the fortran
//! default is), with probably little endian on our sytems and in 4-byte chunks.  

// standard library
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// crate modules
use crate::posvol::*;

// external crates
use anyhow::{anyhow, Result};
use bincode::deserialize;

/// A simple reader for a UKAEA CuV posvol binary
#[derive(Debug, Default)]
pub struct PosvolReader {}

impl PosvolReader {
    /// Just calls Default::default(), nothing special to be initialised
    pub fn new() -> Self {
        Default::default()
    }

    pub fn parse(&self, path: &Path) -> Result<Posvol> {
        let mut reader = Self::init_reader(path);

        // `size_of` is less error prone but could just be 4
        let mut buffer = [0u8; std::mem::size_of::<i32>()];

        // read the first value, should be 24
        reader.read_exact(&mut buffer)?;
        if i32::from_ne_bytes(buffer) != 24 {
            return Err(anyhow!(
                "Expected first value to be 24, found {}",
                i32::from_ne_bytes(buffer)
            ));
        }

        // get the actual useful values, should be 6 of them
        let mut dim_buffer = [0u8; 6 * std::mem::size_of::<i32>()];
        reader.read_exact(&mut dim_buffer)?;

        let dimensions: Dimensions = deserialize(&dim_buffer)?;

        // skip the bookend '24'
        reader.read_exact(&mut buffer)?;

        // next value is bytes to follow, use to define
        reader.read_exact(&mut buffer)?;

        // check to make sure it is the expected value
        if i32::from_ne_bytes(buffer) != dimensions.cell_array_byte_length() {
            return Err(anyhow!(
                "Expected {} found {}",
                dimensions.cell_array_byte_length(),
                i32::from_ne_bytes(buffer)
            ));
        }

        let n_subvoxels = dimensions.number_of_subvoxels();
        let mut posvol = Posvol {
            dimensions,
            cells: Vec::with_capacity(n_subvoxels),
        };

        for _ in 0..posvol.dimensions.number_of_cells() {
            reader.read_exact(&mut buffer)?;
            posvol.cells.push(i32::from_ne_bytes(buffer));
        }

        Ok(posvol)
    }

    pub fn init_reader(path: &Path) -> BufReader<File> {
        BufReader::new(File::open(path).expect("Could not open file"))
    }
}
