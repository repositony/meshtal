// internal modules
use crate::mesh::Geometry;
use crate::point::Point;
use crate::readers::parsers;
use crate::utils::*;

// standard library
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

// external crates
use anyhow::{Context, Result};
use log::trace;
use nom::IResult;

/// A simple reader for a points file
#[derive(Debug, Default)]
pub struct PointsFileReader;

impl PointsFileReader {
    /// Just calls Default::default(), nothing special to be initialised
    pub fn new() -> Self {
        Default::default()
    }

    /// Parses all mesh data from a mcnp meshtal file
    ///
    /// Any whitespace at the beginning of the line is trimmed. Assumes
    /// coordinates given are cartesian xyz unless specified otherwise.
    pub fn parse(&mut self, path: &Path) -> Result<Vec<Point>> {
        // parse the data depending on Format type
        let file = File::open(path).with_context(|| f!("Could not open {}", path.display()))?;
        let reader = BufReader::new(file);

        // Default to the assumption that these are XYZ coordinates
        let mut last_known_keyword = Geometry::Rectangular;
        let mut point_list = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim(); // shadow needed to get around borrow

            // skip empty lines
            if line.is_empty() {
                continue;
            }

            // skip over any comment lines
            if parsers::is_points_file_comment(line) {
                trace!("[Comment] {line}");
                continue;
            }

            // check for keywords first
            if let IResult::Ok((_, c)) = parsers::points_file_keyword(line) {
                last_known_keyword = c;
                trace!("[Keyword] {line}");
                continue;
            }

            // otherwise bother to parse the line for point data
            match &mut parsers::points_file_point(line) {
                IResult::Ok(data) => {
                    let (_, point) = data;
                    point.coordinate_type = last_known_keyword;
                    point_list.push(point.to_owned());
                    trace!("[ Point ] {line}")
                }
                _ => trace!("Failed to parse \"{line}\" to a Point"),
            }
        }

        Ok(point_list)
    }
}
