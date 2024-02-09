//! Module for voxel-related data and implementations

// internal modules
use crate::utils::*;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Representation of a single voxel in the mesh
///
/// The global `index` of the voxel is included to maintain consistency between
/// output [Format](crate::mesh::Format) variants. Parsing line-by-line rather
/// than trying to load an entire file into memory leaves the voxels in an
/// inconsistent order for several formats.
///
/// The minimum information required would be just the result and uncertainty
/// (16 Bytes), and the maximum would include the cooridinates and energy/time
/// groups (80 Bytes).
///
/// The current implementation is a compromise between the two at 24 Bytes, and
/// all other values may be derived from the [Mesh](crate::mesh::Mesh) given the
/// voxel index.
///
/// Several operators are implemented for convenience where it makes sense,
/// including Addition (`+`, `+=`), Subtraction (`-`, `-=`),
/// Multiplication (`*`, `*=`), and Division (`/`, `/=`).
///
/// In all cases, the LHS index is taken, and the RHS may be either another
/// [Voxel] or anything that can be converted into an `f64` primitive.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Voxel {
    /// Global voxel index
    pub index: usize,
    /// Tallied voxel result
    pub result: f64,
    /// Relative error on result
    pub error: f64,
}

impl Voxel {
    /// Returns the absolute error for the voxel
    ///
    /// Example:
    ///
    ///```rust
    /// # use meshtal::mesh::Voxel;
    /// let voxel = Voxel {
    ///     result: 50.0,
    ///     error: 0.10,
    ///     ..Default::default()
    /// };
    ///
    /// /// 10% relative error => 50.0 +/-5.0
    /// assert_eq!(voxel.absolute_error(), 5.0);
    /// ```
    ///
    pub fn absolute_error(&self) -> f64 {
        self.result * self.error
    }
}

impl std::fmt::Display for Voxel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:<5.0}{:>13}{:>13}",
            self.index,
            self.result.sci(5, 2),
            self.error.sci(5, 2)
        )
    }
}

impl Add<Self> for Voxel {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let result = self.result + rhs.result;
        let error = (self.absolute_error().powi(2) + rhs.absolute_error().powi(2)).sqrt() / result;

        Self {
            index: self.index,
            result,
            error,
        }
    }
}

impl AddAssign<Self> for Voxel {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T> Add<T> for Voxel
where
    T: Into<f64>,
{
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        Self {
            index: self.index,
            result: self.result + rhs.into(),
            error: self.error,
        }
    }
}

impl<T> AddAssign<T> for Voxel
where
    T: Into<f64>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs.into();
    }
}

impl Sub<Self> for Voxel {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let result = self.result - rhs.result;
        let error = (self.absolute_error().powi(2) + rhs.absolute_error().powi(2)).sqrt() / result;

        Self {
            index: self.index,
            result,
            error,
        }
    }
}

impl SubAssign<Self> for Voxel {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T> Sub<T> for Voxel
where
    T: Into<f64>,
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        Self {
            index: self.index,
            result: self.result - rhs.into(),
            error: self.error,
        }
    }
}

impl<T> SubAssign<T> for Voxel
where
    T: Into<f64>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs.into();
    }
}

impl Mul<Self> for Voxel {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            index: self.index,
            result: self.result * rhs.result,
            error: (self.error.powi(2) + rhs.error.powi(2)).sqrt(),
        }
    }
}

impl MulAssign<Self> for Voxel {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T> Mul<T> for Voxel
where
    T: Into<f64>,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            index: self.index,
            result: self.result * rhs.into(),
            error: self.error,
        }
    }
}

impl<T> MulAssign<T> for Voxel
where
    T: Into<f64>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs.into();
    }
}

impl Div<Self> for Voxel {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            index: self.index,
            result: self.result / rhs.result,
            error: (self.error.powi(2) + rhs.error.powi(2)).sqrt(),
        }
    }
}

impl DivAssign<Self> for Voxel {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T> Div<T> for Voxel
where
    T: Into<f64>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Self {
            index: self.index,
            result: self.result / rhs.into(),
            error: self.error,
        }
    }
}

impl<T> DivAssign<T> for Voxel
where
    T: Into<f64>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs.into();
    }
}

/// Energy/Time groups are either `Total` or an upper bin edge
///
/// For energies, the meshtal outputs always define `emesh` bounds so there is
/// always at least one group. For a single energy bin this becomes a
/// [Group::Total] no matter the energy.
///
/// | Energy bounds | Groups                            |
/// | ------------- | --------------------------------- |
/// | None          | Total ("0.0 1e36" in output file) |
/// | 0.0 100       | Total                             |
/// | 0.0 20 100    | Value(20.0), Value(100.0), Total  |
///
/// For times, the meshtal outputs will exclude any `tmesh` bounds entirely if
/// none are given in the input file. In the case that there are no time bins
/// or just one, the group is set to [Group::Total] for consistency.
///
/// | Time bounds   | Groups                            |
/// | ------------- | --------------------------------- |
/// | None          | Total (Nothing in output file)    |
/// | 0.0 1e16      | Total                             |
/// | 0.0 1e16 1e36 | Value(1e16), Value(1e36), Total   |
///
#[derive(Debug, PartialEq, Clone, Copy, PartialOrd)]
pub enum Group {
    /// The 'Total' bin group
    Total,
    /// The upper edge of a bin
    Value(f64),
}

impl Group {
    #[inline]
    /// Check if the Group is the `Total` variant
    ///
    /// ```rust
    /// # use meshtal::mesh::Group;
    /// let group = Group::Total;
    /// assert_eq!(group.is_total(), true);
    ///
    /// let group = Group::Value(2.0);
    /// assert_eq!(group.is_total(), false);
    /// ```
    pub const fn is_total(&self) -> bool {
        matches!(*self, Self::Total)
    }

    #[inline]
    /// Check if the Group is the `Value` variant
    ///
    /// ```rust
    /// # use meshtal::mesh::Group;
    /// let group = Group::Total;
    /// assert_eq!(group.is_value(), false);
    ///
    /// let group = Group::Value(2.0);
    /// assert_eq!(group.is_value(), true);
    /// ```
    pub const fn is_value(&self) -> bool {
        !self.is_total()
    }
}

impl std::fmt::Display for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Value(value) => write!(f, "{}", value.sci(5, 2)),
            Self::Total => write!(f, "Total"),
        }
    }
}

/// Convenience structure for collecting voxel coordiante information
///
/// The coordinate data of each [Voxel] are a complete set. Every coordiante is
/// derivable from the global voxel index given knowledge of the
/// [Mesh](crate::mesh::Mesh) fields. It is therefore primarily used in
/// [Mesh](crate::mesh::Mesh) methods.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct VoxelCoordinate {
    /// Energy group (MeV)
    pub energy: Group,
    /// Time group (shakes)
    pub time: Group,
    /// i coordinate at the voxel centre
    pub i: f64,
    /// j coordinate at the voxel centre
    pub j: f64,
    /// k coordinate at the voxel centre
    pub k: f64,
}

impl Default for VoxelCoordinate {
    fn default() -> Self {
        Self {
            energy: Group::Total,
            time: Group::Total,
            i: 0.0,
            j: 0.0,
            k: 0.0,
        }
    }
}
