# Meshtal

[![Build Status][test-img]][test-url]
[![Documentation][doc-img]][doc-url]

[test-img]: https://github.com/repositony/meshtal/actions/workflows/build_tests.yml/badge.svg
[test-url]: https://github.com/repositony/meshtal/actions/workflows/build_tests.yml

[doc-img]: https://github.com/repositony/meshtal/actions/workflows/documentation.yml/badge.svg
[doc-url]: https://repositony.github.io/meshtal/index.html

**A collection of useful analysis tools for interacting with MCNP meshtal files**

*This is a pre-release version for testing and development. The core library is written, so it is just matter of finding the time to extend all the high-level features and functionality*

Full library documentation [here](https://repositony.github.io/meshtal/index.html)

## Install

Direct from github:

```shell
cargo install --git https://github.com/repositony/meshtal.git
```

All executables are under `~/.cargo/bin/`, which should already be in your path
after installing Rust.

<details>
  <summary>Click here if you have never used Rust</summary>

If you have never used the Rust programming language, the toolchain is easily
installed from the [official website](https://www.rust-lang.org/tools/install)

```shell
curl https://sh.rustup.rs -sSf | sh
```

This should have added `source $HOME/.cargo/env` to the bash profile, so update
your environment with `source ~/.bashrc`.

</details>

## Overview

The crate contains several command line tools for quickly performing common
tasks relating to MCNP meshes. More may be added based on time/usefulness.

| Command line   | Description                                             |
| -------------- | ------------------------------------------------------- |
| `mesh2vtk`     | Convert any meshtal tally to various VTK formats        |
| `mesh2ww`      | Convert any meshtal tally to a mesh-based weight window |
| `pointextract` | Extract voxel results for any point(s) in a mesh        |
| `splitmesh`    | Split meshtal tallies into individual files             |
| `posvol`       | Inspect and convert binary UKAEA CuV posvol files       |

All tools are fully documented with detailed `--help` messages, including
examples for common use cases.

### Supported output formats

For more detail, see the `OUT` keyword for the `FMESH` card definition in
the [MCNPv6.2](https://mcnp.lanl.gov/pdf_files/TechReport_2017_LANL_LA-UR-17-29981_WernerArmstrongEtAl.pdf)
or [MCNPv6.3](https://mcnpx.lanl.gov/pdf_files/TechReport_2022_LANL_LA-UR-22-30006Rev.1_KuleszaAdamsEtAl.pdf)
user manuals.

| Output format | Supported? | Description                                         |
| ------------- | ---------- | --------------------------------------------------- |
| COL           | Yes        | Column data (MCNP default)                          |
| CF            | Yes        | Column data including voxel volume                  |
| IJ            | Yes        | 2D matrix of I (col) and J (row) data, grouped by K |
| IK            | Yes        | 2D matrix of I (col) and K (row) data, grouped by J |
| JK            | Yes        | 2D matrix of J (col) and K (row) data, grouped by I |
| CUV (UKAEA)   | Yes        | UKAEA Cell-under-Voxel column data                  |
| NONE          | N/A        | `NONE` or unknown output format                     |

Once I get my paws on MCNPv6.3 this will be extended to include the new
COLSCI, CFSCI, and XDMF/HDF5 formats.

### Supported mesh geometries

Currently spherical meshes are not supported because barely anyone knows
about them, let alone uses them. They are currently a low priority.

| Mesh geometry | Supported? | MCNP designators |
| ------------- | ---------- | ---------------- |
| Rectangular   | Yes        | rec, xyz         |
| Cylindrical   | Yes        | cyl, rzt         |
| Spherical     | No         | sph, rpt         |

## Command line tools

### Mesh2vtk tool

Convert mesh tallies to VTK formats for plotting

```text
Usage: mesh2vtk <meshtal> <number> [options]
```

<details>
  <summary>Click to see detailed examples</summary>

Help is printed with the `-h` flag, and `--help` will show examples, default
values, examples, and any important behaviour.

By default, the results of every time/energy bin are extracted

#### How to include errors

Corresponding uncertainty meshes are optional in case of large meshtal files.

```bash
# Extract every energy and time group, with corresponding error meshes
mesh2vtk /path/to/meshtal.msht 104 --errors
```

#### How to only use the 'Total' energy/time group?

Often only the `Total` energy/time bins are of interest, and a quick way of
only converting this subset is provided.

```bash
# Extract only the 'Total' energy and time groups
mesh2vtk /path/to/meshtal.msht 104 --total
```

#### How to choose specific energy/time groups

If specific energy or time groups are required,

```bash
# Extract specific energy and time groups
mesh2vtk /path/to/meshtal.msht 104    \
        --energy 1.0 1e5              \
        --time 1e12 total
```

For intuitive use, the groups correspond with values defined on the `EMESH`
and `TMESH` cards.

*Note: mesh tallies label groups by the upper bounds defined on EMESH/TMESH
cards. i.e the energy group `1.0` corresponds to the `0.0>=E<1.0` bin,
though in reality 1 MeV particle would end up in the next group up.*

#### How to rescale all values

Mcnp normalises everything so it is often the case that the results must be
rescaled to provide physical values.

```bash
# Rescale the result by a constant multiplier, e.g 6.50E+18 neutrons/s
mesh2vtk /path/to/meshtal.msht 104 --scale 6.50E+18
```

Mesh rotations are a work in progress.

#### How to plot cylindrical meshes

There is no VTK representation of cylindrical meshes, so an unstructured
mesh is generated from verticies based on the RZT bounds.

Unfortunately, this can result in "low-resolution" plots for meshes with
few theta bins. The number of theta bins can be increased to round off these
edges. This simply subdivides the voxels by an integer number of theta bins.

![Cylindrical mesh resolution option](https://github.com/repositony/meshtal/blob/main/data/assets/cylindrical_mesh_resolution.png)

```bash
# Subdivide voxels into 3 to round off cylindrical unstructured mesh
mesh2vtk /path/to/meshtal.msht 104 --resolution 3
```

Note that this will increase the file size and memory usage significantly
in some cases.

#### How to change the output file name

By default the file prefix is `fmesh`, so the output files will be
`fmesh_<number>.vtk`. This may be changed as needed.

```bash
# Change the output name to `myvtk_104.vtr`
mesh2vtk /path/to/meshtal.msht 104 --output myvtk
```

#### How to choose a Vtk format

Most useful may be the ability to decide on output formats. XML and legacy
formats are supported, with both ascii and binary variants.

```bash
# Output as a binary vtk with legacy formatting
mesh2vtk /path/to/meshtal.msht 104 --format legacy-binary
```

#### How to specify compression and byte order

For more advanced usage, things like byte ordering and xml compression
methods are also configurable.

```bash
# Output as an xml using lzma and setting the byte order to big endian
mesh2vtk /path/to/meshtal.msht 104      \
        --format xml                    \
        --compresser lzma               \
        --endian big-endian
```

*Note - [VisIt](https://visit-dav.github.io/visit-website/index.html) only
reads big-endian, but most sytems are natively little-endian. For personal
convenience the default is big, but I am open to arguments for little endian
as the default.*

</details>

### Mesh2ww tool

Conversion of meshes to MCNP mesh-based global weight windows

```bash
Usage: mesh2ww <meshtal> <number> [options]
```

<details>
  <summary>Click to see detailed examples</summary>

Converts a mesh tally of any type to a weight window using the magic method
with configurable de-tuning options.

```bash
# Single particle type weight window
Usage: mesh2ww <meshtal> <number> [options]

# Multiple particle types
Usage: mesh2ww <meshtal> <number> [options] + <meshtal> <number> [options]
```

Help is printed with the `-h` flag, and `--help` will show examples, default
values, examples, and any important behaviour.

#### Tuning weights

Typical usage will generally define a de-tuning factor (`-p`/`--power`) and
possibly a relative error cutoff (`-e`/`--error`) for generating weights.

```bash
mesh2ww run0.msht 104 --power 0.70 --error 0.25
```

The `--power` value modifies calculated weights by `w => w^(power)`, which
helps with softening extreme values. Any voxels with errors above `--error`
(>25% in this case) continue to use analogue transport until the uncertainty
imporves.

#### Renaming output files

Of course, the weight window file may be renamed as needed:

```bash
mesh2ww run0.msht 104 --output mywwmesh.wwinp
```

#### Simplified weight window

It is often desirable to simply generate a global weight window mesh using
only the 'Total' group rather than every explicit energy/time group.

```bash
mesh2ww run0.msht 104 --total
```

This is probably the recommended use case for any finely binned groups, as
nobody should really be trying to optimise for every energy in a 175-group
mesh anyway.

#### Re-scale weights

Generated weights are typically normalised to the total flux for each group.
These may be rescaled by a constant multiplier.

```bash
# Multiply all normalised weights by x2.5
mesh2ww run0.msht 104 --scale 2.5
```

#### Advanced de-tuning

For fine control, the `--power` and `--error` parameters may be set
explicitly for every unique group.

For example, if a mesh had 3 energy groups at 1.0 MeV, 10.0 MeV, and
100.0 MeV, the power factor for each may be set to 0.8, 0.7, and 0.65
respectively.

```bash
# Set energy group power factors individually
mesh2ww run0.msht 104 --power 0.8 0.7 0.65
```

Of course this applies to time bins also. To set values for all unique
groups, the values must be given in order.

For example, a mesh with 3x energy groups and 2x time groups:

```text
Energy 1.0        Power
    Time 1e10      0.9
    Time 1e20      0.7
Energy 10.0
    Time 1e10      0.8
    Time 1e20      0.8
Energy 100.0
    Time 1e10      0.6
    Time 1e20      0.5
```

```bash
# Set energy group power factors individually
mesh2ww run0.msht 104 --power 0.9 0.7 0.8 0.8 0.6 0.5
```

#### Multi-particle weight windows

Multiple tallies may be combined for weight windows covering multiple
particle types. This may be achieved using the `+` operator.

The usage is as simple as combining multiple argument sets with `+` as the
delimiter.

```bash
mesh2ww <meshtal> <number> [options] +      \
        <meshtal> <number> [options] +      \
        <meshtal> <number> [options]
```

For example: 'NP_tallies.msht' contains neutron (FMESH14:n) and photon
(FMESH24:p) tallies, and 'E_tallies.msht' contains an electron (FMESH34:e)
tally.

If all of these are the same geometry, they may be combined with all the
usual optionas applied to each weight set individually:

```bash
mesh2ww NP_tallies.msht 14                   +      \
        NP_tallies.msht 24 -p 0.8 -e 0.15    +      \
        E_tallies.msht  34 --total                  \
```

</details>

### Pointextract tool

Extracts point data from any mesh in a meshtal file.

```text
Usage: pointextract <meshtal> <number> [options]
```

<details>
  <summary>Click to see detailed examples</summary>

Help is printed with the `-h` flag, and `--help` will show examples, default
values, examples, and any important behaviour.

#### Definition of a 'point'

A `Point` in may be defined three ways. Using cartesian `xyz` as an example:

- (x, y, z)
- (energy, x, y, z)
- (energy, time, x, y, z)

where are `energy`/`time` groups are either a value or 'total', and `x`,
`y`, `z` are the location to search for.

The 'Total' group is used by default for omitted `energy` and `time` groups.

Note that no matter what coordinates are provided, they are converted into
the appropriate coordinate system in the background. For example, `xyz`
coordinates are converted to `rzt` if the target mesh is cylindrical.

Points can be retrieved in two ways:

- Single point via the `-p`/`--point` argument
- Multiple points via the `-f`/`--file` argument

#### Single point (--point / --type)

To quickly get a single point:

```bash
# Find the point at (x,y,z) = (1.0, 2.0, 3.0)
pointextract /path/to/meshtal.msht 104 -p 1.0 2.0 3.0

# Find the same point, but for the 100.0 MeV energy group
pointextract /path/to/meshtal.msht 104 -p 1.0E+02 1.0 2.0 3.0

# Find the same point, but for the 'Total' energy group and 6.0e+15 time group
pointextract /path/to/meshtal.msht 104 -p total 6e+15 1.0 2.0 3.0
```

This works for both the `Rectangular` and `Cylindrical` mesh types.

Coordinates default to being XYZ cartesian, but the `--type` argument allows
this to be specified explicitly

```bash
# Find the point at (r,z,t) = (1.0, 2.0, 90.0)
pointextract /path/to/meshtal.msht 104 -p 1.0 2.0 0.53 --type rzt
```

#### Multiple points (--file)

The input file (default `points.txt`) is interpreted with the following
rules for a line:

| Example line               | Interpretation           |
| -------------------------- | ------------------------ |
| Starts with `#`            | comment                  |
| `rzt`, `cyl`, `xyz`, `rec` | geometry keyword         |
| 1.0 2.0 3.0                | i, j, k                  |
| 1e2  1.0 2.0 3.0           | energy, i, j, k          |
| 1e2 total 1.0 2.0 3.0      | energy, time, i, j, k    |

Anything else is ignored. For an example file see `data/points.txt`, though
a simplified input is shown below.

```bash
# this is a line comment in points.txt

xyz                         # points below explicitly interpreted as cartesian
1.0 5.0 7.0                 # 'Total' energy, 'Total' time, (x, y, z)
total 1.0 5.0 7.0           # 'Total' energy, 'Total' time, (x, y, z)
total total 1.0 5.0 7.0     # 'Total' energy, 'Total' time, (x, y, z)

rzt                         # points below explicitly interpreted as cylindrical
4.0 1.0 5.0 0.5             #  4 MeV  energy, 'Total' time, (r, z, t)
4.0 1e16 1.0 5.0 0.5        #  4 MeV  energy,  1e16   time, (r, z, t)
```

This is used as

```bash
# Find all points in file
pointextract /path/to/meshtal.msht 104 --file points.txt
```

It is fine to mix and match coordinates in the same file because all points
are converted into the appropriate coordinate system in the background.

Lines that can not be parsed into a `Point` are ignored, and warnings are
raised for invalid points that are outside of the mesh bounds.

#### Result outputs

Results are written to `results.dat` by default but this can be renamed as
needed.

```bash
# Search for all locations in points.txt, output results to 'myoutput.txt'
pointextract /path/to/meshtal.msht 104 --file points.txt --output myoutput.txt
```

These may also be written to the terminal directly with the `-d`/`--dump`
flag.

Both the parsed user input and search results are presented in tables. For
example, a 'points.txt' file may have the following

```bash
total 1.111e16 0.5 0.0 -1.9    # (energy, time, x, y, z), inside of mesh bounds
total 1.111e16 50.5 0.0 -1.9   # (energy, time, x, y, z), outside of mesh bounds
```

Tables show the user provided points to search for, and detail of the voxel containing these points.

```text
                            Points to search
id     energy        time        i_coord      j_coord      k_coord   system
-----------------------------------------------------------------------------
0      Total     1.11100e+16  5.00000e-01  0.00000e+00 -1.90000e+00   xyz
1      Total     1.11100e+16  5.05000e+01  0.00000e+00 -1.90000e+00   xyz

                    Voxels found (fmesh334, Rectangular)
id     energy        time     i_coord   j_coord   k_coord    result    error
--------------------------------------------------------------------------------
0      Total     1.11100e+16   0.500     0.000    -2.625  1.37928e-02 0.0092
1                             Not found in mesh
```

Note the second point was outside of the mesh and failed, which will warn
you with a `Not found in mesh` entry. Corresponding rows are numbered for
convenience.
*Note - long rows are really inconvenient for reading results on a lot of
screens, so the choice was made to split up the two tables. A case could be
made for doing somthing else with the results for easier parsing.*

</details>

### Splitmesh tool

Command line tool to split up meshtal files

```text
Usage: splitmesh <meshtal> [options]
```

<details>
  <summary>Click to see detailed examples</summary>

Splits up all meshes found in a meshtal file into their own individual
files.

This is very useful for processing large meshtal files with multiple
tallies, or for just reducing file sizes to a minimum for post-processing.

Help is printed with the `-h` flag, and `--help` will show examples, default
values, examples, and any important behaviour.

By default, every tally found in the file is splt into individual files.

#### How to choose specific tallies

Use the `--tallies`  option to specify one or more tallies to be separated
out. Invalid entries are simply ignored.

```bash
# Extract only tallies with ID 104, 204, and 504 from the primary file
splitmesh /path/to/meshtal.msht --tallies 104 204 504
```

#### How to change the file names

The name of the output files is appended with the tally number as
`<output>_<id>.msht`. Output defaults to `fmesh`, but this may be changed.

```bash
# Change output file names to "mymesh_<id>.msht"
splitmesh /path/to/meshtal.msht --output mymesh
```

</details>

### Posvol tool

Command line tool to inspect and convert posvol files

```bash
Usage: posvol <file> [options]
```

<details>
  <summary>Click to see detailed examples</summary>

Very simple reader for UKAEA CuV posvol binaries, skipping the need to open a
special viewer or sort it out manually just to check simple properties.

Allows for 1:1 conversion to ASCII, but also JSON and a more readable
text file.

The endian is assumed to be the same as the native type of the system
this tool is run on. If needed, an option can be provided in future
updates.

Help is printed with the `-h` flag, and `--help` will show examples, default
values, examples, and any important behaviour.

#### Print a summary

By default a simple summary of the posvol dimensions is logged.

```bash
# Print a summary of dimension properties
posvol plot_fmesh_104.bin
```

#### Convert 1:1 to ASCII integers

It can be useful just to have something that can be open and read, so `--ascii`
converts to text.

```bash
# Output a file named 'posvol.txt'
posvol plot_fmesh_104.bin --ascii
```

#### Convert to readable text format

For somthing a bit more human-friendly, the dimensions are split up and cells
grouped into single voxels separated by blank lines.

```bash
# Output a file named 'posvol.txt'
posvol plot_fmesh_104.bin --ascii --pretty
```

#### Convert to JSON file

For lovers of python and other languages there is a JSON output option because
it takes about 5 seconds for me to implement.

```bash
# Output a file named 'posvol.json'
posvol plot_fmesh_104.bin --json
```

#### Change the output file names

By default the file names are 'posvol.txt' for ascii file formats, and
'posvol.json' for a json format.

This can be changed by providing --output with a name

```bash
# Output a files named 'myfile.txt' and 'myfile.json'
posvol plot_fmesh_104.bin       \
            --json              \
            --ascii             \
            --output myfile
```

</details>

## Advanced use

**An API with full cargo documentation is available with details for using the crate**

The command line interface is a set of QoL tools written for colleagues. The
crate is far more useful than the CLI since the challenge with meshtal files is
always just trying to parse the horrible old MCNP outputs.

The crate allows any format to be read into a struct with a one-liner, and
from there you can do whatever you want with the mesh data. All mesh formats
are coerced into the same core `Mesh` structure.

```rust
// read a mesh from any of the meshtal formats
let mesh = meshtal::read_meshtal_target("./data/example_114.msht", 114).unwrap();

// now do whatever you want with it
```

The hull library documentation is published
[here](https://repositony.github.io/meshtal/index.html) for convenience.

## Features under development

Priority:

- HDF5 support is a priority with the release of MCNPv6.3 (behind feature flag as it depends on `libhdf5`)
- Weight window conversion to VTK for plotting
- Various tools for working with/converting to the new HDF5 formats
- Python bindings to make core functionality accessible

Time/need permitting, and in no particular order:

- Cell-under-Voxel sub-voxel plotting
- `Posvol` deserialisation allowing user to specify byte size and endian
- Allow mesh2vtk rotations/translation from command line
- Tools for extracting lines and planes from meshes, not just points
- Voxel result averaging for points exactly on 2-4 voxel boundaries
- Spherical mesh types just in case anyone cares
- WWINP file reader
