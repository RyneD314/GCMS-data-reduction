Below is a completely revised `README.md` for the repository, integrating the unified pipeline, both 3‑ and 6‑dither support, configuration file and command‑line usage, SDSS overlay, sample data generation, and all calibration steps.

---

# GCMS IFU Data Reduction Pipeline

**Unified reduction pipeline for the George & Cynthia Mitchell Spectrograph (GCMS, formerly VIRUS‑P) at McDonald Observatory.**

This pipeline reduces raw GCMS IFU frames to a final 3‑datacube, incorporating:

- Bias, flat‑field, and arc‑lamp calibration (master frames from multiple exposures)
- Automatic fiber trace extraction
- Wavelength calibration (from arc lines or pre‑computed coefficients)
- Sky subtraction (nearest‑in‑frame‑number matching)
- Dither combination (3 or 6 positions, user‑defined offsets)
- Optional sky‑line zero‑point refinement
- Interactive region selection (circle / polygon, include / exclude) with optional SDSS overlay
- Emission line fitting (Gaussian + continuum) and spectrum extraction

Two usage modes are supported:
1. **Configuration file** – full control, reproducible settings.
2. **Command‑line arguments** – simple directory‑based workflow.

---

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Data Organisation](#data-organisation)
- [Auxiliary Files](#auxiliary-files)
  - [Fiber Coordinates](#fiber-coordinates)
  - [Arc Line File](#arc-line-file)
  - [Wavelength Coefficient File](#wavelength-coefficient-file)
- [Usage](#usage)
  - [Command‑Line Example (3‑dither)](#command-line-example-3-dither)
  - [Command‑Line Example (6‑dither)](#command-line-example-6-dither)
  - [Configuration File Example](#configuration-file-example)
- [Interactive Region Selection](#interactive-region-selection)
- [Outputs](#outputs)
- [Generating Sample Data](#generating-sample-data)
- [Environment Setup Script](#environment-setup-script)
- [Shell Scripts](#shell-scripts)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Astropy
- LMFIT
- Astroquery
- Reproject

All dependencies can be installed via `pip` (see [Installation](#installation)).

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/RyneD314/GCMS-data-reduction.git
cd GCMS-data-reduction
./gcms_env.sh          # creates venv and installs packages
source gcms_ifu_env/bin/activate
```

The environment script (`gcms_env.sh`) is provided in the repository.

---

## Data Organisation

Arrange your FITS files into directories **outside the repository** (or inside, but keep them separate from code). Example:

```
/path/to/your/data/
├── biases/          # any number of bias frames (optional)
├── flats/           # any number of flat frames (required)
├── arcs/            # any number of arc lamp frames (required)
├── skies/           # optional sky frames (one per dither recommended)
└── science_3dith/   # exactly 3 FITS files for a 3‑dither observation
```

**Naming convention**: The script matches science and sky frames by the numeric part of the filename (e.g., `sci_001.fits` ↔ `sky_001.fits`). If no match is found, sky subtraction is skipped for that science frame.

---

## Auxiliary Files

### Fiber Coordinates

A CSV file listing the (x, y) position of each fiber in the final cube grid.  
**Format** (CSV, header required):

```csv
fiber_id, x, y
0, 45.2, 30.1
1, 49.8, 30.1
2, 54.3, 30.1
...
```

### Arc Line File

A text file with two columns: pixel position and known wavelength (Å). Used when `--arc_lines` is supplied.  
Example (`arc_lines_auto.txt`):

```
123.4 5577.34
245.7 6300.30
367.9 6548.05
490.2 6562.80
```

### Wavelength Coefficient File

A text file containing the 4 coefficients (cubic polynomial) of the wavelength solution.  
Example (`wavelength_coeffs.txt`):

```
1.234e-06  -2.345e-03   5.678e+00   3.456e+03
```

---

## Usage

The main script is `gcms_reduce_unified.py`. You can run it with command‑line arguments or via a configuration file.

### Command‑Line Example (3‑dither)

```bash
python gcms_reduce_unified.py \
    --sci_dir /path/to/science_3dith \
    --bias_dir /path/to/biases \
    --flat_dir /path/to/flats \
    --arc_dir /path/to/arcs \
    --sky_dir /path/to/skies \
    --n_dither 3 \
    --offsets "0,0;2,0;1,2" \
    --fiber_coords fiber_positions.csv \
    --out_cube gcms_3dith_cube.fits \
    --object_name "NGC 1277" \
    --pixscale 1.0 \
    --arc_lines arc_lines_auto.txt \
    --use_sky_zero \
    --z 0.0172
```

### Command‑Line Example (6‑dither)

```bash
python gcms_reduce_unified.py \
    --sci_dir /path/to/science_6dith \
    --bias_dir /path/to/biases \
    --flat_dir /path/to/flats \
    --arc_dir /path/to/arcs \
    --sky_dir /path/to/skies \
    --n_dither 6 \
    --offsets "0,0;2,0;1,2;1,1;3,1;2,3" \
    --fiber_coords fiber_positions.csv \
    --out_cube gcms_6dith_cube.fits \
    --object_name "NGC 1277" \
    --pixscale 1.0 \
    --coeff_file wavelength_coeffs.txt \
    --use_sky_zero \
    --z 0.0172
```

### Configuration File Example

Create a text file (e.g., `myconfig.cfg`) with the following content:

```ini
sci_dir = /path/to/science_3dith
bias_dir = /path/to/biases
flat_dir = /path/to/flats
arc_dir = /path/to/arcs
sky_dir = /path/to/skies
n_dither = 3
offsets = 0,0;2,0;1,2
fiber_coords = fiber_positions.csv
out_cube = gcms_3dith_cube.fits
object_name = NGC 1277
pixscale = 1.0
arc_lines = arc_lines_auto.txt
use_sky_zero = y
z = 0.0172
```

Then run:

```bash
python gcms_reduce_unified.py --config myconfig.cfg
```

Command‑line arguments always override the configuration file.

---

## Interactive Region Selection

After the cube is built, a collapsed (sum over wavelength) image appears.

- **c** – circle selection tool (click and drag)
- **p** – polygon selection tool (click vertices, press **Enter** to close)
- **e** – toggle between **include** (green) and **exclude** (red) mode
- **r** – reset all selections
- **Enter** – finish selection and extract spectrum

If an SDSS cutout was fetched (via `--object_name`), its contours are overlaid in cyan.

Once selection is complete, the extracted spectrum is shown and you can fit emission lines interactively.

---

## Outputs

- **FITS cube** (name given by `--out_cube`) – 3D datacube (nx, ny, nw) with WCS, ready for DS9 / QFitsView.
- **Extracted spectrum** (optional, saved as text file) – two columns: wavelength (Å) and flux.
- **Line fit plot** (optional, saved as PNG).

All outputs are saved in the current working directory unless a full path is provided.

---

## Generating Sample Data

A script `generate_gcms_simdata.py` is included to create realistic simulated data for testing.

**Usage**:

```bash
# 3‑dither set
python generate_gcms_simdata.py --outdir sample_data_3dith --n_fibers 246 --n_dithers 3 --seed 42

# 6‑dither set
python generate_gcms_simdata.py --outdir sample_data_6dith --n_fibers 246 --n_dithers 6 --seed 42
```

Generated files:

- `flat.fits`, `arc.fits`
- Science files: `science_A1.fits`, `science_A2.fits`, `science_A3.fits` (and `science_B*.fits` for 6‑dither)
- `fiber_coords.csv`
- `arc_lines.txt`

You can then run the pipeline on `sample_data_3dith/` to verify everything works.

---

## Environment Setup Script

The repository includes `gcms_env.sh`:

```bash
#!/bin/bash
python3 -m venv gcms_ifu_env
source gcms_ifu_env/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib astropy lmfit astroquery reproject
```

Make it executable and run it once:

```bash
chmod +x gcms_env.sh
./gcms_env.sh
```

Then activate the environment before each reduction:

```bash
source gcms_ifu_env/bin/activate
```

---

## Shell Scripts

Two example shell scripts are provided as templates:

- `gcms_3dith.sh` – reduces a 3‑dither dataset
- `gcms_6dith.sh` – reduces a 6‑dither dataset

Edit the paths inside to match your data locations and run:

```bash
./gcms_3dith.sh
```

---

## Troubleshooting

| Problem | Possible Solution |
| :--- | :--- |
| `No module named 'astroquery'` | Activate the virtual environment: `source gcms_ifu_env/bin/activate` |
| `ValueError: Need exactly 3 offsets` | Check that `--offsets` has exactly the same number of entries as `--n_dither`. |
| `Error: need either a flat or an arc frame` | Ensure `--flat_dir` or `--arc_dir` contains valid FITS files. |
| Sky zero‑point correction fails | Install `scipy>=1.6` and check that sky frames contain strong sky lines (5577, 6300, 6363 Å). |
| SDSS overlay fails | Your internet connection may be blocking astroquery; try setting a different `--sdss_size` or skip by not providing `--object_name`. |

For further help, please open an issue on the GitHub repository.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

**Enjoy reducing your GCMS data!**
