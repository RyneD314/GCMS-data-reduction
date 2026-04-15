# GCMS IFU Data Reduction Pipeline – Complete Instructions

This document provides all the instructions for reducing raw integral field unit (IFU) data from the **George and Cynthia Mitchell Spectrograph (GCMS)** – formerly VIRUS‑P – on the McDonald Observatory 2.7 m Harlan J. Smith telescope. The pipeline produces a 3D datacube, allows interactive selection of spatial regions, extracts spectra, and fits emission lines.

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation with the Provided Script (`gcms_env.sh`)](#installation-with-the-provided-script-gcms_envsh)
4. [Preparing Input Files](#preparing-input-files)
5. [Running the Script – Step‑by‑Step](#running-the-script--stepbystep)
6. [Using the Interactive Region Selector](#using-the-interactive-region-selector)
7. [Selecting an Emission Line to Fit](#selecting-an-emission-line-to-fit)
   - [Customising the line list](#customising-the-line-list)
8. [Outputs](#outputs)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

## Features

- Handles 3‑ or 6‑position dither patterns (user choice)
- Extracts fiber traces automatically from a flat field
- Wavelength calibration from a user‑provided arc line list (pixel → Å)
- Builds a 3D datacube (x, y, wavelength) from individual fiber spectra
- Combines dither positions with sigma‑clipping to reject cosmic rays
- Optional dither offsets (in cube pixels)
- Fetches SDSS image cutout, aligns it to the cube’s coordinate system (via `astropy.wcs` and `reproject`), and overlays contours
- Interactive region selector:
  - Circle (two clicks: center, edge) and polygon (click vertices, Enter to close)
  - Toggle include (green) / exclude (red) mode
  - Undo last shape, reset all, cancel circle drawing
- Extracts spectrum from selected region (include & not exclude)
- Fits a Gaussian + constant continuum to one of several emission lines (Hα, [O III], Hβ, [N II], [S II])
- Saves combined cube, extracted spectrum, and fit plot

## Requirements

- Python 3.8 or higher
- Internet connection (for SIMBAD/NED queries and SDSS image download)

## Installation with the Provided Script (`gcms_env.sh`)

A convenience shell script `gcms_env.sh` is provided. It creates a Python virtual environment and installs all required packages.

1. **Make the script executable** (only needed once):
   ```bash
   chmod +x gcms_env.sh
   ```

2. **Run the script** to create the environment and install packages:
   ```bash
   ./gcms_env.sh
   ```
   This will:
   - Create a folder `gcms_ifu_env` in the current directory.
   - Install `numpy`, `scipy`, `matplotlib`, `astropy`, `lmfit`, `astroquery`, `reproject`, and optional packages (`jupyter`, `seaborn`, `tqdm`).
   - Verify that all key imports work.

3. **Activate the environment** whenever you want to run the reduction script:
   ```bash
   source gcms_ifu_env/bin/activate
   ```
   *Important*: This only works if you are in the same directory where `gcms_env.sh` was run (i.e., where `gcms_ifu_env` was created). If you move the folder, you must update the path accordingly.

   You should see `(gcms_ifu_env)` at the beginning of your terminal prompt.

4. **Run the reduction script**:
   ```bash
   python gcms_reduce.py
   ```

## Preparing Input Files

### 1. Flat field FITS file
A standard dome flat or twilight flat. The script uses it only to locate fiber traces.

### 2. Arc lamp FITS file
An exposure of a wavelength calibration lamp (e.g., Neon, Argon, Mercury). The script extracts a reference spectrum from the central fiber.

### 3. Arc line file (text file)
Provide a **two‑column** text file (space‑ or tab‑separated) with:
- First column: pixel position (integer or float) of a known emission line in the **reference fiber’s spectrum**.
- Second column: wavelength of that line in Angstroms.

Example (`arc_lines.txt`):
```
100 5577.34
200 6300.30
300 6562.80
400 6717.00
```

You can identify these pixel positions by inspecting the arc frame in a DS9 viewer or by writing a small script to find peaks. The script will fit a cubic polynomial to these points and apply it to all spectral pixels.

### 4. Science FITS files
Place all your target exposures in a single folder. They can have any names. The script will list all `.fits` and `.fit` files and ask you to select the indices corresponding to each dither position (3 or 6 files). For a 3‑dither pattern, you will select three indices; for a 6‑dither pattern, six indices.

### 5. Fiber coordinates CSV (optional)
If you have a calibration file that maps each fiber to (x, y) coordinates in the final cube, provide it as a CSV with header `fiber_id,x,y`. If you do not provide this, the script will simulate a hexagonal grid (spacing 4.0 pixels) – useful for testing but not for real data.

## Running the Script – Step‑by‑Step

1. **Number of dithers** – type `3` or `6`.
2. **Flat field path** – absolute or relative path. Use tab completion in the terminal.
3. **Arc lamp path** – same.
4. **Arc line file path** – same.
5. **Dither offsets** – answer `y` if you have known offsets (in cube pixels). Then enter one pair (`dx dy`) per dither. If you answer `n`, all offsets are assumed zero.
6. **Science folder path** – the folder containing your science FITS files. After listing the files, you will be asked to enter the indices (space‑separated) for each dither.
7. **Fiber coordinates CSV** – provide path or press Enter to simulate.
8. **Target name for SDSS overlay** – e.g., `NGC 1277`. The script will query SIMBAD (fallback NED). Confirm the coordinates.
9. **Pixel scale of cube** – type the approximate scale in arcsec/pixel. For GCMS reduced data with typical dithering, `1.0` is a good starting guess.
10. **Interactive region selection** – a new window opens (see below).
11. **Emission line fitting** – choose a number from the list (1–6). To fit a line not in the list, you can edit the `lines` dictionary in the script (see “Customising the line list”).
12. **Save outputs** – you will be prompted for filenames for the combined cube, the spectrum, and the fit plot.

## Using the Interactive Region Selector

When the plot window appears, you will see your collapsed GCMS data with optional cyan SDSS contours. Use the following keyboard commands:

| Key | Action |
|-----|--------|
| `c` | **Circle mode** – click once to set the centre, then click again to set the radius. A dashed preview appears while dragging. |
| `p` | **Polygon mode** – click to add vertices. Press `Enter` to close the polygon and add the shape. |
| `e` | **Toggle mode** – switches between **include** (green) and **exclude** (red). Included spaxels are kept; excluded spaxels are removed from the final spectrum. |
| `u` | **Undo** – removes the last drawn shape (circle or polygon). |
| `r` | **Reset** – clears all include and exclude masks. |
| `Esc` | **Cancel** – while in circle mode (after first click), cancels the current circle. |
| `Enter` | **Finish** – closes the window and proceeds with the selected region. |

The final spectrum is extracted from spaxels that are **included AND NOT excluded**.

## Selecting an Emission Line to Fit

At the end of the script, you will see a numbered list:

```
Select emission line to fit:
  1. Hβ (4861 Angstrom)
  2. [O III] (5007 Angstrom)
  3. Hα (6563 Angstrom)
  4. [N II] (6583 Angstrom)
  5. [S II] (6717 Angstrom)
  6. [S II] (6731 Angstrom)
```

Enter the corresponding number. The script then fits a Gaussian + constant continuum to that line.

### Customising the line list

If you need a line that is not in the list (e.g., [O II] 3727 Å or He I 5876 Å), edit the `lines` dictionary in `gcms_reduce.py`. Find this section:

```python
lines = {'1': ('Hβ', 4861), '2': ('[O III]', 5007), '3': ('Hα', 6563),
         '4': ('[N II]', 6583), '5': ('[S II]', 6717), '6': ('[S II]', 6731)}
```

Add a new key, for example:

```python
'7': ('[O II]', 3727)
```

The first element is the label (used in the plot title), the second is the rest wavelength in Angstroms. Save the file and rerun the script. Your new line will appear in the menu.

## Outputs

| File | Description |
|------|-------------|
| `<your_cube_name>.fits` | Combined 3D datacube (x, y, wavelength). |
| `<your_spectrum_name>.dat` | Extracted spectrum (two columns: wavelength in Å, flux). |
| `<your_plot_name>.png` | Plot of the spectrum with the Gaussian fit. |

## Troubleshooting

- **No fiber traces found** – The `find_peaks` threshold is set to the 80th percentile of the spatial profile. If your flat field is very noisy, lower this value (e.g., 60) in the script.
- **Arc line file errors** – Ensure the file has exactly two columns, no missing values. Use spaces or tabs.
- **SDSS overlay fails** – The target may not be in SDSS, or you need an internet connection. The script will continue without overlay.
- **Pixel scale wrong** – If the SDSS contours do not align with your object, try a different pixel scale (e.g., 0.8 or 1.2 arcsec/pixel). You can also refine the WCS by editing the `generate_cube_wcs_header` function to include rotation or more accurate astrometry.
- **Region selector window freezes** – On some systems, you may need to install a compatible GUI backend. Try:
  ```bash
  pip install PyQt5
  ```
  or set the backend to TkAgg:
  ```bash
  export MPLBACKEND=TkAgg
  ```

## Citation

If you use this script for published work, please acknowledge the GCMS instrument and the McDonald Observatory. No formal citation is required, but you may link to this repository.
