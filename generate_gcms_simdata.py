#!/usr/bin/env python3
"""
generate_gcms_simdata.py - Create artificial GCMS raw data for testing.

Generates:
- flat.fits
- arc.fits
- science_A1.fits ... science_A3.fits (3-dither set)
- science_B1.fits ... science_B3.fits (additional 3 for 6-dither)
- fiber_coords.csv
- arc_lines.txt (pixel positions and known wavelengths)

Usage:
    python generate_gcms_simdata.py ##outdir ##n_fibers ##n_dithers ##seed
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

###################################################################
# Simulation parameters (adjust as needed)
###################################################################

# Detector size (spatial pixels, spectral pixels)
SPATIAL_PIX = 2048   # typical CCD size in spatial direction
SPEC_PIX = 2048      # spectral pixels

# Fiber parameters
FIBER_WIDTH_PIX = 5      # width of each fiber trace in spatial direction (FWHM)
FIBER_SPACING_PIX = 7.8  # spacing between fiber centers (from VIRUS-P paper)
PEAK_HEIGHT_FLAT = 1000  # counts in flat frame per fiber

# Spectral calibration
WAVE_MIN = 4400.0   # Angstroms
WAVE_MAX = 6800.0
# Arc lamp lines (wavelength, relative intensity)
ARC_LINES = [
    (5577.34, 0.8),   # [O I]
    (6300.30, 0.9),   # [O I]
    (6548.05, 0.6),   # Hα + [N II] blend simplified
    (6562.80, 1.0),   # Hα
    (6583.45, 0.7),   # [N II]
]

# Galaxy spectrum: emission lines (rest frame, then redshifted)
GALAXY_REDSHIFT = 0.0172   # NGC 1277-like
GALAXY_EMISSION_LINES = [
    (4861.0, 0.5),   # Hβ
    (4959.0, 0.4),   # [O III]
    (5007.0, 0.8),   # [O III]
    (6563.0, 1.0),   # Hα
    (6583.0, 0.7),   # [N II]
]
# Continuum (power law)
CONTINUUM_AMP = 100.0
CONTINUUM_INDEX = -1.5

# Galaxy spatial shape (2D Gaussian)
GALAXY_FWHM_PIX = 4.0   # arcseconds, assume 1 pix = 1 arcsec for cube
GALAXY_TOTAL_FLUX = 1e5  # total counts summed over all spaxels

# Noise
READ_NOISE = 5.0   # electrons
SKY_LEVEL = 50.0   # electrons per pixel

# Dither offsets for 3-dither set (in cube pixels, not raw detector pixels)
# These are the shifts applied when building the cube.
DITHER_3_OFFSETS = [(0,0), (2.0, 0), (1.0, 1.732)]  # hexagonal fill
# For 6-dither, second set shifted by half spacing
DITHER_6_OFFSETS_B = [(1.0, 0.866), (3.0, 0.866), (2.0, 2.598)]

###################################################################-
# Helper functions
###################################################################-

def make_fiber_trace_positions(n_fibers, spacing=FIBER_SPACING_PIX, start=50):
    """Generate spatial pixel positions for fiber traces."""
    return np.array([start + i*spacing for i in range(n_fibers)])

def gaussian_spot(x, x0, fwhm):
    """1D Gaussian profile."""
    sigma = fwhm / 2.3548
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

def generate_flat_frame(fiber_positions, shape=(SPATIAL_PIX, SPEC_PIX)):
    """Create a flat frame with Gaussian traces of constant intensity."""
    frame = np.zeros(shape)
    for x0 in fiber_positions:
        ix = int(round(x0))
        if 0 <= ix < shape[0]:
            # Add Gaussian profile in spatial direction, constant along spectral
            profile = gaussian_spot(np.arange(shape[0]), x0, FIBER_WIDTH_PIX)
            frame[:, :] += PEAK_HEIGHT_FLAT * profile[:, np.newaxis]
    # Add noise
    frame += np.random.poisson(np.maximum(frame, 0)) - frame  # Poisson noise
    frame += np.random.normal(0, READ_NOISE, shape)
    return frame.astype(np.float32)

def generate_arc_frame(fiber_positions, arc_lines, wavelength_map, shape=(SPATIAL_PIX, SPEC_PIX)):
    """
    Generate arc frame: for each fiber, add emission lines at the correct spectral pixels.
    wavelength_map: array mapping spectral pixel index -> wavelength.
    """
    frame = np.zeros(shape)
    # For each fiber, add lines with intensity proportional to line strength
    for x0 in fiber_positions:
        ix = int(round(x0))
        if 0 <= ix < shape[0]:
            # Find spectral pixels corresponding to each arc line
            for wl, intens in arc_lines:
                # Find nearest spectral pixel
                pix = np.argmin(np.abs(wavelength_map - wl))
                # Add Gaussian line profile (simple: add to that pixel and neighbors)
                sigma_pix = 1.5  # arc line width in pixels
                for dp in range(-3, 4):
                    p = pix + dp
                    if 0 <= p < shape[1]:
                        frame[ix, p] += intens * np.exp(-0.5 * (dp/sigma_pix)**2)
    # Add noise and constant background
    frame += SKY_LEVEL
    frame += np.random.normal(0, READ_NOISE, shape)
    return frame.astype(np.float32)

def generate_galaxy_spectrum(wavelength, redshift, emission_lines, continuum_amp, continuum_idx):
    """Return a synthetic galaxy spectrum (flux per spectral pixel)."""
    # Observed wavelengths
    observed_wave = wavelength * (1 + redshift)
    # Interpolate to the observed grid (linear)
    flux = np.zeros_like(wavelength)
    # Continuum
    cont = continuum_amp * (observed_wave / observed_wave[0])**continuum_idx
    flux += cont
    # Emission lines: Gaussian profiles
    for line_rest, intens in emission_lines:
        line_obs = line_rest * (1 + redshift)
        sigma_obs = 2.0  # Angstroms, typical line width
        # Convert to pixels (approximate)
        dw = wavelength[1] - wavelength[0]
        sigma_pix = sigma_obs / dw
        center_pix = np.argmin(np.abs(wavelength - line_obs))
        for dp in range(-10, 11):
            p = center_pix + dp
            if 0 <= p < len(wavelength):
                flux[p] += intens * np.exp(-0.5 * (dp/sigma_pix)**2)
    return flux

def generate_science_frame(fiber_positions, fiber_coords_cube, cube_shape, dither_offset,
                           galaxy_spectrum, shape=(SPATIAL_PIX, SPEC_PIX)):
    """
    Simulate a science exposure: for each fiber, project the galaxy spectrum
    from the cube to the fiber, then map to the CCD frame.
    """
    frame = np.zeros(shape)
    nx, ny, nw = cube_shape
    # For each fiber, get its cube coordinates (x,y)
    for i, (x0, y0) in enumerate(fiber_coords_cube):
        # Apply dither offset (shift in cube space)
        x_sky = x0 + dither_offset[0]
        y_sky = y0 + dither_offset[1]
        # Find nearest integer cube pixel (simple resampling)
        ix_sky = int(round(x_sky))
        iy_sky = int(round(y_sky))
        if 0 <= ix_sky < nx and 0 <= iy_sky < ny:
            # Spectrum from that cube pixel
            spec = galaxy_spectrum
        else:
            # Sky spectrum (no galaxy, just sky background)
            spec = np.ones(nw) * SKY_LEVEL
        # Map this fiber to CCD spatial position
        spatial_pos = fiber_positions[i]
        ix_spatial = int(round(spatial_pos))
        if 0 <= ix_spatial < shape[0]:
            # Add the spectrum to the CCD frame (simple: each fiber covers a single spatial pixel)
            # In reality, there is cross-talk and PSF, but for simulation we use a single pixel.
            frame[ix_spatial, :] += spec
    # Add noise
    frame += np.random.normal(0, READ_NOISE, shape)
    return frame.astype(np.float32)

def generate_fiber_coords(n_fibers, cube_nx, cube_ny, pattern='hex'):
    """
    Generate sky coordinates for each fiber in the final cube.
    For GCMS, fibers are arranged in a hexagonal grid with ~1/3 fill factor.
    We'll generate positions that roughly fill the cube with some randomness.
    """
    # Simple approximation: place fibers in a hexagonal lattice over the cube area
    spacing = 4.0  # pixels between fiber centers in cube
    coords = []
    i = 0
    for y in np.arange(0, cube_ny, spacing):
        row_offset = (i % 2) * (spacing/2)
        for x in np.arange(row_offset, cube_nx, spacing):
            if i < n_fibers:
                coords.append((x, y))
                i += 1
            else:
                break
        if i >= n_fibers:
            break
    # If not enough fibers, pad with zeros (should not happen)
    while len(coords) < n_fibers:
        coords.append((0,0))
    return np.array(coords[:n_fibers])

def print_region_selector_instructions():
    """Prints instructions for the region selector to the console."""
    print("\n" + "="*60)
    print("REGION SELECTOR INSTRUCTIONS")
    print("="*60)
    print("Select your region of interest using the interactive plot.")
    print("The final spectrum will be extracted from the selected area.\n")
    print("Key Bindings:")
    print("  [c] - Circle mode (click: center, then click: edge)")
    print("  [p] - Polygon mode (click to add vertices, press Enter to close)")
    print("  [e] - TOGGLE between INCLUDE (green) and EXCLUDE (red) mode")
    print("  [u] - Undo the last drawn shape")
    print("  [r] - Reset all selections (clears include and exclude masks)")
    print("  [Enter] - Finish selection and close the window")
    print("\nClose the plot window after pressing Enter to continue.")
    print("="*60)

###################################################################-
# Main generation routine
###################################################################-

def main():
    parser = argparse.ArgumentParser(description="Generate GCMS simulation data")
    parser.add_argument("outdir", default="gcms_simdata", type=str, help="Output directory")
    parser.add_argument("n_fibers", default=246, type=int, help="Number of fibers")
    parser.add_argument("n_dithers", choices=[3,6], default=3, type=int, 
                        help="Number of dither positions (3 or 6)")
    parser.add_argument("seed", default=42, type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    # print(args)

    np.random.seed(args.seed)

    # Create output directory structure
    raw_dir = os.path.join(args.outdir, "raw")
    calib_dir = os.path.join(args.outdir, "calib")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    # Wavelength solution (linear from WAVE_MIN to WAVE_MAX)
    wavelength = np.linspace(WAVE_MIN, WAVE_MAX, SPEC_PIX)
    dw = wavelength[1] - wavelength[0]

    # Fiber trace positions on detector
    fiber_positions = make_fiber_trace_positions(args.n_fibers)

    # Flat frame
    print("Generating flat frame...")
    flat = generate_flat_frame(fiber_positions)
    fits.writeto(os.path.join(calib_dir, "flat.fits"), flat, overwrite=True)

    # Arc frame: use the same fiber positions and add lines at known wavelengths
    print("Generating arc frame...")
    arc = generate_arc_frame(fiber_positions, ARC_LINES, wavelength)
    fits.writeto(os.path.join(calib_dir, "arc.fits"), arc, overwrite=True)

    # Save arc line list (for calibration)
    # We'll measure the actual pixel positions from the arc frame (simulate measurement)
    # For simplicity, we compute the theoretical pixel positions and add small errors.
    arc_pixel_positions = []
    arc_wavelengths = []
    for wl, intens in ARC_LINES:
        pix = np.argmin(np.abs(wavelength - wl))
        # Add small random error to simulate measurement noise
        pix_err = np.random.normal(0, 0.5)
        arc_pixel_positions.append(pix + pix_err)
        arc_wavelengths.append(wl)
    arc_lines_table = np.column_stack([arc_pixel_positions, arc_wavelengths])
    np.savetxt(os.path.join(calib_dir, "arc_lines.txt"), arc_lines_table,
               header="pixel wavelength_ang", comments="")

    # Galaxy spectrum (observed frame, no redshift applied yet – we will apply redshift)
    # We generate the spectrum at rest frame wavelengths then shift in the science frame generation.
    # But for simplicity, we precompute the galaxy spectrum at observed wavelengths using the galaxy's redshift.
    galaxy_spec = generate_galaxy_spectrum(wavelength, GALAXY_REDSHIFT,
                                           GALAXY_EMISSION_LINES,
                                           CONTINUUM_AMP, CONTINUUM_INDEX)

    # Cube spatial dimensions (arbitrary, but must cover fiber coordinates)
    cube_nx = 100
    cube_ny = 100
    cube_shape = (cube_nx, cube_ny, SPEC_PIX)

    # Generate fiber sky coordinates (mapping fiber index -> cube pixel)
    fiber_coords = generate_fiber_coords(args.n_fibers, cube_nx, cube_ny)
    np.savetxt(os.path.join(calib_dir, "fiber_coords.csv"), fiber_coords,
               header="x_pixel,y_pixel", delimiter=",", comments="")
    print(f"Fiber coordinates saved. Shape: {fiber_coords.shape}")

    # Generate science frames for each dither position
    dither_offsets = DITHER_3_OFFSETS
    if args.n_dithers == 6:
        dither_offsets = DITHER_3_OFFSETS + DITHER_6_OFFSETS_B
        # Rename patterns: first three A1-A3, next three B1-B3
        names = [f"A{i+1}" for i in range(3)] + [f"B{i+1}" for i in range(3)]
    else:
        names = [f"A{i+1}" for i in range(3)]

    for i, (offset, name) in enumerate(zip(dither_offsets, names)):
        print(f"Generating science frame for dither {name} with offset {offset}...")
        sci = generate_science_frame(fiber_positions, fiber_coords, cube_shape, offset,
                                     galaxy_spec)
        fits.writeto(os.path.join(raw_dir, f"science_{name}.fits"), sci, overwrite=True)

    print(f"Simulation complete. Data written to {args.outdir}")
    print("You can now run the reduction scripts:")
    if args.n_dithers == 3:
        print("  python gcms_reduce.py ##rawdir simdata/raw ##outdir output ##flat simdata/calib/flat.fits ##arc simdata/calib/arc.fits ##calib simdata/calib/fiber_coords.csv")
    else:
        print("  python gcms_reduce_6dither.py ##rawdir simdata/raw ##outdir output ##flat simdata/calib/flat.fits ##arc simdata/calib/arc.fits ##calib simdata/calib/fiber_coords.csv")

if __name__ == "__main__":
    main()
