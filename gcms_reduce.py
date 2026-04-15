#!/usr/bin/env python3
"""
gcms_reduce.py - Full GCMS IFU reduction with SDSS overlay via astropy.wcs/reproject.
"""

import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
import astropy.units as u
from scipy.signal import find_peaks
from lmfit.models import GaussianModel, ConstantModel
from matplotlib.path import Path
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned
from astroquery.sdss import SDSS
from reproject import reproject_interp

##############################################################
# Helper functions for data reduction
##############################################################

def extract_fiber_spectra(frame, peak_positions, width=5):
    """
    Extract 1D spectra from a 2D CCD frame for each fiber trace.
    """
    n_fibers = len(peak_positions)
    n_spec = frame.shape[1]                     # spectral axis length
    spectra = np.zeros((n_fibers, n_spec))
    for i, p in enumerate(peak_positions):
        # Sum over a small spatial window centered at the fiber trace.
        # width = 5 pixels is typical for GCMS (fiber traces are ~5 pixels FWHM)
        x0 = max(0, int(p - width//2))
        x1 = min(frame.shape[0], int(p + width//2))
        spectra[i, :] = np.sum(frame[x0:x1, :], axis=0)
    return spectra


def build_cube(fiber_spectra, fiber_coords, cube_shape, dither_offset=(0,0)):
    """
    Place extracted fiber spectra into a 3D datacube.
    """
    nx, ny, nw = cube_shape
    cube = np.full((nx, ny, nw), np.nan)        # initialize with NaNs (empty spaxels)
    for i, (x, y) in enumerate(fiber_coords):
        # Apply dither offset (in cube pixels) to the fiber's nominal position
        x_shifted = int(round(x + dither_offset[0]))
        y_shifted = int(round(y + dither_offset[1]))
        if 0 <= x_shifted < nx and 0 <= y_shifted < ny:
            cube[x_shifted, y_shifted, :] = fiber_spectra[i, :]
    return cube


def fit_gaussian_emission(wavelength, spectrum, guess_center, guess_sigma=5):
    """
    Fit a single Gaussian emission line plus constant continuum.
    """
    model = GaussianModel() + ConstantModel()
    # Initial guesses: amplitude = peak - median continuum, center = user guess,
    # sigma = 5 Angstrom, continuum = median flux.
    params = model.make_params(amplitude=np.max(spectrum)-np.median(spectrum),
                               center=guess_center,
                               sigma=guess_sigma,
                               c=np.median(spectrum))
    # Set bounds to prevent unphysical fits
    params['center'].min = guess_center - 20
    params['center'].max = guess_center + 20
    params['sigma'].min = 1
    params['sigma'].max = 20
    return model.fit(spectrum, params, x=wavelength)


##############################################################
# Functions for astronomical coordinate and image handling
##############################################################

def get_object_coordinates(object_name):
    """Resolve object name to J2000 equatorial coordinates using SIMBAD then NED."""
    print(f"\nResolving coordinates for '{object_name}'...")
    # Try SIMBAD first
    try:
        Simbad.add_votable_fields('ra(d)', 'dec(d)')   # request decimal degrees
        result = Simbad.query_object(object_name)
        if result is not None:
            ra = result['RA_d'][0]    # first row, RA in degrees
            dec = result['DEC_d'][0]
            print(f"SIMBAD found: RA = {ra:.6f}, Dec = {dec:.6f}")
            return ra, dec
    except Exception as e:
        print(f"SIMBAD query failed: {e}")
    # Fallback to NED
    try:
        result = Ned.query_object(object_name)
        if result is not None and len(result) > 0:
            ra_str = result['RA'][0]     # e.g., "03:19:47.6"
            dec_str = result['DEC'][0]   # e.g., "+41:34:37"
            # Convert from sexagesimal to decimal degrees using SkyCoord
            coords = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
            ra = coords.ra.deg
            dec = coords.dec.deg
            print(f"NED found: RA = {ra:.6f}, Dec = {dec:.6f}")
            return ra, dec
    except Exception as e:
        print(f"NED query failed: {e}")
    return None, None


def get_sdss_cutout_and_wcs(ra, dec, size_arcmin=2.0, band='r'):
    """
    Retrieve an SDSS image cutout and its WCS object.
    """
    try:
        # SDSS.get_images returns a list of HDUList objects; radius in arcminutes
        hdulist = SDSS.get_images(coordinates=SkyCoord(ra, dec, unit='deg'),
                                  radius=size_arcmin * u.arcmin,
                                  band=band)
        if not hdulist:
            print("No SDSS data found.")
            return None, None
        hdu = hdulist[0]
        data = hdu.data
        # If data is a cube (e.g., multiple bands or wavelengths), median collapse
        if data.ndim == 3:
            data = np.nanmedian(data, axis=0)
        wcs = WCS(hdu.header)              # extract WCS from FITS header
        return data, wcs
    except Exception as e:
        print(f"Error fetching SDSS image: {e}")
        return None, None


def generate_cube_wcs_header(ra_center, dec_center, cube_shape, pixel_scale_arcsec=1.0):
    """
    Create a FITS header with a simple TAN (gnomonic) projection WCS for the cube.
    """
    nx, ny, nw = cube_shape[:3]
    header = fits.Header()
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = nw
    header['CTYPE1'] = 'RA---TAN'          # Right ascension, gnomonic projection
    header['CTYPE2'] = 'DEC--TAN'          # Declination
    header['CTYPE3'] = 'WAVE'              # Wavelength axis (placeholder)
    # Reference pixel at image center (FITS uses 1-indexed)
    header['CRPIX1'] = nx / 2.0 + 0.5
    header['CRPIX2'] = ny / 2.0 + 0.5
    header['CRPIX3'] = 1.0
    # World coordinates at the reference pixel
    header['CRVAL1'] = ra_center
    header['CRVAL2'] = dec_center
    header['CRVAL3'] = 5000.0              # placeholder wavelength
    # Pixel scale in degrees per pixel. Negative for RA to follow FITS convention.
    header['CDELT1'] = -pixel_scale_arcsec / 3600.0
    header['CDELT2'] = pixel_scale_arcsec / 3600.0
    header['CDELT3'] = 1.0                # wavelength step placeholder
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CUNIT3'] = 'Angstrom'
    return header


##############################################################
# Interactive region selector with include/exclude masking and SDSS overlay
##############################################################

class RegionSelector:
    """
    Interactive tool for selecting spatial regions on a 2D image.
    """
    def __init__(self, image, sdss_reprojected=None, sdss_contours=None):
        self.image = image
        self.sdss_reprojected = sdss_reprojected
        self.sdss_contours = sdss_contours
        self.ny, self.nx = image.shape
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.imshow(image, origin='lower', cmap='inferno')

        # Overlay SDSS contours if provided
        if sdss_reprojected is not None and sdss_contours is not None:
            self.ax.contour(sdss_reprojected, levels=sdss_contours,
                            colors='cyan', alpha=0.5, linewidths=0.8)
            print("SDSS contours overlaid.")

        # Two masks: included spaxels (green) and excluded spaxels (red)
        self.include_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.exclude_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.exclude_mode = False          # False = include, True = exclude
        self.polygon_verts = []            # list of (x,y) for polygon vertices
        self.polygon_active = False
        self.circle_center = None          # for two-click circle
        self.circle_preview = None         # matplotlib patch for preview
        self.history = []                  # store (type, mask) for undo

        self.update_title()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        plt.show(block=False)

    def update_title(self):
        mode = "EXCLUDE" if self.exclude_mode else "INCLUDE"
        self.ax.set_title(f"Mode: {mode} | c:circle | p:polygon | u:undo | e:toggle | r:reset | enter:finish")

    def on_key(self, event):
        # e: toggle include/exclude mode
        if event.key == 'e':
            self.exclude_mode = not self.exclude_mode
            self.update_title()
        # r: reset all masks
        elif event.key == 'r':
            self.include_mask.fill(False)
            self.exclude_mask.fill(False)
            self.history.clear()
            self.redraw()
            print("All masks reset.")
        # u: undo last shape
        elif event.key == 'u':
            self.undo()
        # Escape: cancel circle drawing
        elif event.key == 'escape':
            if self.circle_center is not None:
                self.circle_center = None
                if self.circle_preview:
                    self.circle_preview.remove()
                    self.circle_preview = None
                self.fig.canvas.draw()
                print("Circle drawing cancelled.")
        # c: circle mode
        elif event.key == 'c':
            self.polygon_active = False
            self.circle_center = None
            if self.circle_preview:
                self.circle_preview.remove()
                self.circle_preview = None
            print("Circle mode: click center, then click edge.")
        # p: polygon mode
        elif event.key == 'p':
            self.polygon_active = True
            self.polygon_verts = []
            self.circle_center = None
            print("Polygon mode: click vertices, press Enter to close.")
        # Enter: finish (if polygon, close polygon first)
        elif event.key == 'enter':
            if self.polygon_active and len(self.polygon_verts) >= 3:
                self.finish_polygon()
            else:
                self.finish()

    def undo(self):
        """Remove the last drawn shape."""
        if not self.history:
            print("Nothing to undo.")
            return
        last_type, last_mask = self.history.pop()
        if last_type == 'include':
            self.include_mask = np.logical_and(self.include_mask, ~last_mask)
        else:
            self.exclude_mask = np.logical_and(self.exclude_mask, ~last_mask)
        self.redraw()
        print(f"Undid last {last_type} shape.")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if not self.polygon_active:
            # Circle mode: first click = center, second click = radius
            if self.circle_center is None:
                self.circle_center = (event.xdata, event.ydata)
                print(f"Center: ({self.circle_center[0]:.1f}, {self.circle_center[1]:.1f}) - now click edge.")
            else:
                x2, y2 = event.xdata, event.ydata
                radius = np.hypot(x2 - self.circle_center[0], y2 - self.circle_center[1])
                self.add_circle(self.circle_center[0], self.circle_center[1], radius)
                self.circle_center = None
                if self.circle_preview:
                    self.circle_preview.remove()
                    self.circle_preview = None
        else:
            # Polygon mode: add vertex
            self.polygon_verts.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'ro' if self.exclude_mode else 'go')
            self.fig.canvas.draw()
            print(f"Vertex {len(self.polygon_verts)}: ({event.xdata:.1f}, {event.ydata:.1f})")

    def on_motion(self, event):
        """Preview circle while dragging the second click."""
        if not self.polygon_active and self.circle_center is not None and event.inaxes == self.ax:
            if self.circle_preview:
                self.circle_preview.remove()
            radius = np.hypot(event.xdata - self.circle_center[0], event.ydata - self.circle_center[1])
            self.circle_preview = plt.Circle(self.circle_center, radius, fill=False, color='cyan', linestyle='--')
            self.ax.add_patch(self.circle_preview)
            self.fig.canvas.draw()

    def add_circle(self, cx, cy, radius):
        """Add a circular mask to include or exclude."""
        # Create a meshgrid of pixel indices; ogrid is memory efficient
        yy, xx = np.ogrid[:self.ny, :self.nx]
        circle = (xx - cx)**2 + (yy - cy)**2 <= radius**2
        if self.exclude_mode:
            self.exclude_mask = np.logical_or(self.exclude_mask, circle)
            self.history.append(('exclude', circle))
            color = 'red'
        else:
            self.include_mask = np.logical_or(self.include_mask, circle)
            self.history.append(('include', circle))
            color = 'green'
        self.redraw(color, circle)

    def finish_polygon(self):
        """Close the polygon, add the mask, and redraw."""
        if len(self.polygon_verts) < 3:
            return
        # mgrid creates full 2D coordinate arrays
        yy, xx = np.mgrid[:self.ny, :self.nx]
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        path = Path(self.polygon_verts)          # matplotlib path for polygon
        poly_mask = path.contains_points(points).reshape(self.ny, self.nx)
        if self.exclude_mode:
            self.exclude_mask = np.logical_or(self.exclude_mask, poly_mask)
            self.history.append(('exclude', poly_mask))
            color = 'red'
        else:
            self.include_mask = np.logical_or(self.include_mask, poly_mask)
            self.history.append(('include', poly_mask))
            color = 'green'
        self.redraw(color, poly_mask)
        self.polygon_verts = []
        self.polygon_active = False

    def redraw(self, color=None, mask=None):
        """Redraw the image with current masks and optionally highlight a new shape."""
        self.ax.clear()
        self.ax.imshow(self.image, origin='lower', cmap='inferno')
        # Replot SDSS contours (they are stored as class variables)
        if self.sdss_reprojected is not None and self.sdss_contours is not None:
            self.ax.contour(self.sdss_reprojected, levels=self.sdss_contours,
                            colors='cyan', alpha=0.5, linewidths=0.8)
        # Plot include mask as green points
        if np.any(self.include_mask):
            y, x = np.where(self.include_mask)
            self.ax.scatter(x, y, s=1, c='green', alpha=0.3, label='Include')
        # Plot exclude mask as red points
        if np.any(self.exclude_mask):
            y, x = np.where(self.exclude_mask)
            self.ax.scatter(x, y, s=1, c='red', alpha=0.3, label='Exclude')
        # If a new shape was just added, plot it with its color
        if mask is not None and color is not None:
            y, x = np.where(mask)
            self.ax.scatter(x, y, s=1, c=color, alpha=0.5)
        self.update_title()
        self.fig.canvas.draw()

    def finish(self):
        plt.close(self.fig)

    def get_mask(self):
        """Return final mask: included spaxels that are not excluded."""
        return np.logical_and(self.include_mask, ~self.exclude_mask)


##############################################################
# Main reduction pipeline
##############################################################

def main():
    print("\n=== GCMS IFU Reduction (Interactive with SDSS overlay) ===\n")

    # ---- Step 0: Number of dithers ----
    while True:
        n_dither_str = input("Number of dither positions (3 or 6): ").strip()
        if n_dither_str in ['3','6']:
            n_dither = int(n_dither_str)
            break
        print("Please enter 3 or 6.")
    print(f"Using {n_dither} dither positions.\n")

    # ---- Step 1: Flat field and fiber trace extraction ----
    flat_path = input("Path to flat field FITS file: ").strip()
    flat = fits.getdata(flat_path)
    # Collapse along spectral axis to get a 1D spatial profile of fiber traces
    spatial_profile = np.sum(flat, axis=1)
    # Find peaks above 80th percentile, with a minimum distance of 5 pixels
    threshold = np.percentile(spatial_profile, 80)
    peaks, _ = find_peaks(spatial_profile, height=threshold, distance=5)
    n_fibers = len(peaks)
    print(f"Found {n_fibers} fiber traces.")

    # ---- Step 2: Wavelength calibration from arc lamp ----
    arc_path = input("Path to arc lamp FITS file: ").strip()
    arc_frame = fits.getdata(arc_path)
    line_file = input("Path to arc line file (two columns: pixel wavelength): ").strip()
    try:
        line_data = np.loadtxt(line_file)
        pixel_pos, known_wave = line_data[:,0], line_data[:,1]
    except Exception as e:
        print(f"Error reading arc line file: {e}")
        sys.exit(1)
    # Use the central fiber as reference (assuming it's well‑exposed)
    ref_spec = extract_fiber_spectra(arc_frame, peaks, width=5)[len(peaks)//2, :]
    # Fit a cubic polynomial to map pixel -> wavelength
    coeffs = np.polyfit(pixel_pos, known_wave, 3)
    wavelength = np.polyval(coeffs, np.arange(ref_spec.size))
    print(f"Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Angstrom")

    # ---- Step 3: Dither offsets (optional) ----
    use_offsets = input("Do you have dither offsets? (y/n): ").strip().lower() == 'y'
    offsets = [(0,0)] * n_dither
    if use_offsets:
        print(f"Enter {n_dither} offsets (dx dy) in cube pixels, one per line:")
        for i in range(n_dither):
            vals = input(f"Offset {i+1}: ").strip().split()
            if len(vals) == 2:
                offsets[i] = (float(vals[0]), float(vals[1]))
            else:
                print("Invalid, using (0,0)")

    # ---- Step 4: Science file selection ----
    sci_dir = input("Path to folder with science FITS files: ").strip()
    all_fits = glob.glob(os.path.join(sci_dir, "*.fits")) + glob.glob(os.path.join(sci_dir, "*.fit"))
    if len(all_fits) == 0:
        print("No FITS files found.")
        sys.exit(1)
    print("Found files:")
    for i, f in enumerate(all_fits):
        print(f"  {i}: {os.path.basename(f)}")
    indices = list(map(int, input(f"Select {n_dither} file indices (space-separated): ").strip().split()))
    if len(indices) != n_dither:
        print(f"Need {n_dither} indices.")
        sys.exit(1)
    sci_files = [all_fits[i] for i in indices]

    # ---- Step 5: Fiber coordinates (CSV or simulated hexagonal grid) ----
    calib_file = input("Path to fiber coordinates CSV (fiber_id,x,y) or Enter to simulate: ").strip()
    if calib_file and os.path.exists(calib_file):
        # CSV format: fiber_id, x_pixel, y_pixel (skip header row)
        fiber_coords = np.loadtxt(calib_file, delimiter=',', skiprows=1, usecols=(1,2))
        print(f"Loaded {len(fiber_coords)} fiber coordinates.")
    else:
        print("Simulating hexagonal grid (spacing 4.0 pixels).")
        spacing = 4.0   # approximate spacing between fibers in cube pixels
        coords = []
        i = 0
        # Hexagonal packing: rows are offset by half spacing, vertical spacing = spacing * sqrt(3)/2
        for y in np.arange(20, 80, spacing*0.866):
            row_offset = (i % 2) * (spacing/2)
            for x in np.arange(20 + row_offset, 80, spacing):
                if i < n_fibers:
                    coords.append((x, y))
                    i += 1
        fiber_coords = np.array(coords)
        print(f"Simulated {len(fiber_coords)} positions.")
    # Determine cube size from fiber coordinates, adding margin for dither shifts
    nx = int(np.max(fiber_coords[:,0]) + 20)
    ny = int(np.max(fiber_coords[:,1]) + 20)
    cube_shape = (nx, ny, len(wavelength))

    # ---- Step 6: Process each science exposure and build cubes ----
    all_cubes = []
    for fname, off in zip(sci_files, offsets):
        print(f"Processing {os.path.basename(fname)}...")
        sci_frame = fits.getdata(fname)
        fiber_spec = extract_fiber_spectra(sci_frame, peaks, width=5)
        cube = build_cube(fiber_spec, fiber_coords, cube_shape, off)
        all_cubes.append(cube)
    # Combine cubes with sigma‑clipping to reject cosmic rays (3 sigma)
    stack = np.stack(all_cubes, axis=0)            # shape: (n_dither, nx, ny, nw)
    combined_cube = sigma_clip(stack, axis=0, sigma=3, masked=False)
    combined_cube = np.nanmean(combined_cube, axis=0)   # average over dithers
    out_cube = input("Save combined cube as (e.g., cube.fits): ").strip()
    if out_cube:
        fits.writeto(out_cube, combined_cube, overwrite=True)
        print(f"Saved {out_cube}")

    # ---- Step 7: SDSS overlay (target coordinates, WCS, reprojection) ----
    target_name = input("Enter target name for SDSS overlay (or Enter to skip): ").strip()
    sdss_reprojected = None
    sdss_contours = None
    if target_name:
        ra, dec = get_object_coordinates(target_name)
        if ra is not None:
            confirm = input(f"Coordinates: RA={ra:.6f}, Dec={dec:.6f}. Use these? (y/n): ").strip().lower()
            if confirm == 'y':
                pix_scale = input("Approximate pixel scale of cube (arcsec/pixel) [default 1.0]: ").strip()
                try:
                    pix_scale = float(pix_scale)
                except:
                    pix_scale = 1.0
                # Generate a WCS for the cube (assumes center is target, no rotation)
                cube_wcs_header = generate_cube_wcs_header(ra, dec, cube_shape, pix_scale)
                # Fetch SDSS image and its WCS
                sdss_data, sdss_wcs = get_sdss_cutout_and_wcs(ra, dec, size_arcmin=2.0)
                if sdss_data is not None:
                    # Reproject SDSS image onto the cube's spatial grid
                    sdss_reprojected, _ = reproject_interp(
                        (sdss_data, sdss_wcs),
                        cube_wcs_header,
                        shape_out=cube_shape[:2]   # only spatial dimensions
                    )
                    # Determine contour levels from the reprojected image (5th to 95th percentile)
                    vmin, vmax = np.percentile(sdss_reprojected, (5,95))
                    sdss_contours = np.linspace(vmin, vmax, 10)
                    print("SDSS image reprojected to cube WCS.")
                else:
                    print("Could not retrieve SDSS image.")
            else:
                print("Skipping SDSS overlay.")
        else:
            print("Could not resolve target name. Skipping SDSS overlay.")
    else:
        print("Skipping SDSS overlay.")

    # ---- Step 8: Interactive region selection ----
    # Collapse cube along wavelength to create a 2D image for selection
    collapsed = np.nansum(combined_cube, axis=2)
    print("\n" + "="*60)
    print("REGION SELECTOR INSTRUCTIONS")
    print("  c : circle (click center, then edge)")
    print("  p : polygon (click vertices, Enter to close)")
    print("  e : toggle include (green) / exclude (red)")
    print("  u : undo last shape")
    print("  r : reset all masks")
    print("  Enter : finish and extract spectrum")
    print("="*60)
    input("Press Enter to open interactive window...")
    selector = RegionSelector(collapsed, sdss_reprojected, sdss_contours)
    plt.show()
    final_mask = selector.get_mask()
    if final_mask is None or not np.any(final_mask):
        print("No region selected. Exiting.")
        sys.exit(0)
    # Extract mean spectrum from all spaxels in the final mask
    region_spectrum = np.nanmean(combined_cube[final_mask], axis=0)

    # ---- Step 9: Emission line fitting ----
    lines = {'1': ('Hβ', 4861), '2': ('[O III]', 5007), '3': ('Hα', 6563),
             '4': ('[N II]', 6583), '5': ('[S II]', 6717), '6': ('[S II]', 6731)}
    print("\nSelect emission line to fit:")
    for k, (name, wl) in lines.items():
        print(f"  {k}. {name} ({wl} Angstrom)")
    choice = input("Number: ").strip()
    line_name, line_wave = lines.get(choice, ('Hα', 6563))
    idx_cen = np.argmin(np.abs(wavelength - line_wave))
    guess_center = wavelength[idx_cen]
    result = fit_gaussian_emission(wavelength, region_spectrum, guess_center)
    print(result.fit_report())

    # Plot spectrum and fit
    plt.figure(figsize=(10,5))
    plt.plot(wavelength, region_spectrum, label='Data')
    plt.plot(wavelength, result.best_fit, 'r-', label='Fit')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux')
    plt.title(f'{line_name} fit from selected region')
    plt.legend()
    out_plot = input("Save plot (e.g., line_fit.png): ").strip()
    if out_plot:
        plt.savefig(out_plot)
    plt.show()

    out_spec = input("Save spectrum as (e.g., spectrum.dat): ").strip()
    if out_spec:
        np.savetxt(out_spec, np.column_stack([wavelength, region_spectrum]),
                   header="wavelength flux")
        print(f"Saved {out_spec}")

    print("Done.")


if __name__ == "__main__":
    main()
