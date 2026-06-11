#!/usr/bin/env python3
"""
Unified GCMS IFU Reduction Pipeline

Merges features from:
- gcms_reduce_final.py: Full calibration (bias, flat, arc, sky), config file support.
- gcms_reduce.py (repository): SDSS overlay, coordinate resolution, advanced region selector.

Handles 3- or 6-dither observations, builds a 3D datacube, and provides interactive
region selection and emission line fitting.

Usage:
    # Using a configuration file (recommended for full control)
    python gcms_reduce.py --config myconfig.cfg

    # Using command-line arguments (simpler, directory-based)
    python gcms_reduce.py --sci_dir science/ --bias_dir biases/ --flat_dir flats/ --arc_dir arcs/ \\
        --n_dither 3 --fiber_coords fibers.csv --out_cube cube.fits --object_name "NGC 1277"
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
import astropy.units as u
from scipy.signal import find_peaks
from lmfit.models import GaussianModel, LinearModel, ConstantModel
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned
from astroquery.sdss import SDSS
from reproject import reproject_interp

# =========================================================
# Configuration and helpers
# =========================================================

def load_config(config_path):
    cfg = {}
    if not config_path or not os.path.exists(config_path):
        return cfg
    with open(config_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    return cfg

def cfg_get(cfg, key, prompt=None, default=None):
    if key in cfg and cfg[key] != "":
        print(f"{key} = {cfg[key]}  (from config)")
        return cfg[key]
    if prompt is None:
        return default
    if default is not None:
        value = input(f"{prompt} [{default}]: ").strip()
        return value if value else default
    return input(f"{prompt}: ").strip()

def parse_int_list(text):
    text = str(text).strip()
    if not text:
        return []
    return list(map(int, text.split()))

def parse_bool(text):
    return str(text).strip().lower() in {"y", "yes", "true", "1"}

def parse_offsets(text, n_dither):
    """Parse offset string like '0,0;2,0;1,2' into list of (dx,dy)."""
    text = str(text).strip()
    if not text:
        return [(0.0, 0.0)] * n_dither
    parts = [p.strip() for p in text.split(";") if p.strip()]
    offsets = []
    for p in parts:
        vals = p.split(",")
        if len(vals) != 2:
            raise ValueError("Each offset must be 'dx,dy'")
        offsets.append((float(vals[0]), float(vals[1])))
    if len(offsets) != n_dither:
        raise ValueError(f"Need exactly {n_dither} offsets")
    return offsets

def basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]

def infer_frame_number(path):
    name = basename_noext(path).lower()
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None

def choose_nearest_file(target_file, candidate_files):
    target_num = infer_frame_number(target_file)
    if target_num is None or not candidate_files:
        return None
    scored = []
    for cf in candidate_files:
        n = infer_frame_number(cf)
        if n is not None:
            scored.append((abs(n - target_num), n, cf))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]

def read_fits_float(path):
    return fits.getdata(path).astype(float)

def combine_frames(file_list, method="median"):
    if not file_list:
        return None
    stack = np.stack([read_fits_float(f) for f in file_list], axis=0)
    if method == "mean":
        return np.nanmean(stack, axis=0)
    return np.nanmedian(stack, axis=0)

def make_master_bias(bias_files):
    return combine_frames(bias_files, method="median")

def make_master_flat(flat_files, master_bias=None):
    if not flat_files:
        return None, None
    flat_raw = combine_frames(flat_files, method="median")
    flat_corr = flat_raw.copy()
    if master_bias is not None and flat_corr.shape == master_bias.shape:
        flat_corr = flat_corr - master_bias
    finite = np.isfinite(flat_corr)
    med = np.nanmedian(flat_corr[finite]) if np.any(finite) else np.nan
    if not np.isfinite(med) or med == 0:
        raise ValueError("Cannot normalize master flat")
    master_flat_norm = flat_corr / med
    bad = ~np.isfinite(master_flat_norm) | (master_flat_norm <= 0)
    master_flat_norm[bad] = 1.0
    return flat_corr, master_flat_norm

def make_master_arc(arc_files, master_bias=None, master_flat_norm=None):
    if not arc_files:
        return None
    arc_raw = combine_frames(arc_files, method="median")
    arc_corr = arc_raw.copy()
    if master_bias is not None and arc_corr.shape == master_bias.shape:
        arc_corr = arc_corr - master_bias
    if master_flat_norm is not None and arc_corr.shape == master_flat_norm.shape:
        arc_corr = arc_corr / master_flat_norm
    return arc_corr

def calibrate_frame(frame, master_bias=None, master_flat_norm=None):
    out = frame.astype(float).copy()
    if master_bias is not None and out.shape == master_bias.shape:
        out = out - master_bias
    if master_flat_norm is not None and out.shape == master_flat_norm.shape:
        out = out / master_flat_norm
    return out

# =========================================================
# Spectral extraction and cube building
# =========================================================

def extract_fiber_spectra(frame, peak_positions, width=5):
    n_fibers = len(peak_positions)
    n_spec = frame.shape[1]
    spectra = np.zeros((n_fibers, n_spec), dtype=float)
    for i, p in enumerate(peak_positions):
        x0 = max(0, int(round(p - width // 2)))
        x1 = min(frame.shape[0], int(round(p + width // 2 + 1)))
        spectra[i, :] = np.nansum(frame[x0:x1, :], axis=0)
    return spectra

def build_cube(fiber_spectra, fiber_coords, cube_shape, dither_offset=(0,0)):
    nx, ny, nw = cube_shape
    cube = np.full((nx, ny, nw), np.nan, dtype=float)
    n_use = min(len(fiber_spectra), len(fiber_coords))
    for i in range(n_use):
        x, y = fiber_coords[i]
        x_shifted = int(round(x + dither_offset[0]))
        y_shifted = int(round(y + dither_offset[1]))
        if 0 <= x_shifted < nx and 0 <= y_shifted < ny:
            cube[x_shifted, y_shifted, :] = fiber_spectra[i, :]
    return cube

# =========================================================
# Wavelength calibration and sky zero‑point
# =========================================================

def find_local_peak_near_line(wavelength, spectrum, expected_line, search_window=100.0):
    mask = np.isfinite(wavelength) & np.isfinite(spectrum)
    mask &= (wavelength >= expected_line - search_window) & (wavelength <= expected_line + search_window)
    wave_local = wavelength[mask]
    spec_local = spectrum[mask]
    if len(wave_local) < 5:
        raise ValueError(f"Not enough points near {expected_line:.2f}")
    prom = max(3.0 * np.nanstd(spec_local), 5.0)
    peaks, _ = find_peaks(spec_local, prominence=prom)
    if len(peaks) == 0:
        raise ValueError(f"No peak near {expected_line:.2f}")
    best_peak = peaks[np.argmax(spec_local[peaks])]
    return wave_local[best_peak]

def fit_gaussian_emission(wavelength, spectrum, guess_center, guess_sigma=2.0, window=20,
                          center_tol=8.0, sigma_min=0.8, sigma_max=6.0):
    mask = np.isfinite(wavelength) & np.isfinite(spectrum)
    mask &= (wavelength >= guess_center - window) & (wavelength <= guess_center + window)
    wave_fit = wavelength[mask]
    spec_fit = spectrum[mask]
    if len(wave_fit) < 8:
        raise ValueError(f"Not enough points in window around {guess_center:.1f}")

    # estimate continuum from sidebands
    cont_mask = (wave_fit < guess_center - 4.0) | (wave_fit > guess_center + 4.0)
    if np.sum(cont_mask) >= 4:
        p = np.polyfit(wave_fit[cont_mask], spec_fit[cont_mask], 1)
        slope_guess, intercept_guess = p[0], p[1]
    else:
        slope_guess, intercept_guess = 0.0, np.nanmedian(spec_fit)

    cont_guess = slope_guess * wave_fit + intercept_guess
    line_only = spec_fit - cont_guess
    amp_guess = np.nanmax(line_only)
    if not np.isfinite(amp_guess) or amp_guess <= 0:
        amp_guess = max(np.nanstd(spec_fit), 1.0)
    peak_idx = np.nanargmax(line_only)
    center_guess = wave_fit[peak_idx]

    model = GaussianModel(prefix="g_") + LinearModel(prefix="b_")
    params = model.make_params(
        g_amplitude=amp_guess,
        g_center=center_guess,
        g_sigma=guess_sigma,
        b_slope=slope_guess,
        b_intercept=intercept_guess,
    )
    params["g_amplitude"].min = 0.0
    params["g_center"].min = guess_center - center_tol
    params["g_center"].max = guess_center + center_tol
    params["g_sigma"].min = sigma_min
    params["g_sigma"].max = sigma_max

    result = model.fit(spec_fit, params, x=wave_fit)
    return result, wave_fit, spec_fit

def build_master_sky_spectrum(sky_files, peaks, master_bias=None, master_flat_norm=None):
    if not sky_files:
        raise ValueError("No sky files provided.")
    sky_specs = []
    for sf in sky_files:
        sky_frame = calibrate_frame(read_fits_float(sf), master_bias, master_flat_norm)
        fiber_spec = extract_fiber_spectra(sky_frame, peaks, width=5)
        sky_1d = np.nanmedian(fiber_spec, axis=0)
        sky_specs.append(sky_1d)
    return np.nanmedian(np.stack(sky_specs, axis=0), axis=0)

def measure_sky_zero_point(wavelength, sky_spectrum,
                           sky_lines=(5577.34, 6300.30, 6363.78),
                           search_window=60.0, fit_window=8.0):
    measurements = []
    for line in sky_lines:
        try:
            peak_guess = find_local_peak_near_line(wavelength, sky_spectrum, line, search_window)
            result, _, _ = fit_gaussian_emission(wavelength, sky_spectrum,
                                                 guess_center=peak_guess, guess_sigma=2.0,
                                                 window=fit_window, center_tol=5.0,
                                                 sigma_min=0.5, sigma_max=5.0)
            measured = result.params["g_center"].value
            delta = line - measured
            measurements.append((line, measured, delta))
        except Exception as e:
            print(f"Sky line {line:.2f} failed: {e}")
            continue
    if not measurements:
        raise ValueError("Could not measure any sky lines")
    delta_lambda = np.median([m[2] for m in measurements])
    return delta_lambda, measurements

# =========================================================
# SDSS / WCS / coordinate helpers
# =========================================================

def get_object_coordinates(object_name):
    """Resolve object name to J2000 equatorial coordinates using SIMBAD then NED."""
    print(f"\nResolving coordinates for '{object_name}'...")
    # Try SIMBAD first
    try:
        Simbad.add_votable_fields('ra(d)', 'dec(d)')
        result = Simbad.query_object(object_name)
        if result is not None:
            ra = result['RA_d'][0]
            dec = result['DEC_d'][0]
            print(f"SIMBAD found: RA = {ra:.6f}, Dec = {dec:.6f}")
            return ra, dec
    except Exception as e:
        print(f"SIMBAD query failed: {e}")
    # Fallback to NED
    try:
        result = Ned.query_object(object_name)
        if result is not None and len(result) > 0:
            ra_str = result['RA'][0]
            dec_str = result['DEC'][0]
            coords = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
            ra = coords.ra.deg
            dec = coords.dec.deg
            print(f"NED found: RA = {ra:.6f}, Dec = {dec:.6f}")
            return ra, dec
    except Exception as e:
        print(f"NED query failed: {e}")
    return None, None

def get_sdss_cutout_and_wcs(ra, dec, size_arcmin=2.0, band='r'):
    """Retrieve an SDSS image cutout and its WCS object."""
    try:
        hdulist = SDSS.get_images(coordinates=SkyCoord(ra, dec, unit='deg'),
                                  radius=size_arcmin * u.arcmin, band=band)
        if not hdulist:
            print("No SDSS data found.")
            return None, None
        hdu = hdulist[0]
        data = hdu.data
        if data.ndim == 3:
            data = np.nanmedian(data, axis=0)
        wcs = WCS(hdu.header)
        return data, wcs
    except Exception as e:
        print(f"Error fetching SDSS image: {e}")
        return None, None

def generate_cube_wcs_header(ra_center, dec_center, cube_shape, pixel_scale_arcsec=1.0):
    """Create a FITS header with a simple TAN projection WCS for the cube."""
    nx, ny, nw = cube_shape[:3]
    header = fits.Header()
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = nw
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CTYPE3'] = 'WAVE'
    header['CRPIX1'] = nx / 2.0 + 0.5
    header['CRPIX2'] = ny / 2.0 + 0.5
    header['CRPIX3'] = 1.0
    header['CRVAL1'] = ra_center
    header['CRVAL2'] = dec_center
    header['CRVAL3'] = 5000.0  # placeholder
    header['CDELT1'] = -pixel_scale_arcsec / 3600.0
    header['CDELT2'] = pixel_scale_arcsec / 3600.0
    header['CDELT3'] = 1.0
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CUNIT3'] = 'Angstrom'
    return header

def save_cube_with_wcs(filename, cube_xyz, wavelength, wcs_header):
    """Save cube in (nx,ny,nw) order as FITS with (nw,ny,nx) and WCS."""
    nx, ny, nw = cube_xyz.shape
    cube_fits = np.transpose(cube_xyz, (2, 1, 0))
    wcs_header['NAXIS'] = 3
    wcs_header['NAXIS1'] = nx
    wcs_header['NAXIS2'] = ny
    wcs_header['NAXIS3'] = nw
    wcs_header['CDELT3'] = wavelength[1] - wavelength[0]
    wcs_header['CRVAL3'] = wavelength[0]
    fits.writeto(filename, cube_fits, wcs_header, overwrite=True)

# =========================================================
# Interactive region selector with SDSS overlay
# =========================================================

class RegionSelector:
    def __init__(self, image, sdss_data=None, sdss_wcs=None, cube_wcs=None):
        self.image = np.array(image, dtype=float)
        self.sdss_data = sdss_data
        self.sdss_wcs = sdss_wcs
        self.cube_wcs = cube_wcs
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.include_mask = np.zeros(self.image.shape, dtype=bool)
        self.exclude_mask = np.zeros(self.image.shape, dtype=bool)
        self.exclude_mode = False
        self.selector = None
        self.circle_mode = False
        self.polygon_mode = False
        self.polygon_verts = []
        self.polygon_markers = []
        self.polygon_cid = None
        self._draw_base()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show(block=False)

    def _draw_base(self):
        self.ax.clear()
        # Show the collapsed cube image
        self.ax.imshow(self.image, origin="lower", cmap="inferno", label="GCMS cube")
        # Overlay SDSS contours if available
        if self.sdss_data is not None and self.sdss_wcs is not None and self.cube_wcs is not None:
            try:
                sdss_reprojected, _ = reproject_interp(self.sdss_data, self.cube_wcs,
                                                       shape_out=self.image.shape)
                levels = np.percentile(sdss_reprojected[np.isfinite(sdss_reprojected)],
                                       [70, 80, 90])
                self.ax.contour(sdss_reprojected, levels=levels, colors='cyan',
                                linewidths=0.8, alpha=0.7, origin='lower')
            except Exception as e:
                print(f"Could not overlay SDSS contours: {e}")
        # Draw existing masks
        if np.any(self.include_mask):
            yy, xx = np.where(self.include_mask)
            self.ax.scatter(xx, yy, s=8, c="lime", alpha=0.35)
        if np.any(self.exclude_mask):
            yy, xx = np.where(self.exclude_mask)
            self.ax.scatter(xx, yy, s=8, c="red", alpha=0.35)
        mode = "EXCLUDE" if self.exclude_mode else "INCLUDE"
        tool = "polygon" if self.polygon_mode else ("circle" if self.circle_mode else "none")
        self.ax.set_title(f"Mode: {mode} | Tool: {tool}  (c=circle, p=polygon, e=toggle, r=reset, Enter=finish)")
        self.fig.canvas.draw_idle()

    def _deactivate_selector(self):
        if self.selector is not None:
            try:
                self.selector.set_active(False)
            except Exception:
                pass
            self.selector = None
        self.circle_mode = False

    def _disconnect_polygon(self):
        if self.polygon_cid is not None:
            self.fig.canvas.mpl_disconnect(self.polygon_cid)
            self.polygon_cid = None
        self.polygon_mode = False
        self._clear_polygon_markers()

    def _clear_polygon_markers(self):
        for artist in self.polygon_markers:
            try:
                artist.remove()
            except Exception:
                pass
        self.polygon_markers = []
        self.polygon_verts = []

    def on_key(self, event):
        if event.key == "e":
            self.exclude_mode = not self.exclude_mode
            self._draw_base()
        elif event.key == "r":
            self.include_mask.fill(False)
            self.exclude_mask.fill(False)
            self._draw_base()
            print("All selections reset.")
        elif event.key == "c":
            self._disconnect_polygon()
            self._deactivate_selector()
            self.circle_mode = True
            self.selector = RectangleSelector(self.ax, self.circle_select, useblit=True, button=[1],
                                              minspanx=2, minspany=2, spancoords="pixels", interactive=False)
            print("Circle mode active. Click and drag.")
        elif event.key == "p":
            self._deactivate_selector()
            self._disconnect_polygon()
            self.polygon_mode = True
            self.polygon_verts = []
            self.polygon_markers = []
            self.polygon_cid = self.fig.canvas.mpl_connect("button_press_event", self.polygon_click)
            self._draw_base()
            print("Polygon mode active. Click to add vertices. Press Enter to close polygon.")
        elif event.key == "escape":
            if self.polygon_mode:
                self._clear_polygon_markers()
                self._draw_base()
                print("Polygon cleared.")
        elif event.key == "enter":
            if self.polygon_mode and len(self.polygon_verts) >= 3:
                self.finish_polygon()
            else:
                self.finish()

    def circle_select(self, eclick, erelease):
        if None in (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata):
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        radius = np.hypot(x2 - x1, y2 - y1) / 2.0
        ny, nx = self.image.shape
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        if self.exclude_mode:
            self.exclude_mask |= circle
        else:
            self.include_mask |= circle
        self._draw_base()

    def polygon_click(self, event):
        if not self.polygon_mode or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.polygon_verts.append((event.xdata, event.ydata))
        marker, = self.ax.plot(event.xdata, event.ydata, "ro" if self.exclude_mode else "go", ms=5)
        self.polygon_markers.append(marker)
        if len(self.polygon_verts) > 1:
            x_prev, y_prev = self.polygon_verts[-2]
            line, = self.ax.plot([x_prev, event.xdata], [y_prev, event.ydata],
                                 "r-" if self.exclude_mode else "g-", lw=1)
            self.polygon_markers.append(line)
        self.fig.canvas.draw_idle()

    def finish_polygon(self):
        if len(self.polygon_verts) < 3:
            print("Need at least 3 vertices.")
            return
        ny, nx = self.image.shape
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.column_stack([xx.ravel(), yy.ravel()])
        path = Path(self.polygon_verts)
        poly_mask = path.contains_points(points).reshape(ny, nx)
        if self.exclude_mode:
            self.exclude_mask |= poly_mask
        else:
            self.include_mask |= poly_mask
        self._disconnect_polygon()
        self._draw_base()

    def finish(self):
        self._deactivate_selector()
        self._disconnect_polygon()
        plt.close(self.fig)

    def get_mask(self):
        return self.include_mask & (~self.exclude_mask)

# =========================================================
# Main reduction routine
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Unified GCMS IFU Reduction Pipeline")
    parser.add_argument("--config", help="Configuration file (overrides command‑line options)")
    parser.add_argument("--sci_dir", help="Directory with science FITS files (must contain exactly n_dither files)")
    parser.add_argument("--bias_dir", help="Directory with bias FITS files")
    parser.add_argument("--flat_dir", help="Directory with flat FITS files")
    parser.add_argument("--arc_dir", help="Directory with arc FITS files")
    parser.add_argument("--sky_dir", help="Directory with sky FITS files (optional)")
    parser.add_argument("--n_dither", type=int, choices=[3,6], help="Number of dither positions (3 or 6)")
    parser.add_argument("--offsets", help="Dither offsets as 'dx1,dy1;dx2,dy2;...' (semicolon‑separated)")
    parser.add_argument("--fiber_coords", help="CSV file with fiber coordinates (fiber_id,x,y)")
    parser.add_argument("--out_cube", help="Output cube filename (FITS)")
    parser.add_argument("--object_name", help="Object name for coordinates and SDSS overlay")
    parser.add_argument("--ra", type=float, help="RA center (deg) – overrides object_name")
    parser.add_argument("--dec", type=float, help="Dec center (deg)")
    parser.add_argument("--pixscale", type=float, default=1.0, help="Pixel scale (arcsec/pixel)")
    parser.add_argument("--coeff_file", help="Wavelength coefficient file (txt, poly coefficients)")
    parser.add_argument("--arc_lines", help="Arc line file (pixel wavelength) for calibration")
    parser.add_argument("--use_sky_zero", action="store_true", help="Apply zero‑point correction from sky lines")
    parser.add_argument("--sky_zero_lines", default="5577.34 6300.30 6363.78", help="Sky lines for zero‑point correction")
    parser.add_argument("--sky_zero_window", type=float, default=8.0, help="Fit window for sky lines (Angstrom)")
    parser.add_argument("--z", type=float, default=0.0, help="Redshift for line fitting")
    parser.add_argument("--sdss_size", type=float, default=2.0, help="SDSS cutout size in arcminutes")
    parser.add_argument("--sdss_band", default="r", help="SDSS band (u,g,r,i,z)")
    args = parser.parse_args()

    # Load config file if provided
    cfg = load_config(args.config) if args.config else {}

    # Helper to get value: command line > config > default
    def get_val(key, cmd_val, prompt=None, default=None):
        if cmd_val is not None:
            return cmd_val
        return cfg_get(cfg, key, prompt, default)

    sci_dir = get_val("sci_dir", args.sci_dir, "Science directory")
    bias_dir = get_val("bias_dir", args.bias_dir, "Bias directory", "")
    flat_dir = get_val("flat_dir", args.flat_dir, "Flat directory")
    arc_dir = get_val("arc_dir", args.arc_dir, "Arc directory")
    sky_dir = get_val("sky_dir", args.sky_dir, "Sky directory (optional)", "")
    n_dither = int(get_val("n_dither", args.n_dither, "Number of dither positions (3 or 6)"))
    offsets_str = get_val("offsets", args.offsets, "Dither offsets (e.g., '0,0;2,0;1,2')", "")
    fiber_coords_file = get_val("fiber_coords", args.fiber_coords, "Fiber coordinates CSV")
    out_cube = get_val("out_cube", args.out_cube, "Output cube filename")
    object_name = get_val("object_name", args.object_name, "Object name (e.g., NGC 1277)", "")
    ra_cmd = args.ra
    dec_cmd = args.dec
    pixscale = float(get_val("pixscale", args.pixscale, "Pixel scale (arcsec/pix)", "1.0"))
    coeff_file = get_val("coeff_file", args.coeff_file, "Wavelength coeff file (optional)", "")
    arc_lines_file = get_val("arc_lines", args.arc_lines, "Arc line file (pixel wavelength)", "")
    use_sky_zero = parse_bool(get_val("use_sky_zero", str(args.use_sky_zero), "Use sky zero‑point correction? (y/n)", "n"))
    sky_zero_lines = [float(x) for x in get_val("sky_zero_lines", args.sky_zero_lines, "Sky zero‑point lines", "5577.34 6300.30 6363.78").split()]
    sky_zero_window = float(get_val("sky_zero_window", args.sky_zero_window, "Sky line fit window (Angstrom)", "8"))
    z = float(get_val("z", args.z, "Redshift (z)", "0"))
    sdss_size = float(get_val("sdss_size", args.sdss_size, "SDSS cutout size (arcmin)", "2.0"))
    sdss_band = get_val("sdss_band", args.sdss_band, "SDSS band", "r")

    # Resolve coordinates
    if ra_cmd is not None and dec_cmd is not None:
        ra_center, dec_center = ra_cmd, dec_cmd
        print(f"Using user-provided coordinates: RA = {ra_center:.6f}, Dec = {dec_center:.6f}")
    elif object_name:
        ra_center, dec_center = get_object_coordinates(object_name)
        if ra_center is None or dec_center is None:
            print("Could not resolve object name. Exiting.")
            sys.exit(1)
    else:
        ra_center, dec_center = 331.7537, 10.2309  # default to NGC 7212
        print(f"No coordinates provided. Using default: RA = {ra_center}, Dec = {dec_center}")

    # Find files in directories
    def find_fits(dirpath):
        if not dirpath or not os.path.isdir(dirpath):
            return []
        return sorted(glob.glob(os.path.join(dirpath, "*.fits")) + glob.glob(os.path.join(dirpath, "*.fit")))

    sci_files = find_fits(sci_dir)
    bias_files = find_fits(bias_dir)
    flat_files = find_fits(flat_dir)
    arc_files = find_fits(arc_dir)
    sky_files = find_fits(sky_dir)

    if len(sci_files) != n_dither:
        print(f"Error: need exactly {n_dither} science files in {sci_dir}, found {len(sci_files)}")
        sys.exit(1)

    print(f"Science files: {[os.path.basename(f) for f in sci_files]}")
    if bias_files:
        print(f"Bias files: {[os.path.basename(f) for f in bias_files]}")
    if flat_files:
        print(f"Flat files: {[os.path.basename(f) for f in flat_files]}")
    if arc_files:
        print(f"Arc files: {[os.path.basename(f) for f in arc_files]}")
    if sky_files:
        print(f"Sky files: {[os.path.basename(f) for f in sky_files]}")

    # Build master calibration frames
    print("\nBuilding master bias...")
    master_bias = make_master_bias(bias_files) if bias_files else None
    print("Building master flat...")
    master_flat_corr, master_flat_norm = make_master_flat(flat_files, master_bias) if flat_files else (None, None)
    print("Building master arc...")
    master_arc = make_master_arc(arc_files, master_bias, master_flat_norm) if arc_files else None

    # Find fiber traces from master flat (or arc if flat missing)
    if master_flat_corr is not None:
        spatial_profile = np.nansum(master_flat_corr, axis=1)
    elif master_arc is not None:
        spatial_profile = np.nansum(master_arc, axis=1)
    else:
        print("Error: need either a flat or an arc frame to locate fibers")
        sys.exit(1)

    threshold = np.percentile(spatial_profile[np.isfinite(spatial_profile)], 80)
    peaks, _ = find_peaks(spatial_profile, height=threshold, distance=5)
    n_fibers = len(peaks)
    print(f"Found {n_fibers} fiber traces.")

    # Wavelength calibration
    if coeff_file and os.path.exists(coeff_file):
        coeffs = np.loadtxt(coeff_file).flatten()
        ref_spec = extract_fiber_spectra(master_arc, peaks, width=5)[len(peaks)//2, :] if master_arc is not None else None
        wavelength = np.polyval(coeffs, np.arange(ref_spec.size if ref_spec is not None else 2048))
        print(f"Loaded wavelength coefficients from {coeff_file}")
    elif arc_lines_file and os.path.exists(arc_lines_file):
        if master_arc is None:
            print("Error: arc frame required for wavelength calibration")
            sys.exit(1)
        ref_spec = extract_fiber_spectra(master_arc, peaks, width=5)[len(peaks)//2, :]
        line_data = np.loadtxt(arc_lines_file)
        pixel_pos = line_data[:, 0]
        known_wave = line_data[:, 1]
        coeffs = np.polyfit(pixel_pos, known_wave, 3)
        wavelength = np.polyval(coeffs, np.arange(ref_spec.size))
        print(f"Derived wavelength solution from {arc_lines_file}")
    else:
        print("Error: need either --coeff_file or --arc_lines for wavelength calibration")
        sys.exit(1)

    # Optional sky zero‑point correction
    if use_sky_zero and sky_files:
        try:
            master_sky_spec = build_master_sky_spectrum(sky_files, peaks, master_bias, master_flat_norm)
            sky_delta, measures = measure_sky_zero_point(wavelength, master_sky_spec,
                                                         sky_lines=sky_zero_lines,
                                                         search_window=100,
                                                         fit_window=sky_zero_window)
            print(f"Sky zero‑point correction: {sky_delta:+.2f} Angstrom")
            wavelength += sky_delta
        except Exception as e:
            print(f"Sky zero‑point correction failed: {e}")

    print(f"Wavelength range: {wavelength[0]:.1f} – {wavelength[-1]:.1f} Angstrom")

    # Load fiber coordinates
    if not os.path.exists(fiber_coords_file):
        print(f"Error: fiber coordinate file not found: {fiber_coords_file}")
        sys.exit(1)
    fiber_coords = np.loadtxt(fiber_coords_file, delimiter=",", skiprows=1, usecols=(1,2))
    nx = int(np.nanmax(fiber_coords[:, 0]) + 20)
    ny = int(np.nanmax(fiber_coords[:, 1]) + 20)
    cube_shape = (nx, ny, len(wavelength))

    # Parse dither offsets
    offsets = parse_offsets(offsets_str, n_dither) if offsets_str else [(0,0)] * n_dither

    # Process each science exposure
    all_cubes = []
    for idx, (sci_file, offset) in enumerate(zip(sci_files, offsets)):
        print(f"Processing {os.path.basename(sci_file)} (offset {offset})...")
        sci_frame = calibrate_frame(read_fits_float(sci_file), master_bias, master_flat_norm)

        # Sky subtraction: find nearest sky frame by number
        if sky_files:
            matched_sky = choose_nearest_file(sci_file, sky_files)
            if matched_sky:
                print(f"  Using sky: {os.path.basename(matched_sky)}")
                sky_frame = calibrate_frame(read_fits_float(matched_sky), master_bias, master_flat_norm)
                if sci_frame.shape == sky_frame.shape:
                    sci_frame = sci_frame - sky_frame
                else:
                    print("  WARNING: shape mismatch, skipping sky subtraction")
            else:
                print("  No matching sky frame found")

        fiber_spec = extract_fiber_spectra(sci_frame, peaks, width=5)
        cube = build_cube(fiber_spec, fiber_coords, cube_shape, offset)
        all_cubes.append(cube)

    # Combine cubes with sigma clipping
    stack = np.stack(all_cubes, axis=0)
    clipped = sigma_clip(stack, axis=0, sigma=3, masked=True)
    combined_cube = np.nanmean(clipped.filled(np.nan), axis=0)

    # Generate WCS header for the cube
    cube_wcs_header = generate_cube_wcs_header(ra_center, dec_center, cube_shape, pixscale)

    # Save the cube
    if out_cube:
        save_cube_with_wcs(out_cube, combined_cube, wavelength, cube_wcs_header)
        print(f"Saved cube: {out_cube}")

    # Get SDSS cutout for overlay
    sdss_data, sdss_wcs = get_sdss_cutout_and_wcs(ra_center, dec_center, size_arcmin=sdss_size, band=sdss_band)

    # Interactive region selection with SDSS overlay
    collapsed = np.nansum(combined_cube, axis=2)
    print("\nOpening interactive selector. Use: c=circle, p=polygon, e=toggle include/exclude, r=reset, Enter=finish")
    selector = RegionSelector(collapsed, sdss_data=sdss_data, sdss_wcs=sdss_wcs, cube_wcs=cube_wcs_header)
    plt.show()
    final_mask = selector.get_mask()

    if not np.any(final_mask):
        print("No region selected. Exiting.")
        sys.exit(0)

    selected_spaxels = combined_cube[final_mask]
    region_spectrum = np.nanmean(selected_spaxels, axis=0)

    # Emission line fitting
    lines_rest = {
        "1": ("Hβ", 4861.33),
        "2": ("[O III]", 5006.84),
        "3": ("Hα", 6562.80),
        "4": ("[N II]", 6583.45),
        "5": ("[S II] 6716", 6716.44),
        "6": ("[S II] 6731", 6730.82),
        "7": ("[O I] sky", 5577.34),
    }
    lines = {}
    for key, (name, rest_wl) in lines_rest.items():
        obs_wl = rest_wl if key == "7" else rest_wl * (1.0 + z)
        lines[key] = (name, obs_wl)

    print(f"\nUsing redshift z = {z:.6f}")
    while True:
        print("\nSelect emission line to fit:")
        for k, (name, wl) in lines.items():
            print(f"  {k}. {name} ({wl:.2f} Angstrom)")
        print("  q. quit")
        choice = input("Number: ").strip().lower()
        if choice == "q":
            break
        line_name, line_wave = lines.get(choice, lines["3"])
        idx_cen = np.argmin(np.abs(wavelength - line_wave))
        guess_center = wavelength[idx_cen]
        try:
            result, wave_local, spec_local = fit_gaussian_emission(wavelength, region_spectrum,
                                                                   guess_center, guess_sigma=2, window=10)
        except Exception as e:
            print(f"Fit failed: {e}")
            continue
        print(result.fit_report())
        plt.figure(figsize=(10,5))
        plt.plot(wavelength, region_spectrum, "0.75", lw=1, label="Full spectrum")
        plt.plot(wave_local, spec_local, "b-", lw=1.5, label="Local window")
        plt.plot(wave_local, result.best_fit, "r-", lw=2, label="Fit")
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Flux")
        plt.title(f"{line_name} fit from selected region")
        plt.legend()
        out_plot = input("Save plot (e.g., line_fit.png, Enter to skip): ").strip()
        if out_plot:
            plt.savefig(out_plot, dpi=150)
        plt.show()
        out_spec = input("Save spectrum as (e.g., spectrum.dat, Enter to skip): ").strip()
        if out_spec:
            np.savetxt(out_spec, np.column_stack([wavelength, region_spectrum]), header="wavelength flux")

    print("Done.")

if __name__ == "__main__":
    main()
