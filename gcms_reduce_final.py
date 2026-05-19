#!/usr/bin/env python3

import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.signal import find_peaks
from lmfit.models import GaussianModel, LinearModel
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector


# =========================================================
# Config helpers
# =========================================================

def load_config(config_path):
    cfg = {}
    if not config_path:
        return cfg
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

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

def save_cube_for_qfitsview(filename, cube_xyz, wavelength, ra_deg, dec_deg, pixscale_arcsec=1.0):
    """
    cube_xyz shape: (nx, ny, nw)
    Save as FITS cube with data order (nw, ny, nx) for QFitsView/DS9-like viewers.
    """
    nx, ny, nw = cube_xyz.shape

    cube_fits = np.transpose(cube_xyz, (2, 1, 0))  # -> (wave, y, x)

    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nw

    # Spatial axes
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CRVAL1"] = ra_deg
    hdr["CRVAL2"] = dec_deg
    hdr["CRPIX1"] = nx / 2.0
    hdr["CRPIX2"] = ny / 2.0
    hdr["CDELT1"] = -pixscale_arcsec / 3600.0
    hdr["CDELT2"] =  pixscale_arcsec / 3600.0

    # Spectral axis
    hdr["CTYPE3"] = "WAVE"
    hdr["CUNIT3"] = "Angstrom"
    hdr["CRPIX3"] = 1.0
    hdr["CRVAL3"] = float(wavelength[0])
    hdr["CDELT3"] = float(wavelength[1] - wavelength[0])

    fits.writeto(filename, cube_fits, hdr, overwrite=True)


def cfg_get(cfg, key, prompt=None, default=None):
    if key in cfg and cfg[key] != "":
        value = cfg[key]
        print(f"{key} = {value}  (from config)")
        return value
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
    """
    Format:
    dither_offsets = 0 0; 0 0; 0 0; 0 0; 0 0; 0 0
    """
    text = str(text).strip()
    if not text:
        return [(0.0, 0.0)] * n_dither

    parts = [p.strip() for p in text.split(";") if p.strip()]
    offsets = []
    for p in parts:
        vals = p.split()
        if len(vals) != 2:
            raise ValueError("Each dither offset must have exactly two numbers: dx dy")
        offsets.append((float(vals[0]), float(vals[1])))

    if len(offsets) != n_dither:
        raise ValueError(f"Need exactly {n_dither} dither offsets.")
    return offsets


# =========================================================
# Helpers
# =========================================================

def basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]


def infer_frame_number(path):
    name = basename_noext(path).lower()
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def choose_nearest_file(target_file, candidate_files):
    target_num = infer_frame_number(target_file)
    if target_num is None or len(candidate_files) == 0:
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


def prompt_path(prompt_text, default=None):
    if default:
        value = input(f"{prompt_text} [{default}]: ").strip()
        return value if value else default
    return input(f"{prompt_text}: ").strip()


def read_fits_float(path):
    return fits.getdata(path).astype(float)


def combine_frames(file_list, method="median"):
    if len(file_list) == 0:
        return None

    stack = np.stack([read_fits_float(f) for f in file_list], axis=0)
    if method == "mean":
        return np.nanmean(stack, axis=0)
    return np.nanmedian(stack, axis=0)


def make_master_bias(bias_files):
    if len(bias_files) == 0:
        return None
    return combine_frames(bias_files, method="median")


def make_master_flat(flat_files, master_bias=None):
    if len(flat_files) == 0:
        return None, None

    flat_raw = combine_frames(flat_files, method="median")
    flat_corr = flat_raw.copy()

    if master_bias is not None:
        if flat_corr.shape == master_bias.shape:
            flat_corr = flat_corr - master_bias
        else:
            print("WARNING: master flat and master bias shapes differ; skipping bias subtraction for flat.")

    finite = np.isfinite(flat_corr)
    med = np.nanmedian(flat_corr[finite]) if np.any(finite) else np.nan

    if not np.isfinite(med) or med == 0:
        raise ValueError("Could not normalize master flat; median is invalid.")

    master_flat_norm = flat_corr / med

    bad = ~np.isfinite(master_flat_norm) | (master_flat_norm <= 0)
    master_flat_norm[bad] = 1.0

    return flat_corr, master_flat_norm


def make_master_arc(arc_files, master_bias=None, master_flat_norm=None):
    if len(arc_files) == 0:
        return None

    arc_raw = combine_frames(arc_files, method="median")
    arc_corr = arc_raw.copy()

    if master_bias is not None:
        if arc_corr.shape == master_bias.shape:
            arc_corr = arc_corr - master_bias
        else:
            print("WARNING: master arc and master bias shapes differ; skipping bias subtraction for arc.")

    if master_flat_norm is not None:
        if arc_corr.shape == master_flat_norm.shape:
            arc_corr = arc_corr / master_flat_norm
        else:
            print("WARNING: master arc and master flat shapes differ; skipping flat correction for arc.")

    return arc_corr


def calibrate_frame(frame, master_bias=None, master_flat_norm=None):
    out = frame.astype(float).copy()

    if master_bias is not None:
        if out.shape == master_bias.shape:
            out = out - master_bias
        else:
            print("WARNING: frame and master bias shapes differ; skipping bias subtraction.")

    if master_flat_norm is not None:
        if out.shape == master_flat_norm.shape:
            out = out / master_flat_norm
        else:
            print("WARNING: frame and master flat shapes differ; skipping flat correction.")

    return out


# =========================================================
# Spectral / cube functions
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


def build_cube(fiber_spectra, fiber_coords, cube_shape, dither_offset=(0, 0)):
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


def find_local_peak_near_line(wavelength, spectrum, expected_line, search_window=100.0):
    mask = np.isfinite(wavelength) & np.isfinite(spectrum)
    mask &= (wavelength >= expected_line - search_window) & (wavelength <= expected_line + search_window)

    wave_local = wavelength[mask]
    spec_local = spectrum[mask]

    if len(wave_local) < 5:
        raise ValueError(f"Not enough points near sky line {expected_line:.2f}")

    prom = max(3.0 * np.nanstd(spec_local), 5.0)
    peaks, props = find_peaks(spec_local, prominence=prom)

    if len(peaks) == 0:
        raise ValueError(f"No significant peaks found near sky line {expected_line:.2f}")

    best_peak = peaks[np.argmax(spec_local[peaks])]
    return wave_local[best_peak]

def fit_gaussian_emission(wavelength, spectrum, guess_center, guess_sigma=2.0, window=20,
                          center_tol=8.0, sigma_min=0.8, sigma_max=6.0):
    mask = np.isfinite(wavelength) & np.isfinite(spectrum)
    mask &= (wavelength >= guess_center - window) & (wavelength <= guess_center + window)

    wave_fit = wavelength[mask]
    spec_fit = spectrum[mask]

    if len(wave_fit) < 8:
        raise ValueError(f"Not enough valid points in fit window around {guess_center:.1f} A")

    # Continuum'u line dışındaki sideband'lerden tahmin et
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

#skyline

def build_master_sky_spectrum(sky_files, peaks, master_bias=None, master_flat_norm=None):
    """
    Build a 1D master sky spectrum from the selected sky frames,
    using calibrated (bias/flat corrected) sky exposures BEFORE sky subtraction.
    """
    if len(sky_files) == 0:
        raise ValueError("No sky files provided.")

    sky_specs = []
    for sf in sky_files:
        sky_frame = calibrate_frame(
            read_fits_float(sf),
            master_bias=master_bias,
            master_flat_norm=master_flat_norm
        )
        fiber_spec = extract_fiber_spectra(sky_frame, peaks, width=5)
        sky_1d = np.nanmedian(fiber_spec, axis=0)
        sky_specs.append(sky_1d)

    return np.nanmedian(np.stack(sky_specs, axis=0), axis=0)


def measure_sky_zero_point(wavelength, sky_spectrum,
                           sky_lines=(5577.34, 6300.30, 6363.78),
                           search_window=60.0,
                           fit_window=8.0):
    """
    Measure wavelength zero-point offset from strong sky lines.
    First find a nearby peak in a wide window, then refine with a narrow Gaussian fit.
    Returns:
        delta_lambda : median(true - measured)
        measurements : list of (true_line, measured_center, delta)
    """
    measurements = []

    for line in sky_lines:
        try:
            peak_guess = find_local_peak_near_line(
                wavelength, sky_spectrum, line, search_window=search_window
            )

            result, wave_local, spec_local = fit_gaussian_emission(
                wavelength,
                sky_spectrum,
                guess_center=peak_guess,
                guess_sigma=2.0,
                window=fit_window,
                center_tol=5.0,
                sigma_min=0.5,
                sigma_max=5.0
            )

            measured = result.params["g_center"].value
            delta = line - measured
            measurements.append((line, measured, delta))

        except Exception as e:
            print(f"Sky line {line:.2f} failed: {e}")
            continue
    if len(measurements) == 0:
        raise ValueError("Could not measure any sky lines for zero-point correction.")

    delta_lambda = np.median([m[2] for m in measurements])
    return delta_lambda, measurements

# =========================================================
# Region selector
# =========================================================

class RegionSelector:
    def __init__(self, image):
        self.image = np.array(image, dtype=float)
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
        self.ax.imshow(self.image, origin="lower", cmap="inferno")

        if np.any(self.include_mask):
            yy, xx = np.where(self.include_mask)
            self.ax.scatter(xx, yy, s=8, c="lime", alpha=0.35)

        if np.any(self.exclude_mask):
            yy, xx = np.where(self.exclude_mask)
            self.ax.scatter(xx, yy, s=8, c="red", alpha=0.35)

        mode = "EXCLUDE" if self.exclude_mode else "INCLUDE"
        tool = "polygon" if self.polygon_mode else ("circle" if self.circle_mode else "none")
        self.ax.set_title(
            f"Mode: {mode} | Tool: {tool}  (c=circle, p=polygon, e=toggle, r=reset, Enter=finish)"
        )
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
            self.selector = RectangleSelector(
                self.ax, self.circle_select,
                useblit=True, button=[1],
                minspanx=2, minspany=2,
                spancoords="pixels", interactive=False
            )
            print("Circle mode active. Click and drag to draw a circle.")

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
                print("Polygon selection cleared.")

        elif event.key == "enter":
            if self.polygon_mode and len(self.polygon_verts) >= 3:
                self.finish_polygon()
            else:
                self.finish()

    def circle_select(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        if eclick.ydata is None or erelease.ydata is None:
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        radius = np.hypot(x2 - x1, y2 - y1) / 2.0
        self.add_circle(cx, cy, radius)

    def add_circle(self, cx, cy, radius):
        ny, nx = self.image.shape
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        if self.exclude_mode:
            self.exclude_mask |= circle
        else:
            self.include_mask |= circle

        self._draw_base()

    def polygon_click(self, event):
        if not self.polygon_mode or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.polygon_verts.append((event.xdata, event.ydata))
        marker, = self.ax.plot(
            event.xdata, event.ydata,
            "ro" if self.exclude_mode else "go",
            ms=5
        )
        self.polygon_markers.append(marker)

        if len(self.polygon_verts) > 1:
            x_prev, y_prev = self.polygon_verts[-2]
            line, = self.ax.plot(
                [x_prev, event.xdata],
                [y_prev, event.ydata],
                "r-" if self.exclude_mode else "g-",
                lw=1
            )
            self.polygon_markers.append(line)

        self.fig.canvas.draw_idle()

    def finish_polygon(self):
        if len(self.polygon_verts) < 3:
            print("Need at least 3 vertices for a polygon.")
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
# Main
# =========================================================

def main():
    print("\n=== GCMS IFU Reduction (Bias + Flat + Arc + Sky) ===\n")

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)
    if config_path:
        print(f"Loaded config from: {config_path}")

    default_science_dir = "/Users/astro/Downloads/gcms/alldata/NGC7212"
    default_arc_lines = os.path.join(default_science_dir, "arc_lines_auto.txt")

    while True:
        n_dither_str = cfg_get(cfg, "n_dither", "Number of dither positions (3 or 6)")
        if n_dither_str in {"3", "6"}:
            n_dither = int(n_dither_str)
            break
        print("Please enter 3 or 6.")
    print(f"Using {n_dither} dither positions.\n")

    sci_dir = cfg_get(cfg, "sci_dir", "Path to folder with science FITS files", default_science_dir)
    if not os.path.isdir(sci_dir):
        print(f"Science directory not found: {sci_dir}")
        sys.exit(1)

    all_fits = sorted(
        glob.glob(os.path.join(sci_dir, "*.fits")) +
        glob.glob(os.path.join(sci_dir, "*.fit"))
    )

    if len(all_fits) == 0:
        print("No FITS files found.")
        sys.exit(1)

    print("Found files:")
    for i, f in enumerate(all_fits):
        print(f"  {i}: {os.path.basename(f)}")

    arc_input = cfg_get(cfg, "arc_indices", "Arc frame indices for master arc (space-separated)")
    flat_input = cfg_get(cfg, "flat_indices", "Flat frame indices for master flat (space-separated)")
    bias_input = cfg_get(cfg, "bias_indices", "Bias frame indices for master bias (space-separated, Enter to skip)", "")

    try:
        arc_indices = parse_int_list(arc_input)
        flat_indices = parse_int_list(flat_input)
        bias_indices = parse_int_list(bias_input)
    except ValueError:
        print("Arc/flat/bias indices must be integers.")
        sys.exit(1)

    for group_name, idxs in [("arc", arc_indices), ("flat", flat_indices), ("bias", bias_indices)]:
        if any(i < 0 or i >= len(all_fits) for i in idxs):
            print(f"One or more {group_name} indices are out of range.")
            sys.exit(1)

    arc_files = [all_fits[i] for i in arc_indices]
    flat_files = [all_fits[i] for i in flat_indices]
    bias_files = [all_fits[i] for i in bias_indices]

    print("Selected arc files:")
    for f in arc_files:
        print(" ", os.path.basename(f))
    print("Selected flat files:")
    for f in flat_files:
        print(" ", os.path.basename(f))
    if bias_files:
        print("Selected bias files:")
        for f in bias_files:
            print(" ", os.path.basename(f))
    else:
        print("No bias subtraction will be applied.")

    try:
        sci_input = cfg_get(cfg, "science_indices", f"Select {n_dither} science file indices (space-separated)")
        sci_indices = parse_int_list(sci_input)
    except ValueError:
        print("Science indices must be integers.")
        sys.exit(1)

    if len(sci_indices) != n_dither:
        print(f"Need exactly {n_dither} science indices.")
        sys.exit(1)

    if any(i < 0 or i >= len(all_fits) for i in sci_indices):
        print("One or more science indices are out of range.")
        sys.exit(1)

    sci_files = [all_fits[i] for i in sci_indices]
    print("Selected science files:")
    for f in sci_files:
        print(" ", os.path.basename(f))

    sky_input = cfg_get(cfg, "sky_indices", "Sky frame indices (space-separated, Enter to skip sky subtraction)", "")
    sky_files = []
    if sky_input:
        try:
            sky_indices = parse_int_list(sky_input)
        except ValueError:
            print("Sky indices must be integers.")
            sys.exit(1)

        if any(i < 0 or i >= len(all_fits) for i in sky_indices):
            print("One or more sky indices are out of range.")
            sys.exit(1)

        sky_files = [all_fits[i] for i in sky_indices]
        print("Selected sky files:")
        for f in sky_files:
            print(" ", os.path.basename(f))
    else:
        print("No sky subtraction will be applied.")

    print("\nBuilding master bias/flat/arc...")
    master_bias = make_master_bias(bias_files) if bias_files else None
    if master_bias is not None:
        print("Master bias created.")

    master_flat_corr, master_flat_norm = make_master_flat(flat_files, master_bias=master_bias)
    print("Master flat created and normalized.")

    master_arc = make_master_arc(arc_files, master_bias=master_bias, master_flat_norm=master_flat_norm)
    print("Master arc created.")

    spatial_profile = np.nansum(master_flat_corr, axis=1)
    threshold = np.percentile(spatial_profile[np.isfinite(spatial_profile)], 80)
    peaks, _ = find_peaks(spatial_profile, height=threshold, distance=5)
    n_fibers = len(peaks)
    print(f"Found {n_fibers} fiber traces.")

    # wavelength calibration
    coeff_file = cfg_get(cfg, "coeff_file", "Path to wavelength coefficient file (Enter to use arc line file fit)", "")

    ref_spec = extract_fiber_spectra(master_arc, peaks, width=5)[len(peaks) // 2, :]

    if coeff_file:
        if not os.path.exists(coeff_file):
            print(f"Coefficient file not found: {coeff_file}")
            sys.exit(1)

        try:
            coeffs = np.loadtxt(coeff_file).astype(float).flatten()
        except Exception as e:
            print(f"Error reading coefficient file: {e}")
            sys.exit(1)

        if len(coeffs) < 3:
            print("Coefficient file must contain at least 3 coefficients.")
            sys.exit(1)

        wavelength = np.polyval(coeffs, np.arange(ref_spec.size))
        print(f"Loaded wavelength coefficients from: {coeff_file}")

    else:
        line_file = prompt_path("Path to arc line file (pixel wavelength)", default_arc_lines)
        if not os.path.exists(line_file):
            print(f"Arc line file not found: {line_file}")
            sys.exit(1)

        try:
            line_data = np.loadtxt(line_file)
            if line_data.ndim != 2 or line_data.shape[1] < 2:
                raise ValueError("Arc line file must have at least two columns.")
            pixel_pos = line_data[:, 0]
            known_wave = line_data[:, 1]
        except Exception as e:
            print(f"Error reading arc line file: {e}")
            sys.exit(1)

        if len(pixel_pos) < 4:
            print("Need at least 4 arc lines for a cubic wavelength solution.")
            sys.exit(1)

        coeffs = np.polyfit(pixel_pos, known_wave, 3)
        wavelength = np.polyval(coeffs, np.arange(ref_spec.size))
        print(f"Derived wavelength solution from arc line file: {line_file}")

    delta_input = cfg_get(cfg, "delta_lambda", "Optional wavelength zero-point correction in Angstrom (Enter for 0)", "0")
    try:
        delta_lambda = float(delta_input) if delta_input else 0.0
    except ValueError:
        print("Invalid wavelength correction. Using 0.0 Angstrom.")
        delta_lambda = 0.0

    wavelength = wavelength + delta_lambda
    print(f"Applied wavelength correction: {delta_lambda:+.2f} Angstrom")
    print(f"Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Angstrom")
    
    use_sky_zero = parse_bool(cfg_get(
        cfg,
        "use_sky_zero_point",
        "Use sky lines BEFORE sky subtraction for zero-point correction? (y/n)",
        "y"
    ))

    if use_sky_zero:
        if len(sky_files) == 0:
            print("WARNING: use_sky_zero_point=y but no sky files were provided. Skipping sky zero-point correction.")
        else:
            sky_zero_lines_text = cfg_get(
                cfg,
                "sky_zero_lines",
                "Sky lines for zero-point correction (space-separated Angstrom)",
                "5577.34 6300.30 6363.78"
            )

            try:
                sky_zero_lines = [float(x) for x in sky_zero_lines_text.split()]
            except Exception:
                print("Invalid sky_zero_lines. Using default [5577.34, 6300.30, 6363.78].")
                sky_zero_lines = [5577.34, 6300.30, 6363.78]

            sky_window_text = cfg_get(
                cfg,
                "sky_zero_window",
                "Sky-line fit window in Angstrom",
                "8"
            )

            try:
                sky_window = float(sky_window_text)
            except Exception:
                sky_window = 8.0

            try:
                master_sky_spec = build_master_sky_spectrum(
                    sky_files,
                    peaks,
                    master_bias=master_bias,
                    master_flat_norm=master_flat_norm
                )

                sky_delta, sky_measurements = measure_sky_zero_point(
                    wavelength,
                    master_sky_spec,
                    sky_lines=sky_zero_lines,
                    search_window=100,
                    fit_window=sky_window
                )

                print("\nSky-line zero-point measurements:")
                for true_line, measured, delta in sky_measurements:
                    print(f"  true={true_line:.2f} A   measured={measured:.2f} A   delta={delta:+.2f} A")

                wavelength = wavelength + sky_delta
                print(f"Applied sky-based zero-point correction: {sky_delta:+.2f} Angstrom")
                print(f"New wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Angstrom")

            except Exception as e:
                print(f"Sky zero-point correction failed: {e}")


                # Diagnostic plot for master sky spectrum
                plt.figure(figsize=(12, 5))
                plt.plot(wavelength, master_sky_spec, "k-", lw=1)
                for wl in sky_zero_lines:
                    plt.axvline(wl, color="r", ls="--", alpha=0.6)
                plt.xlim(5500, 5650)
                plt.xlabel("Wavelength (Angstrom)")
                plt.ylabel("Sky Flux")
                plt.title("Master sky spectrum around 5577 A")
                plt.show()

                mask_dbg = (wavelength >= 5500) & (wavelength <= 5650) & np.isfinite(master_sky_spec)
                wave_dbg = wavelength[mask_dbg]
                spec_dbg = master_sky_spec[mask_dbg]

                peaks_dbg, props_dbg = find_peaks(spec_dbg, prominence=max(3*np.nanstd(spec_dbg), 5.0))
                print("\nStrong peaks near 5577 A:")
                for p in peaks_dbg:
                    print(f"  peak at {wave_dbg[p]:.2f} A, flux={spec_dbg[p]:.2f}")


    use_offsets = parse_bool(cfg_get(cfg, "use_offsets", "Do you have dither offsets? (y/n)", "n"))
    offsets = [(0.0, 0.0)] * n_dither

    if use_offsets:
        offsets_text = cfg_get(cfg, "dither_offsets", None, "")
        if offsets_text:
            try:
                offsets = parse_offsets(offsets_text, n_dither)
            except Exception as e:
                print(f"Invalid dither_offsets in config: {e}")
                sys.exit(1)
        else:
            print(f"Enter {n_dither} offsets (dx dy) in cube pixels, one per line:")
            new_offsets = []
            for i in range(n_dither):
                vals = input(f"Offset {i+1}: ").strip().split()
                if len(vals) != 2:
                    print("Please enter exactly two numbers: dx dy")
                    sys.exit(1)
                new_offsets.append((float(vals[0]), float(vals[1])))
            offsets = new_offsets

    calib_file = cfg_get(cfg, "fiber_coords", "Path to fiber coordinates CSV (fiber_id,x,y) or Enter to simulate", "")
    if calib_file and os.path.exists(calib_file):
        fiber_coords = np.loadtxt(calib_file, delimiter=",", skiprows=1, usecols=(1, 2))
    else:
        print("Simulating hexagonal grid...")
        spacing = 4.0
        coords = []
        count = 0
        row_index = 0
        for y in np.arange(20, 80, spacing * 0.866):
            row_offset = (row_index % 2) * (spacing / 2.0)
            for x in np.arange(20 + row_offset, 80, spacing):
                if count < n_fibers:
                    coords.append((x, y))
                    count += 1
            row_index += 1
            if count >= n_fibers:
                break
        fiber_coords = np.array(coords, dtype=float)

    if len(fiber_coords) == 0:
        print("No fiber coordinates available.")
        sys.exit(1)

    nx = int(np.nanmax(fiber_coords[:, 0]) + 20)
    ny = int(np.nanmax(fiber_coords[:, 1]) + 20)
    cube_shape = (nx, ny, len(wavelength))

    all_cubes = []
    for fname, offset in zip(sci_files, offsets):
        print(f"Processing {os.path.basename(fname)}...")
        sci_frame = calibrate_frame(read_fits_float(fname), master_bias=master_bias, master_flat_norm=master_flat_norm)

        matched_sky = choose_nearest_file(fname, sky_files) if sky_files else None
        if matched_sky is not None:
            print(f"  Using sky frame: {os.path.basename(matched_sky)}")
            sky_frame = calibrate_frame(read_fits_float(matched_sky), master_bias=master_bias, master_flat_norm=master_flat_norm)
            if sci_frame.shape == sky_frame.shape:
                proc_frame = sci_frame - sky_frame
            else:
                print("  WARNING: science and sky frame shapes differ; skipping sky subtraction for this file.")
                proc_frame = sci_frame
        else:
            proc_frame = sci_frame

        fiber_spec = extract_fiber_spectra(proc_frame, peaks, width=5)
        cube = build_cube(fiber_spec, fiber_coords, cube_shape, offset)
        all_cubes.append(cube)

    stack = np.stack(all_cubes, axis=0)
    clipped = sigma_clip(stack, axis=0, sigma=3, masked=True)
    with np.errstate(invalid="ignore"):
        combined_cube = np.nanmean(clipped.filled(np.nan), axis=0)

    out_cube = cfg_get(cfg, "out_cube", "Save combined cube as (e.g., cube.fits)", "")
    if out_cube:
        save_cube_for_qfitsview(
            out_cube,
            combined_cube,
            wavelength,
            ra_deg=331.7537231,
            dec_deg=10.2308607,
            pixscale_arcsec=1.0
        )
        print(f"Saved QFitsView-compatible cube: {out_cube}")

    collapsed = np.nansum(combined_cube, axis=2)
    print("Opening interactive selector. Use:")
    print("  c - circle mode, p - polygon mode, e - toggle include/exclude, r - reset, enter - finish")

    selector = RegionSelector(collapsed)
    plt.show()

    final_mask = selector.get_mask()
    if not np.any(final_mask):
        print("No region selected. Exiting.")
        sys.exit(0)

    selected_spaxels = combined_cube[final_mask]
    if selected_spaxels.size == 0:
        print("Selected region contains no valid spaxels.")
        sys.exit(1)

    with np.errstate(invalid="ignore"):
        region_spectrum = np.nanmean(selected_spaxels, axis=0)

    if not np.any(np.isfinite(region_spectrum)):
        print("Region spectrum is empty/non-finite.")
        sys.exit(1)

    z_input = cfg_get(cfg, "z", "Approximate redshift z (press Enter for 0)", "0")
    try:
        z = float(z_input) if z_input else 0.0
    except ValueError:
        print("Invalid redshift value. Using z = 0.0")
        z = 0.0

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

    print(f"\nUsing redshift z = {z:.8f}")

    while True:
        print("Select emission line:")
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
            result, wave_local, spec_local = fit_gaussian_emission(
                wavelength, region_spectrum, guess_center, guess_sigma=2, window=10
            )
        except Exception as e:
            print(f"Fit failed: {e}")
            continue

        print(result.fit_report())

        plt.figure(figsize=(10, 5))
        plt.plot(wavelength, region_spectrum, color="0.75", lw=1, label="Full spectrum")
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
