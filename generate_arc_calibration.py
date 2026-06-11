#!/usr/bin/env python3
"""
generate_arc_calibration.py – Generate arc line list and wavelength coefficients
from raw GCMS arc, bias, and flat frames.
Now includes Sodium (Na) arc lines.
"""

import os
import re
import sys
import argparse
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------
# Known arc lines for common GCMS lamps (wavelength in Angstroms)
# ----------------------------------------------------------------------
KNOWN_LINES = {
    "He": [
        3888.65, 4471.48, 4713.15, 4921.93, 5015.68,
        5875.62, 6678.15, 7065.19, 7281.35
    ],
    "Ne": [
        5852.49, 5881.90, 5944.83, 5975.53, 6074.34,
        6096.16, 6143.06, 6163.59, 6217.28, 6266.49,
        6304.79, 6334.43, 6382.99, 6402.25, 6506.53,
        6532.88, 6598.95, 6678.28, 6717.04, 6929.47,
        7032.41, 7245.17
    ],
    "Ar": [
        3948.98, 4044.42, 4158.59, 4181.88, 4198.32,
        4200.67, 4259.36, 4266.28, 4335.36, 4345.17,
        4510.73, 4545.05, 4596.10, 4657.90, 4702.32,
        4726.89, 4764.87, 4806.02, 4879.86, 4965.08,
        5017.16, 5062.04, 5145.32, 5162.28, 5187.75,
        5221.27, 5285.37, 5292.21, 5355.35, 5379.01,
        5451.66, 5495.87, 5558.70, 5606.73, 5650.70,
        5739.52, 5764.42, 5802.12, 5912.08, 5942.68,
        6043.23, 6059.43, 6114.92, 6145.43, 6172.28,
        6212.50, 6243.29, 6291.21, 6369.58, 6416.31,
        6430.94, 6538.11, 6752.83, 6871.29, 6937.66,
        7030.25, 7067.22, 7147.04, 7272.94, 7311.71,
        7354.43, 7383.98, 7503.87, 7514.65, 7635.11,
        7723.76, 7726.76, 7948.18, 8006.16, 8103.69,
        8115.31, 8264.52, 8408.21, 8424.65, 8521.44,
        8667.94, 9122.97
    ],
    "Hg": [
        3650.15, 4046.56, 4358.33, 5460.74, 5769.60,
        5790.66
    ],
    "Na": [
        5889.95,   # Na I D2
        5895.92    # Na I D1
        # Additional weaker Na lines (rarely needed):
        # 3302.37, 3302.98,   # Na I UV doublet
        # 5682.65, 5688.22,   # Na I
        # 6154.23, 6160.75    # Na I
    ]
}

# ----------------------------------------------------------------------
# Helper functions (same as before)
# ----------------------------------------------------------------------

def expand_file_sequence(start_file, n, base_path=None):
    if base_path:
        start_file = os.path.join(base_path, start_file)
    directory = os.path.dirname(start_file)
    basename = os.path.basename(start_file)
    m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", basename)
    if not m:
        raise ValueError(f"Could not parse numbered filename: {start_file}")
    prefix, number_str, ext = m.groups()
    start_num = int(number_str)
    width = len(number_str)
    files = []
    for i in range(n):
        fname = f"{prefix}{start_num + i:0{width}d}{ext}"
        full = os.path.join(directory, fname) if directory else fname
        files.append(full)
    return files

def read_stack(file_list):
    data = []
    for f in file_list:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")
        data.append(fits.getdata(f).astype(float))
    return np.stack(data, axis=0)

def extract_fiber_spectra(frame, peak_positions, width=5):
    n_fibers = len(peak_positions)
    n_spec = frame.shape[1]
    spectra = np.zeros((n_fibers, n_spec))
    for i, p in enumerate(peak_positions):
        x0 = max(0, int(round(p - width // 2)))
        x1 = min(frame.shape[0], int(round(p + width // 2 + 1)))
        spectra[i, :] = np.nansum(frame[x0:x1, :], axis=0)
    return spectra

def fit_wavelength_solution(pixel_positions, wavelengths, deg=3):
    coeffs = np.polyfit(pixel_positions, wavelengths, deg)
    return coeffs

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate arc line list and wavelength coefficients from GCMS arc exposures."
    )
    parser.add_argument("--path", default=".", help="Base directory containing FITS files")
    parser.add_argument("--arcs", required=True, help="First arc file (e.g., gcms0536.fits)")
    parser.add_argument("--an", required=True, type=int, help="Number of arc files")
    parser.add_argument("--bias", help="First bias file (optional)")
    parser.add_argument("--bn", type=int, default=0, help="Number of bias files")
    parser.add_argument("--flat", required=True, help="First flat file")
    parser.add_argument("--fn", required=True, type=int, help="Number of flat files")
    parser.add_argument("--trace_percentile", type=float, default=80.0,
                        help="Percentile for fiber trace detection")
    parser.add_argument("--arc_percentile", type=float, default=95.0,
                        help="Percentile for arc peak detection")
    parser.add_argument("--distance", type=int, default=5,
                        help="Minimum peak separation (pixels)")
    parser.add_argument("--elements", nargs="+", required=True,
                        help="Lamp elements (e.g., He Ne Ar Na). Use known elements: He, Ne, Ar, Hg, Na")
    parser.add_argument("--out_arc_lines", default="arc_lines_auto.txt",
                        help="Output file for pixel-wavelength pairs")
    parser.add_argument("--out_coeff", default="",
                        help="Output file for wavelength coefficients (optional)")
    parser.add_argument("--match_tolerance", type=float, default=5.0,
                        help="Pixel tolerance for matching detected peaks to known lines")
    parser.add_argument("--poly_degree", type=int, default=3,
                        help="Polynomial degree for wavelength fit")
    args = parser.parse_args()

    outpath = os.path.abspath(args.path)

    # Expand file sequences
    try:
        arc_files = expand_file_sequence(args.arcs, args.an, base_path=args.path)
        flat_files = expand_file_sequence(args.flat, args.fn, base_path=args.path)
        bias_files = expand_file_sequence(args.bias, args.bn, base_path=args.path) if args.bias and args.bn > 0 else []
    except Exception as e:
        print(f"Error expanding file sequences: {e}")
        sys.exit(1)

    print(f"Base path: {outpath}")
    print(f"Bias files: {len(bias_files)}")
    print(f"Flat files: {len(flat_files)}")
    print(f"Arc files:  {len(arc_files)}")

    # Build masters
    master_bias = None
    if bias_files:
        bias_stack = read_stack(bias_files)
        master_bias = np.nanmedian(bias_stack, axis=0)
        print("Master bias created.")

    flat_stack = read_stack(flat_files)
    master_flat_raw = np.nanmedian(flat_stack, axis=0)

    arc_stack = read_stack(arc_files)
    master_arc_raw = np.nanmedian(arc_stack, axis=0)

    # Apply bias subtraction
    if master_bias is not None:
        if master_flat_raw.shape == master_bias.shape:
            master_flat = master_flat_raw - master_bias
        else:
            print("WARNING: flat/bias shape mismatch; skipping bias for flat.")
            master_flat = master_flat_raw
        if master_arc_raw.shape == master_bias.shape:
            master_arc = master_arc_raw - master_bias
        else:
            print("WARNING: arc/bias shape mismatch; skipping bias for arc.")
            master_arc = master_arc_raw
    else:
        master_flat = master_flat_raw
        master_arc = master_arc_raw

    # Normalize flat
    flat_med = np.nanmedian(master_flat[np.isfinite(master_flat)])
    if not np.isfinite(flat_med) or flat_med == 0:
        print("Flat normalization failed.")
        sys.exit(1)
    master_flat_norm = master_flat / flat_med
    bad = ~np.isfinite(master_flat_norm) | (master_flat_norm <= 0)
    master_flat_norm[bad] = 1.0

    # Apply flat correction to arc
    master_arc_corr = master_arc / master_flat_norm

    # Find fiber traces from flat
    spatial_profile = np.nansum(master_flat, axis=1)
    finite_spatial = spatial_profile[np.isfinite(spatial_profile)]
    threshold = np.percentile(finite_spatial, args.trace_percentile)
    peaks, _ = find_peaks(spatial_profile, height=threshold, distance=args.distance)
    print(f"Found {len(peaks)} fiber traces.")

    # Extract central fiber spectrum
    spectra = extract_fiber_spectra(master_arc_corr, peaks, width=5)
    ref_idx = len(peaks) // 2
    ref_spec = spectra[ref_idx]
    print(f"Using reference fiber {ref_idx}")

    # Detect arc peaks
    finite_ref = ref_spec[np.isfinite(ref_spec)]
    peak_thresh = np.percentile(finite_ref, args.arc_percentile)
    arc_peaks, _ = find_peaks(ref_spec, height=peak_thresh, distance=args.distance)
    print(f"Detected {len(arc_peaks)} candidate peaks.")

    # Collect known wavelengths from selected elements
    known_wavelengths = []
    for elem in args.elements:
        if elem not in KNOWN_LINES:
            print(f"Warning: element '{elem}' not in internal database. Skipping.")
            continue
        known_wavelengths.extend(KNOWN_LINES[elem])
    known_wavelengths = np.array(sorted(known_wavelengths))
    print(f"Total known lines from {args.elements}: {len(known_wavelengths)}")

    if len(known_wavelengths) == 0:
        print("No known wavelengths available. Exiting.")
        sys.exit(1)

    # Match peaks to known lines using linear scaling
    arc_peaks_sorted = np.sort(arc_peaks)
    known_waves_sorted = np.sort(known_wavelengths)

    p0 = arc_peaks_sorted[0]
    p1 = arc_peaks_sorted[-1]
    w0 = known_waves_sorted[0]
    w1 = known_waves_sorted[-1]
    if p1 == p0:
        print("Peak range is zero; cannot scale.")
        sys.exit(1)
    a = (w1 - w0) / (p1 - p0)
    b = w0 - a * p0

    matched_pixels = []
    matched_waves = []

    for p in arc_peaks_sorted:
        approx_wave = a * p + b
        idx = np.argmin(np.abs(known_waves_sorted - approx_wave))
        closest = known_waves_sorted[idx]
        if abs(closest - approx_wave) < args.match_tolerance * a:
            if closest not in matched_waves:  # avoid duplicate wavelength matches
                matched_pixels.append(p)
                matched_waves.append(closest)

    matched_pixels = np.array(matched_pixels)
    matched_waves = np.array(matched_waves)

    print(f"Matched {len(matched_pixels)} arc lines.")
    if len(matched_pixels) < args.poly_degree + 1:
        print(f"Need at least {args.poly_degree+1} matched lines for polynomial fit. Exiting.")
        sys.exit(1)

    # Save arc_lines_auto.txt
    arc_lines_data = np.column_stack((matched_pixels, matched_waves))
    np.savetxt(args.out_arc_lines, arc_lines_data, fmt="%d %.2f")
    print(f"Saved arc line list to {args.out_arc_lines}")

    # Fit polynomial wavelength solution
    coeffs = fit_wavelength_solution(matched_pixels, matched_waves, deg=args.poly_degree)
    print(f"Polynomial coefficients (degree {args.poly_degree}): {coeffs}")

    if args.out_coeff:
        np.savetxt(args.out_coeff, [coeffs], fmt="%.8e")
        print(f"Saved wavelength coefficients to {args.out_coeff}")

    # Optional diagnostic plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(ref_spec, lw=1, alpha=0.7)
        plt.plot(matched_pixels, ref_spec[matched_pixels.astype(int)], 'ro', label='Matched lines')
        for p, w in zip(matched_pixels, matched_waves):
            plt.text(p, ref_spec[int(p)], f"{w:.1f}", rotation=45, fontsize=8)
        plt.xlabel("Pixel")
        plt.ylabel("Flux")
        plt.title("Arc spectrum with matched lines")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

    print("Done.")

if __name__ == "__main__":
    main()
