# python inspect_arc_master.py --arcs gcms0536.fits --an 11 --bias gcms0547.fits --bn 11 --flat gcms0558.fits --fn 11 --path "/Users/astro/Downloads/gcms/alldata/NGC7212" --arc_percentile 95 --trace_percentile 68 --distance 8


import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks


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


def save_detected_peaks(outpath, arc_peaks):
    peaks_file = os.path.join(outpath, "candidate_arc_peaks.txt")
    np.savetxt(peaks_file, arc_peaks.reshape(-1, 1), fmt="%d")
    print(f"Saved detected peaks to: {peaks_file}")


def save_anchor_arc_lines(outpath, arc_peaks):
    """
    Very small, safe anchor list for coarse wavelength solution.
    """
    template = [
        (536, 5852.49),
        (681, 6143.06),
        (870, 6598.95),
    ]

    matched = []
    used = set()

    for expected_pix, wave in template:
        idx = np.argmin(np.abs(arc_peaks - expected_pix))
        pix = int(arc_peaks[idx])

        if abs(pix - expected_pix) <= 5 and pix not in used:
            matched.append((pix, wave))
            used.add(pix)

    matched = sorted(matched, key=lambda t: t[0])

    if len(matched) < 3:
        print("Could not build full anchor list automatically.")
        return

    outfile = os.path.join(outpath, "arc_lines_anchor.txt")
    arr = np.array(matched, dtype=float)
    np.savetxt(outfile, arr, fmt=["%d", "%.2f"])
    print(f"Saved anchor arc-line guess to: {outfile}")
    print("Anchor lines:")
    for pix, wave in matched:
        print(f"  {pix:4d}  {wave:8.2f}")


def save_auto_arc_lines(outpath, arc_peaks):
    """
    Conservative automatic subset for this setup.
    """
    template = [
        (495, 5764.42),
        (536, 5852.49),
        (594, 5944.83),
        (604, 5975.53),
        (681, 6143.06),
        (712, 6217.28),
        (734, 6266.49),
        (743, 6304.79),
        (760, 6334.43),
        (791, 6382.99),
        (803, 6402.25),
        (870, 6598.95),
        (888, 6678.28),
        (988, 6929.47),
    ]

    matched = []
    used = set()

    for expected_pix, wave in template:
        idx = np.argmin(np.abs(arc_peaks - expected_pix))
        pix = int(arc_peaks[idx])

        if abs(pix - expected_pix) <= 3 and pix not in used:
            matched.append((pix, wave))
            used.add(pix)

    matched = sorted(matched, key=lambda t: t[0])

    if len(matched) == 0:
        print("No auto arc-line matches found; not writing arc_lines_auto.txt")
        return

    outfile = os.path.join(outpath, "arc_lines_auto.txt")
    arr = np.array(matched, dtype=float)
    np.savetxt(outfile, arr, fmt=["%d", "%.2f"])
    print(f"Saved auto arc-line guess to: {outfile}")
    print("Auto-matched lines:")
    for pix, wave in matched:
        print(f"  {pix:4d}  {wave:8.2f}")


def save_full_guess_arc_lines(outpath, arc_peaks):
    """
    Broader guessed list than auto; looser matching tolerance.
    """
    known_lines = np.array([
        5764.42, 5852.49, 5944.83, 5975.53,
        6143.06, 6217.28, 6266.49, 6304.79,
        6334.43, 6382.99, 6402.25,
        6506.53, 6598.95, 6678.28, 6929.47
    ])

    template_pixels = np.array([
        495, 536, 594, 604,
        681, 712, 734, 743,
        760, 791, 803,
        833, 870, 888, 988
    ])

    matched = []
    used_pix = set()
    used_lines = set()

    for expected_pix, wave in zip(template_pixels, known_lines):
        idx = np.argmin(np.abs(arc_peaks - expected_pix))
        pix = int(arc_peaks[idx])

        if abs(pix - expected_pix) <= 8 and pix not in used_pix and wave not in used_lines:
            matched.append((pix, wave))
            used_pix.add(pix)
            used_lines.add(wave)

    matched = sorted(matched, key=lambda t: t[0])

    if len(matched) == 0:
        print("No full-guess arc-line matches found; not writing arc_lines_full_guess.txt")
        return

    outfile = os.path.join(outpath, "arc_lines_full_guess.txt")
    arr = np.array(matched, dtype=float)
    np.savetxt(outfile, arr, fmt=["%d", "%.2f"])
    print(f"Saved full guessed arc-line list to: {outfile}")
    print("Full guessed lines:")
    for pix, wave in matched:
        print(f"  {pix:4d}  {wave:8.2f}")


def save_all_candidates_predicted(outpath, arc_peaks):
    """
    Use 3 anchor lines to build a coarse quadratic solution, then predict
    wavelengths for all detected peaks.
    """
    # Need these three peaks to exist
    anchors = [
        (536, 5852.49),
        (681, 6143.06),
        (870, 6598.95),
    ]

    pix = []
    wav = []

    for expected_pix, wave in anchors:
        idx = np.argmin(np.abs(arc_peaks - expected_pix))
        found = int(arc_peaks[idx])
        if abs(found - expected_pix) <= 5:
            pix.append(found)
            wav.append(wave)

    if len(pix) < 3:
        print("Could not build coarse solution for all-candidate prediction.")
        return

    pix = np.array(pix, dtype=float)
    wav = np.array(wav, dtype=float)

    coeffs = np.polyfit(pix, wav, 2)
    predicted = np.polyval(coeffs, arc_peaks.astype(float))

    outfile = os.path.join(outpath, "arc_lines_all_candidates_predicted.txt")
    np.savetxt(
        outfile,
        np.column_stack([arc_peaks.astype(int), predicted]),
        fmt=["%d", "%.2f"]
    )
    print(f"Saved predicted wavelengths for all candidate peaks to: {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect master bias/flat/arc and extract reference arc spectrum."
    )
    parser.add_argument("--path", default=".", help="Base directory containing the FITS files")
    parser.add_argument("--arcs", required=True, help="First arc file, e.g. gcms0536.fits")
    parser.add_argument("--an", required=True, type=int, help="Number of arc files")
    parser.add_argument("--bias", required=False, help="First bias file, e.g. gcms0547.fits")
    parser.add_argument("--bn", required=False, type=int, default=0, help="Number of bias files")
    parser.add_argument("--flat", required=True, help="First flat file, e.g. gcms0558.fits")
    parser.add_argument("--fn", required=True, type=int, help="Number of flat files")
    parser.add_argument("--trace_percentile", type=float, default=80.0, help="Percentile for fiber trace threshold")
    parser.add_argument("--arc_percentile", type=float, default=95.0, help="Percentile for arc peak threshold")
    parser.add_argument("--distance", type=int, default=5, help="Minimum peak separation")
    args = parser.parse_args()

    outpath = os.path.abspath(args.path)

    try:
        arc_files = expand_file_sequence(args.arcs, args.an, base_path=args.path)
        flat_files = expand_file_sequence(args.flat, args.fn, base_path=args.path)
        bias_files = expand_file_sequence(args.bias, args.bn, base_path=args.path) if args.bias and args.bn > 0 else []
    except Exception as e:
        print(f"Error expanding file sequences: {e}")
        sys.exit(1)

    print("Base path :", outpath)
    print("Bias files:", len(bias_files))
    print("Flat files:", len(flat_files))
    print("Arc files :", len(arc_files))

    master_bias = None
    if len(bias_files) > 0:
        bias_stack = read_stack(bias_files)
        master_bias = np.nanmedian(bias_stack, axis=0)
        print("Master bias shape:", master_bias.shape)

    flat_stack = read_stack(flat_files)
    arc_stack = read_stack(arc_files)

    master_flat_raw = np.nanmedian(flat_stack, axis=0)
    master_arc_raw = np.nanmedian(arc_stack, axis=0)

    if master_bias is not None:
        if master_flat_raw.shape == master_bias.shape:
            master_flat = master_flat_raw - master_bias
        else:
            print("WARNING: flat and bias shapes differ, skipping bias subtraction for flat.")
            master_flat = master_flat_raw

        if master_arc_raw.shape == master_bias.shape:
            master_arc = master_arc_raw - master_bias
        else:
            print("WARNING: arc and bias shapes differ, skipping bias subtraction for arc.")
            master_arc = master_arc_raw
    else:
        master_flat = master_flat_raw
        master_arc = master_arc_raw

    flat_med = np.nanmedian(master_flat[np.isfinite(master_flat)])
    if not np.isfinite(flat_med) or flat_med == 0:
        print("Flat normalization failed.")
        sys.exit(1)

    master_flat_norm = master_flat / flat_med
    bad = ~np.isfinite(master_flat_norm) | (master_flat_norm <= 0)
    master_flat_norm[bad] = 1.0

    master_arc_corr = master_arc / master_flat_norm

    print("Flat shape:", master_flat.shape)
    print("Arc shape :", master_arc_corr.shape)

    spatial_profile = np.nansum(master_flat, axis=1)
    finite_spatial = spatial_profile[np.isfinite(spatial_profile)]
    threshold = np.percentile(finite_spatial, args.trace_percentile)
    peaks, props = find_peaks(spatial_profile, height=threshold, distance=args.distance)

    print(f"Found {len(peaks)} fiber traces")

    plt.figure(figsize=(12, 5))
    plt.plot(spatial_profile, lw=1)
    plt.plot(peaks, spatial_profile[peaks], "r.")
    plt.xlabel("Spatial pixel")
    plt.ylabel("Summed counts")
    plt.title("Fiber trace detection from master flat")
    plt.tight_layout()
    plt.show()

    spectra = extract_fiber_spectra(master_arc_corr, peaks, width=5)

    ref_idx = len(peaks) // 2
    ref_spec = spectra[ref_idx]

    print("Reference fiber index:", ref_idx)

    finite_ref = ref_spec[np.isfinite(ref_spec)]
    peak_thresh = np.percentile(finite_ref, args.arc_percentile)
    arc_peaks, arc_props = find_peaks(ref_spec, height=peak_thresh, distance=args.distance)

    print("Candidate arc peaks (pixel positions):")
    print(arc_peaks)

    save_detected_peaks(outpath, arc_peaks)
    save_anchor_arc_lines(outpath, arc_peaks)
    save_auto_arc_lines(outpath, arc_peaks)
    save_full_guess_arc_lines(outpath, arc_peaks)
    save_all_candidates_predicted(outpath, arc_peaks)

    plt.figure(figsize=(14, 5))
    plt.plot(ref_spec, lw=1)
    plt.plot(arc_peaks, ref_spec[arc_peaks], "r.")
    for p in arc_peaks:
        plt.text(p, ref_spec[p], str(p), fontsize=8, rotation=90, va="bottom")
    plt.xlabel("Spectral pixel")
    plt.ylabel("Flux")
    plt.title(f"Reference fiber arc spectrum (fiber {ref_idx})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
