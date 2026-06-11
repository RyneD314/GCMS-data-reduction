#!/bin/bash
# 6-dither reduction with the unified pipeline

python gcms_reduce_unified.py \
    --sci_dir raw_data/science_6dith \
    --bias_dir raw_data/biases \
    --flat_dir raw_data/flats \
    --arc_dir raw_data/arcs \
    --sky_dir raw_data/skies \
    --n_dither 6 \
    --offsets "0,0;2,0;1,2;1,1;3,1;2,3" \
    --fiber_coords fiber_positions.csv \
    --out_cube gcms_6dith_cube.fits \
    --object_name "NGC 1277" \
    --pixscale 1.0 \
    --coeff_file wavelength_coeffs.txt \
    --use_sky_zero \
    --z 0.0172
