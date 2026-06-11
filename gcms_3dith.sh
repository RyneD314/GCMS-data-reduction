#!/bin/bash
# 3-dither reduction with the unified pipeline

python gcms_reduce_unified.py \
    --sci_dir raw_data/science_3dith \
    --bias_dir raw_data/biases \
    --flat_dir raw_data/flats \
    --arc_dir raw_data/arcs \
    --sky_dir raw_data/skies \
    --n_dither 3 \
    --offsets "0,0;2,0;1,2" \
    --fiber_coords fiber_positions.csv \
    --out_cube gcms_3dith_cube.fits \
    --object_name "NGC 1277" \
    --pixscale 1.0 \
    --arc_lines arc_lines_auto.txt \
    --use_sky_zero \
    --z 0.0172
