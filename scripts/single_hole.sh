#!/bin/bash

python single_hole.py \
    --output="../data/test.h5" \
    --ncpus=8 \
    --width=500 \
    --height=500 \
    --radius=100 \
    --lambda_=200 \
    --xi=10 \
    --currents 350 450 11 \
    --fields 0 20 3 \
    --solve-time=500 \
    --eval-time=200
