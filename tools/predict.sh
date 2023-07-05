#!/usr/bin/env bash

# This script is used to launch the application.
CUDA_VISIBLE_DEVICES=$1 python run.py test-large-tiles \
    --output-folder $2 \
    --data-folder $3 \
    --large-images-file rasters.txt \