#!/usr/bin/env bash

# This script is used to launch the application.
CUDA_VISIBLE_DEVICES=$1 python run.py test-segmenter --output-folder $2
