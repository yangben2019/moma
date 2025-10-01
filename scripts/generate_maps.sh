#!/bin/bash

set -e

TIMESTAMP=$(date +%s)

printf "Generating test maps ...\n"
moma generate_test_map \
    --N_maps 32 \
    --output run-${TIMESTAMP}/map

printf "Generating test paths ...\n"
moma generate_test_paths \
    --output run-${TIMESTAMP}/paths \
    --fill_metrics \
    --minimum_speed 1.39 \
    --target_speed 9.73