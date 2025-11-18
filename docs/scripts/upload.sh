#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

rsync -av --stats --progress \
    ../build/html/ \
    robot@docs.aarhusrobotics.com:/home/robot/docs/moma/
