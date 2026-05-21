#!/bin/bash

set -e

PROGRESS=plain
VERSION=$(git rev-parse --short HEAD)

cd "$(dirname "${BASH_SOURCE[0]}")"./../

docker build --progress ${PROGRESS} \
             --tag ghcr.io/au-agv/moma:devel \
             --tag ghcr.io/au-agv/moma:devel-${VERSION} \
             --tag ghcr.io/au-agv/moma:devel-stable \
             --file docker/build/devel/Dockerfile \
             .

docker build --progress ${PROGRESS} \
             --tag ghcr.io/au-agv/moma:runtime \
             --tag ghcr.io/au-agv/moma:runtime-${VERSION} \
             --tag ghcr.io/au-agv/moma:runtime-stable \
             --file docker/build/runtime/Dockerfile \
             .