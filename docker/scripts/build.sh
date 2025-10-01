#!/bin/bash

set -e

PROGRESS=plain
VERSION=$(git rev-parse --short HEAD)

cd "$(dirname "${BASH_SOURCE[0]}")"

docker build --progress ${PROGRESS} \
             --tag ghcr.io/aarhus-robotics/moma:devel \
             --tag ghcr.io/aarhus-robotics/moma:devel-${VERSION} \
             --tag ghcr.io/aarhus-robotics/moma:devel-stable \
             ../build/devel

#docker build --progress ${PROGRESS} \
#             --tag ghcr.io/aarhus-robotics/moma:runtime \
#             --tag ghcr.io/aarhus-robotics/moma:runtime-${VERSION} \
#             --tag ghcr.io/aarhus-robotics/moma:runtime-stable \
#             ../build/runtime