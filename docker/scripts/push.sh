#!/bin/bash

set -e

VERSION=$(git rev-parse --short HEAD)

docker push ghcr.io/aarhus-robotics/moma:devel
docker push ghcr.io/aarhus-robotics/moma:devel-${VERSION}
docker push ghcr.io/aarhus-robotics/moma:devel-stable

docker push ghcr.io/aarhus-robotics/moma:runtime
docker push ghcr.io/aarhus-robotics/moma:runtime-${VERSION}
docker push ghcr.io/aarhus-robotics/moma:runtime-stable