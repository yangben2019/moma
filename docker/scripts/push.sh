#!/bin/bash

set -e

VERSION=$(git rev-parse --short HEAD)

docker push ghcr.io/aarhus-robotics/moma:devel
docker push ghcr.io/aarhus-robotics/moma:devel-${VERSION}
docker push ghcr.io/aarhus-robotics/moma:devel-stable