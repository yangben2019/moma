#!/bin/bash

set -e

VERSION=$(git rev-parse --short HEAD)

docker push ghcr.io/au-agv/moma:devel
docker push ghcr.io/au-agv/moma:devel-${VERSION}
docker push ghcr.io/au-agv/moma:devel-stable

docker push ghcr.io/au-agv/moma:runtime
docker push ghcr.io/au-agv/moma:runtime-${VERSION}
docker push ghcr.io/au-agv/moma:runtime-stable