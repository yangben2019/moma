#!/bin/bash

set -e

poetry lock && \
poetry install --with documentation