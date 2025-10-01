<p align="center" width="100%"><img width="256px" src="logo.png"></p>

# Overview

## What is MOMA?

MOMA is a mobility mapping toolbox. When provided with GIS and simulation data
from multiphysics simulation, it allows for the generation of mobility maps and
operational path planning to optimise traversal speed under the specified
conditions.

MOMA allows users to:

* 🌍 Visualize and process geospatial data relevant for vehicle mobility
  assessment for agricultural, defence, robotics and recreational applications

* 📈 Provide a quantitative numerical and visual analysis of vehicle mobility
  across a given geofenced area

* 🛣️ Generate maps and paths for optimal on- and off-road navigation

![moma-map](image.png)

## Features

At this time, MOMA can be used to:

* 🛰️ Load, process and visualize aerial and elevation maps in GeoTIFF format.

* 🚗 Load and process multibody dynamics simulation results to characterize
  vehicle mobility (directly if using
  [DYNO](https://github.com/aarhus-robotics/dyno))

* 🗺️ Generate directional and omni-directional mobility map for a given site

* 📍 Generate optimal traversal paths using the A* algorithm

# Getting started

## Installing MOMA

To get started using MOMA, you can:

* 🐋 Build and run the bundled Docker image by following the [Docker installation guide](https://docs.aarhusrobotics.com/moma/getting-started/docker)

* 🖥️ Build from source by following the [build guide](https://docs.aarhusrobotics.com/moma/getting-started/source)

## Accessing the documentation

You can learn more about how to deploy and use MOMA at
[docs.aarhusrobotics.com/moma](https://docs.aarhusrobotics.com/moma).

# Contributing

MOMA is in active development following the ever-changing needs of the small
research group behind it. If you would like to add a new feature, we invite you
to follow the [contribution guidelines](CONTRIBUTING.md) before opening a new
pull request. We are also open to hearing your feedback, bug reports and feature
requests through [GitHub issues](issues).

# Repository structure

The repository contains the following:

* 🐍 The Python package for the application under [moma/][moma/]

* 🐋 The source files for the Docker image under [docker/](docker/)

* 🗺️ A collection of examples on map and path generation under [examples/](examples/).

* 📜 A collection of convenience scripts for running the command line interface
  under [scripts/](scripts/)

* 📓 A paper for the Journal of Open Source Software (JOSS) under [paper/](paper).

# License

DYNO is licensed using the permissive [MIT software license](LICENSE). You can
do whatever you want as long as you include the original copyright and license
notice in any copy of the software/source. Please refer to the license full text
for more information the license terms.