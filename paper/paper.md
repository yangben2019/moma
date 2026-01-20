---
title: 'MOMA: A ground vehicles mobility mapping toolkit'
tags:
   - gis
   - mobility mapping
   - vehicle simulation
authors:
   - name: Dario Sirangelo
     orcid: 0009-0007-5194-6022
     corresponding: true
     affiliation: "1"
   - name: Nikolaj Jes Petersen
     orcid: 0009-0001-0379-8244
     affiliation: "1"
   - name: Ole Balling
     orcid: 0000-0002-8728-002X
     affiliation: "1"
affilations:
   - name: Aarhus University, Denmark
     ror: 01aj84f44
     index: 1
date: 14 January 2026
bibliography: paper.bib
---

## Summary

The NATO Next-Generation NATO Reference Mobility Model (Figure
\ref{fig:nato-ng-nrmm-framework}) is a shared computer modelling framework for
the prediction of the mobility of ground vehicles across different environments.
The primary goal of the framework is to provide quantitative estimates of where
vehicles can traverse, at what maximum speed, and how safely, using validated
[@gorsich2023cooperative] simulation tools above physical testing alone.

Contrary to older models, such as the previous NATO Reference Mobility Model
(NRMM), NG-NRMM incorporates the combined used of high-fidelity,
three-dimensional vehicle models, multi-layer geographical information systems
(GIS) data, various levels of fidelity for vehicle-terrain interaction, the
generation of optimal routes and the incorporation of uncertainty in input data
related to the vehicle, the environment, or both.

![The Next-Generation NATO Reference Mobility Model (NG-NRMM) framework
\label{fig:nato-ng-nrmm-framework}](figures/nato-ng-nrmm-framework.png){height="180pt"}

ng-nrmm-framework.png

## Statement of need

Contrary to its predecessor, NG-NRMM does not provide an integrated software
toolset to perform mobility analyses, but rather prescribes the architecture,
requirements and, to a degree, a common standard for software to be developed
around. While this encourages the extension of legacy simulation frameworks to
adopt the new standard and ultimately promotes to development of more
independent software solutions, solutions in the open domain are, at the time of
writing, very limited or non-existing.

The popular open source multiphysics simulation middleware Project Chrono has
been successfully proven throughout the development of NG-NRMM by means of
benchmarking. It was the only open source solution part of the effort. While
many popular GIS software solutions are available in the public domain, both as
comprehensive software suites (such as the popular [@graser2025qgis] QGIS) or as
libraries (such as [GDAL](https://gdal.org),
[Rasterio](https://pypi.org/project/rasterio/),
[GeoPandas](https://geopandas.org), [Pronto
Raster](https://ahhz.github.io/raster/)), none are specifically tailored to
mobility analyses and significant development efforts are required to process
the geographical and simulation data into mobility maps.

MOMA is a Python package to streamline the generation of mobility maps. MOMA can
load GIS data in raster format (Figure \ref{fig:moma-elevation-map-3d}) from
online data sources or from disk to build a site specification. Moreover, MOMA
can load simulation data in HDF5 format to characterize vehicle mobility on a
set of terrains with specified slope and soil type and build a surrogate model
suitable for mobility map generation. MOMA can generate mobility maps based on
omnidirectional (averaged) slope information or directional slope maps in the
cardinal and inter-cardinal directions, providing the speed-made-good (SMG)
metric over the imported terrain.

![Three-dimensional elevation map imported in MOMA
\label{fig:moma-elevation-map-3d}](figures/moma-elevation-map-3d.png){height="180pt"}

MOMA is built in an object-oriented approach that allows for easy extension to
custom dataloaders (importing specific data), processing of custom land use or
soil moisture information, importing of custom simulation data. MOMA also
provides basic route planning capability (Figure
\ref{fig:moma-optimal-path-omnidirectional}) in omnidirectional and directional
mode by solving a minimum cost problem over the mobility graph.

![Optimal path based on the omnidirectional mobility map
\label{fig:moma-optimal-path-omnidirectional}](figures/moma-optimal-path-omnidirectional.pdf){height="180pt"}

# References