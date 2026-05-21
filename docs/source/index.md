:html_theme.sidebar_secondary.remove:

![header](header.png){align=center}

# MOMA documentation

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

## Features

At this time, MOMA can be used to:

* 🛰️ Load, process and visualize aerial and elevation maps in GeoTIFF format.

* 🚗 Load and process multibody dynamics simulation results to characterize
  vehicle mobility (directly if using
  [DYNO](https://github.com/au-agv/dyno))

* 🗺️ Generate directional and omni-directional mobility map for a given site

* 📍 Generate optimal traversal paths using the A* algorithm

```{toctree}
:hidden:
pages/getting-started/index.md
pages/user-guide/index.md
pages/examples/index.md
pages/api/index.md
```
