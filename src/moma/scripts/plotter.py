"""
3D plotting script
"""

"""
 #############################################################################
 #                            _     _     _     _                            #
 #                           / \   / \   / \   / \                           #
 #                          ( M ) ( O ) ( M ) ( A )                          #
 #                           \_/   \_/   \_/   \_/                           #
 #                                                                           #
 #   MOMA: Mobility maps and optimal routes generator for ground vehicles    #
 #############################################################################

MIT License

Copyright (c) 2024 Dario Sirangelo, Nikolaj Jes Petersen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import importlib.resources
import logging

from pathlib import Path

from moma.plotting import Plotter3D
from moma.math import CubicSpline2D, PathUtilities

logging.getLogger().setLevel(logging.INFO)

# ---------------------------------------------------------------------------- #
# Define the workspace path.
# ---------------------------------------------------------------------------- #
BASE_PATH = importlib.resources.path("moma", "data")

# ---------------------------------------------------------------------------- #
# Define the input rasters paths.
# ---------------------------------------------------------------------------- #
OMNIDIRECTIONAL_SLOPE_RASTER_PATH = BASE_PATH / Path("Slope map (omni).tif")
DIRECTIONAL_SLOPE_0_RASTER_PATH = BASE_PATH / Path("Slope map (0.0).tif")
ELEVATION_RASTER_PATH = BASE_PATH / Path(
    "elevationmono_1m_clipped_450565_6163390_452401_6164541.tif"
)

# ---------------------------------------------------------------------------- #
# Define the output path.
# ---------------------------------------------------------------------------- #
SAVE_PATH: Path = Path("generated_maps")


def main(show=False):
    plotter3d = Plotter3D(ELEVATION_RASTER_PATH, resample_factor=1, scale_factor=5.0)
    plotter3d.load_values(ELEVATION_RASTER_PATH, "elevation", 5)
    plotter3d.load_values(OMNIDIRECTIONAL_SLOPE_RASTER_PATH, "omnidirectional_slope", 1)
    plotter3d.load_values(DIRECTIONAL_SLOPE_0_RASTER_PATH, "directional_slope_0", 1)

    # Overlay planned path
    # =========================================================================
    # cx, cy = PathUtilities.sample_points(4, (np.min(x), np.max(x)), (np.min(y), np.max(y)))

    # cx, cy = PathUtilities.smooth_line(np.column_stack((cx, cy)).astype(np.float64))

    # pts = np.column_stack((cx, cy, cz)).astype(np.float64)
    # PathUtilities.smooth_line(pts)
    # plotter3d.plot_path()

    # Plot and save results
    # =========================================================================
    for layer, label in zip(
        ("elevation", "omnidirectional_slope", "directional_slope_0"),
        ("Elevation [m]", "Slope [deg]", "Slope [deg]"),
    ):

        plotter3d.clear()
        plotter3d.plot_map(layer=layer, label=label)
        plotter3d.set_camera_position()

        # Save the plot
        plotter3d.save(SAVE_PATH / layer)

        # Optionally, show the plot.
        if show:
            plotter3d.show()


if __name__ == "__main__":
    main()
