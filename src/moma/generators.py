"""
Generators module
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


import importlib
import json
import pathlib

from types import SimpleNamespace

import numpy as np
import jax.random as jrand
import matplotlib.pyplot as plt

from matplotlib import rcParams, gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from PIL import Image

from moma.dataloader import save_new_raster
from moma.mapping import LatinHyperCube

plt.style.use(
    importlib.resources.path("moma", "data") / pathlib.Path("matplotlib/moma.mplstyle")
)

# Ensure that LaTeX is not used for rendering text in plots
#plt.rc('text', usetex=False)


class TestMapGenerator:
    __test__ = False # Mark as non test class for pytest

    def __init__(self):
        pass

    def fill_slopes(self, Z, sections_start_and_end):
        """
        Fills slopes between specified sections in the map array.

        Fills the gaps between sections in the map array by interpolating
        linearly between the start and end points of each section.

        Args:
            Z: A 2D numpy array representing the map array to be filled.

            sections_start_and_end: A list containing two lists of indices.
                The first list contains x-coordinates of section boundaries,
                and the second list contains y-coordinates of section boundaries.

        Returns:
            Z: The modified 2D numpy array with slopes filled between sections.
        """

        for i in range(1, len(sections_start_and_end[0]) - 2, 2):

            x1 = sections_start_and_end[0][i] - 1
            x2 = sections_start_and_end[0][i + 1] + 1

            Z[:, x1:x2] = np.linspace(Z[:, x1], Z[:, x2], x2 - x1).T

        for j in range(1, len(sections_start_and_end[1]) - 2, 2):

            y1 = sections_start_and_end[1][j] - 1
            y2 = sections_start_and_end[1][j + 1] + 1

            Z[y1:y2, :] = np.linspace(Z[y1, :], Z[y2, :], y2 - y1)

        return Z

    def fill_pad(self, Z, start_pad, transition_pad, sections_start_and_end):
        """
        Fills the padding regions of a 2D array with values from the non-padding
        regions.

        The function fills the left, right, bottom, and top padding areas of the
        input array using values from the corresponding non-padding sections. It
        also fills the transition regions between the padding and non-padding
        areas using linear interpolation.

        Args:
            Z: A 2D numpy array representing the data to be padded.

            start_pad: Integer indicating the width of the initial padding
            region.

            transition_pad: Integer indicating the width of the transition
            padding region.
            
            sections_start_and_end: A tuple of two arrays, each containing the
            start and end indices of the non-padding sections along the
            respective axis.

        Returns:
            Z: The modified 2D numpy array with padding filled in.
        """

        # Left
        values = Z[
            np.round(
                (sections_start_and_end[1][:-1] + sections_start_and_end[1][1:]) / 2
            ).astype(int),
            start_pad + transition_pad,
        ]

        Z[: sections_start_and_end[1][1], :start_pad] = values[0]
        for idx in range(1, len(values) - 1):
            Z[
                sections_start_and_end[1][idx] : sections_start_and_end[1][idx + 1],
                :start_pad,
            ] = values[idx]
        Z[sections_start_and_end[1][-2] :, :start_pad] = values[-1]

        # Right
        values = Z[
            np.round(
                (sections_start_and_end[1][:-1] + sections_start_and_end[1][1:]) / 2
            ).astype(int),
            -(start_pad + transition_pad + 1),
        ]

        Z[: sections_start_and_end[1][1], -start_pad:] = values[0]
        for idx in range(1, len(values) - 1):
            Z[
                sections_start_and_end[1][idx] : sections_start_and_end[1][idx + 1],
                -start_pad:,
            ] = values[idx]
        Z[sections_start_and_end[1][-2] :, -start_pad:] = values[-1]

        # Bottom
        values = Z[
            start_pad + transition_pad,
            np.round(
                (sections_start_and_end[0][:-1] + sections_start_and_end[0][1:]) / 2
            ).astype(int),
        ]

        Z[:start_pad, : sections_start_and_end[0][1]] = values[0]
        for idx in range(1, len(values) - 1):
            Z[
                :start_pad,
                sections_start_and_end[0][idx] : sections_start_and_end[0][idx + 1],
            ] = values[idx]
        Z[:start_pad, sections_start_and_end[0][-2] :] = values[-1]

        # Top
        values = Z[
            -(start_pad + transition_pad + 1),
            np.round(
                (sections_start_and_end[0][:-1] + sections_start_and_end[0][1:]) / 2
            ).astype(int),
        ]

        Z[-start_pad:, : sections_start_and_end[0][1]] = values[0]
        for idx in range(1, len(values) - 1):
            Z[
                -start_pad:,
                sections_start_and_end[0][idx] : sections_start_and_end[0][idx + 1],
            ] = values[idx]
        Z[-start_pad:, sections_start_and_end[0][-2] :] = values[-1]

        # Fill transition
        # Left
        x1 = start_pad
        x2 = start_pad + transition_pad
        y1 = start_pad + transition_pad
        y2 = -(start_pad + transition_pad)
        Z[y1:y2, x1:x2] = np.linspace(Z[y1:y2, x1 - 1], Z[y1:y2, x2 + 1], x2 - x1).T

        # Right
        x1 = -(start_pad + transition_pad)
        x2 = -start_pad
        y1 = start_pad + transition_pad
        y2 = -(start_pad + transition_pad)
        Z[y1:y2, x1:x2] = np.linspace(Z[y1:y2, x1 - 1], Z[y1:y2, x2 + 1], x2 - x1).T

        # Bottom
        x1 = start_pad
        x2 = -start_pad
        y1 = start_pad
        y2 = start_pad + transition_pad
        Z[y1:y2, x1:x2] = np.linspace(Z[y1 - 1, x1:x2], Z[y2 + 1, x1:x2], y2 - y1)

        # Top
        x1 = start_pad
        x2 = -start_pad
        y1 = -(start_pad + transition_pad)
        y2 = -start_pad
        Z[y1:y2, x1:x2] = np.linspace(Z[y1 - 1, x1:x2], Z[y2 + 1, x1:x2], y2 - y1)

        return Z

    def generate_test_map(self,
        seed=42,
        max_slope_percent=30,
        map_dim=(500, 500),
        plateaus=(5, 5),
        slope_length=(50, 50),
        resolution=4,# Pixel/meter        
        transition_pad = 15,
        start_pad = 10,
        ):
        """
        Generates a synthetic elevation map with plateaus, slopes, and padding.

        This function creates a 2D elevation map with a specified number of
        plateaus and slopes, using random values within a given slope percentage
        range. It also adds padding regions and fills them with interpolated
        values.

        Args:
            seed: Integer seed for the random number generator.

            max_slope_percent: Maximum slope percentage for the slope sections.

            map_dim: Tuple (width, height) of the map in meters.

            plateaus: Tuple (num_plateaus_x, num_plateaus_y) indicating the
            number of plateaus in each direction.

            slope_length: Tuple (slope_length_x, slope_length_y) indicating the
            length of the slope sections in each direction.

            resolution: Integer representing the number of pixels per meter.

            transition_pad: Integer indicating the width of the transition
            padding region.

            start_pad: Integer indicating the width of the initial padding
            region.

        Returns:
            Z: The generated 2D elevation map with padding.

            map_size: Tuple (width, height) of the generated map in pixels.

            resolution: The resolution used for the map.

            sections: A list of section indices used for padding and slope
            filling.
        """

        slopes_x = plateaus[0] - 1
        slopes_y = plateaus[1] - 1

        plateau_length = (
            (map_dim[0] - slope_length[0] * slopes_x) // plateaus[0],
            (map_dim[1] - slope_length[1] * slopes_y) // plateaus[1],
        )
        # Remainder pixels to distribute among plateaus to ensure correct map size
        plateau_remainder = (
            (map_dim[0] - slope_length[0] * slopes_x) % plateaus[0],
            (map_dim[1] - slope_length[1] * slopes_y) % plateaus[1],
        )

        assert np.min(plateau_length[0]) >= 3, f"Smallest plateau length in x direction less than 3. Plateau lengths: {plateau_length}. Reduce number of plateaus or slope length."
        assert np.min(plateau_length[1]) >= 3, f"Smallest plateau length in y direction less than 3. Plateau lengths: {plateau_length}. Reduce number of plateaus or slope length."

        Slope_length = [
            [slope_length[0] for _ in range(slopes_x)],
            [slope_length[1] for _ in range(slopes_y)],
        ]  # Length of the slope sections in x and y direction
        Flat_length = [
            [plateau_length[0] + (1 if i + 1 <= plateau_remainder[0] else 0) for i in range(plateaus[0])],
            [plateau_length[1] + (1 if i + 1 <= plateau_remainder[1] else 0) for i in range(plateaus[1])],
        ]  # Length of the flat  sections in x and y direction

        Flat_sections = [len(Flat_length[0]), len(Flat_length[1])]

        elevation_grid = np.zeros((Flat_sections[0], Flat_sections[1]))

        # Initialize a PRNG key.
        key = jrand.PRNGKey(seed)
        # Create a key for each cell in the 2D grid.
        cell_keys = jrand.split(key, Flat_sections[0] * Flat_sections[1]).reshape(
            Flat_sections[0], Flat_sections[1], -1
        )

        for i in range(Flat_sections[0]):
            for j in range(Flat_sections[1]):
                if i == 0 and j == 0:
                    pass
                elif i == 0:
                    max_diff = Slope_length[1][j - 1] * max_slope_percent / 100
                    max = elevation_grid[i, j - 1] + max_diff
                    min = elevation_grid[i, j - 1] - max_diff

                    elevation_grid[i, j] = jrand.uniform(
                        cell_keys[i, j, :], shape=(1,), minval=min, maxval=max
                    )[0]
                elif j == 0:
                    max_diff = Slope_length[0][i - 1] * max_slope_percent / 100
                    max = elevation_grid[i - 1, j] + max_diff
                    min = elevation_grid[i - 1, j] - max_diff

                    elevation_grid[i, j] = jrand.uniform(
                        cell_keys[i, j, :], shape=(1,), minval=min, maxval=max
                    )[0]
                else:
                    diff_i = Slope_length[0][i - 1] * max_slope_percent / 100
                    diff_j = Slope_length[1][j - 1] * max_slope_percent / 100
                    max = np.min(
                        [
                            elevation_grid[i - 1, j] + diff_i,
                            elevation_grid[i, j - 1] + diff_j,
                        ]
                    )
                    min = np.max(
                        [
                            elevation_grid[i - 1, j] - diff_i,
                            elevation_grid[i, j - 1] - diff_j,
                        ]
                    )

                    elevation_grid[i, j] = jrand.uniform(
                        cell_keys[i, j, :], shape=(1,), minval=min, maxval=max
                    )[0]

        # Make min value equal to 0
        elevation_grid -= np.min(elevation_grid)

        map_size = (
            sum(Flat_length[0]) + sum(Slope_length[0]),
            sum(Flat_length[1]) + sum(Slope_length[1]),
        )

        sections = [
            [0, Flat_length[0][0]],
            [0, Flat_length[1][0]],
        ]

        for i in range(1, Flat_sections[0]):
            sections[0].append(sections[0][-1] + Slope_length[0][i - 1])
            sections[0].append(sections[0][-1] + Flat_length[0][i])

        for j in range(1, Flat_sections[1]):
            sections[1].append(sections[1][-1] + Slope_length[1][j - 1])
            sections[1].append(sections[1][-1] + Flat_length[1][j])

        sections[0] = np.array(sections[0])
        sections[1] = np.array(sections[1])
        sections_start_and_end = [sections[0] * resolution, sections[1] * resolution]

        Z = np.zeros((map_size[1] * resolution, map_size[0] * resolution))

        for i in range(Flat_sections[0]):
            for j in range(Flat_sections[1]):
                Z[
                    sections_start_and_end[1][j * 2] : sections_start_and_end[1][
                        j * 2 + 1
                    ],
                    sections_start_and_end[0][i * 2] : sections_start_and_end[0][
                        i * 2 + 1
                    ],
                ] = elevation_grid[i, j]

        Z = self.fill_slopes(Z, sections_start_and_end)

        map_size = (
            map_size[0] + 2 * (transition_pad + start_pad),
            map_size[1] + 2 * (transition_pad + start_pad),
        )

        sections[0] += transition_pad + start_pad
        sections[1] += transition_pad + start_pad

        transition_pad = round(transition_pad * resolution)
        start_pad = round(start_pad * resolution)

        sections_start_and_end[0] += transition_pad + start_pad
        sections_start_and_end[1] += transition_pad + start_pad

        if transition_pad > 0 and start_pad > 0:
            new_Z = np.zeros((map_size[1] * resolution, map_size[0] * resolution))

            new_Z[
                (transition_pad + start_pad) : -(transition_pad + start_pad),
                (transition_pad + start_pad) : -(transition_pad + start_pad),
            ] = Z
            Z = new_Z
            Z = self.fill_pad(
                Z, start_pad, transition_pad, sections_start_and_end
            )

        return Z, map_size, resolution, sections

    def generate_test_map_uncertainty(self, map_size, resolution, uncertainty=0.02):
        """
        Generates a 2D uncertainty map with uniform uncertainty values.

        This function creates a 2D array representing uncertainty values across
        the map, where all values are set to a constant uncertainty level.

        Args:
            map_size: Tuple (width, height) of the map in meters.
            
            resolution: Integer representing the number of pixels per meter.
            
            uncertainty: Float indicating the constant uncertainty value for the
            entire map.

        Returns:
            Z: A 2D numpy array of shape (height * resolution, width * resolution)
                containing the constant uncertainty value.
        """
        Z = np.ones((map_size[1] * resolution, map_size[0] * resolution)) * uncertainty

        return Z

    def smooth(self, Z, sigma, truncate):
        """
        Applies a Gaussian filter to smooth the input 2D array.

        This function uses a Gaussian filter to reduce noise and smooth the
        elevation or uncertainty values in the input array.

        Args:
            Z: A 2D numpy array representing the elevation or uncertainty data.
            
            sigma: Standard deviation of the Gaussian kernel. Controls the
            amount of smoothing applied.
            
            truncate: Truncate the filter at this multiple of sigma. A larger
            value results in a wider filter.

        Returns:
            Z: The smoothed 2D numpy array.
        """
        return gaussian_filter(Z, sigma=sigma, truncate=truncate)

    def plot_surface(
        self, output_dir, elevation, sections, map_size, show_plot=False, subname=""
    ):
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[8, 10], wspace=0.4)

        # 2D plot
        ax1 = fig.add_subplot(gs[0])
        img = ax1.imshow(
            elevation, cmap="RdYlBu_r", extent=[0, map_size[0], 0, map_size[1]]
        )

        ax1.vlines(
            sections[0],
            ymin=sections[1][0],
            ymax=sections[1][-1],
            color="grey",
            linestyle="--",
            linewidth=0.5,
        )
        ax1.hlines(
            sections[1],
            xmin=sections[0][0],
            xmax=sections[0][-1],
            color="grey",
            linestyle="--",
            linewidth=0.5,
        )

        ax1.set_title("Elevation Map (2D)")
        ax1.set_xlabel("[m]")
        ax1.set_ylabel("[m]")

        ax1.grid(False)

        # Match colorbar size to plot size
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        colorbar = plt.colorbar(img, cax=cax)
        colorbar.set_label("Elevation [m]")

        ax2 = fig.add_subplot(gs[1], projection="3d")
        ax2.set_facecolor(fig.get_facecolor())

        # Create a meshgrid for the x and y coordinates
        x = np.linspace(0, map_size[0], elevation.shape[1])
        y = np.linspace(0, map_size[1], elevation.shape[0])
        X, Y = np.meshgrid(x, y)
        surf = ax2.plot_surface(
            X, Y, np.flip(elevation, axis=0), cmap="RdYlBu_r"
        )  # , edgecolor="k")
        ax2.set_title("Elevation Map (3D Surface)")
        ax2.set_xlabel("[m]")
        ax2.set_ylabel("[m]")
        ax2.set_zlabel("[m]")

        ax2.view_init(30, 225)

        ax2.set_box_aspect((1, 1, 1), zoom=0.9)

        fig.savefig(
            output_dir / f"test_map{subname}.pdf", format="pdf", bbox_inches="tight"
        )  # , pad_inches=0.5)

        if show_plot:
            plt.show()
        plt.close(fig)

    def save_data_for_moma(
        self,
        raster_dir,
        elevation,
        uncertainty_map,
        map_size,
        resolution,
        friction_coefficient,
        friction_coefficient_std,
        diffuse_map,
        coordinates_lower_left=(0, 0),
    ):
        """
        Plots a 2D and 3D surface representation of the elevation map.

        This function generates a figure with two subplots: a 2D color map and a
        3D surface plot of the elevation data. It also adds vertical and
        horizontal lines to indicate section boundaries.

        Args:
            output_dir: Directory path where the plot will be saved.
            
            elevation: 2D numpy array representing the elevation data.
            
            sections: A list of section indices used to define boundaries in the
            elevation map.
            
            map_size: Tuple (width, height) of the map in meters.
            
            show_plot: Boolean indicating whether to display the plot.
            
            subname: Optional string to append to the filename for
            differentiation.

        Returns:
            None
        """
        # Save elevation raster
        print(f"Save rasters in folder: {raster_dir}...")

        # Coordinates of lower left corner
        lat = coordinates_lower_left[0]  # E
        long = coordinates_lower_left[1]  # N

        # Site.yaml file
        with open(raster_dir / ".." / "site.yaml", "w") as site:
            site.write(
f"""    DataSource: DataforsyningenFriction
    Coordinates: [{lat}, {long}, {lat + map_size[0]}, {lat + map_size[0]}] # [W, S, E, N]
    NoGoZones:
    #   - [x1, x2, y1, y2, (Description)]: Box
    #   - [x, y, r, (Description)]: Circle
    """
            )

        coor_string = f"{lat}_{long}_{lat + map_size[0]}_{lat + map_size[0]}"

        # Reduce elevation to 1x1m grid and add extra dimension
        elevation = elevation[
            resolution // 2 :: resolution, resolution // 2 :: resolution
        ][np.newaxis, :]
        uncertainty_map = uncertainty_map[
            resolution // 2 :: resolution, resolution // 2 :: resolution
        ][np.newaxis, :]

        elevation_combined = np.concatenate((elevation, uncertainty_map))

        transform = Affine(1, 0, lat, 0, -1, map_size[1] + long)
        path = raster_dir / f"elevationmono_1m_clipped_{coor_string}.tif"
        save_new_raster(elevation_combined, path, transform)

        # Save diffuse map as areal view
        diffuse_map = np.stack(
            (
                diffuse_map[::resolution, ::resolution, 0],
                diffuse_map[::resolution, ::resolution, 1],
                diffuse_map[::resolution, ::resolution, 2],
                np.zeros_like(diffuse_map[::resolution, ::resolution, 0]),
            ),
            axis=0,
        ).astype(np.uint8)
        path = raster_dir / f"aerial_1m_clipped_{coor_string}.tif"
        save_new_raster(diffuse_map, path, transform)

        # Create land use raster
        land_use = np.concatenate(
            (
                np.ones_like(elevation) * 0,
                np.ones_like(elevation) * 127,
                np.ones_like(elevation) * 20,
                np.zeros_like(elevation),
            )
        )

        path = raster_dir / f"landuse_1m_clipped_{coor_string}.tif"
        save_new_raster(land_use, path, transform)

        # Create soil type raster with hard terrain
        soil_type = np.concatenate(
            (
                np.ones_like(elevation) * 0,
                np.ones_like(elevation) * 0,
                np.ones_like(elevation) * 0,
                np.zeros_like(elevation),
            )
        )

        soils = {
            0: {"label": "hard clay", "color": [0, 0, 0]},
            1: {"label": "hard clay", "color": [0, 0, 0]},
            #1: {"label": "hard gravel", "color": [50, 50, 50]},
        }

        shape = soil_type.shape[1:]
        for i in range(3):
            soil_type[i, : int(shape[0] / 2), :] = soils[0]["color"][i]
            soil_type[i, int(shape[0] / 2) :, :] = soils[1]["color"][i]

        path = raster_dir / f"soiltype_1m_clipped_{coor_string}.tif"
        save_new_raster(soil_type, path, transform)

        # Saves soil to use in DYNO
        soil_export = {"shape": shape, "resolution": 1, "labels": soils, "major": "row"}
        soil_data_export = np.zeros(shape)
        for i, soil in soils.items():

            soil_data_export[
                (soil_type[0, :, :] == soil["color"][0])
                & (soil_type[1, :, :] == soil["color"][1])
                & (soil_type[2, :, :] == soil["color"][2])
            ] = i

        soil_export["data"] = list(soil_data_export.flatten())

        with open(
            raster_dir / f"soiltype_1m_clipped_{coor_string}.json", "w"
        ) as json_file:
            json.dump(soil_export, json_file)

        # Create moisture content raster
        moisture_content = np.concatenate(
            (
                np.ones_like(elevation) * 150,
                np.ones_like(elevation) * 150,
                np.ones_like(elevation) * 255,
                np.zeros_like(elevation),
                np.zeros_like(elevation),
            )
        )

        path = raster_dir / f"moisturecontent_1m_clipped_{coor_string}.tif"
        save_new_raster(moisture_content, path, transform)

        # Create friction coefficient raster
        friction_low = 0.4
        friction_high = 0.5
        friction = np.ones_like(elevation)
        
        friction[
            0,
            (soil_type[0, :, :] == 0)
            & (soil_type[1, :, :] == 0)
            & (soil_type[2, :, :] == 0),
        ] = friction_low
        friction[
            0,
            (soil_type[0, :, :] == 50)
            & (soil_type[1, :, :] == 50)
            & (soil_type[2, :, :] == 50),
        ] = friction_high
        friction_coefficient_map = np.concatenate(
            (friction, np.ones_like(elevation) * friction_coefficient_std)
        )

        path = raster_dir / f"friction_coefficient_1m_clipped_{coor_string}.tif"
        save_new_raster(friction_coefficient_map, path, transform)

    def metadata(self, elev_min, elev_max, map_size, friction_coefficient):
        """
        Generate metadata dictionary for the elevation map.

        This function creates a dictionary containing key information about the
        generated elevation map, including elevation range, map dimensions, and
        friction coefficient.

        Args:
            elev_min: Minimum elevation value in the map.

            elev_max: Maximum elevation value in the map.
            
            map_size: Tuple (width, height) representing the map dimensions in
            meters. friction_coefficient: Coefficient representing the friction
            of the surface.

        Returns:
            meta: Dictionary containing metadata for the elevation map.
        """
        meta = {
            "description": "slope test map",
            "elevation": {"max": float(elev_max), "min": float(elev_min)},
            "map_size": {"width": float(map_size[0]), "height": float(map_size[1])},
            "friction_coefficient": float(friction_coefficient),
        }
        return meta

    def generate_diffuse_map(self, elevation):
        """
        Generate a diffuse map from the elevation data by interpreting the
        slopes in the x and y directions as R and G channels for RGB color.

        This function computes the gradient of the elevation data to determine
        the slope in the x and y directions. These slopes are then normalized
        and used to create an RGB image where the red channel corresponds to the
        x-slope and the green channel corresponds to the y-slope. The blue
        channel is set to zero.

        Args:
            elevation: 2D numpy array representing the elevation data.

        Returns:
            diffuse_map: A 3D numpy array of shape (height, width, 3)
            representing the diffuse map in RGB format.
        """
        slope_x, slope_y = np.gradient(elevation)

        slope_x = self.smooth(slope_x, sigma=2, truncate=15)
        slope_y = self.smooth(slope_y, sigma=2, truncate=15)

        x_min, x_max = np.min(slope_x), np.max(slope_x)
        y_min, y_max = np.min(slope_y), np.max(slope_y)

        x_normalized = 230 * (slope_x - x_min) / (x_max - x_min) + 255 - 230
        y_normalized = 230 * (slope_y - y_min) / (y_max - y_min) + 255 - 230

        x_normalized = x_normalized.astype(np.uint8)
        y_normalized = y_normalized.astype(np.uint8)

        return np.dstack((x_normalized, y_normalized, np.zeros_like(x_normalized)))

    def save_diffuse_map(self, output_dir, elevation, subname=""):
        """
        Saves the diffuse map as a PNG file.

        This function generates a diffuse map from the elevation data and saves
        it as a PNG file in the specified output directory.

        Args:
            output_dir: Directory path where the diffuse map will be saved.

            elevation: 2D numpy array representing the elevation data.

            subname: Optional string to append to the filename for
            differentiation.

        Returns:
            None
        """
        # Save the diffuse map as a .png file
        diffuse_map = self.generate_diffuse_map(elevation)
        diffuse_map_img = Image.fromarray(diffuse_map)

        diffuse_map_img.save(output_dir / f"map-diffuse{subname}.png")

    def save_bitmap(self,
        output_dir,
        elevation,
        map_size,
        friction_coefficient,
        subname="",
        output_resolution=6,
    ):
        """
        Saves the elevation data as a .bmp file and metadata as a .json file.

        This function downsamples the elevation data to a specified resolution
        and normalizes it to the 0-255 range. It then saves the resulting image
        as a .bmp file and the metadata as a .json file.

        Args:
            output_dir: Directory path where the files will be saved.

            elevation: 2D numpy array representing the elevation data.
            
            map_size: Tuple (width, height) representing the map dimensions in
            meters.
            
            friction_coefficient: Coefficient representing the friction of the
            surface.
            
            subname: Optional string to append to the filename for
            differentiation.
            
            output_resolution: Integer indicating the resolution for
            downsampling.

        Returns:
            None
        """  

        # Downsample the elevation data to (0.25*output_resolution)x(0.25*output_resolution)m grid
        elevation_out = elevation[::output_resolution, ::output_resolution]

        # Normalize the data to the 0-255 range:
        elev_min, elev_max = elevation_out.min(), elevation_out.max()
        elevation_normalized = 255 * (elevation_out - elev_min) / (elev_max - elev_min)

        elevation_normalized = elevation_normalized.astype(np.uint8)

        img = Image.fromarray(elevation_normalized)

        # Save the image as a .bmp file and the metadata as a .json file
        img.save(output_dir / f"map-elevation{subname}.bmp")

        with open(output_dir / f"map-meta{subname}.json", "w") as f:
            json.dump(
                self.metadata(
                    elev_min, elev_max, map_size, friction_coefficient
                ),
                f,
                indent=4,
            )

    def save_obj(self,
        output_dir,
        elevation,
        map_size,
        resolution,
        friction_coefficient,
        subname="",
        output_resolution=6,
    ):
        """
        Saves the elevation data as an .obj file and metadata as a .json file.

        This function downsamples the elevation data to a specified resolution
        and generates a 3D mesh in the .obj format. It also saves the metadata
        as a .json file.

        Args:
            output_dir: Directory path where the files will be saved.
            
            elevation: 2D numpy array representing the elevation data.
            
            map_size: Tuple (width, height) representing the map dimensions in
            meters.
            
            resolution: Integer representing the number of pixels per meter.
            
            friction_coefficient: Coefficient representing the friction of the
            surface.
            
            subname: Optional string to append to the filename for
            differentiation.
            
            output_resolution: Integer indicating the resolution for
            downsampling.

        Returns:
            None
        """
        # Downsample the elevation data to (0.25*output_resolution)x(0.25*output_resolution)m grid
        elevation_out = elevation[::output_resolution, ::output_resolution]

        elev_min, elev_max = elevation_out.min(), elevation_out.max()

        rows, cols = elevation_out.shape

        # Create a list for vertices and faces
        vertices = []
        faces = []

        # Generate vertices
        for i in range(rows):
            for j in range(cols):
                # x=j, y=i, and z is the elevation value
                vertices.append(
                    f"v {j/resolution} {i/resolution} {elevation_out[i, j]}"
                )

        # Generate faces.
        # Each cell of the grid (except the last row/col) is divided into two triangles.
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Calculate the indices of the four corners in the vertices list.
                # OBJ files use 1-based indexing.
                top_left = i * cols + j + 1
                top_right = top_left + 1
                bottom_left = (i + 1) * cols + j + 1
                bottom_right = bottom_left + 1

                faces.append(f"f {top_left} {bottom_left} {top_right}")
                faces.append(f"f {top_right} {bottom_left} {bottom_right}")

        # Save the elevation data as .obj file and the metadata as a .json file

        with open(output_dir / f"map-elevation{subname}.obj", "w") as f:
            f.write("\n".join(vertices) + "\n")
            f.write("\n".join(faces) + "\n")

        with open(output_dir / f"map-meta{subname}.json", "w") as f:
            json.dump(
                self.metadata(
                    elev_min, elev_max, map_size, friction_coefficient
                ),
                f,
                indent=4,
            )

    def slope_hist(self, output_dir, elevation, resolution, show_plot=False):
        """
        Plots a histogram of the slope distribution in the elevation map.

        This function calculates the slope magnitude at each point in the
        elevation map and generates a histogram of the slope distribution. The
        histogram is saved as a PNG file in the specified output directory.

        Args:
            output_dir: Directory path where the histogram will be saved.

            elevation: 2D numpy array representing the elevation data.

            resolution: Integer representing the number of pixels per meter.

            show_plot: Boolean indicating whether to display the plot.

        Returns:
            None
        """
        # Calculate the slope of the elevation map
        gradient_x, gradient_y = np.gradient(elevation)
        slope = np.sqrt(gradient_x**2 + gradient_y**2).flatten() * resolution * 100

        # Plot the histogram of slopes
        plt.figure(figsize=(10, 6))
        plt.hist(slope, bins=100, density=True, alpha=0.6, color="g")
        plt.title("Slope in steepest direction")
        plt.xlabel("Slope $\%$")
        plt.ylabel("Density")

        plt.tight_layout()

        plt.savefig(output_dir / "test_map_slope_histogram.png", bbox_inches="tight")

        if show_plot:
            plt.show()
        plt.close()

    def main_generate(self,
        create_rasters=False,
        base_output_dir=None,
        plot_additional_files=False,
        N_test_maps=1,
        elevation_std=0.00,
        friction_coefficient=0.5,
        friction_coefficient_std=0.00,
        map_type="bmp",
        output_resolution=6,
        coordinates_lower_left=(0, 0),
        test_map_params=SimpleNamespace(
            seed=42,
            max_slope_percent=30,
            map_dim=(20, 20),
            plateaus=(2, 2),
            slope_length=(10, 10),        
            transition_pad = 15,
            start_pad = 10,
            resolution=4,
        )
    ):
        """
        Main function to generate test maps and save them in the specified
        format.

        This function generates a synthetic elevation map with plateaus and
        slopes, applies smoothing, generates uncertainty maps, and saves the
        results in either .bmp or .obj format. It also optionally saves diffuse
        maps and generates multiple test maps with random variations.

        Args:
            create_rasters: Boolean indicating whether to create raster files.
            
            base_output_dir: Directory path where the output files will be
            saved.
            
            plot_additional_files: Boolean indicating whether to generate
            additional plots.
            
            N_test_maps: Number of test maps to generate.
            
            elevation_std: Standard deviation for the elevation uncertainty.
            
            friction_coefficient: Friction coefficient for the map.
            
            friction_coefficient_std: Standard deviation for the friction
            coefficient uncertainty.
            
            map_type: Type of map to save, either 'bmp' or 'obj'.
            
            output_resolution: Resolution for downsampling the elevation data.
            
            coordinates_lower_left: Tuple (x, y) representing the lower-left
            coordinates.
            
            test_map_params: Namespace containing parameters for generating the
            test map.

        Returns:
            None
        """

        # Fill in default parameters if not provided
        if getattr(test_map_params, "seed", None) is None:
            test_map_params.seed = 42
        if getattr(test_map_params, "max_slope_percent", None) is None:
            test_map_params.max_slope_percent = 30
        if getattr(test_map_params, "map_dim", None) is None:
            test_map_params.map_dim = (500, 500)
        if getattr(test_map_params, "plateaus", None) is None: 
            test_map_params.plateaus = (5, 5)
        if getattr(test_map_params, "slope_length", None) is None:
            test_map_params.slope_length = (50, 50)
        if getattr(test_map_params, "resolution", None) is None:
            test_map_params.resolution = 4
        if getattr(test_map_params, "transition_pad", None) is None:
            test_map_params.transition_pad = 15
        if getattr(test_map_params, "start_pad", None) is None:
            test_map_params.start_pad = 10

        # If map is not returned to caller function as an array, create the output directory
        if base_output_dir is None:
            base_output_dir = pathlib.Path("data") / "sites" / "testmap"
        else:
            base_output_dir = pathlib.Path(base_output_dir)

        raster_dir = base_output_dir / "rasters"
        output_dir = base_output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Output directory: \n\tBitmaps: {output_dir}"
        )

        if create_rasters:
            raster_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"\tRasters: {raster_dir}"
            )

        print("Generating test map...")
        elevation, map_size, resolution, sections = self.generate_test_map(
            seed=test_map_params.seed,
            max_slope_percent=test_map_params.max_slope_percent,
            map_dim=test_map_params.map_dim,
            plateaus=test_map_params.plateaus,
            slope_length=test_map_params.slope_length,
            resolution=test_map_params.resolution,
            transition_pad=test_map_params.transition_pad,
            start_pad=test_map_params.start_pad,
        )
        
        smoothed_elevation = self.smooth(elevation, sigma=15, truncate=60)

        uncertainty_map = self.generate_test_map_uncertainty(
            map_size, resolution, uncertainty=elevation_std
        )

        if plot_additional_files:
            print("Plotting test map...")
            self.plot_surface(
                output_dir, elevation, sections, map_size, show_plot=False
            )
            self.plot_surface(
                output_dir,
                smoothed_elevation,
                sections,
                map_size,
                show_plot=False,
                subname="_smoothed",
            )

            print("Plotting slope histogram...")
            self.slope_hist(output_dir, smoothed_elevation, resolution)

        # Save the diffuse map as a .png file for terrain texture

        if create_rasters:
            print("Saving test map for MOMA...")
            diffuse = self.generate_diffuse_map(smoothed_elevation)
            self.save_data_for_moma(
                raster_dir,
                smoothed_elevation,
                uncertainty_map,
                map_size,
                resolution,
                friction_coefficient,
                friction_coefficient_std,
                diffuse,
                coordinates_lower_left
            )

        print("Generation samples for DYNO...")
        smoothed_sampled_elevation = np.array(
            LatinHyperCube(smoothed_elevation, uncertainty_map, N_test_maps, seed=21)
        )
        friction_coefficient_sampled = np.array(
            LatinHyperCube(
                np.array([[friction_coefficient]]),
                np.array([[friction_coefficient_std]]),
                N_test_maps,
                seed=23,
            )
        ).reshape(N_test_maps)

        if plot_additional_files:
            print("Plotting one test map with uncertainty...")
            self.plot_surface(
                output_dir,
                smoothed_sampled_elevation[0, :, :],
                sections,
                map_size,
                show_plot=False,
                subname="_smoothed_sampled",
            )

        # Save a single map without
        if map_type == "bmp":
            self.save_bitmap(
                output_dir,
                smoothed_elevation,
                map_size,
                friction_coefficient=friction_coefficient,
                subname="-without",
                output_resolution=output_resolution,
            )
        elif map_type == "obj":
            self.save_obj(
                output_dir,
                smoothed_elevation,
                map_size,
                resolution,
                friction_coefficient=friction_coefficient,
                subname="-without",
                output_resolution=output_resolution,
            )
        else:
            raise ValueError(
                "Map type {map_type} not supported, must be either 'bmp' or 'obj'."
            )

        # Save the diffuse map as a .png file for terrain texture
        self.save_diffuse_map(
            output_dir, smoothed_elevation, subname="-without"
        )

        for i in range(smoothed_sampled_elevation.shape[0]):
            print(f"Saving test map {i + 1} of {smoothed_sampled_elevation.shape[0]}...")
            if map_type == "bmp":
                self.save_bitmap(
                    output_dir,
                    smoothed_sampled_elevation[i, :, :],
                    map_size,
                    friction_coefficient=friction_coefficient_sampled[i],
                    subname=f"-{i}",
                    output_resolution=output_resolution,
                )
            elif map_type == "obj":
                self.save_obj(
                    output_dir,
                    smoothed_sampled_elevation[i, :, :],
                    map_size,
                    resolution,
                    friction_coefficient=friction_coefficient_sampled[i],
                    subname=f"-{i}",
                    output_resolution=output_resolution,
                )

            # Save the diffuse map as a .png file for terrain texture
            self.save_diffuse_map(
                output_dir, smoothed_sampled_elevation[i, :, :], subname=f"-{i}"
            )
