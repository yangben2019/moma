"""
Mapping module
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

import logging

from abc import ABC

import colorama
import jax.numpy as jnp
import jax.random as jrand
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from jax import jit, vmap
from jax.scipy.stats import norm
from matplotlib.colors import ListedColormap

log = logging.getLogger("MOMA")


@jit
def uniform(key, lower_bound, upper_bound):
    """
    Generates a uniform random number between lower_bound and upper_bound.

    Args:
        key (PRNGKey): PRNG key.
        lower_bound (float): Lower bound of the range.
        upper_bound (float): Upper bound of the range.

    Returns:
        float: Uniform random number.
    """
    return jrand.uniform(key, shape=(), minval=lower_bound, maxval=upper_bound)


def LatinHyperCube(means, stds, n_samples, seed=None):
    """
    Generates samples from a normal distribution using a Latin Hypercube Sampling (LHS) method.

    Args:
        means (np.ndarray): 2D array of means for each cell in the grid.
        stds (np.ndarray): 2D array of standard deviations for each cell in the grid.
        n_samples (int): Number of samples to generate.
        seed (int): Seed for the PRNG key.

    Returns:
        lhs_samples (np.ndarray): 3D array of samples from the normal distribution.
    """
    if seed is None:
        seed = np.random.randint(0, 2**16)

    means = jnp.array(means)
    stds = jnp.array(stds)
    rows, cols = means.shape

    # Initialize a PRNG key.
    key = jrand.PRNGKey(seed)
    # Create a key for each cell in the 2D grid.
    cell_keys = jrand.split(key, rows * cols).reshape(rows, cols, -1)

    @jit
    def lhs_sample(row, col):
        """
        Generates LHS samples for a single cell.

        Args:
            row (int): Row index of the cell.
            col (int): Column index of the cell.

        Returns:
            np.ndarray: LHS samples for the cell.
        """
        # Use one key per cell.
        cell_key = cell_keys[row, col, :]

        # Split the key into n+1 parts: one for permutation, n for interval sampling.
        keys = jrand.split(cell_key, n_samples + 1)
        perm_key = keys[0]
        sample_keys = keys[1:]

        # Define n intervals in [0, 1]
        intervals = jnp.linspace(0, 1, n_samples + 1)
        lower_bounds = intervals[:-1]
        upper_bounds = intervals[1:]

        # Use vmap to generate one sample per interval using a separate key
        samples = vmap(uniform, in_axes=(0, 0, 0))(
            sample_keys, lower_bounds, upper_bounds
        )  # Map over the intervals

        # Shuffle the stratified samples to remove any ordering bias.
        shuffled_samples = jrand.permutation(perm_key, samples)

        # Clip to avoid inf and -inf (and very unlikely outputs)
        epsilon = 1e-6
        shuffled_samples = jnp.clip(shuffled_samples, epsilon, 1 - epsilon)

        # Transform the uniform samples to normal samples using the inverse CDF.
        cell_samples = norm.ppf(
            shuffled_samples, loc=means[row, col], scale=stds[row, col]
        )

        return cell_samples

    # Map over each cell to generate LHS samples and then transform them to the normal distribution.
    vmap_lhs_sample = vmap(
        lhs_sample, in_axes=(0, 0), out_axes=-1
    )  # Map over the rows and columns

    index_row, index_col = np.unravel_index(np.arange(0, means.size), means.shape)

    lhs_samples = vmap_lhs_sample(index_row, index_col)

    return lhs_samples.reshape((n_samples,) + means.shape)


def sample_points(data, n_samples=1):
    """
    Samples points from a raster.

    Args:
        data (np.ndarray): 3D array of shape (n_layers, n_rows, n_cols).
        n_layers: Is either 1 (mean), 2 (mean and std) or more (mean, and different noise layers).
        n_samples (int): Number of samples to draw.

    Returns:
        np.ndarray: 3D array of shape (n_samples, n_rows, n_cols) with the sampled points.
    """

    uncertainty_layers = data.shape[0] - 1
    if uncertainty_layers > 1:
        # Mean values are in the first layer, and different noise layers are in the rest.
        return data[0] + data[1:]

    if n_samples > 1 and uncertainty_layers == 1:
        # Mean values in the first layer, and std in the second layer.
        return LatinHyperCube(data[0], data[1], n_samples)

    # If only one sample is requested or no uncertainty layers exist, return the mean value.
    return data[0][np.newaxis, ...]


class Map(ABC):
    """
    Abstract base class for maps.
    """

    pass


class AerialMap(Map):
    """Aerial map"""

    def __init__(self, name, raster):
        """
        Initialises a new aerial map.

        Args:
            name (str): Name of the map.
            raster (Raster): Raster object containing the map data.
        """
        log.info(f"Initialising new aerial map for site {name}.")

        self.name = name
        self.raster = raster

    def plot(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the aerial map.

        Args:
            export_path (str): Path to export the plot.
            show (bool): Whether to display the plot.
            export (bool): Whether to export the plot.
            save_maps (bool): Whether to save the aerial map as an image.
            hide_utm (bool): Whether to hide the UTM coordinates.
        """
        rgb = np.copy(self.raster.data)

        # Set nodata values
        rgb[rgb == 0] = 255

        rgb = np.moveaxis(rgb, 0, -1)

        self.raster.plot(
            rgb=rgb,
            export_path=export_path / "site/aerial_view.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )


class ElevationMap(Map):
    """Elevation map"""

    def __init__(self, name, raster):
        """
        Initialises a new elevation map.

        Args:
            name (str): Name of the map.
            raster (Raster): Raster object containing the map data.
        """
        log.info(f"Initialising new elevation map for site {name}.")

        self.name = name
        self.raster = raster

    def plot(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the elevation map.

        Args:
            export_path (str): Path to export the plot.
            show (bool): Whether to display the plot.
            export (bool): Whether to export the plot.
            save_maps (bool): Whether to save the elevation map as an image.
            hide_utm (bool): Whether to hide the UTM coordinates.
        """
        return_vals = self.raster.plot(
            colorbar="Elevation [m]",
            export_path=export_path / "site/elevation_map.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )

        if not (show, export, save_maps):
            return return_vals


class DirectionalMap(Map):
    """
    Directional map.
    """

    directional_rasters = {}

    # Unit distance for each direction
    DIRECTIONAL_DISTANCES_MULTIPLIER = {
        "N": 1,
        "S": 1,
        "E": 1,
        "W": 1,
        "NW": np.sqrt(2),
        "SE": np.sqrt(2),
        "NE": np.sqrt(2),
        "SW": np.sqrt(2),
    }


def direction2compass_angle(direction):
    """
    Convert a direction to a compass angle.

    Args:
        direction (str): Direction as a string, e.g. "N", "NNE", "NE",
            "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W",
            "WNW", "NW", "NNW".

    Returns:
        float: Compass angle in degrees, where 0 is North, 90 is East,
            180 is South, and 270 is West.
    """
    direction = direction.upper()
    compass_directions = {
        "N": 0,
        "NNE": 22.5,
        "NE": 45,
        "ENE": 67.5,
        "E": 90,
        "ESE": 112.5,
        "SE": 135,
        "SSE": 157.5,
        "S": 180,
        "SSW": 202.5,
        "SW": 225,
        "WSW": 247.5,
        "W": 270,
        "WNW": 292.5,
        "NW": 315,
        "NNW": 337.5,
    }

    if direction in compass_directions:
        return np.round(compass_directions[direction], 3)
    elif direction == "OMNI":
        return "omni"
    else:
        return np.round(float(direction), 3)


class SlopeMap(DirectionalMap):
    """
    Slope map.
    """

    def __init__(self, name, map_elevation, directions, path_planning):
        """
        Initialises a slope map.

        Args:
            name (str): Name of the map.
            map_elevation (ElevationMap): Elevation map.
            directions (dict): Dictionary containing the directions and angles.
        """
        log.info(f"Initialising slope map from elevation data.")

        self.name = name
        self.elevation_raster = map_elevation.raster

        self.slope_raster = self.elevation_raster.return_empty_copy()
        self.aspect_raster = self.elevation_raster.return_empty_copy()
        self.aspect_raster.colormap = "twilight"  # Set cyclic colormap

        self.slope_raster.data, self.aspect_raster.data = self.slope_aspect_vmap()(
            jnp.array(self.elevation_raster.data),
            self.elevation_raster.resolution_x,
            self.elevation_raster.resolution_y,
        )

        new_directions = []
        new_angles = []
        # all 8 primary compass directions are present for omnidirectional predictions
        if "omni" in directions["angles"] and path_planning:
            for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                angle = direction2compass_angle(direction)
                if angle not in directions["angles"]:
                    log.info(
                        f"Adding slope map for direction {direction} degrees to have all slopes available for omnidirectional predictions."
                    )
                    new_directions.append(angle)
                    new_angles.append(angle)

        # Ensure that all perpendicular slopemaps are availibe for sideslope computations
        for direction, angle in zip(directions["names"], directions["angles"]):
            new_directions.append(direction)
            new_angles.append(angle)
            if not angle == "omni":
                rotated_angle = np.mod(angle + 90, 360)
                if (
                    rotated_angle not in directions["angles"]
                    and rotated_angle not in new_angles
                ):
                    log.info(
                        f"Adding slope map for direction {rotated_angle} degrees to have sideslopes available for {direction}."
                    )
                    new_directions.append(angle)
                    new_angles.append(rotated_angle)

        # Compute directional slopes
        for direction, angle in zip(new_directions, new_angles):
            self.directional_rasters[angle] = self.elevation_raster.return_empty_copy()
            self.directional_rasters[angle].direction = direction
            self.directional_slope(direction, angle)

    def slope_aspect_vmap(self):
        """
        Creates a vectorized function to compute slope and aspect from elevation data.

        Returns:
            callable: Function that takes elevation data, x-resolution, and y-resolution as input and returns slope and aspect.
        """

        @jit
        def slope_aspect(elevation, dx, dy):
            """
            Computes slope and aspect from elevation data.

            Args:
                elevation (jnp.ndarray): Elevation data.
                dx (float): X-resolution of the elevation data.
                dy (float): Y-resolution of the elevation data.

            Returns:
                tuple[jnp.ndarray, jnp.ndarray]: Tuple containing slope and aspect data.
            """

            # Pads data to handle edges
            padded = jnp.pad(elevation, 1, mode="reflect", reflect_type="odd")

            # Slope and aspect with Horn's algorithm according to: https://xdem.readthedocs.io/en/stable/terrain.html
            h_0m = padded[1:-1, 2:]
            h_pm = padded[2:, 2:]
            h_p0 = padded[2:, 1:-1]
            h_pp = padded[2:, :-2]
            h_0p = padded[1:-1, :-2]
            h_mp = padded[:-2, :-2]
            h_m0 = padded[:-2, 1:-1]
            h_mm = padded[:-2, 2:]

            p = ((h_pp + 2 * h_p0 + h_pm) - (h_mp + 2 * h_m0 + h_mm)) / (8 * dx)
            q = ((h_pp + 2 * h_0p + h_mp) - (h_pm + 2 * h_0m + h_mm)) / (8 * dy)

            slope = jnp.sqrt(p**2 + q**2)  # Slope in ratio (rise/run)
            aspect = jnp.mod(
                jnp.arctan2(q, p), 2 * jnp.pi
            )  # Aspect (downward direction of slope) given in radians

            return slope, aspect

        return vmap(
            slope_aspect, in_axes=(0, None, None), out_axes=0
        )  # Maps over the samples

    def directional_slope(self, direction="N", direction_angle=0):
        """
        Computes the slope map in a specific direction.

        Args:
            direction (str): Direction to compute the slope map in.
            direction_angle (float): Angle of the direction in degrees.
        """
        log.info(f"Computing slope map in direction {direction}.")

        if direction == "omni":
            # Using steepest slope for omni-directional slope
            self.directional_rasters[direction_angle].data = self.slope_raster.data
        else:
            # Directional slope as given by: https://gis.stackexchange.com/a/169722
            self.directional_rasters[direction_angle].data = self.slope_raster.data * (
                -np.cos(np.deg2rad(direction_angle)) * np.cos(self.aspect_raster.data)
                - np.sin(np.deg2rad(direction_angle)) * np.sin(self.aspect_raster.data)
            )

        log.info(
            f"Slope max in direction {direction}: {colorama.Fore.BLUE}Mean {np.mean(self.directional_rasters[direction_angle].data):0.2g} deg{colorama.Fore.RESET}, Std. deviation: {np.std(self.directional_rasters[direction_angle].data):0.2g} deg."
        )

    def plot_slope_aspect(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the slope and aspect maps.

        Args:
            export_path (str): Path to export the plots to.
            show (bool): Whether to show the plots.
            export (bool): Whether to export the plots.
            save_maps (bool): Whether to save the slope and aspect maps.
            hide_utm (bool): Whether to hide the UTM coordinates on the plots.
        """
        # Plot steepest decline map
        self.slope_raster.plot(
            data=self.slope_raster.data * 100,
            colorbar="Slope [\%]",
            export_path=export_path / "slope_maps" / f"slope_steepest.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )

        # Plot aspect decline map
        self.aspect_raster.plot(
            data=np.rad2deg(self.aspect_raster.data),
            colorbar=r"Direction of steepest decline [\textdegree]",
            clim=(0, 360),
            export_path=export_path / "slope_maps" / "aspect.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )

    def plot(
        self,
        directions,
        export_path=None,
        show=True,
        export=False,
        save_maps=False,
        hide_utm=False,
    ):
        """
        Plots the slope maps for each direction.

        Args:
            directions (dict): Dictionary containing the directions and angles.
            export_path (str): Path to export the plots to.
            show (bool): Whether to show the plots.
            export (bool): Whether to export the plots.
            save_maps (bool): Whether to save the slope maps.
            hide_utm (bool): Whether to hide the UTM coordinates on the plots.
        """
        for direction, angle in zip(directions["names"], directions["angles"]):

            direction_name = direction.lower() if type(direction) == str else str(direction).replace('.', '_')

            self.directional_rasters[angle].plot(
                data=self.directional_rasters[angle].data * 100,
                colorbar="Slope [\%]",
                clim=(-20, 20),
                export_path=export_path / "slope_maps" / f"slope_map_{direction_name}.pdf",
                show=show,
                export=export,
                save_maps=save_maps,
                hide_utm=hide_utm,
            )


class ParameterMap(Map):
    """
    Parameter map.
    """

    pass


class LandUseMap(ParameterMap):
    """Land use map coded through an RGB raster mapped to ECDS classification
    codes. Used to delineate no-go zones and hard surfaces on the map."""

    # More information on land use codes at the following link:
    # https://www.sedris.org/sdk_4.1/src/lib/edcs/docs/dictionary/ECC_C.htm
    LANDUSE_CODES = {
        "AL020": {
            "edcs_classification_label": "BUILT_UP_REGION",
            "edcs_classification_symbolic_constant": "ECC_BUILT_UP_REGION",
            "edcs_classification_code": 177,
            "definition": "A <TRACT> containing a concentration of <BUILDING>s and other <STRUCTURE>s; a built-up region.",
            "edcs_group_labels": ["SHELTER"],
            "reference_type": "IR",
            "reference": "S57A, Object: BUAARE",
        },
        "EA010": {
            "edcs_classification_label": "CROP_LAND",
            "edcs_classification_symbolic_constant": "ECC_CROP_LAND",
            "edcs_classification_code": 295,
            "definition": "A <TRACT> that has been tilled for the planting of crops.",
            "edcs_group_labels": [
                "AGRICULTURE",
                "LIVING_ORGANISM",
                "PLANT",
                "USAGE_REGION",
            ],
            "reference_type": "IR",
            "reference": "FACC, Annex A: EA010",
        },
        "EC030": {
            "edcs_classification_label": "TREED_TRACT",
            "edcs_classification_symbolic_constant": "ECC_TREED_TRACT",
            "edcs_classification_code": 1282,
            "definition": "A tract covered by a non empty set of trees; a treed tract. examples: copse, thicket, stand of trees.",
            "edcs_group_labels": ["LIVING_ORGANISM", "PHYSIOGRAPHY", "PLANT"],
            "reference_type": "IR",
            "reference": "FACC, Annex A: EC030",
        },
    }

    def __init__(self, name, raster, value_map, nogo):
        """
        Initialises a land use map.

        Args:
            name (str): Name of the map.
            raster (RasterRGBA): RGBA land use raster.
            value_map (dict): Dictionary of {Hex color}: {EDCS code} mappings.
            nogo (dict): Dictionary of {EDCS code}: {True/False} mappings indicating nogo areas.
        """

        log.info(f"Initialising new land use map for site {name}.")

        self.name = name
        self.raster = raster
        self.value_map = value_map

        log.info(f"Provided value map: {value_map}")
        log.info(f"Provided nogo map: {nogo}")

        # Map colors from RGB integer values to corresponding map quantity (e.g. "255: "EC030")
        self.col_dict = {hex_to_rgb(k): v for k, v in self.value_map.items()}

        self.rgb = jnp.dstack(
            (self.raster.data[0], self.raster.data[1], self.raster.data[2])
        ).astype(jnp.uint8)

        # List of landuse RGB that is nogo
        self.rgb_nogo = np.zeros(256**3)

        # Convert the nogo landuse to a boolean array
        for rgb, landuse in self.col_dict.items():
            if nogo[landuse]:
                self.rgb_nogo[rgb] = 1
        self.rgb_nogo = jnp.array(self.rgb_nogo)

        # Sets the boolean value for each pixel in the raster
        @jit
        def is_nogo(index_y, index_x):
            rgb_value = self.rgb[index_y, index_x, :].astype(jnp.uint8)
            combined_rgb = (
                rgb_value[0].astype(jnp.int32) * 256 * 256
                + rgb_value[1].astype(jnp.int32) * 256
                + rgb_value[2].astype(jnp.int32)
            )
            return self.rgb_nogo[combined_rgb]

        self.vmap_is_nogo = vmap(is_nogo)  # Maps over the rows and columns

        index_y, index_x = jnp.unravel_index(
            np.arange(0, self.raster.data[0].size), self.raster.data[0].shape
        )
        self.nogo = self.vmap_is_nogo(index_y, index_x)
        self.nogo = np.array(self.nogo).reshape(self.raster.data[0].shape)

    def plot(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the land use map.

        Args:
            export_path (str): Path to export the plot to.
            show (bool): Whether to show the plot.
            export (bool): Whether to export the plot.
            save_maps (bool): Whether to save the land use map.
            hide_utm (bool): Whether to hide the UTM coordinates on the plot.
        """
        # Generate a colormap based on available colors
        discrete_colormap = ListedColormap([k for k, _ in self.value_map.items()])

        # Plots colormap
        self.raster.plot(
            rgb=self.rgb,
            discrete_colormap=(discrete_colormap, self.col_dict),
            export_path=export_path / "site/landuse_map.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )


class SoilTypeMap(ParameterMap):
    """Soil type map"""

    SOIL_LIST = {
        "GW": "Well-graded gravel, fine to coarse gravel",
        "GP": "Poorly graded gravel",
        "GM": "Silty gravel",
        "GC": "Clayey gravel",
        "SW": "Well-graded sand, fine to coarse sand",
        "SP": "Poorly graded sand",
        "SM": "Silty sand",
        "SC": "Clayed sand",
        "ML": "Silt",
        "CL": "Lean clay",
        "OL": "Organic silt, organic clay",
        "MH": "Elastic silt",
        "CH": "Fat clay",
        "OH": "Organic clay, organic silt",
        "PT": "Peat",
    }

    def __init__(self, name, raster, value_map):
        """
        Initialises a soil type map.

        Args:
            name (str): Name of the map.
            raster (Raster): RGBA soil type raster.
            value_map (dict): Dictionary of {Hex color}: {Soil type name} mappings.
        """
        log.info(f"Initialising new soil type map for site {name}")

        self.name = name
        self.raster = raster
        self.value_map = value_map

        log.info(f"Registered soil types:\n\t{value_map}")

        self.make_rgb_int_map()
        self.make_soil_array()

    def make_rgb_int_map(self):
        """
        Maps hexadecimal color codes to integer RGB values.
        """

        self.rgb_map = {}
        for key, value in self.value_map.items():
            self.rgb_map[hex_to_rgb(key)] = value

    def make_soil_array(self):
        """
        Converts the soil type raster from integers to soil labels.
        """
        log.info("Converting soil type raster from integers to soil labels.")

        self.soilarray = np.zeros(self.raster.data[0].shape, dtype=object)

        rgb = (
            self.raster.data[0].astype(np.int64) * 256 * 256
            + self.raster.data[1].astype(np.int64) * 256
            + self.raster.data[2].astype(np.int64)
        )

        for rgb_key, soil in self.rgb_map.items():
            index = np.where(rgb == rgb_key)
            self.soilarray[index] = soil

    def plot(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the soil type map.

        Args:
            export_path (str): Path to export the plot to.
            show (bool): Whether to show the plot.
            export (bool): Whether to export the plot.
            save_maps (bool): Whether to save the soil type map.
            hide_utm (bool): Whether to hide the UTM coordinates on the plot.
        """
        # Map colors from RGB integer values to corresponding map quantity (e.g. "255: "EC030")
        col_dict = {hex_to_rgb(k): v for k, v in self.value_map.items()}

        # Generate a colormap based on available colors
        discrete_colormap = ListedColormap([k for k, _ in self.value_map.items()])

        rgb = np.dstack((self.raster.data[0], self.raster.data[1], self.raster.data[2])).astype(np.uint8)

        # Plots colormap
        self.raster.plot(
            rgb=rgb,
            discrete_colormap=(discrete_colormap, col_dict),
            export_path=export_path / "site/soil_map.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )


class MoistureContentMap(ParameterMap):
    """Moisture content map"""

    def __init__(self, name, raster, value_map):
        """
        Initialises a moisture content map.

        Args:
            name (str): Name of the map.
            raster (Raster): RGBA moisture content raster.
            value_map (dict): Dictionary of {Hex color}: {Moisture content value} mappings.
        """

        log.info(f"Initialising new moisture content map for site {name}.")

        self.name = name
        self.raster = raster
        self.value_map = value_map
        self.make_rgb_int_map()
        self.make_moisture_array()

        self.raster.colormap = "RdYlBu"

        log.info(f"Registered moisture content values:\n\t{value_map}")

    def make_rgb_int_map(self):
        """
        Maps hexadecimal color codes to integer RGB values.
        """
        self.rgb_map = {}
        for key, value in self.value_map.items():
            self.rgb_map[hex_to_rgb(key)] = value

    def make_moisture_array(self):
        """
        Converts the moisture content raster from integers to moisture content labels.
        """
        self.moisturearray = np.zeros_like(self.raster.data[0])

        rgb = (
            self.raster.data[0] * 256 * 256
            + self.raster.data[1] * 256
            + self.raster.data[2]
        )

        for rgb_key, moisture in self.rgb_map.items():
            index = np.where(rgb == rgb_key)
            self.moisturearray[index] = moisture

        self.moisturearray = np.concatenate(
            (self.moisturearray[np.newaxis, ...], self.raster.data[4:]), axis=0
        )

    def plot(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the moisture content map.

        Args:
            export_path (str): Path to export the plot to.
            show (bool): Whether to show the plot.
            export (bool): Whether to export the plot.
            save_maps (bool): Whether to save the moisture content map.
            hide_utm (bool): Whether to hide the UTM coordinates on the plot.
        """
        # Plots colormap
        self.raster.plot(
            colorbar="Moisture content [\%]",
            data=self.moisturearray,
            reverse_colormap=False,
            export_path=export_path / "site/moisture_map.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )


class FrictionMap(ParameterMap):
    """Friction map"""

    def __init__(self, name, raster):
        """
        Initialises a friction map.

        Args:
            name (str): Name of the map.
            raster (Raster): Friction raster.
        """
        log.info(f"Initialising new friction map for site {name}.")

        self.name = name
        self.raster = raster

    def plot(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the friction map.

        Args:
            export_path (str): Path to export the plot to.
            show (bool): Whether to show the plot.
            export (bool): Whether to export the plot.
            save_maps (bool): Whether to save the friction map.
            hide_utm (bool): Whether to hide the UTM coordinates on the plot.
        """
        self.raster.plot(
            colorbar="Friction",
            export_path=export_path / "site/friction_map.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )


def convert_coordinate_to_pixel(data, raster, shape, coordinate_system="utm"):
    """
    Converts a coordinate to a pixel index in a raster.

    Args:
        data (Tuple[float, float]): The coordinate to convert.
        raster (Raster): The raster to convert the coordinate to.
        shape (Tuple[int, int]): The shape of the raster.
        coordinate_system (str): The coordinate system of the coordinate. Can be "utm" or "map". Defaults to "utm".

    Returns:
        Tuple[int, int]: The pixel index in the raster.
    """
    x, y = data

    if coordinate_system == "utm" and (
        x < raster.x_min or x > raster.x_max or y < raster.y_min or y > raster.y_max
    ):
        raise ValueError(
            f"UTM coordinate ({x}, {y}) is outside the map bounds ([{round(raster.x_min)}, {round(raster.x_max)}], [{round(raster.y_min)}, {round(raster.y_max)}])"
        )
    if coordinate_system == "map" and (
        x < 0 or x > raster.x_max or y < 0 or y > raster.y_max
    ):
        raise ValueError(
            f"Map coordinate ({x}, {y}) is outside the map bounds ([{0}, {round(raster.x_max - raster.x_min)}], [{0}, {round(raster.y_max - raster.y_min)}])"
        )

    new_x = (
        x - (raster.x_min if coordinate_system == "utm" else 0)
    ) / raster.resolution_x
    new_y = (
        y - (raster.y_min if coordinate_system == "utm" else 0)
    ) / raster.resolution_y

    return np.floor(new_x).astype(int), shape[0] - np.floor(new_y).astype(int) - 1


def convert_coordinate_length_to_pixel_length(l: float, raster) -> float:
    """
    Converts a coordinate length to a pixel length in a raster.

    Args:
        l (float): The coordinate length.
        raster (Raster): The raster to convert the coordinate length to.

    Returns:
        float: The pixel length in the raster.
    """

    return l / raster.resolution_x


class NoGoMap(Map):
    """No-go map"""

    def __init__(self, name, vehicle, aerial, landuse):
        """
        Initialises a no-go zones map.

        Args:
            name (str): Name of the map.
            vehicle (str): Type of vehicle to consider.
            aerial (Raster): Aerial raster.
            landuse (str): Landuse type.
        """

        log.info(f"Initialising no-go zones map site {name}, vehicle {vehicle}.")

        self.vehicle = vehicle
        self.aerial = aerial
        self.data = np.zeros(
            aerial.raster.data[0].shape
        )  # We are using the aerial red channel to deduct shape
        self.landuse = landuse
        self.zones = []

    def add_zone(self, data, description=None):
        """
        Adds a zone to the no-go zones map.

        Args:
            data (np.ndarray): A binary array where True indicates a no-go zone.
            description (str): A description of the zone. Defaults to None.
        """

        def add_box(data, description):
            """
            Adds a bounding box zone to a no-go zones map.

            Args:
                data (Tuple[float, float, float, float]): A tuple containing the bounding box coordinates (x_min, x_max, y_min, y_max).
                description (str): A description of the zone.
            """
            x_min, x_max, y_min, y_max = data
            log.info(
                f'Adding zone "{description}" with bounding box [{x_min}, {x_max}, {y_min}, {y_max}]'
            )
            x_pixel_min, y_pixel_min = convert_coordinate_to_pixel(
                (x_min, y_min), self.landuse.raster, self.data.shape
            )
            x_pixel_max, y_pixel_max = convert_coordinate_to_pixel(
                (x_max, y_max), self.landuse.raster, self.data.shape
            )
            self.data[y_pixel_max:y_pixel_min, x_pixel_min:x_pixel_max] = 1

        def add_circle(data, description):
            """
            Adds a circular zone to a no-go zones map.

            Args:
                data (Tuple[float, float, float]): A tuple containing the circle coordinates (x, y, r).
                description (str): A description of the zone.
            """

            x, y, r = data
            log.info(
                f'Adding zone "{description}" with bounding circle centrum [{x}, {y}] and radius {r}'
            )
            x_pixel, y_pixel = convert_coordinate_to_pixel(
                (x, y), self.landuse.raster, self.data.shape
            )
            r_pixel = convert_coordinate_length_to_pixel_length(r, self.landuse.raster)
            index_y, index_x = np.unravel_index(
                np.arange(0, self.data.size), self.data.shape
            )
            nogo_index_x = index_x[
                (index_x - x_pixel) ** 2 + (index_y - y_pixel) ** 2 <= r_pixel**2
            ]
            nogo_index_y = index_y[
                (index_x - x_pixel) ** 2 + (index_y - y_pixel) ** 2 <= r_pixel**2
            ]
            self.data[nogo_index_y, nogo_index_x] = 1

        # If no description is provided, use zone number
        if description is None:
            description = f"Zone {len(self.zones) + 1}"

        self.zones.append({})
        self.zones[-1]["description"] = description
        self.zones[-1]["data"] = data

        # Store zone type
        if len(data) == 4:
            self.zones[-1]["type"] = "Box"
            add_box(data, description)
        elif len(data) == 3:
            self.zones[-1]["type"] = "Circle"
            add_circle(data, description)
        else:
            log.info(
                f'Unsupported zone type for "{description}" with bounding data [{data}]'
            )

        log.info(f"Total number of zones updated to: {len(self.zones)}")

    def plot(
        self, export_path=None, show=True, export=False, save_maps=False, hide_utm=False
    ):
        """
        Plots the no-go zones map.

        Args:
            export_path (str): Path to export the plot to.
            show (bool): Whether to display the plot. Defaults to True.
            export (bool): Whether to export the plot. Defaults to False.
            save_maps (bool): Whether to save the map in the current working directory. Defaults to False.
            hide_utm (bool): Whether to hide the UTM grid lines. Defaults to False.
        """
        # Open new figure
        (figure, axes) = plt.subplots(1, 1)

        rgb = np.dstack(
            (
                self.aerial.raster.data[0],
                self.aerial.raster.data[1],
                self.aerial.raster.data[2],
            )
        )
        # Set nodata values
        rgb[rgb == 0] = 255

        (_, axes, _, _) = self.aerial.raster.plot(
            rgb=rgb, show=False, export=False, save_maps=False
        )

        # Set style for the no-go zone labels
        label_style = dict(boxstyle="round", facecolor="indianred", alpha=0.9)
        font_style = dict(fontsize="x-small", color="white")

        # Create custom colormap
        colormap1 = mpl.colors.ListedColormap(["none", "darkorange"])
        colormap2 = mpl.colors.ListedColormap(["none", "crimson"])

        # Add no-go from land use
        extent = [
            self.landuse.raster.x_min,
            self.landuse.raster.x_max,
            self.landuse.raster.y_min,
            self.landuse.raster.y_max,
        ]
        axes.imshow(
            self.landuse.nogo, vmin=0, vmax=1, extent=extent, cmap=colormap1, alpha=0.7
        )

        # Plot no-go zones
        axes.imshow(self.data, vmin=0, vmax=1, extent=extent, cmap=colormap2, alpha=0.7)

        # Add no-go zone labels
        for entry in self.zones:
            # Get no-go zone label position
            data = entry["data"]

            # Get label position
            offset = 15
            if entry["type"] == "Box":
                x_pixel_min, x_pixel_max, y_pixel_min, y_pixel_max = data
                x = (x_pixel_max + x_pixel_min) / 2

                txt_x = x
                txt_y = y_pixel_min - offset

            elif entry["type"] == "Circle":
                x_pixel, y_pixel, r_pixel = data

                txt_x = x_pixel
                txt_y = y_pixel - r_pixel - offset

            axes.text(
                txt_x,
                txt_y,
                entry["description"],
                bbox=label_style,
                fontdict=font_style,
                ha="center",
                va="top",
            )

        if show:
            plt.show()

        if export:
            log.info(f"Exporting raster plot to {export_path}.")
            plt.savefig(
                export_path / f"mobility_{self.vehicle}" / "nogo_maps/nogo_map.pdf",
                format="pdf",
                bbox_inches="tight",
            )

        # Saves the raster as a GeoTIFF file if save_maps is True
        if save_maps:
            log.info(f"Exporting no go as GeoTiff is not yet supported.")
            # TODO: add possibility to save the no-go map as a GeoTIFF file

        plt.close()


class MapEditor:
    """Map editor"""

    def __init__(self):
        """
        Initialises the map editor.
        """

        pass

    @staticmethod
    def convert_geohornet(raster_rgba, value_null=0):
        """
        GeoHornet 32-bit RGBA raster to 8-bit single-channel raster

        Convert the 32-bit RGBA elevation raster from the GeoHornet server source to an 8-bit single-channel raster suitable for processing in the MOMA.

        Args:
            raster_rgba (rasterio.raster): 32-bit RGBA elevation raster
            value_null (int, optional): Value to assign to cells without data. Defaults to 0.

        Returns:
            numpy.array: 8-bit single-channel elevation raster
        """

        # Assemble elevation information
        # - Alpha channel (0: no data, 255: data available)
        # - Red channel (bits 17 to 24)
        # - Green channel (bits 9 to 16)
        # - Blue channel (bits 1 to 8)
        h_alpha = raster_rgba[3].astype("float")
        h_red = raster_rgba[0].astype("float") * 255 * 255
        h_green = raster_rgba[1].astype("float") * 255
        h_blue = raster_rgba[2].astype("float")
        h_final = h_red + h_green + h_blue

        # Return elevation value only where available
        h_mono = np.where(h_alpha == 255, h_final, value_null)

        # Convert the elevation values from mm to m
        h_mono = h_mono * 1e-3

        return h_mono


def hex_to_rgb(hex_color):
    """Convert hexadecimal color to RGB integer

    Args:
        hex_color (string): Hexadecimal color value, including "#"

    Returns:
        int: RGB integer
    """

    # Convert hex value to RGB triplet
    rgb_triplet = mpl.colors.hex2color(hex_color)

    # Map RGB triplet from [0, 1] to [0, 255]
    rgb_triplet = tuple([int(255 * x) for x in rgb_triplet])

    # Convert RGB triplet to an integer
    rgb_integer = rgb_triplet[0] * 256 * 256 + rgb_triplet[1] * 256 + rgb_triplet[2]

    return rgb_integer
