"""
Mobility module
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
import logging
import os
import pathlib

from abc import ABC

import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate as spi
import yaml
import rasterio as rio
import tables

from jax import jit, vmap
from jax.scipy.interpolate import RegularGridInterpolator
from matplotlib import cm
from matplotlib.colors import Normalize

from moma.mapping import DirectionalMap

log = logging.getLogger("MOMA")

plt.style.use(
    importlib.resources.path("moma", "data") / pathlib.Path("matplotlib/moma.mplstyle")
)

def generate_vehicle_mobility_file(vehicle, minimum_speed=0.01, plot_mobility=False):
    """
    Generates a mobility file for a given vehicle.

    Args:
        vehicle (str): Name of the vehicle.
        minimum_speed (float): Minimum speed for valid runs.
        plot_mobility (bool): Whether to plot the mobility data.

    Returns:
        None
    """

    print(f"Generating vehicle mobility file for {vehicle}")
    print("\nLoading data\n")

    vehicle_dir = pathlib.Path("data") / "vehicles" / f"{vehicle}"

    data = np.zeros((4, 0))

    for test_run in os.listdir(vehicle_dir / "straight_line_acceleration"):
        if test_run[0:3] == "run":
            run_number = int(test_run[4:])
            if run_number % 50 == 0:
                print(f"Run number: {run_number:5}")

            run_dir = vehicle_dir / "straight_line_acceleration" / test_run

            with open(run_dir / "case.json", "r") as json_file:
                json_file = json.load(json_file)

                # Only load and save data for runs with side slope = 0
                sideslope_percentage = json_file["scenario"]["sideSlopePercentage"]
                if sideslope_percentage == 0:

                    friction_coefficient = json_file["scenario"]["terrain"]["rigid"][
                        "frictionCoefficient"
                    ]
                    slope_percentage = json_file["scenario"]["slopePercentage"]

                    with tables.open_file(
                        run_dir / "case.h5", driver="H5FD_CORE"
                    ) as h5file:
                        rotation = h5file.root.data.pose.rotation[:]
                        velocity = h5file.root.data.velocity.linear[:]

                    stable_z_rotation = rotation[-1000:, 2]
                    longitudinal_velocity = velocity[:, 0]

                    # If speed is below the minimum speed or z rotation after the initial setup is not pointing forward the run is failed
                    if (
                        longitudinal_velocity[-1] > minimum_speed
                        and np.max(np.abs(stable_z_rotation)) < np.pi / 4
                    ):
                        # Takes the mean of the last values of the linear velocity
                        longitudinal_velocity = np.mean(longitudinal_velocity[-10:])
                    else:
                        # If failed run, set to 0
                        longitudinal_velocity = 0.0

                    data = np.append(
                        data,
                        np.array(
                            [
                                [slope_percentage],
                                [friction_coefficient],
                                [longitudinal_velocity],
                                [run_number],
                            ]
                        ),
                        axis=1,
                    )

    data[0, :] = slope_percentage_to_angle(data[0, :])

    slope_percentage = np.unique(data[0, :])
    friction_coefficient = np.unique(data[1, :])

    for idx, val in enumerate(slope_percentage):
        data[0, data[0, :] == val] = idx
    for idx, val in enumerate(friction_coefficient):
        data[1, data[1, :] == val] = idx

    # The data is put into a grid to make sure that missing grid points are set to zero
    data_grid = np.zeros((len(slope_percentage), len(friction_coefficient), 2))
    data_grid[data[0, :].astype(int), data[1, :].astype(int)] = data[2:, :].T

    print("\nWriting mobility file\n")

    # Overwrite the mobility.yaml file with the new values
    with open(vehicle_dir / "mobility.yaml", "w") as f:

        for i, slope in enumerate(slope_percentage):
            for j, friction in enumerate(friction_coefficient):
                speed = data_grid[i, j, 0]
                run_number = int(data_grid[i, j, 1])

                if run_number % 50 == 0:
                    print(f"Run number: {run_number:5}")

                f.write(
                    f"""
- soil:
    type: HT
    friction_coefficient: {friction}
  test:
    run_number: {run_number}
    type: slope
    slope_angle: {slope}
  value: {speed}"""
                )

    if plot_mobility:
        print(f"Plotting the data")
        # Plot the mobility data as a 3D surface
        fig = plt.figure()

        fig = plt.figure(facecolor=fig.get_facecolor())
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(fig.get_facecolor())

        ax.set_xlabel("Slope Angle (degrees)")
        ax.set_ylabel("Friction Coefficient")
        ax.set_zlabel("Speed [m/s]")

        color_val = np.sqrt(data_grid[:, :, 0].T)
        norm = Normalize(vmin=color_val.min(), vmax=color_val.max())
        colors = cm.viridis(norm(color_val))

        # Plots the data as a surface
        X, Y = np.meshgrid(slope_percentage, friction_coefficient)
        ax.plot_surface(
            X,
            Y,
            data_grid[:, :, 0].T,
            facecolors=colors,
            cmap="viridis",
            edgecolor="none",
        )

        plt.savefig(
            vehicle_dir / f"Vehicle mobility vs slope and friction.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.5,
        )
        plt.close(fig)


def slope_percentage_to_angle(slope_percentage):
    """
    Converts slope percentage to slope angle in degrees.

    Args:
        slope_percentage (np.ndarray): Slope percentage values.

    Returns:
        np.ndarray: Slope angle values in degrees.
    """

    return np.arctan(slope_percentage / 100) * 180 / np.pi

class ReductionFunction:
    """
    Class that returns the wanted sample reduction functions.

    __call__ methods must be jit-able.
    """

    def __init__(self, parameters):
        """
        Initializes the SampleReduction object with a specified function.

        Args:
            parameters (List[str]): List of function parameters.
        """

        function_name = parameters[0]
        function_parameters = parameters[1:]
        if not hasattr(self, function_name):
            raise ValueError(f"Sample reduction function {function_name} not supported")
        self.function = getattr(self, function_name)(*function_parameters)

    class mean:
        """
        Mean reduction function.
        """

        def __init__(self):
            """
            Initializes the mean reduction function.
            """

            self.name = "mean"

        def __call__(self, values):
            """
            Calculates the mean of the input values.

            Args:
                values (jnp.ndarray): Input values.

            Returns:
                jnp.ndarray: Mean of the input values.
            """
            return jnp.mean(values)

    class quantile:
        """
        Quantile reduction function.
        """

        def __init__(self, q_val):
            """
            Initializes the quantile reduction function.

            Args:
                q_val (float): Quantile value.
            """

            q_val = float(q_val)
            self.q_val = jnp.array(
                [1 - q_val]
            )  # Flips the quantile since the highest quantile part is wanted
            self.name = f"${100*q_val:.0f}\%$ confident quantile"

        def __call__(self, values):
            """
            Calculates the quantile of the input values.

            Args:
                values (jnp.ndarray): Input values.

            Returns:
                jnp.ndarray: Quantile of the input values.
            """

            return jnp.quantile(values, self.q_val)[0]


class MobilityAnalysis:
    """Mobility analysis object class.

    A mobility analysis object encapsulates a vehicle and its corresponding
    simulation results database subset, an arbitrary number of soil types and an
    arbitrary number of soil parameters for the soil types set. On the basis of
    the provided data, it allows for the generation of mobility tensors, one for
    each soil type, allowing for interpolation of simulated data over the
    n-dimensional hyperplane defined by the chosen parameters.
    """

    # List of parameters for each mobility test
    TESTS_PARAMETERS_LIST = {"slope": ["slope_angle"]}

    def __init__(
        self,
        database_path,
        vehicle,
        soil_types,
        soil_parameters,
        sample_reduction,
        minimum_speed,
        target_speed,
        clip_mobility_to_range=False,
    ):
        """
        Initializes the mobility analysis object.

        Args:
            database_path (str): Path to the simulation results database.
            vehicle (str): Name of the vehicle.
            soil_types (List[str]): List of soil types.
            soil_parameters (List[str]): List of soil parameters.
            sample_reduction (ReductionFunction): Sample reduction function.
            minimum_speed (float): Minimum speed for valid runs.
            target_speed (float): Target speed for mobility interpolation.
            clip_mobility_to_range (bool): Clip mobility to the tabulated range
                instead of extrapolating outside it.
        """

        log.info(
            f"""Starting new mobility analysis for:
            \tVehicle: {vehicle}
            \tSoil parameters: {soil_parameters}
            \tSoil types: {soil_types}"""
        )

        # Initialise vehicle simulation results database subset
        self.database = self.load_vehicle_metrics(database_path)

        # Initialise soil types list
        self.soil_types = soil_types

        # Initialise soil parameters list
        self.soil_parameters = soil_parameters

        # Pre-allocate mobility tensors list ordered by soil type
        self.mobility_tensors = []
        self.sideslope_factors = None

        self.vehicle = vehicle

        self.minimum_speed = minimum_speed
        self.target_speed = target_speed
        self.clip_mobility_to_range = clip_mobility_to_range

        # Sample reduction function to reduce sample space
        self.sample_reduction = ReductionFunction(sample_reduction)

    def load_vehicle_metrics(self, database_path):
        """
        Loads simulation results database subset for a given vehicle.

        Args:
            database_path (str): Path to the simulation results database.

        Returns:
            dict: Simulation results database subset.
        """

        # Load mobility database
        with open(database_path) as database:
            mobility_database = yaml.load(database, Loader=yaml.SafeLoader)

        # Return simulation results database for the vehicle
        return mobility_database

    def generate_mobility_tensors(self, test):
        """
        Generates mobility tensors for each soil type.

        Args:
            test (str): Name of the mobility test.
        """

        for soil_type in self.soil_types:

            valid_entries = []
            soil_parameter_names = []
            test_parameter_names = []

            # Get all entries for the given soil type
            for entry in self.database:

                if entry["soil"]["type"] == soil_type:
                    soil_parameters = []
                    test_parameters = []

                    # Collects soil parameters
                    for soil_parameter in self.soil_parameters:
                        if soil_parameter in entry["soil"]:
                            soil_parameters.append(entry["soil"][soil_parameter])
                            if soil_parameter not in soil_parameter_names:
                                soil_parameter_names.append(soil_parameter)

                    # Collects test parameters
                    for test_parameter in self.TESTS_PARAMETERS_LIST[test]:
                        if test_parameter in entry["test"]:
                            test_parameters.append(entry["test"][test_parameter])
                            if test_parameter not in test_parameter_names:
                                test_parameter_names.append(test_parameter)

                    max_speed = entry["value"]
                    database_entry = soil_parameters + test_parameters + [max_speed]

                    valid_entries.append(database_entry)

            if len(valid_entries) > 0:
                valid_params_list = list(zip(*valid_entries))
                parameters = valid_params_list[0:-1]
                values = valid_params_list[-1]

                # Find unique entries
                parameters = [np.unique(parameter) for parameter in parameters]

                values = (
                    np.array(values)
                    .reshape(
                        np.flip(np.array([len(parameter) for parameter in parameters]))
                    )
                    .T
                )

                log.info(
                    f"""Initialising new mobility tensor for soil: {soil_type} with settings:
                    \tSoil parameters: {soil_parameter_names} 
                    \tTest parameters: {test_parameter_names}"""
                )

                self.mobility_tensors.append(
                    MobilityTensor(
                        soil_type,
                        parameters,
                        soil_parameter_names,
                        test_parameter_names,
                        values,
                        self.vehicle,
                        clip_mobility_to_range=self.clip_mobility_to_range,
                    )
                )
            else:
                log.info(
                    f"No valid entries found for soil type: {soil_type} and test: {test}. This terrain is set to 0 mobility"
                )
                self.mobility_tensors.append(
                    lambda *args: 0.0  # Mobility tensor for no valid entries, returns zero speed
                )

    def generate_sideslope_factor(self):
        """
        Generates sideslope factor for interpolation of sideslope data.
        """

        sideslopes = []
        max_sideslope_speed = []

        for entry in self.database:
            if entry["soil"]["type"] == "sideslope":
                sideslopes.append(entry["test"]["slope_angle"])
                max_sideslope_speed.append(entry["value"])

        # Uses factor one if no sideslope data is available
        if len(sideslopes) == 0:
            self.sideslope_factors = lambda sideslope, **_: jnp.ones_like(sideslope)
            log.info(
                "No sideslope data available, sideslope factor set to 1 everywhere."
            )
            return

        self.sideslope_factors = SideslopeFactor(
            sideslopes=sideslopes, speeds=max_sideslope_speed
        )

    def create_evaluate_functions(self):
        """
        Creates functions to evaluate mobility tensors.
        """

        @jit
        def evaluate_mobility_tensor(soil, moisture, friction, slope, sideslope):
            """
            Evaluates mobility tensor for a given soil type and parameters.

            Args:
                soil (int): Index of the soil type.
                moisture (float): Moisture value.
                friction (float): Friction value.
                slope (float): Slope value.
                sideslope (float): Sideslope value.

            Returns:
                float: Steady-state speed.
            """

            params = jnp.array([moisture, friction, slope])
            steady_state_speed = jnp.minimum(
                jax.lax.switch(soil, self.mobility_tensors, params), self.target_speed
            )
            sideslope_factor = self.sideslope_factors(
                sideslope,
                minimum_speed=self.minimum_speed,
                target_speed=self.target_speed,
            )
            return steady_state_speed * sideslope_factor

        @jit
        def evaluate_point_in_space(soil, moisture, friction, slope, sideslope):
            """
            Evaluates point in space for mobility evaluation.

            Args:
                soil (int): Index of the soil type.
                moisture (float): Moisture value.
                friction (float): Friction value.
                slope (float): Slope value.
                sideslope (float): Sideslope value.

            Returns:
                float: Reduced speed based on sample reduction function.
            """

            # TODO: Handle high negative slopes as no-go zones.
            # Take care of negative slopes and filter out for small slopes
            low_slope_mask = slope < 1
            slope_masked = jnp.where(low_slope_mask, 0.0, slope)

            # Get corresponding tensor speed
            speeds = self.vmap_evaluate_mobility_tensor(
                soil, moisture, friction, slope_masked, sideslope
            )

            # Return reduction in samples based on the sample reduction function
            return self.sample_reduction.function(speeds)

        @jit
        def vmap_evaluate_point_in_space(soil, moisture, friction, slope, sideslope):
            """
            Vmaps evaluation of point in space for mobility evaluation.

            Args:
                soil (int): Index of the soil type.
                moisture (float): Moisture value.
                friction (float): Friction value.
                slope (float): Slope value.
                sideslope (float): Sideslope value.

            Returns:
                ndarray: Reduced speeds based on sample reduction function.
            """

            self.vmap_evaluate_mobility_tensor = vmap(
                evaluate_mobility_tensor,
                in_axes=(
                    0 if soil.ndim == 2 else None,
                    0 if moisture.ndim == 2 else None,
                    0 if friction.ndim == 2 else None,
                    0 if slope.ndim == 2 else None,
                    0 if sideslope.ndim == 2 else None,
                ),
            )  # Maps over the LHS samples for a given point in space, only map over parameters if they are not scalars

            return vmap(evaluate_point_in_space, in_axes=(-1, -1, -1, -1, -1))(
                soil, moisture, friction, slope, sideslope
            )  # Maps over the points in space

        self.vmap_evaluate_point_in_space = vmap_evaluate_point_in_space


class MobilityTensor:
    """Mobility tensor object class.

    A mobility tensor object is an n-dimensional interpolator defined for a
    given soil type and an arbitrarily long set of parameter for the given soil.
    When called with an n-tuple of parameters as argument, it returns the
    interpolator evaluation for the given soil parameters coordinate.
    """

    def __init__(
        self,
        soil_type,
        parameters,
        soil_parameter_name,
        test_parameter_name,
        values,
        vehicle,
        clip_mobility_to_range=False,
    ):
        """
        Initialises a mobility tensor.

        Args:
            soil_type (str): Soil type.
            parameters (ndarray): Grid points.
            soil_parameter_name (list[str]): Soil parameters.
            test_parameter_name (list[str]): Test parameters.
            values (ndarray): Values at grid points.
            vehicle (str): Vehicle type.
            clip_mobility_to_range (bool): If True, return zero speed for
                slopes outside the tabulated range instead of extrapolating.
        """

        self.soil_type = soil_type
        self.parameters = parameters
        self.vehicle = vehicle
        self.clip_mobility_to_range = clip_mobility_to_range

        all_parameter_names = [
            "moisture_content",
            "friction_coefficient",
            "slope_angle",
        ]
        used_parameter_names = soil_parameter_name + test_parameter_name
        self.parameter_mask = jnp.array(
            [
                True if parameter in used_parameter_names else False
                for parameter in all_parameter_names
            ]
        )

        log.info(
            f"Initialising new RegularGridInterpolator for soil {soil_type}, "
            f"{'clipping to the tabulated range' if clip_mobility_to_range else 'using extrapolation for points outside the grid'}"
        )

        # TODO: Standardize the structure of the parameters and values in the database.

        # Initialise linear radial-basis function interpolator
        self.interpolator = RegularGridInterpolator(
            self.parameters,
            values,
            method="linear",
            fill_value=0.0 if clip_mobility_to_range else None,
        )

    def __call__(self, call):
        """
        Interpolates values at given points.

        Args:
            call (ndarray): Points to interpolate.

        Returns:
            float: Interpolated value.
        """

        return self.interpolator(
            call[self.parameter_mask].reshape(1, len(self.parameters))
        )[0]


class SideslopeFactor:
    """Class to calculate the penalty factor due to sideslope"""

    def __init__(self, sideslopes, speeds):
        """Initialise the sideslope factor with the sideslopes and speeds

        Args:
            sideslopes (list): List of sideslopes in degrees.
            speeds (list): List of maximum speeds for the corresponding sideslopes.
        """
        sideslopes = jnp.array(sideslopes)
        speeds = jnp.array(speeds)

        # Initialise the interpolator
        self.interpolator = RegularGridInterpolator(
            (sideslopes,), speeds, method="linear", fill_value=None
        )

    def __call__(self, sideslope, target_speed, minimum_speed, power=5):
        """Calculate the sideslope factor for a given sideslope
        sideslope_speed <= minimum_speed: 0
        sideslope_speed >= target_speed: 1
        minimum_speed < sideslope_speed < target_speed: [(v_sideslope - v_min)/(v_target - v_min)]^(1/power)
        """
        sideslope_speeds = self.interpolator(sideslope.reshape(1, 1))

        sideslope_factor = jnp.zeros_like(sideslope)
        sideslope_factor = jnp.where(sideslope_speeds >= target_speed, 1, 0)
        sideslope_factor = jnp.where(
            (sideslope_speeds > minimum_speed) & (sideslope_speeds < target_speed),
            jnp.pow(
                ((sideslope_speeds - minimum_speed) / (target_speed - minimum_speed)),
                1 / power,
            ),
            sideslope_factor,
        )

        return sideslope_factor


class Map(ABC):
    """
    Abstract base class for maps.

    Provides a common interface for different map types.
    """

    def overlay(self, axis, map):
        """
        Overlays a map on the given axis.

        Args:
            axis (matplotlib.axes.Axes): Axis to overlay the map on.
            map (ndarray): Map to overlay.
        """
        cmap2 = matplotlib.colors.ListedColormap(["none", "red"])

        axis.imshow(map, alpha=1.0, cmap=cmap2)


class DirectionalMobilityMap(DirectionalMap):
    """
    Directional mobility map.

    Extends the `DirectionalMap` class to create a map of mobility values for different soil types.
    """

    def __init__(self, mobility_maps):
        """
        Initialises a directional mobility map container.

        Args:
            mobility_maps (list[MobilityMap]): List of mobility maps.
        """

        log.info(f"Initialising directional mobility map container.")

        self.mobility_maps = mobility_maps

    def show_overview(self, directions="N"):
        """
        Displays an overview of the directional mobility map.

        Args:
            directions (str): Directions to show.

        Returns:
            list[matplotlib.axes.Axes]: List of axes with the overview.
        """

        axes = []

        _, axis = plt.subplots(len(directions))

        for direction in directions:

            axis.title(f"Directional mobility map\n{direction} direction")
            axis.set_xlabel("X coordinate [m]")
            axis.set_ylabel("Y coordinate [m]")

            axes.append(axis)

        return axes

    def show_separate(self, directions="N"):
        """
        Displays separate overview of the directional mobility map.

        Args:
            directions (str): Directions to show.

        Returns:
            list[matplotlib.axes.Axes]: List of axes with the separate overview.
        """
        axes = []

        for _ in directions:
            _, axis = plt.subplots(1, 1)
            axis.title(f"Directional mobility map\n{direction} direction")
            axis.set_xlabel("X coordinate [m]")
            axis.set_ylabel("Y coordinate [m]")

            axes.append(axis)

        return axes

    def fill_path_with_mobility_data(
        self,
        path_json=None,
        nodes=None,
        DiGraph=None,
        elevation_raster=None,
        slope_maps=None,
        minimum_speed=None,
        target_speed=None,
        fill_omnidirectional_info=False,
    ):
        """
        Fills in mobility metrics for a given path.

        Args:
            path_json (Optional[str]): Path to the json file containing the path coordinates.
            nodes (Optional[list[Node]]): List of nodes representing the path.
            DiGraph (Optional[nx.DiGraph]): Directed graph of the map.
            elevation_raster (Optional[Raster]): Raster of elevation data.
            slope_maps (Optional[list[Map]]): List of slope maps.
            minimum_speed (Optional[float]): Minimum speed for mobility calculations.
            target_speed (Optional[float]): Target speed for mobility calculations.
            fill_omnidirectional_info (bool): Whether to fill in omnidirectional mobility information.
        """

        def get_direction(delta_map_coor):
            """
            Calculates the direction of movement between two points.

            Args:
                delta_map_coor (Tuple[float, float]): Difference in map coordinates between the two points.

            Returns:
                angle (float): Angle in radians between the positive x-axis and the vector from the first point to the second point.
                direction (str): Direction of the movement in compass notation.
                compass_direction (int): Direction of the movement in degrees, where 0 is north, 90 is east, 180 is south and 270 is west.
            """
            angle = np.arctan2(delta_map_coor[1], delta_map_coor[0])

            if angle > np.pi * 7 / 8:
                return angle, "W", 270
            elif angle > np.pi * 5 / 8:
                return angle, "NW", 315
            elif angle > np.pi * 3 / 8:
                return angle, "N", 0
            elif angle > np.pi * 1 / 8:
                return angle, "NE", 45
            elif angle > -np.pi * 1 / 8:
                return angle, "E", 90
            elif angle > -np.pi * 3 / 8:
                return angle, "SE", 135
            elif angle > -np.pi * 5 / 8:
                return angle, "S", 180
            else:
                return angle, "SW", 225

        def info_from_node_num(node):
            """
            Returns the map coordinates and indices of a node in the raster.

            Args:
                node (int): Node number in the path.

            Returns:
                indices (np.ndarray): Indices of the node in the raster.
                UTM_coor (Tuple[float, float]): UTM coordinates of the node.
                map_coor (Tuple[float, float]): Map coordinates of the node.
            """

            indices = np.unravel_index(node, elevation_raster.data.shape[-2:])
            UTM_coor = rio.transform.xy(
                elevation_raster.transform, indices[0], indices[1], offset="center"
            )
            map_coor = (
                UTM_coor[0] - elevation_raster.x_min,
                UTM_coor[1] - elevation_raster.y_min,
            )
            return indices, UTM_coor, map_coor

        def info_from_map_coor(map_coor):
            """
            Returns the map coordinates and indices of a node in the raster.

            Args:
                map_coor (dict): Map coordinates of the node.

            Returns:
                indices (np.ndarray): Indices of the node in the raster.
                UTM_coor (Tuple[float, float]): UTM coordinates of the node.
                map_coor (Tuple[float, float]): Map coordinates of the node.
            """
            raster = list(self.mobility_maps.values())[0].raster
            inv_transform = ~raster.transform

            map_coor = (map_coor["x"], map_coor["y"])
            UTM_coor = map_coor[0] + raster.x_min, map_coor[1] + raster.y_min
            indices = inv_transform * UTM_coor
            indices = (int(indices[1] - 0.5), int(indices[0] - 0.5))

            return indices, UTM_coor, map_coor

        if path_json is None:
            # If the path is from the MOMA path planner the path is created from scratch
            path_json = {"waypoints": []}

        path_json["metadata"] = {
            "minimum_speed": minimum_speed,
            "target_speed": target_speed,
        }

        # Retrieves first point on path
        if nodes is None:
            num_points = len(path_json["waypoints"])
            map_coor = path_json["waypoints"][0]["map"]
            indices, UTM_coor, map_coor = info_from_map_coor(map_coor)
        else:
            num_points = len(nodes)
            node = nodes[0]
            indices, UTM_coor, map_coor = info_from_node_num(node)

        # Setting initial values
        total_length = 0.0
        total_time = 0.0
        if fill_omnidirectional_info:
            omni_total_time = 0.0

        for idx in range(num_points - 1):
            # Retrieves next point on path
            if nodes is None:
                map_coor_next = path_json["waypoints"][idx + 1]["map"]
                indices_next, UTM_coor_next, map_coor_next = info_from_map_coor(
                    map_coor_next
                )
                pathfinder_used_angle = None
            else:
                node_next = nodes[idx + 1]
                indices_next, UTM_coor_next, map_coor_next = info_from_node_num(
                    node_next
                )
                pathfinder_used_angle = DiGraph.get_edge_data(node, node_next)["angle"]

            # Calculate the segment length and transverse time based on the cell speed and distance to the next cell
            delta_map_coor = np.subtract(map_coor_next, map_coor)
            segment_length = np.sqrt(np.sum(delta_map_coor**2))
            total_length += segment_length

            direction_angle, direction, compass_direction = get_direction(
                delta_map_coor
            )

            speed = self.mobility_maps[compass_direction].raster.data[indices]
            transverse_time = (
                segment_length / speed if speed != 0 else segment_length / 0.001
            )
            total_time += transverse_time
            if fill_omnidirectional_info:
                omni_speed = self.mobility_maps["omni"].raster.data[indices]
                omni_transverse_time = (
                    segment_length / omni_speed
                    if omni_speed != 0
                    else segment_length / 0.001
                )
                omni_total_time += omni_transverse_time

            if nodes is not None:
                # If the path is from the MOMA path planner the waypoint has to be added to the list with information about the points location
                path_json["waypoints"].append(
                    {
                        "node": int(node),
                        "map": {
                            "x": float(map_coor[0]),
                            "y": float(map_coor[1]),
                        },
                        "elevation": float(
                            np.mean(elevation_raster.data[:, indices[0], indices[1]])
                        ),
                    }
                )

                node = node_next

            # Fills in the data needed for both path types

            path_json["waypoints"][idx]["path"] = {
                "move_direction": float(direction_angle),
                "move_compass_direction": direction,
                "pathfinder_used_angle": (
                    float(pathfinder_used_angle)
                    if pathfinder_used_angle is not None
                    else "N/A"
                ),
                "slope": {
                    "forward": float(
                        np.mean(
                            slope_maps.directional_rasters[compass_direction].data[
                                :, indices[0], indices[1]
                            ]
                        )
                    ),
                    "side": float(
                        np.mean(
                            slope_maps.directional_rasters[
                                np.mod(compass_direction + 90, 360)
                            ].data[:, indices[0], indices[1]]
                        )
                    ),
                },
                "speed": float(speed),
                "transverse_times": float(transverse_time),
                "total_time": float(total_time),
                "total_length": float(total_length),
            }
            if fill_omnidirectional_info:
                path_json["waypoints"][idx]["path_omni_based_info"] = {
                    "move_direction": float(direction_angle),
                    "move_compass_direction": direction,
                    "pathfinder_used_angle": (
                        float(pathfinder_used_angle)
                        if pathfinder_used_angle is not None
                        else "N/A"
                    ),
                    "speed": float(omni_speed),
                    "transverse_times": float(omni_transverse_time),
                    "total_time": float(omni_total_time),
                    "total_length": float(total_length),
                }
            path_json["waypoints"][idx]["indices"] = {
                "x": int(indices[1]),
                "y": int(indices[0]),
            }
            path_json["waypoints"][idx]["UTM"] = {
                "x": float(UTM_coor[0]),
                "y": float(UTM_coor[1]),
            }

            indices = indices_next
            UTM_coor = UTM_coor_next
            map_coor = map_coor_next

        if nodes is not None:
            path_json["waypoints"].append(
                {
                    "node": int(node),
                    "map": {
                        "x": float(map_coor[0]),
                        "y": float(map_coor[1]),
                    },
                    "elevation": float(
                        np.mean(elevation_raster.data[:, indices[0], indices[1]])
                    ),
                }
            )

        # Fill in the last waypoint
        path_json["waypoints"][-1]["path"] = {
            # FIXME: Should a null move direction be zero?
            "move_direction": float(0.0),
            "move_compass_direction": "N/A",
            "pathfinder_used_angle": "N/A",
            "slope": {
                "forward": float(0.0),
                "side": float(0.0),
            },
            "speed": float(0),
            "transverse_times": float(0),
            "total_time": float(total_time),
            "total_length": float(total_length),
        }
        if fill_omnidirectional_info:
            path_json["waypoints"][-1]["path_omni_based_info"] = {
                # FIXME: Should a null move direction be zero?
                "move_direction": float(0.0),
                "move_compass_direction": "N/A",
                "pathfinder_used_angle": "N/A",
                "speed": float(0),
                "transverse_times": float(0),
                "total_time": float(omni_total_time),
                "total_length": float(total_length),
            }
        path_json["waypoints"][-1]["indices"] = {
            "x": int(indices[1]),
            "y": int(indices[0]),
        }
        path_json["waypoints"][-1]["UTM"] = {
            "x": float(UTM_coor[0]),
            "y": float(UTM_coor[1]),
        }

        # Mark if the speed gets under the minimum speed
        lowest_speed = min(
            [waypoint["path"]["speed"] for waypoint in path_json["waypoints"][:-1]]
        )
        if lowest_speed < minimum_speed:
            path_json["metadata"]["minimum_speed_violated"] = True
        else:
            path_json["metadata"]["minimum_speed_violated"] = False

        return path_json

    def calculate_mobility_for_path(
        self,
        path_folder,
        slope_maps,
        minimum_speed,
        target_speed,
        fill_omnidirectional_info=False,
    ):
        """
        Calculates mobility metrics for paths in a folder.

        Args:
            path_folder (Path): Path to the folder containing the path files.
            slope_maps (list[Map]): List of slope maps.
            minimum_speed (float): Minimum speed for mobility calculations.
            target_speed (float): Target speed for mobility calculations.
            fill_omnidirectional_info (bool): Whether to fill in omnidirectional mobility information.
        """
        for folder in path_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("path-"):
                for file in folder.iterdir():
                    if file.is_file() and file.suffix == ".json":
                        log.info(f"Calculating mobility for path in {folder / file}")
                        # Load the path
                        with open(file, "r") as f:
                            path = json.load(f)

                        path = self.fill_path_with_mobility_data(
                            path_json=path,
                            slope_maps=slope_maps,
                            minimum_speed=minimum_speed,
                            target_speed=target_speed,
                            fill_omnidirectional_info=fill_omnidirectional_info,
                        )

                        # Save changes
                        with open(file, "w") as json_file:
                            json.dump(path, json_file, indent=4)


class MobilityMap(Map):

    def __init__(
        self,
        direction,
        slope_map,
        land_use_map,
        nogo_map,
        moisture_content_map,
        friction_map,
        soil_type_map,
        mobility_analysis,
        minimum_speed,
        target_speed,
        clip_mobility_to_range=False,
    ):
        """
        Initialises a new mobility map.

        Args:
            direction (tuple): Tuple containing the direction and angle of the mobility map.
            slope_map (Map): Slope map.
            land_use_map (Map): Land use map.
            nogo_map (Map): Nogo map.
            moisture_content_map (Map): Moisture content map.
            friction_map (Map): Friction map.
            soil_type_map (Map): Soil type map.
            mobility_analysis (MobilityAnalysis): Mobility analysis object.
            minimum_speed (float): Minimum speed for mobility calculations.
            target_speed (float): Target speed for mobility calculations.
            clip_mobility_to_range (bool): Use slope magnitude and zero speed
                outside the tabulated range.
        """

        log.info("Initialising new mobility map.")

        # Initialise a raster object based on the slope map
        self.raster = slope_map.slope_raster.return_empty_copy()

        # The mobility map is initialized with zero-speeds, using land use map for shape reference
        self.raster.data = np.zeros_like(land_use_map.nogo)

        # Set direction associated with mobility map
        self.direction = direction[0]
        self.angle = direction[1]

        self.slope_raster = slope_map.directional_rasters[self.angle]
        self.sideslope_data = (
            jnp.zeros_like(self.slope_raster.data)
            if self.direction == "omni"
            else slope_map.directional_rasters[np.mod(self.angle + 90, 360)].data
        )  # zero sideslope is the same as not thinking about sideslope
        self.land_use_map = land_use_map
        self.soil_type_map = soil_type_map
        self.moisture_content_map = moisture_content_map
        self.friction_map = friction_map
        self.nogo_map = nogo_map
        self.mobility_analysis = mobility_analysis
        self.clip_mobility_to_range = clip_mobility_to_range

        # Converts soil type map to a map of integers corresponding to the index of the mobility tensor in the mobility analysis object
        for i, soil_type in enumerate(self.mobility_analysis.soil_types):
            self.soil_type_map.soilarray[self.soil_type_map.soilarray == soil_type] = i

        log.info("Populating mobility map.")

        self.populate_map(minimum_speed=minimum_speed, target_speed=target_speed)

    def __call__(self):
        """
        Returns the mobility map as a NumPy array.

        Returns:
            np.ndarray: Mobility map as a NumPy array.
        """
        return np.array(self.data)

    def populate_map(self, minimum_speed, target_speed):
        """
        Populates the mobility map with speeds for each cell.

        Args:
            minimum_speed (float): Minimum speed for mobility calculations.
            target_speed (float): Target speed for mobility calculations.
        """
        index_x, index_y = np.unravel_index(
            np.arange(0, self.raster.data.size), self.raster.data.shape
        )

        # Check if cell is a no-go zone based on defined danger zones and land use
        nogo_defined_zones = self.nogo_map.data[index_x, index_y]
        land_use_nogo_mask = self.land_use_map.nogo[index_x, index_y]

        nogo_mask = np.logical_or(nogo_defined_zones, land_use_nogo_mask)

        # Vectors with index for go and no-go zones
        go_index_x = index_x[~nogo_mask]
        go_index_y = index_y[~nogo_mask]

        # Get soil type, moisture content and directional slope for go zones
        soil_type = jnp.array(self.soil_type_map.soilarray, dtype=jnp.int32)
        moisture_content = jnp.array(self.moisture_content_map.moisturearray)
        friction = jnp.array(self.friction_map.raster.data)
        directional_slope = jnp.rad2deg(
            jnp.arctan(jnp.array(self.slope_raster.data))
        )  # Convert slope from ratio (raise/run) to degrees
        if self.clip_mobility_to_range:
            directional_slope = jnp.abs(directional_slope)
        sideslope = jnp.abs(
            jnp.rad2deg(jnp.arctan(jnp.array(self.sideslope_data)))
        )  # Same to sideslope, uses only absolute value of the sideslope (= does not care about slope to the left or right)

        n_sample = np.max(
            (moisture_content.shape[0], friction.shape[0], directional_slope.shape[0])
        )

        # Evaluate the mobility tensor for the go zones
        # slice(None) if moisture_content.shape[0]  == n_sample else 0 takes care of the case if the parameter is not sampled.
        speeds = self.mobility_analysis.vmap_evaluate_point_in_space(
            soil_type[go_index_x, go_index_y],
            moisture_content[
                slice(None) if moisture_content.shape[0] == n_sample else 0,
                go_index_x,
                go_index_y,
            ],
            friction[
                slice(None) if friction.shape[0] == n_sample else 0,
                go_index_x,
                go_index_y,
            ],
            directional_slope[
                slice(None) if directional_slope.shape[0] == n_sample else 0,
                go_index_x,
                go_index_y,
            ],
            sideslope[
                slice(None) if sideslope.shape[0] == n_sample else 0,
                go_index_x,
                go_index_y,
            ],
        )

        speeds = jnp.where(
            speeds < minimum_speed, 0, speeds
        )  # Set speeds below the minimum speed to zero
        speeds = jnp.where(
            speeds > target_speed, target_speed, speeds
        )  # Set speeds above the target speed to the target speed

        self.raster.data[go_index_x, go_index_y] = speeds

    def get_values(self, slope_dataset, slope_mobility):
        """
        Sets the speed values in the mobility map based on the slope mobility dictionary.

        Args:
            slope_dataset (np.ndarray): Slope values for each cell in the mobility map.
            slope_mobility (dict): Dictionary mapping slope values to speeds.
        """
        # Initialise interpolator for speed values
        speed_interpolator = spi.interp1d(
            list(slope_mobility.keys()), list(slope_mobility.values()), kind="linear"
        )

        for i in range(0, slope_dataset.size):
            index = np.unravel_index(i, self.raster.data.shape)
            self.raster.data[index] = speed_interpolator(slope_dataset[index])

    def plot(
        self,
        n_samples=0,
        export_path=None,
        show=True,
        export=False,
        save_maps=False,
        hide_utm=False,
    ):
        """
        Plots the mobility map.

        Args:
            n_samples (int): Number of samples to plot.
            export_path (str): Path to export the plot.
            show (bool): Whether to display the plot.
            export (bool): Whether to export the plot.
            save_maps (bool): Whether to save the mobility map as an image.
            hide_utm (bool): Whether to hide the UTM coordinates.
        """
        title = (
            None
            if n_samples <= 1
            else f"Mobility map with {n_samples} samples and sample reduction function: {self.mobility_analysis.sample_reduction.function.name}"
        )

        direction_name = self.direction.lower() if type(self.direction) == str else str(self.direction).replace('.', '_')

        self.raster.plot(
            colorbar="Speed [m/s]",
            clim=(0, self.mobility_analysis.target_speed),
            title=title,
            export_path=export_path
            / f"mobility_{self.mobility_analysis.mobility_tensors[0].vehicle}"
            / f"mobility_maps/mobility_map_{direction_name}.pdf",
            show=show,
            export=export,
            save_maps=save_maps,
            hide_utm=hide_utm,
        )