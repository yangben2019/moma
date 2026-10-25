"""
Planning module
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


import json
import logging

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rasterio as rio
import scipy.interpolate as spi

from mpl_toolkits.mplot3d.art3d import Line3DCollection

log = logging.getLogger("MOMA")


class Path:
    """Path"""

    def __init__(self, waypoints, DiGraph):
        """Path object constructor. Defines a path using a provided set of waypoints.

        Args:
            waypoints (list): Iterable of waypoints defined as 2-element tuples.
        """

        self.waypoints = waypoints
        self.DiGraph = DiGraph

    # def get_optimal_path_raster(self, shape):
    #    """Optimal path raster generator
    #
    #    Args:
    #        shape (tuple): Output raster shape
    #    """
    #
    #    # Pre-allocate shortest path raster
    #    optimal_path_raster = np.zeros(shape)
    #
    #    # Highlight points which are part of the shortest path
    #    for index, _ in enumerate(optimal_path_raster.flat):
    #        if index in self.waypoints:
    #            optimal_path_raster[np.unravel_index(index, shape)] = 1
    #        else:
    #            optimal_path_raster[np.unravel_index(index, shape)] = 0
    #
    #    self.raster = optimal_path_raster

    def plot(
        self,
        overlay,
        vehicle,
        omnidirectional_based=False,
        show=False,
        export_path=False,
    ):
        import matplotlib as mpl  # Used for color maps definition

        (_, ax, _, _) = overlay.plot(show=False)

        # Transform path to UTM coordinates
        x, y = rio.transform.xy(
            overlay.transform, self.yint, self.xint, offset="center"
        )

        ax.scatter(
            x,
            y,
            color="k",
            marker="o",
            s=1.25,
            # cmap="seismic",
        )
        ax.scatter(x[0], y[0], color="b", marker="s", s=7.5)
        ax.scatter(x[-1], y[-1], color="g", marker="X", s=7.5)


        if export_path:
            plt.savefig(
                export_path
                / f"mobility_{vehicle}" / "paths"
                / f"path_optimal{'_omnidirectional' if omnidirectional_based else '_directional'}.pdf",
                format="pdf",
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close()

    # TODO Review spline smoothing algorithm
    def smoothen_spline(self, shape, points=400):

        logging.info(
            f"Densifying and smoothing path using a cubic spline interpolator over {points} points."
        )

        y, x = np.unravel_index(self.waypoints, shape)
        f, u = spi.splprep([x, y], k=1, s=0.15)
        self.xint, self.yint = spi.splev(np.linspace(0, 1, points), f)

    def print_path(
        self,
        elevation_raster,
        slope_maps,
        directional_mobility_map,
        vehicle,
        minimum_speed,
        target_speed,
        fill_omnidirectional_info=False,
        omnidirectional_based=False,
        export=False,
    ):
        log.info(f"Shortest path found! Printing cell list to .JSON file.")

        path_json = directional_mobility_map.fill_path_with_mobility_data(
            nodes=self.waypoints,
            DiGraph=self.DiGraph,
            elevation_raster=elevation_raster,
            slope_maps=slope_maps,
            minimum_speed=minimum_speed,
            target_speed=target_speed,
            fill_omnidirectional_info=fill_omnidirectional_info,
        )

        if export:
            with open(
                export
                / f"mobility_{vehicle}" / "paths"
                / f"path_optimal{'_omnidirectional' if omnidirectional_based else '_directional'}.json",
                "w",
            ) as f:
                json.dump(path_json, f, indent=4)

    def plot_path(self, vehicle, export, omnidirectional_based=False, show=False):
        # load json file
        with open(
            export
            / f"mobility_{vehicle}" / "paths"
            / f"path_optimal{'_omnidirectional' if omnidirectional_based else '_directional'}.json",
            "r",
        ) as f:
            data = json.load(f)["waypoints"]

        # Extract the coordinates from the JSON data
        coordinates = []
        speeds = []
        for value in data:
            x = value["map"]["x"]
            y = value["map"]["y"]
            z = value["elevation"]
            speed = value["path"]["speed"]
            coordinates.append((x, y, z))
            speeds.append(speed)

        # Convert to segments for individual coloring
        speeds = np.array(speeds)
        coordinates = np.array(np.moveaxis(coordinates, 0, -1)).T.reshape(-1, 1, 3)
        segments = np.concatenate([coordinates[:-1], coordinates[1:]], axis=1)

        lc = Line3DCollection(
            segments, cmap="plasma", norm=plt.Normalize(speeds.min(), speeds.max())
        )
        lc.set_array(speeds[:-1])  # set the value used for coloring
        lc.set_linewidth(2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(lc)

        fig.colorbar(lc, ax=ax, label="Speed")
        ax.set_xlabel("Map longitude [m]")
        ax.set_ylabel("Map latitude [m]")
        ax.set_zlabel("Elevation [m]")

        ax.set_xlim(coordinates[:, 0, 0].min(), coordinates[:, 0, 0].max())
        ax.set_ylim(coordinates[:, 0, 1].min(), coordinates[:, 0, 1].max())
        ax.set_zlim(coordinates[:, 0, 2].min(), coordinates[:, 0, 2].max())

        if export:
            plt.savefig(
                export
                / f"mobility_{vehicle}" / "paths"
                / f"path_optimal_3d{'_omnidirectional' if omnidirectional_based else '_directional'}.pdf",
                format="pdf",
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close()

    # TODO Define Savgol filter algorithm and compare with the spline algorithm
    # def smoothen_savgol(self):
    #
    #    #yint = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
    #    #shortpath = np.ravel_multi_index((xint, yint), (125,125))

    # TODO Define an AUVAS specification for importing coordinates from a planner.
    # def convert_path_to_coordinates(xmin, xmax, ymin, ymax, xres, yres, path):
    #
    #    pass
    #
    # def convert_to_vehicle_origin(waypoints, vehiclex, vehicley):
    #
    #    new_waypoints = []
    #
    #    for waypointx, waypointy in waypoints:
    #        new_waypoints.append(waypointx+vehiclex)


class MobilityPlanner:
    """Mobility planner object"""

    def __init__(self, site):

        self.site = site


def make_speed_graph(directional_mobility_map, hard_no_go=False):
    """Returns the weighted digraph for a given mobility map

    Returns the weighted digraph for a directional mobility map by defining a node for each mobility map cell and a set of at least 3 and at most 8 weighted edges connecting each cell to its neighbours through an appropriate speed weight as specified by the mobility map layer in the given direction.

    Args:
        mobility_map (dict): Directional mobility map with cardinal and intercardinal direction as keys
        hard_no_go (bool): If True, zero-speed cells are impassable. If False,
            they receive a large finite penalty (backwards compatible).

    Returns:
       networkx.digraph: Directional mobility map weighted digraph
    """

    log.info(f"Generating directional mobility graph from mapped data.")

    # Initialise empty graph
    DG = nx.DiGraph()

    # Add weighted edges
    we = define_weighted_edges(directional_mobility_map, hard_no_go=hard_no_go)
    DG.add_edges_from(we)

    return DG


def define_weighted_edges(directional_mobility_map, hard_no_go=False):
    """Builds weighted edges, optionally treating zero-speed cells as impassable."""

    def compass_angle2next_cell(angle):
        """Convert an compass angle to the next cell direction

        Args:
            angle (float): Angle in degrees

        Returns:
            str: Direction string corresponding to the compass angle
            tuple (int, int): A (dy, dx) tuple indicating the relative position of a neighboring cell in grid cell units
        """

        angle = (
            np.mod(np.deg2rad(angle) + np.pi, 2 * np.pi) - np.pi
        )  # Convert to radians and normalize to [-pi, pi]

        if angle < np.arctan2(-0.5, -1) or angle > np.arctan2(0.5, -1):
            return "S", (1, 0)
        elif angle < np.arctan2(-1, -0.5):
            return "SW", (1, -1)
        elif angle < np.arctan2(-1, 0.5):
            return "W", (0, -1)
        elif angle < np.arctan2(-0.5, 1):
            return "NW", (-1, -1)
        elif angle < np.arctan2(0.5, 1):
            return "N", (-1, 0)
        elif angle < np.arctan2(1, 0.5):
            return "NE", (-1, 1)
        elif angle < np.arctan2(1, -0.5):
            return "E", (0, 1)
        elif angle < np.arctan2(0.5, -1):
            return "SE", (1, 1)

    log.info(f"Computing mobility graph edge weights across the map.")

    weighted_edges_dict = {}
    shape = list(directional_mobility_map.mobility_maps.values())[0].raster.data.shape

    for angle, map in directional_mobility_map.mobility_maps.items():
        if not map.direction == "omni":
            log.info(f"Processing mobility map for direction {map.direction}.")

            next_cell_direction, next_cell = compass_angle2next_cell(angle)

            # Finds valid starting and ending indices
            if next_cell[0] == 1:
                begin_i = np.arange(0, shape[0] - 1)
            elif next_cell[0] == 0:
                begin_i = np.arange(0, shape[0])
            elif next_cell[0] == -1:
                begin_i = np.arange(1, shape[0])

            if next_cell[1] == 1:
                begin_j = np.arange(0, shape[1] - 1)
            elif next_cell[1] == 0:
                begin_j = np.arange(0, shape[1])
            elif next_cell[1] == -1:
                begin_j = np.arange(1, shape[1])

            end_i = begin_i + next_cell[0]
            end_j = begin_j + next_cell[1]

            begin_i, begin_j = np.meshgrid(begin_i, begin_j, indexing="ij")
            end_i, end_j = np.meshgrid(end_i, end_j, indexing="ij")

            # Putting each pair into the graph
            cells = (begin_i, begin_j)

            # Compute indices
            index = np.ravel_multi_index(cells, shape)
            neighbor_index = np.ravel_multi_index((end_i, end_j), shape)

            cell_vals = map.raster.data[cells]
            valid_cells = cell_vals > 0
            valid_neighbors = map.raster.data[(end_i, end_j)] > 0
            # If the neighbor is speed 0 (nogo or steep slope), set a high weight equal til going 0.001 m/s ~equal to max weight on the Vejlby map
            cell_vals = np.where(
                cell_vals == 0, 0.001, cell_vals
            )  # Avoid division by zero

            p = 1  # Power for the weight function
            neighbor_weight = cell_vals ** (-p)
            neighbor_weight *= (
                directional_mobility_map.DIRECTIONAL_DISTANCES_MULTIPLIER[
                    next_cell_direction
                ]
            )

            # Average the weight for a 5x5 kernel
            average_kernel = np.ones((5, 5)) / 25.0
            average_neighbor_weight = cv2.filter2D(neighbor_weight, -1, average_kernel)

            index_shape = index.shape

            for i in range(index_shape[0]):
                for j in range(index_shape[1]):
                    if hard_no_go and not (valid_cells[i, j] and valid_neighbors[i, j]):
                        continue
                    key = (index[i, j], neighbor_index[i, j])

                    # Find centre of the kernel for the next cell direction
                    kernel_i = i + next_cell[0] * 2
                    kernel_j = j + next_cell[1] * 2
                    # If the kernel is out of bound go to bounds
                    if kernel_i < 0:
                        kernel_i = 0
                    elif kernel_i > index_shape[0] - 1:
                        kernel_i = index_shape[0] - 1

                    if kernel_j < 0:
                        kernel_j = 0
                    elif kernel_j > index_shape[1] - 1:
                        kernel_j = index_shape[1] - 1

                    # Kernel penalty from an average in the move direction

                    kernel_penalty = average_neighbor_weight[kernel_i, kernel_j]
                    weight = neighbor_weight[i, j] + 1 / 20 * kernel_penalty

                    if key not in weighted_edges_dict:
                        weighted_edges_dict[key] = {
                            "weight": weight,
                            "angle": angle,
                        }
                    elif weighted_edges_dict[key]["weight"] > weight:
                        # If the weight is lower, update it
                        weighted_edges_dict[key]["weight"] = weight
                        weighted_edges_dict[key]["angle"] = angle

    # Convert the dictionary to a list
    weighted_edges_list = [
        (start, end, data) for (start, end), data in weighted_edges_dict.items()
    ]

    return weighted_edges_list


def find_shortest_path(
    mobility_digraph, source, target, omnidirectional=False, verbose=True
):
    """Returns the shortest path between a source and target nodes in a directional mobility digraph

    Returns the shortest path between a source and target nodes in a directional mobility digraph.

    Args:
        mobility_digraph (_type_): _description_
        source (_type_): _description_
        target (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    log.info(
        f"Finding shortest path between {source} and {target}{', with omnidirectional predictions' if omnidirectional else ''}."
    )

    best_path = nx.shortest_path(
        mobility_digraph, source=source, target=target, weight="weight"
    )

    return Path(best_path, mobility_digraph)


# TODO Implement weighted edges search through a convolution kernel
# def define_weighted_edges_kernel(mobility_maps):
#
#    weighted_edges_list = []
#
#    kernels = {
#        "N":  np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
#        "S":  np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
#        "E":  np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
#        "W":  np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
#        "NW": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
#        "SW": np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
#        "NE": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
#        "SE": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
#    }
#    dirs = ["N", "S", "E", "W", "NW", "SW", "NE", "SE"]
#
#    shape = mobility_maps["N"].data.shape
#    indices = np.linspace(0, mobility_maps["N"].data.size, mobility_maps["N"].data.size).reshape(shape)
#
#
#
#    # Compute all neighbors
#    neighbors = {}
#    for dire in dirs:
#        neighbors[dire] = cv2.filter2D(indices, -1, kernels[dire])
#        neighbors[dire] = neighbors[dire].astype(int)
#    indices = indices.astype(int)
#    arrays = [indices, *neighbors.values()]
#    merged_arrays = {}
#    #merged_arrays = np.stack(arrays, axis=2).reshape(indices.size, 9)
#
#    merged_arrays["N"] = np.stack([indices, neighbors["N"]], axis=2).reshape(indices.size, 2).astype(int).tolist()
#    merged_arrays["N"] = [(*element, {"weight": 1}) for element in merged_arrays["N"]]
#    #for item in merged_arrays:
#    #    #TODO fix the weights
#    #    [weighted_edges_list.append((item[0], myitem, {"weight": 1})) for myitem in item[1:-1]]
#
#    return merged_arrays["N"]
