"""
Toolboxes module
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
import pathlib
import yaml

import numpy as np

from coloredlogs import ColoredFormatter
from matplotlib import rcParams

from moma.dataloader import get_dataloader
from moma.mapping import (
    convert_coordinate_to_pixel,
    AerialMap,
    direction2compass_angle,
    ElevationMap,
    LandUseMap,
    MoistureContentMap,
    FrictionMap,
    NoGoMap,
    sample_points,
    SlopeMap,
    SoilTypeMap,
)
from moma.mobility import DirectionalMobilityMap, MobilityAnalysis, MobilityMap
from moma.planning import find_shortest_path, make_speed_graph


class MobilityToolbox:
    def __init__(
        self,
        sites_path,
        vehicles_path,
        site_name,
        vehicle_name,
        n_samples,
        directions,
        minimum_speed,
        target_speed,
        export_plots=True,
        save_maps=False,
        show_plots=False,
        hide_utm=False,
        path_planning=False,
    ):
        """
        Initialises the mobility toolbox.

        Args:
            sites_path (Path): Path to the directory containing site data.
            vehicles_path (Path): Path to the directory containing vehicle data.
            site_name (str): Name of the site.
            vehicle_name (str): Name of the vehicle.
            n_samples (int): Number of samples to generate.
            directions (list): List of directions for slope maps and mobility maps.
            minimum_speed (float): Minimum speed for mobility simulations.
            target_speed (float): Target speed for mobility simulations.
            export_plots (bool): Whether to export plots.
            save_maps (bool): Whether to save mobility maps.
            show_plots (bool): Whether to display plots.
            hide_utm (bool): Whether to hide UTM coordinates in plots.
        """

        # Initialise logger
        LOG_LEVEL = logging.INFO
        LOGFORMAT = "  %(asctime)s %(levelname)s %(message)s"

        formatter = ColoredFormatter(LOGFORMAT, datefmt="%H:%M:%S")
        stream = logging.StreamHandler()
        stream.setLevel(LOG_LEVEL)
        stream.setFormatter(formatter)
        self.log = logging.getLogger("MOMA")
        self.log.setLevel(LOG_LEVEL)
        self.log.addHandler(stream)

        base_path = sites_path / site_name
        self.base_path = base_path
        self.site_name = site_name

        # Database for vehicle mobility and landuse_nogo
        self.vehicles_path = vehicles_path / vehicle_name
        self.vehicle = vehicle_name

        # Creating and export directory if it does not exist
        self.EXPORT_PATH = base_path / "export/"
        self.EXPORT_PATH.mkdir(exist_ok=True)
        (self.EXPORT_PATH / f"mobility_{vehicle_name}").mkdir(exist_ok=True)
        (self.EXPORT_PATH / f"mobility_{vehicle_name}/mobility_maps").mkdir(exist_ok=True)
        (self.EXPORT_PATH / f"mobility_{vehicle_name}/nogo_maps").mkdir(exist_ok=True)
        (self.EXPORT_PATH / f"mobility_{vehicle_name}/paths").mkdir(exist_ok=True)
        (self.EXPORT_PATH / "site").mkdir(exist_ok=True)
        (self.EXPORT_PATH / "slope_maps").mkdir(exist_ok=True)
        self.export_plots = export_plots
        self.save_maps = save_maps
        self.show_plots = show_plots
        self.hide_utm = hide_utm

        self.n_samples = n_samples

        self.minimum_speed = minimum_speed
        self.target_speed = target_speed
        self.path_planning = path_planning

        # Directions for slope maps and mobility maps, both name and angle in degrees
        self.directions = {
            "names": directions[0],
            "angles": [
                direction2compass_angle(direction) for direction in directions[0]
            ],
        }

        # Use primary directions if no directions are provided
        if len(self.directions["names"]) == 0 and directions[1] == 0:
            self.directions = {
                "names": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                "angles": [0, 45, 90, 135, 180, 225, 270, 315],
            }

        # Add directions given by N_directions
        if not directions[1] == 0:
            angles = np.round(np.linspace(0, 360, directions[1], endpoint=False), 3)
            for angle in angles:
                if angle not in self.directions["angles"]:
                    self.directions["angles"].append(angle)
                    self.directions["names"].append(angle)

        self.log.info(
            f"Initialising MOMA for \n\tsite: {self.site_name} \n\tvehicle: {self.vehicle} \n\tn_samples: {n_samples} \n\tdirections: {self.directions['names']}"
        )

        with open(base_path / "site.yaml") as site_yaml:
            site_yaml = yaml.load(site_yaml, Loader=yaml.SafeLoader)
        self.datasource = site_yaml["DataSource"]
        self.coordinates = site_yaml["Coordinates"]
        self.nogo_zones = site_yaml["NoGoZones"]

        self.dataloader = get_dataloader(base_path, self.datasource, self.coordinates)
        self.log.info(f"Data source: {self.dataloader.source}")

        self.RESAMPLING_FACTOR = 0.2
        self.RESAMPLING_ALGORITHM = "bilinear"

        self.SOIL_TYPES = ["HT", "SC", "GW", "CL"]

        # Sets default for plotting
        rcParams.update(
            {
                "font.size": 14,
                "axes.labelpad": 10,
            }
        )

    def generate_aerial_map(self, show=True):
        """
        Generates an aerial map for the site.

        Args:
            show (bool): Whether to display the map.

        Returns:
            None
        """

        # Import and process the aerial map raster
        self.aerial_raster = self.dataloader.get_aerial_raster()

        # Resample the aerial raster by the specified factor
        self.aerial_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")
        
        # Instantiate an aerial map object
        self.aerial_map = AerialMap(self.site_name, self.aerial_raster)

        # Show the aerial map (and save plot to disk)
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.aerial_map.plot(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_elevation_map(self, show=True):
        """
        Generates an elevation map for the site.

        Args:
            show (bool): Whether to display the map.

        Returns:
            None
        """
        # Import and process the elevation map raster
        self.elevation_raster = self.dataloader.get_elevation_raster()

        # Resample the elevation raster by the specified factor
        self.elevation_raster.resample(
            self.RESAMPLING_FACTOR, resampling=self.RESAMPLING_ALGORITHM
        )

        # Sample point from the uncertainty distribution
        self.elevation_raster.data = sample_points(
            self.elevation_raster.data, self.n_samples
        )

        # Instantiate an elevation map object
        self.elevation_map = ElevationMap(self.site_name, self.elevation_raster)

        # Show the elevation map
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.elevation_map.plot(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_directional_slope_maps(self, show=True):
        """
        Generates directional slope maps for the site.

        Args:
            show (bool): Whether to display the maps.

        Returns:
            None
        """

        # TODO Switch directional slope maps generation to user-initiated
        # Instantiate a slope map object (this will automatically compute directional slope maps!)
        self.slope_map = SlopeMap(
            self.site_name, self.elevation_map, directions=self.directions, path_planning=self.path_planning
        )

        # Show the directional slope maps
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.slope_map.plot_slope_aspect(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )
            self.slope_map.plot(
                directions=self.directions,
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_land_use_map(self, show=True):
        """
        Generates a land use map for the site.

        Args:
            show (bool): Whether to display the map.

        Returns:
            None
        """
        # Load land use raster and value mappings
        landuse_raster, landuse_valuemap = self.dataloader.get_landuse_raster()

        # Resample the land use raster by the specified factor
        landuse_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")

        # Import land use no-go mappings
        # TODO Get rid of the hard-coded land use no-go mappings
        with open(
            self.vehicles_path / "landuse_nogo.yaml", encoding="utf8"
        ) as landuse_nogo_file:
            landuse_nogo = yaml.load(landuse_nogo_file, Loader=yaml.SafeLoader)

        # Instantiate a land use map object
        self.landuse_map = LandUseMap(
            self.site_name, landuse_raster, landuse_valuemap, landuse_nogo
        )

        # Show the land use map
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.landuse_map.plot(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_soil_type_map(self, show=True):
        """
        Generates a soil type map for the site.

        Args:
            show (bool): Whether to display the map.

        Returns:
            None
        """
        # Generate soil type map
        soiltype_raster, soiltype_valuemap = self.dataloader.get_soiltype_raster()

        # Resample the soil type raster by the specified factor
        soiltype_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")

        # Instantiate a soil type map object
        self.soiltype_map = SoilTypeMap(
            self.site_name, soiltype_raster, soiltype_valuemap
        )

        # Show the soil type map
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.soiltype_map.plot(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_moisture_content_map(self, show=True):
        """
        Generates a moisture content map for the site.

        Args:
            show (bool): Whether to display the map.

        Returns:
            None
        """

        # Generate moisture content map
        moisturecontent_raster, moisturecontent_valuemap = (
            self.dataloader.get_moisturecontent_raster()
        )

        # Resample the moisture content raster by the specified factor
        moisturecontent_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")

        self.moisturecontent_map = MoistureContentMap(
            self.site_name, moisturecontent_raster, moisturecontent_valuemap
        )

        # Sample point from the uncertainty distribution
        self.moisturecontent_map.moisturearray = sample_points(
            self.moisturecontent_map.moisturearray, self.n_samples
        )

        # Show the moisture content map
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.moisturecontent_map.plot(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_friction_map(self, show=True):
        """
        Generates a friction map for the site.

        Args:
            show (bool): Whether to display the map.

        Returns:
            None
        """
        # Generate friction map
        friction_raster = self.dataloader.get_friction_raster()

        # Resample the friction raster by the specified factor
        friction_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")

        # Sample point from the uncertainty distribution
        friction_raster.data = sample_points(friction_raster.data, self.n_samples)

        self.friction_map = FrictionMap(self.site_name, friction_raster)

        # Show the soil friction map
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.friction_map.plot(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_no_go_map(self, show=True):
        """
        Generates a no-go map for the site.

        Args:
            show (bool): Whether to display the map.

        Returns:
            None
        """

        # Generate no-go map, use aerial map for visualization
        self.nogo_map = NoGoMap(
            self.site_name, self.vehicle, self.aerial_map, self.landuse_map
        )

        # Add no-go zones from a list of zones with the format: [[x1, x2, y1, y2, "description"], ...]
        if self.nogo_zones is not None:
            for zone in self.nogo_zones:
                if type(zone[-1]) == str:
                    self.nogo_map.add_zone(zone[0:-1], zone[-1])
                else:
                    self.nogo_map.add_zone(zone)

        # Show the no-go map
        if show and (self.show_plots or self.export_plots or self.save_maps):
            self.nogo_map.plot(
                export_path=self.EXPORT_PATH,
                show=self.show_plots,
                export=self.export_plots,
                save_maps=self.save_maps,
                hide_utm=self.hide_utm,
            )

    def generate_directional_mobility_maps(self, sample_reduction, show=True):
        """
        Generates directional mobility maps for the site.

        Args:
            sample_reduction (float): Reduction factor for sampling points.
            show (bool): Whether to display the maps.

        Returns:
            None
        """

        # Generate mobility tensors
        soil_parameter = ["friction_coefficient", "moisture_content"]
        mobility_analysis = MobilityAnalysis(
            self.vehicles_path / "mobility.yaml",
            self.vehicle,
            self.SOIL_TYPES,
            soil_parameter,
            sample_reduction,
            self.minimum_speed,
            self.target_speed,
        )
        mobility_analysis.generate_mobility_tensors("slope")
        mobility_analysis.generate_sideslope_factor()
        mobility_analysis.create_evaluate_functions()

        n_samples = list(self.slope_map.directional_rasters.values())[0].data.shape[0]

        # Adds all 8 compass directions if omnidirectional is used and pathplanning is enabled
        calculate_directions = []
        calculate_angles = []
        for direction, angle in zip(
            self.directions["names"], self.directions["angles"]
        ):
            calculate_directions.append(direction)
            calculate_angles.append(angle)
            if direction == "omni" and self.path_planning:
                for compass in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                    compass_angle = direction2compass_angle(compass)
                    if (
                        compass_angle not in self.directions["angles"]
                        and compass_angle not in calculate_angles
                    ):
                        calculate_directions.append(compass)
                        calculate_angles.append(compass_angle)

        # Generate mobility maps
        self.mobility_maps = {}
        for direction, angle in zip(calculate_directions, calculate_angles):
            self.log.info(f"Generating mobility map in the {direction} direction.")
            self.mobility_maps[angle] = MobilityMap(
                (direction, angle),
                self.slope_map,
                self.landuse_map,
                self.nogo_map,
                self.moisturecontent_map,
                self.friction_map,
                self.soiltype_map,
                mobility_analysis,
                minimum_speed=self.minimum_speed,
                target_speed=self.target_speed,
            )

            # Show the mobility map
            if (
                show
                and (self.show_plots or self.export_plots or self.save_maps)
                and direction in self.directions["names"]
            ):
                self.log.info(f"Showing/exporting mobility map.")
                self.mobility_maps[angle].plot(
                    n_samples,
                    export_path=self.EXPORT_PATH,
                    show=self.show_plots,
                    export=self.export_plots,
                    save_maps=self.save_maps,
                    hide_utm=self.hide_utm,
                )

    def calculate_mobility(self, path_folder, omnidirectional=False):
        """
        Calculates mobility for a path.

        Args:
            path_folder (str): Path to the folder containing the path.
            omnidirectional (bool): Whether to use omnidirectional mobility.

        Returns:
            None
        """

        path_folder = pathlib.Path(path_folder)

        directional_mobility_map = DirectionalMobilityMap(self.mobility_maps)

        directional_mobility_map.calculate_mobility_for_path(
            path_folder,
            self.slope_map,
            self.minimum_speed,
            self.target_speed,
            omnidirectional,
        )

    def path_planner(self, waypoint_start, waypoint_end, coordinate_system, show=True):
        """
        Plans a path between two waypoints.

        Args:
            waypoint_start (Tuple[float, float]): Starting waypoint coordinates.
            waypoint_end (Tuple[float, float]): Ending waypoint coordinates.
            coordinate_system (str): Coordinate system for the waypoints.
            show (bool): Whether to display the path.

        Returns:
            None
        """

        # Convert waypoint coordinates to pixel coordinates
        if coordinate_system in ["map", "utm"]:
            waypoint_start = convert_coordinate_to_pixel(
                waypoint_start,
                self.elevation_raster,
                self.elevation_raster.data[0].shape,
                coordinate_system,
            )
            waypoint_end = convert_coordinate_to_pixel(
                waypoint_end,
                self.elevation_raster,
                self.elevation_raster.data[0].shape,
                coordinate_system,
            )
        else:
            raise ValueError(
                f"Coordinate system {coordinate_system} not supported. Use 'map' or 'utm'."
            )

        # Flattens the start and end waypoints
        self.waypoint_start = np.ravel_multi_index(
            np.flip(waypoint_start), self.elevation_raster.data[0].shape
        )
        self.waypoint_end = np.ravel_multi_index(
            np.flip(waypoint_end), self.elevation_raster.data[0].shape
        )

        # Find optimal path, sort away the directions that are added due to omnidirectional predictions
        directional_mobility_map = DirectionalMobilityMap(
            {angle: self.mobility_maps[angle] for angle in self.directions["angles"]}
        )

        # Generate directional mobility graph
        directional_mobility_graph = make_speed_graph(directional_mobility_map)

        # Generate directional mobility graph based on omnidirectional mobility maps
        # Works by using the directional framework, but with omnidirectional values in all directions
        omnidirectional = True if "omni" in self.directions["names"] else False
        if omnidirectional:
            directional_mobility_map_omni = DirectionalMobilityMap({})
            for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                direction_angle = direction2compass_angle(direction)
                directional_mobility_map_omni.mobility_maps[direction_angle] = (
                    self.mobility_maps["omni"]
                )
                directional_mobility_map_omni.mobility_maps[
                    direction_angle
                ].direction = "omni for pseudo direction"

            directional_mobility_graph_omni = make_speed_graph(
                directional_mobility_map_omni
            )

        # Creates a directional mobility map with the 8 primary compass directions
        directional_mobility_map_for_filling_metric = DirectionalMobilityMap({})
        for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            if direction not in self.directions["names"] and not omnidirectional:
                continue
            direction_angle = direction2compass_angle(direction)
            directional_mobility_map_for_filling_metric.mobility_maps[
                direction_angle
            ] = self.mobility_maps[direction_angle]
        if omnidirectional:
            directional_mobility_map_for_filling_metric.mobility_maps["omni"] = (
                self.mobility_maps["omni"]
            )

        # Find the shortest path between two waypoints if three or more directions that are not omnidirectional are available
        if len([angle for angle in directional_mobility_map.mobility_maps if angle != "omni"]) >= 3:
            shortpath = find_shortest_path(
                directional_mobility_graph, self.waypoint_start, self.waypoint_end
            )
            shortpath.print_path(
                elevation_raster=self.elevation_map.raster,
                directional_mobility_map=directional_mobility_map_for_filling_metric,
                slope_maps=self.slope_map,
                vehicle=self.vehicle,
                minimum_speed=self.minimum_speed,
                target_speed=self.target_speed,
                fill_omnidirectional_info=omnidirectional,
                export=self.EXPORT_PATH,
            )

        if omnidirectional:
            shortpath_omni = find_shortest_path(
                directional_mobility_graph_omni,
                self.waypoint_start,
                self.waypoint_end,
                omnidirectional=True,
            )

            shortpath_omni.print_path(
                elevation_raster=self.elevation_map.raster,
                directional_mobility_map=directional_mobility_map_for_filling_metric,
                slope_maps=self.slope_map,
                vehicle=self.vehicle,
                minimum_speed=self.minimum_speed,
                target_speed=self.target_speed,
                fill_omnidirectional_info=True,
                export=self.EXPORT_PATH,
                omnidirectional_based=True,
            )

        # Plots a 3D projection of the path
        # shortpath.plot_path(vehicle=self.vehicle, show=show, export=self.EXPORT_PATH)
        if len([angle for angle in directional_mobility_map.mobility_maps if angle != "omni"]) >= 3:
            shortpath.smoothen_spline(
                list(self.mobility_maps.values())[0].raster.data.shape
            )

        if omnidirectional:
            shortpath_omni.smoothen_spline(
                list(self.mobility_maps.values())[0].raster.data.shape
            )

        # Show optimal path on the aerial map
        if show and (self.show_plots or self.export_plots):
            if len([angle for angle in directional_mobility_map.mobility_maps if angle != "omni"]) >= 3:
                shortpath.plot(
                    overlay=self.elevation_raster,
                    vehicle=self.vehicle,
                    show=self.show_plots,
                    export_path=self.EXPORT_PATH,
                )

            if omnidirectional:
                shortpath_omni.plot(
                    overlay=self.elevation_raster,
                    vehicle=self.vehicle,
                    show=self.show_plots,
                    export_path=self.EXPORT_PATH,
                    omnidirectional_based=True,
                )
