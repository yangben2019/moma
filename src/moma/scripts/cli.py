"""
Command line interface for MOMA scripts
"""

import os
import platform

import pathlib
import argparse

from types import SimpleNamespace

from moma.toolboxes import MobilityToolbox
from moma.mobility import generate_vehicle_mobility_file

from moma.generators import TestMapGenerator

# ============================================
# Settings for parsing command line arguments
# ============================================
parser = argparse.ArgumentParser()
# Clear the console
parser.add_argument(
    "--clear", action="store_true", help="clear the console before running the script"
)
# Add subparsers
subparsers = parser.add_subparsers(required=True, help="subcommand help")


# ==========================
# MOMA Mobility calculation
# ==========================
def calculate_mobility(args):
    # Location of sites and vehicle mobilities inside MOMA
    sites_path = pathlib.Path(args.path) / "sites"
    vehicles_path = pathlib.Path(args.path) / "vehicles"

    omnidirectional = True if "omni" in args.directions else False

    moma = MobilityToolbox(
        sites_path,
        vehicles_path,
        args.site,
        args.vehicle,
        args.n_samples,
        (args.directions, args.n_directions),
        args.minimum_speed,
        args.target_speed,
        export_plots=not args.dont_export,
        save_maps=args.save_maps,
        show_plots=args.show_plots,
        hide_utm=args.hide_utm,
        path_planning=(not args.path_coordinate_system == "None" or not args.path_folder == "None")
    )

    # By adding show=False as argument to the following generate and path_planner functions the plots/maps are not generated stored regardless of the other inputs
    moma.generate_aerial_map()
    moma.generate_elevation_map()
    moma.generate_directional_slope_maps()
    moma.generate_land_use_map()
    moma.generate_soil_type_map()
    moma.generate_moisture_content_map()
    moma.generate_friction_map()
    moma.generate_no_go_map()
    moma.generate_directional_mobility_maps(args.sample_reduction)

    if not args.path_coordinate_system == "None":
        moma.path_planner(args.path_start, args.path_end, args.path_coordinate_system)

    if not args.path_folder == "None":
        moma.calculate_mobility(args.path_folder, omnidirectional=omnidirectional)


# Subparser for mobility calculation
mobility_parser = subparsers.add_parser("mobility", help="predict mobility")
mobility_parser.set_defaults(func=calculate_mobility)
# Show plots
mobility_parser.add_argument(
    "--show_plots", action="store_true", help="show popup with plots"
)
# Vehicle
mobility_parser.add_argument(
    "--vehicle",
    help="name of the vehicle to generate plots for, default mock",
    type=str,
    default="mock",
)
# Path to the site folder
mobility_parser.add_argument(
    "--site", help="name of the site folder", type=str, default="testmap"
)
# Number of sample points if one the mean value is used
mobility_parser.add_argument(
    "--n_samples",
    help="number of sample points, if 1 the mean value is used, default 1",
    type=int,
    default=1,
)
# Method used to reduce sample space to one value
mobility_parser.add_argument(
    "--sample_reduction",
    nargs="+",
    help="Method used to reduce sample space to one value. Support 'mean' and 'quantile q', default mean",
    default=["mean"],
)
# Start and end waypoints for the path planner
mobility_parser.add_argument(
    "--path", help="Path used for loading data", type=str, default="data"
)

mobility_parser.add_argument(
    "--path_coordinate_system",
    help="coordinate system for the path planner, 'map' or 'utm'",
    type=str,
    default="None",
)
mobility_parser.add_argument(
    "--path_start",
    help="start waypoint for the path planner",
    type=int,
    nargs=2,
    default=(0, 0),
)
mobility_parser.add_argument(
    "--path_end",
    help="end waypoint for the path planner",
    type=int,
    nargs=2,
    default=(0, 0),
)

mobility_parser.add_argument(
    "--path_folder",
    help="path to the folder with paths to predict mobility for",
    type=str,
    default="None",
)
mobility_parser.add_argument(
    "--dont_export", action="store_true", help="don't export plots to pdf files"
)

mobility_parser.add_argument(
    "--directions",
    nargs="+",
    help="List of directions to generate directional slope maps for specified as either 'omni', angle in degrees or abbreviations ('N', 'NNE', 'NW', etc.). default: all 8 primary compass directions",
    default=[],
)
mobility_parser.add_argument(
    "--n_directions", help="Number of directions for mobility maps", type=int, default=0
)
mobility_parser.add_argument(
    "--save_maps",
    action="store_true",
    help="Save the generated maps as GeoTIFF files in the export folder",
    default=False,
)
mobility_parser.add_argument(
    "--hide_utm",
    action="store_true",
    help="Hide the UTM coordinate system in the plots",
    default=False,
)
mobility_parser.add_argument(
    "--minimum_speed",
    type=float,
    help="minimum speed for the mobility calculation, default 0.5",
    default=0.5,
)
mobility_parser.add_argument(
    "--target_speed",
    type=float,
    help="target speed for the mobility calculation, default 10.0",
    default=10.0,
)


# ====================
# Test map generation
# ====================
def generate_test_map(args):
    test_map_params = SimpleNamespace()
    test_map_params.seed = args.seed
    test_map_params.max_slope_percent = args.max_slope_percent
    test_map_params.map_dim = args.map_dim
    test_map_params.plateaus = args.plateaus
    test_map_params.slope_length = args.slope_length
    test_map_params.resolution = args.resolution
    test_map_params.transition_pad = args.transition_pad
    test_map_params.start_pad = args.start_pad

    test_map_generator = TestMapGenerator()
    test_map_generator.main_generate(
        base_output_dir=args.output if args.output != "None" else None,
        N_test_maps=args.N_maps,
        elevation_std=args.elevation_std,
        friction_coefficient=args.friction_coefficient,
        friction_coefficient_std=args.friction_coefficient_std,
        plot_additional_files=args.plot_additional_files,
        map_type=args.map_type,
        output_resolution=args.output_resolution,
        create_rasters=args.create_rasters,
        test_map_params=test_map_params,
        coordinates_lower_left=args.coordinates_lower_left,
    )


# Subparser for generation test map
test_map_parser = subparsers.add_parser("generate_test_map", help="generate test map")
test_map_parser.set_defaults(func=generate_test_map)
test_map_parser.add_argument(
    "--N_maps", type=int, help="number of maps to generate, default 1", default=1
)
test_map_parser.add_argument(
    "--elevation_std",
    type=float,
    help="standard deviation of the elevation, default 0.02",
    default=0.00,
)
test_map_parser.add_argument(
    "--friction_coefficient",
    type=float,
    help="friction coefficient for the map, default 0.5",
    default=0.5,
)
test_map_parser.add_argument(
    "--friction_coefficient_std",
    type=float,
    help="standard deviation of the friction coefficient for the map, default 0.05",
    default=0.00,
)
test_map_parser.add_argument(
    "--output", type=str, help="path to output directory", default="None"
)
test_map_parser.add_argument(
    "--map_type",
    type=str,
    help='file type of map, "obj" or "bmp", default bmp',
    default="bmp",
)
test_map_parser.add_argument(
    "--output_resolution",
    type=int,
    help="resolution of the generated map, output is resolution x 0.25 m",
    default=6,
)
test_map_parser.add_argument(
    "--create_rasters",
    action="store_true",
    help="create rasters of test map for use in MOMA",
    default=False,
)
test_map_parser.add_argument(
    "--plot_additional_files",
    action="store_true",
    help="plot additional files such as elevation histogram",
    default=False,
)
test_map_parser.add_argument(
    "--seed", 
    type=int, 
    help="seed for random map generation", 
    default=2
)
test_map_parser.add_argument(
    "--max_slope_percent",
    type=float,
    help="maximum slope percentage for the test map, default 20.0",
    default=20.0,
)
test_map_parser.add_argument(
    "--map_dim",
    nargs=2,
    type=int,
    help="dimensions of the map in number of cells, default 200 200",
    default=(200, 200),
)
test_map_parser.add_argument(
    "--plateaus",
    nargs=2,
    type=int,
    help="number of plateaus in the test map, default 4 4",
    default=(4, 4),
)
test_map_parser.add_argument(
    "--slope_length",
    nargs=2,
    type=int,
    help="average length of slopes in the test map, default 30 30",
    default=(30, 30),
)
test_map_parser.add_argument(
    "--resolution",
    type=int,
    help="resolution of the test map in pixels per meter, default 4",
    default=4,
)
test_map_parser.add_argument(
    "--transition_pad",
    type=int,
    help="length of transition padding around the test map in meters, default 15",
    default=15,
)
test_map_parser.add_argument(
    "--start_pad",
    type=int,
    help="length of start padding around the test map in meters, default 10",
    default=10,
)
test_map_parser.add_argument(
    "--coordinates_lower_left",
    nargs=2,
    type=int,
    help="UTM Coordinates (easting, northing) of the lower left corner of the map, default (0, 0)",
    default=(0, 0),
)

# =========================
# Mobility file generation
# =========================

def generate_mobility_file(args):
    generate_vehicle_mobility_file(
        vehicle=args.vehicle,
        minimum_speed=args.minimum_speed,
        plot_mobility=args.plot,
    )


# Subparser for mobility file generation
mobility_file_parser = subparsers.add_parser(
    "generate_mobility_file", help="generate mobility file"
)
mobility_file_parser.set_defaults(func=generate_mobility_file)
mobility_file_parser.add_argument(
    "--vehicle", type=str, help="name of the vehicle to generate mobility file for"
)
mobility_file_parser.add_argument(
    "--minimum_speed",
    type=float,
    help="minimum speed for the mobility file, default 0.01",
    default=0.01,
)
mobility_file_parser.add_argument(
    "--plot",
    action="store_true",
    help="plot the data from generated mobility file",
    default=False,
)


# ====================
# Executing functions
# ====================
def main():
    args = parser.parse_args()

    if args.clear:
        if platform.system() == "Windows":
            os.system("cls")
        if platform.system() == "Linux":
            os.system("clear")

    args.func(args)


if __name__ == "__main__":
    main()
