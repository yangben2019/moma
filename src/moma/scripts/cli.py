""" This file runs the moma_class.py with the given arguments

Example call with CLI for calculating mobility:
    python src/use_moma_class.py --clear mobility --show_plots --vehicle mungo --site Vejlby --n_samples 10 --sample_reduction quantile 0.9 --path_coordinate_system utm --path_start 573500 6230500 --path_end 575900 6230250 
    python src/use_moma_class.py --clear mobility --vehicle mungo --site oksbol --n_samples 50 --sample_reduction mean --directions N NE E SE S SW W NW omni --path_coordinate_system utm --path_start 450750 6163600 --path_end 452250 6164200

Example call with CLI for calculating mobility on test map:
    python src/use_moma_class.py --clear mobility --site testmap --vehicle olav --path_folder path/to/path/folder 
    python src/use_moma_class.py --clear mobility --site testmap --vehicle olav --path_coordinate_system map --path_start 7 65 --path_end 210 140 --sample_reduction quantile 0.9 --n_samples 100
    python src/use_moma_class.py --clear mobility --site testmap --vehicle olav --directions N NE E SE S SW W NW omni --dont_export

Example call with CLI for generating test map:
    python src/use_moma_class.py --clear generate_test_map --N_maps 10
    python src/use_moma_class.py generate_test_map --output path/to/output/folder --create_rasters

Example call from CLI for generating straight line paths:
    python src/use_moma_class.py --clear generate_test_paths --output path/to/output/folder --plot_paths
    python src/use_moma_class.py --clear generate_test_paths --output path/to/output/folder --plot_paths --fill_metrics

Example call with CLI for generating mobility file:
    python src/use_moma_class.py --clear generate_mobility_file --vehicle olav
    python src/use_moma_class.py --clear generate_mobility_file --vehicle olav --plot
"""

import os
import platform

import pathlib
import argparse

from moma.toolboxes import MobilityToolbox
from moma.mobility import generate_vehicle_mobility_file

from moma.generators import TestMapGenerator
from moma.generators import TestPathsGenerator

# ============================================
# Settings for parsing command line arguments 
# ============================================
parser = argparse.ArgumentParser()
# Clear the console
parser.add_argument("--clear", action="store_true", 
                    help="clear the console before running the script") 
# Add subparsers
subparsers = parser.add_subparsers(required=True, help='subcommand help') 


# ==========================
# MOMA Mobility calculation
# ==========================
def calculate_mobility(args):
    # Location of sites and vehicle mobilities inside MOMA
    sites_path    = args.path / "sites"
    vehicles_path = args.path / "vehicles"

    omnidirectional = True if "omni" in args.directions else False

    moma = MobilityToolbox(
        sites_path, vehicles_path, 
        args.site, args.vehicle, 
        args.n_samples, (args.directions, args.n_directions), 
        args.minimum_speed, args.target_speed,
        export_plots=not args.dont_export, save_maps=args.save_maps, show_plots=args.show_plots, hide_utm=args.hide_utm
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
mobility_parser = subparsers.add_parser('mobility', help='predict mobility')
mobility_parser.set_defaults(func=calculate_mobility)
# Show plots
mobility_parser.add_argument("--show_plots", action="store_true", help="show popup with plots") 
# Vehicle
mobility_parser.add_argument("--vehicle", help="name of the vehicle to generate plots for, default mungo", type=str, default="mungo")
# Path to the site folder
mobility_parser.add_argument("--site", help="name of the site folder", type=str, default="wtd41_copy")
# Number of sample points if one the mean value is used
mobility_parser.add_argument("--n_samples", help="number of sample points, if 1 the mean value is used, default 1", type=int, default=1) 
# Method used to reduce sample space to one value
mobility_parser.add_argument("--sample_reduction", nargs='+', help="Method used to reduce sample space to one value. Support ['mean'] and ['quantile', q], default mean", default=["mean"])
# Start and end waypoints for the path planner
mobility_parser.add_argument("--path", help="Path used for loading data", type=str, default=pathlib.Path.cwd())

mobility_parser.add_argument("--path_coordinate_system", help="coordinate system for the path planner, 'map' or 'utm'", type=str, default="None")
mobility_parser.add_argument("--path_start", help="start waypoint for the path planner", type=int, nargs=2, default=(0, 0))
mobility_parser.add_argument("--path_end", help="end waypoint for the path planner", type=int, nargs=2, default=(0, 0))

mobility_parser.add_argument("--path_folder", help="path to the folder with paths to predict mobility for", type=str, default="None")
mobility_parser.add_argument("--dont_export", action="store_true", help="don't export plots to pdf files")
#mobility_parser.add_argument("--omnidirectional", action="store_true", help="Add omnidirectional mobility prediction to the paths", default=False)
mobility_parser.add_argument("--directions", nargs='+', help="List of directions to generate directional slope maps for specified as either 'omni', angle in degrees or abbreviations ('N', 'NNE', 'NW', etc.). default: all 8 primary compass directions", default=[])
mobility_parser.add_argument("--n_directions", help="Number of directions for mobility maps", type=int, default=0)
mobility_parser.add_argument("--save_maps", action="store_true", help="Save the generated maps as .tif files in the export folder", default=False)
mobility_parser.add_argument("--hide_utm", action="store_true", help="Hide the UTM coordinate system in the plots", default=False)
mobility_parser.add_argument("--minimum_speed", type=float, help="minimum speed for the mobility calculation, default 0.5", default=0.5)
mobility_parser.add_argument("--target_speed", type=float, help="target speed for the mobility calculation, default 5.0", default=5.0)


# ====================
# Test map generation
# ====================
def generate_test_map(args):

    TestMapGenerator.main_generate(
        base_output_dir=args.output if args.output != "None" else None,
        N_test_maps=args.N_maps,
        std_elevation=args.std_elevation,
        friction_coefficient=args.friction_coefficient,
        friction_coefficient_std=args.friction_coefficient_std, 
        map_type=args.map_type,
        output_resolution=args.resolution, 
        create_rasters=args.create_rasters)

# Subparser for generation test map
test_map_parser = subparsers.add_parser('generate_test_map', help='generate test map')
test_map_parser.set_defaults(func=generate_test_map)
test_map_parser.add_argument('--N_maps', type=int, help='number of maps to generate, default 1', default=1)
test_map_parser.add_argument('--std_elevation', type=float, help='standard deviation of the elevation, default 0.02', default=0.00)
test_map_parser.add_argument('--friction_coefficient', type=float, help='friction coefficient for the map, default 0.5', default=0.5)
test_map_parser.add_argument('--friction_coefficient_std', type=float, help='standard deviation of the friction coefficient for the map, default 0.05', default=0.00)
test_map_parser.add_argument('--output', type=str, help='path to output directory', default="None")
test_map_parser.add_argument('--map_type', type=str, help='file type of map, "obj" or "bmp", default bmp', default="bmp")
test_map_parser.add_argument('--resolution', type=int, help='resolution of the map, output is resolution x 0.25 m', default=6)
test_map_parser.add_argument('--create_rasters', action='store_true', help='create rasters of test map for use in MOMA', default=False)


# =====================
# Test path generation
# =====================
def generate_test_paths(args):

    TestPathsGenerator.main_generate(
        output_dir=args.output,
        plot_paths=args.plot_paths,
        fill_metrics=args.fill_metrics,
        n_samples=args.n_samples,
        sample_reduction=args.sample_reduction,
        minimum_speed=args.minimum_speed,
        target_speed=args.target_speed
    )

test_path_parser = subparsers.add_parser('generate_test_paths', help='generate test paths')
test_path_parser.set_defaults(func=generate_test_paths)
test_path_parser.add_argument('--output', type=str, help='path to output directory', default=".")
test_path_parser.add_argument('--plot_paths', action='store_true', help='plot the generated paths', default=False)
test_path_parser.add_argument('--fill_metrics', action='store_true', help='fill the generated paths with MOMA metrics', default=False)

# Argument for creating the correct mobility maps when using fill metrics
test_path_parser.add_argument("--n_samples", help="number of sample points, if one the mean value is used, default 1", type=int, default=1) 
test_path_parser.add_argument("--sample_reduction", nargs='+', help="Method used to reduce sample space to one value. Support ['mean'] and ['quantile', q], default mean", default=["mean"])
test_path_parser.add_argument("--minimum_speed", type=float, help="minimum speed for the mobility calculation, default 0.5", default=0.5)
test_path_parser.add_argument("--target_speed", type=float, help="target speed for the mobility calculation, default 5.0", default=5.0)

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
mobility_file_parser = subparsers.add_parser('generate_mobility_file', help='generate mobility file')
mobility_file_parser.set_defaults(func=generate_mobility_file)
mobility_file_parser.add_argument('--vehicle', type=str, help='name of the vehicle to generate mobility file for')
mobility_file_parser.add_argument('--minimum_speed', type=float, help='minimum speed for the mobility file, default 0.01', default=0.01)
mobility_file_parser.add_argument('--plot', action='store_true', help='plot the data from generated mobility file', default=False)


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

