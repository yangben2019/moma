'''
Generators module
'''

'''
 #############################################################################
 #                            _     _     _     _                            #
 #                           / \   / \   / \   / \                           #
 #                          ( M ) ( O ) ( M ) ( A )                          #
 #                           \_/   \_/   \_/   \_/                           #
 #                                                                           #
 #               MOMA: Ground vehicles mobility mapping toolkit              #
 #############################################################################

MIT License

Copyright (c) 2024 Dario Sirangelo

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
'''


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

plt.style.use(importlib.resources.path(
        "moma", "data") / pathlib.Path(
        "matplotlib/moma.mplstyle"))

class TestMapGenerator:

    def fill_slopes(Z, sections_start_and_end):
        for i in range(1, len(sections_start_and_end[0]) - 2, 2):

            x1 = sections_start_and_end[0][i] - 1
            x2 = sections_start_and_end[0][i + 1] + 1

            Z[:, x1:x2] = np.linspace(Z[:, x1], Z[:, x2], x2 - x1).T

        for j in range(1, len(sections_start_and_end[1]) - 2, 2):

            y1 = sections_start_and_end[1][j] - 1
            y2 = sections_start_and_end[1][j + 1] + 1

            Z[y1:y2, :] = np.linspace(Z[y1, :], Z[y2, :], y2 - y1)

        return Z

    def fill_pad(Z, start_pad, transition_pad, sections_start_and_end):
        # Left
        values = Z[np.round((sections_start_and_end[1][:-1] + sections_start_and_end[1][1:])/2).astype(int), start_pad + transition_pad]

        Z[:sections_start_and_end[1][1], :start_pad] = values[0]
        for idx in range(1, len(values) - 1):
            Z[sections_start_and_end[1][idx]:sections_start_and_end[1][idx + 1], :start_pad] = values[idx]
        Z[sections_start_and_end[1][-2]:, :start_pad] = values[-1]

        # Right
        values = Z[np.round((sections_start_and_end[1][:-1] + sections_start_and_end[1][1:])/2).astype(int), - (start_pad + transition_pad + 1)]

        Z[:sections_start_and_end[1][1], -start_pad:] = values[0]
        for idx in range(1, len(values) - 1):
            Z[sections_start_and_end[1][idx]:sections_start_and_end[1][idx + 1], -start_pad:] = values[idx]
        Z[sections_start_and_end[1][-2]:, -start_pad:] = values[-1]

        # Bottom
        values = Z[start_pad + transition_pad, np.round((sections_start_and_end[0][:-1] + sections_start_and_end[0][1:])/2).astype(int)]

        Z[:start_pad, :sections_start_and_end[0][1]] = values[0]
        for idx in range(1, len(values) - 1):
            Z[:start_pad, sections_start_and_end[0][idx]:sections_start_and_end[0][idx + 1]] = values[idx]
        Z[:start_pad, sections_start_and_end[0][-2]:] = values[-1]

        # Top
        values = Z[-(start_pad + transition_pad + 1), np.round((sections_start_and_end[0][:-1] + sections_start_and_end[0][1:])/2).astype(int)]

        Z[-start_pad:, :sections_start_and_end[0][1]] = values[0]
        for idx in range(1, len(values) - 1):
            Z[-start_pad:, sections_start_and_end[0][idx]:sections_start_and_end[0][idx + 1]] = values[idx]
        Z[-start_pad:, sections_start_and_end[0][-2]:] = values[-1]

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

    def generate_test_map(scale_map=1, seed=42):

        max_slope_percent = 30

        # Gives test map like the following:

        Slope_length = [[50 for _ in range(9)], [50 for _ in range(9)]]   # Length of the slope sections in x and y direction
        Flat_length  = [[50 for _ in range(10)], [50 for _ in range(10)]] # Length of the flat  sections in x and y direction

        Flat_sections = [len(Flat_length[0]), len(Flat_length[1])]

        elevation_grid = np.zeros((Flat_sections[0], Flat_sections[1]))

        # Initialize a PRNG key.
        key = jrand.PRNGKey(seed)
        # Create a key for each cell in the 2D grid.
        cell_keys = jrand.split(key, Flat_sections[0] * Flat_sections[1]).reshape(Flat_sections[0], Flat_sections[1], -1)

        for i in range(Flat_sections[0]):
            for j in range(Flat_sections[1]):
                if i == 0 and j == 0:
                    pass
                elif i == 0:
                    max_diff = Slope_length[1][j-1] * max_slope_percent/ 100
                    max = elevation_grid[i, j-1] + max_diff
                    min = elevation_grid[i, j-1] - max_diff

                    elevation_grid[i, j] = jrand.uniform(cell_keys[i, j, :], shape=(1,), minval=min, maxval=max)[0]
                elif j == 0:
                    max_diff = Slope_length[0][i-1] * max_slope_percent/ 100
                    max = elevation_grid[i-1, j] + max_diff
                    min = elevation_grid[i-1, j] - max_diff

                    elevation_grid[i, j] = jrand.uniform(cell_keys[i, j, :], shape=(1,), minval=min, maxval=max)[0]
                else:
                    diff_i = Slope_length[0][i-1] * max_slope_percent/ 100
                    diff_j = Slope_length[1][j-1] * max_slope_percent/ 100
                    max = np.min([elevation_grid[i-1, j] + diff_i, elevation_grid[i, j-1] + diff_j])
                    min = np.max([elevation_grid[i-1, j] - diff_i, elevation_grid[i, j-1] - diff_j])

                    elevation_grid[i, j] = jrand.uniform(cell_keys[i, j, :], shape=(1,), minval=min, maxval=max)[0]

        # Make min value equal to 0
        elevation_grid -= np.min(elevation_grid)

        # Scale the map
        elevation_grid *= scale_map

        resolution = 4           # Pixel/meter

        map_size = (sum(Flat_length[0]) + sum(Slope_length[0]), sum(Flat_length[1]) + sum(Slope_length[1]))

        sections = [
            [0, Flat_length[0][0]],
            [0, Flat_length[1][0]],
        ]

        for i in range(1, Flat_sections[0]):
            sections[0].append(sections[0][-1] + Slope_length[0][i-1])
            sections[0].append(sections[0][-1] + Flat_length[0][i])

        for j in range(1, Flat_sections[1]):
            sections[1].append(sections[1][-1] + Slope_length[1][j-1])
            sections[1].append(sections[1][-1] + Flat_length[1][j])

        sections[0] = np.array(sections[0])
        sections[1] = np.array(sections[1])
        sections_start_and_end = [sections[0]*resolution, sections[1]*resolution]

        Z = np.zeros((map_size[1]*resolution, map_size[0]*resolution))

        for i in range(Flat_sections[0]):
            for j in range(Flat_sections[1]):
                Z[sections_start_and_end[1][j*2]:sections_start_and_end[1][j*2+1], sections_start_and_end[0][i*2]:sections_start_and_end[0][i*2+1]] = elevation_grid[i, j]

        Z = TestMapGenerator.fill_slopes(Z, sections_start_and_end)

        # Create the flat starting spots for the vehicle
        transition_pad = 15
        start_pad = 10

        map_size = (map_size[0] + 2*(transition_pad + start_pad), map_size[1] + 2*(transition_pad + start_pad))

        sections[0] += transition_pad + start_pad
        sections[1] += transition_pad + start_pad

        transition_pad = round(transition_pad * resolution)
        start_pad = round(start_pad * resolution)

        sections_start_and_end[0] += transition_pad + start_pad
        sections_start_and_end[1] += transition_pad + start_pad

        new_Z = np.zeros((map_size[1]*resolution, map_size[0]*resolution))

        new_Z[
            (transition_pad + start_pad):-(transition_pad + start_pad),
            (transition_pad + start_pad):-(transition_pad + start_pad)
        ] = Z
        Z = new_Z

        Z = TestMapGenerator.fill_pad(Z, start_pad, transition_pad, sections_start_and_end)

        return Z, map_size, resolution, sections

    def generate_test_map_uncertainty(map_size, resolution, uncertainty=0.02):
        Z = np.ones((map_size[1]*resolution, map_size[0]*resolution))*uncertainty

        return Z

    def smooth(Z, sigma, truncate):
        return gaussian_filter(Z, sigma=sigma, truncate=truncate)

    def plot_surface(output_dir, elevation, sections, map_size, show_plot=False, subname=""):
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[8, 10], wspace=0.4)

        # 2D plot
        ax1 = fig.add_subplot(gs[0])
        img = ax1.imshow(elevation, cmap="RdYlBu_r", extent=[0, map_size[0], 0, map_size[1]])

        ax1.vlines(sections[0], ymin=sections[1][0], ymax=sections[1][-1], color="grey", linestyle="--", linewidth=0.5)
        ax1.hlines(sections[1], xmin=sections[0][0], xmax=sections[0][-1], color="grey", linestyle="--", linewidth=0.5)

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
        surf = ax2.plot_surface(X, Y, np.flip(elevation, axis=0), cmap="RdYlBu_r")#, edgecolor="k")
        ax2.set_title("Elevation Map (3D Surface)")
        ax2.set_xlabel("[m]")
        ax2.set_ylabel("[m]")
        ax2.set_zlabel("[m]")

        ax2.view_init(30, 225)

        ax2.set_box_aspect((1,1,1), zoom=0.9)

        fig.savefig(output_dir / f"test_map{subname}.pdf", format="pdf", bbox_inches="tight")#, pad_inches=0.5)

        if show_plot:
            plt.show()
        plt.close(fig)

    def save_data_for_moma(raster_dir, elevation, uncertainty_map, map_size, resolution, friction_coefficient, friction_coefficient_std, diffuse_map, coordinates_lower_left=(542616, 6217995)):
        # Save elevation raster
        print(f"Save rasters in folder: {raster_dir}...")


        # Coordinates of lower left corner
        # Default coordinates is Himmelbjerget
        lat = coordinates_lower_left[0]  # E
        long = coordinates_lower_left[1]  # N

        # Site.yaml file
        with open(raster_dir / ".." / "site.yaml", "w") as site:
            site.write(f"""
    DataSource: DataforsyningenFriction
    Coordinates: [{lat}, {long}, {lat + map_size[0]}, {lat + map_size[0]}] # [W, S, E, N]
    NoGoZones:
    #   - [x1, x2, y1, y2, (Description)]: Box
    #   - [x, y, r, (Description)]: Circle
    """)

        coor_string = f"{lat}_{long}_{lat + map_size[0]}_{lat + map_size[0]}"

        # Reduce elevation to 1x1m grid and add extra dimension
        elevation   = elevation[resolution//2::resolution, resolution//2::resolution][np.newaxis, :]
        uncertainty_map = uncertainty_map[resolution//2::resolution, resolution//2::resolution][np.newaxis, :]

        elevation_combined = np.concatenate((elevation, uncertainty_map))

        transform = Affine(1, 0, lat, 0, -1, map_size[1] + long)
        path = raster_dir / f"elevationmono_1m_clipped_{coor_string}.tif"
        save_new_raster(elevation_combined, path, transform)

        # Save diffuse map as areal view
        diffuse_map = np.stack((
            diffuse_map[::resolution, ::resolution, 0],
            diffuse_map[::resolution, ::resolution, 1],
            diffuse_map[::resolution, ::resolution, 2],
            np.zeros_like(diffuse_map[::resolution, ::resolution, 0])
            ), axis=0).astype(np.uint8)
        path = raster_dir / f"aerial_1m_clipped_{coor_string}.tif"
        save_new_raster(diffuse_map, path, transform)

        # Create land use raster
        land_use = np.concatenate((np.ones_like(elevation) * 0,
                                np.ones_like(elevation) * 127,
                                np.ones_like(elevation) * 20,
                                np.zeros_like(elevation)))

        path = raster_dir / f"landuse_1m_clipped_{coor_string}.tif"
        save_new_raster(land_use, path, transform)

        # Create soil type raster with hard terrain
        soil_type = np.concatenate((np.ones_like(elevation) * 0,
                                    np.ones_like(elevation) * 0,
                                    np.ones_like(elevation) * 0,
                                    np.zeros_like(elevation)))

        soils = {
            0: {
                "label": "hard clay",
                "color": [0, 0, 0]
            },
            1: {
                "label": "hard gravel",
                "color": [50, 50, 50]
            }
        }

        shape = soil_type.shape[1:]
        for i in range(3):
            soil_type[i, :int(shape[0]/2), :] = soils[0]["color"][i]
            soil_type[i, int(shape[0]/2):, :] = soils[1]["color"][i]

        path = raster_dir / f"soiltype_1m_clipped_{coor_string}.tif"
        save_new_raster(soil_type, path, transform)

        # Saves soil to use in Chrono
        soil_export = {
            "shape": shape,
            "resolution": 1,
            "labels": soils,
            "major": "row"
        }
        soil_data_export = np.zeros(shape)
        for i, soil in soils.items():

            soil_data_export[
                (soil_type[0, :, :] == soil["color"][0])  &
                (soil_type[1, :, :] == soil["color"][1])  &
                (soil_type[2, :, :] == soil["color"][2])
            ] = i

        soil_export["data"] = list(soil_data_export.flatten())

        with open(raster_dir / f"soiltype_1m_clipped_{coor_string}.json", "w") as json_file:
            json.dump(soil_export, json_file)

        # Create moisture content raster
        moisture_content = np.concatenate((np.ones_like(elevation) * 150,
                                        np.ones_like(elevation) * 150,
                                        np.ones_like(elevation) * 255,
                                        np.zeros_like(elevation),
                                        np.zeros_like(elevation)))

        path = raster_dir / f"moisturecontent_1m_clipped_{coor_string}.tif"
        save_new_raster(moisture_content, path, transform)

        # Create friction coefficient raster
        friction_low = 0.4
        friction_high = 0.6
        friction = np.ones_like(elevation)
        friction[0, (soil_type[0, :, :] == 0)  & (soil_type[1, :, :] == 0)  & (soil_type[2, :, :] == 0)]  = friction_low
        friction[0, (soil_type[0, :, :] == 50) & (soil_type[1, :, :] == 50) & (soil_type[2, :, :] == 50)] = friction_high
        friction_coefficient_map = np.concatenate((friction, np.ones_like(elevation) * friction_coefficient_std))

        path = raster_dir / f"friction_coefficient_1m_clipped_{coor_string}.tif"
        save_new_raster(friction_coefficient_map, path, transform)

    def metadata(elev_min, elev_max, map_size, friction_coefficient):
        """
        Generate metadata dictionary for the elevation map.
        """
        meta = {
            "description": "slope test map",
            "elevation": {
                "max": float(elev_max),
                "min": float(elev_min)
            },
            "map_size": {
                "width": float(map_size[0]),
                "height": float(map_size[1])},
            "friction_coefficient": float(friction_coefficient),
        }
        return meta

    def generate_diffuse_map(elevation):
        """
        Generate a diffuse map from the elevation data by interpret the slopes in the x and y directions as R and G channels for RGB color.
        """
        slope_x, slope_y = np.gradient(elevation)

        slope_x = TestMapGenerator.smooth(slope_x, sigma=2, truncate=15)
        slope_y = TestMapGenerator.smooth(slope_y, sigma=2, truncate=15)

        x_min, x_max = np.min(slope_x), np.max(slope_x)
        y_min, y_max = np.min(slope_y), np.max(slope_y)

        x_normalized = 230 * (slope_x - x_min) / (x_max - x_min) + 255 - 230
        y_normalized = 230 * (slope_y - y_min) / (y_max - y_min) + 255 - 230

        x_normalized = x_normalized.astype(np.uint8)
        y_normalized = y_normalized.astype(np.uint8)

        return np.dstack((x_normalized, y_normalized, np.zeros_like(x_normalized)))

    def save_diffuse_map(output_dir, elevation, subname=""):
        # Save the diffuse map as a .png file
        diffuse_map = TestMapGenerator.generate_diffuse_map(elevation)
        diffuse_map_img = Image.fromarray(diffuse_map)

        diffuse_map_img.save(output_dir / f"map-diffuse{subname}.png")

    def save_bitmap(output_dir, elevation, map_size, friction_coefficient, subname="", output_resolution=6):
        # Save the elevation data as .bmp file for use in Chrono

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
            json.dump(TestMapGenerator.metadata(elev_min, elev_max, map_size, friction_coefficient), f, indent=4)

    def save_obj(output_dir, elevation, map_size, resolution, friction_coefficient, subname="", output_resolution=6):
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
                vertices.append(f"v {j/resolution} {i/resolution} {elevation_out[i, j]}")

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
            json.dump(TestMapGenerator.metadata(elev_min, elev_max, map_size, friction_coefficient), f, indent=4)

    def slope_hist(output_dir, elevation, resolution, show_plot=False):
        # Calculate the slope of the elevation map
        gradient_x, gradient_y = np.gradient(elevation)
        slope = np.sqrt(gradient_x**2 + gradient_y**2).flatten()*resolution*100

        # Plot the histogram of slopes
        plt.figure(figsize=(10, 6))
        plt.hist(slope, bins=100, density=True, alpha=0.6, color='g')
        plt.title("Slope in steepest direction")
        plt.xlabel("Slope $\%$")
        plt.ylabel("Density")

        plt.tight_layout()

        plt.savefig(output_dir / "test_map_slope_histogram.png", bbox_inches="tight")

        if show_plot:
            plt.show()
        plt.close()

    def main_generate(
            create_rasters=False,
            base_output_dir=None,
            plot_additional_files=False,
            scale_map=1,
            N_test_maps=1,
            std_elevation=0.00,
            friction_coefficient=0.5,
            friction_coefficient_std=0.00,
            map_type="bmp",
            output_resolution=6,
            return_map=False):

        # If map is not returned as an array, create the output directory
        if base_output_dir is None:
            base_output_dir = pathlib.Path(".") / "examples" / "sites" / "testmap"
        else:
            base_output_dir = pathlib.Path(base_output_dir)


        raster_dir = base_output_dir / "rasters"
        output_dir = base_output_dir

        if create_rasters:
            raster_dir.mkdir(parents=True, exist_ok=True)

        if not return_map:
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Output directory: \n\tBitmaps: {output_dir}\n\tRasters: {raster_dir}")

        print("Generating test map...")
        elevation, map_size, resolution, sections = TestMapGenerator.generate_test_map(scale_map)
        smoothed_elevation = TestMapGenerator.smooth(elevation, sigma=15, truncate=60)
        if return_map and not create_rasters:
            return smoothed_elevation, resolution

        uncertainty_map = TestMapGenerator.generate_test_map_uncertainty(map_size, resolution, uncertainty=std_elevation)

        if plot_additional_files:
            print("Plotting test map...")
            TestMapGenerator.plot_surface(output_dir, elevation, sections, map_size, show_plot=False)
            TestMapGenerator.plot_surface(output_dir, smoothed_elevation, sections, map_size, show_plot=False, subname="_smoothed")

            print("Plotting slope histogram...")
            TestMapGenerator.slope_hist(output_dir, smoothed_elevation, resolution)

        # Save the diffuse map as a .png file for terrain texture

        if create_rasters:
            print("Saving test map for MOMA...")
            diffuse = TestMapGenerator.generate_diffuse_map(smoothed_elevation)
            TestMapGenerator.save_data_for_moma(raster_dir, smoothed_elevation, uncertainty_map, map_size, resolution, friction_coefficient, friction_coefficient_std, diffuse)

            if return_map:
                return smoothed_elevation, resolution

        print("Generation samples for Chrono...")
        smoothed_sampled_elevation = np.array(LatinHyperCube(smoothed_elevation, uncertainty_map, N_test_maps, seed=21))
        friction_coefficient_sampled = np.array(LatinHyperCube(np.array([[friction_coefficient]]), np.array([[friction_coefficient_std]]),
                                                            N_test_maps, seed=23)).reshape(N_test_maps)

        if plot_additional_files:
            print("Plotting one test map with uncertainty...")
            TestMapGenerator.plot_surface(output_dir, smoothed_sampled_elevation[0, :, :], sections, map_size, show_plot=False, subname="_smoothed_sampled")

        # Save a single map without
        if map_type == "bmp":
            TestMapGenerator.save_bitmap(output_dir, smoothed_elevation, map_size, friction_coefficient=friction_coefficient, subname="-without", output_resolution=output_resolution)
        elif map_type == "obj":
            TestMapGenerator.save_obj(output_dir, smoothed_elevation, map_size, resolution, friction_coefficient=friction_coefficient, subname="-without", output_resolution=output_resolution)
        else:
            raise ValueError("Map type {map_type} not supported, must be either 'bmp' or 'obj'.")

        # Save the diffuse map as a .png file for terrain texture
        TestMapGenerator.save_diffuse_map(output_dir, smoothed_elevation, subname="-without")

        for i in range(smoothed_sampled_elevation.shape[0]):
            print(f"Saving test map {i} of {smoothed_sampled_elevation.shape[0]}...")
            if map_type == "bmp":
                TestMapGenerator.save_bitmap(output_dir, smoothed_sampled_elevation[i, :, :], map_size, friction_coefficient=friction_coefficient_sampled[i], subname=f"-{i}", output_resolution=output_resolution)
            elif map_type == "obj":
                TestMapGenerator.save_obj(output_dir, smoothed_sampled_elevation[i, :, :], map_size, resolution, friction_coefficient=friction_coefficient_sampled[i], subname=f"-{i}", output_resolution=output_resolution)

            # Save the diffuse map as a .png file for terrain texture
            TestMapGenerator.save_diffuse_map(output_dir, smoothed_sampled_elevation[i, :, :], subname=f"-{i}")


    if __name__ == "__main__":
        main_generate(create_rasters=True, N_test_maps=2, plot_additional_files=True, return_map=False)

class TestPathsGenerator:

    def get_paths(map):

        max_x, max_y = map.shape
        distance_to_edge = 7.5
        cell_offset = 2.5
        block_size = 50
        middle_block = ((max_x - block_size) / block_size) / 2 - 1 # Zero indexed

        # Zero indexed block numbers
        non_diagonal_blocks = [2, 3, 5, 6, 12, 13, 15, 16] # Block number x and y direction
        diagonal_blocks = [1, 2, 3, 4] # Number of block away from the true diagonal

        # [x1, y1, x2, y2]
        paths = {
            "N": [
                [
                    (x_block + 1)*block_size + np.sign(middle_block - x_block)*cell_offset,
                    distance_to_edge,
                    (x_block + 1)*block_size + np.sign(middle_block - x_block)*cell_offset,
                    max_y - distance_to_edge,
                ] for x_block in non_diagonal_blocks
            ],
            "S": [
                [
                    (x_block + 1)*block_size + np.sign(middle_block - x_block)*cell_offset,
                    max_y - distance_to_edge,
                    (x_block + 1)*block_size + np.sign(middle_block - x_block)*cell_offset,
                    distance_to_edge,
                ] for x_block in non_diagonal_blocks
            ],
            "E": [
                [
                    distance_to_edge,
                    (y_block + 1)*block_size + np.sign(middle_block - y_block)*cell_offset,
                    max_x - distance_to_edge,
                    (y_block + 1)*block_size + np.sign(middle_block - y_block)*cell_offset,
                ] for y_block in non_diagonal_blocks
            ],
            "W": [
                [
                    max_x - distance_to_edge,
                    (y_block + 1)*block_size + np.sign(middle_block - y_block)*cell_offset,
                    distance_to_edge,
                    (y_block + 1)*block_size + np.sign(middle_block - y_block)*cell_offset,
                ] for y_block in non_diagonal_blocks
            ],
            "NE": [
                [distance_to_edge, distance_to_edge, max_x - distance_to_edge, max_y - distance_to_edge]
            ] + [
                [
                    distance_to_edge,
                    blocks*block_size + distance_to_edge,
                    max_x - (blocks*block_size + distance_to_edge),
                    max_y - distance_to_edge,
                ] for blocks in diagonal_blocks
            ] + [
                [
                    blocks*block_size + distance_to_edge,
                    distance_to_edge,
                    max_x - distance_to_edge,
                    max_y - (blocks*block_size + distance_to_edge),
                ] for blocks in diagonal_blocks
            ],
            "SW": [
                [max_x - distance_to_edge, max_y - distance_to_edge, distance_to_edge, distance_to_edge]
            ] + [
                [
                    max_x - (blocks*block_size + distance_to_edge),
                    max_y - distance_to_edge,
                    distance_to_edge,
                    blocks*block_size + distance_to_edge,
                ] for blocks in diagonal_blocks
            ] + [
                [
                    max_x - distance_to_edge,
                    max_y - (blocks*block_size + distance_to_edge),
                    blocks*block_size + distance_to_edge,
                    distance_to_edge,
                ] for blocks in diagonal_blocks
            ],
            "SE": [
                [distance_to_edge, max_y - distance_to_edge, max_x - distance_to_edge, distance_to_edge]
            ] + [
                [
                    distance_to_edge,
                    max_y - (blocks*block_size + distance_to_edge),
                    max_x - (blocks*block_size + distance_to_edge),
                    distance_to_edge,
                ] for blocks in diagonal_blocks
            ] + [
                [
                    blocks*block_size + distance_to_edge,
                    max_y - distance_to_edge,
                    max_x - distance_to_edge,
                    blocks*block_size + distance_to_edge,
                ] for blocks in diagonal_blocks
            ],
            "NW": [
                [max_x - distance_to_edge, distance_to_edge, distance_to_edge, max_y - distance_to_edge]
            ] + [
                [
                    max_x - (blocks*block_size + distance_to_edge),
                    distance_to_edge,
                    distance_to_edge,
                    max_y - (blocks*block_size + distance_to_edge),
                ] for blocks in diagonal_blocks
            ] + [
                [
                    max_x - distance_to_edge,
                    blocks*block_size + distance_to_edge,
                    blocks*block_size + distance_to_edge,
                    max_y - distance_to_edge,
                ] for blocks in diagonal_blocks
            ],
        }
        return paths

    def add_waypoints(path, direction, x, y, map):
        elevation = map[map.shape[0] - 1 - int(np.floor(y)), int(np.floor(x))]
        path["waypoints"].append(
            {
                "map": {
                    "x": x,
                    "y": y
                },
                "elevation": elevation,
                "path": {
                    "move_direction": direction,
                }
            }
        )

    def fill_path(path, direction, map, grid_size=5):
        json_path = {"waypoints": []}
        x, y, x_end, y_end = path
        TestPathsGenerator.add_waypoints(json_path, direction, x, y, map)

        while x != x_end or y != y_end:
            x_diff = x_end - x
            y_diff = y_end - y

            x = x + np.sign(x_diff) * grid_size
            y = y + np.sign(y_diff) * grid_size
            TestPathsGenerator.add_waypoints(json_path, direction, x, y, map)

        return json_path

    def densify_paths(paths, map):

        full_paths = []

        for direction, p in paths.items():
            for path in p:
                path = TestPathsGenerator.fill_path(path, direction, map)
                path["direction"] = direction
                full_paths.append(path)

        return full_paths

    def load_map(fill_metrics=False):
        if fill_metrics:
            map, resolution = TestMapGenerator.main_generate(
                create_rasters=True,
                return_map=True
            )
        else:
            map, resolution = TestMapGenerator.main_generate(return_map=True)

        map = map[::resolution, ::resolution]
        return map

    def plot(paths, map, output_dir, grid_size=5):
        fig, ax = plt.subplots()

        extent = [0, map.shape[1], 0, map.shape[0]]
        img = ax.imshow(map[::grid_size, ::grid_size], cmap='RdYlBu_r', extent=extent)

        color_bar = fig.colorbar(img, ax=ax, pad=0.025)
        color_bar.ax.set_ylabel("Elevation (m)", rotation=270, labelpad=25)

        ax.set_xlabel("[m]")
        ax.set_ylabel("[m]")

        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.grid(True, which='both')

        for path in paths:
            if path["direction"] in ["S", "W", "NW", "NE"]:
                waypoints = path["waypoints"]
                x = [wp["map"]["x"] for wp in waypoints]
                y = [wp["map"]["y"] for wp in waypoints]
                ax.plot(x, y, color='black', )

        out_path = output_dir / "test_paths.pdf"
        print(f"Saving plotted paths to: {out_path}")
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    def save_paths(paths, output_dir):
        for idx, path in enumerate(paths):
            out_path = output_dir / f"path-{idx}"
            out_path.mkdir(exist_ok=True)

            with open(out_path / "path.json", "w") as json_file:
                json.dump(path, json_file, indent=4)

    def main_generate(output_dir, plot_paths=False, fill_metrics=False, n_samples=None, sample_reduction=None, minimum_speed=0.5, target_speed=5.0):

        # Create the output directory if it doesn't exist
        output_dir = pathlib.Path(output_dir)
        print(f"Output directory for test paths: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Calling map generation to get map")
        # Generate the test map
        map = TestPathsGenerator.load_map(fill_metrics)

        # Get the paths
        paths = TestPathsGenerator.get_paths(map)

        # Densify the paths
        paths = TestPathsGenerator.densify_paths(paths, map)

        # Plots the path on the map
        if plot_paths:
            TestPathsGenerator.plot(paths, map, output_dir)

        # Save the paths
        TestPathsGenerator.save_paths(paths, output_dir)

        # Fills the path MOMA metrics
        if fill_metrics:
            print("Fill paths with MOMA metrics")

            # Imported here to avoid circular import
            from moma.scripts.cli import calculate_mobility
            import importlib

            args = SimpleNamespace(
                site="testmap",
                vehicle="olav",
                n_samples=n_samples,
                dont_export=True,
                show_plots=False,
                path=importlib.resources.path("moma", "data") / pathlib.Path("examples"),
                path_coordinate_system="None",
                path_folder=str(output_dir),
                sample_reduction=sample_reduction,
                directions=["N", "NE", "E", "SE", "S", "SW", "W", "NW", "omni"],
                n_directions=0,
                minimum_speed=minimum_speed,
                target_speed=target_speed,
                save_maps=False,
                hide_utm=True,
            )
            calculate_mobility(args)


