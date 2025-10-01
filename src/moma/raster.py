'''
Raster module
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


import logging
from abc import ABC

import numpy as np  # Used for raster data array operations
import matplotlib as mpl  # Used for color maps definition
import matplotlib.pyplot as plt  # Used for generating raster plots
import shapely.geometry as shpgeo  # Used for no-go zones definition
import rasterio  # Used for raster data import
import json
from rasterio.enums import Resampling  # Used for rasters resampling operations
from matplotlib.ticker import MultipleLocator  # Used for minor grid ticks
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

log = logging.getLogger("MOMA")

class Raster(ABC):
    """Raster
    """

    COLORMAPS = ["RdYlBu"]

    colormap = "RdYlBu"

    RESAMPLING_METHODS = {
        "average": Resampling.average,
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest
    }

    def set_colormap(self, key):
        """Set raster colormap

        Args:
            key (string): Colormap name. Possible values are "RdYlBu". Defaults to "RdYlBu".
        """

        self.colormap = getattr(mpl.cm, self.COLORMAPS.index(
            key)) if key in self.COLORMAPS else self.COLORMAP_DEFAULT
        
    def return_empty_copy(self):
        """Return empty copy of raster
        """
        new_raster = self.__class__(self.path)
        new_raster.data = np.zeros_like(self.data)
        new_raster.dtype = self.dtype

        new_raster.width = self.width
        new_raster.height = self.height

        new_raster.resolution_x = self.resolution_x
        new_raster.resolution_y = self.resolution_y

        new_raster.x_min = self.x_min
        new_raster.x_max = self.x_max
        new_raster.y_min = self.y_min
        new_raster.y_max = self.y_max
        
        new_raster.crs = self.crs
        new_raster.transform = self.transform

        return new_raster

    def plot(self, 
             data=None, rgb=None,
             colorbar=None, clim=None, discrete_colormap=None, reverse_colormap=True,
             title=None,
             export_path=None, show=True, export=False, save_maps=False, hide_utm=False):
        """Show raster
        """

        # Use stored data in raster if no data is provided
        if data is not None:
            plot_data = data
        elif rgb is None:
            plot_data = self.data

        # Open new figure
        (figure, axes) = plt.subplots(1, 1)

        # Set and invert colormap
        cmap = mpl.cm.get_cmap(self.colormap + ("_r" if reverse_colormap else ""))

        # Normalise colormap range. Defaults to a range of three standard deviations around the mean value.
        def norm(data):
            return mpl.colors.Normalize(
                vmin=np.mean(data) - 1.25 * np.std(data),
                vmax=np.mean(data) + 1.25 * np.std(data))

        # Draw raster data
        extent = [self.x_min, self.x_max, self.y_min, self.y_max]
        if rgb is None:
            if plot_data.ndim == 3:
                means = np.mean(plot_data, axis=0)
            else:
                means = plot_data

            image = axes.imshow(
                    means,
                    extent=extent,
                    cmap=cmap,
                    norm=norm(means))
        else:
            if discrete_colormap is None:
                image = axes.imshow(rgb, extent=extent)
            else:
                col_dict = discrete_colormap[1]
                discrete_colormap = discrete_colormap[0]

                # Define a description for each colour category - order matters!
                labels = [str(value) for value in col_dict.values()]

                # Generate normalization bins
                normalization_bins = np.sort([*col_dict.keys()]) + 0.5
                normalization_bins = np.insert(normalization_bins, 0,
                                            np.min(normalization_bins) - 1.0)
                
                # Set up normaliser and formatter
                normalization = mpl.colors.BoundaryNorm(normalization_bins,
                                                        len(labels),
                                                        clip=True)
                colorbar_formatter = mpl.ticker.FuncFormatter(
                    lambda x, pos: labels[normalization(x)])
                
                image = axes.imshow(rgb, extent=extent, cmap=discrete_colormap, norm=normalization)

        # Add minor grid showing the cells for small sites
        if self.width + self.height < 200: 
            axes.xaxis.set_minor_locator(MultipleLocator(5))
            axes.yaxis.set_minor_locator(MultipleLocator(5))
            axes.grid(which='major', linestyle='--')
        if hide_utm:
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xlabel("")
            axes.set_ylabel("")
        else:
            # Add UTM ticks
            axes.grid(True, which='both')
            axes.set_xlabel("UTM Longitude [m]")
            axes.set_ylabel("UTM Latitude [m] ")
        
        # Add map ticks
        axes.secondary_xaxis('bottom' if hide_utm else 'top',   functions=(lambda x: x - self.x_min, lambda x: x + self.x_min)).set_xlabel("Length [m]")
        axes.secondary_yaxis('left' if hide_utm else 'right', functions=(lambda y: y - self.y_min, lambda y: y + self.y_min)).set_ylabel("Length [m]")

        # Disable scientific notation for ticks
        axes.ticklabel_format(style="plain", useOffset=False)

        if colorbar is not None or discrete_colormap is not None:
            divider = make_axes_locatable(axes)
            pad_val = 0.5 if hide_utm else 1
            pad_val = pad_val if discrete_colormap is None else pad_val - 0.3
            cax = divider.new_vertical(size = '5%', pad = pad_val)
            figure.add_axes(cax)
        # Add color_bar
        if colorbar is not None:

            color_bar = figure.colorbar(image, cax=cax, orientation='horizontal')
            color_bar.set_label(colorbar, labelpad=-55)
            color_bar.minorticks_on()
            # Adjust clim
            if clim is not None:
                image.set_clim(clim[0], clim[1])
        elif discrete_colormap is not None:
            # Set spacing between color_bar entries
            spacing = normalization_bins[1:] - normalization_bins[:-1]
            ticks = normalization_bins[:-1] + spacing / 2

            color_bar = figure.colorbar(image, cax=cax, orientation='horizontal', format=colorbar_formatter, ticks=ticks)

            color_bar.ax.xaxis.set_ticks_position('top')
            color_bar.ax.xaxis.set_label_position('top')
            tick_spacing = ticks[1] - ticks[0]
            for label in color_bar.ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('left')
                x, y = label.get_position()
                label.set_position((x - 0.5*tick_spacing, y))
        else:
            color_bar = None

        # Add title
        if title is not None:
            axes.title(title, pad=20)

        # Set horizontal and vertical labels
        # plt.tight_layout()

        if show:
            plt.show()

        if export:
            log.info(f"Exporting raster plot to {export_path}.")
            plt.savefig(export_path, format="pdf", bbox_inches="tight")

        # Saves the raster as a GeoTIFF file if save_maps is True
        if save_maps:
            export_path = export_path.with_suffix(".tif")
            log.info(f"Exporting raster plot to {export_path}.")
            if rgb is not None:
                # Convert RGB data to HSV hue
                hsv_img = mcolors.rgb_to_hsv(rgb[:, :, :3])  
                hue_map = hsv_img[:, :, 0]

                if discrete_colormap is not None:
                    hue_map = np.round(hue_map*255)

                    # Converts label rgb to hue 
                    label_to_hue = {}
                    for rgb_int, label in col_dict.items():
                        r = (rgb_int >> 16) & 0xFF
                        g = (rgb_int >> 8) & 0xFF
                        b = rgb_int & 0xFF

                        rgb = np.array([r, g, b]) / 255.0
                        hsv = mcolors.rgb_to_hsv(rgb)
                        hue = np.round(hsv[0]*255)

                        label_to_hue[label] = hue

                    with open(export_path.with_suffix(".json"), "w") as json_file:
                        json.dump(label_to_hue, json_file, indent=4)

                self.save(export_path, data=hue_map)
            elif data is not None:
                self.save(export_path, data=means)
            else:
                self.save(export_path, data=means)

        if export or show or save_maps:
            plt.close()
        else:
            return (figure, axes, color_bar, image)

    def save(self, path, data=None):
        """Save raster to disk
        Args:
            path (str): Path to save raster to
            data (np.ndarray, optional): Raster data to save. Defaults to None, which uses the raster's internal stored data.
        """

        if data is None:
            data = self.data
        
        # Expand data dimensions if it is a single band raster
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Open new raster handle
        raster = rasterio.open(path,
                               "w",
                               driver="GTiff",
                               width=self.width,
                               height=self.height,
                               count=data.shape[0],
                               dtype=data.dtype,
                               crs=self.crs,
                               transform=self.transform)

        # Export raster and close handle
        raster.write(data)
        raster.close()


class RasterMono(Raster):
    """Single-band raster
    """

    def __init__(self, path):

        log.info(f"Initialising new single-channel raster from file {path}.")

        # Set raster path
        self.path = path

        # Create RasterIO raster handle
        raster = rasterio.open(path, "r")

        # Get raster data and data type
        self.data = raster.read()
        self.dtype = raster.dtypes

        # Get raster dimensions
        self.width = raster.width
        self.height = raster.height

        # Get raster resolution
        self.resolution_x = raster.transform[0]
        self.resolution_y = -raster.transform[4]

        # Get raster boundaries
        self.x_min, self.y_min = raster.transform * (0, raster.height)
        self.x_max, self.y_max = raster.transform * (raster.width, 0)

        # Get coordinate reference system
        self.crs = raster.crs
        self.transform = raster.transform

        # Close Rasterio raster handle
        raster.close()

    def resample(self, factor, resampling="average"):
        """Resample raster

        Args:
            factor (float): Scaling factor
            resampling (str, optional): Resampling algorithm. Valid options are "average", "bilinear", "cubic", "lanczos" or "nearest". Defaults to "average".
        """

        # Compute new raster size and shape
        height = int(self.height * factor)
        width = int(self.width * factor)
        shape = (height, width)
        
        # Open RasterIO raster handle
        with rasterio.open(self.path, "r") as raster:

            # Resample original raster data
            self.data = raster.read(
                out_shape=shape,
                resampling=self.RESAMPLING_METHODS.get(resampling))

            # Update transform
            self.transform = self.transform * raster.transform.scale(
                (self.width / shape[-1]), (self.height / shape[-2]))

        # Update raster information
        self.height = height
        self.width = width
        
        # Get new raster resolution
        self.resolution_x = self.transform[0]
        self.resolution_y = -self.transform[4] 

    def add_cylinder(self, coordinate_x: float, coordinate_y: float,
                     radius: float, height: float):
        """Add cylindrical feature

        Args:
            coordinate_x (float): Cylinder center x coordinate
            coordinate_y (float): Cylinder center y coordinate
            radius (float): Cylinder radius
            height (float): Cylinder height
        """

        circle = shpgeo.Point(coordinate_x, coordinate_y).buffer(radius)

        iterator = np.nditer(self.data, flags=["multi_index"])
        for _ in iterator:
            i, j = iterator.multi_index
            coordinates = self.coordinates(i, j)
            if circle.contains(shpgeo.Point(coordinates)):
                self.data[i, j] = height

    def coordinates(self, i, j):
        """Retrieve raster coordinates for index pair (i, j).

        Args:
            i (int): X coordinate index
            j (int): Y coordinate index

        Returns:
            double: Raster value at cell (i, j)
        """

        return self.transform * (i, j)


class RasterRGBA(Raster):
    """4-band raster
    """

    def __init__(self, path):

        log.info(f"Initialising new four-channel raster from file {path}.")

        # Set raster path
        self.path = path

        # Create RasterIO raster handle
        raster = rasterio.open(path, "r")

        # Get raster data and data type
        self.data = raster.read()       
        self.dtype = raster.dtypes

        # Get raster dimensions
        self.width = raster.width
        self.height = raster.height

        # Get raster resolution
        self.resolution_x = raster.transform[0]
        self.resolution_y = -raster.transform[4]

        # Get raster boundaries
        self.x_min, self.y_min = raster.transform * (0, raster.height)
        self.x_max, self.y_max = raster.transform * (raster.width, 0)

        # Get coordinate reference system
        self.crs = raster.crs
        self.transform = raster.transform

        # Close RasterIO raster handle
        raster.close()

    def resample(self, factor, resampling="average"):
        """Resample raster

        Args:
            factor (float): Scaling factor
            resampling (str, optional): Resampling algorithm. Valid options are "average", "bilinear", "cubic", "lanczos" or "nearest". Defaults to "average".
        """

        log.info(
            f"Resampling raster by a factor {factor} using resampling algorithm {resampling}."
        )

        # Compute new raster size and shape
        height = int(self.height * factor)
        width = int(self.width * factor)
        shape = (height, width)

        # Create RasterIO raster handle
        with rasterio.open(self.path, "r") as raster:

            # Resample RGBA data for all 4 channels
            self.data = raster.read(out_shape=shape, resampling=self.RESAMPLING_METHODS.get(resampling))

            # Update transform
            self.transform = self.transform * raster.transform.scale(
                (self.width / shape[-1]), (self.height / shape[-2]))

        # Update raster information
        self.height = height
        self.width = width
        
        # Get new raster resolution
        self.resolution_x = self.transform[0]
        self.resolution_y = -self.transform[4]

    def get_indices(self, lon, lat):

        # Create RasterIO raster handle
        with rasterio.open(self.path, "r") as raster:
            (py, px) = raster.index(lon, lat)
        
        return (py, -px)