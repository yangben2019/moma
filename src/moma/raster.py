"""
Raster module
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

import json
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shpgeo
import rasterio

from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.enums import Resampling


log = logging.getLogger("MOMA")


class Raster(ABC):

    COLORMAPS = ["RdYlBu"]

    colormap = "RdYlBu"

    RESAMPLING_METHODS = {
        "average": Resampling.average,
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
    }

    def set_colormap(self, key):
        """Sets the colormap for the raster.

        Args:
            key (str): Name of the colormap. Possible values are "RdYlBu". Defaults to "RdYlBu".

        Raises:
            ValueError: If the colormap name is not valid.
        """

        self.colormap = (
            getattr(mpl.cm, self.COLORMAPS.index(key))
            if key in self.COLORMAPS
            else self.COLORMAP_DEFAULT
        )

    def return_empty_copy(self):
        """Returns an empty copy of the raster object.

        Returns:
            Raster: A new raster object with the same dimensions and metadata as the original raster.
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

    def plot(
        self,
        data=None,
        rgb=None,
        colorbar=None,
        clim=None,
        discrete_colormap=None,
        reverse_colormap=True,
        title=None,
        export_path=None,
        show=True,
        export=False,
        save_maps=False,
        hide_utm=False,
    ):
        """Plots raster data.

        This function plots raster data, including both continuous and discrete
        data. It provides options for customizing the plot, such as setting a
        title, adding a colorbar, and hiding UTM ticks. The function also
        supports exporting the plot to a file or displaying it on the screen.

        Args:
            data (numpy.ndarray): Data to plot.
            rgb (numpy.ndarray): RGB data to plot.
            colorbar (str): Label for the colorbar.
            clim (tuple): Minimum and maximum values for the colormap / colorbar.
            discrete_colormap (tuple): List of colors and labels for a discrete colormap.
            reverse_colormap (bool): Whether to reverse the colormap.
            title (str): Title of the plot.
            export_path (str): Path to export the plot to.
            show (bool): Whether to display the plot.
            export (bool): Whether to export the plot to a file.
            save_maps (bool): Whether to save the raster as a GeoTIFF file.
            hide_utm (bool): Whether to hide the UTM ticks.

        Returns:
            tuple: A tuple containing the figure, axes, colorbar, and image.
        """

        # Use stored data in raster if no data is provided
        if data is not None:
            plot_data = data
        elif rgb is None:
            plot_data = self.data

        # Open new figure
        (figure, axes) = plt.subplots(1, 1)

        # Set and invert colormap
        cmap = mpl.colormaps.get_cmap(
            self.colormap + ("_r" if reverse_colormap else "")
        )

        # Normalise colormap range. Defaults to a range of three standard deviations around the mean value.
        def norm(data):
            vmin = np.mean(data) - 1.25 * np.std(data)
            vmax = np.mean(data) + 1.25 * np.std(data)

            if vmax - vmin < 0.01 * np.abs(np.mean(data)):
                vmin = np.min(data) - 0.1 * np.abs(np.min(data))
                vmax = np.max(data) + 0.1 * np.abs(np.max(data))

            return mpl.colors.Normalize(
                vmin=vmin,
                vmax=vmax,
            )

        # Draw raster data
        extent = [self.x_min, self.x_max, self.y_min, self.y_max]
        if rgb is None:
            if plot_data.ndim == 3:
                means = np.mean(plot_data, axis=0)
            else:
                means = plot_data

            image = axes.imshow(means, extent=extent, cmap=cmap, norm=norm(means))
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
                normalization_bins = np.insert(
                    normalization_bins, 0, np.min(normalization_bins) - 1.0
                )

                # Set up normaliser and formatter
                normalization = mpl.colors.BoundaryNorm(
                    normalization_bins, len(labels), clip=True
                )
                colorbar_formatter = mpl.ticker.FuncFormatter(
                    lambda x, pos: labels[normalization(x)]
                )

                image = axes.imshow(
                    rgb, extent=extent, cmap=discrete_colormap, norm=normalization
                )

        if hide_utm:
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xlabel("")
            axes.set_ylabel("")
        else:
            # Add UTM ticks
            axes.grid(True, which="both")
            axes.set_xlabel("UTM Longitude [m]")
            axes.set_ylabel("UTM Latitude [m] ")
            axes.tick_params(axis='x', labelrotation=30)

        # Add map ticks
        axes.secondary_xaxis(
            "bottom" if hide_utm else "top",
            functions=(lambda x: x - self.x_min, lambda x: x + self.x_min),
        ).set_xlabel("Length [m]")
        axes.secondary_yaxis(
            "left" if hide_utm else "right",
            functions=(lambda y: y - self.y_min, lambda y: y + self.y_min),
        ).set_ylabel("Length [m]")

        # Disable scientific notation for ticks
        axes.ticklabel_format(style="plain", useOffset=False)

        if colorbar is not None or discrete_colormap is not None:
            divider = make_axes_locatable(axes)
            pad_val = 0.5 if hide_utm else 1
            pad_val = pad_val if discrete_colormap is None else pad_val - 0.3
            cax = divider.new_vertical(size="5%", pad=pad_val)
            figure.add_axes(cax)
        # Add color_bar
        if colorbar is not None:

            color_bar = figure.colorbar(image, cax=cax, orientation="horizontal")
            color_bar.set_label(colorbar, labelpad=-55)
            color_bar.minorticks_on()
            # Adjust clim
            if clim is not None:
                image.set_clim(clim[0], clim[1])
        elif discrete_colormap is not None:
            # Set spacing between color_bar entries
            spacing = normalization_bins[1:] - normalization_bins[:-1]
            ticks = normalization_bins[:-1] + spacing / 2

            color_bar = figure.colorbar(
                image,
                cax=cax,
                orientation="horizontal",
                format=colorbar_formatter,
                ticks=ticks,
            )
            color_bar.minorticks_off()

            color_bar.ax.xaxis.set_ticks_position("top")
            color_bar.ax.xaxis.set_label_position("top")
            tick_spacing = ticks[1] - ticks[0]
            for label in color_bar.ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("left")
                x, y = label.get_position()
                label.set_position((x - 0.5 * tick_spacing, y))
        else:
            color_bar = None

        if title is not None:
            axes.set_title(title, pad=120)

        if show:
            plt.show()

        if export:
            log.info(f"Exporting raster plot to {export_path}.")
            plt.savefig(export_path, format="pdf", bbox_inches="tight")

        if save_maps:
            export_path = export_path.with_suffix(".tif")
            log.info(f"Exporting raster plot to {export_path}.")
            if rgb is not None:
                # Convert RGB data to HSV hue
                hsv_img = mcolors.rgb_to_hsv(rgb[:, :, :3])
                hue_map = hsv_img[:, :, 0]

                if discrete_colormap is not None:
                    hue_map = np.round(hue_map * 255)

                    # Converts label rgb to hue
                    label_to_hue = {}
                    for rgb_int, label in col_dict.items():
                        r = (rgb_int >> 16) & 0xFF
                        g = (rgb_int >> 8) & 0xFF
                        b = rgb_int & 0xFF

                        rgb = np.array([r, g, b]) / 255.0
                        hsv = mcolors.rgb_to_hsv(rgb)
                        hue = np.round(hsv[0] * 255)

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
        """Saves the raster data to a GeoTIFF file.

        Args:
            path (str): Path to the GeoTIFF file.
            data (numpy.ndarray): Data to save. If not provided, the data from the raster object will be used.

        Raises:
            ValueError: If the data is not in the correct shape.
        """

        if data is None:
            data = self.data

        # Expand data dimensions if it is a single band raster
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Open new raster handle
        raster = rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=self.width,
            height=self.height,
            count=data.shape[0],
            dtype=data.dtype,
            crs=self.crs,
            transform=self.transform,
        )

        # Export raster and close handle
        raster.write(data)
        raster.close()


class RasterMono(Raster):

    def __init__(self, path):
        """Initialises a single-channel raster object from a file.

        Args:
            path (str): Path to the raster file.

        Raises:
            RasterioError: If the raster file cannot be opened.
        """

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
        """Resamples the raster data to a new size.

        Args:
            factor (float): Scaling factor for the new raster.
            resampling (str): Resampling method. Valid options are "average", "bilinear", "cubic", "lanczos" or "nearest". Defaults to "average".

        Raises:
            ValueError: If the resampling method is not valid.
        """

        # Compute new raster size and shape
        height = int(self.height * factor)
        width = int(self.width * factor)
        shape = (height, width)

        # Open RasterIO raster handle
        with rasterio.open(self.path, "r") as raster:

            # Resample original raster data
            self.data = raster.read(
                out_shape=shape, resampling=self.RESAMPLING_METHODS.get(resampling)
            )

            # Update transform
            self.transform = self.transform * raster.transform.scale(
                (self.width / shape[-1]), (self.height / shape[-2])
            )

        # Update raster information
        self.height = height
        self.width = width

        # Get new raster resolution
        self.resolution_x = self.transform[0]
        self.resolution_y = -self.transform[4]

    def add_cylinder(
        self, coordinate_x: float, coordinate_y: float, radius: float, height: float
    ):
        """Adds a cylinder shape to the raster.

        Args:
            coordinate_x (float): X-coordinate of the center of the cylinder.
            coordinate_y (float): Y-coordinate of the center of the cylinder.
            radius (float): Radius of the cylinder.
            height (float): Height of the cylinder.
        """

        circle = shpgeo.Point(coordinate_x, coordinate_y).buffer(radius)

        iterator = np.nditer(self.data, flags=["multi_index"])
        for _ in iterator:
            i, j = iterator.multi_index
            coordinates = self.coordinates(i, j)
            if circle.contains(shpgeo.Point(coordinates)):
                self.data[i, j] = height

    def coordinates(self, i, j):
        """Converts raster coordinates to geographic coordinates.

        Args:
            i (int): Row index of the raster.
            j (int): Column index of the raster.

        Returns:
            tuple: Tuple containing the geographic coordinates (x, y).
        """

        return self.transform * (i, j)


class RasterRGBA(Raster):

    def __init__(self, path):
        """Initialises a four-channel raster object from a file.

        Args:
            path (str): Path to the raster file.

        Raises:
            RasterioError: If the raster file cannot be opened.
        """

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
        """Resamples the raster data to a new size.

        Args:
            factor (float): Scaling factor for the new raster.
            resampling (str): Resampling method. Valid options are "average", "bilinear", "cubic", "lanczos" or "nearest". Defaults to "average".

        Raises:
            ValueError: If the resampling method is not valid.
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
            self.data = raster.read(
                out_shape=shape, resampling=self.RESAMPLING_METHODS.get(resampling)
            )

            # Update transform
            self.transform = self.transform * raster.transform.scale(
                (self.width / shape[-1]), (self.height / shape[-2])
            )

        # Update raster information
        self.height = height
        self.width = width

        # Get new raster resolution
        self.resolution_x = self.transform[0]
        self.resolution_y = -self.transform[4]

    def get_indices(self, lon, lat):
        """Converts geographic coordinates to raster indices.

        Args:
            lon (float): Longitude of the geographic coordinate.
            lat (float): Latitude of the geographic coordinate.

        Returns:
            tuple: Tuple containing the raster indices (row, column).
        """
        # Create RasterIO raster handle
        with rasterio.open(self.path, "r") as raster:
            (py, px) = raster.index(lon, lat)

        return (py, -px)
