"""
Plotting module
"""

"""
 #############################################################################
 #                            _     _     _     _                            #
 #                           / \   / \   / \   / \                           #
 #                          ( M ) ( O ) ( M ) ( A )                          #
 #                           \_/   \_/   \_/   \_/                           #
 #                                                                           #
 #               MOMA: Ground vehicles mobility mapping toolkit              #
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

from pathlib import Path

import numpy as np

from pyvista import Plotter, StructuredGrid
from rasterio.enums import Resampling
from rioxarray import open_rasterio
from scipy.ndimage import gaussian_filter


class Plotter3D:

    def __init__(self, path: str, resample_factor: int = 1, scale_factor: float = 1.0):

        self._layers = {}

        assert type(resample_factor) == int
        assert resample_factor > 0
        self._resample_factor: int = resample_factor

        assert type(scale_factor) == float
        assert scale_factor > 0
        self._scale_factor: int = scale_factor

        self._plotter = Plotter()

        assert Path(path).exists()
        self._load_elevation(Path(path))

    def load_values(self, path: str, key, factor: float = 1.0) -> None:
        logging.info('Loading GeoTIFF "%s" to layer "%s" ...', Path(path).name, key)

        data = open_rasterio(path)

        if len(data.shape) > 2:
            data = data[0]

        if data.shape != self._data.shape:

            logging.info(
                'Resampling input layer "%s" of size (%i, %i) to (%i, %i) ...',
                Path(path).name,
                data.shape[0],
                data.shape[1],
                self._data.shape[0],
                self._data.shape[1],
            )
            data = self._resample_from_reference(data, self._data)
        self._layers[key] = data

    def find_elevations(self, cx, cy):

        return np.array(
            [
                self._values.sel(x=x, y=y, method="nearest").values
                for x, y in zip(cx, cy)
            ]
        )

    def _load_elevation(self, path: Path) -> None:

        # Reading in the data
        self._data = open_rasterio(path)
        self._data = self._resample_from_factor(self._data, self._resample_factor)
        self._values = np.asarray(self._data[0])

        # logging.info("Adding values for layer at path \"%s\"", path)
        logging.info(
            "Layer size: (%i, %i)", self._values.shape[0], self._values.shape[1]
        )

        self._create_mesh()

    def _resample_from_factor(
        self, raster, factor: float, resampling=Resampling.average
    ):

        return self._resample(
            raster,
            (int(raster.rio.height * factor), int(raster.rio.width * factor)),
            resampling,
        )

    def _resample_from_reference(
        self, raster, reference_raster, resampling=Resampling.average
    ):

        return self._resample(
            raster,
            (reference_raster.rio.height, reference_raster.rio.width),
            resampling,
        )

    def _resample(self, raster, shape, resampling=Resampling.average):

        logging.info(
            "Layer size before/after: (%i x %i) => (%i x %i)",
            raster.rio.width,
            raster.rio.height,
            shape[1],
            shape[0],
        )

        return raster.rio.reproject(
            raster.rio.crs,
            shape=(shape[0], shape[1]),
            resampling=resampling,
        )

    def _filter(self, radius) -> None:

        self._data[:] = gaussian_filter(self._data, 1.0, radius=radius)

    def _flatten(self) -> None:

        pass

    def _get_meshgrid(self) -> None:

        x, y = np.meshgrid(self._data["x"], self._data["y"])
        z = np.zeros_like(x)

        return x, y, z

    def _create_mesh(self) -> None:

        x, y, z = self._get_meshgrid()
        self._mesh = StructuredGrid(x, y, z)
        self._mesh["Elevation"] = self._values.ravel(order="F")

        self._topo = self._mesh.warp_by_scalar(
            scalars="Elevation", factor=self._scale_factor
        )

    def plot_path(self, plotter, pts) -> None:

        pts = np.multiply(pts, [1.0, 1.0, self._scale_factor])
        plotter.add_points(
            pts,
            cmap="viridis",
            scalars=pts[:, 2],
            render_points_as_spheres=True,
            point_size=10.0,
            show_scalar_bar=False,
        )

    def plot_map(self, layer: str, label: str) -> None:

        self._plotter.add_mesh(
            mesh=self._topo,
            scalars=np.array(self._layers[layer]).ravel(order="F"),
            cmap="terrain",
            interpolate_before_map=True,
            flip_scalars=False,
            show_scalar_bar=False,
            smooth_shading=True,
            split_sharp_edges=True,
            reset_camera=True,
            log_scale=False,
        )

        self._plotter.add_scalar_bar(
            label,
            vertical=False,
            width=0.8,
            position_x=0.1,
            font_family="courier",
            bold=False,
            fmt="%0.2f",
            fill=True,
            title_font_size=24,
            label_font_size=24,
        )

    def clear(self):

        # self._plotter.clear()
        self._plotter = Plotter()

    def set_camera_position(self):

        self._translate_camera((0.0, 70.0, 0.0))
        self._plotter.camera.azimuth = 45
        self._plotter.camera.elevation = 10
        self._plotter.zoom_camera(1.8)

    def save(self, path: str) -> None:

        self._plotter.off_screen = False
        self._plotter.window_size = (3 * 1920, 3 * 1080)
        self._plotter.save_graphic(
            Path(path).with_stem(Path(path).stem + "_3d").with_suffix(".pdf")
        )
        self._plotter.off_screen = True

    def show(self) -> None:

        self._plotter.window_size = (1920, 1080)
        self._plotter.show()

    def _translate_camera(self, translation_vector, flip_perspective=True):
        camera = np.array(self._plotter.camera.position)
        focus = np.array(self._plotter.camera.focal_point)
        up = np.array(self._plotter.camera.up)
        if flip_perspective:
            # Flip translation to get desired movement of the mesh
            translation_vector = -np.array(translation_vector)

        # Calculate and normalize viewing direction
        view_dir = focus - camera
        view_dir = view_dir / np.linalg.norm(view_dir)

        # The "right" vector is perpendicular to both view_dir and up
        # This will be our first basis vector in the normal plane
        right = np.cross(view_dir, up)
        right = right / np.linalg.norm(right)

        # Now recompute a proper "up" vector that's exactly perpendicular to view_dir and right
        # This ensures our basis vectors are orthogonal
        new_up = np.cross(right, view_dir)
        new_up = new_up / np.linalg.norm(new_up)

        # Our translation is now in terms of the right and new_up vectors
        translation = (
            translation_vector[0] * right
            + translation_vector[1] * new_up
            + translation_vector[2] * view_dir
        )

        # Apply translation to both camera and focus
        self._plotter.camera.position = tuple(camera + translation)
        self._plotter.camera.focal_point = tuple(focus + translation)
