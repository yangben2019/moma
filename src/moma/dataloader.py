"""
Dataloader module
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

import io
import logging
import json
import pathlib
import requests
import urllib.request

from collections import namedtuple
from urllib.parse import urlencode, urlunparse

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import rasterio
import yaml

from rasterio.transform import Affine
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.features import rasterize
from PIL import Image

from moma.raster import RasterRGBA, RasterMono

log = logging.getLogger("MOMA")


def get_dataloader(site_path, data_source, coordinates):
    """
    Factory generator that initializes the correct dataloader based on the data source.

    Args:
        site_path (str): The path to the site data.
        data_source (str): The data source.
        coordinates (str): The coordinate reference system.

    Returns:
        Union[DataloaderGeoHornet, DataloaderDataforsyningen, DataloaderDataforsyningenFriction, DataLoaderOksbolPaper]: The dataloader.
    """
    if data_source == "GeoHornet":
        return DataloaderGeoHornet(site_path)

    if data_source == "Dataforsyningen":
        return DataloaderDataforsyningen(site_path, coordinates)

    if data_source == "DataforsyningenFriction":
        return DataloaderDataforsyningenFriction(site_path, coordinates)

    if data_source == "OksbolPaper":
        return DataLoaderOksbolPaper(site_path, coordinates)

    raise ValueError(f"Unknown data source: {data_source}")


def save_new_raster(array, path, transform, epsg=25832):
    """
    Save a new raster.

    Args:
        array (numpy.ndarray): The array containing the raster data.
        path (str): The path to save the raster to.
        transform (rasterio.Affine): The affine transformation for the raster.
        epsg (int, optional): The EPSG code for the coordinate reference system. Defaults to 25832.
    """
    new_dataset = rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        crs=rasterio.crs.CRS.from_epsg(epsg),
        transform=transform,
    )
    new_dataset.write(array)
    new_dataset.close()


def generate_synthetic_data(h, w, grid_size, bias, scale):
    """
    Generates synthetic elevation data.

    Args:
        h (int): Height of the synthetic data.
        w (int): Width of the synthetic data.
        grid_size (int): Size of the coarse grid.
        bias (float): Bias to add to the synthetic data.
        scale (float): Scale to multiply the synthetic data by.

    Returns:
        numpy.ndarray: Synthetic elevation data.
    """
    # Generate a coarse grid of random values
    gw, gh = w // grid_size + 1, h // grid_size + 1
    coarse_grid = np.random.rand(gh, gw)  # Random elevations in [0,1]

    # Upsample using bilinear interpolation
    x = np.linspace(0, gw - 1, w)
    y = np.linspace(0, gh - 1, h)
    X, Y = np.meshgrid(x, y)

    # Bilinear interpolation using NumPy
    x0, y0 = X.astype(int), Y.astype(int)
    x1, y1 = np.clip(x0 + 1, 0, gw - 1), np.clip(y0 + 1, 0, gh - 1)

    fx, fy = X - x0, Y - y0  # Fractional part
    v00 = coarse_grid[y0, x0]  # Bottom-left
    v10 = coarse_grid[y0, x1]  # Bottom-right
    v01 = coarse_grid[y1, x0]  # Top-left
    v11 = coarse_grid[y1, x1]  # Top-right

    # Bilinear interpolation formula
    synthetic = (
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )

    return bias + synthetic * scale


# Create a url with parameters for a get request
def create_url_with_parameters(netloc, query_params):
    Components = namedtuple(
        typename="Components",
        field_names=["scheme", "netloc", "url", "path", "query", "fragment"],
    )
    url = urlunparse(
        Components(
            scheme="https",
            netloc=netloc,
            query=urlencode(query_params),
            path="",
            url="",
            fragment="",
        )
    )
    return url


class Dataloader:
    """
    Data loader for site data.

    A valid data loader should return the following rasters in a consistent grid along with some color mappings for them:
       - Aerial
       - Elevation
       - Land Use
       - Soil type
       - Moisture content
    """

    pass


class DataloaderGeoHornet(Dataloader):
    def __init__(self, site_path):
        """
        Initializes the GeoHornet dataloader.

        Args:
            site_path (Path): The path to the site data.
        """

        self.site_path = site_path
        self.source = "Dataloader for GeoHornet data"

        # Raster locations
        self.AERIAL_PATH = site_path / "rasters" / "aerial_1m_clipped.tif"
        self.ELEVATION_PATH = site_path / "rasters" / "elevationmono_1m_clipped.tif"
        self.LANDUSE_PATH = site_path / "rasters" / "landuse_1m_clipped.tif"
        self.SOILTYPE_PATH = site_path / "rasters" / "soiltype30_1m_clipped.tif"
        self.MOISTURECONTENT_PATH = (
            site_path / "rasters" / "moisturecontent15_1m_clipped.tif"
        )

        # Additional mapping information
        self.LANDUSE_MAPPING = (
            site_path.parents[1] / "dataloaders" / "geohornet" / "landuse_mapping.yaml"
        )
        self.SOILTYPE_MAPPING = (
            site_path.parents[1] / "dataloaders" / "geohornet" / "soiltype_mapping.yaml"
        )
        self.MOISTURECONTENT_MAPPING = (
            site_path.parents[1]
            / "dataloaders"
            / "geohornet"
            / "moisturecontent_mapping.yaml"
        )

    def get_aerial_raster(self):
        """
        Returns the aerial raster.

        Returns:
            RasterRGBA: The aerial raster.
        """

        processed_path = str(self.AERIAL_PATH)[0:-4] + "_processed.tif"

        # Add one line of pixels to make to match dimensions of the other rasters
        if not pathlib.Path.exists(processed_path):
            with rasterio.open(self.AERIAL_PATH) as raster:
                data = [raster.read(i) for i in range(1, 5)]
                transform = raster.transform

            h, w = data[0].shape
            aerial = np.zeros((4, h + 1, w), dtype=np.uint8)
            aerial[:, :-1, :] = np.stack(data)

            save_new_raster(aerial, processed_path, transform)

        return RasterRGBA(processed_path)

    def get_elevation_raster(self):
        """
        Returns the elevation raster.

        Returns:
            RasterMono: The elevation raster.
        """
        return RasterMono(self.ELEVATION_PATH)

    def get_landuse_raster(self):
        """
        Returns the land use raster.

        Returns:
            Tuple[RasterRGBA, dict]: The land use raster and value map.
        """
        # TODO Get rid of the hard-coded land use value mappings
        with open(self.LANDUSE_MAPPING, encoding="utf8") as landuse_valuemap_file:
            landuse_valuemap = yaml.load(landuse_valuemap_file, Loader=yaml.SafeLoader)

        return RasterRGBA(self.LANDUSE_PATH), landuse_valuemap

    def get_soiltype_raster(self):
        """
        Returns the soil type raster.

        Returns:
            Tuple[RasterMono, dict]: The soil type raster and value map.
        """
        with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_mapping_file:
            soiltype_valuemap = yaml.load(soiltype_mapping_file, Loader=yaml.SafeLoader)

        return RasterRGBA(self.SOILTYPE_PATH), soiltype_valuemap

    def get_moisturecontent_raster(self):
        """
        Returns the moisture content raster.

        Returns:
            Tuple[RasterRGBA, dict]: The moisture content raster and value map.
        """

        with open(
            self.MOISTURECONTENT_MAPPING, encoding="utf8"
        ) as moisturecontent_valuemap_file:
            moisturecontent_valuemap = yaml.load(
                moisturecontent_valuemap_file, Loader=yaml.SafeLoader
            )

        return RasterRGBA(self.MOISTURECONTENT_PATH), moisturecontent_valuemap


class DataloaderDataforsyningen:
    def __init__(self, site_path, coordinates, username, password):
        """
        Initializes the Dataforsyningen dataloader.

        Args:
            site_path (Path): The path to the site data.
            coordinates (List[float]): The coordinate reference system.
        """
        self.site_path = site_path
        self.coordinates = coordinates
        self.source = "Dataforsyningen, Datafordeleren, GEUS and BaseMap04"
        self.file_subname = (
            f"_{coordinates[0]}_{coordinates[1]}_{coordinates[2]}_{coordinates[3]}"
        )

        self.transform = Affine(1, 0, coordinates[0], 0, -1, coordinates[3])

        (pathlib.Path(site_path) / "rasters").mkdir(exist_ok=True)

        # Location for maps stored locally
        # TODO: Set the basemap location toa path in the data folder
        self.basemap_location = pathlib.Path.cwd()
        self.jordartskort_location = pathlib.Path.cwd()

        # Query parameters for datafordeleren
        self.query_params = {
            "username": username,
            "password": password,
            "BBOX": f"{self.coordinates[0]},{self.coordinates[1]},{self.coordinates[2]},{self.coordinates[3]}",
            "WIDTH": f"{self.coordinates[2] - self.coordinates[0]}",
            "HEIGHT": f"{self.coordinates[3] - self.coordinates[1]}",
        }

        # Raster locations
        self.AERIAL_PATH = (
            site_path / "rasters" / f"aerial_1m_clipped{self.file_subname}.tif"
        )
        self.ELEVATION_PATH = (
            site_path / "rasters" / f"elevationmono_1m_clipped{self.file_subname}.tif"
        )
        self.LANDUSE_PATH = (
            site_path / "rasters" / f"landuse_1m_clipped{self.file_subname}.tif"
        )
        self.SOILTYPE_PATH = (
            site_path / "rasters" / f"soiltype_1m_clipped{self.file_subname}.tif"
        )
        self.MOISTURECONTENT_PATH = (
            site_path / "rasters" / f"moisturecontent_1m_clipped{self.file_subname}.tif"
        )

        # Additional mapping information
        self.LANDUSE_MAPPING = (
            site_path.parents[1]
            / "dataloaders"
            / "dataforsyningen"
            / "landuse_mapping.yaml"
        )
        self.SOILTYPE_MAPPING = (
            site_path.parents[1]
            / "dataloaders"
            / "dataforsyningen"
            / "soiltype_mapping.yaml"
        )
        self.MOISTURECONTENT_MAPPING = (
            site_path.parents[1]
            / "dataloaders"
            / "dataforsyningen"
            / "moisturecontent_mapping.yaml"
        )

    def get_aerial_raster(self):
        """
        Downloads the aerial raster from Dataforsyningen and saves it locally.

        Data source: https://datafordeler.dk/dataoversigt/geodanmark-ortofoto/ortofoto-foraar-wms/

        Returns:
            RasterRGBA: The aerial raster.
        """
        if pathlib.Path.exists(self.AERIAL_PATH):
            log.info(
                "Aerial raster is already stored locally and will not be downloaded again"
            )
        else:
            local_query_params = dict(self.query_params)
            # Specific parameters for the aerial raster
            local_query_params.update(
                {
                    "SERVICE": "WMS",
                    "REQUEST": "GetMap",
                    "VERSION": "1.3.0",
                    "STYLES": "",
                    "FORMAT": "image/png",
                    "CRS": "EPSG:25832",
                    "DPI": "96",
                    "MAP_RESOLUTION": "96",
                    "FORMAT_OPTIONS": "dpi:96",
                    "Layers": "orto_foraar",
                }
            )
            url = create_url_with_parameters(
                "services.datafordeler.dk/GeoDanmarkOrto/orto_foraar/1.0.0/WMS",
                local_query_params,
            )

            log.info(f"Downloading aerial raster from datafordeler, url: \n{url}")
            res = requests.get(url)
            if res.status_code == 200:
                # Convert image bytes to numpy array
                img = Image.open(io.BytesIO(res.content))
                arr = np.array(img)

                # Move bands to be the first dimension
                arr = np.moveaxis(arr, 2, 0)
                # Adding alpha layer
                arr = np.concatenate(
                    (arr, 255 * np.ones((1, arr.shape[1], arr.shape[2]), dtype=int))
                )

                save_new_raster(arr, self.AERIAL_PATH, self.transform)

            else:
                log.info("Data could not be downloaded from source")

        return RasterRGBA(self.AERIAL_PATH)

    def get_elevation_raster(self):
        """
        Downloads the elevation raster from Dataforsyningen and saves it locally.

        Data source: https://datafordeler.dk/dataoversigt/danmarks-hoejdemodel-historisk-dhm/dhm-2007-historik-wcs/

        Returns:
            RasterMono: The elevation raster.
        """

        if pathlib.Path.exists(self.ELEVATION_PATH):
            log.info(
                "Elevation raster is already stored locally and will not be downloaded again"
            )
        else:
            local_query_params = dict(self.query_params)
            # Specific parameters for the elevation raster
            local_query_params.update(
                {
                    "COVERAGE": "dhm2007_terraen",
                    "SERVICE": "WCS",
                    "REQUEST": "GetCoverage",
                    "VERSION": "1.0.0",
                    "STYLE": "default",
                    "FORMAT": "GTiff",
                    "CRS": "EPSG:25832",
                    "RESPONSE_CRS": "EPSG:25832",
                }
            )
            url = create_url_with_parameters(
                "services.datafordeler.dk/DHMterraenBro2007/dhm_historik/1.0.0/WCS",
                local_query_params,
            )

            # TODO fix problem with downloading data that occurs sometimes, quickfix delete created raster and try again
            log.info(f"Downloading elevation raster from datafordeler, url: \n{url}")
            urllib.request.urlretrieve(
                url, self.ELEVATION_PATH
            )  # Save the raster to the specified location

            # Loads raster and adds synthetic data to illustrate feature
            with rasterio.open(self.ELEVATION_PATH) as raster:
                data = raster.read(1)
                transform = raster.transform

            h, w = int(self.query_params["HEIGHT"]), int(self.query_params["WIDTH"])
            elevation_std = generate_synthetic_data(
                h, w, grid_size=200, bias=0.05, scale=0.10
            ).reshape(1, h, w)
            elevation = np.concatenate((data.reshape(1, h, w), elevation_std))
            save_new_raster(elevation, self.ELEVATION_PATH, transform)

        return RasterMono(self.ELEVATION_PATH)

    def get_landuse_raster(self):
        """
        Creates a land use raster from BaseMap04.

        Data source: https://envs.au.dk/om-instituttet-1/faglige-omraader/samfund-miljoe-og-ressourcer/arealanvendelse-og-gis/basemap/basemap03-til-download-1

        Returns:
            Tuple[RasterRGBA, dict]: The land use raster and value map.
        """

        with open(self.LANDUSE_MAPPING, encoding="utf8") as landuse_valuemap_file:
            loaded_data = yaml.load(landuse_valuemap_file, Loader=yaml.SafeLoader)
            basemap_to_landuse_label = loaded_data["basemap_to_landuse_label"]
            landuse_label_to_color = loaded_data["label_to_color"]

        if pathlib.Path.exists(self.LANDUSE_PATH):
            log.info(
                "Landuse raster is already stored locally and will not be downloaded again"
            )
        else:
            log.info("Creating landuse raster from BaseMap04")

            with rasterio.open(self.basemap_location) as raster:
                # Find area of interest in basemap
                y1, x1 = rowcol(
                    raster.transform, self.coordinates[0], self.coordinates[3]
                )
                y2, x2 = rowcol(
                    raster.transform, self.coordinates[2], self.coordinates[1]
                )

                # Widen area to ensure that need area is not cut off in rounding
                x1 -= 1
                x2 += 1
                y1 -= 1
                y2 += 1

                # Creates window and read data
                window = Window(x1, y1, x2 - x1, y2 - y1)
                data = raster.read(1, window=window)
                win_transform = raster.window_transform(window)

            # Initiates map with color corresponding to non classified
            arr = np.zeros((data.shape[0], data.shape[1], 3), dtype=int)
            arr[:, :, 0] = landuse_label_to_color["Non classified"][0]
            arr[:, :, 1] = landuse_label_to_color["Non classified"][1]
            arr[:, :, 2] = landuse_label_to_color["Non classified"][2]

            # Fill in colors for landuse
            for base_map, label in basemap_to_landuse_label.items():
                arr[data == base_map] = landuse_label_to_color[label]

            # Upscale to 1 meter grid
            arr = arr.repeat(10, 0).repeat(10, 1)
            transform = Affine(1, 0, win_transform[2], 0, -1, win_transform[5])

            # Get correct location on the upscaled grid
            y1, x1 = rowcol(transform, self.coordinates[0], self.coordinates[3])
            y2, x2 = rowcol(transform, self.coordinates[2], self.coordinates[1])
            arr = arr[y1:y2, x1 + 1 : x2 + 1, :]
            transform = Affine(1, 0, self.coordinates[0], 0, -1, self.coordinates[3])

            # Move bands to be the first dimension
            arr = np.moveaxis(arr, 2, 0)
            # Adding alpha layer
            arr = np.concatenate(
                (arr, 255 * np.ones((1, arr.shape[1], arr.shape[2]), dtype=int))
            )

            save_new_raster(arr, self.LANDUSE_PATH, transform)

        landuse_valuemap = {}

        for label, color in landuse_label_to_color.items():
            landuse_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = label

        return RasterRGBA(self.LANDUSE_PATH), landuse_valuemap

    def get_soiltype_raster(self):
        """
        Creates a soil type raster from Jordartskort.

        Data source: https://www.geus.dk/produkter-ydelser-og-faciliteter/data-og-kort/danske-kort/download-jordartskort

        Returns:
            Tuple[RasterRGBA, dict]: The soil type raster and value map.
        """

        with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_valuemap_file:
            loaded_data = yaml.load(soiltype_valuemap_file, Loader=yaml.SafeLoader)
            # Mappings between different representations of soil type
            jordart_to_soil_number = loaded_data["jordart_to_soil_number"]
            soil_number_to_label = loaded_data["number_to_label"]
            soil_label_to_color = loaded_data["label_to_color"]
            label_to_program_label = loaded_data["label_to_program_label"]

        if pathlib.Path.exists(self.SOILTYPE_PATH):
            log.info(
                "Soil type raster is already stored locally and will not be downloaded again"
            )
        else:
            log.info("Creating soil type raster from Jordartskort")
            log.info("Importing data from local version of GEUS Jordartskort")

            gdf = gpd.read_file(self.jordartskort_location)

            for jordart, number in jordart_to_soil_number.items():
                gdf.loc[gdf.sediment == jordart, "sediment"] = number

            # Fix problem with values being: 'X - Bed unkown, no infomation', was not present in earlier versions of MOMA
            shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.sediment))

            log.info("Rasterize data")
            rasterized = rasterize(
                shapes,
                out_shape=(
                    int(self.query_params["HEIGHT"]),
                    int(self.query_params["WIDTH"]),
                ),
                transform=self.transform,
                fill=0,
                dtype=np.int32,
            )

            # Initiates map with color corresponding to non classified
            arr = np.zeros((rasterized.shape[0], rasterized.shape[1], 3), dtype=int)
            arr[:, :, 0] = soil_label_to_color["non classified"][0]
            arr[:, :, 1] = soil_label_to_color["non classified"][1]
            arr[:, :, 2] = soil_label_to_color["non classified"][2]

            # Fill in colors for soil type
            for number, label in soil_number_to_label.items():
                arr[rasterized == number] = soil_label_to_color[label]

            # Move bands to be the first dimension
            arr = np.moveaxis(arr, 2, 0)
            # Adding alpha layer
            arr = np.concatenate(
                (arr, 255 * np.ones((1, arr.shape[1], arr.shape[2]), dtype=int))
            )

            save_new_raster(arr, self.SOILTYPE_PATH, self.transform)

        soiltype_valuemap = {}

        for label, color in soil_label_to_color.items():
            soiltype_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = (
                label_to_program_label[label]
            )

        return RasterRGBA(self.SOILTYPE_PATH), soiltype_valuemap

    def get_moisturecontent_raster(self):
        """
        Creates a synthetic moisture content raster.

        Returns:
            Tuple[RasterRGBA, dict]: The moisture content raster and value map.
        """

        with open(
            self.MOISTURECONTENT_MAPPING, encoding="utf8"
        ) as moisturecontent_valuemap_file:
            moisturecontent_valuemap = yaml.load(
                moisturecontent_valuemap_file, Loader=yaml.SafeLoader
            )

        if pathlib.Path.exists(self.MOISTURECONTENT_PATH):
            log.info(
                "Moisture raster is already stored locally and will not be downloaded again"
            )
        else:
            log.info(f"Creating synthetic moisture content raster")

            h, w = int(self.query_params["HEIGHT"]), int(self.query_params["WIDTH"])
            moisture = generate_synthetic_data(h, w, grid_size=200, bias=28, scale=9)

            # Fill in color data
            arr = np.zeros((h, w, 3), dtype=int)
            for color, value in moisturecontent_valuemap.items():
                arr[moisture >= value] = [
                    int(255 * x) for x in mpl.colors.hex2color(color)
                ]

            # Move bands to be the first dimension
            arr = np.moveaxis(arr, 2, 0)
            # Adding alpha layer
            arr = np.concatenate(
                (arr, 255 * np.ones((1, arr.shape[1], arr.shape[2]), dtype=int))
            )

            moisture_std = generate_synthetic_data(
                h, w, grid_size=200, bias=0.5, scale=1
            ).reshape(1, h, w)

            save_new_raster(
                np.concatenate((arr, moisture_std)),
                self.MOISTURECONTENT_PATH,
                self.transform,
            )

        return RasterRGBA(self.MOISTURECONTENT_PATH), moisturecontent_valuemap


class DataloaderDataforsyningenFriction(DataloaderDataforsyningen):
    """
    A class for integrating synthethic terrain friction data for Dataforsyningen rasters.
    """

    def __init__(self, site_path, coordinates, username, password):
        """
        Initializes the `DataloaderDataforsyningen` class.

        Args:
            site_path: The path to the site directory.
            coordinates: The coordinates of the area of interest.
            username: The username for Dataforsyningen authentication.
            password: The password for Dataforsyningen authentication.
        """

        super().__init__(site_path, coordinates, username, password)
        self.FRICTION_PATH = (
            site_path
            / "rasters"
            / f"friction_coefficient_1m_clipped{self.file_subname}.tif"
        )

    def get_friction_raster(self):
        """
        Creates a synthetic friction coefficient raster.

        Returns:
            RasterMono: The friction raster.
        """
        if pathlib.Path.exists(self.FRICTION_PATH):
            log.info(
                "Friction raster is already stored locally and will not be downloaded again"
            )
        else:
            log.info("Creating synthetic friction coefficient raster")

            h, w = int(self.query_params["HEIGHT"]), int(self.query_params["WIDTH"])
            friction = generate_synthetic_data(h, w, grid_size=200, bias=0.5, scale=0)

            # Fill in values
            friction = friction.reshape((h, w, 1))

            # Move bands to be the first dimension
            friction = np.moveaxis(friction, 2, 0)

            save_new_raster(friction, self.FRICTION_PATH, self.transform)

        return RasterMono(self.FRICTION_PATH)


class DataLoaderOksbolPaper(DataloaderDataforsyningenFriction):
    """
    A class for downloading Oksbøl site data from Dataforsyningen.
    """

    def __init__(self, site_path, coordinates):
        """
        Initializes the `DataLoaderOksbolPaper` class.

        Args:
            site_path: The path to the site directory.
            coordinates: The coordinates of the area of interest.
        """

        super().__init__(site_path, coordinates)

        self.LANDUSE_JSON_PATH = (
            site_path / "rasters" / f"landuse_1m_clipped{self.file_subname}.json"
        )
        self.LANDUSE_SINGLE_BAND = (
            site_path
            / "rasters"
            / f"landuse_1m_clipped{self.file_subname}_single_band.tif"
        )
        self.SOILTYPE_JSON_PATH = (
            site_path / "rasters" / f"soiltype_1m_clipped{self.file_subname}.json"
        )
        self.SOILTYPE_SINGLE_BAND = (
            site_path
            / "rasters"
            / f"soiltype_1m_clipped{self.file_subname}_single_band.tif"
        )

    def get_landuse_raster(self):
        """
        Creates landuse and soil type rasters from BaseMap04.

        Data source: https://envs.au.dk/om-instituttet-1/faglige-omraader/samfund-miljoe-og-ressourcer/arealanvendelse-og-gis/basemap/basemap03-til-download-1

        Returns:
            Tuple[RasterRGBA, dict]: The landuse raster and value map.
        """
        with open(self.LANDUSE_MAPPING, encoding="utf8") as landuse_valuemap_file:
            loaded_data = yaml.load(landuse_valuemap_file, Loader=yaml.SafeLoader)
            basemap_to_landuse_label = loaded_data["basemap_to_landuse_label"]
            landuse_label_to_color = loaded_data["label_to_color"]
            label_to_soil_type = loaded_data["label_to_soil_type"]

        if pathlib.Path.exists(self.LANDUSE_PATH):
            log.info(
                "Landuse raster is already stored locally and will not be downloaded again"
            )
        else:
            log.info("Creating landuse raster from BaseMap04")

            with rasterio.open(self.basemap_location) as raster:
                # Find area of interest in basemap
                y1, x1 = rowcol(
                    raster.transform, self.coordinates[0], self.coordinates[3]
                )
                y2, x2 = rowcol(
                    raster.transform, self.coordinates[2], self.coordinates[1]
                )

                # Widen area to ensure that need area is not cut off in rounding
                x1 -= 1
                x2 += 1
                y1 -= 1
                y2 += 1

                # Creates window and read data
                window = Window(x1, y1, x2 - x1, y2 - y1)
                data = raster.read(1, window=window)
                win_transform = raster.window_transform(window)

            # Initiates map with color corresponding to non classified
            arr = np.zeros((data.shape[0], data.shape[1], 3), dtype=int)
            arr[:, :, 0] = landuse_label_to_color["Non classified"][0]
            arr[:, :, 1] = landuse_label_to_color["Non classified"][1]
            arr[:, :, 2] = landuse_label_to_color["Non classified"][2]

            # Fill in colors for landuse
            for base_map, label in basemap_to_landuse_label.items():
                arr[data == base_map] = landuse_label_to_color[label]

            # Upscale to 1 meter grid
            arr = arr.repeat(10, 0).repeat(10, 1)
            transform = Affine(1, 0, win_transform[2], 0, -1, win_transform[5])

            # Get correct location on the upscaled grid
            y1, x1 = rowcol(transform, self.coordinates[0], self.coordinates[3])
            y2, x2 = rowcol(transform, self.coordinates[2], self.coordinates[1])
            arr = arr[y1:y2, x1 + 1 : x2 + 1, :]
            transform = Affine(1, 0, self.coordinates[0], 0, -1, self.coordinates[3])

            # Move bands to be the first dimension
            arr = np.moveaxis(arr, 2, 0)
            # Adding alpha layer
            arr = np.concatenate(
                (arr, 255 * np.ones((1, arr.shape[1], arr.shape[2]), dtype=int))
            )

            save_new_raster(arr, self.LANDUSE_PATH, transform)

            # Saves landuse as single band raster
            single_band = np.zeros((1, arr.shape[1], arr.shape[2]), dtype=int)
            landuse_export = {}
            for i, (label, color) in enumerate(landuse_label_to_color.items()):
                single_band[
                    0,
                    (arr[0, :, :] == color[0])
                    & (arr[1, :, :] == color[1])
                    & (arr[2, :, :] == color[2]),
                ] = i
                landuse_export[i] = ({"label": label, "color": color},)

            save_new_raster(single_band, self.LANDUSE_SINGLE_BAND, transform)
            with open(self.LANDUSE_JSON_PATH, "w") as json_file:
                json.dump(landuse_export, json_file, indent=4)

            # Generate the soil type raster
            with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_valuemap_file:
                loaded_data = yaml.load(soiltype_valuemap_file, Loader=yaml.SafeLoader)
                # Mappings between different representations of soil type
                jordart_to_soil_number = loaded_data["jordart_to_soil_number"]
                soil_number_to_label = loaded_data["number_to_label"]
                soil_label_to_color = loaded_data["label_to_color"]
                label_to_program_label = loaded_data["label_to_program_label"]

            # Initiates map with color corresponding to non classified
            arr_soil = np.zeros_like(arr, dtype=int)
            arr_soil[0, :, 0] = soil_label_to_color["non classified"][0]
            arr_soil[1, :, 1] = soil_label_to_color["non classified"][1]
            arr_soil[2, :, 2] = soil_label_to_color["non classified"][2]

            for label, soil_type in label_to_soil_type.items():
                # Get the color for the soil type
                color = soil_label_to_color[soil_type]
                # Set the color in the array where the landuse matches the label

                for key, landuse in landuse_export.items():
                    landuse = landuse[0]
                    if landuse["label"] == label:
                        arr_soil[0, :, :][single_band[0, :, :] == key] = color[0]
                        arr_soil[1, :, :][single_band[0, :, :] == key] = color[1]
                        arr_soil[2, :, :][single_band[0, :, :] == key] = color[2]
                        break

            # Adding alpha layer
            arr_soil = np.concatenate(
                (
                    arr_soil,
                    255 * np.ones((1, arr_soil.shape[1], arr_soil.shape[2]), dtype=int),
                )
            )

            save_new_raster(arr_soil, self.SOILTYPE_PATH, self.transform)

            with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_valuemap_file:
                loaded_data = yaml.load(soiltype_valuemap_file, Loader=yaml.SafeLoader)
                # Mappings between different representations of soil type
                soil_label_to_color = loaded_data["label_to_color"]

            soils = {
                0: {"label": "hard clay", "color": soil_label_to_color["hard clay"]},
                1: {
                    "label": "hard gravel",
                    "color": soil_label_to_color["hard gravel"],
                },
                2: {"label": "soft clay", "color": soil_label_to_color["soft clay"]},
                3: {"label": "soft sand", "color": soil_label_to_color["soft sand"]},
            }

            soil_export = {
                "shape": [arr_soil.shape[1], arr_soil.shape[2]],
                "resolution": 1,
                "labels": soils,
                "major": "row",
            }
            soil_data_export = np.zeros(soil_export["shape"])
            for i, soil in soils.items():

                soil_data_export[
                    (arr_soil[0, :, :] == soil["color"][0])
                    & (arr_soil[1, :, :] == soil["color"][1])
                    & (arr_soil[2, :, :] == soil["color"][2])
                ] = i

            soil_export["data"] = list(soil_data_export.flatten())

            save_new_raster(
                np.expand_dims(soil_data_export, axis=0),
                self.SOILTYPE_SINGLE_BAND,
                self.transform,
            )
            with open(self.SOILTYPE_JSON_PATH, "w") as json_file:
                json.dump(soil_export, json_file)

        # Saves soil to use in DYNO
        landuse_valuemap = {}

        for label, color in landuse_label_to_color.items():
            landuse_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = label

        if not pathlib.Path.exists(self.LANDUSE_JSON_PATH):
            log.info("Saving landuse raster as json")

        return RasterRGBA(self.LANDUSE_PATH), landuse_valuemap

    def get_soiltype_raster(self):
        """
        Creates soil type raster from soil type mapping.

        Data source: https://www.geus.dk/produkter-ydelser-og-faciliteter/data-og-kort/danske-kort/download-jordartskort

        Returns:
            Tuple[RasterRGBA, dict]: The soil type raster and value map.
        """

        with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_valuemap_file:
            loaded_data = yaml.load(soiltype_valuemap_file, Loader=yaml.SafeLoader)
            # Mappings between different representations of soil type
            jordart_to_soil_number = loaded_data["jordart_to_soil_number"]
            soil_number_to_label = loaded_data["number_to_label"]
            soil_label_to_color = loaded_data["label_to_color"]
            label_to_program_label = loaded_data["label_to_program_label"]

        if pathlib.Path.exists(self.SOILTYPE_PATH):
            log.info(
                "Soil type raster is already stored locally and will not be downloaded again"
            )
        else:
            raise FileNotFoundError(
                "Soil type raster should be generated under landuse raster generation, but was not found."
            )

        soiltype_valuemap = {}

        for label, color in soil_label_to_color.items():
            soiltype_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = (
                label_to_program_label[label]
            )

        return RasterRGBA(self.SOILTYPE_PATH), soiltype_valuemap
