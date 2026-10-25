"""Regression tests for RGB -> integer lookups.

Both bugs only appear when the combined RGB value exceeds 255:
  - LandUseMap used JAX uint8 arithmetic, so only the blue channel survived.
  - SoilTypeMap cast the combined value to uint8, so only the blue channel survived.
"""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from moma.mapping import LandUseMap, SoilTypeMap
from moma.raster import RasterRGBA


def _write_rgba(path, pixels):
    """Write a 1-row RGBA GeoTIFF; pixels is a list of (R, G, B)."""
    rgba = np.zeros((4, 1, len(pixels)), dtype=np.uint8)
    for i, (r, g, b) in enumerate(pixels):
        rgba[0, 0, i] = r
        rgba[1, 0, i] = g
        rgba[2, 0, i] = b
        rgba[3, 0, i] = 255
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=len(pixels),
        count=4,
        dtype="uint8",
        crs=CRS.from_epsg(32650),
        transform=from_origin(0, 1, 1, 1),
    ) as ds:
        ds.write(rgba)


def test_landuse_nogo_uses_full_rgb(tmp_path):
    # Built-up red (127, 0, 0) -> 8,323,072; forest (0, 127, 20) -> 32,532.
    path = tmp_path / "landuse.tif"
    _write_rgba(path, [(127, 0, 0), (0, 127, 20)])

    value_map = {"#7F0000": "Built-up area", "#007F14": "Forest"}
    nogo = {"Built-up area": True, "Forest": False}
    landuse = LandUseMap("test", RasterRGBA(path), value_map, nogo)

    assert np.asarray(landuse.nogo).astype(bool).ravel().tolist() == [True, False]


def test_soil_type_uses_full_rgb(tmp_path):
    # #8B4513 = 9,127,187; #0000FF = 255.
    path = tmp_path / "soil.tif"
    _write_rgba(path, [(139, 69, 19), (0, 0, 255)])

    value_map = {"#8B4513": "CL", "#0000FF": "HT"}
    soil = SoilTypeMap("test", RasterRGBA(path), value_map)

    assert soil.soilarray[0, 0] == "CL"
    assert soil.soilarray[0, 1] == "HT"
