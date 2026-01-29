"""
Test module
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

import pytest

from types import SimpleNamespace
import pathlib
import shutil

from moma.generators import TestMapGenerator

from moma.toolboxes import MobilityToolbox

import matplotlib.pyplot as plt

#plt.rc('text', usetex=False)

def test_map_generator():
    map_dim=(10, 10)
    slope_length=(3, 3)
    plateaus=(2, 2)
    resolution=2
    transition_pad=2
    start_pad=2

    pytest_map_dir = pathlib.Path(".") / "test/output/sites/testmap"

    test_map_generator = TestMapGenerator()
    test_map_generator.main_generate(
        test_map_params=SimpleNamespace(
            map_dim=map_dim,
            slope_length=slope_length,
            plateaus=plateaus,
            resolution=resolution,
            transition_pad=transition_pad,
            start_pad=start_pad,
        ), 
        create_rasters=True,
        base_output_dir=pytest_map_dir
        )

    # check that elevation raster is created
    map_title = f"elevationmono_1m_clipped_0_0_{map_dim[0] + 2*(start_pad + transition_pad)}_{map_dim[1] + 2*(start_pad + transition_pad)}.tif"
    elevation_raster_path = pytest_map_dir / "rasters" / map_title
    assert elevation_raster_path.exists(), "Elevation raster was not created correctly."

# Creates link from the temporary test directory to the example mapping files to have the correct relative paths
@pytest.fixture
def mapping_links():
    yamls = [
        "landuse_mapping.yaml",
        "moisturecontent_mapping.yaml",
        "soiltype_mapping.yaml",
    ]

    copies = []

    for yaml in yamls:
        src = (
            pathlib.Path("data/dataloaders/dataforsyningen")
            / yaml
        )
        dst = (
            pathlib.Path("test/output/dataloaders/dataforsyningen")
            / yaml
        )

        dst.parent.mkdir(parents=True, exist_ok=True)

        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)

        copies.append(dst)

    return copies


def test_main_mobility_generation(mapping_links):
    for link in mapping_links:
        assert link.exists(), f"Mapping link from temp test output to {link} does not exist."

    pytest_sites_dir = pathlib.Path(".") / "test/output/sites"
    pytest_vehicles_dir = pathlib.Path(".") / "data/vehicles"
    moma = MobilityToolbox(
        sites_path=pytest_sites_dir,
        vehicles_path=pytest_vehicles_dir,
        site_name="testmap",
        vehicle_name="mock",
        n_samples=1,
        directions=([], 1),
        minimum_speed=1,
        target_speed=5,
        export_plots=True,
        hide_utm=True,)

    for name, func in [
        ("aerial_map", moma.generate_aerial_map),
        ("elevation_map", moma.generate_elevation_map),
        ("directional_slope_maps", moma.generate_directional_slope_maps),
        ("land_use_map", moma.generate_land_use_map),
        ("soil_type_map", moma.generate_soil_type_map),
        ("moisture_content_map", moma.generate_moisture_content_map),
        ("friction_map", moma.generate_friction_map),
        ("no_go_map", moma.generate_no_go_map),
    ]:
        # Runs the functions and checks they complete without error
        print(f"Testing {name}...")
        func()

    moma.generate_directional_mobility_maps(["mean"])

    # check that mobility map is created
    mobility_map_path = pytest_sites_dir / "testmap" / "export" / "mobility_mock" / "mobility_maps" / "mobility_map_0_0.pdf"
    assert mobility_map_path.exists(), "Mobility map was not created correctly."