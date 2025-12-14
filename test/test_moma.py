import pytest

from types import SimpleNamespace
import pathlib

from moma.generators import TestMapGenerator
from moma.generators import TestPathsGenerator

from moma.toolboxes import MobilityToolbox

import matplotlib.pyplot as plt
plt.rc('text', usetex=False)

def test_map_generator():
    map_dim=(10, 10)
    slope_length=(3, 3)
    plateaus=(2, 2)
    resolution=2
    transition_pad=2
    start_pad=2

    pytest_map_dir = pathlib.Path(".") / "test/output/sites/testmap"

    test_map_generator = TestMapGenerator()
    map, resolution = test_map_generator.main_generate(
        test_map_params=SimpleNamespace(
            map_dim=map_dim,
            slope_length=slope_length,
            plateaus=plateaus,
            resolution=resolution,
            transition_pad=transition_pad,
            start_pad=start_pad,
        ), 
        return_map=True, 
        create_rasters=True,
        base_output_dir=pytest_map_dir
        )

    assert resolution == resolution, "Resolution does not match expected value."
    assert (map.shape[0] == (map_dim[0] + (transition_pad + start_pad)*2)*resolution) \
       and (map.shape[1] == (map_dim[1] + (transition_pad + start_pad)*2)*resolution), \
       "Map dimensions do not match expected size."

# Creates link from the temporary test directory to the example mapping files to have the correct reletive paths
@pytest.fixture
def mapping_links():
    yamls = [
        "landuse_mapping.yaml",
        "moisturecontent_mapping.yaml",
        "soiltype_mapping.yaml",
    ]

    links = []

    for yaml in yamls:
        real = pathlib.Path(
            "src/moma/data/examples/dataloaders/dataforsyningen"
        ) / yaml
        link = pathlib.Path(
            "test/output/dataloaders/dataforsyningen1"
        ) / yaml

        link.parent.mkdir(parents=True, exist_ok=True)

        if link.exists() or link.is_symlink():
            link.unlink()

        link.symlink_to(real.resolve())
        links.append(link)

    return links


def test_main_mobility_generation(mapping_links):
    for link in mapping_links:
        assert link.exists(), f"Mapping link from temp test output to {link} does not exist."
        assert link.is_symlink(), f"Mapping link from temp test output to {link} is not a symlink."

    pytest_sites_dir = pathlib.Path(".") / "test/output/sites"
    pytest_vehicles_dir = pathlib.Path(".") / "src/moma/data/examples/vehicles"
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

    print("Testing directional_mobility_maps...")
    moma.generate_directional_mobility_maps(["mean"])
    
#def test_paths_generator():
#    test_paths_generator = TestPathsGenerator()
#    result = test_paths_generator.main_generate()
#
#    assert result is not None