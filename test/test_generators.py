import pytest

from types import SimpleNamespace
import pathlib

from moma.generators import TestMapGenerator
from moma.generators import TestPathsGenerator

def test_map_generator():
    map_dim=(10, 10)
    slope_length=(3, 3)
    plateaus=(2, 2)
    resolution=2
    transition_pad=2
    start_pad=2

    pytest_map_dir = pathlib.Path(".") / "test" / "output" / "testmap"

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
    
#def test_paths_generator():
#    test_paths_generator = TestPathsGenerator()
#    result = test_paths_generator.main_generate()
#
#    assert result is not None

if __name__ == "__main__":
    test_map_generator()