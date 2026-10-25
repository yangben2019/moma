"""Tests for the optional hard no-go graph policy."""

from types import SimpleNamespace

import networkx as nx
import numpy as np

from moma.mobility import DirectionalMobilityMap
from moma.planning import make_speed_graph


def _directional_map():
    # 5x5 grid with a vertical zero-speed barrier in column 2.
    speed = np.ones((5, 5))
    speed[:, 2] = 0.0
    directional_map = DirectionalMobilityMap({})
    directional_map.mobility_maps = {
        0: SimpleNamespace(direction="N", raster=SimpleNamespace(data=speed)),
        90: SimpleNamespace(direction="E", raster=SimpleNamespace(data=speed)),
        180: SimpleNamespace(direction="S", raster=SimpleNamespace(data=speed)),
        270: SimpleNamespace(direction="W", raster=SimpleNamespace(data=speed)),
    }
    return directional_map


def test_soft_no_go_allows_crossing():
    graph = make_speed_graph(_directional_map(), hard_no_go=False)
    assert nx.has_path(graph, 0, 4)


def test_hard_no_go_blocks_crossing():
    graph = make_speed_graph(_directional_map(), hard_no_go=True)
    assert not nx.has_path(graph, 0, 4)
