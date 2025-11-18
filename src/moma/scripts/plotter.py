import logging

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from moma.plotting import Plotter3D

logging.getLogger().setLevel(logging.INFO)


def plot_results():
    N = 25
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.viridis(np.linspace(0, 1, N))
    )

    RESULTS_FOLDER = Path(
        Path.home() / "Downloads" / "2025-05-31-00-34-09-straight_line_acceleration"
    )

    times = []
    velocities = []
    fc = []
    tslopes = []
    for folder in natsorted(
        [entry for entry in RESULTS_FOLDER.iterdir() if entry.is_dir()]
    ):

        h5file = open_file(folder / "case.h5", "r")

        time = np.array(h5file.root.data.time)
        velocity = np.array(h5file.root.data.velocity.linear[:, 0])
        if max(velocity) > 20.0:
            continue
        if min(velocity) < -2.0:
            continue

        fc1 = np.array(h5file.root.metadata.frictionCoefficient)[0]
        if fc1 != 0.8:
            continue

        tslope = np.array(h5file.root.metadata.targetSlope)[0]

        # plt.plot(time, velocity)
        # plt.show()
        tslopes.append(tslope)
        fc.append(fc1)
        times.append(time)
        velocities.append(velocity)

        h5file.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    [
        ax.plot(
            slope,
            time,
            velocity,
        )
        for time, velocity, slope in zip(times, velocities, tslopes)
    ]
    ax.set_ylim((0.0, 20.0))
    ax.set_xlabel("Slope")
    ax.set_ylabel("Time")
    ax.set_zlabel("Velocity")
    plt.show()


import importlib.resources


def main():
    SAVE_PATH: Path = Path("generated_maps")
    OMNIDIRECTIONAL_SLOPE_RASTER_PATH = importlib.resources.path("moma", "data") / Path(
        "Slope map (omni).tif"
    )
    DIRECTIONAL_SLOPE_0_RASTER_PATH = importlib.resources.path("moma", "data") / Path(
        "Slope map (0.0).tif"
    )
    ELEVATION_RASTER_PATH = importlib.resources.path("moma", "data") / Path(
        "elevationmono_1m_clipped_450565_6163390_452401_6164541.tif"
    )

    plotter3d = Plotter3D(ELEVATION_RASTER_PATH, resample_factor=1, scale_factor=5.0)
    plotter3d.load_values(ELEVATION_RASTER_PATH, "elevation", 5)
    plotter3d.load_values(OMNIDIRECTIONAL_SLOPE_RASTER_PATH, "omnidirectional_slope", 1)
    plotter3d.load_values(DIRECTIONAL_SLOPE_0_RASTER_PATH, "directional_slope_0", 1)

    # Overlay planned path
    # =========================================================================
    # cx, cy = sample_points(4, (np.min(x), np.max(x)), (np.min(y), np.max(y)))

    # cx, cy = smooth_line(np.column_stack((cx, cy)).astype(np.float64))

    # pts = np.column_stack((cx, cy, cz)).astype(np.float64)
    # smooth_line(pts)
    # plotter3d.plot_path()

    # Plot and save results
    # =========================================================================
    for layer, label in zip(
        ("elevation", "omnidirectional_slope", "directional_slope_0"),
        ("Elevation [m]", "Slope [deg]", "Slope [deg]"),
    ):

        plotter3d.clear()
        plotter3d.plot_map(layer=layer, label=label)
        plotter3d.set_camera_position()
        plotter3d.save(SAVE_PATH / layer)
        # plotter3d.show()


def smooth_line(pts):
    sp = CubicSpline2D(pts[:, 0], pts[:, 1])

    s = np.arange(0, sp.s[-1], 1.0)
    n_x = []
    n_y = []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        n_x.append(ix)
        n_y.append(iy)

    return n_x, n_y


def sample_points(
    count: int, x_range: Tuple[float, float], y_range: Tuple[float, float]
):

    x_min, x_max = x_range
    y_min, y_max = y_range

    return uniform(x_min, x_max, count), uniform(y_min, y_max, count)


if __name__ == "__main__":
    main()
