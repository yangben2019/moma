import json
import numpy as np
import pathlib

import matplotlib.pyplot as plt
import tables

from matplotlib import rcParams
from scipy.interpolate import interp1d

# Set default for plotting
rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelpad": 10,
        "axes.titlepad": 10,
    }
)

test_path_MOMA_dir = pathlib.Path() / "examples" / "sites" / "testmap" / "test_paths"
test_path_Chrono_dir = (
    pathlib.Path() / "Result generation" / "validation runs"
)  # "validation runs limited"
output_dir = pathlib.Path() / "Result generation" / "Export"
output_dir.mkdir(exist_ok=True)
single_runs = output_dir / "single_runs"
single_runs.mkdir(exist_ok=True)


def get_lim(limits, padding=0.05):
    lim_min = limits[0]
    lim_max = limits[1]
    pad = padding * (lim_max - lim_min)
    limits = (lim_min - pad, lim_max + pad)
    return limits


def direction_to_angle(direction):
    """
    Convert direction to angle in radians.
    """
    if direction == "E":
        return 0
    elif direction == "NE":
        return np.pi / 4
    elif direction == "N":
        return np.pi / 2
    elif direction == "NW":
        return np.pi * 3 / 4
    elif direction == "W":
        return np.pi
    elif direction == "SW":
        return -np.pi * 3 / 4
    elif direction == "S":
        return -np.pi / 2
    elif direction == "SE":
        return -np.pi / 4


def get_MOMA_metric(run_number):
    file = test_path_MOMA_dir / f"path-{run_number}" / "path.json"

    with open(file, "r") as json_file:
        path = json.load(json_file)

    time = np.zeros(len(path["waypoints"]))
    time_omni = np.zeros(len(path["waypoints"]))

    direction = path["direction"]
    elevation = np.zeros(len(path["waypoints"]))
    slope = np.zeros(len(path["waypoints"]))
    sideslope = np.zeros(len(path["waypoints"]))
    position = np.zeros((2, len(path["waypoints"])))
    speed = np.zeros(len(path["waypoints"]))
    speed_omni = np.zeros(len(path["waypoints"]))

    for idx, waypoint in enumerate(path["waypoints"]):
        time[idx] = waypoint["path"]["total_time"]
        time_omni[idx] = waypoint["omni_path"]["total_time"]
        elevation[idx] = waypoint["elevation"]
        slope[idx] = waypoint["path"]["slope"]["forward"]
        sideslope[idx] = waypoint["path"]["slope"]["side"]
        position[:, idx] = np.array([waypoint["map"]["x"], waypoint["map"]["y"]])
        speed[idx] = waypoint["path"]["speed"]
        speed_omni[idx] = waypoint["omni_path"]["speed"]

    # Only works for paths without turning
    distance = np.sqrt(
        (position[0, :] - position[0, 0]) ** 2 + (position[1, :] - position[1, 0]) ** 2
    )

    # Move distance to the middle of the segment that it predicts speed for
    new_dist = distance[:-1] + (distance[1:] - distance[:-1]) / 2
    speed = speed[:-1]
    speed_omni = speed_omni[:-1]

    # Pads the end of the data
    new_dist = np.concatenate(([distance[0]], new_dist, [distance[-1]]))
    speed = np.concatenate(([speed[0]], speed, [speed[-1]]))
    speed_omni = np.concatenate(([speed_omni[0]], speed_omni, [speed_omni[-1]]))

    elevation_dist = distance

    return {
        "time": time,
        "time_omni": time_omni,
        "distance": new_dist,
        "speed": speed,
        "speed_omni": speed_omni,
        "slope": slope,
        "sideslope": sideslope,
        "elevation": elevation,
        "elevation_dist": elevation_dist,
        "direction": direction,
    }


def get_Chrono_metric(run_number, plot=0):
    h5_file = test_path_Chrono_dir / f"path-{run_number}" / "results.h5"
    json_file = test_path_Chrono_dir / f"path-{run_number}" / "path.json"

    with tables.open_file(h5_file, driver="H5FD_CORE") as h5file:
        time = h5file.root.data.time[:]

        position = h5file.root.data.pose.position[:]
        rotation = h5file.root.data.pose.rotation[:]
        velocity = h5file.root.data.velocity.linear[:]

    with open(json_file, "r") as json_file:
        path = json.load(json_file)

    direction = path["direction"]

    stable_z_rotation = rotation[100:, 2]

    forward_direction_angle = direction_to_angle(direction)

    # Only works for paths without turning
    distance = np.sqrt(
        (position[:, 0] - position[0, 0]) ** 2 + (position[:, 1] - position[0, 1]) ** 2
    )
    longitudinal_velocity = velocity[:, 0]

    failed = False
    angle_diff = np.abs(
        np.mod(stable_z_rotation - forward_direction_angle + np.pi, np.pi * 2) - np.pi
    )
    if np.max(angle_diff) > np.pi / 4:
        failed = True
        print(
            f"\n\t!!!!!! Run {run_number}, direction {direction} failed: \nThe vehicle is turned more that 45 degrees from the forward direction."
        )

        if plot > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(distance[100:], angle_diff * 10, label="angle difference [deg]")
            ax.plot(
                distance, longitudinal_velocity, label="Longitudinal Velocity [m/s]"
            )
            ax.set_xlabel("Distance [m]")

            ax.legend()

            fig.savefig(
                single_runs / f"run-{run_number} failed.pdf", bbox_inches="tight"
            )
            plt.close(fig)

    return {
        "time": time,
        "distance": distance,
        "speed": longitudinal_velocity,
        "failed": failed,
    }


def compare_run(run_number, plot=0):
    MOMA = get_MOMA_metric(run_number)
    DYNO = get_Chrono_metric(run_number, plot=plot)

    return_dict = {
        "run_number": run_number,
        "failed": DYNO["failed"],
        "direction": MOMA["direction"],
    }

    if not DYNO["failed"]:
        MOMA["interpolators"] = {
            "distance": interp1d(MOMA["distance"], MOMA["speed"], kind="linear"),
            "distance_omni": interp1d(
                MOMA["distance"], MOMA["speed_omni"], kind="linear"
            ),
            "time": interp1d(MOMA["elevation_dist"], MOMA["time"], kind="linear"),
            "time_omni": interp1d(
                MOMA["elevation_dist"], MOMA["time_omni"], kind="linear"
            ),
        }

        # plot limits
        min_x = min(MOMA["distance"][0], DYNO["distance"][0])
        max_x = max(MOMA["distance"][-1], DYNO["distance"][-1])
        x_lim = get_lim((min_x, max_x))

        # Error between the two methods
        end = min(
            MOMA["distance"][-1], MOMA["distance"][-1]
        )  # Last point where both models have data
        end_mask = DYNO["distance"] <= end

        return_dict["end_time"] = {
            "MOMA": MOMA["interpolators"]["time"](end),
            "MOMA_omni": MOMA["interpolators"]["time_omni"](end),
            "DYNO": DYNO["time"][end_mask][-1],
        }

        # ============
        # Single runs
        # ============
        if plot == 1:
            output_dir_run = single_runs / "combined results"
            output_dir_run.mkdir(exist_ok=True)

            error_dist = DYNO["distance"][end_mask]
            error = (
                MOMA["interpolators"]["distance"](error_dist)
                - DYNO["speed"][end_mask]
            )
            error_percentage = np.abs(error / DYNO["speed"][end_mask]) * 100

            error_omni = (
                MOMA["interpolators"]["distance_omni"](error_dist)
                - DYNO["speed"][end_mask]
            )
            error_percentage_omni = np.abs(error_omni / DYNO["speed"][end_mask]) * 100

            # Plotting the results
            fig, ax = plt.subplots(4, 1, figsize=(10, 12))

            ax[0].plot(
                MOMA["elevation_dist"],
                MOMA["slope"],
                label="Slope [deg]",
                color="purple",
            )
            ax[0].plot(
                MOMA["elevation_dist"],
                MOMA["sideslope"],
                label="Sideslope [deg]",
                color="brown",
            )
            ax[0].set_xlabel("Distance [m]")
            ax[0].set_ylabel("value")

            ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax[0].set_title(f"Run {run_number} - Direction: {MOMA['direction']}")

            speed = np.min((np.ones_like(MOMA["speed"]) * 5, MOMA["speed"]), axis=0)
            speed_omni = np.min(
                (np.ones_like(MOMA["speed_omni"]) * 5, MOMA["speed_omni"]), axis=0
            )
            ax[1].plot(MOMA["distance"], speed, label="MOMA directional", color="blue")
            ax[1].plot(
                MOMA["distance"],
                speed_omni,
                label="MOMA omnidirectional",
                color="orange",
            )
            ax[1].plot(DYNO["distance"], DYNO["speed"], label="DYNO", color="red")

            ax[1].set_xlabel("Distance [m]")
            ax[1].set_ylabel("Speed (m/s)")

            ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax[2].plot(error_dist, error, label="Error directional", color="blue")
            ax[2].plot(
                error_dist, error_omni, label="Error omnidirectional ", color="orange"
            )

            ax[2].set_xlabel("Distance [m]")
            ax[2].set_ylabel("Error (m/s)")

            ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax[3].plot(
                error_dist,
                error_percentage,
                label=r"$y = \frac{|v_{diff}|}{v_{DYNO}}$",
                color="blue",
            )
            ax[3].plot(
                error_dist,
                error_percentage_omni,
                label=r"$y = \frac{|v_{diff, omni}|}{v_{DYNO}}$",
                color="orange",
            )

            ax[3].set_xlabel("Distance [m]")
            ax[3].set_ylabel("Error (\%)")

            ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax[0].set_xlim(x_lim)
            ax[1].set_xlim(x_lim)
            ax[2].set_xlim(x_lim)
            ax[3].set_xlim(x_lim)
            ax[3].set_ylim((-10, 1510))

            fig.savefig(output_dir_run / f"run-{run_number}.pdf", bbox_inches="tight")
            plt.close(fig)

        elif plot == 2:
            output_dir_run = single_runs / "separated results"
            output_dir_run.mkdir(exist_ok=True)

            # Plotting the results
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))

            ax.plot(
                MOMA["elevation_dist"],
                MOMA["slope"],
                label="Slope [deg]",
                color="purple",
            )
            ax.plot(
                MOMA["elevation_dist"],
                MOMA["sideslope"],
                label="Sideslope [deg]",
                color="brown",
            )

            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("value", labelpad=10)

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax.set_xlim(x_lim)

            fig.savefig(output_dir_run / f"run-{run_number}-a.pdf", bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(10, 3))

            speed = np.min((np.ones_like(MOMA["speed"]) * 5, MOMA["speed"]), axis=0)
            speed_omni = np.min(
                (np.ones_like(MOMA["speed_omni"]) * 5, MOMA["speed_omni"]), axis=0
            )
            ax.plot(MOMA["distance"], speed, label="MOMA directional", color="blue")
            ax.plot(
                MOMA["distance"],
                speed_omni,
                label="MOMA omnidirectional",
                color="orange",
            )
            ax.plot(
                DYNO["distance"],
                DYNO["speed"],
                label="Vehicle simulation",
                color="red",
            )

            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("Speed [m/s]", labelpad=10)

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax.set_xlim(x_lim)

            fig.savefig(output_dir_run / f"run-{run_number}-b.pdf", bbox_inches="tight")
            plt.close(fig)

    return return_dict


runs = []
for run in range(32):
    print(f"run number: {run:3}")
    return_dict = compare_run(run, plot=1)
    if not return_dict["failed"]:
        runs.append(return_dict)

# Plot end time
MOMA_directional = np.array([run["end_time"]["MOMA"] for run in runs])
MOMA_omnidirectional = np.array([run["end_time"]["MOMA_omni"] for run in runs])
DYNO = np.array([run["end_time"]["DYNO"] for run in runs])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(MOMA_directional / 60, DYNO / 60, "o", label="Directional approach")
ax.plot(MOMA_omnidirectional / 60, DYNO / 60, "o", label="Omnidirectional approach")
for xd, xo, y in zip(MOMA_directional, MOMA_omnidirectional, DYNO):
    ax.plot(
        [xd / 60, xo / 60], [y / 60, y / 60], color="gray", linewidth=0.5, zorder=-1
    )
ax.plot([-10, 100], [-10, 100], "--", color="black", label="y=x")

ax.set_xlabel("Predicted end time (min)")
ax.set_ylabel("Simulated end time (min)")

ax.set_xlim(get_lim((0, 3)))
ax.set_ylim(get_lim((0, 0.5)))

ax.legend(loc="lower right")

fig.savefig(output_dir / "end_time.pdf", bbox_inches="tight")
plt.close(fig)
