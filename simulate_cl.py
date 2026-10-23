# simulate_collision.py

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from collision import CollisionChecker, capsules_from_fk
from fk import forward_kinematics
from robot_model import robot
from visualization import _box_trace, _sphere_trace


def robot_trace(
    transforms: list[np.ndarray],
    collision_checker: CollisionChecker,
    obstacles: list,
):
    links = capsules_from_fk(
        transforms,
        radii=collision_checker.link_radii,
    )

    collisions = collision_checker.check(
        transforms,
        obstacles=obstacles,
    )

    colliding_links = set()

    for collision in collisions:
        colliding_links.add(collision.first)
        colliding_links.add(collision.second)

    traces = []

    for index, link in enumerate(links):
        link_name = link.name or f"link_{index}"
        color = "red" if link_name in colliding_links else "royalblue"

        points = np.vstack([link.start, link.end])

        traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=12),
                marker=dict(color=color, size=4),
                name=link_name,
            )
        )

    return traces


def main() -> None:
    obstacles = [
        # Add Sphere(...) and Box(...) instances here.
    ]

    checker = CollisionChecker(
        link_radii=[0.05] * 5,
        obstacles=obstacles,
    )

    configurations = []

    for value in np.linspace(-1.0, 1.0, 40):
        q = np.array([
            value,
            0.4 * np.sin(value),
            0.0,
            0.0,
            0.0,
            0.0,
        ])

        configurations.append(q)

    first_transforms = forward_kinematics(
        robot,
        configurations[0],
    )

    initial_traces = robot_trace(
        first_transforms,
        checker,
        obstacles,
    )

    frames = []

    for frame_index, q in enumerate(configurations):
        transforms = forward_kinematics(robot, q)

        frames.append(
            go.Frame(
                data=robot_trace(
                    transforms,
                    checker,
                    obstacles,
                ),
                name=f"frame_{frame_index}",
            )
        )

    figure = go.Figure(
        data=initial_traces,
        frames=frames,
    )

    figure.update_layout(
        title="Robot Collision Simulation",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 100,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                            },
                        ],
                    },
                ],
            }
        ],
    )

    figure.show()


if __name__ == "__main__":
    main()