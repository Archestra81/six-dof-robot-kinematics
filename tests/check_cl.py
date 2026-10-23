import numpy as np

from collision import Box, CollisionChecker, Sphere


obstacles = [
    Sphere(
        center=np.array([0.45, 0.0, 0.25]),
        radius=0.10,
        name="workpiece",
    ),
    Box(
        minimum=np.array([0.20, -0.20, -0.05]),
        maximum=np.array([0.30, 0.20, 0.20]),
        name="wall",
    ),
]

collision_checker = CollisionChecker(
    link_radii=[0.04, 0.04, 0.035, 0.03, 0.025],
    obstacles=obstacles,
)


def configuration_is_valid(robot, q: np.ndarray) -> bool:
    transforms = forward_kinematics(robot, q)

    collisions = collision_checker.check(
        transforms,
        names=[
            "link_1",
            "link_2",
            "link_3",
            "link_4",
            "link_5",
        ],
    )

    if collisions:
        for collision in collisions:
            print(
                f"{collision.kind}: "
                f"{collision.first} vs {collision.second}, "
                f"clearance={collision.distance:.4f}"
            )
        return False

    return True
def valid_trajectory(robot, trajectory: list[np.ndarray]) -> bool:
    return all(configuration_is_valid(robot, q) for q in trajectory)