import numpy as np

from collision import Box, CollisionChecker, Sphere
from fk import forward_kinematics
from visualization import show_robot

# Change this import to match your project.
# Examples:
# from robot_model import robot
# from robot_model import create_robot
# from robot_model import Robot
from robot_model import robot


def main() -> None:
    q = np.zeros(6, dtype=float)

    obstacles = [
        Sphere(
            center=np.array([0.40, 0.00, 0.20]),
            radius=0.10,
            name="sphere_1",
        ),
        Box(
            minimum=np.array([0.20, -0.20, -0.05]),
            maximum=np.array([0.30, 0.20, 0.10]),
            name="box_1",
        ),
    ]

    link_names = [
        "link_0",
        "link_1",
        "link_2",
        "link_3",
        "link_4",
    ]

    checker = CollisionChecker(
        link_radii=[0.05] * len(link_names),
        obstacles=obstacles,
    )

    transforms = forward_kinematics(robot, q)

    # Use only the positional argument first.
    collisions = checker.check(transforms)

    if collisions:
        print("COLLISION DETECTED")

        for collision in collisions:
            print(
                f"{collision.kind}: "
                f"{collision.first} vs {collision.second}, "
                f"clearance={collision.clearance:.6f}"
            )
    else:
        print("Collision-free configuration")

    show_robot(
        transforms,
        obstacles=obstacles,
        collisions=collisions,
        names=link_names,
        title="Static robot collision test",
    )


if __name__ == "__main__":
    main()