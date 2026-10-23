import numpy as np
import matplotlib.pyplot as plt


def _get_value(object_or_dict, name, default=None):
    """Read a value from either an object or a dictionary."""
    if isinstance(object_or_dict, dict):
        return object_or_dict.get(name, default)

    return getattr(object_or_dict, name, default)


def _get_collision_names(collisions):
    """Find link names involved in collisions."""
    collision_names = set()

    for collision in collisions:
        first = _get_value(collision, "first")
        second = _get_value(collision, "second")

        if first is not None:
            collision_names.add(str(first))

        if second is not None:
            collision_names.add(str(second))

    return collision_names


def _draw_sphere(ax, sphere):
    """Draw a sphere obstacle."""
    center = _get_value(sphere, "center")
    radius = _get_value(sphere, "radius")

    if center is None or radius is None:
        raise AttributeError(
            "Sphere must have center and radius attributes"
        )

    center = np.asarray(center, dtype=float)
    radius = float(radius)

    if center.shape != (3,):
        raise ValueError(
            "Sphere center must contain exactly 3 values"
        )

    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 20)

    x = (
        center[0]
        + radius * np.outer(
            np.cos(u),
            np.sin(v),
        )
    )

    y = (
        center[1]
        + radius * np.outer(
            np.sin(u),
            np.sin(v),
        )
    )

    z = (
        center[2]
        + radius * np.outer(
            np.ones_like(u),
            np.cos(v),
        )
    )

    ax.plot_surface(
        x,
        y,
        z,
        color="orange",
        alpha=0.35,
    )


def _draw_box(ax, box):
    """Draw an axis-aligned box obstacle."""
    minimum = _get_value(box, "minimum")
    maximum = _get_value(box, "maximum")

    if minimum is None or maximum is None:
        raise AttributeError(
            "Box must have minimum and maximum attributes"
        )

    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)

    x0, y0, z0 = minimum
    x1, y1, z1 = maximum

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for start_index, end_index in edges:
        start = vertices[start_index]
        end = vertices[end_index]

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="orange",
            linewidth=2,
        )


def _draw_obstacle(ax, obstacle):
    """Draw a Sphere or Box obstacle."""
    has_sphere_attributes = (
        _get_value(obstacle, "center") is not None
        and _get_value(obstacle, "radius") is not None
    )

    has_box_attributes = (
        _get_value(obstacle, "minimum") is not None
        and _get_value(obstacle, "maximum") is not None
    )

    if has_sphere_attributes:
        _draw_sphere(ax, obstacle)

    elif has_box_attributes:
        _draw_box(ax, obstacle)

    else:
        raise TypeError(
            "Obstacle must contain either "
            "center/radius or minimum/maximum"
        )


def plot_robot(
    transforms,
    obstacles=None,
    collisions=None,
    link_names=None,
):
    """
    Plot the robot and collision geometry.

    Parameters
    ----------
    transforms:
        Sequence of 4x4 homogeneous transformation matrices.

    obstacles:
        List of Sphere or Box objects.

    collisions:
        Output from checker.check(transforms).

    link_names:
        Names of the links. If there are N transforms,
        provide N-1 names.
    """
    if obstacles is None:
        obstacles = []

    if collisions is None:
        collisions = []

    transforms = list(transforms)

    if len(transforms) < 2:
        raise ValueError(
            "At least two transforms are required"
        )

    points = []

    for index, transform in enumerate(transforms):
        transform = np.asarray(transform, dtype=float)

        if transform.shape != (4, 4):
            raise ValueError(
                f"Transform {index} must have shape (4, 4)"
            )

        points.append(transform[:3, 3])

    points = np.asarray(points, dtype=float)

    number_of_links = len(points) - 1

    if link_names is None:
        link_names = [
            f"link_{index}"
            for index in range(number_of_links)
        ]
    else:
        link_names = list(link_names)

        if len(link_names) != number_of_links:
            raise ValueError(
                f"Expected {number_of_links} link names, "
                f"but received {len(link_names)}"
            )

    collision_names = _get_collision_names(collisions)

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    for index in range(number_of_links):
        start = points[index]
        end = points[index + 1]
        link_name = str(link_names[index])

        if link_name in collision_names:
            color = "red"
            width = 5
        else:
            color = "royalblue"
            width = 3

        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=width,
        )

    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="black",
        s=40,
        label="Joints",
    )

    for obstacle in obstacles:
        _draw_obstacle(axis, obstacle)

    if collisions:
        axis.set_title(
            "6-DOF Robot - COLLISION DETECTED"
        )
    else:
        axis.set_title(
            "6-DOF Robot - COLLISION FREE"
        )

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")

    axis.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    plt.show()

    return figure, axis