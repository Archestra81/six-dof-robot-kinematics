"""Simple collision detection for robot links and obstacles.

Robot links are represented as capsules:
    capsule = line segment + radius

FK input:
    a sequence of 4x4 homogeneous transformation matrices

For N transforms, N-1 robot links are created:
    transform[0] -> transform[1] = link_0
    transform[1] -> transform[2] = link_1
    ...
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _point(value):
    """Convert a value to a NumPy 3D point."""
    result = np.asarray(value, dtype=float)

    if result.shape != (3,):
        raise ValueError(
            "Expected a 3D point with shape (3), "
            f"but received shape {result.shape}"
        )

    return result


def _transform(value):
    """Convert a value to a NumPy 4x4 transform."""
    result = np.asarray(value, dtype=float)

    if result.shape != (4, 4):
        raise ValueError(
            "Expected a 4x4 transform, "
            f"but received shape {result.shape}"
        )

    return result


def _radius(value):
    """Validate a radius."""
    result = float(value)

    if result < 0.0:
        raise ValueError("Radius cannot be negative")

    return result


# ---------------------------------------------------------------------
# Geometry classes
# ---------------------------------------------------------------------

@dataclass
class Capsule:
    """A capsule represented by a line segment and radius."""

    start: np.ndarray
    end: np.ndarray
    radius: float
    name: str = ""

    def __post_init__(self):
        self.start = _point(self.start)
        self.end = _point(self.end)
        self.radius = _radius(self.radius)


@dataclass
class Sphere:
    """A spherical obstacle."""

    center: np.ndarray
    radius: float
    name: str = ""

    def __post_init__(self):
        self.center = _point(self.center)
        self.radius = _radius(self.radius)


@dataclass
class Box:
    """An axis-aligned box obstacle."""

    minimum: np.ndarray
    maximum: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.minimum = _point(self.minimum)
        self.maximum = _point(self.maximum)

        if np.any(self.maximum < self.minimum):
            raise ValueError(
                "Box maximum must be greater than or equal "
                "to box minimum"
            )


@dataclass
class Collision:
    """One detected collision."""

    kind: str
    first: str
    second: str
    clearance: float

    @property
    def distance(self):
        """Alias for clearance."""
        return self.clearance


# ---------------------------------------------------------------------
# Distance calculations
# ---------------------------------------------------------------------

def point_to_segment_distance(point, start, end):
    """Return distance from a point to a line segment."""
    point = _point(point)
    start = _point(start)
    end = _point(end)

    direction = end - start
    length_squared = float(np.dot(direction, direction))

    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))

    parameter = float(
        np.dot(point - start, direction) / length_squared
    )

    parameter = float(np.clip(parameter, 0.0, 1.0))

    closest_point = start + parameter * direction

    return float(np.linalg.norm(point - closest_point))


def segment_to_segment_distance(
    first_start,
    first_end,
    second_start,
    second_end,
):
    """Return the shortest distance between two 3D segments."""
    p = _point(first_start)
    q = _point(second_start)

    u = _point(first_end) - p
    v = _point(second_end) - q
    w = p - q

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))

    epsilon = 1e-12

    # Both segments are points.
    if a <= epsilon and c <= epsilon:
        return float(np.linalg.norm(p - q))

    # First segment is a point.
    if a <= epsilon:
        return point_to_segment_distance(p, q, q + v)

    # Second segment is a point.
    if c <= epsilon:
        return point_to_segment_distance(q, p, p + u)

    denominator = a * c - b * b

    if denominator > epsilon:
        first_parameter = (
            b * e - c * d
        ) / denominator
    else:
        first_parameter = 0.0

    first_parameter = float(
        np.clip(first_parameter, 0.0, 1.0)
    )

    second_parameter = (
        b * first_parameter + e
    ) / c

    if second_parameter < 0.0:
        second_parameter = 0.0
        first_parameter = float(
            np.clip(-d / a, 0.0, 1.0)
        )

    elif second_parameter > 1.0:
        second_parameter = 1.0
        first_parameter = float(
            np.clip((b - d) / a, 0.0, 1.0)
        )

    else:
        second_parameter = float(
            np.clip(second_parameter, 0.0, 1.0)
        )

    first_closest = p + first_parameter * u
    second_closest = q + second_parameter * v

    return float(
        np.linalg.norm(first_closest - second_closest)
    )


def capsule_to_capsule_clearance(first, second):
    """Return clearance between two capsules."""
    centerline_distance = segment_to_segment_distance(
        first.start,
        first.end,
        second.start,
        second.end,
    )

    return (
        centerline_distance
        - first.radius
        - second.radius
    )


def capsule_to_sphere_clearance(capsule, sphere):
    """Return clearance between a capsule and sphere."""
    centerline_distance = point_to_segment_distance(
        sphere.center,
        capsule.start,
        capsule.end,
    )

    return (
        centerline_distance
        - capsule.radius
        - sphere.radius
    )


def point_to_box_distance(point, box):
    """Return distance from a point to an axis-aligned box."""
    point = _point(point)

    distance_below = np.maximum(
        box.minimum - point,
        0.0,
    )

    distance_above = np.maximum(
        point - box.maximum,
        0.0,
    )

    outside_distance = distance_below + distance_above

    return float(np.linalg.norm(outside_distance))


def capsule_to_box_clearance(capsule, box):
    """Return approximate clearance between a capsule and box."""
    segment = capsule.end - capsule.start
    length = float(np.linalg.norm(segment))

    if length == 0.0:
        return (
            point_to_box_distance(capsule.start, box)
            - capsule.radius
        )

    step_size = max(capsule.radius, 0.001)
    sample_count = max(
        20,
        int(np.ceil(length / step_size)) * 4,
    )

    parameters = np.linspace(
        0.0,
        1.0,
        sample_count,
    )

    distances = []

    for parameter in parameters:
        point = capsule.start + parameter * segment
        distance = point_to_box_distance(point, box)
        distances.append(distance)

    return float(
        min(distances) - capsule.radius
    )


# ---------------------------------------------------------------------
# Convert FK transforms to link capsules
# ---------------------------------------------------------------------

def capsules_from_fk(
    transforms,
    radii,
    names=None,
):
    """Create link capsules from FK transforms.

    Parameters
    ----------
    transforms:
        Sequence of 4x4 homogeneous transforms.

    radii:
        One radius for every link, or one scalar radius.

    names:
        Optional names for every link.

    Returns
    -------
    list[Capsule]
    """
    transforms = [
        _transform(value)
        for value in transforms
    ]

    if len(transforms) < 2:
        raise ValueError(
            "At least two transforms are required."
        )

    positions = [
        transform[:3, 3].copy()
        for transform in transforms
    ]

    link_count = len(positions) - 1

    if np.isscalar(radii):
        link_radii = [
            float(radii)
            for _ in range(link_count)
        ]
    else:
        link_radii = [
            float(value)
            for value in radii
        ]

        if len(link_radii) != link_count:
            raise ValueError(
                f"Expected {link_count} radii, "
                f"but received {len(link_radii)}"
            )

    for value in link_radii:
        _radius(value)

    if names is None:
        link_names = [
            f"link_{index}"
            for index in range(link_count)
        ]
    else:
        link_names = list(names)

        if len(link_names) != link_count:
            raise ValueError(
                f"Expected {link_count} names, "
                f"but received {len(link_names)}"
            )

    links = []

    for index in range(link_count):
        links.append(
            Capsule(
                start=positions[index],
                end=positions[index + 1],
                radius=link_radii[index],
                name=link_names[index],
            )
        )

    return links


# Backward-compatible alias.
capsules_from_transforms = capsules_from_fk


# ---------------------------------------------------------------------
# Collision checker
# ---------------------------------------------------------------------

class CollisionChecker:
    """Check robot self-collisions and obstacle collisions."""

    def __init__(
        self,
        link_radii,
        obstacles=None,
        ignore_adjacent_links=True,
    ):
        self.link_radii = link_radii
        self.obstacles = list(obstacles or [])
        self.ignore_adjacent_links = bool(
            ignore_adjacent_links
        )

        for obstacle in self.obstacles:
            if not isinstance(obstacle, (Sphere, Box)):
                raise TypeError(
                    "Each obstacle must be a Sphere or Box"
                )

    def check_self_collisions(self, links):
        """Check non-adjacent robot links."""
        collisions = []

        for first_index, second_index in combinations(
            range(len(links)),
            2,
        ):
            if (
                self.ignore_adjacent_links
                and second_index - first_index <= 1
            ):
                continue

            first = links[first_index]
            second = links[second_index]

            clearance = capsule_to_capsule_clearance(
                first,
                second,
            )

            if clearance <= 0.0:
                collisions.append(
                    Collision(
                        kind="self",
                        first=first.name,
                        second=second.name,
                        clearance=float(clearance),
                    )
                )

        return collisions

    def check_obstacle_collisions(
        self,
        links,
        obstacles=None,
    ):
        """Check every robot link against every obstacle."""
        if obstacles is None:
            obstacles = self.obstacles
        else:
            obstacles = list(obstacles)

        collisions = []

        for link_index, link in enumerate(links):
            link_name = link.name

            if not link_name:
                link_name = f"link_{link_index}"

            for obstacle_index, obstacle in enumerate(
                obstacles
            ):
                obstacle_name = getattr(
                    obstacle,
                    "name",
                    "",
                )

                if not obstacle_name:
                    obstacle_name = (
                        f"obstacle_{obstacle_index}"
                    )

                if isinstance(obstacle, Sphere):
                    clearance = capsule_to_sphere_clearance(
                        link,
                        obstacle,
                    )

                elif isinstance(obstacle, Box):
                    clearance = capsule_to_box_clearance(
                        link,
                        obstacle,
                    )

                else:
                    raise TypeError(
                        "Each obstacle must be a Sphere or Box"
                    )

                if clearance <= 0.0:
                    collisions.append(
                        Collision(
                            kind="obstacle",
                            first=link_name,
                            second=obstacle_name,
                            clearance=float(clearance),
                        )
                    )

        return collisions

    def check(
        self,
        transforms,
        obstacles=None,
        names=None,
    ):
        """Check FK transforms for all collision types."""
        links = capsules_from_fk(
            transforms=transforms,
            radii=self.link_radii,
            names=names,
        )

        self_collisions = self.check_self_collisions(
            links
        )

        obstacle_collisions = (
            self.check_obstacle_collisions(
                links=links,
                obstacles=obstacles,
            )
        )

        return self_collisions + obstacle_collisions

    def is_collision_free(
        self,
        transforms,
        obstacles=None,
        names=None,
    ):
        """Return True when no collision is detected."""
        collisions = self.check(
            transforms=transforms,
            obstacles=obstacles,
            names=names,
        )

        return len(collisions) == 0
