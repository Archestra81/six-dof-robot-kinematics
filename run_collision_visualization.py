import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Simple collision geometry
# ============================================================

class Sphere:
    def __init__(self, center, radius, name="sphere"):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.name = name

        if self.center.shape != (3,):
            raise ValueError(
                "Sphere center must contain exactly 3 values"
            )

        if self.radius < 0:
            raise ValueError(
                "Sphere radius cannot be negative"
            )


class Box:
    def __init__(self, minimum, maximum, name="box"):
        self.minimum = np.asarray(minimum, dtype=float)
        self.maximum = np.asarray(maximum, dtype=float)
        self.name = name

        if self.minimum.shape != (3,):
            raise ValueError(
                "Box minimum must contain exactly 3 values"
            )

        if self.maximum.shape != (3,):
            raise ValueError(
                "Box maximum must contain exactly 3 values"
            )

        if np.any(self.maximum < self.minimum):
            raise ValueError(
                "Box maximum must be greater than minimum"
            )


class Collision:
    def __init__(
        self,
        collision_type,
        first,
        second,
        clearance,
    ):
        self.kind = collision_type
        self.first = first
        self.second = second
        self.clearance = float(clearance)


# ============================================================
# Geometry calculations
# ============================================================

def point_to_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    length_squared = np.dot(direction, direction)

    if length_squared == 0:
        return float(np.linalg.norm(point - start))

    parameter = np.dot(point - start, direction)
    parameter = parameter / length_squared
    parameter = np.clip(parameter, 0.0, 1.0)

    closest_point = start + parameter * direction

    return float(np.linalg.norm(point - closest_point))


def point_to_box_distance(point, box):
    point = np.asarray(point, dtype=float)

    below = np.maximum(box.minimum - point, 0.0)
    above = np.maximum(point - box.maximum, 0.0)

    outside_distance = below + above

    return float(np.linalg.norm(outside_distance))


def segment_to_segment_distance(
    first_start,
    first_end,
    second_start,
    second_end,
):
    first_start = np.asarray(first_start, dtype=float)
    first_end = np.asarray(first_end, dtype=float)
    second_start = np.asarray(second_start, dtype=float)
    second_end = np.asarray(second_end, dtype=float)

    first_direction = first_end - first_start
    second_direction = second_end - second_start

    first_length = np.linalg.norm(first_direction)
    second_length = np.linalg.norm(second_direction)

    if first_length == 0:
        return point_to_segment_distance(
            first_start,
            second_start,
            second_end,
        )

    if second_length == 0:
        return point_to_segment_distance(
            second_start,
            first_start,
            first_end,
        )

    values = []

    for first_parameter in np.linspace(0.0, 1.0, 30):
        first_point = (
            first_start
            + first_parameter * first_direction
        )

        values.append(
            point_to_segment_distance(
                first_point,
                second_start,
                second_end,
            )
        )

    for second_parameter in np.linspace(0.0, 1.0, 30):
        second_point = (
            second_start
            + second_parameter * second_direction
        )

        values.append(
            point_to_segment_distance(
                second_point,
                first_start,
                first_end,
            )
        )

    return float(min(values))


# ============================================================
# Collision checker
# ============================================================

class CollisionChecker:
    def __init__(
        self,
        link_radii,
        obstacles=None,
    ):
        self.link_radii = list(link_radii)
        self.obstacles = [] if obstacles is None else list(
            obstacles
        )

    def check(self, transforms):
        transforms = [
            np.asarray(transform, dtype=float)
            for transform in transforms
        ]

        if len(transforms) < 2:
            raise ValueError(
                "At least two transforms are required"
            )

        for transform in transforms:
            if transform.shape != (4, 4):
                raise ValueError(
                    "Every transform must have shape (4, 4)"
                )

        link_count = len(transforms) - 1

        if len(self.link_radii) != link_count:
            raise ValueError(
                f"Expected {link_count} link radii, "
                f"but received {len(self.link_radii)}"
            )

        collisions = []

        # --------------------------------------------------------
        # Self-collision checks
        # --------------------------------------------------------
        for first_index in range(link_count):
            for second_index in range(
                first_index + 2,
                link_count,
            ):
                first_start = transforms[first_index][:3, 3]
                first_end = transforms[first_index + 1][:3, 3]

                second_start = transforms[second_index][:3, 3]
                second_end = transforms[second_index + 1][:3, 3]

                distance = segment_to_segment_distance(
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                )

                clearance = (
                    distance
                    - self.link_radii[first_index]
                    - self.link_radii[second_index]
                )

                if clearance <= 0:
                    collisions.append(
                        Collision(
                            "self",
                            f"link_{first_index}",
                            f"link_{second_index}",
                            clearance,
                        )
                    )

        # --------------------------------------------------------
        # Environment collision checks
        # --------------------------------------------------------
        for link_index in range(link_count):
            start = transforms[link_index][:3, 3]
            end = transforms[link_index + 1][:3, 3]
            radius = self.link_radii[link_index]

            for obstacle in self.obstacles:
                if isinstance(obstacle, Sphere):
                    distance = point_to_segment_distance(
                        obstacle.center,
                        start,
                        end,
                    )

                    clearance = (
                        distance
                        - radius
                        - obstacle.radius
                    )

                elif isinstance(obstacle, Box):
                    distance = min(
                        point_to_box_distance(
                            start,
                            obstacle,
                        ),
                        point_to_box_distance(
                            end,
                            obstacle,
                        ),
                    )

                    clearance = distance - radius

                else:
                    raise TypeError(
                        "Obstacle must be Sphere or Box"
                    )

                if clearance <= 0:
                    collisions.append(
                        Collision(
                            "obstacle",
                            f"link_{link_index}",
                            obstacle.name,
                            clearance,
                        )
                    )

        return collisions


# ============================================================
# Simple forward kinematics
# ============================================================

def dh_transform(a, alpha, d, theta):
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.array(
        [
            [
                cos_theta,
                -sin_theta * cos_alpha,
                sin_theta * sin_alpha,
                a * cos_theta,
            ],
            [
                sin_theta,
                cos_theta * cos_alpha,
                -cos_theta * sin_alpha,
                a * sin_theta,
            ],
            [
                0.0,
                sin_alpha,
                cos_alpha,
                d,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )


def forward_kinematics(q):
    q = np.asarray(q, dtype=float)

    if q.shape != (6,):
        raise ValueError(
            "q must contain exactly 6 joint angles"
        )

    a_values = [
        0.55,
        0.30,
        0.25,
        0.20,
        0.10,
        0.10,
    ]

    alpha_values = [
        np.pi / 2,
        0.0,
        0.0,
        np.pi / 2,
        -np.pi / 2,
        0.0,
    ]

    d_values = [
        0.99,
        0.00,
        0.00,
        0.00,
        0.00,
        0.10,
    ]

    transforms = [np.eye(4)]
    current = np.eye(4)

    for index in range(6):
        current = current @ dh_transform(
            a_values[index],
            alpha_values[index],
            d_values[index],
            q[index],
        )

        transforms.append(current.copy())

    return transforms


# ============================================================
# Visualization
# ============================================================

def draw_sphere(axis, sphere):
    center = sphere.center
    radius = sphere.radius

    u = np.linspace(
        0.0,
        3.0 * np.pi,
        30,
    )

    v = np.linspace(
        0.0,
        np.pi,
        20,
    )

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

    axis.plot_surface(
        x,
        y,
        z,
        color="orange",
        alpha=0.35,
    )


def draw_box(axis, box):
    x0, y0, z0 = box.minimum
    x1, y1, z1 = box.maximum

    vertices = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]

    edges = [
        (0, 5),
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

    for first, second in edges:
        start = vertices[first]
        end = vertices[second]

        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="orange",
            linewidth=2,
        )


def plot_robot(transforms, obstacles, collisions):
    points = np.array(
        [
            transform[:3, 3]
            for transform in transforms
        ]
    )

    collision_links = set()

    for collision in collisions:
        collision_links.add(collision.first)
        collision_links.add(collision.second)

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        link_name = f"link_{index}"

        if link_name in collision_links:
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
    )

    for obstacle in obstacles:
        if isinstance(obstacle, Sphere):
            draw_sphere(axis, obstacle)
        elif isinstance(obstacle, Box):
            draw_box(axis, obstacle)

    if collisions:
        axis.set_title(
            "Collision detected"
        )
    else:
        axis.set_title(
            "Collision-free robot"
        )

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    plt.show()


# ============================================================
# Main program
# ============================================================

def main():
    q = np.array(
        [
            0.4,
            0.6,
            0.7,
            7.0,
            0.0,
            0.8,
        ],
        dtype=float,
    )

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

    checker = CollisionChecker(
        link_radii=[
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        obstacles=obstacles,
    )

    transforms = forward_kinematics(q)
    collisions = checker.check(transforms)

    if collisions:
        print("COLLISION DETECTED")

        for collision in collisions:
            print(
                collision.kind,
                collision.first,
                collision.second,
                collision.clearance,
            )
    else:
        print("COLLISION FREE")

    plot_robot(
        transforms,
        obstacles,
        collisions,
    )


if __name__ == "__main__":
    main()