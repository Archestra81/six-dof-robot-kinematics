from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union
import numpy as np

# --- Primitive Data Structures ---

@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float
    name: str = "sphere"

@dataclass(frozen=True)
class Box:
    minimum: np.ndarray
    maximum: np.ndarray
    name: str = "box"

@dataclass(frozen=True)
class Capsule:
    p1: np.ndarray
    p2: np.ndarray
    radius: float
    name: str = "capsule"

@dataclass(frozen=True)
class CollisionResult:
    kind: str
    first: str
    second: str
    distance: float


def _as_point(value: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"Expected a 3D point, got shape {point.shape}")
    return point


def _validate_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a homogenous 4x4 transform, got {transform.shape}")
    return transform


def capsules_from_fk(transforms: Union[np.ndarray, Sequence[np.ndarray]], link_radii: Sequence[float]) -> list[Capsule]:
    # Unpack 3D numpy array or sequence into individual 4x4 transforms safely
    if isinstance(transforms, np.ndarray) and transforms.ndim == 3:
        transform_list = [transforms[i] for i in range(transforms.shape[0])]
    else:
        transform_list = list(transforms)

    validated_transforms = [_validate_transform(t) for t in transform_list]
    
    capsules = []
    for i in range(len(validated_transforms) - 1):
        p1 = validated_transforms[i][:3, 3]
        p2 = validated_transforms[i + 1][:3, 3]
        radius = link_radii[i] if i < len(link_radii) else link_radii[-1]
        capsules.append(Capsule(p1=p1, p2=p2, radius=radius, name=f"link_{i}"))
        
    return capsules


class CollisionChecker:
    def __init__(self, link_radii: Sequence[float], obstacles: list[Union[Sphere, Box]]):
        self.link_radii = link_radii
        self.obstacles = obstacles

    def check(self, transforms: Union[np.ndarray, Sequence[np.ndarray]]) -> list[CollisionResult]:
        links = capsules_from_fk(transforms, self.link_radii)
        collisions = []

        for link in links:
            for obs in self.obstacles:
                if isinstance(obs, Sphere):
                    # Basic distance check placeholder for demonstration
                    dist = float(np.linalg.norm(link.p2 - obs.center) - (link.radius + obs.radius))
                    if dist <= 0.0:
                        collisions.append(CollisionResult(
                            kind="link_obstacle",
                            first=link.name,
                            second=obs.name,
                            distance=dist
                        ))
                elif isinstance(obs, Box):
                    # Bounding check or basic clearance placeholder
                    pass

        return collisions
    import numpy as np

from collision import CollisionChecker, capsules_from_fk
from fk import forward_kinematics
from robot_model import robot
from visualization import plot_collision_state


q = np.zeros(6)

transforms = forward_kinematics(robot, q)

obstacles = [
    # Your Sphere and Box objects here.
]

checker = CollisionChecker(
    link_radii=[0.05] * 5,
    obstacles=obstacles,
)

links = capsules_from_fk(
    transforms,
    radii=[0.05] * 5,
    names=[
        "link_1",
        "link_2",
        "link_3",
        "link_4",
        "link_5",
    ],
)

collisions = checker.check(
    transforms,
    obstacles=obstacles,
)

figure = plot_collision_state(
    links,
    obstacles,
    collisions,
)

figure.show()