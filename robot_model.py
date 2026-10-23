from dataclasses import dataclass
import numpy as np

@dataclass
class Joint:
    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0
    limits: tuple = (-np.pi, np.pi)
    joint_type: str = "revolute"
# Create the 6-DOF robot model instance (list of Joint objects)
robot = [
    Joint(a=0.0, alpha=np.pi/2, d=0.33, theta_offset=0.0),
    Joint(a=0.425, alpha=0.0, d=0.0, theta_offset=0.0),
    Joint(a=0.392, alpha=0.0, d=0.0, theta_offset=0.0),
    Joint(a=0.0, alpha=np.pi/2, d=0.109, theta_offset=0.0),
    Joint(a=0.0, alpha=-np.pi/2, d=0.094, theta_offset=0.0),
    Joint(a=0.0, alpha=0.0, d=0.082, theta_offset=0.0)
]