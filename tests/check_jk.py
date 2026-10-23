import numpy as np

from robot_model import robot
from fk import forward_kinematics
from jacobian import geometric_jacobian


q = np.array([
    0.2,
    0.3,
    0.2,
    0.1,
    0.2,
    0.1,
])

T, transforms = forward_kinematics(robot, q)
J = geometric_jacobian(robot, transforms)

print("Jacobian:")
print(J)

print("\nJacobian shape:")
print(J.shape)