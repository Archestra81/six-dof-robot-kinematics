import numpy as np

from robot_model import robot
from fk import forward_kinematics
from visualization import plot_robot


q = np.array([
    1.0,
    0.0,
    7.0,
    0.0,
    0.0,
    0.0,
])

T, transforms = forward_kinematics(robot, q)

print("End-effector position:")
print(T[:3, 3])

print("\nEnd-effector transformation matrix:")
print(T)

plot_robot(transforms)
fig = plot_robot(transforms)
fig.show()