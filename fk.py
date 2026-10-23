import numpy as np
from dh import dh_transform

def forward_kinematics(joints, q):
    T = np.eye(4)
    # ... rest of your code ...
    transforms = [T.copy()]

    for joint, qi in zip(joints, q):
        if joint.joint_type == "revolute":
            theta = qi + joint.theta_offset
            d = joint.d
        else:
            theta = joint.theta_offset
            d = qi + joint.d

        A = dh_transform(joint.a, joint.alpha, d, theta)
        T = T @ A
        transforms.append(T.copy())

    return T, transforms