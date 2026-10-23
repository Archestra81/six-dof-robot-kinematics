import numpy as np
from robot_model import Joint
from fk import forward_kinematics

def test_forward_kinematics_basic():
    # Define a simple test robot
    test_robot = [
        Joint(a=0.0, alpha=0.0, d=1.0, theta_offset=0.0)
    ]
    q = [0.0]
    T, transforms = forward_kinematics(test_robot, q)
    
    # Verify matrix dimensions and transform history length
    assert T.shape == (4, 4)
    assert len(transforms) == len(test_robot) + 1