import numpy as np
def geometric_jacobian(joints, transforms):
    n = len(transforms) - 1
    J = np.zeros((6, n))
    o_n = transforms[-1][:3, 3]

    for i, joint in enumerate(joints):
        T_prev = transforms[i]
        z = T_prev[:3, 2]
        o = T_prev[:3, 3]
        
        if joint.joint_type == "revolute":
            J[:3, i] = np.cross(z, o_n - o)
            J[3:, i] = z
        else:
            J[:3, i] = z
            J[3:, i] = 0.0

    return J