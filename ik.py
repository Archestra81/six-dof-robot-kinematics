import numpy as np
from scipy.spatial.transform import Rotation
from fk import forward_kinematics
from jacobian import geometric_jacobian

def pose_error(T_current, T_target):
    p_error = T_target[:3, 3] - T_current[:3, 3]
    
    R_current = Rotation.from_matrix(T_current[:3, :3])
    R_target = Rotation.from_matrix(T_target[:3, :3])
    
    R_error = R_target * R_current.inv()
    w_error = R_error.as_rotvec()
    
    return np.concatenate([p_error, w_error])

def ik_solve(
    joints,
    target,
    q_initial,
    max_iterations=500,
    position_tolerance=1e-4,
    orientation_tolerance=1e-4,
    damping=1e-2,
    step_size=1.0
):
    q = np.array(q_initial, dtype=float).copy()
    
    for iteration in range(max_iterations):
        T, transforms = forward_kinematics(joints, q)
        error = pose_error(T, target)
        
        position_norm = np.linalg.norm(error[:3])
        orientation_norm = np.linalg.norm(error[3:])
        
        if position_norm < position_tolerance and orientation_norm < orientation_tolerance:
            return {
                "q": q,
                "success": True,
                "iterations": iteration,
                "position_error": position_norm,
                "orientation_error": orientation_norm,
            }
            
        J = geometric_jacobian(joints, transforms)
        identity = np.eye(6)
        
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * identity,
            error,
        )
        
        q += step_size * dq
        
        for i, joint in enumerate(joints):
            q[i] = np.clip(q[i], joint.limits[0], joint.limits[1])
            
    return {
        "q": q,
        "success": False,
        "iterations": max_iterations,
        "position_error": position_norm,
        "orientation_error": orientation_norm,
    }
    