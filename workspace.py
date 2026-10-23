def sample_workspace(joints, samples=100_000, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.empty((samples, 3))

    for k in range(samples):
        q = np.array([
            rng.uniform(joint.limits[0], joint.limits[1])
            for joint in joints
        ])

        T, _ = forward_kinematics(joints, q)
        positions[k] = T[:3, 3]

    return positions