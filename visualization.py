import numpy as np
import matplotlib.pyplot as plt


def plot_robot(transforms):
    points = np.array([
        T[:3, 3]
        for T in transforms
    ])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        marker="o",
        linewidth=3,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("6-DOF Robot")

    plt.show()