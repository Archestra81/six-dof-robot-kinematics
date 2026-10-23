import numpy as np

def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,        d],
        [0.0,    0.0,     0.0,      1.0],
    ])
if __name__ == "__main__":
    # Replace dh_transform with your actual function name if it's different
    T = dh_transform(0.0, 0.0, 1.0, 0.0)
    print(T)