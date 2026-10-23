# 6-DOF Robot Collision Visualization (Python)

This project is a small Python toolkit to visualize a 6-DOF robot arm in 3D and check simple collisions with obstacles.

It uses:
- Forward kinematics with standard Denavit–Hartenberg (DH) parameters.
- Capsule-based link geometry (thick line segments).
- Sphere and axis-aligned box obstacles.
- NumPy for math and Matplotlib for 3D plotting.

The goal is educational: understand how FK relates to collision detection and see the result in 3D.

---

## Project structure

```text
your_project/
│
├── run_collision_visualization.py  # Main script: FK + collision + visualization
└── (optional) other modules        # e.g. collision.py, visualization.py, robot_model.py
```

> Note: In the simplified version, `run_collision_visualization.py` is standalone.  
> It defines its own `Sphere`, `Box`, `CollisionChecker`, FK, and plotting code.

---

## How it works

1. **Joint configuration**  
   The script defines a 6-element joint vector `q` (joint angles in radians):

   ```python
   q = np.array(
       [
           0.0,
           0.0,
           0.0,
           0.0,
           0.0,
           0.0,
       ],
       dtype=float,
   )
   ```

2. **Forward kinematics**  
   `forward_kinematics(q)` computes a sequence of 4×4 homogeneous transforms.  
   There are 7 transforms total (base + 6 joints). These are used to extract the 3D positions of the joints. Forward kinematics is a standard tool for robot motion prediction and collision detection. [web:170]

3. **Collision geometry**  
   - Each robot link is modeled as a capsule: a line segment between two joint positions plus a radius.
   - Obstacles are:
     - `Sphere(center, radius)`
     - `Box(minimum, maximum)`

4. **Collision checking**  
   The `CollisionChecker`:
   - Computes distances between link capsules and obstacles.
   - Computes distances between non-adjacent link capsules (self-collision).
   - Converts distances to clearances:

     ```text
     clearance = distance - (link_radius + obstacle_radius)
     ```

   - If `clearance <= 0`, it reports a collision.

5. **Visualization**  
   - Matplotlib’s 3D axes plot the robot links and joints. [web:65][web:155]
   - Links in collision are drawn in red; others are blue.
   - Obstacles are drawn as:
     - Orange spheres (surface).
     - Orange box edges.

---

## Running the project

1. Install Python dependencies:

   ```bash
   python -m pip install numpy matplotlib
   ```

2. Run the main script:

   ```bash
   python run_collision_visualization.py
   ```

   On Windows you can also use:

   ```bash
   py run_collision_visualization.py
   ```

   On Linux/macOS:

   ```bash
   python3 run_collision_visualization.py
   ```

3. A Matplotlib 3D window opens:
   - Robot links and joints are shown.
   - Obstacles appear around the robot.
   - The console prints either:

     ```text
     COLLISION DETECTED
     ```

     or:

     ```text
     COLLISION FREE
     ```

---

## Changing the robot pose

Edit the `q` definition in `run_collision_visualization.py`:

```python
q = np.array(
    [
        0.5,
        -0.3,
        0.8,
        0.2,
        -0.4,
        0.1,
    ],
    dtype=float,
)
```

Values are in radians. If you prefer degrees:

```python
q = np.radians(
    [
        30,
        -20,
        45,
        10,
        -15,
        5,
    ]
)
```

---

## Changing obstacles

### Sphere

```python
Sphere(
    center=np.array([0.40, 0.00, 0.20]),
    radius=0.10,
    name="sphere_1",
)
```

Change `center` and `radius` to move or resize the sphere.

### Box

```python
Box(
    minimum=np.array([0.20, -0.20, -0.05]),
    maximum=np.array([0.30, 0.20, 0.10]),
    name="box_1",
)
```

Change `minimum` and `maximum` to reposition/resize the box.

---

## Changing link thickness

`link_radii` defines the collision radius for each link:

```python
link_radii = [
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
]
```

Increasing these radii makes collisions more likely (thicker links).

---

## Typical use cases

- Teaching forward kinematics and collision detection.
- Visualizing how obstacles affect robot reachability.
- Experimenting with simple collision-avoidance ideas.

This is not meant as a production-grade real-time collision engine; it prioritizes clarity over performance.

---

