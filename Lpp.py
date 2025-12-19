import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from mpl_toolkits.mplot3d import Axes3D

def solve_3d_lpp():
    print("--- 3D LPP Solver (x, y, z) ---")
    
    # 1. Objective Function
    cx = float(input("Z coefficient for x: "))
    cy = float(input("Z coefficient for y: "))
    cz = float(input("Z coefficient for z: "))
    obj = [-cx, -cy, -cz] # Negate for maximization

    # 2. Constraints
    n = int(input("Number of constraints (ax + by + cz <= d): "))
    lhs = []
    rhs = []
    for i in range(n):
        print(f"\nConstraint {i+1}:")
        lhs.append([float(input("  a: ")), float(input("  b: ")), float(input("  c: "))])
        rhs.append(float(input("  d: ")))

    # 3. Solve Numerically (Simplex)
    res = linprog(c=obj, A_ub=lhs, b_ub=rhs, bounds=(0, None), method='highs')
    
    if not res.success:
        print("No feasible solution found.")
        return

    print(f"\nOptimal Value (Max Z): {-res.fun:.2f}")
    print(f"Coordinates: x={res.x[0]:.2f}, y={res.x[1]:.2f}, z={res.x[2]:.2f}")

    # 4. 3D Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid for the planes
    limit = max(rhs) / 1.0 # Rough scaling
    d1 = np.linspace(0, limit, 10)
    d2 = np.linspace(0, limit, 10)
    X, Y = np.meshgrid(d1, d2)

    colors = ['blue', 'red', 'green', 'yellow', 'purple']
    for i in range(len(lhs)):
        a, b, c = lhs[i]
        d = rhs[i]
        # Plane equation: z = (d - ax - by) / c
        if c != 0:
            Z_plane = (d - a*X - b*Y) / c
            Z_plane[Z_plane < 0] = np.nan # Don't plot below ground
            ax.plot_surface(X, Y, Z_plane, alpha=0.3, color=colors[i % len(colors)])

    # Mark the Optimal Point
    ax.scatter(res.x[0], res.x[1], res.x[2], color='black', s=100, label='Optimal Point')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title(f"3D LPP: Optimal Z = {-res.fun:.2f}")
    plt.show()

if __name__ == "__main__":
    solve_3d_lpp() 