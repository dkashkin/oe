# EVOLVE-BLOCK-START
"""
Expert Constructor-based Circle Packing for n=26 Circles in a Unit Square.
The goal is to maximize the sum of radii of 26 non-overlapping circles.
This implementation uses several high-density candidate arrangements (hexagonal,
sunflower, and relaxed grids) and then solves a Linear Program (LP) for each 
arrangement to find the optimal set of radii that maximizes their sum.
"""

import numpy as np
from scipy.optimize import linprog

def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that maximizes the sum of their radii.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1
    
    # 1. Define high-potential center patterns
    configs = []
    
    # Pattern A: Sunflower Spirals (Vogel's model)
    # Good for distributing points uniformly in many container shapes.
    def generate_sunflower(n, scale=0.45):
        phi = (1 + np.sqrt(5)) / 2
        golden_angle = 2 * np.pi * (1 - 1/phi)
        centers = []
        for i in range(n):
            r = np.sqrt(i + 0.5) / np.sqrt(n)
            theta = i * golden_angle
            x = 0.5 + scale * r * np.cos(theta)
            y = 0.5 + scale * r * np.sin(theta)
            centers.append([x, y])
        return np.clip(np.array(centers), 0.01, 0.99)

    # Pattern B: Hexagonal Row-based Layouts
    # Hexagonal packing is the densest possible in infinite space.
    def generate_rows(counts, pad_x=0.1, pad_y=0.1):
        y_coords = np.linspace(pad_y, 1.0 - pad_y, len(counts))
        centers = []
        for i, count in enumerate(counts):
            x_coords = np.linspace(pad_x, 1.0 - pad_x, count)
            for x in x_coords:
                centers.append([x, y_coords[i]])
        return np.array(centers)

    def generate_staggered_rows(counts, pad=0.1):
        y_coords = np.linspace(pad, 1.0 - pad, len(counts))
        centers = []
        for i, count in enumerate(counts):
            # Apply slight horizontal stagger to improve interlocking
            offset = 0.02 if i % 2 == 1 else 0.0
            x_coords = np.linspace(pad + offset, 1.0 - pad - offset, count)
            for x in x_coords:
                centers.append([x, y_coords[i]])
        return np.array(centers)

    # Pattern C: Deterministic Force-Directed Relaxation
    # Spreads centers out to find better positions for the LP to exploit.
    def relax_centers(centers, steps=40):
        c = centers.copy()
        n_pts = len(c)
        for _ in range(steps):
            forces = np.zeros_like(c)
            # Mutual repulsion
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    diff = c[i] - c[j]
                    dist_sq = np.sum(diff**2) + 1e-6
                    dist = np.sqrt(dist_sq)
                    force = (diff / dist) * (0.01 / dist_sq)
                    forces[i] += force
                    forces[j] -= force
                # Boundary repulsion
                forces[i, 0] += 0.005 / (c[i, 0]**2 + 1e-5)
                forces[i, 0] -= 0.005 / ((1.0 - c[i, 0])**2 + 1e-5)
                forces[i, 1] += 0.005 / (c[i, 1]**2 + 1e-5)
                forces[i, 1] -= 0.005 / ((1.0 - c[i, 1])**2 + 1e-5)
            c += 0.002 * forces
            c = np.clip(c, 0.01, 0.99)
        return c

    # Add layouts to candidates
    configs.append(generate_sunflower(n, 0.44))
    configs.append(generate_sunflower(n, 0.47))
    
    hex_patterns = [[5, 6, 5, 6, 4], [6, 5, 4, 5, 6], [5, 5, 6, 5, 5], [6, 5, 5, 5, 5]]
    for counts in hex_patterns:
        for p in [0.08, 0.1, 0.12]:
            configs.append(generate_rows(counts, p, p))
    
    configs.append(generate_staggered_rows([5, 5, 6, 5, 5], 0.1))
    
    # Add a "disturbed" 5x5 grid + 1
    grid_25 = generate_rows([5, 5, 5, 5, 5], 0.1, 0.1)
    grid_26 = np.vstack([grid_25, [0.5, 0.5]])
    configs.append(relax_centers(grid_26))
    
    # 2. Optimize radii for each configuration using Linear Programming
    for c_np in configs:
        if len(c_np) != n: continue
        radii = compute_max_radii(c_np)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = c_np
            best_radii = radii
            
    return best_centers, best_radii, best_sum

def compute_max_radii(centers):
    """
    Computes optimal radii for fixed centers by maximizing sum(r_i)
    subject to non-overlap and boundary constraints.
    """
    n = len(centers)
    # Objective: Maximize sum of radii -> Minimize negative sum
    obj = -np.ones(n)
    
    # Non-overlap constraints: r_i + r_j <= distance(c_i, c_j)
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
    
    # Boundary constraints: r_i <= dist(c_i, wall)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0, max_r))
        
    # Solve the Linear Program
    try:
        # Use 'highs' method for efficiency if available
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # Ensure strict adherence to constraints by applying a tiny safety margin
            final_radii = np.maximum(res.x - 1e-10, 0.0)
            return final_radii
    except Exception:
        pass
        
    # Fallback: Simple robust inflation if LP fails
    radii = np.array([min(x, 1.0-x, y, 1.0-y) for x, y in centers])
    for _ in range(20):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    scale = (d - 1e-11) / (radii[i] + radii[j] + 1e-12)
                    radii[i] *= scale
                    radii[j] *= scale
    return radii
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")