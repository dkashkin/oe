import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START
"""
Expert constructor for maximizing the sum of radii of 26 circles in a unit square.
This implementation uses a variety of geometric seeds (staggered hexagonal grids, 
Vogel spirals, and perturbed 5x5 grids), refines their positions with a 
force-directed relaxation, and then solves a Linear Programming problem to find 
the optimal radii for each configuration.
"""

def construct_packing():
    """
    Constructs an arrangement of 26 circles in a unit square that maximizes the sum of radii.
    Returns: (centers, radii, sum_radii)
    """
    n = 26
    phi = (1 + 5**0.5) / 2
    seeds = []
    
    # 1. Sunflower / Vogel Spiral Configurations (diverse scales)
    for scale in [0.44, 0.46, 0.48]:
        c = []
        for i in range(n):
            # r = scale * sqrt(i+0.5)/sqrt(n), theta = golden angle
            r = scale * np.sqrt(i + 0.5) / np.sqrt(n)
            theta = 2 * np.pi * i / phi**2
            c.append([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
        seeds.append(np.array(c))
        
    # 2. Hexagonal Row-based Configurations
    # These patterns are designed such that the sum of circles per row is exactly 26.
    row_patterns = [
        [5, 6, 5, 6, 4], 
        [6, 5, 6, 5, 4], 
        [4, 6, 6, 6, 4], 
        [5, 5, 6, 5, 5], 
        [4, 5, 4, 5, 4, 4]
    ]
    for p in row_patterns:
        for pad in [0.08, 0.1]:
            # Normal row grid
            c1, ys = [], np.linspace(pad, 1 - pad, len(p))
            for i, count in enumerate(p):
                xs = np.linspace(pad, 1 - pad, count)
                for x in xs: c1.append([x, ys[i]])
            if len(c1) == n: seeds.append(np.array(c1))
            
            # Staggered row grid (interlocking centers)
            c2 = []
            for i, count in enumerate(p):
                offset = 0.02 if i % 2 == 1 else 0
                xs = np.linspace(pad + offset, 1 - pad - offset, count)
                for x in xs: c2.append([x, ys[i]])
            if len(c2) == n: seeds.append(np.array(c2))
            
    # 3. Modified 5x5 Grid (25 circles) + 1 additional circle in a gap
    grid_5x5 = [[x, y] for x in np.linspace(0.1, 0.9, 5) for y in np.linspace(0.1, 0.9, 5)]
    grid_26 = grid_5x5 + [[0.2, 0.2]] # Start the 26th circle in a gap
    seeds.append(np.array(grid_26))

    def relax_centers(pts, steps=80):
        """Refines center positions using short-range mutual repulsion and boundary forces."""
        pts = pts.copy()
        for _ in range(steps):
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=-1) + np.eye(n)
            # Short-range 1/r^4 force for hard-sphere-like interaction
            forces = np.sum(diffs / (dists**5)[..., np.newaxis], axis=1)
            # Subtle boundary repulsion to keep points from getting stuck at the very edge
            forces[:, 0] += 0.001 * (1/pts[:,0]**2 - 1/(1-pts[:,0])**2)
            forces[:, 1] += 0.001 * (1/pts[:,1]**2 - 1/(1-pts[:,1])**2)
            pts += 0.0002 * forces
            pts = np.clip(pts, 0.001, 0.999)
        return pts

    best_sum, best_centers, best_radii = -1, None, None
    
    # Evaluate all seeds (raw and relaxed) using Linear Programming to find optimal radii
    for seed_c in seeds:
        # Test both the original pattern and its force-relaxed variation
        for candidate_c in [seed_c, relax_centers(seed_c)]:
            # Radius Optimization Problem:
            # Maximize: sum(r_i)
            # Subject to: r_i + r_j <= distance(c_i, c_j)
            #             0 <= r_i <= distance(c_i, boundary)
            num = len(candidate_c)
            obj = -np.ones(num)
            A_ub, b_ub = [], []
            for i in range(num):
                for j in range(i + 1, num):
                    dist = np.linalg.norm(candidate_c[i] - candidate_c[j])
                    row = np.zeros(num)
                    row[i], row[j] = 1, 1
                    A_ub.append(row)
                    b_ub.append(dist)
            bounds = [(0, min(x, 1-x, y, 1-y)) for x, y in candidate_c]
            
            try:
                # Use HiGHS solver for high performance and numerical stability
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    current_sum = np.sum(res.x)
                    if current_sum > best_sum:
                        best_sum, best_centers, best_radii = current_sum, candidate_c, res.x
            except:
                continue
                
    # Final strictly enforced validation to prevent any floating point constraint violations
    eps = 1e-10
    final_radii = best_radii - eps
    for i in range(n):
        x, y = best_centers[i]
        final_radii[i] = min(final_radii[i], x - eps, 1 - x - eps, y - eps, 1 - y - eps)
        final_radii[i] = max(0, final_radii[i])
        
    for _ in range(10): # Iterative overlap cleanup
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(best_centers[i] - best_centers[j])
                if final_radii[i] + final_radii[j] > dist:
                    overlap = (final_radii[i] + final_radii[j] - dist + eps) / 2
                    final_radii[i] = max(0, final_radii[i] - overlap)
                    final_radii[j] = max(0, final_radii[j] - overlap)

    return best_centers, final_radii, np.sum(final_radii)

# EVOLVE-BLOCK-END

def run_packing():
    """Execute the circle packing constructor for n=26"""
    return construct_packing()

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii for n=26: {sum_radii:.6f}")