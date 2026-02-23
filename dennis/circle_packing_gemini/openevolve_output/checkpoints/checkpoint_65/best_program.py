import numpy as np
from scipy.optimize import linprog

def run_packing():
    """
    Constructs an arrangement of 26 circles in a unit square to maximize the sum of radii.
    
    The strategy utilizes several candidate structural layouts (primarily row-based 
    staggered configurations) and then applies Linear Programming (LP) to determine 
    the optimal radii for those fixed centers. This ensures that the circles 
    maximize the total sum while strictly adhering to non-overlap and boundary constraints.
    """
    n = 26
    
    def solve_lp(P):
        """
        Solves a Linear Program to maximize the sum of radii for a fixed set of centers.
        Objective: Maximize sum(r_i)
        Constraints:
            r_i + r_j <= distance between center_i and center_j
            r_i <= distance from center_i to any wall
        """
        # Objective: minimize -sum(r_i) which is maximizing sum(r_i)
        c_obj = -np.ones(n)
        
        # Non-overlapping constraints: r_i + r_j <= dist(C_i, C_j)
        num_constraints = n * (n - 1) // 2
        A_ub = np.zeros((num_constraints, n))
        b_ub = np.zeros(num_constraints)
        
        curr = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(P[i] - P[j])
                A_ub[curr, i] = 1.0
                A_ub[curr, j] = 1.0
                # Subtract tiny epsilon to ensure strict non-overlap and account for float precision
                b_ub[curr] = max(0, dist - 1e-12)
                curr += 1
        
        # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
        bounds = []
        for i in range(n):
            x, y = P[i]
            d_wall = min(x, 1.0 - x, y, 1.0 - y)
            # Ensure radius is non-negative and fits within the square
            bounds.append((0, max(0, d_wall - 1e-12)))
            
        try:
            # Use 'highs' method for efficiency if available (Scipy >= 1.5.0)
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        except Exception:
            # Fallback for older Scipy versions or environments
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
            
        if res.success:
            return res.x, -res.fun
        return np.zeros(n), 0.0

    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Define several candidate staggered row configurations to evaluate
    # These represent row counts that total 26 circles
    configs = [
        [5, 5, 6, 5, 5],
        [5, 5, 5, 5, 6],
        [6, 5, 5, 5, 5],
        [4, 5, 4, 5, 4, 4],
        [5, 4, 5, 4, 5, 3],
        [5, 6, 5, 6, 4],
        [4, 6, 6, 6, 4]
    ]

    seeds = []
    for cfg in configs:
        pts = []
        rows = len(cfg)
        for i, nx in enumerate(cfg):
            # Distribute rows in the y-dimension from 0.1 to 0.9
            y = 0.1 + 0.8 * i / (rows - 1)
            for j in range(nx):
                # Standard horizontal spacing
                x = 0.1 + 0.8 * j / (nx - 1)
                # Stagger alternate rows to achieve hexagonal-like efficiency
                if i % 2 == 1:
                    x = 0.12 + 0.76 * j / (nx - 1)
                pts.append([x, y])
        seeds.append(np.array(pts[:n]))
    
    # Add a grid-based seed: 5x5 grid plus one circle tucked in a gap
    grid_seed = [[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)]
    grid_seed.append([0.2, 0.2]) # Placing 26th circle at an internal gap
    seeds.append(np.array(grid_seed[:n]))

    # Evaluate all candidate structural seeds using LP
    for p_arr in seeds:
        # Clip centers to ensure they are strictly within the unit square
        p_arr = np.clip(p_arr, 0.001, 0.999)
        radii, current_sum = solve_lp(p_arr)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_radii = radii
            best_centers = p_arr

    # Return the configuration with the highest sum of radii found
    final_centers = np.array(best_centers)
    final_radii = np.array(best_radii)
    final_sum = float(np.sum(final_radii))
    
    return final_centers, final_radii, final_sum

def construct_packing():
    """Alias for run_packing to ensure compatibility with different evaluator calls."""
    return run_packing()