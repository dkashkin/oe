import numpy as np
from scipy.optimize import linprog, minimize

def run_packing():
    """
    Maximizes the sum of radii for 26 circles in a unit square.
    
    The strategy uses a two-stage optimization:
    1. Linear Programming (LP): Evaluates multiple structural row-based and grid layouts 
       by finding the optimal radii for fixed center positions.
    2. Nonlinear Optimization (SLSQP): Refines both center positions and radii of the 
       top candidates to further improve the total sum.
    """
    n = 26
    
    def solve_lp(P):
        """Finds optimal radii for fixed centers using Linear Programming."""
        c = -np.ones(n)
        A = []
        b = []
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i], row[j] = 1.0, 1.0
                A.append(row)
                dist = np.linalg.norm(P[i] - P[j])
                # Small epsilon to ensure strict validity
                b.append(max(0, dist - 1e-12))
        
        bounds = []
        for i in range(n):
            x, y = P[i]
            d_wall = min(x, 1.0 - x, y, 1.0 - y)
            bounds.append((0, max(0, d_wall - 1e-12)))
            
        try:
            res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method='highs')
        except:
            res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds)
            
        if res.success and res.x is not None:
            return res.x, -res.fun
        return None, 0

    # 1. Generate structural seeds
    seeds = []
    # Candidate row configurations (Total circles = 26)
    configs = [
        [5, 5, 6, 5, 5],
        [6, 7, 7, 6],
        [4, 5, 4, 5, 4, 4],
        [5, 6, 5, 6, 4],
        [5, 5, 5, 5, 6],
        [6, 5, 5, 5, 5]
    ]
    
    for cfg in configs:
        num_rows = len(cfg)
        # Try different y-margins and stagger offsets to cover different packing basins
        for margin in [0.08, 0.1, 0.12]:
            for stagger_offset in [0.0, 0.02, 0.04]:
                pts = []
                y_coords = np.linspace(margin, 1.0 - margin, num_rows)
                for r_idx, count in enumerate(cfg):
                    y = y_coords[r_idx]
                    # Standard horizontal spacing
                    x_vals = np.linspace(margin, 1.0 - margin, count)
                    if r_idx % 2 == 1:
                        # Stagger rows to encourage hexagonal density
                        x_vals = np.clip(x_vals + stagger_offset, 0.01, 0.99)
                    for x in x_vals:
                        pts.append([x, y])
                seeds.append(np.array(pts[:n]))
            
    # Add a 5x5 grid seed with one circle tucked in a gap
    grid_seed = [[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)]
    grid_seed.append([0.5, 0.5]) # Optimizer will push it out of overlap
    seeds.append(np.array(grid_seed[:n]))
    
    # 2. Evaluate all seeds with LP to find the best starting points for refinement
    results = []
    for s in seeds:
        radii, total = solve_lp(s)
        if radii is not None:
            results.append((total, s, radii))
    
    if not results:
        # Emergency fallback
        return np.random.rand(n, 2), np.zeros(n), 0.0
        
    results.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Refine top layouts using Nonlinear Optimization (SLSQP)
    # Optimization variables v: [x_0..x_n-1, y_0..y_n-1, r_0..r_n-1]
    def objective(v):
        return -np.sum(v[2*n:])
        
    def jac_objective(v):
        j = np.zeros(3*n)
        j[2*n:] = -1.0
        return j
        
    def constraints(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        res = []
        # Non-overlapping: dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                res.append((x[i] - x[j])**2 + (y[i] - y[j])**2 - (r[i] + r[j])**2)
        # Boundary: r_i <= x, 1-x, y, 1-y
        for i in range(n):
            res.extend([x[i] - r[i], 1.0 - x[i] - r[i], y[i] - r[i], 1.0 - y[i] - r[i]])
        return np.array(res)

    def jac_constraints(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        num_non_overlap = n * (n - 1) // 2
        J = np.zeros((num_non_overlap + 4 * n, 3 * n))
        row = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy, dr_sum = x[i] - x[j], y[i] - y[j], r[i] + r[j]
                J[row, i], J[row, j] = 2 * dx, -2 * dx
                J[row, n + i], J[row, n + j] = 2 * dy, -2 * dy
                J[row, 2 * n + i], J[row, 2 * n + j] = -2 * dr_sum, -2 * dr_sum
                row += 1
        for i in range(n):
            # x[i] - r[i] >= 0
            J[row, i], J[row, 2*n + i] = 1, -1; row += 1
            # 1 - x[i] - r[i] >= 0
            J[row, i], J[row, 2*n + i] = -1, -1; row += 1
            # y[i] - r[i] >= 0
            J[row, n + i], J[row, 2*n + i] = 1, -1; row += 1
            # 1 - y[i] - r[i] >= 0
            J[row, n + i], J[row, 2*n + i] = -1, -1; row += 1
        return J

    best_total_sum, best_centers, best_radii = results[0]
    
    # Limit refinement to top promising seeds to manage execution time
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    for i in range(min(8, len(results))):
        _, s_centers, s_radii = results[i]
        # Jitter slightly to break symmetry if necessary
        x0 = np.concatenate([s_centers[:, 0], s_centers[:, 1], s_radii])
        
        opt_res = minimize(
            objective, x0, jac=jac_objective, method='SLSQP', bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints, 'jac': jac_constraints},
            options={'maxiter': 300, 'ftol': 1e-9}
        )
        
        if opt_res.success and -opt_res.fun > best_total_sum:
            best_total_sum = -opt_res.fun
            best_centers = np.column_stack((opt_res.x[:n], opt_res.x[n:2*n]))
            best_radii = opt_res.x[2*n:]

    # Final check: Ensure strict validity by subtracting a minute epsilon
    final_radii = best_radii - 1e-11
    final_centers = np.clip(best_centers, 0, 1)
    final_sum = float(np.sum(final_radii))
    
    return final_centers, final_radii, final_sum

def construct_packing():
    """Alias for consistency with different evaluation call patterns."""
    return run_packing()

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Total sum of radii: {sum_r:.5f}")