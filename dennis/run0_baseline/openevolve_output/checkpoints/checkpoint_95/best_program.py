import numpy as np
from scipy.optimize import linprog, minimize

def run_packing():
    """
    Maximizes the sum of radii for 26 circles in a unit square using a two-stage strategy:
    1. Systematic generation of structural layouts (seeds) based on hexagonal row counts.
    2. Linear Programming (LP) to determine optimal radii for fixed centers to filter seeds.
    3. Nonlinear Optimization (SLSQP) to refine both centers and radii of the top candidates.
    """
    n = 26
    np.random.seed(42)

    def solve_lp(P):
        """Calculates optimal radii for fixed centers using Linear Programming."""
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i], row[j] = 1.0, 1.0
                A_ub.append(row)
                dist = np.linalg.norm(P[i] - P[j])
                b_ub.append(max(0, dist - 1e-12))
        
        bounds = []
        for i in range(n):
            x, y = P[i]
            d_wall = min(x, 1.0 - x, y, 1.0 - y)
            bounds.append((0, max(0, d_wall - 1e-12)))
            
        try:
            res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
            if res.success and res.x is not None:
                return res.x, -res.fun
        except:
            pass
        return None, 0

    # 1. Structural Seed Generation
    # Various row configurations that total 26 circles
    configs = [
        [5, 5, 6, 5, 5], [6, 7, 7, 6], [4, 6, 6, 6, 4],
        [5, 6, 5, 6, 4], [4, 5, 4, 5, 4, 4], [5, 5, 5, 5, 6],
        [6, 6, 7, 7], [4, 4, 5, 4, 4, 5], [4, 4, 4, 4, 5, 5]
    ]
    
    seeds = []
    for cfg in configs:
        num_rows = len(cfg)
        # Vary margins and staggers to cover different packing densities
        for margin in [0.07, 0.09, 0.11]:
            for stagger in [0.0, 0.03]:
                pts = []
                for r_idx, count in enumerate(cfg):
                    y = margin + (1.0 - 2.0 * margin) * r_idx / (num_rows - 1)
                    x_vals = np.linspace(margin, 1.0 - margin, count)
                    if r_idx % 2 == 1:
                        x_vals = np.clip(x_vals + stagger, margin/2, 1.0 - margin/2)
                    for x in x_vals:
                        pts.append([x, y])
                seeds.append(np.array(pts[:n]))
    
    # 2. Evaluate all structural seeds using LP to find the best starting points
    candidate_results = []
    for s in seeds:
        radii, total = solve_lp(s)
        if radii is not None:
            candidate_results.append((total, s, radii))
    
    candidate_results.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Refinement using Nonlinear Optimization (SLSQP)
    def objective(v):
        return -np.sum(v[2*n:])
        
    def jac_objective(v):
        j = np.zeros(3*n)
        j[2*n:] = -1.0
        return j
        
    def constraints(v):
        x, y, r = v[0:n], v[n:2*n], v[2*n:3*n]
        res = []
        # Non-overlapping: dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                res.append((x[i] - x[j])**2 + (y[i] - y[j])**2 - (r[i] + r[j])**2)
        # Boundary: r_i <= wall dist
        for i in range(n):
            res.extend([x[i] - r[i], 1.0 - x[i] - r[i], y[i] - r[i], 1.0 - y[i] - r[i]])
        return np.array(res)

    def jac_constraints(v):
        x, y, r = v[0:n], v[n:2*n], v[2*n:3*n]
        J = np.zeros((n * (n - 1) // 2 + 4 * n, 3 * n))
        row = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy, dr_sum = x[i] - x[j], y[i] - y[j], r[i] + r[j]
                J[row, i], J[row, j] = 2 * dx, -2 * dx
                J[row, n + i], J[row, n + j] = 2 * dy, -2 * dy
                J[row, 2 * n + i], J[row, 2 * n + j] = -2 * dr_sum, -2 * dr_sum
                row += 1
        for i in range(n):
            # x[i]-r[i], 1-x[i]-r[i], y[i]-r[i], 1-y[i]-r[i]
            for k, val in enumerate([1, -1, 1, -1]):
                J[row, (k//2)*n + i] = val
                J[row, 2*n + i] = -1
                row += 1
        return J

    best_total_sum, best_centers, best_radii = candidate_results[0]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    # Refine top candidates to manage execution time within ~1.5s
    for idx in range(min(6, len(candidate_results))):
        _, s_centers, s_radii = candidate_results[idx]
        x0 = np.concatenate([s_centers[:, 0], s_centers[:, 1], s_radii])
        
        opt_res = minimize(
            objective, x0, jac=jac_objective, method='SLSQP', bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints, 'jac': jac_constraints},
            options={'maxiter': 400, 'ftol': 1e-9}
        )
        
        if opt_res.success and -opt_res.fun > best_total_sum:
            best_total_sum = -opt_res.fun
            best_centers = np.column_stack((opt_res.x[:n], opt_res.x[n:2*n]))
            best_radii = opt_res.x[2*n:]

    # Final validity correction: Ensure boundary/overlap constraints are strictly met
    final_radii = best_radii - 1e-10
    final_centers = np.clip(best_centers, 0, 1)
    final_sum = float(np.sum(final_radii))
    
    return final_centers, final_radii, final_sum

def construct_packing():
    return run_packing()