import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp(P, n):
    """
    Calculates the optimal radii for a fixed set of centers using Linear Programming.
    This provides a fast way to screen thousands of center arrangements.
    """
    # Objective: Maximize the sum of radii (minimize -sum(r))
    c = -np.ones(n)
    
    # Calculate pairwise distances for overlap constraints: r_i + r_j <= distance(P_i, P_j)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2) + 1e-18)
    ii, jj = np.triu_indices(n, k=1)
    num_overlap = len(ii)
    
    A_ub = np.zeros((num_overlap, n))
    A_ub[np.arange(num_overlap), ii] = 1.0
    A_ub[np.arange(num_overlap), jj] = 1.0
    b_ub = dist_matrix[ii, jj]
    
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    limits = np.minimum(np.minimum(P[:, 0], 1.0 - P[:, 0]), 
                        np.minimum(P[:, 1], 1.0 - P[:, 1]))
    bounds = [(0, max(0.0, lim)) for lim in limits]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except:
        pass
    return None, 0

def get_constraints(v, n):
    """Vectorized constraints for SLSQP: (d^2 - (r1+r2)^2 >= 0) and square boundary."""
    x, y, r = v[:n], v[n:2*n], v[2*n:]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    r_sum2 = (r[:, np.newaxis] + r[np.newaxis, :])**2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.concatenate([
        (d2 - r_sum2)[mask], 
        x - r, 1.0 - x - r, 
        y - r, 1.0 - y - r
    ])

def get_jacobian(v, n):
    """Efficient vectorized Jacobian for overlap and boundary constraints."""
    x, y, r = v[:n], v[n:2*n], v[2*n:]
    ii, jj = np.triu_indices(n, k=1)
    num_overlap = len(ii)
    J = np.zeros((num_overlap + 4 * n, 3 * n))
    
    dx, dy, rs = x[ii] - x[jj], y[ii] - y[jj], r[ii] + r[jj]
    idx = np.arange(num_overlap)
    
    # Overlap derivatives
    J[idx, ii], J[idx, jj] = 2 * dx, -2 * dx
    J[idx, n + ii], J[idx, n + jj] = 2 * dy, -2 * dy
    J[idx, 2 * n + ii], J[idx, 2 * n + jj] = -2 * rs, -2 * rs
    
    # Boundary derivatives
    b_idx = num_overlap
    for i in range(4):
        slc = slice(b_idx, b_idx + n)
        if i < 2: J[slc, :n] = np.eye(n) if i == 0 else -np.eye(n)
        else: J[slc, n:2*n] = np.eye(n) if i == 2 else -np.eye(n)
        J[slc, 2*n:] = -np.eye(n)
        b_idx += n
    return J

def run_packing():
    n = 26
    np.random.seed(1337)
    seeds = []
    
    # 1. Broad Seed Generation Strategy
    # Staggered patterns tuned for n=26
    configs = [
        [5, 6, 5, 6, 4], [6, 7, 7, 6], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [4, 5, 4, 5, 4, 4], [5, 5, 5, 5, 6],
        [4, 4, 5, 5, 4, 4], [3, 4, 6, 6, 4, 3], [5, 5, 5, 5, 5, 1]
    ]
    for cfg in configs:
        for m in [0.07, 0.085, 0.10, 0.115]:
            for stagger in [0.0, 0.015, 0.03]:
                pts, rows = [], len(cfg)
                for r_idx, count in enumerate(cfg):
                    y = m + (1.0 - 2.0 * m) * r_idx / (rows - 1)
                    x_vals = np.linspace(m, 1.0 - m, count)
                    if r_idx % 2 == 1:
                        x_vals = np.clip(x_vals + stagger, 0.01, 0.99)
                    for x in x_vals: pts.append([x, y])
                seeds.append(np.array(pts[:n]))

    # Fibonacci / Sunflower spiral variations
    phi = (1 + 5**0.5) / 2
    for scale in [0.42, 0.45, 0.48, 0.51, 0.54]:
        indices = np.arange(n) + 0.5
        r_s, theta = np.sqrt(indices / n), 2 * np.pi * indices * phi
        seeds.append(np.column_stack([0.5 + scale * r_s * np.cos(theta), 0.5 + scale * r_s * np.sin(theta)]))

    # Grid 5x5 + extra point in various regions
    for grid_m in [0.09, 0.1, 0.11]:
        gx, gy = np.meshgrid(np.linspace(grid_m, 1-grid_m, 5), np.linspace(grid_m, 1-grid_m, 5))
        grid25 = np.column_stack([gx.ravel(), gy.ravel()])
        for extra in [[0.5, 0.5], [0.02, 0.02], [0.98, 0.98], [0.02, 0.5], [0.5, 0.98]]:
            seeds.append(np.vstack([grid25, extra]))

    # Power-law repulsion seeds
    for power in [1.5, 2.0, 3.0]:
        for _ in range(10):
            p = np.random.rand(n, 2)
            for _ in range(40):
                diff = p[:, None, :] - p[None, :, :]
                d2 = np.sum(diff**2, axis=-1) + 1e-9
                f = np.sum(diff / (d2**((power+1)/2))[:, :, None], axis=1)
                p = np.clip(p + 0.012 * f, 0.005, 0.995)
            seeds.append(p)

    # 2. Screening Top Candidates via LP
    results = []
    for s in seeds:
        s_cl = np.clip(s, 0.001, 0.999)
        radii, score = solve_lp(s_cl, n)
        if radii is not None:
            results.append((score, s_cl, radii))
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    # 3. SLSQP Non-linear Refinement
    best_v, best_sum = None, 0
    obj_fn = lambda v: -np.sum(v[2*n:])
    jac_obj = lambda v: np.concatenate([np.zeros(2*n), -np.ones(n)])

    # Optimize top candidates from screening
    for i in range(min(25, len(results))):
        _, s_pts, s_rad = results[i]
        x0 = np.concatenate([s_pts[:, 0], s_pts[:, 1], s_rad * 0.997])
        res = minimize(obj_fn, x0, jac=jac_obj, method='SLSQP',
                       bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                       constraints={'type': 'ineq', 'fun': get_constraints, 'jac': get_jacobian, 'args': (n,)},
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.success and -res.fun > best_sum:
            best_sum, best_v = -res.fun, res.x

    # 4. Multi-scale Jitter Polish (Local search to break through plateaus)
    if best_v is not None:
        for scale in [0.0004, 0.0001, 0.00005]:
            for _ in range(3):
                # Apply small perturbations to centers and radii and re-polish
                x_jitter = best_v.copy()
                x_jitter[:2*n] += np.random.normal(0, scale, 2*n)
                x_jitter[2*n:] *= 0.998 # Shrink slightly to maintain feasibility
                res = minimize(obj_fn, x_jitter, jac=jac_obj, method='SLSQP',
                               bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                               constraints={'type': 'ineq', 'fun': get_constraints, 'jac': get_jacobian, 'args': (n,)},
                               options={'maxiter': 600, 'ftol': 1e-15})
                if res.success and -res.fun > best_sum:
                    best_sum, best_v = -res.fun, res.x

    # Final construction with a tiny safety buffer for validity
    final_centers = np.column_stack((best_v[:n], best_v[n:2*n]))
    final_radii = np.maximum(best_v[2*n:] - 1e-14, 1e-15)
    return final_centers, final_radii, float(np.sum(final_radii))

def construct_packing():
    """Main entry point for the circle packing optimizer."""
    return run_packing()