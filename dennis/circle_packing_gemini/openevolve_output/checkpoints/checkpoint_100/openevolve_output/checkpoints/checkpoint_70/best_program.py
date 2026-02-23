import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp(P, n):
    """Calculates optimal radii for fixed centers using Linear Programming."""
    c = -np.ones(n)
    dist_matrix = np.sqrt(np.sum((P[:, np.newaxis, :] - P[np.newaxis, :, :])**2, axis=2) + 1e-15)
    num_overlap = n * (n - 1) // 2
    A_ub = np.zeros((num_overlap, n))
    b_ub = np.zeros(num_overlap)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist_matrix[i, j]
            idx += 1
    bounds = [(0, max(0, min(P[i, 0], 1.0 - P[i, 0], P[i, 1], 1.0 - P[i, 1]))) for i in range(n)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success: return res.x, -res.fun
    except: pass
    return None, 0

def get_constraints(v, n):
    """Vectorized constraints: No overlaps (d^2 - (r1+r2)^2 >= 0) and square boundary."""
    x, y, r = v[:n], v[n:2*n], v[2*n:]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    r_sum2 = (r[:, np.newaxis] + r[np.newaxis, :])**2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.concatenate([(d2 - r_sum2)[mask], x - r, 1.0 - x - r, y - r, 1.0 - y - r])

def get_jacobian(v, n):
    """Highly efficient vectorized Jacobian for the constraint set."""
    x, y, r = v[:n], v[n:2*n], v[2*n:]
    num_overlap = n * (n - 1) // 2
    J = np.zeros((num_overlap + 4 * n, 3 * n))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    ii, jj = np.where(mask)
    dx, dy, rs = x[ii] - x[jj], y[ii] - y[jj], r[ii] + r[jj]
    idx = np.arange(num_overlap)
    J[idx, ii], J[idx, jj] = 2 * dx, -2 * dx
    J[idx, n + ii], J[idx, n + jj] = 2 * dy, -2 * dy
    J[idx, 2 * n + ii], J[idx, 2 * n + jj] = -2 * rs, -2 * rs
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
    np.random.seed(42)
    seeds = []
    
    # 1. Structural Seeds (Row-based, Grid-based, Spiral, Symmetry)
    configs = [[6, 7, 7, 6], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 6, 5, 6, 4], [4, 5, 4, 5, 4, 4], [5, 5, 5, 5, 6], [3, 4, 6, 6, 4, 3]]
    for cfg in configs:
        for m in [0.07, 0.08, 0.09, 0.10, 0.11]:
            pts, rows = [], len(cfg)
            for r_idx, count in enumerate(cfg):
                y = m + (1.0 - 2.0 * m) * r_idx / (rows - 1)
                x_vals = np.linspace(m, 1.0 - m, count)
                if r_idx % 2 == 1: x_vals = np.clip(x_vals + 0.02 * (np.random.rand()-0.5), m/2, 1.0 - m/2)
                for x in x_vals: pts.append([x, y])
            seeds.append(np.array(pts[:n]))

    # Fibonacci spirals and axial symmetry
    phi = (1 + 5**0.5) / 2
    for scale in [0.42, 0.45, 0.48, 0.50]:
        indices = np.arange(n) + 0.5
        r_s, theta = np.sqrt(indices / n), 2 * np.pi * indices * phi
        seeds.append(np.column_stack([0.5 + scale * r_s * np.cos(theta), 0.5 + scale * r_s * np.sin(theta)]))

    # Repulsion-based seeds
    for _ in range(12):
        p = np.random.rand(n, 2)
        for _ in range(40):
            diff = p[:, None, :] - p[None, :, :]
            d2 = np.sum(diff**2, axis=-1) + 1e-8
            p = np.clip(p + 0.012 * np.sum(diff / d2[:, :, None], axis=1), 0.01, 0.99)
        seeds.append(p)

    # 2. Screening and SLSQP Polish
    results = []
    for s in seeds:
        s_clip = np.clip(s, 0.001, 0.999)
        radii, score = solve_lp(s_clip, n)
        if radii is not None: results.append((score, s_clip, radii))
    results.sort(key=lambda x: x[0], reverse=True)
    
    best_v, best_sum = None, 0
    obj_fn = lambda v: -np.sum(v[2*n:])
    jac_fn = lambda v: np.concatenate([np.zeros(2*n), -np.ones(n)])

    for i in range(min(22, len(results))):
        _, s_pts, s_rad = results[i]
        x0 = np.concatenate([s_pts[:, 0], s_pts[:, 1], s_rad * 0.98]) # start slightly smaller to allow movement
        res = minimize(obj_fn, x0, jac=jac_fn, method='SLSQP',
                       bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                       constraints={'type': 'ineq', 'fun': get_constraints, 'jac': get_jacobian, 'args': (n,)},
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.success and -res.fun > best_sum:
            best_sum, best_v = -res.fun, res.x

    # 3. Final Multi-scale Jitter and Polish
    if best_v is not None:
        for scale in [0.0004, 0.0001]:
            for _ in range(2):
                x_jitter = best_v + (np.random.rand(3 * n) - 0.5) * scale
                x_jitter[2*n:] = np.clip(x_jitter[2*n:], 1e-7, 0.45)
                res = minimize(obj_fn, x_jitter, jac=jac_fn, method='SLSQP',
                               bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                               constraints={'type': 'ineq', 'fun': get_constraints, 'jac': get_jacobian, 'args': (n,)},
                               options={'maxiter': 400, 'ftol': 1e-14})
                if res.success and -res.fun > best_sum:
                    best_sum, best_v = -res.fun, res.x

    final_centers = np.column_stack((best_v[:n], best_v[n:2*n]))
    final_radii = best_v[2*n:] - 1e-14 # Minimal safety buffer
    return final_centers, final_radii, float(np.sum(final_radii))

def construct_packing():
    """Main entry point for the circle packing optimizer."""
    return run_packing()