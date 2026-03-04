import numpy as np
from scipy.optimize import minimize, linprog


def run_packing():
    """
    Directly produces an arrangement of 26 circles in a unit square,
    maximizing the sum of their radii using a multi-start optimization
    followed by Linear Programming for exact radius refinement.
    """
    n = 26
    best_sum = 0
    best_c = np.zeros((n, 2))
    best_r = np.zeros(n)

    # Multi-start optimization with diverse initializations
    # 1. Perturbed 5x5 grid + 1 extra circle
    # 2. Staggered hexagonal-style packing
    # 3. Random distribution
    seeds = [1, 42, 123, 777]
    for seed in seeds:
        np.random.seed(seed)

        if seed == 1:
            # 5x5 grid (25 circles) + 26th circle in the corner
            gv = np.linspace(0.1, 0.9, 5)
            xv, yv = np.meshgrid(gv, gv)
            c_init = np.vstack([np.column_stack([xv.ravel(), yv.ravel()]), [0.95, 0.05]])
        elif seed == 42:
            # Staggered hexagonal-ish: 5-6-5-6-4 row layout
            c_init = []
            for r_idx, count in enumerate([5, 6, 5, 6, 4]):
                for c_idx in range(count):
                    # Stagger every other row
                    x = (c_idx + 0.5 + (0.3 if r_idx % 2 == 1 else 0)) / 6.5
                    y = (r_idx + 0.5) / 5.5
                    c_init.append([x, y])
            c_init = np.array(c_init)
        else:
            # Random starting centers
            c_init = np.random.uniform(0.1, 0.9, (n, 2))

        # Add jitter and enforce bounds
        c_init += np.random.uniform(-0.02, 0.02, (n, 2))
        c_init = np.clip(c_init, 0.0, 1.0)
        r_init = np.full(n, 0.09)

        # Optimization phase: Search for center configurations that allow large radii
        def penalty_objective(params, weight):
            curr_c = params[:2*n].reshape((n, 2))
            curr_r = params[2*n:]

            # Maximize sum(r_i)
            obj_val = -np.sum(curr_r)

            # Boundary constraints penalty: r_i <= distance to wall
            dist_to_wall = np.minimum(np.minimum(curr_c[:, 0], 1.0 - curr_c[:, 0]),
                                     np.minimum(curr_c[:, 1], 1.0 - curr_c[:, 1]))
            b_pen = np.sum(np.maximum(0, curr_r - dist_to_wall)**2)

            # Overlap constraints penalty: r_i + r_j <= distance between centers
            diffs = curr_c[:, np.newaxis, :] - curr_c[np.newaxis, :, :]
            # Stability constant 1e-12 avoids sqrt(0) issues
            dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-12)
            r_sum = curr_r[:, np.newaxis] + curr_r[np.newaxis, :]
            o_violations = np.maximum(0, r_sum - dists)
            np.fill_diagonal(o_violations, 0)
            o_pen = 0.5 * np.sum(o_violations**2)

            return obj_val + weight * (b_pen + o_pen)

        # Optimization refined in two stages (soft constraints then hard)
        init_params = np.concatenate([c_init.ravel(), r_init])
        bnds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        
        # Stage 1: Preliminary search
        res = minimize(penalty_objective, init_params, args=(100.0,), method='L-BFGS-B', 
                       bounds=bnds, options={'maxiter': 500, 'ftol': 1e-6})
        # Stage 2: Refined search
        res = minimize(penalty_objective, res.x, args=(10000.0,), method='L-BFGS-B', 
                       bounds=bnds, options={'maxiter': 700, 'ftol': 1e-8})
        
        c_refined = res.x[:2*n].reshape((n, 2))
        c_refined = np.clip(c_refined, 0.0, 1.0)

        # Radii Refinement: Solve exact Linear Program for the sum of radii at fixed centers
        A_ub = []
        b_ub = []
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i], row[j] = 1.0, 1.0
                A_ub.append(row)
                b_ub.append(np.linalg.norm(c_refined[i] - c_refined[j]))
        
        r_bounds = []
        for i in range(n):
            dist_w = min(c_refined[i, 0], 1.0 - c_refined[i, 0],
                         c_refined[i, 1], 1.0 - c_refined[i, 1])
            r_bounds.append((0.0, max(0.0, dist_w)))
        
        lp_res = linprog(-np.ones(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                         bounds=r_bounds, method='highs')
        
        if lp_res.success:
            curr_sum = -lp_res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_c, best_r = c_refined, lp_res.x

    # Ensure strict feasibility against precision limits
    best_r *= (1.0 - 1e-12)
    for i in range(n):
        best_r[i] = min(best_r[i], best_c[i, 0], 1.0 - best_c[i, 0],
                        best_c[i, 1], 1.0 - best_c[i, 1])
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_c[i] - best_c[j])
            if best_r[i] + best_r[j] > d:
                overlap = (best_r[i] + best_r[j] - d)
                best_r[i] -= overlap / 2.0 + 1e-15
                best_r[j] -= overlap / 2.0 + 1e-15
    
    best_r = np.maximum(best_r, 0.0)
    final_sum = np.sum(best_r)
    
    return best_c, best_r, final_sum


# Execute the solver to generate the arrangement
centers, radii, sum_radii = run_packing()