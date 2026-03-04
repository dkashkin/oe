import numpy as np
from scipy.optimize import linprog


def run_packing():
    """
    Arrangement of 26 circles in a unit square to maximize the sum of their radii.
    Employs a multi-start strategy with physics-inspired Adam optimization 
    to find center configurations and Linear Programming (HiGHS) to solve 
    for the exact optimal radii.
    """
    n = 26

    def optimize_arrangement(c_init, seed_val, iters=3000):
        """
        Moves centers and expands radii using the Adam optimizer to maximize
        the sum of radii while satisfying boundary and non-overlap constraints.
        """
        np.random.seed(seed_val)
        c = c_init + np.random.normal(0, 0.015, c_init.shape)
        c = np.clip(c, 0.05, 0.95)
        r = np.full(n, 0.06)

        m_c, v_c = np.zeros_like(c), np.zeros_like(c)
        m_r, v_r = np.zeros_like(r), np.zeros_like(r)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = 0.0025

        for t in range(1, iters + 1):
            penalty_coeff = 2.0 + t * 0.22
            dc = np.zeros_like(c)
            dr = -np.ones(n)  # Minimize -sum(r) to maximize sum(r)

            # 1. Boundary violations
            for i in range(2):
                # Low side: r - c_i <= 0
                v_low = r - c[:, i]
                mask_low = v_low > 0
                grad_low = 2 * penalty_coeff * v_low[mask_low]
                dr[mask_low] += grad_low
                dc[mask_low, i] -= grad_low

                # High side: c_i + r - 1 <= 0
                v_high = c[:, i] + r - 1.0
                mask_high = v_high > 0
                grad_high = 2 * penalty_coeff * v_high[mask_high]
                dr[mask_high] += grad_high
                dc[mask_high, i] += grad_high

            # 2. Pairwise overlaps: r_i + r_j - dist_ij <= 0
            for i in range(n):
                diff = c[i] - c[i + 1:]
                dists = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
                sum_r = r[i] + r[i + 1:]
                overlap = sum_r - dists
                mask = overlap > 0

                if np.any(mask):
                    idx = np.where(mask)[0] + i + 1
                    ov = overlap[mask]
                    di = dists[mask]
                    df = diff[mask]

                    grad_pair = 2 * penalty_coeff * ov
                    dr[i] += np.sum(grad_pair)
                    dr[idx] += grad_pair

                    # Center movement grads (pushing circles apart)
                    grad_c = (grad_pair / di)[:, np.newaxis] * df
                    dc[i] -= np.sum(grad_c, axis=0)
                    dc[idx] += grad_c

            # Adam updates for centers
            m_c = beta1 * m_c + (1 - beta1) * dc
            v_c = beta2 * v_c + (1 - beta2) * (dc**2)
            m_c_hat = m_c / (1 - beta1**t)
            v_c_hat = v_c / (1 - beta2**t)
            c -= lr * m_c_hat / (np.sqrt(v_c_hat) + eps)

            # Adam updates for radii
            m_r = beta1 * m_r + (1 - beta1) * dr
            v_r = beta2 * v_r + (1 - beta2) * (dr**2)
            m_r_hat = m_r / (1 - beta1**t)
            v_r_hat = v_r / (1 - beta2**t)
            r -= lr * m_r_hat / (np.sqrt(v_r_hat) + eps)

            c = np.clip(c, 0.0, 1.0)
            r = np.clip(r, 0.0, 0.5)

        return c, r

    # Phase 1: Multiple Initializations
    initial_configs = []
    
    # Init A: 6x5 grid (30 slots), remove 4 corners
    points_a = []
    xs_a = np.linspace(0.1, 0.9, 6)
    ys_a = np.linspace(0.1, 0.9, 5)
    for i in range(6):
        for j in range(5):
            if not ((i == 0 or i == 5) and (j == 0 or j == 4)):
                points_a.append([xs_a[i], ys_a[j]])
    initial_configs.append((np.array(points_a)[:n], 42, 3000))

    # Init B: 7x4 grid (28 slots), remove 2 corners
    points_b = []
    xs_b = np.linspace(0.08, 0.92, 7)
    ys_b = np.linspace(0.12, 0.88, 4)
    for i in range(7):
        for j in range(4):
            if not ((i == 0 or i == 6) and (j == 0)):
                points_b.append([xs_b[i], ys_b[j]])
    initial_configs.append((np.array(points_b)[:n], 123, 2000))

    best_total_sum = 0
    best_c_found = None
    best_r_found = None

    # Phase 2: Run Optimizations and LP Refinement
    for c_init, seed, iterations in initial_configs:
        c_refined, r_adam = optimize_arrangement(c_init, seed, iterations)
        
        # Linear Programming to solve for exact optimal radii for these centers
        A_ub, b_ub, bounds = [], [], []
        dist_to_edge = np.min([c_refined[:, 0], 1.0 - c_refined[:, 0], 
                               c_refined[:, 1], 1.0 - c_refined[:, 1]], axis=0)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(c_refined[i] - c_refined[j])
                row = np.zeros(n)
                row[i], row[j] = 1.0, 1.0
                A_ub.append(row)
                b_ub.append(dist)
            bounds.append((0, max(0.0, dist_to_edge[i])))

        res = linprog(-np.ones(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                      bounds=bounds, method='highs')
        
        if res.success:
            r_lp = res.x
            current_sum = np.sum(r_lp)
        else:
            r_lp = r_adam
            current_sum = np.sum(r_adam)

        if current_sum > best_total_sum:
            best_total_sum = current_sum
            best_c_found = c_refined
            best_r_found = r_lp

    # Phase 3: Post-processing for strict validity
    final_c = best_c_found
    final_r = best_r_found * (1.0 - 1e-11)
    
    # Enforce boundary constraints strictly
    for i in range(n):
        final_r[i] = min(final_r[i], final_c[i, 0], 1.0 - final_c[i, 0], 
                         final_c[i, 1], 1.0 - final_c[i, 1])
    
    # Enforce non-overlap constraints strictly
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(final_c[i] - final_c[j])
            if final_r[i] + final_r[j] > dist:
                over = (final_r[i] + final_r[j]) - dist
                final_r[i] -= over * 0.5 + 1e-15
                final_r[j] -= over * 0.5 + 1e-15

    final_r = np.maximum(final_r, 0.0)
    final_sum = np.sum(final_r)

    return final_c, final_r, final_sum


# Final execution variables
centers, radii, sum_radii = run_packing()