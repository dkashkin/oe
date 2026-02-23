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

    def solve_lp(c_fixed):
        """Solves for exact optimal radii given fixed center positions."""
        num = len(c_fixed)
        # Linear Programming: Maximize sum(r) <=> Minimize sum(-r)
        # Subject to: r_i + r_j <= distance(c_i, c_j)
        # and r_i <= distance to the nearest wall (boundary constraint)
        a_ub, b_ub, bounds = [], [], []
        for i in range(num):
            for j in range(i + 1, num):
                diff = c_fixed[i] - c_fixed[j]
                dist = np.sqrt(np.sum(diff**2) + 1e-15)
                row = np.zeros(num)
                row[i], row[j] = 1.0, 1.0
                a_ub.append(row)
                b_ub.append(dist)
            # Boundary constraint for each circle
            dist_to_wall = min(c_fixed[i, 0], 1.0 - c_fixed[i, 0], 
                               c_fixed[i, 1], 1.0 - c_fixed[i, 1])
            bounds.append((0, max(0.0, dist_to_wall)))

        res = linprog(-np.ones(num), A_ub=np.array(a_ub), b_ub=np.array(b_ub), 
                      bounds=bounds, method='highs')
        return res.x if res.success else None

    def adam_optimize(c_init, seed_val, iters=4000):
        """Refines circle centers using Adam optimization with dynamic penalty scaling."""
        np.random.seed(seed_val)
        c = c_init.copy() + np.random.normal(0, 0.01, c_init.shape)
        c = np.clip(c, 0.0, 1.0)
        r = np.full(n, 0.08)
        
        m_c, v_c = np.zeros_like(c), np.zeros_like(c)
        m_r, v_r = np.zeros_like(r), np.zeros_like(r)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        base_lr = 0.003
        
        for t in range(1, iters + 1):
            # Gradual penalty increase to allow centers to reorganize early
            if t < iters * 0.2:
                penalty_coeff = 0.8
            else:
                penalty_coeff = 0.8 + 800.0 * ((t - iters * 0.2) / (iters * 0.8))**2
                
            dc, dr = np.zeros_like(c), -np.ones(n)
            
            # 1. Boundary penalties: minimize max(0, r - distance_to_wall)^2
            for i in range(2):
                v_lo, v_hi = r - c[:, i], c[:, i] + r - 1.0
                m_lo, m_hi = v_lo > 0, v_hi > 0
                g_lo = 2 * penalty_coeff * v_lo[m_lo]
                dr[m_lo] += g_lo
                dc[m_lo, i] -= g_lo
                g_hi = 2 * penalty_coeff * v_hi[m_hi]
                dr[m_hi] += g_hi
                dc[m_hi, i] += g_hi
            
            # 2. Pairwise overlap penalties: minimize max(0, r_i + r_j - dist_ij)^2
            for i in range(n):
                diff = c[i] - c[i + 1:]
                dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
                overlap = (r[i] + r[i + 1:]) - dist
                mask = overlap > 0
                if np.any(mask):
                    idx = np.where(mask)[0] + i + 1
                    g_pair = 2 * penalty_coeff * overlap[mask]
                    dr[i] += np.sum(g_pair)
                    dr[idx] += g_pair
                    g_c = (g_pair / dist[mask])[:, np.newaxis] * diff[mask]
                    dc[i] -= np.sum(g_c, axis=0)
                    dc[idx] += g_c
            
            # Adaptive learning rate decay
            lr = base_lr * (1.0 - 0.5 * t / iters)
            
            # Adam optimizer updates for centers
            m_c = beta1 * m_c + (1 - beta1) * dc
            v_c = beta2 * v_c + (1 - beta2) * (dc**2)
            c -= lr * (m_c / (1 - beta1**t)) / (np.sqrt(v_c / (1 - beta2**t)) + eps)
            
            # Adam optimizer updates for radii
            m_r = beta1 * m_r + (1 - beta1) * dr
            v_r = beta2 * v_r + (1 - beta2) * (dr**2)
            r -= lr * (m_r / (1 - beta1**t)) / (np.sqrt(v_r / (1 - beta2**t)) + eps)
            
            # Projected constraints
            c, r = np.clip(c, 0, 1), np.clip(r, 0, 0.5)
            
        return c

    # Create various initial configurations
    grids = []
    
    # 1. 6x5 grid (30 slots), removing 4 corners
    pts_6x5 = []
    for i in range(6):
        for j in range(5):
            if not ((i == 0 or i == 5) and (j == 0 or j == 4)):
                pts_6x5.append([(i + 0.5) / 6.0, (j + 0.5) / 5.0])
    grids.append(np.array(pts_6x5))

    # 2. 5x6 grid (30 slots), removing 4 corners
    pts_5x6 = []
    for i in range(5):
        for j in range(6):
            if not ((i == 0 or i == 4) and (j == 0 or j == 5)):
                pts_5x6.append([(i + 0.5) / 5.0, (j + 0.5) / 6.0])
    grids.append(np.array(pts_5x6))

    # 3. Staggered hex-like grid (5-6-5-6-4 configuration)
    pts_hex = []
    for r_idx, count in enumerate([5, 6, 5, 6, 4]):
        for c_idx in range(count):
            x = (c_idx + 0.5 + (0.25 if r_idx % 2 else 0)) / 6.5
            y = (r_idx + 0.5) / 5.5
            pts_hex.append([x, y])
    grids.append(np.array(pts_hex))

    # 4. 7x4 grid (28 slots), removing 2 corners
    pts_7x4 = []
    for i in range(7):
        for j in range(4):
            if not ((i == 0 and j == 0) or (i == 6 and j == 3)):
                pts_7x4.append([(i + 0.5) / 7.0, (j + 0.5) / 4.0])
    grids.append(np.array(pts_7x4))

    # 5. Concentrated spiral-like start
    pts_spiral = []
    for i in range(n):
        phi = i * 0.5 * np.pi * (1 + 5**0.5)
        dist = 0.45 * np.sqrt(i / n)
        pts_spiral.append([0.5 + dist * np.cos(phi), 0.5 + dist * np.sin(phi)])
    grids.append(np.array(pts_spiral))

    best_c, best_r, best_sum = None, None, 0
    
    # Evaluate each starting configuration
    for idx, g in enumerate(grids):
        c_optimized = adam_optimize(g[:n], seed_val=100 + idx, iters=4500)
        r_lp = solve_lp(c_optimized)
        if r_lp is not None:
            current_sum = np.sum(r_lp)
            if current_sum > best_sum:
                best_sum, best_c, best_r = current_sum, c_optimized, r_lp

    # Final refinement: Ensure strict feasibility against precision limits
    # Scale down slightly to provide a buffer for floating point errors
    best_r *= (1.0 - 1e-12)
    
    # 1. Enforce boundary constraints
    for i in range(n):
        best_r[i] = min(best_r[i], best_c[i, 0], 1.0 - best_c[i, 0], 
                        best_c[i, 1], 1.0 - best_c[i, 1])
    
    # 2. Enforce non-overlap constraints (greedy correction)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_c[i] - best_c[j])
            if best_r[i] + best_r[j] > dist:
                over = (best_r[i] + best_r[j]) - dist
                best_r[i] -= over * 0.5 + 2e-15
                best_r[j] -= over * 0.5 + 2e-15
    
    best_r = np.maximum(best_r, 0.0)
    final_sum = np.sum(best_r)
    
    return best_c, best_r, final_sum


# Define results
centers, radii, sum_radii = run_packing()