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
        # Linear Programming: Maximize sum(r)
        a_ub, b_ub, bounds = [], [], []
        for i in range(num):
            for j in range(i + 1, num):
                dist = np.linalg.norm(c_fixed[i] - c_fixed[j])
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

    def adam_optimize(c_init, seed_val, iters=3800):
        """Refines circle centers using Adam optimization with dynamic penalty scaling."""
        np.random.seed(seed_val)
        c = c_init.copy() + np.random.normal(0, 0.012, c_init.shape)
        c = np.clip(c, 0.01, 0.99)
        r = np.full(n, 0.095)
        
        m_c, v_c = np.zeros_like(c), np.zeros_like(c)
        m_r, v_r = np.zeros_like(r), np.zeros_like(r)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        base_lr = 0.0035
        
        for t in range(1, iters + 1):
            # Gradual penalty increase to allow early reconfiguration
            penalty_coeff = 1.0 + 900.0 * (t / iters)**2
            dc, dr = np.zeros_like(c), -np.ones(n)
            
            # 1. Boundary penalties
            for i in range(2):
                v_lo, v_hi = r - c[:, i], c[:, i] + r - 1.0
                m_lo, m_hi = v_lo > 0, v_hi > 0
                g_lo = 2 * penalty_coeff * v_lo[m_lo]
                dr[m_lo] += g_lo
                dc[m_lo, i] -= g_lo
                g_hi = 2 * penalty_coeff * v_hi[m_hi]
                dr[m_hi] += g_hi
                dc[m_hi, i] += g_hi
            
            # 2. Pairwise overlap penalties
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
            
            # Learning rate with cosine decay
            lr = base_lr * 0.5 * (1 + np.cos(np.pi * t / iters))
            
            # Adam optimizer updates
            m_c = beta1 * m_c + (1 - beta1) * dc
            v_c = beta2 * v_c + (1 - beta2) * (dc**2)
            c -= lr * (m_c / (1 - beta1**t)) / (np.sqrt(v_c / (1 - beta2**t)) + eps)
            
            m_r = beta1 * m_r + (1 - beta1) * dr
            v_r = beta2 * v_r + (1 - beta2) * (dr**2)
            r -= lr * (m_r / (1 - beta1**t)) / (np.sqrt(v_r / (1 - beta2**t)) + eps)
            
            c, r = np.clip(c, 0, 1), np.clip(r, 0, 0.5)
            
        return c

    # Diverse initial layouts for multi-start
    grids = []
    
    # Layout 1: 6x5 Grid (30 slots), remove corners
    pts = []
    for i in range(6):
        for j in range(5):
            if not ((i == 0 or i == 5) and (j == 0 or j == 4)):
                pts.append([i / 5.0, j / 4.0])
    grids.append(np.array(pts)[:n])

    # Layout 2: 5x6 Grid (30 slots), remove corners
    pts = []
    for i in range(5):
        for j in range(6):
            if not ((i == 0 or i == 4) and (j == 0 or j == 5)):
                pts.append([i / 4.0, j / 5.0])
    grids.append(np.array(pts)[:n])

    # Layout 3: Staggered rows (5-6-5-6-4)
    pts = []
    for r_idx, count in enumerate([5, 6, 5, 6, 4]):
        for c_idx in range(count):
            pts.append([(c_idx + 0.5 * (r_idx % 2)) / 6.0, r_idx / 4.5])
    grids.append(np.array(pts)[:n])

    # Layout 4: Staggered columns
    pts = []
    for c_idx, count in enumerate([5, 6, 5, 6, 4]):
        for r_idx in range(count):
            pts.append([c_idx / 4.5, (r_idx + 0.5 * (c_idx % 2)) / 6.0])
    grids.append(np.array(pts)[:n])

    # Layout 5: Hex-like 6-5-6-5-4
    pts = []
    for r_idx, count in enumerate([6, 5, 6, 5, 4]):
        for c_idx in range(count):
            pts.append([(c_idx + 0.5 * ((r_idx + 1) % 2)) / 6.0, r_idx / 4.5])
    grids.append(np.array(pts)[:n])

    # Layout 6: Random uniform distribution
    np.random.seed(42)
    grids.append(np.random.rand(n, 2))

    best_c, best_r, best_sum = None, None, 0
    
    # Run multi-start optimization
    for idx, g in enumerate(grids):
        c_opt = adam_optimize(g, seed_val=100 + idx)
        r_lp = solve_lp(c_opt)
        if r_lp is not None:
            current_sum = np.sum(r_lp)
            if current_sum > best_sum:
                best_sum, best_c, best_r = current_sum, c_opt, r_lp

    # Final post-processing for precision and feasibility
    best_r *= (1.0 - 1e-12)
    for i in range(n):
        best_r[i] = min(best_r[i], best_c[i, 0], 1.0 - best_c[i, 0], 
                        best_c[i, 1], 1.0 - best_c[i, 1])
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_c[i] - best_c[j])
            if best_r[i] + best_r[j] > dist:
                over = (best_r[i] + best_r[j]) - dist
                best_r[i] -= over * 0.5 + 1e-15
                best_r[j] -= over * 0.5 + 1e-15
    
    best_r = np.maximum(best_r, 0.0)
    final_sum = np.sum(best_r)
    
    return best_c, best_r, final_sum


# Execute the solver
centers, radii, sum_radii = run_packing()