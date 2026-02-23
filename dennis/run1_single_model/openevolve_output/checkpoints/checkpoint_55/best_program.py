import numpy as np
from scipy.optimize import linprog


def run_packing():
    """
    Arrangement of 26 circles in a unit square to maximize the sum of radii.
    Employs a physics-inspired optimization (Adam) to locate centers and 
    Linear Programming (HiGHS) to determine the mathematically optimal radii.
    The start is a refined 6x5 grid (minus corners) to ensure good coverage.
    """
    n = 26
    
    # 1. Initialization: 6x5 grid and remove 4 corners to get 26 points
    points = []
    for i in range(6):
        for j in range(5):
            # Remove corner indices to leave exactly 26 points
            if not ((i == 0 or i == 5) and (j == 0 or j == 4)):
                points.append([0.1 + i * 0.16, 0.1 + j * 0.2])
    c = np.array(points)[:n]
    
    # Break symmetry and jitter to help the optimizer find global trends
    np.random.seed(42)
    c += np.random.normal(0, 0.01, c.shape)
    c = np.clip(c, 0.05, 0.95)
    
    # Initial radii guess
    r = np.full(n, 0.08)
    
    # 2. Optimization Phase (Adam)
    # This phase moves centers and radii to minimize overlaps and maximize growth.
    m_c, v_c = np.zeros_like(c), np.zeros_like(c)
    m_r, v_r = np.zeros_like(r), np.zeros_like(r)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.002
    
    iters = 3000
    for t in range(1, iters + 1):
        # Penalty grows to transition from soft to hard constraints
        penalty_coeff = 5.0 + t * 0.2
        
        # Gradients: dr objective is -1 (maximizing sum is minimizing negative sum)
        dc = np.zeros_like(c)
        dr = -np.ones(n)
        
        # Boundary constraints gradients (minimize square of violation)
        for i in range(2):
            # Low boundary: r_i - c_i <= 0
            v_low = r - c[:, i]
            mask_low = v_low > 0
            dr[mask_low] += 2 * penalty_coeff * v_low[mask_low]
            dc[mask_low, i] -= 2 * penalty_coeff * v_low[mask_low]
            
            # High boundary: c_i + r_i - 1 <= 0
            v_high = c[:, i] + r - 1.0
            mask_high = v_high > 0
            dr[mask_high] += 2 * penalty_coeff * v_high[mask_high]
            dc[mask_high, i] += 2 * penalty_coeff * v_high[mask_high]
            
        # Pairwise overlap constraints gradients: r_i + r_j - dist_ij <= 0
        for i in range(n):
            diff = c[i] - c[i + 1:]
            # Distance with stability constant to avoid division by zero
            dists = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
            sum_r = r[i] + r[i + 1:]
            overlap = sum_r - dists
            mask = overlap > 0
            
            if np.any(mask):
                idx = np.where(mask)[0] + i + 1
                ov = overlap[mask]
                di = dists[mask]
                df = diff[mask]
                
                # Radii penalty gradients
                grad_r = 2 * penalty_coeff * ov
                dr[i] += np.sum(grad_r)
                dr[idx] += grad_r
                
                # Center penalty gradients (pushing overlapping circles apart)
                grad_c = (grad_r / di)[:, np.newaxis] * df
                dc[i] -= np.sum(grad_c, axis=0)
                dc[idx] += grad_c
        
        # Adam update for centers
        m_c = beta1 * m_c + (1 - beta1) * dc
        v_c = beta2 * v_c + (1 - beta2) * (dc**2)
        m_c_hat = m_c / (1 - beta1**t)
        v_c_hat = v_c / (1 - beta2**t)
        c -= lr * m_c_hat / (np.sqrt(v_c_hat) + eps)
        
        # Adam update for radii
        m_r = beta1 * m_r + (1 - beta1) * dr
        v_r = beta2 * v_r + (1 - beta2) * (dr**2)
        m_r_hat = m_r / (1 - beta1**t)
        v_r_hat = v_r / (1 - beta2**t)
        r -= lr * m_r_hat / (np.sqrt(v_r_hat) + eps)
        
        # Keeping centers and radii within logical bounds
        c = np.clip(c, 0.0, 1.0)
        r = np.clip(r, 0.0, 0.5)

    # 3. Radius Refinement (Linear Programming)
    # Fix center locations and solve for mathematically optimal radii values.
    A_ub, b_ub, bounds = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(c[i] - c[j])
            row = np.zeros(n)
            row[i], row[j] = 1.0, 1.0
            A_ub.append(row)
            b_ub.append(dist)
        # Radius for circle i is limited by its distance to the 4 edges
        dist_to_edge = min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
        bounds.append((0, max(0.0, dist_to_edge)))
    
    # Solve LP: Minimize -sum(r) to maximize sum(r)
    res_lp = linprog(-np.ones(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                     bounds=bounds, method='highs')
    
    if res_lp.success:
        r_final = res_lp.x
    else:
        r_final = r

    # 4. Final Feasibility Check and Post-processing
    # Ensure zero overlap and zero boundary violations under precision.
    r_final *= (1.0 - 1e-10)
    for i in range(n):
        r_final[i] = min(r_final[i], c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(c[i] - c[j])
            if r_final[i] + r_final[j] > dist:
                over = (r_final[i] + r_final[j]) - dist
                r_final[i] -= over * 0.5
                r_final[j] -= over * 0.5
    
    r_final = np.maximum(r_final, 0.0)
    sum_r = np.sum(r_final)
    
    return c, r_final, sum_r


# Execution to define required output variables
centers, radii, sum_radii = run_packing()