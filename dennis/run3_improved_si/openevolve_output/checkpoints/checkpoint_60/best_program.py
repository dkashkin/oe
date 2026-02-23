import time
import numpy as np

def compute_valid_scale(P, R):
    """
    Computes a strict scaling factor to guarantee that no circles overlap and 
    all remain completely within the [0, 1] boundaries. This acts as a robust 
    correction step resolving any floating point precision inconsistencies.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    
    R_sum = R[:, np.newaxis] + R[np.newaxis, :]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scale_pairs = np.min(dist / (R_sum + 1e-12))
        scale_left = np.min(P[:, 0] / (R + 1e-12))
        scale_right = np.min((1 - P[:, 0]) / (R + 1e-12))
        scale_bottom = np.min(P[:, 1] / (R + 1e-12))
        scale_top = np.min((1 - P[:, 1]) / (R + 1e-12))
        
    return max(0.0, min(1.0, scale_pairs, scale_left, scale_right, scale_bottom, scale_top))

def optimize_radii_lp(P):
    """
    Given a fixed topology of centers, calculates the mathematically strict maximum
    possible radii using a Linear Programming solver.
    """
    N = len(P)
    try:
        from scipy.optimize import linprog
        c = -np.ones(N)
        A_ub = []
        b_ub = []
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = float(np.linalg.norm(P[i] - P[j]))
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
                
        bounds = [(0.0, max(0.0, min(float(P[i, 0]), 1.0 - float(P[i, 0]), float(P[i, 1]), 1.0 - float(P[i, 1])))) for i in range(N)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def optimize_centers_and_radii(P, R, steps, initial_lr, initial_C, inject_noise=True):
    """
    Physics-based constraint optimization via Adam gradient descent.
    Uses simulated annealing with an increasing penalty constraint (C) and decaying learning rate.
    """
    lr = initial_lr
    C = initial_C
    m_P, v_P = np.zeros_like(P), np.zeros_like(P)
    m_R, v_R = np.zeros_like(R), np.zeros_like(R)
    
    for step in range(steps):
        left = np.maximum(0, R - P[:, 0])
        right = np.maximum(0, R - (1 - P[:, 0]))
        bottom = np.maximum(0, R - P[:, 1])
        top = np.maximum(0, R - (1 - P[:, 1]))
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        dist_safe = np.maximum(dist, 1e-12)
        
        R_sum = R[:, np.newaxis] + R[np.newaxis, :]
        overlap = R_sum - dist
        np.fill_diagonal(overlap, -np.inf)
        overlap_masked = np.maximum(0, overlap)
        
        grad_R = -1.0 + 2 * C * (left + right + bottom + top + np.sum(overlap_masked, axis=1))
        
        grad_P = np.zeros_like(P)
        grad_P[:, 0] += 2 * C * (-left + right)
        grad_P[:, 1] += 2 * C * (-bottom + top)
        
        weight = 2 * C * overlap_masked / dist_safe
        grad_P -= np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        
        t = step + 1
        
        m_P = 0.9 * m_P + 0.1 * grad_P
        v_P = 0.999 * v_P + 0.001 * (grad_P ** 2)
        m_hat_P = m_P / (1 - 0.9 ** t)
        v_hat_P = v_P / (1 - 0.999 ** t)
        P -= lr * m_hat_P / (np.sqrt(v_hat_P) + 1e-8)
        
        m_R = 0.9 * m_R + 0.1 * grad_R
        v_R = 0.999 * v_R + 0.001 * (grad_R ** 2)
        m_hat_R = m_R / (1 - 0.9 ** t)
        v_hat_R = v_R / (1 - 0.999 ** t)
        R -= lr * m_hat_R / (np.sqrt(v_hat_R) + 1e-8)
        
        R = np.maximum(1e-5, R)
        P = np.clip(P, 1e-5, 1 - 1e-5)
        
        C *= 1.002
        lr *= 0.9995
        
        if inject_noise and step % 600 == 0 and step < steps * 0.6:
            P += np.random.randn(*P.shape) * 0.0005
            
    return P, R

def run_packing():
    """
    Main execution function called by the evaluator.
    Finds the optimal packing of 26 non-overlapping circles to maximize sum of radii.
    Returns: (centers, radii, sum_radii)
    """
    N = 26
    np.random.seed(42)
    start_time = time.time()
    
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    attempt = 0
    while attempt < 100 and (time.time() - start_time) < 45.0:
        P = np.random.rand(N, 2) * 0.8 + 0.1
        
        # Strategic topological initialization to explore diverse layout seeds
        mode = attempt % 7
        if mode == 0:
            P[0:4] = [[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]]
            P[4:8] = [[0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
            P[8] = [0.5, 0.5]
        elif mode == 1:
            for i in range(10):
                angle = 2 * np.pi * i / 10
                P[i] = [0.5 + 0.35 * np.cos(angle), 0.5 + 0.35 * np.sin(angle)]
            for i in range(8):
                angle = 2 * np.pi * i / 8
                P[10+i] = [0.5 + 0.15 * np.cos(angle), 0.5 + 0.15 * np.sin(angle)]
            P[18] = [0.5, 0.5]
            P[19:] = np.random.rand(7, 2) * 0.8 + 0.1
        elif mode == 2:
            idx = 0
            for x in np.linspace(0.1, 0.9, 5):
                for y in np.linspace(0.1, 0.9, 5):
                    if idx < 25:
                        P[idx] = [x, y]
                        idx += 1
            P[25] = [0.5, 0.5]
        elif mode == 3:
            idx = 0
            for row in range(5):
                cols = 5 if row % 2 == 0 else 6
                for col in range(cols):
                    if idx < N:
                        P[idx] = [0.1 + 0.8 * col / max(1, cols - 1), 0.1 + 0.8 * row / 4.0]
                        idx += 1
        elif mode == 4:
            for i in range(12):
                P[i] = np.random.rand(2) * 0.2
                if i % 4 == 1: P[i, 0] += 0.8
                elif i % 4 == 2: P[i, 1] += 0.8
                elif i % 4 == 3: P[i] += 0.8
            P[12:] = np.random.rand(14, 2) * 0.8 + 0.1
        elif mode == 5:
            P = np.random.rand(N, 2) * 0.8 + 0.1
        elif mode == 6:
            P[0] = [0.5, 0.5]
            for i in range(6): 
                P[i+1] = [0.5 + 0.15*np.cos(2*np.pi*i/6), 0.5 + 0.15*np.sin(2*np.pi*i/6)]
            for i in range(12): 
                P[i+7] = [0.5 + 0.3*np.cos(2*np.pi*i/12), 0.5 + 0.3*np.sin(2*np.pi*i/12)]
            for i in range(7): 
                P[i+19] = [0.5 + 0.45*np.cos(2*np.pi*i/7), 0.5 + 0.45*np.sin(2*np.pi*i/7)]
            P = np.clip(P, 0.05, 0.95)
            
        # Initialize sizes inversely proportional to distance from the centroid
        dist_to_center = np.linalg.norm(P - 0.5, axis=1)
        R = 0.08 - 0.04 * dist_to_center
        
        # Two-stage physics optimization
        P, R = optimize_centers_and_radii(P, R, steps=3000, initial_lr=0.015, initial_C=1.0, inject_noise=True)
        P, R = optimize_centers_and_radii(P, R, steps=1000, initial_lr=0.002, initial_C=100.0, inject_noise=False)
        
        # Upgrade optimized positions to rigorous maximum values through LP
        R_lp = optimize_radii_lp(P)
        if R_lp is not None:
            R_final = R_lp * 0.9999999
        else:
            scale = compute_valid_scale(P, R)
            R_final = R * scale * 0.9999999
            
        # Enforce exact rigidness fallback resolving precision faults
        scale2 = compute_valid_scale(P, R_final)
        if scale2 < 1.0:
            R_final *= scale2 * 0.9999999
            
        current_sum = np.sum(R_final)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = P.copy()
            best_radii = R_final.copy()
            
        attempt += 1
            
    return best_centers.tolist(), best_radii.tolist(), float(best_sum)