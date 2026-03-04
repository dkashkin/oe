import time
import numpy as np


def compute_valid_scale(P, R):
    """
    Computes a scaling factor to guarantee that no circles overlap and all remain
    strictly within the [0, 1] bounds, correcting any floating point inaccuracies.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    
    R_sum = R[:, np.newaxis] + R[np.newaxis, :]
    scale_pairs = np.min(dist / (R_sum + 1e-10))
    
    scale_left = np.min(P[:, 0] / (R + 1e-10))
    scale_right = np.min((1 - P[:, 0]) / (R + 1e-10))
    scale_bottom = np.min(P[:, 1] / (R + 1e-10))
    scale_top = np.min((1 - P[:, 1]) / (R + 1e-10))
    
    return max(0.0, min(1.0, scale_pairs, scale_left, scale_right, scale_bottom, scale_top))


def optimize_radii_lp(P):
    """
    Given a fixed set of centers, calculates the mathematically optimal radii 
    using Linear Programming to maximize their sum.
    """
    N = len(P)
    try:
        from scipy.optimize import linprog
        c = -np.ones(N)
        A_ub = []
        b_ub = []
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(P[i] - P[j])
                A_row = np.zeros(N)
                A_row[i] = 1
                A_row[j] = 1
                A_ub.append(A_row)
                b_ub.append(max(float(dist), 1e-8))
                
        bounds = [(0.0, max(0.0, min(float(P[i, 0]), 1.0 - float(P[i, 0]), float(P[i, 1]), 1.0 - float(P[i, 1])))) for i in range(N)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if res.success:
            return res.x * 0.9999999  # Slightly scale to ensure strict bound compliance
    except Exception:
        pass
    return None


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
    # Run multiple restarts to avoid local minima, stopping well before timeout
    while attempt < 40 and (time.time() - start_time) < 25.0:
        P = np.random.rand(N, 2) * 0.8 + 0.1
        
        # Strategic topological initializations
        mode = attempt % 5
        if mode == 0:
            # Emphasize corners and edges
            P[0:4] = [[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]]
            P[4:8] = [[0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
            P[8] = [0.5, 0.5]
        elif mode == 1:
            # Concentric distribution
            for i in range(8):
                angle = 2 * np.pi * i / 8
                P[i] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
        elif mode == 2:
            # Standard grid structure
            idx = 0
            for x in np.linspace(0.1, 0.9, 5):
                for y in np.linspace(0.1, 0.9, 5):
                    if idx < 25:
                        P[idx] = [x, y]
                        idx += 1
            P[25] = [0.5, 0.5]
        elif mode == 3:
            # Dense corner clusters
            for i in range(12):
                P[i] = np.random.rand(2) * 0.2
                if i % 4 == 1: P[i, 0] += 0.8
                elif i % 4 == 2: P[i, 1] += 0.8
                elif i % 4 == 3: P[i] += 0.8
                
        # Initialize initial radii inversely proportional to their distance from centroid
        dist_to_center = np.linalg.norm(P - 0.5, axis=1)
        R = 0.06 - 0.03 * dist_to_center
        
        # Optimization hyperparameters
        lr = 0.015
        C = 1.0
        steps = 4000
        
        m_P, v_P = np.zeros_like(P), np.zeros_like(P)
        m_R, v_R = np.zeros_like(R), np.zeros_like(R)
        
        # Physics-based constraint optimization via Adam gradient descent
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
            
            # Gradients for radii maximization + penalty bounds
            grad_R = -1.0 + 2 * C * (left + right + bottom + top + np.sum(overlap_masked, axis=1))
            
            # Gradients for center positional adjustments
            grad_P = np.zeros_like(P)
            grad_P[:, 0] += 2 * C * (-left + right)
            grad_P[:, 1] += 2 * C * (-bottom + top)
            
            weight = 2 * C * overlap_masked / dist_safe
            grad_P -= np.sum(weight[:, :, np.newaxis] * diff, axis=1)
            
            t = step + 1
            
            # Apply Adam updates
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
            
            # Simulated Annealing mechanism
            C *= 1.002
            lr *= 0.9995
            
            # Inject micro-perturbations to break layout symmetries
            if step % 800 == 0 and step < 2500:
                P += np.random.randn(*P.shape) * 0.001
                
        # Resolve any residual errors strictly
        P = np.clip(P, 1e-5, 1 - 1e-5)
        
        # Elevate to absolute mathematical optimum via Linear Programming
        R_lp = optimize_radii_lp(P)
        if R_lp is not None:
            R_final = R_lp
        else:
            scale = compute_valid_scale(P, R)
            R_final = R * scale * 0.9999999
            
        current_sum = np.sum(R_final)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = P.copy()
            best_radii = R_final.copy()
            
        attempt += 1
            
    return best_centers.tolist(), best_radii.tolist(), float(best_sum)