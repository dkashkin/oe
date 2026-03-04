import numpy as np
from scipy.optimize import linprog
import time

def solve_max_radii(pts):
    """
    Computes optimal radii for fixed centers using Linear Programming.
    Maximizes sum(r_i) subject to r_i + r_j <= distance(P_i, P_j) and
    boundary constraints r_i <= wall_distance.
    Returns optimal radii, dual variables (marginals), and pairwise info.
    """
    n = pts.shape[0]
    # Scipy linprog minimizes -sum(r_i) to maximize sum(r_i)
    c_lp = -np.ones(n)
    
    # Precompute indices for pairwise distance constraints
    i_idx, j_idx = np.triu_indices(n, 1)
    dists = np.linalg.norm(pts[i_idx] - pts[j_idx], axis=1)
    num_pairs = len(i_idx)
    
    # Combined pairwise and boundary constraints into one A_ub matrix
    # Total constraints: num_pairs + 4 * n
    A_ub = np.zeros((num_pairs + 4 * n, n))
    b_ub = np.zeros(num_pairs + 4 * n)
    
    # Pairwise constraints: r_i + r_j <= distance_ij
    for k, (i, j) in enumerate(zip(i_idx, j_idx)):
        A_ub[k, i] = 1
        A_ub[k, j] = 1
        b_ub[k] = dists[k] - 1e-11
        
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    x, y = pts[:, 0], pts[:, 1]
    for i in range(n):
        base = num_pairs + 4 * i
        A_ub[base, i], b_ub[base] = 1, x[i] - 1e-11
        A_ub[base + 1, i], b_ub[base + 1] = 1, 1.0 - x[i] - 1e-11
        A_ub[base + 2, i], b_ub[base + 2] = 1, y[i] - 1e-11
        A_ub[base + 3, i], b_ub[base + 3] = 1, 1.0 - y[i] - 1e-11
        
    try:
        # Use HiGHS solver for speed and access to dual marginals
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success and hasattr(res, 'ineqlin') and res.ineqlin is not None:
            return res.x, res.ineqlin.marginals, i_idx, j_idx, dists
    except Exception:
        pass
        
    return np.zeros(n), None, i_idx, j_idx, dists

def construct_packing():
    """
    Optimizes the arrangement of 26 circles to maximize the sum of their radii.
    Utilizes multi-start configurations and dual-gradient descent.
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    start_time = time.time()
    
    # Diverse initial layouts for multi-start optimization
    inits = []
    
    # Layout 1: 5x5 Grid with one circle near the center
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    c1 = np.vstack([np.column_stack([gx.ravel(), gy.ravel()]), [0.51, 0.49]])
    inits.append(c1)
    
    # Layout 2: Staggered layers (5, 4, 5, 4, 5, 3) to fill the square
    c2 = []
    layers = [5, 4, 5, 4, 5, 3]
    for r_idx, n_row in enumerate(layers):
        for i_idx in range(n_row):
            c2.append([(i_idx + 0.5) / n_row, (r_idx + 0.5) / len(layers)])
    inits.append(np.array(c2))
    
    # Layout 3: Ring distribution (1 center, 8 middle, 17 outer)
    c3 = [[0.5, 0.5]]
    for rad, count in [(0.23, 8), (0.45, 17)]:
        for i in range(count):
            a = 2 * np.pi * i / count
            c3.append([0.5 + rad * np.cos(a), 0.5 + rad * np.sin(a)])
    inits.append(np.array(c3))

    # Layout 4: Another staggered layout (4, 5, 4, 5, 4, 4)
    c4 = []
    layers4 = [4, 5, 4, 5, 4, 4]
    for r_idx, n_row in enumerate(layers4):
        for i_idx in range(n_row):
            c4.append([(i_idx + 0.5) / n_row, (r_idx + 0.5) / len(layers4)])
    inits.append(np.array(c4))
    
    # Layout 5: Random distribution
    np.random.seed(42)
    inits.append(np.random.uniform(0.1, 0.9, (n, 2)))
    
    # Optimization loop per initialization
    for c_start in inits:
        curr_c = np.clip(c_start.copy(), 1e-7, 1.0 - 1e-7)
        lr = 0.008  # Initial learning rate
        
        for step in range(350):
            # Periodically check time limit
            if time.time() - start_time > 560:
                break
                
            radii, duals, i_idx, j_idx, dists = solve_max_radii(curr_c)
            if duals is None:
                break
                
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = curr_c.copy()
                best_radii = radii.copy()
            
            # Construct gradient based on dual variables (marginals)
            # Duals tell us how the sum of radii changes with center positions
            grad = np.zeros((n, 2))
            num_pairs = len(i_idx)
            
            # Gradients from pairwise distances
            lambdas = np.abs(duals[:num_pairs])
            active_mask = lambdas > 1e-8
            if np.any(active_mask):
                ii = i_idx[active_mask]
                jj = j_idx[active_mask]
                lams = lambdas[active_mask]
                diffs = curr_c[ii] - curr_c[jj]
                ds = dists[active_mask][:, np.newaxis]
                # Direction of movement to increase the distance
                g_vals = (lams[:, np.newaxis] * diffs / (ds + 1e-12))
                np.add.at(grad, ii, g_vals)
                np.add.at(grad, jj, -g_vals)
            
            # Gradients from boundary constraints
            # Sensitivity to moving away from the four walls
            mu0 = np.abs(duals[num_pairs + 4 * np.arange(n)])      # r_i <= x_i
            mu1 = np.abs(duals[num_pairs + 4 * np.arange(n) + 1])  # r_i <= 1-x_i
            mu2 = np.abs(duals[num_pairs + 4 * np.arange(n) + 2])  # r_i <= y_i
            mu3 = np.abs(duals[num_pairs + 4 * np.arange(n) + 3])  # r_i <= 1-y_i
            grad[:, 0] += mu0 - mu1
            grad[:, 1] += mu2 - mu3
            
            # Apply update with normalized gradient for stability
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-10:
                curr_c += lr * grad / gnorm
            
            # Keep centers within square and decay learning rate
            curr_c = np.clip(curr_c, 1e-7, 1.0 - 1e-7)
            lr *= 0.995
            
    # Final cleanup to ensure radii are optimal for the best centers found
    final_radii, _, _, _, _ = solve_max_radii(best_centers)
    return best_centers, final_radii, np.sum(final_radii)

def run_packing():
    """Entry point for evaluation scripts."""
    return construct_packing()

if __name__ == "__main__":
    centers, radii, total_sum = construct_packing()
    print(f"Total sum of radii: {total_sum:.6f}")