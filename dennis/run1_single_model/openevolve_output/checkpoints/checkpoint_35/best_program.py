# EVOLVE-BLOCK-START
"""
Constructor-based circle packing for n=26 circles in a unit square.
This program uses a dual-gradient optimization approach to find circle centers
that maximize the sum of radii. For any set of fixed centers, the optimal radii
are computed using Linear Programming. The gradients for the centers are then
derived from the dual variables of this LP.
"""
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
    
    # We combine pairwise and boundary constraints into one A_ub matrix
    # to obtain dual variables for both simultaneously.
    # Total constraints: num_pairs + 4 * n
    A_ub = np.zeros((num_pairs + 4 * n, n))
    b_ub = np.zeros(num_pairs + 4 * n)
    
    # Pairwise constraints: r_i + r_j <= distance_ij
    for k, (i, j) in enumerate(zip(i_idx, j_idx)):
        A_ub[k, i] = 1
        A_ub[k, j] = 1
        # Subtract a tiny epsilon to ensure strict validity against float errors
        b_ub[k] = max(0, dists[k] - 1e-11)
        
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    x, y = pts[:, 0], pts[:, 1]
    for i in range(n):
        base = num_pairs + 4 * i
        A_ub[base, i] = 1
        b_ub[base] = max(0, x[i] - 1e-11)
        A_ub[base + 1, i] = 1
        b_ub[base + 1] = max(0, 1.0 - x[i] - 1e-11)
        A_ub[base + 2, i] = 1
        b_ub[base + 2] = max(0, y[i] - 1e-11)
        A_ub[base + 3, i] = 1
        b_ub[base + 3] = max(0, 1.0 - y[i] - 1e-11)
        
    try:
        # Use HiGHS solver for speed and access to dual marginals
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    except Exception:
        # Fallback for environments with older scipy versions
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
        
    if res.success and hasattr(res, 'ineqlin') and res.ineqlin is not None:
        return res.x, res.ineqlin.marginals, i_idx, j_idx, dists
    return np.zeros(n), None, i_idx, j_idx, dists


def run_packing():
    """
    Core function to optimize the arrangement of 26 circles.
    Utilizes multiple starting configurations and dual-gradient descent.
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    start_time = time.time()
    
    # Define diverse initial layouts for multi-start optimization
    inits = []
    
    # Layout 1: 5x5 Grid with one extra circle at center
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    c1 = np.vstack([np.column_stack([gx.ravel(), gy.ravel()]), [0.5, 0.5]])
    inits.append(c1)
    
    # Layout 2: Staggered layers (5, 4, 5, 4, 5, 3) to fill the square
    c2 = []
    layers = [5, 4, 5, 4, 5, 3]
    for r_idx, n_row in enumerate(layers):
        for i_idx in range(n_row):
            c2.append([(i_idx + 0.5) / n_row, (r_idx + 0.5) / len(layers)])
    inits.append(np.array(c2)[:n])
    
    # Layout 3: Concentric circles (1 center, 7 middle, 18 outer)
    c3 = [[0.5, 0.5]]
    for rad, count in [(0.22, 7), (0.42, 18)]:
        for i in range(count):
            a = 2 * np.pi * i / count
            c3.append([0.5 + rad * np.cos(a), 0.5 + rad * np.sin(a)])
    inits.append(np.array(c3)[:n])
    
    # Layout 4: Random distribution
    np.random.seed(42)
    inits.append(np.random.uniform(0.1, 0.9, (n, 2)))
    
    # Optimization loop per initialization
    for c_start in inits:
        curr_c = np.clip(c_start.copy(), 1e-7, 1.0 - 1e-7)
        lr = 0.006  # Initial learning rate
        
        for step in range(400):
            # Periodically check time limit (600s total)
            if time.time() - start_time > 550:
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
            # Scipy returns negative duals for minimization; we use absolute values
            lambdas = np.abs(duals[:num_pairs])
            active_mask = lambdas > 1e-7
            if np.any(active_mask):
                ii = i_idx[active_mask]
                jj = j_idx[active_mask]
                lams = lambdas[active_mask]
                diffs = curr_c[ii] - curr_c[jj]
                ds = dists[active_mask][:, np.newaxis]
                g_vals = (lams[:, np.newaxis] * diffs / (ds + 1e-12))
                np.add.at(grad, ii, g_vals)
                np.add.at(grad, jj, -g_vals)
            
            # Gradients from boundary constraints
            mu0 = np.abs(duals[num_pairs + 4 * np.arange(n)])
            mu1 = np.abs(duals[num_pairs + 4 * np.arange(n) + 1])
            mu2 = np.abs(duals[num_pairs + 4 * np.arange(n) + 2])
            mu3 = np.abs(duals[num_pairs + 4 * np.arange(n) + 3])
            grad[:, 0] += mu0 - mu1
            grad[:, 1] += mu2 - mu3
            
            # Apply update with normalized gradient for stability
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-12:
                curr_c += lr * grad / gnorm
            
            # Keep centers strictly within square and decay learning rate
            curr_c = np.clip(curr_c, 1e-7, 1.0 - 1e-7)
            lr *= 0.994
            
    return best_centers, best_radii, best_sum


def construct_packing():
    """Wrapper to meet the standard constructor signature."""
    return run_packing()


if __name__ == "__main__":
    centers, radii, total_sum = run_packing()
    print(f"Total sum of radii: {total_sum:.6f}")
# EVOLVE-BLOCK-END