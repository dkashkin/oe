# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def make_valid(centers, radii):
    """Ensure strict geometric validity by sequentially scaling offending radii."""
    n = len(radii)
    # Strictly bind to [0, 1] box limits
    radii = np.minimum(radii, centers[:, 0])
    radii = np.minimum(radii, 1.0 - centers[:, 0])
    radii = np.minimum(radii, centers[:, 1])
    radii = np.minimum(radii, 1.0 - centers[:, 1])
    radii = np.maximum(radii, 1e-9)
    
    # Iteratively remove all circle-to-circle overlaps
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j])
                    radii[i] *= scale * 0.9999999
                    radii[j] *= scale * 0.9999999
                    changed = True
        if not changed:
            break
            
    # Final global shrinking pass for absolute safety against floating point comparison discrepancies
    radii *= 0.999999
    return radii


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    to maximize the sum of their radii using a simulated physics model 
    (Iterative Projected Adam Optimization with penalty scaling).

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    steps = 4000
    runs = 15
    
    best_sum = -1.0
    best_X = None
    best_R = None
    
    # Prior heuristic 1: Seed strategically to bias toward edges and centers 
    p_corners = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
    p_edges = [
        [0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5],
        [0.25, 0.05], [0.75, 0.05], [0.25, 0.95], [0.75, 0.95]
    ]
    p_ring1 = [
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
        [0.5, 0.3], [0.5, 0.7], [0.3, 0.5], [0.7, 0.5]
    ]
    p_center = [[0.5, 0.5], [0.4, 0.4], [0.6, 0.4], [0.4, 0.6], [0.6, 0.6]]
    p_extra = [[0.8, 0.5]]
    
    initial_template = np.array(p_corners + p_edges + p_ring1 + p_center + p_extra)
    
    # Optimizer hyperparams
    beta1, beta2 = 0.9, 0.999
    base_lr = 0.02
    
    # Break perfect symmetry & evaluate multiple starting basins configurations
    for seed in range(runs):
        np.random.seed(42 + seed)
        
        X = initial_template.copy()
        
        # Heuristic 2: Introduce slight random perturbations that grow iteratively based on the run phase
        noise_level = 0.01 + 0.04 * (seed / max(1, runs - 1))
        if seed > 0:
            X += np.random.normal(0, noise_level, (n, 2))
        X = np.clip(X, 0.05, 0.95)
        
        # Heuristic 3: Placement by size - Larger circles initialized robustly near the focal center space
        d_center = np.linalg.norm(X - np.array([0.5, 0.5]), axis=1)
        R = 0.15 - 0.12 * d_center
        R = np.clip(R, 0.02, 0.2)
        # Adding some uniform radii randomization affords topological escaping capability
        R += np.random.uniform(-0.01, 0.01, n)
        
        m_X, v_X = np.zeros_like(X), np.zeros_like(X)
        m_R, v_R = np.zeros_like(R), np.zeros_like(R)
        
        # Heuristic 4: Tight constrained optimization loop with annealed learning rate and temperature bounds 
        for step in range(1, steps + 1):
            progress = step / steps
            
            # Simulated annealing profile: Decaying learning momentum & steeply scaling physical repulsion forces 
            lr = base_lr * (0.001 ** progress)
            lam = 5.0 * (100000.0 / 5.0) ** progress
            
            dX = np.zeros_like(X)
            dR = np.full(n, -1.0)
            
            # Sub-graph pairwise distance calculations globally mapped vector-wise directly in NumPy internals
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :] 
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            
            # Identification of constraint overlapping geometry masks
            overlap = (R[:, np.newaxis] + R[np.newaxis, :]) - dist
            overlap_mask = overlap > 0
            np.fill_diagonal(overlap_mask, False)
            
            # Map structural boundary forces against overlapping intersection logic 
            dR += lam * np.sum(np.where(overlap_mask, overlap, 0.0), axis=1)
            grad_dist_factor = np.where(overlap_mask, lam * overlap / dist, 0.0)
            dX -= np.sum(grad_dist_factor[:, :, np.newaxis] * diff, axis=1)
            
            # Impose soft edge boundary penalties forcing positions safely toward validity throughout 
            bl = R - X[:, 0]
            mask_l = bl > 0
            dR += lam * np.where(mask_l, bl, 0.0)
            dX[:, 0] -= lam * np.where(mask_l, bl, 0.0)
            
            br = R + X[:, 0] - 1.0
            mask_r = br > 0
            dR += lam * np.where(mask_r, br, 0.0)
            dX[:, 0] += lam * np.where(mask_r, br, 0.0)
            
            bb = R - X[:, 1]
            mask_b = bb > 0
            dR += lam * np.where(mask_b, bb, 0.0)
            dX[:, 1] -= lam * np.where(mask_b, bb, 0.0)
            
            bt = R + X[:, 1] - 1.0
            mask_t = bt > 0
            dR += lam * np.where(mask_t, bt, 0.0)
            dX[:, 1] += lam * np.where(mask_t, bt, 0.0)
            
            # Smoothly integrate parameters leveraging purely continuous Adam trajectory gradients  
            m_X = beta1 * m_X + (1 - beta1) * dX
            v_X = beta2 * v_X + (1 - beta2) * (dX**2)
            m_X_hat = m_X / (1 - beta1**step)
            v_X_hat = v_X / (1 - beta2**step)
            X -= lr * m_X_hat / (np.sqrt(v_X_hat) + 1e-8)
            
            m_R = beta1 * m_R + (1 - beta1) * dR
            v_R = beta2 * v_R + (1 - beta2) * (dR**2)
            m_R_hat = m_R / (1 - beta1**step)
            v_R_hat = v_R / (1 - beta2**step)
            R -= lr * m_R_hat / (np.sqrt(v_R_hat) + 1e-8)
            
            R = np.maximum(R, 1e-6)
            
        # Repair potential mathematically miniature validity drifts rigidly enforcing non-overlappable 0 tolerance geometries
        R_valid = make_valid(X, R.copy())
        
        # Save globally maximum radii configuration topologically tested across independent iteration basins
        current_sum = np.sum(R_valid)
        if current_sum > best_sum:
            best_sum = current_sum
            best_X = X.copy()
            best_R = R_valid.copy()
            
    return best_X, best_R, best_sum
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    # visualize(centers, radii)