# EVOLVE-BLOCK-START
"""Physics-based iterative optimizer for circle packing n=26"""
import numpy as np


def adam_optimization(seed, n, steps):
    """
    Runs a physics-simulated Adam optimizer to find maximum sum-of-radii 
    configuration of N circles in a unit square.
    """
    np.random.seed(seed)
    
    # Initialize randomly
    X = np.random.uniform(0.1, 0.9, (n, 2))
    
    # 1. Bias initial placements toward corners and edges to maximize space
    # 4 corners
    X[0] = [0.05, 0.05]
    X[1] = [0.05, 0.95]
    X[2] = [0.95, 0.05]
    X[3] = [0.95, 0.95]
    
    # 8 edges
    for i in range(8):
        u = np.random.uniform(0.1, 0.9)
        if i % 4 == 0: X[i + 4] = [0.05, u]
        elif i % 4 == 1: X[i + 4] = [0.95, u]
        elif i % 4 == 2: X[i + 4] = [u, 0.05]
        elif i % 4 == 3: X[i + 4] = [u, 0.95]

    # Bias largest initial toward the center
    X[12] = [0.5, 0.5]
    
    # Radii initialization: larger in center
    R = np.ones(n) * 0.05
    R[0:4] = 0.08   # start corners moderately
    R[12] = 0.15    # start center larger
    
    mX, vX = np.zeros_like(X), np.zeros_like(X)
    mR, vR = np.zeros_like(R), np.zeros_like(R)
    
    eye_n = np.eye(n, dtype=bool)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    for t in range(1, steps + 1):
        # Decaying learning rate and simulated annealing of the collision penalty
        lr = 0.02 * (1.0 - t / steps) + 0.001
        lam = 20.0 + 800.0 * (t / steps) ** 2 
        
        # 2. Break perfect symmetry during intermediate phases to escape local maxima
        if t % 500 == 0 and t < steps - 500:
            X += np.random.normal(0, 0.005, X.shape)
            
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :] 
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.maximum(dist, 1e-8)
        dist[eye_n] = 1.0
        
        sum_R = R[:, np.newaxis] + R[np.newaxis, :]
        overlap = np.maximum(0, sum_R - dist)
        overlap[eye_n] = 0.0
        
        # Penalize collision
        grad_R_overlap = 2 * lam * np.sum(overlap, axis=1)
        overlap_weight = 2 * lam * overlap / dist
        grad_X_overlap = -np.einsum('ij,ijk->ik', overlap_weight, diff)
        
        # Boundary collision calculations
        bx0 = np.maximum(0, R - X[:, 0])
        bx1 = np.maximum(0, R + X[:, 0] - 1)
        by0 = np.maximum(0, R - X[:, 1])
        by1 = np.maximum(0, R + X[:, 1] - 1)
        
        grad_R_bound = 2 * lam * (bx0 + bx1 + by0 + by1)
        
        grad_X_bound = np.zeros_like(X)
        grad_X_bound[:, 0] = 2 * lam * (-bx0 + bx1)
        grad_X_bound[:, 1] = 2 * lam * (-by0 + by1)
        
        # Always expand unless penalized
        grad_R_growth = -1.0
        
        grad_X = grad_X_overlap + grad_X_bound
        grad_R = grad_R_overlap + grad_R_bound + grad_R_growth
        
        mX = beta1 * mX + (1 - beta1) * grad_X
        vX = beta2 * vX + (1 - beta2) * (grad_X ** 2)
        mX_hat = mX / (1 - beta1 ** t)
        vX_hat = vX / (1 - beta2 ** t)
        X -= lr * mX_hat / (np.sqrt(vX_hat) + eps)
        
        mR = beta1 * mR + (1 - beta1) * grad_R
        vR = beta2 * vR + (1 - beta2) * (grad_R ** 2)
        mR_hat = mR / (1 - beta1 ** t)
        vR_hat = vR / (1 - beta2 ** t)
        R -= lr * mR_hat / (np.sqrt(vR_hat) + eps)
        
        # Strongly box particles into unit square to maintain simulation stability
        X = np.clip(X, 1e-4, 1.0 - 1e-4)
        R = np.maximum(R, 1e-4)

    # 3. Clean-up phase ensures exactly mathematically strictly valid final arrangements
    # Trim radii against edges first
    for i in range(n):
        R[i] = min(R[i], X[i, 0], X[i, 1], 1 - X[i, 0], 1 - X[i, 1])
        R[i] = max(0, R[i])
        
    # Strictly reduce proportional pairwise to eradicate any trace of overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if R[i] + R[j] > d:
                # Calculate scale that guarantees safety, and add buffer to offset floats
                scale = d / (R[i] + R[j] + 1e-12)
                buffer = 0.999999
                R[i] *= scale * buffer
                R[j] *= scale * buffer
                
    return X, R, float(np.sum(R))


def construct_packing():
    """
    Repeatedly searches over various starting orientations
    to report a global packing configuration of n=26 
    minimizing interstitial volume perfectly.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_X = None
    best_R = None
    best_sum = -1.0
    
    # Searching across diverse seeds reliably avoids local failures
    for seed in range(42, 50):  # 8 distinct initialized runs
        X, R, total = adam_optimization(seed, n, steps=3000)
        if total > best_sum:
            best_sum = total
            best_X = X.copy()
            best_R = R.copy()
            
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