# EVOLVE-BLOCK-START
"""Physics-based iterative optimizer for circle packing n=26"""
import numpy as np


def adam_optimization(seed, n, steps):
    """
    Runs a physics-simulated Adam optimizer to find maximum sum-of-radii 
    configuration of N circles in a unit square.
    """
    np.random.seed(seed)
    init_mode = seed % 4
    
    # Heuristic 1: Geometric seeding strategies to cover wide space of arrangements
    if init_mode == 0:
        # Mitchell's Best Candidate starting from center
        X = np.zeros((n, 2))
        X[0] = [0.5, 0.5]
        for i in range(1, n):
            cands = np.random.uniform(0.05, 0.95, (50, 2))
            diff = cands[:, np.newaxis, :] - X[np.newaxis, :i, :]
            dists = np.min(np.linalg.norm(diff, axis=-1), axis=1)
            X[i] = cands[np.argmax(dists)]
            
    elif init_mode == 1:
        # Four corners bias + Best Candidate
        X = np.zeros((n, 2))
        X[0:4] = [[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]]
        for i in range(4, n):
            cands = np.random.uniform(0.05, 0.95, (50, 2))
            diff = cands[:, np.newaxis, :] - X[np.newaxis, :i, :]
            dists = np.min(np.linalg.norm(diff, axis=-1), axis=1)
            X[i] = cands[np.argmax(dists)]
            
    elif init_mode == 2:
        # Edge placements + Best Candidate
        X = np.zeros((n, 2))
        edges = []
        for i in range(8):
            u = np.random.uniform(0.1, 0.9)
            if i % 4 == 0: edges.append([0.05, u])
            elif i % 4 == 1: edges.append([0.95, u])
            elif i % 4 == 2: edges.append([u, 0.05])
            else: edges.append([u, 0.95])
        X[0:8] = edges
        for i in range(8, n):
            cands = np.random.uniform(0.05, 0.95, (50, 2))
            diff = cands[:, np.newaxis, :] - X[np.newaxis, :i, :]
            dists = np.min(np.linalg.norm(diff, axis=-1), axis=1)
            X[i] = cands[np.argmax(dists)]
            
    else:
        # Baseline uniform scatter 
        X = np.random.uniform(0.1, 0.9, (n, 2))
    
    # Initialize radii uniformly; the physics simulation adaptively sizes them into gaps
    R = np.ones(n) * 0.05
    
    mX, vX = np.zeros_like(X), np.zeros_like(X)
    mR, vR = np.zeros_like(R), np.zeros_like(R)
    
    eye_n = np.eye(n, dtype=bool)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    for t in range(1, steps + 1):
        frac = t / steps
        # Decaying learning rate smoothly and a steep power-law annealing profile
        lr = 0.03 * (1.0 - frac)**2 + 0.0005
        lam = 5.0 + 3000.0 * (frac ** 3) 
        
        # 2. Symmetry breaking to occasionally bump out of saddle equilibrium constraints
        if t % 400 == 0 and t < steps * 0.7:
            X += np.random.normal(0, 0.003, X.shape)
            
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :] 
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.maximum(dist, 1e-8)
        dist[eye_n] = 1.0
        
        sum_R = R[:, np.newaxis] + R[np.newaxis, :]
        overlap = np.maximum(0, sum_R - dist)
        overlap[eye_n] = 0.0
        
        # Compute overlapping geometry collision force and structural penalty
        grad_R_overlap = 2 * lam * np.sum(overlap, axis=1)
        overlap_weight = 2 * lam * overlap / dist
        grad_X_overlap = -np.einsum('ij,ijk->ik', overlap_weight, diff)
        
        # Calculate forces from bounds of unit area domain dynamically
        bx0 = np.maximum(0, R - X[:, 0])
        bx1 = np.maximum(0, R + X[:, 0] - 1)
        by0 = np.maximum(0, R - X[:, 1])
        by1 = np.maximum(0, R + X[:, 1] - 1)
        
        grad_R_bound = 2 * lam * (bx0 + bx1 + by0 + by1)
        
        grad_X_bound = np.zeros_like(X)
        grad_X_bound[:, 0] = 2 * lam * (-bx0 + bx1)
        grad_X_bound[:, 1] = 2 * lam * (-by0 + by1)
        
        # Constant tendency: optimize to maximize area without size prejudice directly
        grad_R_growth = -1.0
        
        grad_X = grad_X_overlap + grad_X_bound
        grad_R = grad_R_overlap + grad_R_bound + grad_R_growth
        
        # Iteration of parameters using adaptive Adam heuristics for positional momentum
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
        
        # Protect positional bounds minimally against explosive initial impulses 
        X = np.clip(X, 1e-4, 1.0 - 1e-4)
        R = np.maximum(R, 1e-4)

    # 3. Soft relaxation step: geometrically shunts nodes iteratively avoiding cutting bounds immediately
    for _ in range(100):
        for i in range(n):
            for j in range(i + 1, n):
                diff = X[i] - X[j]
                d = np.linalg.norm(diff)
                overlap_amt = R[i] + R[j] - d
                if overlap_amt > 1e-12:
                    push = (diff / (d + 1e-12)) * overlap_amt * 0.5
                    X[i] += push
                    X[j] -= push
        # Box positioning properly again using the strict current radii as border constraints
        for i in range(n):
            safe_margin = max(R[i], 1.0 - R[i])
            X[i, 0] = np.clip(X[i, 0], R[i], safe_margin)
            X[i, 1] = np.clip(X[i, 1], R[i], safe_margin)

    # 4. Rigorous numerical sub-float resolution guarantees strictly zero infractions legally mathematically
    for i in range(n):
        R[i] = min(R[i], X[i, 0], X[i, 1], 1 - X[i, 0], 1 - X[i, 1])
        R[i] = max(0, R[i])
        
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if R[i] + R[j] > d:
                # Retains proportionally accurate local scaling offsets explicitly avoiding false overlap
                scale = d / (R[i] + R[j] + 1e-12)
                buffer = 0.9999999
                R[i] *= scale * buffer
                R[j] *= scale * buffer
                
    return X, R, float(np.sum(R))


def construct_packing():
    """
    Searches diverse structural permutations via topological biasing, resolving a 
    globally strong configuration sequentially guaranteeing optimally reduced void volumes.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_X = None
    best_R = None
    best_sum = -1.0
    
    # Run iterations ensuring broad spectrum discovery utilizing 4 separate initialization branches
    for seed in range(42, 52):  # Tests spanning 10 distinct simulation trials
        X, R, total = adam_optimization(seed, n, steps=6000)
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