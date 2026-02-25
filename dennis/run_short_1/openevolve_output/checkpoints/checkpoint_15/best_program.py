# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Construct a heavily optimized arrangement of 26 circles in a unit square
    that tightly maximizes the sum of their radii via simulated annealing physics.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Run a continuous restart scheme to traverse and select best optimal geometries
    for seed in range(12):
        X = optimize_seed(seed)
        final_radii = compute_max_radii(X)
        score = np.sum(final_radii)
        
        if score > best_sum:
            best_sum = score
            best_centers = X
            best_radii = final_radii
            
    return best_centers, best_radii, best_sum


def optimize_seed(seed):
    """Executes gradient ascent physical optimizer applying non-linear repulsions"""
    n = 26
    np.random.seed(seed * 42 + 7)
    X = []
    
    # Generate structural layout biases targeting interstitial properties 
    if seed % 4 == 0:
        grid = np.linspace(0.1, 0.9, 5)
        for x in grid:
            for y in grid:
                X.append([x, y])
        X.append([0.5, 0.5])
    elif seed % 4 == 1:
        X.append([0.5, 0.5])
        for i in range(7):
            a = 2 * np.pi * i / 7
            X.append([0.5 + 0.2 * np.cos(a), 0.5 + 0.2 * np.sin(a)])
        for i in range(18):
            a = 2 * np.pi * i / 18
            X.append([0.5 + 0.45 * np.cos(a), 0.5 + 0.45 * np.sin(a)])
    elif seed % 4 == 2:
        # Bias boundaries strategically
        X.extend([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]])
        for t in np.linspace(0.2, 0.8, 4):
            X.extend([[t, 0.05], [t, 0.95], [0.05, t], [0.95, t]])
        for _ in range(n - len(X)):
            X.append(np.random.uniform(0.2, 0.8, 2).tolist())
    else:
        # Radial symmetry fibonacci break layout 
        for i in range(n):
            r = 0.45 * np.sqrt((i + 1) / float(n))
            theta = i * 2.3999632
            X.append([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
            
    # Apply thermal random breaks ensuring robust positional gradient responses
    X = np.array(X)[:n] + np.random.randn(n, 2) * 0.005
    X = np.clip(X, 0.02, 0.98)
    
    R = np.full(n, 0.06)
    
    m_X, v_X = np.zeros_like(X), np.zeros_like(X)
    m_R, v_R = np.zeros_like(R), np.zeros_like(R)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    steps = 2200
    
    # Advanced gradient scheduling simulated layout physics settling iterations
    for step in range(1, steps + 1):
        progress = step / steps
        lr_base = 0.005 * (1.0 - progress) + 0.0002
        
        # Exponential penalization scaling solidifying boundary geometries gracefully
        C = 5.0 * (2000.0 ** progress)
        
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-12
        
        overlap = R[:, np.newaxis] + R[np.newaxis, :] - dist
        np.fill_diagonal(overlap, -1.0)
        overlap_vals = overlap * (overlap > 0)
        
        grad_X = np.zeros_like(X)
        grad_R = np.full(n, -1.0)
        
        # Dynamic internal forces generated inversely proportional towards interstitial gaps 
        factor = -2.0 * C * overlap_vals / dist 
        grad_X += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        grad_R += np.sum(2.0 * C * overlap_vals, axis=1)
        
        w_l = np.maximum(0, R - X[:, 0])
        w_r = np.maximum(0, R - (1 - X[:, 0]))
        w_b = np.maximum(0, R - X[:, 1])
        w_t = np.maximum(0, R - (1 - X[:, 1]))
        
        grad_X[:, 0] += 2.0 * C * (-w_l + w_r)
        grad_X[:, 1] += 2.0 * C * (-w_b + w_t)
        grad_R += 2.0 * C * (w_l + w_r + w_b + w_t)
        
        # Anneal structural constraints breaking edge symmetry lock-ups securely
        if step < steps * 0.35:
            grad_X += np.random.randn(n, 2) * 0.03 * (1.0 - step / (steps * 0.35))
            
        m_X = beta1 * m_X + (1 - beta1) * grad_X
        v_X = beta2 * v_X + (1 - beta2) * (grad_X ** 2)
        X -= lr_base * (m_X / (1 - beta1 ** step)) / (np.sqrt(v_X / (1 - beta2 ** step)) + eps)
        
        m_R = beta1 * m_R + (1 - beta1) * grad_R
        v_R = beta2 * v_R + (1 - beta2) * (grad_R ** 2)
        R -= lr_base * (m_R / (1 - beta1 ** step)) / (np.sqrt(v_R / (1 - beta2 ** step)) + eps)
        
        X = np.clip(X, 0, 1)
        R = np.maximum(R, 0.0)
        
    return X


def compute_max_radii(centers):
    """
    Compute mathematically optimized strict distributions securely enforcing non-overlap, 
    evaluating geometrical capabilities resiliently without user sequencing prejudice.
    """
    n = centers.shape[0]
    
    # Mathematical attempt evaluating precise extreme maximum geometric boundaries limits
    try:
        from scipy.optimize import linprog
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        for i in range(n):
            for j in range(i+1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(np.linalg.norm(centers[i] - centers[j]))
                
        bounds = []
        for i in range(n):
            safe_wall = max(0.0, min(centers[i, 0], centers[i, 1], 
                                     1.0 - centers[i, 0], 1.0 - centers[i, 1]) - 1e-8)
            bounds.append((0, safe_wall))
            
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if res.success:
            return np.maximum(0.0, res.x - 1e-6)
    except Exception:
        pass

    # Unprejudiced stable dynamic geometric compression graceful-recovery fallback
    r = np.zeros(n)
    for i in range(n):
        r[i] = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
        
    dist = np.sqrt(np.sum((centers[:, None] - centers[None, :]) ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    for _ in range(3000):
        r += 0.002
        max_walls = np.minimum.reduce([centers[:, 0], 1 - centers[:, 0], 
                                       centers[:, 1], 1 - centers[:, 1]])
        r = np.minimum(r, max_walls)
        
        overlap = r[:, None] + r[None, :] - dist
        if np.max(overlap) > 1e-9:
            excess = np.maximum(0, overlap)
            scale = r[:, None] / (r[:, None] + r[None, :] + 1e-9)
            r -= np.sum(excess * scale, axis=1) * 0.5

    # Assured residual overlap removal phase mathematically assuring clean submission states
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if r[i] + r[j] > dist[i, j] + 1e-9:
                    excess = r[i] + r[j] - dist[i, j]
                    r[i] -= excess / 2.0
                    r[j] -= excess / 2.0
                    changed = True
        if not changed:
            break
            
    return np.maximum(0.0, r - 1e-6)

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