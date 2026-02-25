# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii without overlap.
    
    Uses an iterative constrained penalty optimization initialized via
    a diverse multi-start technique. A sequence of Adam updates minimizes
    constraint violations while maintaining outward expansion of radii.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii strictly valid
    """
    B = 100
    N = 26
    steps = 1500
    
    np.random.seed(42)
    
    # 1: Pure random initialization strictly inside valid margins
    B1 = B // 3
    X1 = np.random.uniform(0.05, 0.95, (B1, N))
    Y1 = np.random.uniform(0.05, 0.95, (B1, N))
    
    # 2: Grid-like initialized configuration (removes spatial cluttering risks)
    B2 = B // 3
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 6), np.linspace(0.1, 0.9, 5))
    grid_pts = np.c_[grid_x.ravel(), grid_y.ravel()] 
    X2 = np.zeros((B2, N))
    Y2 = np.zeros((B2, N))
    for b in range(B2):
        idx = np.random.choice(30, N, replace=False)
        pts = grid_pts[idx] + np.random.normal(0, 0.015, (N, 2))
        X2[b] = pts[:, 0]
        Y2[b] = pts[:, 1]
        
    # 3: Edge-biased initialization (creates favorable layout patterns naturally)
    B3 = B - B1 - B2
    X3 = np.random.beta(0.5, 0.5, (B3, N)) * 0.9 + 0.05
    Y3 = np.random.beta(0.5, 0.5, (B3, N)) * 0.9 + 0.05
    
    X = np.vstack([X1, X2, X3])
    Y = np.vstack([Y1, Y2, Y3])
    R = np.random.uniform(0.01, 0.05, (B, N))
    
    # Custom vectorized multi-agent Adam Optimizer setup
    lr = 0.04
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    m_X, v_X = np.zeros_like(X), np.zeros_like(X)
    m_Y, v_Y = np.zeros_like(Y), np.zeros_like(Y)
    m_R, v_R = np.zeros_like(R), np.zeros_like(R)
    
    # Self-exclusion matrix mask for pairwise constraint calculations
    mask = ~np.eye(N, dtype=bool)
    
    # Exponential increasing penalty acts identically to a mathematical cooling regime
    start_pen = 50.0
    end_pen = 3000.0
    pen_factor = (end_pen / start_pen) ** (1.0 / steps)
    penalty_weight = start_pen
    
    for step in range(1, steps + 1):
        # Linearly decay learning rate completely
        current_lr = lr * (1.0 - step / steps)
        
        # Gather full batch metric evaluations efficiently
        dX = X[:, :, np.newaxis] - X[:, np.newaxis, :]
        dY = Y[:, :, np.newaxis] - Y[:, np.newaxis, :]
        dist = np.sqrt(dX**2 + dY**2 + 1e-12)
        
        # Detect strict overlaps safely separated off exact pairs structurally
        overlap = np.maximum(0, R[:, :, np.newaxis] + R[:, np.newaxis, :] - dist)
        overlap = overlap * mask[np.newaxis, :, :]
        
        # Calculate wall overlap margins directly avoiding outer boundaries 
        v_x1 = np.maximum(0, R - X)
        v_x2 = np.maximum(0, X + R - 1.0)
        v_y1 = np.maximum(0, R - Y)
        v_y2 = np.maximum(0, Y + R - 1.0)
        
        # Composite spatial gradients and size pressure calculations
        grad_R_overlap = 2.0 * np.sum(overlap, axis=2)
        grad_R_bounds = 2.0 * (v_x1 + v_x2 + v_y1 + v_y2)
        
        dir_X = dX / dist
        dir_Y = dY / dist
        grad_X_overlap = -2.0 * np.sum(overlap * dir_X, axis=2)
        grad_Y_overlap = -2.0 * np.sum(overlap * dir_Y, axis=2)
        
        grad_X_bounds = 2.0 * (-v_x1 + v_x2)
        grad_Y_bounds = 2.0 * (-v_y1 + v_y2)
        
        # Objective expansion gradient pushes R continuously + weighted constraints pull 
        grad_R = -1.0 + penalty_weight * (grad_R_overlap + grad_R_bounds)
        grad_X = penalty_weight * (grad_X_overlap + grad_X_bounds)
        grad_Y = penalty_weight * (grad_Y_overlap + grad_Y_bounds)
        
        # Momentum & variance stabilization step integrations internally
        bias1_corr = 1.0 - beta1**step
        bias2_corr = 1.0 - beta2**step
        
        m_X = beta1 * m_X + (1.0 - beta1) * grad_X
        v_X = beta2 * v_X + (1.0 - beta2) * (grad_X**2)
        X -= current_lr * (m_X / bias1_corr) / (np.sqrt(v_X / bias2_corr) + epsilon)
        
        m_Y = beta1 * m_Y + (1.0 - beta1) * grad_Y
        v_Y = beta2 * v_Y + (1.0 - beta2) * (grad_Y**2)
        Y -= current_lr * (m_Y / bias1_corr) / (np.sqrt(v_Y / bias2_corr) + epsilon)
        
        m_R = beta1 * m_R + (1.0 - beta1) * grad_R
        v_R = beta2 * v_R + (1.0 - beta2) * (grad_R**2)
        R -= current_lr * (m_R / bias1_corr) / (np.sqrt(v_R / bias2_corr) + epsilon)
        
        penalty_weight *= pen_factor

    # Find fundamentally stable and correct instance safely mapping coordinates
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    for b in range(B):
        x = np.copy(X[b])
        y = np.copy(Y[b])
        r = np.copy(R[b])
        
        # Hard lock mathematical conditions sequentially safely overriding variations
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        y = np.clip(y, 1e-6, 1.0 - 1e-6)
        r = np.clip(r, 0.0, 0.5)
        
        # Trim off immediately exterior overlaps
        r = np.minimum.reduce([r, x, 1.0 - x, y, 1.0 - y])
        
        # Scale back safely into fully acceptable layout ensuring valid overlap metric natively 
        for i in range(N):
            for j in range(i + 1, N):
                dist_val = np.hypot(x[i] - x[j], y[i] - y[j])
                if r[i] + r[j] > dist_val:
                    scale = (dist_val - 1e-9) / (r[i] + r[j])
                    if scale < 0: 
                        scale = 0.0
                    r[i] *= scale
                    r[j] *= scale
                    
        # Apply extremely soft padding preserving structure absolutely securely
        r *= 0.99999
        r = np.maximum(r, 0.0)
        
        current_sum = np.sum(r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = np.column_stack((x, y))
            best_radii = r
            
    return best_centers, best_radii, best_sum

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