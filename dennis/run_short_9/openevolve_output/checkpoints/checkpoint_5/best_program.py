# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def compute_max_radii(centers):
    """
    Compute exactly maximal valid radii for this configuration 
    using linprog to maximize sum of radii exactly given fixed centers.
    Falls back to a safe heuristic method if scipy isn't present.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    
    try:
        from scipy.optimize import linprog
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # 1. Square border boundary constraints
        for i in range(n):
            x, y = centers[i]
            bnd = min(x, y, 1.0 - x, 1.0 - y)
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bnd)
            
        # 2. Pairwise boundary distance constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist)
                
        # Resolve via optimal mapping mathematically guaranteeing no overlaps natively
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return res.x
    except ImportError:
        pass
        
    # Original order-dependent heuristic projection if strictly falling back
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


def get_loss_and_grad(Z, penalty_weight):
    """
    Analytical forward propagation and exact gradient derivation 
    for solving optimal configurations via physical expansion simulations.
    """
    x, y, r = Z[:, 0], Z[:, 1], Z[:, 2]
    
    grad = np.zeros_like(Z)
    grad[:, 2] = -1.0  # Encourage uniform mathematical growth
    loss = -np.sum(r)
    
    # Boundary x min overlap physics penalties
    viol_xmin = -x + r
    m_mask = viol_xmin > 0
    if np.any(m_mask):
        loss += penalty_weight * np.sum(viol_xmin[m_mask] ** 2)
        grad[m_mask, 0] -= 2 * penalty_weight * viol_xmin[m_mask]
        grad[m_mask, 2] += 2 * penalty_weight * viol_xmin[m_mask]
        
    # Boundary x max
    viol_xmax = x + r - 1.0
    m_mask = viol_xmax > 0
    if np.any(m_mask):
        loss += penalty_weight * np.sum(viol_xmax[m_mask] ** 2)
        grad[m_mask, 0] += 2 * penalty_weight * viol_xmax[m_mask]
        grad[m_mask, 2] += 2 * penalty_weight * viol_xmax[m_mask]

    # Boundary y min
    viol_ymin = -y + r
    m_mask = viol_ymin > 0
    if np.any(m_mask):
        loss += penalty_weight * np.sum(viol_ymin[m_mask] ** 2)
        grad[m_mask, 1] -= 2 * penalty_weight * viol_ymin[m_mask]
        grad[m_mask, 2] += 2 * penalty_weight * viol_ymin[m_mask]

    # Boundary y max
    viol_ymax = y + r - 1.0
    m_mask = viol_ymax > 0
    if np.any(m_mask):
        loss += penalty_weight * np.sum(viol_ymax[m_mask] ** 2)
        grad[m_mask, 1] += 2 * penalty_weight * viol_ymax[m_mask]
        grad[m_mask, 2] += 2 * penalty_weight * viol_ymax[m_mask]

    # Bubble pairwise collision penalty overlaps natively computed safely
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx ** 2 + dy ** 2
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    safe_dist = np.maximum(dist, 1e-12)
    
    viol = r[:, None] + r[None, :] - dist
    np.fill_diagonal(viol, -1.0)
    
    m_mask = viol > 0
    if np.any(m_mask):
        loss += penalty_weight * np.sum(viol[m_mask] ** 2) / 2.0
        term = 2.0 * penalty_weight * viol * m_mask  
        grad[:, 2] += np.sum(term, axis=1)
        
        dviol_dx = np.where(m_mask, -dx / safe_dist, 0.0)
        dviol_dy = np.where(m_mask, -dy / safe_dist, 0.0)
        grad[:, 0] += np.sum(term * dviol_dx, axis=1)
        grad[:, 1] += np.sum(term * dviol_dy, axis=1)
        
    # Prevent parameters shifting heavily outside strict size parameters negative limits
    viol_neg = -r
    m_mask = viol_neg > 0
    if np.any(m_mask):
        loss += penalty_weight * np.sum(viol_neg[m_mask] ** 2)
        grad[m_mask, 2] -= 2 * penalty_weight * viol_neg[m_mask]
        
    return loss, grad


def get_initial_Z(n=26, mode=0):
    """Yield geometrically heuristical setups per pass heavily breaking initial symmetry loops"""
    Z = np.zeros((n, 3))
    
    if mode == 0:
        # Central large setup allowing diverse smaller interpolating gaps natively later
        Z[0] = [0.5, 0.5, 0.35]
        Z[1:5, 0] = [0.15, 0.85, 0.15, 0.85]
        Z[1:5, 1] = [0.15, 0.15, 0.85, 0.85]
        Z[1:5, 2] = 0.15
        for i in range(5, n):
            Z[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05]
    elif mode == 1:
        # Corners specifically locked heuristical configurations mapping properly mapping gaps 
        Z[0:4, 0] = [0.25, 0.75, 0.25, 0.75]
        Z[0:4, 1] = [0.25, 0.25, 0.75, 0.75]
        Z[0:4, 2] = 0.25
        for i in range(4, n):
            Z[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.03]
    elif mode == 2:
        # Heavily tight geometrical spacing arrays
        idx = 0
        for x in np.linspace(0.1, 0.9, 5):
            for y in np.linspace(0.1, 0.9, 5):
                if idx < n:
                    Z[idx] = [x, y, 0.08]
                    idx += 1
        Z[25] = [0.5, 0.5, 0.05]
    else:
        # Generic space search
        Z[:, 0] = np.random.uniform(0.1, 0.9, n)
        Z[:, 1] = np.random.uniform(0.1, 0.9, n)
        Z[:, 2] = np.random.uniform(0.02, 0.1, n)

    # Adding uniform mathematical distribution perturbations
    Z[:, 0] += np.random.normal(0, 0.005, n)
    Z[:, 1] += np.random.normal(0, 0.005, n)
    Z[:, 2] += np.abs(np.random.normal(0, 0.005, n))
    
    Z[:, :2] = np.clip(Z[:, :2], 0.01, 0.99)
    return Z


def construct_packing():
    """
    Construct heuristically iteratively refined configurations driving tightly mapping overlaps seamlessly maximizing.
    Uses dynamic geometry simulation via heavily damped simulated annealing continuous expansions algorithms mapping exact bounds.
    """
    np.random.seed(42)
    n = 26
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    iterations = 2500
    n_restarts = 20
    
    # Executing parallelized physical exploration starts guaranteeing convergence diversity mapped across random configurations 
    for restart in range(n_restarts):
        mode = restart % 4
        Z = get_initial_Z(n, mode)
        
        m_moment = np.zeros_like(Z)
        v_moment = np.zeros_like(Z)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        lr_start = 0.01
        lr_end = 0.0001
        penalty = 5.0
        
        for t in range(1, iterations + 1):
            loss, grad = get_loss_and_grad(Z, penalty)
            
            m_moment = beta1 * m_moment + (1 - beta1) * grad
            v_moment = beta2 * v_moment + (1 - beta2) * (grad ** 2)
            m_hat = m_moment / (1 - beta1 ** t)
            v_hat = v_moment / (1 - beta2 ** t)
            
            # Using progressively stable updates mappings avoiding oscillations tightly squeezing final configurations
            current_lr = lr_start * (1 - t / iterations) + lr_end
            Z -= current_lr * m_hat / (np.sqrt(v_hat) + eps)
            
            np.clip(Z[:, 0], 0.0001, 0.9999, out=Z[:, 0])
            np.clip(Z[:, 1], 0.0001, 0.9999, out=Z[:, 1])
            np.clip(Z[:, 2], 0.0000, 0.5000, out=Z[:, 2])
            
            if t % 300 == 0:
                penalty *= 2.0
                
        # We capture tightly clustered mathematically locked outputs via exact exact-constraint derivations cleanly
        centers = np.clip(Z[:, :2], 0.001, 0.999)
        radii = compute_max_radii(centers)
        
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
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
    visualize(centers, radii)