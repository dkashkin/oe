# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def optimize_packing(seed):
    """
    Simulated expanding bubble optimization algorithm via Adam gradient descent.
    Places sizes strategically and uses simulated annealing noise schedule.
    """
    np.random.seed(seed)
    n = 26
    
    # Initialize radii with bias (large to small)
    r = np.random.uniform(0.05, 0.15, n)
    r[::-1].sort()  # Sort descending so early indices get larger radius initially
    
    pos = np.zeros((n, 2))
    # Strategically place largest in center
    pos[0] = [0.5 + np.random.randn() * 0.01, 0.5 + np.random.randn() * 0.01]
    
    # First inner ring
    for i in range(1, 6):
        a = 2 * np.pi * i / 5 + np.random.randn() * 0.2
        r_dist = 0.15 + np.random.randn() * 0.02
        pos[i] = [0.5 + r_dist * np.cos(a), 0.5 + r_dist * np.sin(a)]
        
    # Second outer ring
    for i in range(6, 14):
        a = 2 * np.pi * i / 8 + np.random.randn() * 0.2
        r_dist = 0.3 + np.random.randn() * 0.02
        pos[i] = [0.5 + r_dist * np.cos(a), 0.5 + r_dist * np.sin(a)]
        
    # Send small bubbles heavily to the boundaries and tight corners
    pos[14] = [0.05, 0.05]
    pos[15] = [0.95, 0.05]
    pos[16] = [0.05, 0.95]
    pos[17] = [0.95, 0.95]
    
    # Specific edge placements
    edge_coords = [
        [0.33, 0.05], [0.66, 0.05], [0.33, 0.95], [0.66, 0.95],
        [0.05, 0.33], [0.05, 0.66], [0.95, 0.33], [0.95, 0.66]
    ]
    for i in range(18, 26):
        pos[i] = edge_coords[i - 18]
        
    # Soft perturb boundaries to break any remaining perfect symmetries
    pos[14:26] += np.random.normal(0, 0.01, (12, 2))
    
    m_pos, v_pos = np.zeros_like(pos), np.zeros_like(pos)
    m_r, v_r = np.zeros_like(r), np.zeros_like(r)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    T = 3500
    for step in range(T):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-12
        np.fill_diagonal(dist, 100.0)
        
        overlap = r[:, np.newaxis] + r[np.newaxis, :] - dist
        np.fill_diagonal(overlap, 0.0)
        overlap = np.maximum(0, overlap)
        
        left_ov = np.maximum(0, r - pos[:, 0])
        right_ov = np.maximum(0, r - (1 - pos[:, 0]))
        bottom_ov = np.maximum(0, r - pos[:, 1])
        top_ov = np.maximum(0, r - (1 - pos[:, 1]))
        
        # Exponential growth of strictness from soft overlaps to strictly exclusive boundaries
        K = 10.0 * (10000.0) ** (step / (T - 1.0))
        
        # Maximize sum(r) by treating radius growth as steady baseline force pulling down cost gradient
        grad_r = -np.ones(n)
        grad_pos = np.zeros_like(pos)
        
        # Apply strictly escalating gradient forces resolving geometric intersects
        grad_r += 2 * K * np.sum(overlap, axis=1)
        direction = diff / dist[..., np.newaxis]
        grad_pos -= 2 * K * np.sum(overlap[..., np.newaxis] * direction, axis=1)
        
        grad_r += 2 * K * (left_ov + right_ov + bottom_ov + top_ov)
        grad_pos[:, 0] -= 2 * K * left_ov
        grad_pos[:, 0] += 2 * K * right_ov
        grad_pos[:, 1] -= 2 * K * bottom_ov
        grad_pos[:, 1] += 2 * K * top_ov
        
        # Cosine learning rate decay to tune smoothly to optimal dense configuration settling out oscillations
        lr_pos_step = 1e-5 + 0.5 * (0.01 - 1e-5) * (1 + np.cos(np.pi * step / T))
        lr_r_step = 1e-5 + 0.5 * (0.005 - 1e-5) * (1 + np.cos(np.pi * step / T))
        
        # Adam descent state parameter update
        m_pos = beta1 * m_pos + (1 - beta1) * grad_pos
        v_pos = beta2 * v_pos + (1 - beta2) * grad_pos**2
        m_hat_pos = m_pos / (1 - beta1**(step + 1))
        v_hat_pos = v_pos / (1 - beta2**(step + 1))
        pos -= lr_pos_step * m_hat_pos / (np.sqrt(v_hat_pos) + eps)
        
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * grad_r**2
        m_hat_r = m_r / (1 - beta1**(step + 1))
        v_hat_r = v_r / (1 - beta2**(step + 1))
        r -= lr_r_step * m_hat_r / (np.sqrt(v_hat_r) + eps)
        
        # Simulated annealing initialization injects jiggle energy safely bypassing sub-optimal geometric ruts
        if step < T - 1000:
            noise = 0.001 * (1 - step / (T - 1000.0))
            pos += np.random.normal(0, noise, pos.shape)
            r += np.random.normal(0, noise * 0.1, r.shape)
            
        pos = np.clip(pos, 0.001, 0.999)
        r = np.clip(r, 0.001, 0.5)
        
    return pos, r


def fix_radii(pos, r):
    """
    Fallback exact shrinkage solver cleanly eliminating infinitesimally remaining penalty overlaps.
    """
    r = np.copy(r)
    # Perform strict conservative corrections
    for _ in range(500):
        # Clip absolutely strictly within exterior boundaries first 
        max_r_x = np.minimum(pos[:, 0], 1 - pos[:, 0])
        max_r_y = np.minimum(pos[:, 1], 1 - pos[:, 1])
        r = np.minimum(r, max_r_x)
        r = np.minimum(r, max_r_y)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, 100.0)
        
        overlap = r[:, np.newaxis] + r[np.newaxis, :] - dist
        if np.max(overlap) <= 1e-9:
            break
            
        i, j = np.unravel_index(np.argmax(overlap), overlap.shape)
        # Apply scaled tight elimination
        dist_ij = dist[i, j]
        total_r = r[i] + r[j]
        if total_r > dist_ij:
            scale = (dist_ij / total_r) - 1e-8
            r[i] *= scale
            r[j] *= scale
            
    # Perform a fast final check confirming uniform and reliable spatial integrity
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, 100.0)
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            if r[i] + r[j] > dist[i, j]:
                scale = dist[i, j] / (r[i] + r[j]) - 1e-10
                r[i] *= scale
                r[j] *= scale
                
    for i in range(len(r)):
        r[i] = min([r[i], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1]])
        
    return r


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    to maximize the sum of their radii, utilizing a combination of physics simulation 
    with noise annealing and diverse start seed topologies for the fittest layout execution.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    best_sum = -1.0
    best_pos = None
    best_r = None

    # Evolve multiple initialization topology searches locating globally maximal fits 
    for seed in range(42, 48):
        pos, r = optimize_packing(seed)
        r_fixed = fix_radii(pos, r)
        current_sum = np.sum(r_fixed)
        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = pos.copy()
            best_r = r_fixed.copy()

    return best_pos, best_r, best_sum
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