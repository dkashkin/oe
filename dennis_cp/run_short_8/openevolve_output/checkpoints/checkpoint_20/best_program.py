# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def optimize_packing(seed):
    """
    Optimizes positions and sizes for 26 circles via simulated expansion using 
    an escalating penalty-augmented gradient descent method (Adam). 
    Strategic hex-layer heuristics maximize convergence stability towards the ideal fit.
    """
    np.random.seed(seed)
    n = 26
    
    # Initialize radii with decay sizing bias allowing diverse central placement
    r = np.random.uniform(0.04, 0.16, n)
    r[::-1].sort()
    
    pos = np.zeros((n, 2))
    
    # Construct geometric initial symmetry layout matching expected tight formations
    # [0] Center piece
    pos[0] = [0.5, 0.5]
    
    # [1-6] 6-sided inner core wrapping
    for i in range(1, 7):
        angle = 2 * np.pi * i / 6.0 + np.random.randn() * 0.1
        r_dist = 0.16 + np.random.randn() * 0.02
        pos[i] = [0.5 + r_dist * np.cos(angle), 0.5 + r_dist * np.sin(angle)]
        
    # [7-18] 12-sided outer corona ring
    for i in range(7, 19):
        angle = 2 * np.pi * (i - 7) / 12.0 + np.random.randn() * 0.1
        r_dist = 0.35 + np.random.randn() * 0.02
        pos[i] = [0.5 + r_dist * np.cos(angle), 0.5 + r_dist * np.sin(angle)]
        
    # [19-22] Exact corner filler bubbles
    corners = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
    for i in range(19, 23):
        pos[i] = corners[i - 19]
    
    # [23-25] Interstitial edge mappings biased variably across multiple seeds
    if seed % 2 == 0:
        edge_coords = [[0.5, 0.05], [0.05, 0.5], [0.95, 0.5]]
    else:
        edge_coords = [[0.33, 0.95], [0.66, 0.95], [0.5, 0.05]]
        
    for i in range(23, 26):
        pos[i] = edge_coords[i - 23]
        
    # Break rigid symmetry smoothly 
    pos += np.random.normal(0, 0.015, pos.shape)
    pos = np.clip(pos, 0.02, 0.98)
    
    # Adam parameters tracking physical momentums 
    m_pos, v_pos = np.zeros_like(pos), np.zeros_like(pos)
    m_r, v_r = np.zeros_like(r), np.zeros_like(r)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    T = 3200
    for step in range(T):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-12
        np.fill_diagonal(dist, np.inf)
        
        overlap = np.maximum(0, r[:, np.newaxis] + r[np.newaxis, :] - dist)
        
        # Determine geometric wall constraint overrides limiting space
        left_ov = np.maximum(0, r - pos[:, 0])
        right_ov = np.maximum(0, r - (1 - pos[:, 0]))
        bottom_ov = np.maximum(0, r - pos[:, 1])
        top_ov = np.maximum(0, r - (1 - pos[:, 1]))
        
        # Continuously expanding hardness factor driving strictly mutually exclusive spaces
        K = 10.0 * (10000.0) ** (step / float(T - 1.0))
        
        grad_r = -np.ones(n)
        grad_r += 2 * K * np.sum(overlap, axis=1)
        grad_r += 2 * K * (left_ov + right_ov + bottom_ov + top_ov)
        
        direction = diff / dist[:, :, np.newaxis]
        grad_pos = -2 * K * np.sum(overlap[:, :, np.newaxis] * direction, axis=1)
        
        grad_pos[:, 0] -= 2 * K * left_ov
        grad_pos[:, 0] += 2 * K * right_ov
        grad_pos[:, 1] -= 2 * K * bottom_ov
        grad_pos[:, 1] += 2 * K * top_ov
        
        # Implement cosine schedule driving precise stable equilibrium settlement
        lr_pos = 1e-5 + 0.5 * (0.01 - 1e-5) * (1 + np.cos(np.pi * step / T))
        lr_r = 1e-5 + 0.5 * (0.005 - 1e-5) * (1 + np.cos(np.pi * step / T))
        
        m_pos = beta1 * m_pos + (1 - beta1) * grad_pos
        v_pos = beta2 * v_pos + (1 - beta2) * grad_pos**2
        m_hat_pos = m_pos / (1 - beta1**(step + 1))
        v_hat_pos = v_pos / (1 - beta2**(step + 1))
        pos -= lr_pos * m_hat_pos / (np.sqrt(v_hat_pos) + eps)
        
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * grad_r**2
        m_hat_r = m_r / (1 - beta1**(step + 1))
        v_hat_r = v_r / (1 - beta2**(step + 1))
        r -= lr_r * m_hat_r / (np.sqrt(v_hat_r) + eps)
        
        # Gentle random thermal energy breaks possible metastable suboptimal lattice geometries
        if step < int(T * 0.7):
            noise_scale = 0.001 * (1 - step / (T * 0.7))
            pos += np.random.normal(0, noise_scale, pos.shape)
            r += np.random.normal(0, noise_scale * 0.05, r.shape)
            
        pos = np.clip(pos, 0.001, 0.999)
        r = np.clip(r, 0.001, 0.5)
        
    return pos, r


def fix_and_expand_radii(pos, r):
    """
    Absolutely solidifies domain limits strictly verifying zero spatial overlaps globally.
    Additionally incorporates dynamic inflation passes expanding strictly permitted gaps.
    """
    r = np.copy(r)
    n = len(r)
    
    # 1. Fallback exact shrinking enforcing reliable constraints
    for _ in range(500):
        # Apply strict coordinate limiting
        max_r_x = np.minimum(pos[:, 0], 1 - pos[:, 0])
        max_r_y = np.minimum(pos[:, 1], 1 - pos[:, 1])
        r = np.minimum(r, max_r_x)
        r = np.minimum(r, max_r_y)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        overlap = r[:, np.newaxis] + r[np.newaxis, :] - dist
        if np.max(overlap) <= 1e-9:
            break
            
        i, j = np.unravel_index(np.argmax(overlap), overlap.shape)
        dist_ij = dist[i, j]
        total_r = r[i] + r[j]
        if total_r > dist_ij:
            scale = (dist_ij / total_r) - 1e-8
            r[i] *= scale
            r[j] *= scale
            
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Assured check establishing global domain validation
    for i in range(n):
        for j in range(i + 1, n):
            if r[i] + r[j] > dist[i, j]:
                scale = dist[i, j] / (r[i] + r[j]) - 1e-10
                r[i] *= scale
                r[j] *= scale
                
    for i in range(n):
        r[i] = min([r[i], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1]])

    # 2. Re-expansion safely scaling previously bounded circles to capture missing volume bits
    dr_scales = [0.0005, 0.00005]
    for dr in dr_scales:
        active = np.ones(n, dtype=bool)
        for _ in range(1000):
            if not np.any(active):
                break
                
            r[active] += dr
            
            # Bound condition drops actively scaling indices failing boundary guarantees
            wall_mask = (r > pos[:, 0]) | (r > 1 - pos[:, 0]) | \
                        (r > pos[:, 1]) | (r > 1 - pos[:, 1])
            if np.any(wall_mask & active):
                r[wall_mask & active] -= dr
                active[wall_mask & active] = False
                
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            collisions = r_sum > dist + 1e-11
            if np.any(collisions):
                rows, cols = np.where(collisions)
                for i, j in zip(rows, cols):
                    if active[i]:
                        r[i] -= dr
                        active[i] = False
                    if active[j]:
                        r[j] -= dr
                        active[j] = False
                        
    # Safety final clip guaranteeing zero rounding discrepancy overlap penalties internally
    for i in range(n):
        for j in range(i + 1, n):
            if r[i] + r[j] > dist[i, j]:
                scale = dist[i, j] / (r[i] + r[j]) - 1e-12
                r[i] *= scale
                r[j] *= scale
                
    for i in range(n):
        r[i] = min([r[i], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1]])
        
    return r


def construct_packing():
    """
    Constructs an optimized 2D layout topology maximizing radii totals
    mapping simulated Adam physics annealing combining distinct geometric restarts.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    best_sum = -1.0
    best_pos = None
    best_r = None

    # Diversify seed conditions securing globally optimized configurations avoiding isolated traps
    for seed in range(42, 52):  # Execute across 10 random variations
        pos, r = optimize_packing(seed)
        r_fixed = fix_and_expand_radii(pos, r)
        current_sum = np.sum(r_fixed)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = pos.copy()
            best_r = r_fixed.copy()

    return best_pos, best_r, best_sum
# EVOLVE-BLOCK-END

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
    # AlphaEvolve improved this to significantly higher spatial allocation efficiency.

    # Uncomment to visualize:
    # visualize(centers, radii)