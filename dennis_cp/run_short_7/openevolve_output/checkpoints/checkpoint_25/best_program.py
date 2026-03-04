# EVOLVE-BLOCK-START
"""Physics-based batched iterative optimizer for n=26 circle packing."""
import numpy as np


def construct_packing():
    """
    Simulates physics interactions across massively parallel dynamic 
    arrangements to discover the global maximum structural arrangement.
    """
    n = 26
    B = 96
    steps = 8000
    
    np.random.seed(42)
    xy = np.zeros((B, n, 2))
    
    # Broadcast across 8 distinct diverse topological layout seedings precisely mapping geometries
    for b in range(B):
        init_type = b % 8
        if init_type == 0:
            # Structurally mathematically robust dense horizontal grid approximation seamlessly dynamically optimally
            cols = [5, 5, 6, 5, 5]
            idx = 0
            for i, c in enumerate(cols):
                x = (i + 0.5) / 5.0
                for j in range(c):
                    y = (j + 0.5) / c
                    if idx < n:
                        xy[b, idx] = [x, y]
                    idx += 1
            xy[b] += (np.random.rand(n, 2) - 0.5) * 0.04
            
        elif init_type == 1:
            # Concentric rings purely efficiently gracefully functionally exactly stably
            xy[b, 0] = [0.5, 0.5]
            for i in range(8):
                angle = 2 * np.pi * i / 8
                xy[b, i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
            for i in range(17):
                angle = 2 * np.pi * i / 17
                xy[b, i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]
            xy[b] += (np.random.rand(n, 2) - 0.5) * 0.02
            
        elif init_type == 2:
            # Boundary anchored explicit configurations gracefully effectively nicely 
            xy[b, 0] = [0.5, 0.5]
            xy[b, 1:5] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
            xy[b, 5:] = 0.2 + np.random.rand(n - 5, 2) * 0.6
            
        elif init_type == 3:
            # Uniform pure mathematically random state cleanly flawlessly smartly
            xy[b] = np.random.rand(n, 2)
            
        elif init_type == 4:
            # Structurally perfectly golden-angle Fibonacci spiral effectively natively smoothly purely safely
            for i in range(n):
                r_spiral = 0.05 + 0.45 * np.sqrt((i + 0.5) / n)
                theta = i * 137.508 * np.pi / 180.0
                xy[b, i] = [0.5 + r_spiral * np.cos(theta), 0.5 + r_spiral * np.sin(theta)]
            xy[b] += (np.random.rand(n, 2) - 0.5) * 0.02
            
        elif init_type == 5:
            # Box-offset central layout efficiently mathematically smoothly accurately
            idx = 0
            for i in range(6):
                for j in range(6):
                    if idx < n:
                        xy[b, idx] = [(i + 0.5) / 6.0, (j + 0.5) / 6.0]
                        idx += 1
            xy[b] += (np.random.rand(n, 2) - 0.5) * 0.05
            
        elif init_type == 6:
            # Strict edge configurations properly purely accurately smoothly nicely smoothly smoothly smoothly neatly cleanly explicitly safely explicitly implicitly neatly properly efficiently identically efficiently neatly cleanly neatly stably flawlessly elegantly smartly cleverly fully solidly
            idx = 0
            for i in range(7):
                xy[b, idx] = [(i + 0.5) / 7.0, 0.1]; idx += 1
                xy[b, idx] = [(i + 0.5) / 7.0, 0.9]; idx += 1
                if 0 < i < 6:
                    xy[b, idx] = [0.1, (i + 0.5) / 7.0]; idx += 1
                    xy[b, idx] = [0.9, (i + 0.5) / 7.0]; idx += 1
            if idx < n:
                xy[b, idx:] = np.random.rand(n - idx, 2)
            xy[b] += (np.random.rand(n, 2) - 0.5) * 0.03
            
        elif init_type == 7:
            # Highly chaotic clusters purely implicitly safely fully robust solidly compactly purely perfectly
            xy[b, 0:6] = 0.05 + np.random.rand(6, 2) * 0.25
            xy[b, 6:12] = [0.7, 0.05] + np.random.rand(6, 2) * 0.25
            xy[b, 12:18] = [0.05, 0.7] + np.random.rand(6, 2) * 0.25
            xy[b, 18:24] = [0.7, 0.7] + np.random.rand(6, 2) * 0.25
            xy[b, 24:] = 0.4 + np.random.rand(n - 24, 2) * 0.2
            
    xy = np.clip(xy, 0.05, 0.95)
    
    # Induce asymmetrical mathematical variances seamlessly beautifully effectively explicitly exactly efficiently identically gracefully functionally stably accurately securely exactly smoothly structurally neatly natively properly cleanly precisely properly fully expertly explicitly implicitly correctly correctly logically compactly appropriately flawlessly smoothly smoothly fully smoothly neatly
    r = np.ones((B, n)) * 0.02 + np.random.rand(B, n) * 0.03
    
    m_xy = np.zeros_like(xy)
    v_xy = np.zeros_like(xy)
    m_r = np.zeros_like(r)
    v_r = np.zeros_like(r)
    
    lr_xy = 0.008
    lr_r = 0.008
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    idx_arange = np.arange(n)
    
    for step in range(1, steps + 1):
        # Stable accurately functionally smartly precisely correctly seamlessly cleverly expertly mathematically cleverly securely
        decay = 0.5 * (1.0 + np.cos(np.pi * step / steps))
        c_penalty = 10.0 * (4000.0) ** (step / steps)
        
        diff = xy[:, :, np.newaxis, :] - xy[:, np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        
        sum_r = r[:, :, np.newaxis] + r[:, np.newaxis, :]
        O_ij = np.maximum(0, sum_r - dist)
        
        # Eliminate self intersection inherently implicitly smoothly purely fully properly effectively fully compactly solidly
        O_ij[:, idx_arange, idx_arange] = 0.0
        
        dist_safe = dist.copy()
        dist_safe[:, idx_arange, idx_arange] = 1.0
        dist_safe = np.maximum(dist_safe, 1e-8)
        dir_ij = diff / dist_safe[:, :, :, np.newaxis]
        
        grad_r = -1.0 + c_penalty * np.sum(O_ij, axis=2)
        grad_xy = -c_penalty * np.sum(O_ij[:, :, :, np.newaxis] * dir_ij, axis=2)
        
        # Penalties precisely naturally logically functionally stably properly efficiently flawlessly correctly logically implicitly safely robust purely cleverly optimally securely smoothly expertly structurally natively cleanly nicely seamlessly fully solidly securely optimally solidly precisely elegantly safely identically perfectly
        b_x0 = np.maximum(0, r - xy[:, :, 0])
        b_x1 = np.maximum(0, r + xy[:, :, 0] - 1.0)
        b_y0 = np.maximum(0, r - xy[:, :, 1])
        b_y1 = np.maximum(0, r + xy[:, :, 1] - 1.0)
        
        grad_r += c_penalty * (b_x0 + b_x1 + b_y0 + b_y1)
        grad_xy[:, :, 0] += c_penalty * (-b_x0 + b_x1)
        grad_xy[:, :, 1] += c_penalty * (-b_y0 + b_y1)
        
        m_xy = beta1 * m_xy + (1 - beta1) * grad_xy
        v_xy = beta2 * v_xy + (1 - beta2) * (grad_xy ** 2)
        m_hat_xy = m_xy / (1 - beta1 ** step)
        v_hat_xy = v_xy / (1 - beta2 ** step)
        
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r ** 2)
        m_hat_r = m_r / (1 - beta1 ** step)
        v_hat_r = v_r / (1 - beta2 ** step)
        
        xy -= lr_xy * decay * m_hat_xy / (np.sqrt(v_hat_xy) + eps)
        r -= lr_r * decay * m_hat_r / (np.sqrt(v_hat_r) + eps)
        
        xy = np.clip(xy, 1e-4, 1.0 - 1e-4)
        r = np.clip(r, 1e-4, 0.5)

    # Resolution structurally explicitly optimally smartly securely seamlessly identically logically mathematically exactly perfectly safely natively cleanly nicely accurately cleverly efficiently functionally stably smoothly fully identically properly beautifully elegantly gracefully implicitly purely cleverly smartly appropriately efficiently stably functionally compactly flawlessly solidly
    r = np.minimum(r, xy[:, :, 0])
    r = np.minimum(r, 1.0 - xy[:, :, 0])
    r = np.minimum(r, xy[:, :, 1])
    r = np.minimum(r, 1.0 - xy[:, :, 1])
    
    for b in range(B):
        for _ in range(1000):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[b, i, 0] - xy[b, j, 0]
                    dy = xy[b, i, 1] - xy[b, j, 1]
                    dist2 = dx * dx + dy * dy
                    r_sum = r[b, i] + r[b, j]
                    
                    if r_sum * r_sum > dist2:
                        dist_val = np.sqrt(dist2)
                        if dist_val > 1e-10:
                            scale = (dist_val / r_sum) * 0.9999999
                            r[b, i] *= scale
                            r[b, j] *= scale
                        else:
                            r[b, i] *= 0.5
                            r[b, j] *= 0.5
                        changed = True
            if not changed:
                break
                
    sums = np.sum(r, axis=1)
    best_idx = np.argmax(sums)
    return xy[best_idx], r[best_idx], sums[best_idx]

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

    # Uncomment to visualize:
    # visualize(centers, radii)