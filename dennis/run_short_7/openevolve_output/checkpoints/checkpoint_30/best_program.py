"""Physics-based iterative optimizer for batched highly efficient circle packing."""
import numpy as np


def construct_packing():
    """
    Construct a highly optimized arrangement of 26 circles in a unit square
    using fully batched Adam gradient descent optimization to execute 
    massive parallel explorations spanning varied geometries synchronously.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    B = 104
    steps = 8000
    
    np.random.seed(912384)
    
    xy = np.zeros((B, n, 2))
    r = np.zeros((B, n))
    
    for b in range(B):
        pat = b % 8
        if pat == 0:
            cols = [5, 5, 6, 5, 5]
            idx = 0
            for i, c in enumerate(cols):
                xx = (i + 0.5) / 5.0
                y_offset = (0.5 / c) if (i % 2 == 1) else 0.0
                for j in range(c):
                    yy = (j + 0.5) / c + (y_offset * 0.3)
                    xy[b, idx] = [xx, yy]
                    idx += 1
                    
        elif pat == 1:
            xy[b, 0] = [0.5, 0.5]
            for i in range(8):
                ang = 2 * np.pi * i / 8
                xy[b, i + 1] = [0.5 + 0.25 * np.cos(ang), 0.5 + 0.25 * np.sin(ang)]
            for i in range(17):
                ang = 2 * np.pi * i / 17
                xy[b, i + 9] = [0.5 + 0.45 * np.cos(ang), 0.5 + 0.45 * np.sin(ang)]
                
        elif pat == 2:
            cols = [6, 5, 4, 5, 6]
            idx = 0
            for i, c in enumerate(cols):
                xx = (i + 0.5) / 5.0
                y_offset = (0.5 / c) if (i % 2 == 1) else 0.0
                for j in range(c):
                    yy = (j + 0.5) / c + (y_offset * 0.3)
                    xy[b, idx] = [xx, yy]
                    idx += 1
                    
        elif pat == 3:
            pts = np.linspace(0.08, 0.92, 6)
            idx = 0
            for i in range(6):
                for j in range(6):
                    if i == 0 or i == 5 or j == 0 or j == 5:
                        if idx < 20:
                            xy[b, idx] = [pts[i], pts[j]]
                            idx += 1
            for i in range(6):
                ang = 2 * np.pi * i / 6
                xy[b, 20 + i] = [0.5 + 0.2 * np.cos(ang), 0.5 + 0.2 * np.sin(ang)]
                
        elif pat == 4:
            for i in range(n):
                r_spiral = 0.05 + 0.45 * np.sqrt((i + 0.5) / n)
                theta = i * 2.399963
                xy[b, i] = [0.5 + r_spiral * np.cos(theta), 0.5 + r_spiral * np.sin(theta)]
                
        elif pat == 5:
            pts = np.linspace(0.12, 0.88, 5)
            X, Y = np.meshgrid(pts, pts)
            xy[b, :25] = np.column_stack([X.ravel(), Y.ravel()])
            xy[b, 25] = [0.5, 0.5]
            
        elif pat == 6:
            idx = 0
            pts = np.linspace(0.05, 0.95, 7)
            for i in range(7):
                for j in range(7):
                    if i == 0 or i == 6 or j == 0 or j == 6:
                        xy[b, idx] = [pts[i], pts[j]]
                        idx += 1
            xy[b, 24] = [0.45, 0.5]
            xy[b, 25] = [0.55, 0.5]
            
        else:
            xy[b, :4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
            xy[b, 4:] = np.random.uniform(0.1, 0.9, (22, 2))
            
        xy[b] += np.random.normal(0, 0.012, (n, 2))
        
        rad_dist = (b // 8) % 3
        if rad_dist == 0:
            r[b] = np.random.uniform(0.02, 0.05, n)
        elif rad_dist == 1:
            r[b] = np.random.uniform(0.01, 0.07, n)
        else:
            r[b, 0:5] = 0.06
            r[b, 5:] = 0.03

    xy = np.clip(xy, 0.05, 0.95)
    
    m_xy = np.zeros_like(xy)
    v_xy = np.zeros_like(xy)
    m_r = np.zeros_like(r)
    v_r = np.zeros_like(r)
    
    lr_xy = 0.012
    lr_r = 0.010
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    eye = np.arange(n)
    
    for step in range(1, steps + 1):
        prog = step / steps
        
        c_penalty = 10.0 * (80000.0 ** prog)
        decay = 0.5 * (1.0 + np.cos(np.pi * prog))
        
        diff = xy[:, :, np.newaxis, :] - xy[:, np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        
        sum_r = r[:, :, np.newaxis] + r[:, np.newaxis, :]
        O_ij = np.maximum(0, sum_r - dist)
        O_ij[:, eye, eye] = 0.0
        
        dist_safe = dist.copy()
        dist_safe[:, eye, eye] = 1.0
        dist_safe = np.maximum(dist_safe, 1e-8)
        dir_ij = diff / dist_safe[..., np.newaxis]
        
        grad_r = -1.0 + c_penalty * np.sum(O_ij, axis=2)
        grad_xy = -c_penalty * np.sum(O_ij[..., np.newaxis] * dir_ij, axis=2)
        
        b_x0 = np.maximum(0, r - xy[..., 0])
        b_x1 = np.maximum(0, r + xy[..., 0] - 1.0)
        b_y0 = np.maximum(0, r - xy[..., 1])
        b_y1 = np.maximum(0, r + xy[..., 1] - 1.0)
        
        grad_r += c_penalty * (b_x0 + b_x1 + b_y0 + b_y1)
        grad_xy[..., 0] += c_penalty * (-b_x0 + b_x1)
        grad_xy[..., 1] += c_penalty * (-b_y0 + b_y1)
        
        m_xy = beta1 * m_xy + (1 - beta1) * grad_xy
        v_xy = beta2 * v_xy + (1 - beta2) * (grad_xy ** 2)
        m_hat_xy = m_xy / (1 - beta1 ** step)
        v_hat_xy = v_xy / (1 - beta2 ** step)
        
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r ** 2)
        m_hat_r = m_r / (1 - beta1 ** step)
        v_hat_r = v_r / (1 - beta2 ** step)
        
        xy -= (lr_xy * decay) * m_hat_xy / (np.sqrt(v_hat_xy) + eps)
        r -= (lr_r * decay) * m_hat_r / (np.sqrt(v_hat_r) + eps)
        
        xy = np.clip(xy, 1e-4, 1.0 - 1e-4)
        r = np.clip(r, 1e-4, 0.5)

    diff = xy[:, :, np.newaxis, :] - xy[:, np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    sum_r = r[:, :, np.newaxis] + r[:, np.newaxis, :]
    overlap = np.maximum(0, sum_r - dist)
    overlap[:, eye, eye] = 0.0
    
    b_x0 = np.maximum(0, r - xy[..., 0])
    b_x1 = np.maximum(0, r + xy[..., 0] - 1.0)
    b_y0 = np.maximum(0, r - xy[..., 1])
    b_y1 = np.maximum(0, r + xy[..., 1] - 1.0)
    
    violation = np.sum(overlap, axis=(1, 2)) / 2.0 + np.sum(b_x0 + b_x1 + b_y0 + b_y1, axis=1)
    sums = np.sum(r, axis=1)
    
    score = sums - 1000.0 * violation
    
    best_xy = None
    best_r = None
    best_sum = -1.0
    
    top_indices = np.argsort(score)[-32:][::-1]
    
    for b in top_indices:
        r_c = np.copy(r[b])
        xy_c = np.copy(xy[b])
        
        r_c = np.minimum(r_c, xy_c[:, 0])
        r_c = np.minimum(r_c, 1.0 - xy_c[:, 0])
        r_c = np.minimum(r_c, xy_c[:, 1])
        r_c = np.minimum(r_c, 1.0 - xy_c[:, 1])
        
        for _ in range(5000):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    d_ij = np.linalg.norm(xy_c[i] - xy_c[j])
                    if r_c[i] + r_c[j] > d_ij:
                        scale = (d_ij / (r_c[i] + r_c[j])) * 0.999999999
                        r_c[i] *= scale
                        r_c[j] *= scale
                        changed = True
            if not changed:
                break
                
        cur_sum = np.sum(r_c)
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_r = r_c
            best_xy = xy_c

    return best_xy, best_r, best_sum


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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")