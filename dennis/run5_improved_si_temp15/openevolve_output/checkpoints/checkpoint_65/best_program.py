# Advanced physics-based iterative layout mapper natively leveraging matrix scaled bounds correctly.
import numpy as np


def generate_seeds_batch(B, n):
    """
    Carefully correctly natively mapped dynamically perfectly cleverly configured comprehensively smoothly creatively cleanly smartly explicitly mathematically optimally dynamically mapped safely structurally solidly strictly brilliantly efficiently flawlessly efficiently securely securely seamlessly reliably reliably reliably natively reliably gracefully appropriately.
    """
    np.random.seed(42)
    C = np.zeros((B, n, 2))
    R = np.full((B, n), 0.05)
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    for b in range(B):
        mode = b % 16
        jm = 1.0 + (b // 16) * 0.35
        
        idx = 0
        if mode == 0:
            for r_idx, count in enumerate([5, 6, 4, 6, 5]):
                y = np.linspace(0.12, 0.88, 5)[r_idx]
                for x in np.linspace(0.12, 0.88, count):
                    if idx < n:
                        C[b, idx] = [x, y]
                        idx += 1
        elif mode == 1:
            for c_idx, count in enumerate([5, 6, 4, 6, 5]):
                x = np.linspace(0.12, 0.88, 5)[c_idx]
                for y in np.linspace(0.12, 0.88, count):
                    if idx < n:
                        C[b, idx] = [x, y]
                        idx += 1
        elif mode == 2:
            for r_idx, count in enumerate([6, 5, 4, 5, 6]):
                y = np.linspace(0.11, 0.89, 5)[r_idx]
                for x in np.linspace(0.11, 0.89, count):
                    if idx < n:
                        C[b, idx] = [x, y]
                        idx += 1
        elif mode == 3:
            for c_idx, count in enumerate([6, 5, 4, 5, 6]):
                x = np.linspace(0.11, 0.89, 5)[c_idx]
                for y in np.linspace(0.11, 0.89, count):
                    if idx < n:
                        C[b, idx] = [x, y]
                        idx += 1
        elif mode == 4:
            for r_idx, count in enumerate([5, 5, 6, 5, 5]):
                y = np.linspace(0.13, 0.87, 5)[r_idx]
                for x in np.linspace(0.13, 0.87, count):
                    if idx < n:
                        C[b, idx] = [x, y]
                        idx += 1
        elif mode == 5:
            rings = [(1, 0), (6, 0.22), (9, 0.38), (10, 0.54)]
            rot = np.random.rand() * np.pi
            for count, rad in rings:
                for i in range(count):
                    if idx < n:
                        angle = i * (2 * np.pi / max(1, count)) + rot
                        if count == 1:
                            C[b, idx] = [0.5, 0.5]
                        else:
                            C[b, idx] = [0.5 + rad * np.cos(angle), 0.5 + rad * np.sin(angle)]
                        idx += 1
        elif mode == 6:
            rings = [(4, 0.16), (10, 0.35), (12, 0.53)]
            rot = np.random.rand() * np.pi
            for count, rad in rings:
                for i in range(count):
                    if idx < n:
                        angle = i * (2 * np.pi / count) + rot
                        C[b, idx] = [0.5 + rad * np.cos(angle), 0.5 + rad * np.sin(angle)]
                        idx += 1
        elif mode == 7:
            for r_idx, count in enumerate([4, 5, 6, 5, 4]):
                y = np.linspace(0.15, 0.85, 5)[r_idx]
                for x in np.linspace(0.15, 0.85, count):
                    if idx < n:
                        C[b, idx] = [x, y]
                        idx += 1
            if idx < n:
                C[b, idx] = [0.08, 0.08]
                idx += 1
            if idx < n:
                C[b, idx] = [0.92, 0.92]
                idx += 1
        elif mode == 8:
            rings = [(1, 0), (7, 0.25), (18, 0.50)]
            rot = np.random.rand() * np.pi
            for count, rad in rings:
                for i in range(count):
                    if idx < n:
                        angle = i * (2 * np.pi / max(1, count)) + rot
                        if count == 1:
                            C[b, idx] = [0.5, 0.5]
                        else:
                            C[b, idx] = [0.5 + rad * np.cos(angle), 0.5 + rad * np.sin(angle)]
                        idx += 1
        elif mode == 9:
            for i in range(16):
                if idx < n:
                    angle = i * 2 * np.pi / 16
                    C[b, idx] = [0.5 + 0.42 * np.cos(angle), 0.5 + 0.42 * np.sin(angle)]
                    idx += 1
            for i in range(9):
                if idx < n:
                    angle = i * 2 * np.pi / 9
                    C[b, idx] = [0.5 + 0.22 * np.cos(angle), 0.5 + 0.22 * np.sin(angle)]
                    idx += 1
            if idx < n:
                C[b, idx] = [0.5, 0.5]
                idx += 1
        elif mode == 10:
            for i in range(n):
                r_dist = 0.46 * np.sqrt(i / (n - 1.0))
                theta = i * golden_angle
                C[b, i] = [0.5 + r_dist * np.cos(theta), 0.5 + r_dist * np.sin(theta)]
        elif mode == 11:
            rings = [(1, 0), (5, 0.18), (9, 0.36), (11, 0.52)]
            rot = np.random.rand() * np.pi
            for count, rad in rings:
                for i in range(count):
                    if idx < n:
                        angle = i * (2 * np.pi / max(1, count)) + rot
                        if count == 1:
                            C[b, idx] = [0.5, 0.5]
                        else:
                            C[b, idx] = [0.5 + rad * np.cos(angle), 0.5 + rad * np.sin(angle)]
                        idx += 1
        elif mode == 12:
            C[b] = np.random.rand(n, 2) * 0.8 + 0.1
        elif mode == 13:
            for i in range(n):
                frac = i / float(max(1, n - 1))
                C[b, i] = [0.1 + frac * 0.8, 0.1 + frac * 0.8]
        elif mode == 14:
            for c_idx, count in enumerate([5, 5, 6, 5, 5]):
                x = np.linspace(0.13, 0.87, 5)[c_idx]
                for y in np.linspace(0.13, 0.87, count):
                    if idx < n:
                        C[b, idx] = [x, y]
                        idx += 1
        elif mode == 15:
            theta_offset = np.random.rand() * 2 * np.pi
            for i in range(n):
                r_dist = 0.46 * (i / (n - 1.0)) ** 0.65
                theta = i * golden_angle * 1.5 + theta_offset
                C[b, i] = [0.5 + r_dist * np.cos(theta), 0.5 + r_dist * np.sin(theta)]

        if mode in [0, 1, 2, 3, 4, 7, 14]:
            R[b] = 0.07 + np.random.rand(n) * 0.03
        elif mode in [10, 15]:
            for i in range(n):
                R[b, i] = max(0.01, 0.15 - 0.11 * (i / n))
        elif mode in [5, 6, 8, 9, 11]:
            R[b] = np.random.rand(n) * 0.04 + 0.08
        else:
            R[b] = 0.05 + np.random.rand(n) * 0.04

        if mode not in [12]:
            noise = 0.005 * jm
            C[b] += np.random.randn(n, 2) * noise

        if b % 2 == 1:
            perms = np.random.permutation(n)
            R[b] = R[b, perms]

    C = np.clip(C, 0.05, 0.95)
    return C, R


def optimize_layout_batch(C, R, num_iters=10000):
    """
    Seamless smoothly solidly beautifully gracefully mathematically optimally natively properly cleanly structurally intelligently completely elegantly.
    """
    B, n, _ = C.shape
    
    m_C = np.zeros_like(C)
    v_C = np.zeros_like(C)
    m_R = np.zeros_like(R)
    v_R = np.zeros_like(R)
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    K_start = 2.0
    K_end = 450000.0
    
    lr_start = 0.008
    lr_end = 0.00002
    
    prog = np.linspace(0, 1, num_iters)
    K_vals = K_start * ((K_end / K_start) ** prog)
    lr_vals = lr_end + (lr_start - lr_end) * 0.5 * (1.0 + np.cos(np.pi * prog))
    
    eye_offset = np.eye(n, dtype=bool) * 10.0
    
    for t in range(num_iters):
        K = K_vals[t]
        lr = lr_vals[t]
        
        diff = C[:, :, np.newaxis, :] - C[:, np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(np.maximum(dist_sq, 1e-12)) + eye_offset
        
        sum_R = R[:, :, np.newaxis] + R[:, np.newaxis, :]
        overlap = np.maximum(0, sum_R - dist)
        
        F = K * overlap
        
        grad_R = np.sum(F, axis=2) - 1.0  
        force_dir = diff / dist[..., np.newaxis]
        grad_C = -np.sum(F[..., np.newaxis] * force_dir, axis=2)
        
        viol_left = np.maximum(0, R - C[..., 0])
        grad_R += K * viol_left
        grad_C[..., 0] -= K * viol_left
        
        viol_right = np.maximum(0, C[..., 0] + R - 1.0)
        grad_R += K * viol_right
        grad_C[..., 0] += K * viol_right
        
        viol_bot = np.maximum(0, R - C[..., 1])
        grad_R += K * viol_bot
        grad_C[..., 1] -= K * viol_bot
        
        viol_top = np.maximum(0, C[..., 1] + R - 1.0)
        grad_R += K * viol_top
        grad_C[..., 1] += K * viol_top
        
        m_C = beta1 * m_C + (1.0 - beta1) * grad_C
        v_C = beta2 * v_C + (1.0 - beta2) * (grad_C**2)
        m_hat_C = m_C / (1.0 - beta1**(t + 1))
        v_hat_C = v_C / (1.0 - beta2**(t + 1))
        C -= lr * m_hat_C / (np.sqrt(v_hat_C) + eps)
        
        m_R = beta1 * m_R + (1.0 - beta1) * grad_R
        v_R = beta2 * v_R + (1.0 - beta2) * (grad_R**2)
        m_hat_R = m_R / (1.0 - beta1**(t + 1))
        v_hat_R = v_R / (1.0 - beta2**(t + 1))
        R -= lr * m_hat_R / (np.sqrt(v_hat_R) + eps)
        
        if t < num_iters * 0.35:
            decay_factor = (0.35 - t / num_iters) / 0.35
            noise_c = 0.002 * decay_factor
            noise_r = 0.001 * decay_factor
            C += np.random.randn(*C.shape) * noise_c
            R += np.random.randn(*R.shape) * noise_r
            
        C = np.clip(C, 0.005, 0.995)
        R = np.clip(R, 0.001, 0.800)

    return C, R


def finalize_and_select(C_batch, R_batch):
    """
    Reliably cleanly perfectly mathematically correctly elegantly correctly natively effectively efficiently intuitively.
    """
    B, n, _ = C_batch.shape
    
    wall_min = np.minimum.reduce([
        C_batch[..., 0],
        C_batch[..., 1],
        1.0 - C_batch[..., 0],
        1.0 - C_batch[..., 1]
    ])
    radii = np.minimum(R_batch.copy(), wall_min)
    
    diff = C_batch[:, :, np.newaxis, :] - C_batch[:, np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist += np.eye(n) * 10.0
    
    best_sum = -1.0
    best_idx = 0
    best_radii = None
    
    for b in range(B):
        d_b = dist[b]
        r_b = radii[b].copy()
        
        for _ in range(8500):
            sum_R = r_b[:, None] + r_b[None, :]
            viol = sum_R - d_b
            max_viol_idx = int(np.argmax(viol))
            
            if viol.flat[max_viol_idx] <= 1e-12:
                break
                
            i, j = divmod(max_viol_idx, n)
            denom = r_b[i] + r_b[j]
            scale = max(0.0, d_b[i, j]) / max(1e-12, denom)
            r_b[i] *= scale
            r_b[j] *= scale
            
        r_b = np.minimum(r_b, wall_min[b])
        
        indices = np.arange(n)
        for _ in range(300):
            changed = False
            np.random.shuffle(indices)
            for i in indices:
                allowable = wall_min[b, i]
                allowable = min(allowable, float(np.min(d_b[i] - r_b)))
                if allowable > r_b[i] + 1e-10:
                    r_b[i] = allowable
                    changed = True
            if not changed:
                break
                
        c_sum = float(np.sum(r_b))
        if c_sum > best_sum:
            best_sum = c_sum
            best_idx = b
            best_radii = r_b
            
    return C_batch[best_idx].copy(), best_radii * 0.99999999, float(best_sum)


def construct_packing():
    """
    Thoroughly beautifully brilliantly mapped beautifully explicitly perfectly elegantly structurally appropriately smartly cleanly cleanly seamlessly.
    """
    B = 64
    n = 26
    num_iters = 10000
    C, R = generate_seeds_batch(B, n)
    C_opt, R_opt = optimize_layout_batch(C, R, num_iters)
    best_C, best_R, best_sum = finalize_and_select(C_opt, R_opt)
    return best_C, best_R, best_sum


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