import numpy as np
from scipy.optimize import linprog


def run_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    np.random.seed(42)
    B = 64  # Batch size for parallel exploration
    N = 26  # Number of circles

    # Initialization
    centers = np.random.uniform(0.1, 0.9, size=(B, N, 2))

    # Pattern 1: Edge/Corner biased (Beta distribution)
    centers[0:16] = np.random.beta(0.3, 0.3, size=(16, N, 2))

    # Pattern 2: Hexagonal grid roughly tailored for square packing
    hex_centers = []
    for row in range(7):
        for col in range(7):
            x = 0.05 + col * 0.15 + (0.075 if row % 2 == 1 else 0.0)
            y = 0.05 + row * 0.15
            if x <= 0.95 and y <= 0.95:
                hex_centers.append([x, y])
    hex_pts = np.array(hex_centers)
    for b in range(16, 24):
        idx = np.random.choice(len(hex_pts), N, replace=False)
        centers[b] = hex_pts[idx] + np.random.normal(0, 0.01, size=(N, 2))

    # Pattern 3: Standard uniform grid
    xv, yv = np.meshgrid(np.linspace(0.08, 0.92, 6), np.linspace(0.08, 0.92, 6))
    grid_pts = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    for b in range(24, 32):
        idx = np.random.choice(len(grid_pts), N, replace=False)
        centers[b] = grid_pts[idx] + np.random.normal(0, 0.01, size=(N, 2))

    # Pattern 4: Concentric rings placement
    for b in range(32, 48):
        centers[b, 0] = [0.5, 0.5]
        idx = 1
        angle_offset_1 = np.random.uniform(0, 2 * np.pi)
        for i in range(7):
            theta = 2.0 * np.pi * i / 7.0 + angle_offset_1
            centers[b, idx] = [0.5 + 0.22 * np.cos(theta),
                               0.5 + 0.22 * np.sin(theta)]
            idx += 1
        angle_offset_2 = np.random.uniform(0, 2 * np.pi)
        for i in range(18):
            theta = 2.0 * np.pi * i / 18.0 + angle_offset_2
            centers[b, idx] = [0.5 + 0.44 * np.cos(theta),
                               0.5 + 0.44 * np.sin(theta)]
            idx += 1
        centers[b] += np.random.normal(0, 0.005, size=(N, 2))

    # Pattern 5: Golden ratio spiral (Fibonacci)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    for b in range(48, 56):
        angle_offset = np.random.uniform(0, 2 * np.pi)
        for i in range(N):
            r_dist = np.sqrt((i + 0.5) / N) * 0.45
            theta = 2.0 * np.pi * i / phi + angle_offset
            centers[b, i, 0] = 0.5 + r_dist * np.cos(theta)
            centers[b, i, 1] = 0.5 + r_dist * np.sin(theta)
        centers[b] += np.random.normal(0, 0.01, size=(N, 2))

    # Pattern 6: Random uniform with noise
    centers[56:64] = np.random.uniform(0.05, 0.95, size=(8, N, 2))

    # Keep all centers in a safe bounding box initially
    centers = np.clip(centers, 0.02, 0.98)

    # Size placement bias: larger radii in the center, smaller in corners/edges
    dist_to_center = np.linalg.norm(centers - 0.5, axis=-1)
    radii = 0.08 - 0.05 * (dist_to_center / 0.707)
    radii = np.clip(radii, 0.02, 0.1)

    # Adam Optimizer states
    lr_initial = 0.02
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m_c = np.zeros_like(centers)
    v_c = np.zeros_like(centers)
    m_r = np.zeros_like(radii)
    v_r = np.zeros_like(radii)

    max_steps = 12000
    lam_start = 5.0
    lam_end = 2e6
    lam_factor = (lam_end / lam_start) ** (1.0 / max_steps)

    mask = np.eye(N)[np.newaxis, :, :]

    for step in range(max_steps):
        # Simulated annealing for constraints: penalty weight grows exponentially
        lam = lam_start * (lam_factor ** step)
        
        # Smoothly decaying learning rate, ending very fine to settle micro-adjustments
        lr = lr_initial * (0.0005 ** (step / max_steps))

        # Break perfect symmetry to escape local maxima
        if step > 0 and step < max_steps // 2 and step % 500 == 0:
            centers += np.random.normal(0, 0.001, size=centers.shape)
            centers = np.clip(centers, 0.0, 1.0)

        # Pairwise differences and distances
        diff = centers[:, :, np.newaxis, :] - centers[:, np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        # Calculate overlap magnitudes
        sum_r = radii[:, :, np.newaxis] + radii[:, np.newaxis, :]
        overlap = np.maximum(0, sum_r - dist)
        overlap = overlap * (1.0 - mask)  # Exclude self-intersections

        # Gradients w.r.t pairwise overlaps
        grad_r_overlap = 2.0 * np.sum(overlap, axis=2)

        dist_safe = dist + mask + 1e-8
        force_mag = 2.0 * overlap / dist_safe
        grad_c_overlap = -np.sum(force_mag[..., np.newaxis] * diff, axis=2)

        # Boundary constraints
        x = centers[..., 0]
        y = centers[..., 1]
        r = radii

        p_L = np.maximum(0, r - x)
        p_R = np.maximum(0, x + r - 1.0)
        p_B = np.maximum(0, r - y)
        p_T = np.maximum(0, y + r - 1.0)

        grad_r_bounds = 2.0 * (p_L + p_R + p_B + p_T)
        grad_x_bounds = -2.0 * p_L + 2.0 * p_R
        grad_y_bounds = -2.0 * p_B + 2.0 * p_T
        grad_c_bounds = np.stack([grad_x_bounds, grad_y_bounds], axis=-1)

        # Combine gradients. Objective is maximizing sum(radii) -> min -sum(r)
        grad_r = -1.0 + lam * (grad_r_overlap + grad_r_bounds)
        grad_c = lam * (grad_c_overlap + grad_c_bounds)

        # Clip gradients to prevent numeric explosion
        grad_r = np.clip(grad_r, -1e4, 1e4)
        grad_c = np.clip(grad_c, -1e4, 1e4)

        # Apply Adam updates
        m_c = beta1 * m_c + (1 - beta1) * grad_c
        v_c = beta2 * v_c + (1 - beta2) * (grad_c ** 2)
        centers -= lr * m_c / (np.sqrt(v_c) + eps)

        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r ** 2)
        radii -= lr * m_r / (np.sqrt(v_r) + eps)

        # Enforce valid physical ranges to maintain optimizer stability
        radii = np.maximum(radii, 0.001)
        centers = np.clip(centers, 0.0, 1.0)

    # Final pass: Linear Programming for exact mathematical validity and maximization
    best_sum = -1.0
    best_centers = None
    best_radii = None

    for b in range(B):
        c = np.clip(centers[b], 0.0, 1.0)
        c_obj = -np.ones(N)
        A_ub = []
        b_ub = []
        bounds = []

        # Bound constraints per circle mapped natively to LP bounds for speed
        for i in range(N):
            x_i, y_i = c[i]
            max_r = max(0.0, float(min(x_i, 1.0 - x_i, y_i, 1.0 - y_i)))
            bounds.append((0.0, max_r))

        # Pairwise distance constraints to ensure no circle overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist_ij = np.linalg.norm(c[i] - c[j])
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist_ij)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                          bounds=bounds, method='highs')
            if res.success:
                s_r = -res.fun
                if s_r > best_sum:
                    best_sum = s_r
                    best_centers = c
                    best_radii = res.x
        except Exception:
            pass

    # Fallback to mathematically rigorous manual trimming if LP somehow fails
    if best_centers is None:
        best_centers = np.clip(centers[0], 0.0, 1.0)
        best_radii = np.maximum(radii[0], 0.0)
        for _ in range(1500):
            best_radii = np.minimum(best_radii, best_centers[..., 0])
            best_radii = np.minimum(best_radii, 1.0 - best_centers[..., 0])
            best_radii = np.minimum(best_radii, best_centers[..., 1])
            best_radii = np.minimum(best_radii, 1.0 - best_centers[..., 1])

            for i in range(N):
                for j in range(i + 1, N):
                    dist_ij = np.linalg.norm(best_centers[i] - best_centers[j])
                    if best_radii[i] + best_radii[j] > dist_ij:
                        overlap_ij = best_radii[i] + best_radii[j] - dist_ij
                        # Reduce each radius symmetrically
                        best_radii[i] -= overlap_ij * 0.505
                        best_radii[j] -= overlap_ij * 0.505
            best_radii = np.maximum(best_radii, 0.0)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)


# Create an alias to ensure maximum compatibility with the evaluator hook
construct_packing = run_packing