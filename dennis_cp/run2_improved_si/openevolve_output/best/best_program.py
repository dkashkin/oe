import numpy as np
from scipy.optimize import linprog


def construct_packing():
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
    B = 96  # Batch size for parallel exploration
    N = 26  # Number of circles

    centers = np.random.uniform(0.1, 0.9, size=(B, N, 2))

    # Pattern 1: Edge/Corner biased (Beta distribution)
    centers[0:16] = np.random.beta(0.3, 0.3, size=(16, N, 2))

    # Pattern 2: Concentric rings placement
    for b in range(16, 32):
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

    # Pattern 3: Golden ratio spiral (Fibonacci)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    for b in range(32, 48):
        angle_offset = np.random.uniform(0, 2 * np.pi)
        for i in range(N):
            r_dist = np.sqrt((i + 0.5) / N) * 0.45
            theta = 2.0 * np.pi * i / phi + angle_offset
            centers[b, i, 0] = 0.5 + r_dist * np.cos(theta)
            centers[b, i, 1] = 0.5 + r_dist * np.sin(theta)
        centers[b] += np.random.normal(0, 0.01, size=(N, 2))

    # Pattern 4: Hexagonal grid roughly tailored for square packing
    hex_centers = []
    xv, yv = np.meshgrid(np.linspace(0.05, 0.95, 7), np.linspace(0.05, 0.95, 7))
    xv[1::2] += (0.9 / 6) / 2.0  # Shift alternate rows
    hex_pts = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    hex_pts = hex_pts[(hex_pts[:, 0] <= 0.95) & (hex_pts[:, 1] <= 0.95)]
    for b in range(48, 64):
        idx = np.random.choice(len(hex_pts), N, replace=False)
        centers[b] = hex_pts[idx] + np.random.normal(0, 0.005, size=(N, 2))

    # Pattern 5: Standard uniform grid
    xv, yv = np.meshgrid(np.linspace(0.08, 0.92, 6), np.linspace(0.08, 0.92, 6))
    grid_pts = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    for b in range(64, 80):
        idx = np.random.choice(len(grid_pts), N, replace=False)
        centers[b] = grid_pts[idx] + np.random.normal(0, 0.01, size=(N, 2))

    # Pattern 6: Random uniform with noise
    centers[80:96] = np.random.uniform(0.05, 0.95, size=(16, N, 2))

    # Keep all centers in a safe bounding box initially
    np.clip(centers, 0.02, 0.98, out=centers)

    # Size placement bias: larger radii in the center, smaller in corners/edges
    dist_to_center = np.linalg.norm(centers - 0.5, axis=-1)
    radii = 0.08 - 0.05 * (dist_to_center / 0.707)
    np.clip(radii, 0.02, 0.1, out=radii)

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

    diag_indices = np.arange(N)

    for step in range(max_steps):
        # Simulated annealing for constraints: penalty weight grows exponentially
        lam = lam_start * (lam_factor ** step)
        
        # Smoothly decaying learning rate, ending very fine to settle micro-adjustments
        lr = lr_initial * (0.0005 ** (step / max_steps))

        # Break perfect symmetry to escape local maxima
        if 0 < step < max_steps // 2 and step % 500 == 0:
            centers += np.random.normal(0, 0.001, size=centers.shape)
            np.clip(centers, 0.0, 1.0, out=centers)

        # Pairwise differences and fast distance calculation
        diff = centers[:, :, np.newaxis, :] - centers[:, np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        
        # Ignore self-intersection by setting diagonal distance arbitrarily high
        dist[:, diag_indices, diag_indices] = 10.0

        # Calculate overlap magnitudes
        sum_r = radii[:, :, np.newaxis] + radii[:, np.newaxis, :]
        overlap = np.maximum(0, sum_r - dist)

        # Gradients w.r.t pairwise overlaps
        grad_r_overlap = 2.0 * np.sum(overlap, axis=2)

        force_mag = 2.0 * overlap / (dist + 1e-8)
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
        np.clip(grad_r, -1e4, 1e4, out=grad_r)
        np.clip(grad_c, -1e4, 1e4, out=grad_c)

        # Apply in-place Adam updates for computation speed
        m_c *= beta1
        m_c += (1.0 - beta1) * grad_c
        v_c *= beta2
        v_c += (1.0 - beta2) * (grad_c * grad_c)
        centers -= lr * m_c / (np.sqrt(v_c) + eps)

        m_r *= beta1
        m_r += (1.0 - beta1) * grad_r
        v_r *= beta2
        v_r += (1.0 - beta2) * (grad_r * grad_r)
        radii -= lr * m_r / (np.sqrt(v_r) + eps)

        # Enforce valid physical ranges to maintain optimizer stability
        np.maximum(radii, 0.001, out=radii)
        np.clip(centers, 0.0, 1.0, out=centers)

    # Final pass: Linear Programming for exact mathematical validity and maximization
    best_sum = -1.0
    best_centers = None
    best_radii = None

    idx_i, idx_j = np.triu_indices(N, 1)
    A_ub_base = np.zeros((len(idx_i), N))
    A_ub_base[np.arange(len(idx_i)), idx_i] = 1.0
    A_ub_base[np.arange(len(idx_i)), idx_j] = 1.0

    # Sort batches and only apply linprog to the top candidates to save compute
    batch_scores = np.sum(radii, axis=1)
    top_batches = np.argsort(batch_scores)[-20:][::-1]

    for b in top_batches:
        c = np.clip(centers[b], 0.0, 1.0)
        c_obj = -np.ones(N)
        
        bounds = []
        # Bound constraints per circle mapped natively to LP bounds for speed
        for i in range(N):
            x_i, y_i = c[i]
            max_r = max(0.0, float(min(x_i, 1.0 - x_i, y_i, 1.0 - y_i)))
            bounds.append((0.0, max_r))

        b_ub = np.linalg.norm(c[idx_i] - c[idx_j], axis=1)

        try:
            res = linprog(c_obj, A_ub=A_ub_base, b_ub=b_ub,
                          bounds=bounds, method='highs')
            if res.success:
                s_r = -res.fun
                if s_r > best_sum:
                    best_sum = s_r
                    best_centers = c.copy()
                    best_radii = res.x.copy()
        except Exception:
            pass

    # Fallback to mathematically rigorous manual trimming if LP somehow fails entirely
    if best_centers is None:
        best_idx = int(top_batches[0])
        best_centers = np.clip(centers[best_idx], 0.0, 1.0)
        best_radii = np.maximum(radii[best_idx], 0.0)
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
run_packing = construct_packing