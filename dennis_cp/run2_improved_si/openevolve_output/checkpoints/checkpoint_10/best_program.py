# EVOLVE-BLOCK-START
"""Iterative optimization-based circle packing for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Construct a highly optimized arrangement of 26 circles in a unit square
    using physics-based relaxation followed by linear programming to
    strictly maximize the sum of their radii without any overlaps.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    centers = optimize_centers(n, iters=3000)
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def optimize_centers(n, iters):
    """
    Simulate a force-directed physical model to find optimal circle centers.
    """
    centers = np.zeros((n, 2))

    # 1. Strategic initialization
    # Push larger circles to center, smaller to corners
    centers[0] = [0.5, 0.5]  # Center

    centers[1] = [0.1, 0.1]  # Corners
    centers[2] = [0.1, 0.9]
    centers[3] = [0.9, 0.1]
    centers[4] = [0.9, 0.9]

    for i in range(7):       # Inner ring
        angle = 2 * np.pi * i / 7
        centers[i + 5] = [0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)]

    for i in range(14):      # Outer ring
        angle = 2 * np.pi * i / 14 + np.pi / 14
        centers[i + 12] = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]

    np.random.seed(42)
    # 2. Add slight initial random perturbations to break perfect symmetry
    centers += np.random.normal(0, 0.01, size=(n, 2))
    centers = np.clip(centers, 0.01, 0.99)

    radii = np.ones(n) * 0.02

    for step in range(iters):
        progress = step / iters

        # 3. Tune optimization parameters: decaying learning rate
        lr_pos = 0.1 * (1.0 - progress) + 0.001
        lr_rad = 0.05 * (1.0 - progress) + 0.001

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)

        grad_centers = np.zeros((n, 2))

        # 4. Size placement: Bias center circles to grow larger
        dist_to_center = np.linalg.norm(centers - 0.5, axis=1)
        grad_radii = 0.1 + 0.05 * (1.0 - 2 * dist_to_center)

        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = radii_sum - dist

        mask = overlap > 0
        i_idx, j_idx = np.where(mask)

        for u, v in zip(i_idx, j_idx):
            if u >= v:
                continue

            d = dist[u, v]
            if d > 1e-6:
                dir_vec = diff[u, v] / d
            else:
                dir_vec = np.random.randn(2)
                dir_vec /= np.linalg.norm(dir_vec)

            force = overlap[u, v] * 2.0

            grad_centers[u] += dir_vec * force
            grad_centers[v] -= dir_vec * force
            grad_radii[u] -= force
            grad_radii[v] -= force

        overlap_x0 = radii - centers[:, 0]
        overlap_x1 = radii - (1.0 - centers[:, 0])
        overlap_y0 = radii - centers[:, 1]
        overlap_y1 = radii - (1.0 - centers[:, 1])

        for u in range(n):
            if overlap_x0[u] > 0:
                grad_centers[u, 0] += overlap_x0[u] * 2.0
                grad_radii[u] -= overlap_x0[u]
            if overlap_x1[u] > 0:
                grad_centers[u, 0] -= overlap_x1[u] * 2.0
                grad_radii[u] -= overlap_x1[u]
            if overlap_y0[u] > 0:
                grad_centers[u, 1] += overlap_y0[u] * 2.0
                grad_radii[u] -= overlap_y0[u]
            if overlap_y1[u] > 0:
                grad_centers[u, 1] -= overlap_y1[u] * 2.0
                grad_radii[u] -= overlap_y1[u]

        # Apply gradients
        centers += lr_pos * grad_centers
        radii += lr_rad * grad_radii
        
        centers = np.clip(centers, 0.001, 0.999)
        radii = np.clip(radii, 0.001, 0.5)

    return centers


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses linear programming for optimal assignment, with a fallback.
    """
    n = centers.shape[0]
    
    try:
        from scipy.optimize import linprog
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
                
        bounds = []
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            bounds.append((0, max_r))
            
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
        
    # Fallback if LP fails or scipy is not available
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
        
    for _ in range(150):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist + 1e-9:
                    diff = radii[i] + radii[j] - dist
                    if radii[i] > 0 and radii[j] > 0:
                        ratio = radii[i] / (radii[i] + radii[j])
                        radii[i] -= diff * ratio
                        radii[j] -= diff * (1.0 - ratio)
                    elif radii[i] > 0:
                        radii[i] -= diff
                    elif radii[j] > 0:
                        radii[j] -= diff
                    changed = True
        if not changed:
            break
            
    return np.maximum(radii, 0.0)

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
        if radius > 0:
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