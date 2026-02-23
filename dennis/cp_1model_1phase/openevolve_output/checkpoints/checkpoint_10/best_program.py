"""Constructor-based circle packing for n=26 circles in a unit square"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that maximizes the sum of their radii using a force-directed
    simulation and a greedy radius optimization.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    # Seed for reproducibility
    np.random.seed(42)

    # Initial layout: 5x5 grid plus one extra circle
    centers = np.zeros((n, 2))
    for i in range(5):
        for j in range(5):
            centers[i * 5 + j] = [0.2 * i + 0.1, 0.2 * j + 0.1]
    
    # Place the 26th circle in the center and add jitter to break symmetry
    centers[25] = [0.5, 0.5]
    centers += np.random.uniform(-0.02, 0.02, (n, 2))

    # Force-directed simulation parameters
    # A target distance of ~0.203 aims for a sum of radii around 2.639
    iters = 1200
    dt = 0.05
    target_dist = 0.203
    target_radius = target_dist / 2.0

    # Physics simulation loop
    for _ in range(iters):
        forces = np.zeros((n, 2))
        # Pairwise repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = (dx * dx + dy * dy)**0.5 + 1e-9
                if dist < target_dist:
                    f = (target_dist - dist) / dist
                    forces[i, 0] += dx * f
                    forces[i, 1] += dy * f
                    forces[j, 0] -= dx * f
                    forces[j, 1] -= dy * f
        
        # Repulsion from square boundaries
        for i in range(n):
            # Distance to left wall
            if centers[i, 0] < target_radius:
                forces[i, 0] += (target_radius - centers[i, 0])
            # Distance to right wall
            if centers[i, 0] > 1.0 - target_radius:
                forces[i, 0] -= (centers[i, 0] - (1.0 - target_radius))
            # Distance to bottom wall
            if centers[i, 1] < target_radius:
                forces[i, 1] += (target_radius - centers[i, 1])
            # Distance to top wall
            if centers[i, 1] > 1.0 - target_radius:
                forces[i, 1] -= (centers[i, 1] - (1.0 - target_radius))
        
        # Update centers and constrain within unit square
        centers += forces * dt
        centers = np.clip(centers, 0.0, 1.0)

    # Compute final maximized radii for these center positions
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Maximize the sum of radii for fixed centers by solving the
    overlap and boundary constraints iteratively.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)

    # 1. Initialize radii to the distance to the closest boundary
    for i in range(n):
        radii[i] = min(centers[i, 0], centers[i, 1], 
                       1.0 - centers[i, 0], 1.0 - centers[i, 1])

    # 2. Proportionally scale down radii to eliminate overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = (dx * dx + dy * dy)**0.5
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j] + 1e-12)
                radii[i] *= scale
                radii[j] *= scale

    # 3. Iteratively expand circles greedily to maximize the sum of radii
    # We alternate the order to ensure fair space distribution
    for it in range(60):
        order = range(n) if it % 2 == 0 else range(n - 1, -1, -1)
        for i in order:
            # Maximum radius is limited by distance to boundaries
            r_limit = min(centers[i, 0], centers[i, 1], 
                          1.0 - centers[i, 0], 1.0 - centers[i, 1])
            # And distance to every other already-sized circle
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = (dx * dx + dy * dy)**0.5
                    r_limit = min(r_limit, dist - radii[j])
            radii[i] = max(0.0, r_limit)

    return radii


def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")