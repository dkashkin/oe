# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles in a unit square"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii using a 
    force-directed simulation and greedy radius optimization.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    # Initial layout: 5x5 grid plus one circle in the middle
    centers = np.zeros((n, 2))
    for i in range(25):
        centers[i] = [0.1 + (i % 5) * 0.2, 0.1 + (i // 5) * 0.2]
    centers[25] = [0.5, 0.5]

    # Add small jitter to break symmetry for better packing exploration
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, (n, 2))

    # Parameters for the physics-based repulsion simulation
    # Target distance corresponds to an average radius of ~0.1015
    iters = 500
    dt = 0.08
    target_dist = 0.203

    # Simulation loop: push overlapping circles apart
    for _ in range(iters):
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.sqrt(d[0]**2 + d[1]**2)
                if dist < target_dist:
                    # Prevent division by zero with small epsilon
                    if dist < 1e-8:
                        d = np.array([0.01, 0.0])
                        dist = 0.01
                    # Repulsion force proportional to overlap
                    f = (target_dist - dist) / dist
                    forces[i] += d * f
                    forces[j] -= d * f
        
        centers += forces * dt
        # Constrain centers to the unit square
        centers = np.clip(centers, 0.0, 1.0)

    # Compute final optimized radii for these center positions
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    using proportional scaling followed by iterative greedy refinement.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)

    # 1. Initialize radii based on distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # 2. Proportionally scale radii to remove any circle-circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    # 3. Greedy refinement to maximize the sum of radii
    # In each step, maximize r_i given all other fixed radii
    for _ in range(20):
        for i in range(n):
            # Distance to boundaries
            m = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
            # Distance to all other circles
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist - radii[j] < m:
                        m = dist - radii[j]
            radii[i] = max(0.0, m)

    return radii


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
    # AlphaEvolve improved this to 2.635
    # visualize(centers, radii)