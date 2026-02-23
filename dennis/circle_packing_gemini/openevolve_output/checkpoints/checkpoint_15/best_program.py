# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1
    
    configs = []
    
    # Config Group 1: Sunflower Spirals (Vogel's model mapped to a square)
    # This naturally distributes points uniformly but not strictly grid-like, 
    # allowing varying radii perfectly.
    golden_angle = np.pi * (3 - np.sqrt(5))
    for scale in [0.4, 0.44, 0.48]:
        c = []
        for i in range(n):
            r = np.sqrt(i + 0.5) / np.sqrt(n)
            theta = i * golden_angle
            x = 0.5 + scale * r * np.cos(theta)
            y = 0.5 + scale * r * np.sin(theta)
            c.append([x, y])
        configs.append(c)
        
    # Config Group 2: Symmetrical Hexagonal Arrays
    # Try different shapes and paddings to find the optimal boundary interactions.
    hex_shapes = [[5, 6, 4, 6, 5], [6, 5, 4, 5, 6], [4, 6, 6, 6, 4]]
    for shape in hex_shapes:
        for pad in [0.08, 0.1, 0.12]:
            c = []
            y_starts = np.linspace(pad, 1 - pad, 5)
            for r_idx, count in enumerate(shape):
                y = y_starts[r_idx]
                xs = np.linspace(pad, 1 - pad, count)
                for x in xs:
                    c.append([x, y])
            if len(c) == n:
                configs.append(c)

    # Config Group 3: Concentric Patterns
    for rings, radii_r in [
        ([1, 6, 9, 10], [0.0, 0.18, 0.35, 0.48]),
        ([5, 9, 12], [0.15, 0.32, 0.46])
    ]:
        c = []
        for ring_idx, count in enumerate(rings):
            r = radii_r[ring_idx]
            if count == 1:
                c.append([0.5, 0.5])
            else:
                for i in range(count):
                    angle = 2 * np.pi * i / count + (np.pi/count if ring_idx % 2 == 1 else 0)
                    c.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
        if len(c) == n:
            configs.append(c)
            
    # Config Group 4: Fractal / Apollonian-like structural layout
    c = [[0.5, 0.5]]
    for r in [0.15, 0.85]:
        for p in [0.15, 0.85]:
            c.append([r, p])
    for e in [0.1, 0.9]:
        c.append([0.5, e])
        c.append([e, 0.5])
    for i in range(8):
        a = 2 * np.pi * i / 8 + np.pi/8
        c.append([0.5 + 0.3 * np.cos(a), 0.5 + 0.3 * np.sin(a)])
    for i in range(9):
        a = 2 * np.pi * i / 9
        c.append([0.5 + 0.45 * np.cos(a), 0.5 + 0.45 * np.sin(a)])
    if len(c) == n:
        configs.append(c)
        
    # Config Group 5: Small deterministic perturbations of the 6-5-4-5-6 grid
    base_shape = [6, 5, 4, 5, 6]
    base_c = []
    y_starts = np.linspace(0.1, 0.9, 5)
    for r_idx, count in enumerate(base_shape):
        y = y_starts[r_idx]
        xs = np.linspace(0.1, 0.9, count)
        for x in xs:
            base_c.append([x, y])
    base_c_np = np.array(base_c)
    
    rng = np.random.RandomState(42)
    for _ in range(3):
        noise = rng.uniform(-0.015, 0.015, (n, 2))
        configs.append((base_c_np + noise).tolist())
        
    # Evaluate all explicit configurations and choose the absolute best
    for c_list in configs:
        c_np = np.clip(np.array(c_list), 0.001, 0.999)
        radii = compute_max_radii(c_np)
        s = np.sum(radii)
        if s > best_sum:
            best_sum = s
            best_centers = c_np
            best_radii = radii
            
    return best_centers, best_radii, best_sum

def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    by solving a Linear Programming problem to directly maximize sum of radii.
    """
    n = centers.shape[0]
    radii = None
    
    try:
        from scipy.optimize import linprog
        c = -np.ones(n) # Maximize sum of radii
        A_ub = []
        b_ub = []
        bounds = []
        
        # Limit by distance to square borders
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, 1-x, y, 1-y)
            bounds.append((0, max_r))
            
        # Limit by distance to other circles (r_i + r_j <= distance)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
                
        # Suppress warnings and execute fast LP solver
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if res.success:
            radii = res.x
    except Exception:
        pass

    if radii is None:
        # Fallback solver: Projected Gradient Ascent physically simulating inflating balloons
        radii = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            radii[i] = min(x, 1-x, y, 1-y)
            
        lr = 0.02
        for step in range(800):
            radii += lr * (1.0 - step/800)
            # Iteratively resolve constraints
            for _ in range(2):
                for i in range(n):
                    x, y = centers[i]
                    radii[i] = min(radii[i], x, 1-x, y, 1-y)
                for i in range(n):
                    for j in range(i+1, n):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if radii[i] + radii[j] > dist:
                            diff = (radii[i] + radii[j] - dist) / 2
                            radii[i] -= diff
                            radii[j] -= diff

    # Final robust, strict enforcement to absolutely prevent floating point constraint violations
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x - 1e-8, 1-x - 1e-8, y - 1e-8, 1-y - 1e-8)
        radii[i] = max(0.0, radii[i])
        
    for _ in range(10):
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist + 1e-9:
                    if dist < 1e-6:
                        radii[i] = 0.0
                        radii[j] = 0.0
                    else:
                        sum_r = radii[i] + radii[j]
                        if sum_r > 0:
                            scale = (dist - 1e-8) / sum_r
                            radii[i] *= scale
                            radii[j] *= scale
                            
    return radii
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
    visualize(centers, radii)