# EVOLVE-BLOCK-START
"""Iterative Adam-based optimizer for packing n=26 unequal circles"""
import numpy as np


def construct_packing():
    """
    Construct an arrangement of 26 circles using iterative gradient 
    optimization with Adam. Maximizes sum of their radii without overlapping.
    
    Returns:
        Tuple of (centers, radii, sum_radii)
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialize strategically biased starting configurations
    inits = []
    
    # Initialization A: Hexagonal layout broadly adapted for N=26
    c_hex = []
    for row in range(6):
        for col in range(5):
            if len(c_hex) < n:
                x = 0.15 + col * 0.15 + (0.075 if row % 2 else 0)
                y = 0.15 + row * 0.14
                c_hex.append([x, y])
    inits.append((np.clip(c_hex, 0.05, 0.95), np.ones(n) * 0.03))
    
    # Initialization B: Center-biased symmetric pattern layout
    c_ring = np.zeros((n, 2))
    c_ring[0] = [0.5, 0.5]
    for i in range(8):
        c_ring[i+1] = [0.5 + 0.2*np.cos(2*np.pi*i/8), 0.5 + 0.2*np.sin(2*np.pi*i/8)]
    for i in range(17):
        c_ring[i+9] = [0.5 + 0.4*np.cos(2*np.pi*i/17), 0.5 + 0.4*np.sin(2*np.pi*i/17)]
    inits.append((np.clip(c_ring, 0.05, 0.95), np.ones(n) * 0.02))
    
    # Initializations C: Random starts mimicking spatial variations
    for _ in range(8):
        c = np.random.uniform(0.1, 0.9, (n, 2))
        r = np.random.uniform(0.01, 0.06, n)
        inits.append((c, r))

    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Perform simulation steps across different seed spaces
    for c_init, r_init in inits:
        c_final, r_final = optimize_single(c_init, r_init, steps=6000)
        
        # Eliminate floating precision issues enforcing perfectly safe bounding boxes
        r_valid = make_strictly_valid(c_final, r_final)
        
        current_sum = np.sum(r_valid)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = c_final
            best_radii = r_valid
            
    return best_centers, best_radii, best_sum


def optimize_single(c_init, r_init, steps=6000):
    """Core computational physics optimization kernel for a single setup"""
    n = len(r_init)
    params = np.zeros((n, 3))
    params[:, :2] = c_init
    params[:, 2] = r_init
    
    # Adam optim parameters
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    
    base_lr = 0.01
    wp = 20.0 
    
    for step in range(steps):
        # 1. Distances and continuous overlaps evaluations
        diff = params[:, np.newaxis, :2] - params[np.newaxis, :, :2]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        
        target = params[:, np.newaxis, 2] + params[np.newaxis, :, 2]
        overlap = np.maximum(0, target - dist)
        
        grad = np.zeros_like(params)
        
        # Uniform force pulling radius sizes up!
        grad[:, 2] = -1.0 
        
        # Simulated simulated annealing / symmetry momentum breaker 
        if step < 2000 and step % 250 == 0:
            grad[:, :2] += np.random.normal(0, 0.5, (n, 2))
            
        if np.any(overlap > 0):
            # Enact heavy force pushing overlapping neighbors away directly proportional to overlap
            overlap_grad = 2 * wp * overlap
            grad[:, 2] += np.sum(overlap_grad, axis=1)
            
            valid_dist = dist > 1e-12
            dir_norm = np.zeros_like(diff)
            dir_norm[valid_dist] = diff[valid_dist] / dist[valid_dist][..., np.newaxis]
            
            c_grad = -overlap_grad[..., np.newaxis] * dir_norm
            grad[:, :2] += np.sum(c_grad, axis=1)
            
        # Hard limits boundary constraints enforcement penalty calculation
        bx1 = np.maximum(0, params[:, 2] - params[:, 0])
        grad[:, 2] += 2 * wp * bx1
        grad[:, 0] -= 2 * wp * bx1
        
        bx2 = np.maximum(0, params[:, 0] + params[:, 2] - 1.0)
        grad[:, 2] += 2 * wp * bx2
        grad[:, 0] += 2 * wp * bx2
        
        by1 = np.maximum(0, params[:, 2] - params[:, 1])
        grad[:, 2] += 2 * wp * by1
        grad[:, 1] -= 2 * wp * by1
        
        by2 = np.maximum(0, params[:, 1] + params[:, 2] - 1.0)
        grad[:, 2] += 2 * wp * by2
        grad[:, 1] += 2 * wp * by2
        
        # Floor boundaries minimum 
        min_rad = 0.005
        neg_r = np.maximum(0, min_rad - params[:, 2])
        grad[:, 2] -= 2 * wp * neg_r
        
        # Exponential decreasing decay scale learning phase rate limit factor map parameter down!
        lr = base_lr * (0.01 ** (step / max(1, steps - 1)))
        
        # Applying iterative adam states transition variables momentum array combinations smoothing gradient updates  
        t = step + 1
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad ** 2)
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        
        params -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        # Iteratively strengthening restrictions gradually tightening overlapping limit gaps limits tolerances bounds values! 
        if step % 500 == 499:
            wp *= 1.4

    return params[:, :2], params[:, 2]


def make_strictly_valid(centers, radii):
    """
    Absolute final constraint cleanup safety wrapper ensuring perfectly validated valid radius arrays returns
    Strictly trims and ensures sizes stay structurally locked internally!
    """
    centers = np.clip(centers, 0.0, 1.0)
    
    # Establish borders distance sizes absolute limiting boundaries mapping
    b_rad = np.min([
        centers[:, 0], 
        1.0 - centers[:, 0], 
        centers[:, 1], 
        1.0 - centers[:, 1]
    ], axis=0)
    
    radii = np.minimum(radii, b_rad)
    radii = np.maximum(0.0, radii)
    
    n = len(radii)
    # Strictly execute non overlapping reduction pairwise loops! 
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                margin_dist = max(0.0, dist - 1e-7)
                if radii[i] + radii[j] > 1e-9:
                    scale = margin_dist / (radii[i] + radii[j])
                    scale = min(1.0, scale)
                    radii[i] *= scale
                    radii[j] *= scale
    return radii


# Alias retaining structure interface from legacy wrapper if used
def compute_max_radii(centers):
    """Compatibility method."""
    radii = np.ones(len(centers)) * 0.01
    return make_strictly_valid(centers, radii)

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
    # visualize(centers, radii)