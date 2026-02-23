# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using physics-based optimization"""
import numpy as np

def fix_radii(centers, radii):
    """
    Ensure all circles strictly satisfy boundary and non-overlap constraints.
    Shrinks circles iteratively to resolve any minor violations.
    """
    radii = np.copy(radii)
    n = len(radii)
    
    # Boundary constraints
    for i in range(n):
        max_r = min(
            centers[i, 0], 
            centers[i, 1], 
            1.0 - centers[i, 0], 
            1.0 - centers[i, 1]
        )
        radii[i] = min(radii[i], max_r * 0.9999999)
        radii[i] = max(0.0, radii[i])
        
    # Pairwise non-overlap constraints
    while True:
        max_violation = 0.0
        max_pair = None
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > max_violation + 1e-12:
                    max_violation = overlap
                    max_pair = (i, j)
                    
        if max_pair is None:
            break
            
        i, j = max_pair
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if radii[i] + radii[j] > 0:
            scale = (dist / (radii[i] + radii[j])) * 0.9999999
        else:
            scale = 0.0
            
        radii[i] *= scale
        radii[j] *= scale
        
    return radii


def optimize_packing(n=26, max_steps=3000, seed=42):
    """
    Optimizes circle centers and radii using a physics-based approach
    with Adam optimizer and simulated annealing.
    """
    np.random.seed(seed)
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Initialization strategies
    if seed % 3 == 0:
        # Sunflower spiral setup for even distribution
        for i in range(n):
            r = 0.45 * np.sqrt((i + 0.5) / n)
            theta = i * 2.39996323  # Golden angle
            centers[i] = [0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)]
            radii[i] = 0.05
    elif seed % 3 == 1:
        # Grid layout
        grid_dim = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            row = i // grid_dim
            col = i % grid_dim
            centers[i] = [
                0.1 + 0.8 * col / max(1, grid_dim - 1), 
                0.1 + 0.8 * row / max(1, grid_dim - 1)
            ]
            radii[i] = 0.05
    else:
        # Random initialization
        centers = np.random.uniform(0.1, 0.9, (n, 2))
        radii = np.random.uniform(0.03, 0.08, n)
        
    # Break perfect symmetry with random perturbations
    centers += np.random.normal(0, 0.005, size=(n, 2))
    centers = np.clip(centers, 0.01, 0.99)
    
    # Adam optimizer parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    m_c = np.zeros_like(centers)
    v_c = np.zeros_like(centers)
    m_r = np.zeros_like(radii)
    v_r = np.zeros_like(radii)
    
    # Penalty coefficients
    lambda_init = 10.0
    lambda_final = 1e5
    
    for step in range(1, max_steps + 1):
        progress = step / max_steps
        
        # Decaying learning rate and penalty scheduling
        current_lr = lr * (0.01 ** progress)
        current_lambda = lambda_init * ((lambda_final / lambda_init) ** progress)
        
        # Simulated annealing noise
        if step < max_steps * 0.5:
            noise_scale = 0.001 * (1.0 - progress / 0.5)
            centers += np.random.normal(0, noise_scale, size=centers.shape)
            
        # Compute pairwise distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Compute overlaps
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0.0, r_sum - dists)
        np.fill_diagonal(overlap, 0.0)
        
        # Compute boundary violations
        v_x0 = np.maximum(0.0, radii - centers[:, 0])
        v_x1 = np.maximum(0.0, centers[:, 0] + radii - 1.0)
        v_y0 = np.maximum(0.0, radii - centers[:, 1])
        v_y1 = np.maximum(0.0, centers[:, 1] + radii - 1.0)
        
        # Initialize gradients
        grad_r = -1.0 * np.ones_like(radii)
        grad_c = np.zeros_like(centers)
        
        # Add gradients from boundary penalties
        grad_r += current_lambda * (v_x0 + v_x1 + v_y0 + v_y1)
        grad_c[:, 0] += current_lambda * (-v_x0 + v_x1)
        grad_c[:, 1] += current_lambda * (-v_y0 + v_y1)
        
        # Add gradients from overlap penalties
        grad_r += current_lambda * np.sum(overlap, axis=1)
        
        safe_dists = np.maximum(dists, 1e-10)
        overlap_factor = current_lambda * overlap / safe_dists
        grad_c -= np.sum(overlap_factor[:, :, np.newaxis] * diffs, axis=1)
        
        # Gradient clipping to prevent instability
        grad_c = np.clip(grad_c, -100.0, 100.0)
        grad_r = np.clip(grad_r, -100.0, 100.0)
        
        # Adam step for centers
        m_c = beta1 * m_c + (1 - beta1) * grad_c
        v_c = beta2 * v_c + (1 - beta2) * (grad_c ** 2)
        m_c_hat = m_c / (1 - beta1 ** step)
        v_c_hat = v_c / (1 - beta2 ** step)
        centers -= current_lr * m_c_hat / (np.sqrt(v_c_hat) + epsilon)
        
        # Adam step for radii
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r ** 2)
        m_r_hat = m_r / (1 - beta1 ** step)
        v_r_hat = v_r / (1 - beta2 ** step)
        radii -= current_lr * m_r_hat / (np.sqrt(v_r_hat) + epsilon)
        
        # Constrain to sensible space
        centers = np.clip(centers, 0.001, 0.999)
        radii = np.maximum(radii, 0.001)
        
    # Ensure rigorous validity at the end
    radii = fix_radii(centers, radii)
    return centers, radii, np.sum(radii)


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Run optimization with a few different seeds to find the best configuration
    for seed in [11, 42, 1337, 2024, 99]:
        centers, radii, sum_r = optimize_packing(n=26, max_steps=2500, seed=seed)
        if sum_r > best_sum:
            best_sum = sum_r
            best_centers = centers
            best_radii = radii
            
    return best_centers, best_radii, best_sum
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