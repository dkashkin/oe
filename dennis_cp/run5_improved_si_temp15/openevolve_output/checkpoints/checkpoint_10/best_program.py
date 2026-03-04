# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def generate_initial_states(batch_size, n_circles):
    """
    Strategically biases seeds into corners, edges, and dense center structures,
    while breaking symmetry through micro-randomizations.
    """
    positions = np.zeros((batch_size, n_circles, 2))
    radii = np.zeros((batch_size, n_circles))
    
    for b in range(batch_size):
        idx = 0
        
        # Central circles (varying cluster arrangements to maximize topology search space)
        num_center = b % 4 + 1
        for i in range(num_center):
            if idx < n_circles:
                angle = 2.0 * np.pi * i / num_center
                radius = 0.05 if num_center > 1 else 0.0
                positions[b, idx] = [
                    0.5 + radius * np.cos(angle), 
                    0.5 + radius * np.sin(angle)
                ]
                radii[b, idx] = 0.15 - 0.02 * num_center
                idx += 1
                
        corners = [[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]]
        edges = [[0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
        
        places = corners + edges
        np.random.shuffle(places)
        
        # Deploy at edges/corners iteratively
        for pos in places:
            if idx < n_circles:
                p_x = pos[0] + np.random.uniform(-0.02, 0.02)
                p_y = pos[1] + np.random.uniform(-0.02, 0.02)
                positions[b, idx] = [p_x, p_y]
                radii[b, idx] = 0.08
                idx += 1
                
        # Fill interstitials 
        while idx < n_circles:
            positions[b, idx] = np.random.uniform(0.1, 0.9, 2)
            radii[b, idx] = np.random.uniform(0.02, 0.06)
            idx += 1
            
    positions = np.clip(positions, 0.0, 1.0)
    return positions, radii


def make_valid(positions, radii):
    """
    Rigorously cleans the batch solution down to exactly 100% precision valid states
    where there are mathematically strictly zero structural overlaps.
    """
    r_out = radii.copy()
    n_circles = len(positions)
    
    # Boundary limiting pass
    for i in range(n_circles):
        x, y = positions[i]
        max_r = min(x, y, 1.0 - x, 1.0 - y)
        if r_out[i] > max_r:
            r_out[i] = max_r
            
    # Successive intersection reduction mapping
    for _ in range(100):
        has_overlap = False
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                d = np.linalg.norm(positions[i] - positions[j])
                
                # Check bounding against tiny floating scale offset
                if r_out[i] + r_out[j] > d + 1e-9:
                    if d < 1e-7:
                        r_out[i] *= 0.5
                        r_out[j] *= 0.5
                    else:
                        scale = d / (r_out[i] + r_out[j])
                        # Safety compression margin mitigates Zeno-locking bounds loops
                        scale *= 0.99999 
                        r_out[i] *= scale
                        r_out[j] *= scale
                    has_overlap = True
        
        # Perfect stable-packing breaks early
        if not has_overlap:
            break
            
    return r_out


def construct_packing():
    """
    Constructs an optimized mathematical geometric simulation packing 
    applying physics-modeled continuous gradient Adam momentum mechanics.
    Provides natural pressure growth to systematically seek maximized cumulative bounds.
    """
    batch_size = 32
    n_circles = 26
    iters = 3000
    
    positions, radii = generate_initial_states(batch_size, n_circles)
    radii = radii.reshape(batch_size, n_circles, 1)
    
    # Configure Adam state allocations
    m_p, v_p = np.zeros_like(positions), np.zeros_like(positions)
    m_r, v_r = np.zeros_like(radii), np.zeros_like(radii)
    
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    base_learning_rate = 0.005
    
    # Perform Continuous Constraint and Target Growth Vector Mapping (Numpy SIMD Processed)
    for step in range(1, iters + 1):
        # Gradual simulated annealing constraints 
        lr = base_learning_rate * (1.0 - 0.9 * step / iters)
        penalty = 10.0 * (1000.0 ** (step / iters))
        
        x = positions[:, :, 0:1]
        y = positions[:, :, 1:2]
        
        dx = x - x.transpose(0, 2, 1)
        dy = y - y.transpose(0, 2, 1)
        
        # Formulate non-diverging bounds matrix for pair overlaps 
        dist = np.sqrt(dx**2 + dy**2 + 1e-8)
        dist += np.eye(n_circles).reshape(1, n_circles, n_circles) * 100.0
        
        sum_r = radii + radii.transpose(0, 2, 1)
        
        overlap_pair = np.maximum(0.0, sum_r - dist)
        force_pair = 2.0 * penalty * overlap_pair
        
        # Accumulate matrix gradients mathematically symmetrical interactions
        grad_x_pair = np.sum(-force_pair * dx / dist, axis=2, keepdims=True)
        grad_y_pair = np.sum(-force_pair * dy / dist, axis=2, keepdims=True)
        grad_r_pair = np.sum(force_pair, axis=2, keepdims=True)
        
        # Construct constraint mechanics against geometry bounds limitation
        ox0 = np.maximum(0.0, radii - x)
        fx0 = 2.0 * penalty * ox0
        ox1 = np.maximum(0.0, radii + x - 1.0)
        fx1 = 2.0 * penalty * ox1
        
        oy0 = np.maximum(0.0, radii - y)
        fy0 = 2.0 * penalty * oy0
        oy1 = np.maximum(0.0, radii + y - 1.0)
        fy1 = 2.0 * penalty * oy1
        
        grad_x = grad_x_pair - fx0 + fx1
        grad_y = grad_y_pair - fy0 + fy1
        grad_positions = np.concatenate([grad_x, grad_y], axis=-1)
        
        grad_radii = grad_r_pair + fx0 + fx1 + fy0 + fy1
        grad_radii -= 2.0  # Imparts uniform target radius sum maximizing bounds pressure 
        
        # Compute discrete Adam velocity steps 
        m_p = beta1 * m_p + (1.0 - beta1) * grad_positions
        v_p = beta2 * v_p + (1.0 - beta2) * (grad_positions ** 2)
        step_p = (m_p / (1.0 - beta1**step)) / (np.sqrt(v_p / (1.0 - beta2**step)) + epsilon)
        positions -= lr * step_p
        
        m_r = beta1 * m_r + (1.0 - beta1) * grad_radii
        v_r = beta2 * v_r + (1.0 - beta2) * (grad_radii ** 2)
        step_r = (m_r / (1.0 - beta1**step)) / (np.sqrt(v_r / (1.0 - beta2**step)) + epsilon)
        radii -= lr * step_r
        
        # Keep components locally rigid to mathematical possibility structures
        positions = np.clip(positions, 0.0, 1.0)
        radii = np.clip(radii, 0.001, 0.5)

    best_score = -1.0
    best_positions = None
    best_radii = None
    
    # Discover supreme outcome enforcing totally correct unoverlapped borders
    for b in range(batch_size):
        curr_positions = positions[b]
        curr_radii = radii[b, :, 0]
        
        cleaned_radii = make_valid(curr_positions, curr_radii)
        eval_score = np.sum(cleaned_radii)
        
        if eval_score > best_score:
            best_score = eval_score
            best_positions = curr_positions
            best_radii = cleaned_radii
            
    return best_positions, best_radii, best_score


def compute_max_radii(centers):
    """
    Functionally preserved for pipeline continuity if strictly examined standalone.
    Yields overlapping safety-bounds dynamically mirroring strict shrinkage model mapping.
    """
    n = centers.shape[0]
    radii = np.ones(n)

    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
                
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
    # Uncomment to visualize:
    # visualize(centers, radii)