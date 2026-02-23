"""
Iterative optimization-based circle packing for n=26 circles.
Uses Adam optimizer with a custom physics-based objective to maximize 
the sum of radii while strictly penalizing overlapping and boundary violations.
Runs multiple restarts from diverse strategic initializations to escape local minima.
"""
import numpy as np

def run_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    to maximize the sum of their radii without overlapping.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    B = 250  # Batch size for parallel topology exploration
    N = 26
    iters = 3500
    
    np.random.seed(42)
    
    C = np.zeros((B, N, 2))
    
    # 1. Seed initial positions strategically:
    idx = 0
    step = B // 6
    
    # Strategy A: Uniform random
    C[idx:idx+step] = np.random.rand(step, N, 2)
    idx += step
    
    # Strategy B: Centered normal (clusters in middle, leaves edges for expansion)
    C[idx:idx+step] = np.random.normal(0.5, 0.15, size=(step, N, 2))
    idx += step
    
    # Strategy C: Beta distribution (pushes to corners and edges)
    C[idx:idx+step] = np.random.beta(0.4, 0.4, size=(step, N, 2))
    idx += step
    
    # Strategy D: Concentric rings pattern
    ring_centers = np.zeros((N, 2))
    ring_centers[0] = [0.5, 0.5]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        ring_centers[i + 1] = [0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)]
    for i in range(17):
        angle = 2 * np.pi * i / 17
        ring_centers[i + 9] = [0.5 + 0.45 * np.cos(angle), 0.5 + 0.45 * np.sin(angle)]
    C[idx:idx+step] = ring_centers + np.random.normal(0, 0.01, size=(step, N, 2))
    idx += step
    
    # Strategy E: Grid-like patterns
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 6))
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    C[idx:idx+step] = grid + np.random.normal(0, 0.01, size=(step, N, 2))
    idx += step
    
    # Strategy F: Uniform with corners explicitly anchored
    rem = B - idx
    if rem > 0:
        C[idx:] = np.random.uniform(0.1, 0.9, size=(rem, N, 2))
        C[idx:, :4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    
    # Boundary enforcement for initialization
    C = np.clip(C, 0.05, 0.95)
    
    # 2. Size placement bias: Push larger circles toward center
    dist_to_center = np.linalg.norm(C - 0.5, axis=2)
    R = 0.03 + 0.04 * (0.707 - dist_to_center)
    R = np.clip(R, 0.01, 0.15)
    
    # Adam optimizer state
    m_C = np.zeros_like(C)
    v_C = np.zeros_like(C)
    m_R = np.zeros_like(R)
    v_R = np.zeros_like(R)
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    mask = np.eye(N)[np.newaxis, :, :]
    
    # Optimization Loop
    for t in range(1, iters + 1):
        # 3. Tune optimization parameters: Simulated Annealing penalty phase
        phase_ratio = min(1.0, t / (iters * 0.8))
        if t <= iters * 0.8:
            lr = 0.01 * (1.0 - 0.5 * phase_ratio)
            lambda_val = 10.0 * (10000.0 ** phase_ratio)
        else:
            lr = 0.001
            lambda_val = 100000.0
            
        # Calculate wall overlap violations
        viol_x0 = np.maximum(0, R - C[:, :, 0])
        viol_x1 = np.maximum(0, R + C[:, :, 0] - 1)
        viol_y0 = np.maximum(0, R - C[:, :, 1])
        viol_y1 = np.maximum(0, R + C[:, :, 1] - 1)
        
        # Calculate pairwise overlap violations safely
        diff = C[:, :, np.newaxis, :] - C[:, np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(dist_sq + mask + 1e-12)
        
        R_sum = R[:, :, np.newaxis] + R[:, np.newaxis, :]
        viol_pair = np.maximum(0, R_sum - dist) * (1 - mask)
        
        # Compute exact analytical gradients
        grad_R = -1.0 + 2 * lambda_val * (viol_x0 + viol_x1 + viol_y0 + viol_y1 + np.sum(viol_pair, axis=2))
        
        grad_C_x = 2 * lambda_val * (-viol_x0 + viol_x1)
        grad_C_y = 2 * lambda_val * (-viol_y0 + viol_y1)
        grad_C_bound = np.stack([grad_C_x, grad_C_y], axis=-1)
        
        dir = diff / dist[..., np.newaxis]
        grad_C_pair = 2 * lambda_val * np.sum(viol_pair[..., np.newaxis] * (-dir), axis=2)
        grad_C = grad_C_bound + grad_C_pair
        
        # Apply Adam update for Centers
        m_C = beta1 * m_C + (1 - beta1) * grad_C
        v_C = beta2 * v_C + (1 - beta2) * (grad_C**2)
        m_C_hat = m_C / (1 - beta1**t)
        v_C_hat = v_C / (1 - beta2**t)
        C -= lr * m_C_hat / (np.sqrt(v_C_hat) + eps)
        
        # Apply Adam update for Radii
        m_R = beta1 * m_R + (1 - beta1) * grad_R
        v_R = beta2 * v_R + (1 - beta2) * (grad_R**2)
        m_R_hat = m_R / (1 - beta1**t)
        v_R_hat = v_R / (1 - beta2**t)
        R -= lr * m_R_hat / (np.sqrt(v_R_hat) + eps)
        
        # 4. Break perfect symmetry: Introduce small perturbations early on
        if t % 250 == 0 and t < iters * 0.6:
            C += np.random.randn(*C.shape) * 0.002
            
        # Constrain variables logically
        C = np.clip(C, 0, 1)
        R = np.maximum(R, 0)
        
    # Post Processing Phase: Guarantee mathematically perfect validity
    best_sum_R = -1
    best_C = None
    best_R = None
    
    for b in range(B):
        Cb = C[b].copy()
        Rb = R[b].copy()
        
        # Limit to boundaries explicitly (incorporating fp safety margin)
        Rb = np.clip(Rb, 0, None)
        Rb = np.minimum(Rb, Cb[:, 0] - 1e-12)
        Rb = np.minimum(Rb, 1 - Cb[:, 0] - 1e-12)
        Rb = np.minimum(Rb, Cb[:, 1] - 1e-12)
        Rb = np.minimum(Rb, 1 - Cb[:, 1] - 1e-12)
        Rb = np.maximum(Rb, 0)
        
        # Iterate strict greedy shrink pass to resolve infinitesimal remaining overlaps
        for _ in range(5):
            for i in range(N):
                for j in range(i + 1, N):
                    dist_ij = np.linalg.norm(Cb[i] - Cb[j])
                    if Rb[i] + Rb[j] > dist_ij:
                        if dist_ij > 0:
                            scale = (dist_ij - 1e-12) / (Rb[i] + Rb[j])
                            Rb[i] *= scale
                            Rb[j] *= scale
                        else:
                            Rb[i] = 0
                            Rb[j] = 0
                            
        sum_R = np.sum(Rb)
        if sum_R > best_sum_R:
            best_sum_R = sum_R
            best_C = Cb
            best_R = Rb
            
    return best_C, best_R, best_sum_R

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Optimal Sum of Radii achieved: {sum_radii:.6f}")