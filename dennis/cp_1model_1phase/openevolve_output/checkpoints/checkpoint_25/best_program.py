import numpy as np


def compute_max_radii(centers):
    """
    Maximizes the sum of radii for a given set of center coordinates
    within a unit square, ensuring no overlaps and respect for boundaries.
    Uses coordinate descent with multiple greedy expansion orders.
    """
    n = centers.shape[0]
    # Precompute pairwise distances and boundary limits
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    # Distance to the nearest boundary for each center
    boundary_limits = np.min(np.concatenate([centers, 1.0 - centers], axis=1), axis=1)

    best_radii = np.zeros(n)
    best_sum = -1.0

    # Try several expansion orders to maximize the sum of radii
    for trial in range(13):
        radii = np.zeros(n)
        if trial == 0:
            order = np.arange(n)
        elif trial == 1:
            order = np.argsort(boundary_limits)
        else:
            order = np.random.permutation(n)

        # Coordinate descent to solve for optimal radii for fixed centers
        for _ in range(25):
            for i in order:
                # Maximum radius r_i is limited by boundary and dist(i,j) - r_j
                r_i = boundary_limits[i]
                for j in range(n):
                    if i != j:
                        r_i = min(r_i, dist_matrix[i, j] - radii[j])
                radii[i] = max(0.0, r_i)

        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_radii = radii.copy()

    return best_radii


def construct_packing():
    """
    Constructs an arrangement of 26 circles in a unit square using a
    force-directed physics simulation with cooling and jitter, followed
    by a greedy radii optimization to maximize the sum of radii.
    """
    n = 26
    iters = 1600
    dt_base = 0.05
    
    best_overall_sum = -1.0
    best_overall_centers = None
    best_overall_radii = None
    
    # Range of target distances to explore across multiple restarts
    target_dists = np.linspace(0.198, 0.208, 10)
    
    for restart in range(10):
        # Set seed for reproducibility per restart
        np.random.seed(200 + restart)
        target_dist = target_dists[restart]
        
        # Initial layout: 5x5 grid plus one extra circle
        centers = np.zeros((n, 2))
        for i in range(5):
            for j in range(5):
                centers[i * 5 + j] = [0.2 * i + 0.1, 0.2 * j + 0.1]
        
        # Placement of the 26th circle varies to diversify the search
        if restart % 3 == 0:
            centers[25] = [0.5, 0.5]
        elif restart % 3 == 1:
            centers[25] = [0.05, 0.05]
        else:
            centers[25] = [0.95, 0.95]
            
        # Add jitter to break initial symmetry and encourage exploration
        centers += np.random.uniform(-0.02, 0.02, (n, 2))
        centers = np.clip(centers, 0.0, 1.0)
        
        # Physics simulation loop
        for k in range(iters):
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-10
            
            # Pairwise repulsion between circles
            overlap = np.maximum(0, target_dist - dist)
            np.fill_diagonal(overlap, 0)
            # Magnitude proportional to overlap distance
            f_mag = overlap / dist
            forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
            
            # Repulsion from square boundaries
            r_target = target_dist / 2.0
            forces[:, 0] += np.maximum(0, r_target - centers[:, 0])
            forces[:, 0] -= np.maximum(0, centers[:, 0] - (1.0 - r_target))
            forces[:, 1] += np.maximum(0, r_target - centers[:, 1])
            forces[:, 1] -= np.maximum(0, centers[:, 1] - (1.0 - r_target))
            
            # Update center positions with a cooling schedule
            dt = dt_base * (1.0 - 0.9 * k / iters)
            centers += forces * dt
            
            # Add small noise early in the simulation to help skip local minima
            if k < iters * 0.6:
                noise_mag = 0.002 * (1.0 - k / (iters * 0.6))
                centers += np.random.normal(0, noise_mag, (n, 2))
                
            centers = np.clip(centers, 0.0, 1.0)
            
        # Compute maximized radii for these center positions
        radii = compute_max_radii(centers)
        current_sum = np.sum(radii)
        
        # Maintain the best configuration found across all restarts
        if current_sum > best_overall_sum:
            best_overall_sum = current_sum
            best_overall_centers = centers.copy()
            best_overall_radii = radii.copy()
            
    return best_overall_centers, best_overall_radii, best_overall_sum


def run_packing():
    """
    Standard entry point to run the packing algorithm for 26 circles.
    Returns: (centers, radii, sum_radii)
    """
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    # Test execution
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")