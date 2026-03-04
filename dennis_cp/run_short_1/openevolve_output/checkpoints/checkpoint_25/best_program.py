"""Constructor-based circle packing for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Construct a heavily optimized arrangement of 26 circles in a unit square
    using an Adam-based simulated physical packing expansion.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    best_sum = -1.0
    best_X = None
    best_R = None

    # Explore many distinct structural topologies alongside semi-random variants to locate optimal global minimum
    for seed in range(12):
        X, R = optimize_seed(seed, n, steps=3500)
        R_polished = polish_radii(X, R)
        current_sum = np.sum(R_polished)

        if current_sum > best_sum:
            best_sum = current_sum
            best_X = X.copy()
            best_R = R_polished.copy()

    return best_X, best_R, best_sum


def generate_seed_data(seed_id, n):
    """Deterministically output geometrically motivated configurations combined with random variances."""
    rng = np.random.RandomState(42 + seed_id)
    X = rng.uniform(0.1, 0.9, (n, 2))
    R = rng.uniform(0.01, 0.05, n)

    if seed_id == 0:
        # Concentric central focus layout promoting core symmetry breaking
        X[0] = [0.5, 0.5]
        R[0] = 0.25
        X[1:5] = [[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]]
        R[1:5] = 0.15
        for i in range(8):
            ang = i * np.pi / 4
            X[i + 5] = [0.5 + 0.25 * np.cos(ang), 0.5 + 0.25 * np.sin(ang)]
            R[i + 5] = 0.1
        for i in range(13):
            ang = i * 2 * np.pi / 13
            X[i + 13] = [0.5 + 0.42 * np.cos(ang), 0.5 + 0.42 * np.sin(ang)]
            R[i + 13] = 0.05
    elif seed_id == 1:
        # Heavily organized generic structural grid distribution mimicking typical hexagonal boundaries
        for i in range(5):
            for j in range(5):
                X[i * 5 + j] = [0.1 + i * 0.2, 0.1 + j * 0.2]
                R[i * 5 + j] = 0.08
        X[25] = [0.5, 0.5]
    elif seed_id == 2:
        # Uniform dense sunburst forcing intense compression upon expansion limits natively
        X[0] = [0.5, 0.5]
        R[0] = 0.15
        for i in range(7):
            ang = 2 * np.pi * i / 7
            X[i + 1] = [0.5 + 0.2 * np.cos(ang), 0.5 + 0.2 * np.sin(ang)]
            R[i + 1] = 0.08
        for i in range(18):
            ang = 2 * np.pi * i / 18
            X[i + 8] = [0.5 + 0.4 * np.cos(ang), 0.5 + 0.4 * np.sin(ang)]
            R[i + 8] = 0.05
    elif seed_id == 3:
        # Dense mathematical geometric layout utilizing Fibonacci nodes capturing irrational angle filling gaps
        for i in range(n):
            r_dist = 0.05 + 0.4 * (i / n)
            theta = i * np.pi * (3.0 - np.sqrt(5.0))
            X[i] = [0.5 + r_dist * np.cos(theta), 0.5 + r_dist * np.sin(theta)]
            R[i] = 0.06
    elif seed_id == 4:
        # Four intensely inflated boundaries drawing layout into strict corners
        X[0:4] = [[0.22, 0.22], [0.22, 0.78], [0.78, 0.22], [0.78, 0.78]]
        R[0:4] = 0.2
    elif seed_id == 5:
        # Two oppositely positioned massive origins generating asymmetrical squeezing across midline plane
        X[0:2] = [[0.25, 0.25], [0.75, 0.75]]
        R[0:2] = 0.24
        X[2:4] = [[0.2, 0.8], [0.8, 0.2]]
        R[2:4] = 0.15
    else:
        # Stagger variance classes intentionally resolving Voronoi jamming efficiently
        mod_strategy = seed_id % 4
        if mod_strategy == 0:
            R = rng.exponential(0.06, n)
        elif mod_strategy == 1:
            R = rng.uniform(0.01, 0.15, n)
        elif mod_strategy == 2:
            R = np.full(n, 0.05)
        else:
            R = rng.uniform(0.02, 0.08, n)
            R[0] = 0.3
            X[0] = [0.5, 0.5]

    # Quick clip to sensible starting bounds establishing smooth optimization entries natively
    R = np.clip(R, 0.005, 0.4)
    X = np.clip(X, 0.01, 0.99)
    return X, R, rng


def optimize_seed(seed_id, n, steps):
    """Execute dynamic Adam iterations managing force physics matching structural geometric barriers strictly."""
    X, R, rng = generate_seed_data(seed_id, n)

    m_X = np.zeros_like(X)
    v_X = np.zeros_like(X)
    m_R = np.zeros_like(R)
    v_R = np.zeros_like(R)

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    base_lr = 0.006

    for t in range(1, steps + 1):
        # Progressively escalate boundary repulsions matching barrier scheduling identically equivalent outer-penalty mechanisms
        lam = 5.0 * (10000.0 ** (t / steps))

        # Promote structural stabilization through correctly balanced annealing schedule smoothly settling constraints 
        if t < 100:
            lr = base_lr * (t / 100.0)
        else:
            progress = (t - 100) / (steps - 100)
            lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = max(lr, 1e-5)

        grad_X = np.zeros_like(X)
        
        # Maximize global variables enforcing native expanding radiuses directly through strict minimization pathways natively
        grad_R = np.full_like(R, -1.0) 

        # Left Boundary Check x = 0
        b1 = np.maximum(0, R - X[:, 0])
        grad_R += lam * 2 * b1
        grad_X[:, 0] -= lam * 2 * b1

        # Right Boundary Check x = 1
        b2 = np.maximum(0, X[:, 0] + R - 1.0)
        grad_R += lam * 2 * b2
        grad_X[:, 0] += lam * 2 * b2

        # Bottom Boundary Check y = 0
        b3 = np.maximum(0, R - X[:, 1])
        grad_R += lam * 2 * b3
        grad_X[:, 1] -= lam * 2 * b3

        # Top Boundary Check y = 1
        b4 = np.maximum(0, X[:, 1] + R - 1.0)
        grad_R += lam * 2 * b4
        grad_X[:, 1] += lam * 2 * b4

        # Complete pair overlaps correctly vectorized matching distances strictly representing native N-body constraints effectively
        dx = X[:, 0].reshape(-1, 1) - X[:, 0]
        dy = X[:, 1].reshape(-1, 1) - X[:, 1]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1.0)

        R_sum = R.reshape(-1, 1) + R
        overlap = np.triu(R_sum - dist, 1)
        o_vals = np.maximum(0, overlap)

        if np.any(o_vals > 0):
            # Sum symmetrically capturing full structural node derivatives 
            grad_R += 2 * lam * (np.sum(o_vals, axis=1) + np.sum(o_vals, axis=0))

            mask = o_vals > 0
            d_dist = np.zeros_like(dist)
            d_dist[mask] = -2 * lam * o_vals[mask] / dist[mask]

            gx_matrix = d_dist * dx
            gy_matrix = d_dist * dy

            # Maintain correct symmetric mappings guaranteeing precise displacement calculations perfectly opposing overlapping barriers strictly
            grad_X[:, 0] += np.sum(gx_matrix, axis=1) - np.sum(gx_matrix, axis=0)
            grad_X[:, 1] += np.sum(gy_matrix, axis=1) - np.sum(gy_matrix, axis=0)

        # Incorporate Momentum
        b1_t = 1 - beta1**t
        b2_t = 1 - beta2**t

        m_X = beta1 * m_X + (1 - beta1) * grad_X
        v_X = beta2 * v_X + (1 - beta2) * (grad_X**2)
        X -= lr * (m_X / b1_t) / (np.sqrt(v_X / b2_t) + epsilon)

        m_R = beta1 * m_R + (1 - beta1) * grad_R
        v_R = beta2 * v_R + (1 - beta2) * (grad_R**2)
        R -= lr * (m_R / b1_t) / (np.sqrt(v_R / b2_t) + epsilon)

        R = np.clip(R, 0.005, 1.0)
        X = np.clip(X, 0.0, 1.0)

        # Apply structural spatial shaking resolving isolated jammed formations heavily stuck preventing Voronoi escape precisely  
        if t % 150 == 0 and t < steps * 0.7:
            noise_scale = 0.005 * (1.0 - t / (steps * 0.7))
            X += rng.randn(n, 2) * noise_scale
            X = np.clip(X, 0.01, 0.99)

    return X, R


def polish_radii(X, R):
    """
    Ensure mathematical precision guarantees removing microscopic collisions natively natively eliminating bounds problems natively.
    Re-inflates structural vacuums sequentially capturing maximum possible geometrical spaces optimally.
    """
    N = X.shape[0]
    R_valid = R.copy()

    # Step down structural constraint margins capturing correct bounds
    for i in range(N):
        x, y = X[i]
        R_valid[i] = min(R_valid[i], x, y, 1.0 - x, 1.0 - y)

    # Completely remove all remaining structural floating errors rapidly utilizing guaranteed scaling shrink cascades seamlessly 
    for _ in range(500):
        violation = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i, 0] - X[j, 0]
                dy = X[i, 1] - X[j, 1]
                dist = np.sqrt(dx**2 + dy**2)

                if R_valid[i] + R_valid[j] > dist + 1e-12:
                    scale = dist / (R_valid[i] + R_valid[j])
                    R_valid[i] *= (scale * 0.999999)
                    R_valid[j] *= (scale * 0.999999)
                    violation = True
        if not violation:
            break

    # Extract optimally maximal values across completely resolved strict mappings dynamically increasing radius where margin gaps safely accommodate natively
    for _ in range(15):
        expanded_any = False
        for i in range(N):
            max_r = min(X[i, 0], X[i, 1], 1.0 - X[i, 0], 1.0 - X[i, 1])
            for j in range(N):
                if i != j:
                    dist = np.sqrt((X[i, 0] - X[j, 0])**2 + (X[i, 1] - X[j, 1])**2)
                    max_r = min(max_r, dist - R_valid[j])
            
            # Sequentially implement size recovery guaranteeing perfectly zero collisions while raising cumulative radii cleanly natively
            if max_r > R_valid[i] + 1e-10:
                R_valid[i] = max_r
                expanded_any = True
        
        if not expanded_any:
            break

    return R_valid


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