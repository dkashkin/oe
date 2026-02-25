# EVOLVE-BLOCK-START
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

    # Try different topological seeds to find best global optimum
    for seed in range(5):
        X, R = optimize_seed(seed, n, steps=3500)
        R_valid = make_strictly_valid(X, R)
        current_sum = np.sum(R_valid)

        if current_sum > best_sum:
            best_sum = current_sum
            best_X = X.copy()
            best_R = R_valid.copy()

    return best_X, best_R, best_sum


def optimize_seed(seed_id, n, steps):
    """Run an optimization using Adam on position and radius variables."""
    np.random.seed(42 + seed_id)
    X = np.random.uniform(0.1, 0.9, (n, 2))
    R = np.random.uniform(0.01, 0.05, n)

    if seed_id == 0:
        # Concentric layout focusing large circle heavily towards center
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
        # 5x5 Grid loosely filled with center focus
        for i in range(5):
            for j in range(5):
                X[i * 5 + j] = [0.1 + i * 0.2, 0.1 + j * 0.2]
                R[i * 5 + j] = 0.08
        X[25] = [0.5, 0.5]
    elif seed_id == 2:
        # Dense sunburst packing strategy
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
        # Fibonacci spiral packing for maximum density utilization
        for i in range(n):
            r_dist = 0.05 + 0.4 * (i / n)
            theta = i * np.pi * (3.0 - np.sqrt(5.0))
            X[i] = [0.5 + r_dist * np.cos(theta), 0.5 + r_dist * np.sin(theta)]
            R[i] = 0.06

    # Apply positional jitter to strictly break axis alignments (escapes trapping geometry local minima)
    if seed_id != 4:
        X += np.random.normal(0, 0.005, size=X.shape)

    X = np.clip(X, 0.01, 0.99)

    m_X = np.zeros_like(X)
    v_X = np.zeros_like(X)
    m_R = np.zeros_like(R)
    v_R = np.zeros_like(R)

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    base_lr = 0.005

    for t in range(1, steps + 1):
        # Schedule the constraint logic continuously up like a rigorous exterior penalty path method
        lam = 5.0 * (10000.0 ** (t / steps))

        # Cosine annealed learning rate heavily promotes settling down firmly into stable packed states
        if t < 100:
            lr = base_lr * (t / 100.0)
        else:
            progress = (t - 100) / (steps - 100)
            lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = max(lr, 1e-5)

        grad_X = np.zeros_like(X)
        grad_R = np.zeros_like(R)

        # Continually instruct variables mapping circle radiuses to expand
        grad_R += -1.0

        # Boundary checks against four edges, creating firm resistance fields ensuring internal fitting
        b1 = np.maximum(0, R - X[:, 0])
        grad_R += lam * 2 * b1
        grad_X[:, 0] -= lam * 2 * b1

        b2 = np.maximum(0, X[:, 0] + R - 1.0)
        grad_R += lam * 2 * b2
        grad_X[:, 0] += lam * 2 * b2

        b3 = np.maximum(0, R - X[:, 1])
        grad_R += lam * 2 * b3
        grad_X[:, 1] -= lam * 2 * b3

        b4 = np.maximum(0, X[:, 1] + R - 1.0)
        grad_R += lam * 2 * b4
        grad_X[:, 1] += lam * 2 * b4

        # Map complete N-body geometric interference and deduce optimal repulsion separation force structures
        dx = X[:, 0].reshape(-1, 1) - X[:, 0]
        dy = X[:, 1].reshape(-1, 1) - X[:, 1]
        dist = np.sqrt(dx**2 + dy**2 + 1e-16)
        np.fill_diagonal(dist, 1.0)

        R_sum = R.reshape(-1, 1) + R
        overlap = np.triu(R_sum - dist, 1)
        mask = overlap > 0

        if np.any(mask):
            o_vals = np.zeros_like(overlap)
            o_vals[mask] = overlap[mask]

            # Collect symmetric radius forces reflecting intersection limits natively
            grad_R += 2 * lam * (np.sum(o_vals, axis=1) + np.sum(o_vals, axis=0))

            # Distribute correctly signed vector displacement adjustments tracking Euclidean pathways exactly
            d_dist = np.zeros_like(dist)
            d_dist[mask] = -2 * lam * o_vals[mask] / dist[mask]

            gx_matrix = d_dist * dx
            gy_matrix = d_dist * dy

            grad_X[:, 0] += np.sum(gx_matrix, axis=1) - np.sum(gx_matrix, axis=0)
            grad_X[:, 1] += np.sum(gy_matrix, axis=1) - np.sum(gy_matrix, axis=0)

        # Integrate parameter derivatives via full momentum caching structures
        m_X = beta1 * m_X + (1 - beta1) * grad_X
        v_X = beta2 * v_X + (1 - beta2) * (grad_X**2)
        X -= lr * (m_X / (1 - beta1**t)) / (np.sqrt(v_X / (1 - beta2**t)) + epsilon)

        m_R = beta1 * m_R + (1 - beta1) * grad_R
        v_R = beta2 * v_R + (1 - beta2) * (grad_R**2)
        R -= lr * (m_R / (1 - beta1**t)) / (np.sqrt(v_R / (1 - beta2**t)) + epsilon)

        # Establish sensible numerical floors
        R = np.clip(R, 0.005, 1.0)
        X = np.clip(X, 0.0, 1.0)

    return X, R


def make_strictly_valid(X, R):
    """
    Ensure absolutely zero remaining overlaps mathematically via rapid and tiny exact shrinkage.
    The optimized positions will barely move meaning fitness score practically identical to Adam output.
    """
    N = X.shape[0]
    R_valid = R.copy()

    # Step down edges first strictly satisfying bounds 
    for i in range(N):
        x, y = X[i]
        R_valid[i] = min(R_valid[i], x, y, 1.0 - x, 1.0 - y)

    # Reconcile pair connections guaranteeing correct minimum clearances explicitly preventing float glitches
    for _ in range(500):
        violation = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i, 0] - X[j, 0]
                dy = X[i, 1] - X[j, 1]
                dist = np.sqrt(dx**2 + dy**2)

                if R_valid[i] + R_valid[j] > dist + 1e-12:
                    # Provide an immediate safety jump explicitly removing endless mathematical looping traps
                    scale = dist / (R_valid[i] + R_valid[j])
                    R_valid[i] *= (scale * 0.999999)
                    R_valid[j] *= (scale * 0.999999)
                    violation = True
        if not violation:
            break

    return R_valid
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