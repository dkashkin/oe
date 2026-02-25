# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    that dynamically finds optimal placements to maximize the sum of radii.
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    centers, radii, sum_radii = solve_multi_start()
    return centers, radii, sum_radii


def solve_multi_start():
    best_sum = -1.0
    best_X = None
    best_R = None

    # Explore 6 distinct seeded starts to bypass poor local minima
    for seed in range(42, 48):
        X_opt, R_opt = single_optimization(seed=seed)
        X_fixed, R_fixed = resolve_constraints(X_opt, R_opt)

        current_sum = np.sum(R_fixed)
        if current_sum > best_sum:
            best_sum = current_sum
            best_X = X_fixed.copy()
            best_R = R_fixed.copy()

    return best_X, best_R, best_sum


def single_optimization(seed):
    np.random.seed(seed)
    n = 26
    X = np.zeros((n, 2))

    # Geometrical seeding
    # Center core starts largest
    X[0] = [0.5, 0.5]

    idx = 1
    corners = [[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]]
    for p in corners:
        X[idx] = p
        idx += 1

    edges = [[0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
    for p in edges:
        X[idx] = p
        idx += 1

    ring1 = [[0.3, 0.3], [0.3, 0.7], [0.7, 0.3], [0.7, 0.7]]
    for p in ring1:
        if idx < n:
            X[idx] = p
            idx += 1

    while idx < n:
        X[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
        idx += 1

    # Apply spatial jitter to evade perfectly symmetric standstills
    X += np.random.normal(0, 0.02, size=(n, 2))
    X = np.clip(X, 0.05, 0.95)

    R = np.random.uniform(0.04, 0.08, size=(n,))
    R[0] = 0.15

    lr_initial_X = 0.01
    lr_initial_R = 0.005
    beta1 = 0.9
    beta2 = 0.999

    m_X = np.zeros_like(X)
    v_X = np.zeros_like(X)
    m_R = np.zeros_like(R)
    v_R = np.zeros_like(R)

    max_iter = 3500

    for t in range(1, max_iter + 1):
        progress = t / max_iter
        W_bound = 10.0 + 4000.0 * (progress ** 2)
        W_overlap = W_bound
        W_neg = W_bound

        # Annealing kicks to reshuffle blockages roughly
        if t % 500 == 0 and progress < 0.6:
            X += np.random.normal(0, 0.003, size=(n, 2))

        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D2 = np.sum(diff ** 2, axis=-1)
        D = np.sqrt(D2 + 1e-12)
        np.fill_diagonal(D, np.inf)

        R_sum = R[:, np.newaxis] + R[np.newaxis, :]
        V_overlap = np.maximum(0, R_sum - D)
        np.fill_diagonal(V_overlap, 0)

        grad_D_wrt_X = diff / D[..., np.newaxis]
        grad_X_overlap = -2.0 * np.sum(V_overlap[..., np.newaxis] * grad_D_wrt_X, axis=1)
        grad_R_overlap = 2.0 * np.sum(V_overlap, axis=1)

        v_L = np.maximum(0, R - X[:, 0])
        v_R_bound = np.maximum(0, R + X[:, 0] - 1.0)
        v_B = np.maximum(0, R - X[:, 1])
        v_T = np.maximum(0, R + X[:, 1] - 1.0)

        grad_X_bounds = np.zeros_like(X)
        grad_X_bounds[:, 0] = W_bound * (-2.0 * v_L + 2.0 * v_R_bound)
        grad_X_bounds[:, 1] = W_bound * (-2.0 * v_B + 2.0 * v_T)
        grad_R_bounds = W_bound * 2.0 * (v_L + v_R_bound + v_B + v_T)

        v_neg = np.maximum(0, -R)
        grad_R_neg = W_neg * (-2.0 * v_neg)

        grad_X = grad_X_bounds + W_overlap * grad_X_overlap
        # Standardize -1 as the target descent driver to constantly attempt scaling R radially upwards
        grad_R = -1.0 + grad_R_bounds + W_overlap * grad_R_overlap + grad_R_neg

        m_X = beta1 * m_X + (1 - beta1) * grad_X
        v_X = beta2 * v_X + (1 - beta2) * (grad_X ** 2)
        m_R = beta1 * m_R + (1 - beta1) * grad_R
        v_R = beta2 * v_R + (1 - beta2) * (grad_R ** 2)

        m_X_hat = m_X / (1 - beta1 ** t)
        v_X_hat = v_X / (1 - beta2 ** t)
        m_R_hat = m_R / (1 - beta1 ** t)
        v_R_hat = v_R / (1 - beta2 ** t)

        lr_factor = (1.0 - progress) ** 0.5
        X = X - lr_initial_X * lr_factor * m_X_hat / (np.sqrt(v_X_hat) + 1e-8)
        R = R - lr_initial_R * lr_factor * m_R_hat / (np.sqrt(v_R_hat) + 1e-8)

        # Retain circles reasonably in-range through iterations purely to avoid overflow drifts
        X = np.clip(X, 1e-4, 1.0 - 1e-4)

    return X, R


def resolve_constraints(X, R_guess):
    n = len(X)
    try:
        from scipy.optimize import linprog
        c = -np.ones(n)
        bounds = []
        for i in range(n):
            limit = max(0.0, min([X[i, 0], 1.0 - X[i, 0], X[i, 1], 1.0 - X[i, 1]]))
            bounds.append((0, limit))

        A_ub = []
        b_ub = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(X[i] - X[j])
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(max(0.0, dist - 1e-10))

        if A_ub:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                R_opt = res.x.copy()

                for i in range(n):
                    limit = max(0.0, min([X[i, 0], 1.0 - X[i, 0], X[i, 1], 1.0 - X[i, 1]]))
                    R_opt[i] = min(R_opt[i], limit)

                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.linalg.norm(X[i] - X[j])
                        if R_opt[i] + R_opt[j] > dist:
                            overlap = R_opt[i] + R_opt[j] - dist + 1e-10
                            R_opt[i] = max(0.0, R_opt[i] - overlap / 2.0)
                            R_opt[j] = max(0.0, R_opt[j] - overlap / 2.0)
                return X, R_opt
    except Exception:
        pass

    return X, fallback_resolve(X, R_guess)


def fallback_resolve(X, R):
    R = R.copy()
    n = len(R)

    for i in range(n):
        limit = max(0.0, min([X[i, 0], 1.0 - X[i, 0], X[i, 1], 1.0 - X[i, 1]]))
        R[i] = min(max(0.0, R[i]), limit)

    for _ in range(3000):
        changed = False

        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D2 = np.sum(diff ** 2, axis=-1)
        D = np.sqrt(D2)
        np.fill_diagonal(D, np.inf)

        R_sum = R[:, np.newaxis] + R[np.newaxis, :]
        V_overlap = R_sum - D

        if np.max(V_overlap) <= 1e-10:
            break

        i_idx, j_idx = np.where(V_overlap > 1e-10)
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            if i < j:
                dist = D[i, j]
                overlap = R[i] + R[j] - dist + 1e-9
                if overlap > 0:
                    R[i] = max(0.0, R[i] - overlap / 2.0)
                    R[j] = max(0.0, R[j] - overlap / 2.0)
                    changed = True

        for i in range(n):
            limit = max(0.0, min([X[i, 0], 1.0 - X[i, 0], X[i, 1], 1.0 - X[i, 1]]))
            R[i] = min(R[i], limit)

        if not changed:
            break

    # Hard defensive scan
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(X[i] - X[j])
            if R[i] + R[j] > dist:
                overlap = R[i] + R[j] - dist + 1e-9
                R[i] = max(0.0, R[i] - overlap / 2.0)
                R[j] = max(0.0, R[j] - overlap / 2.0)

    for i in range(n):
        limit = max(0.0, min([X[i, 0], 1.0 - X[i, 0], X[i, 1], 1.0 - X[i, 1]]))
        R[i] = min(R[i], limit)

    return R

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