# EVOLVE-BLOCK-START
"""
Multi-start, batched Adam optimization-based circle packing for n=26 circles.
Runs several highly parallel starting layouts leveraging identical physics to gracefully 
dodge local geometric minima, producing mathematical precision flawlessly seamlessly.
"""
import numpy as np


def generate_initial_layouts():
    """
    Produce 12 varying initial symmetry configurations to span possible topologies.
    """
    B, N = 12, 26
    centers = np.zeros((B, N, 2))

    # 0, 1: Basic grid slight offsets
    for b in [0, 1]:
        idx = 0
        for i, count in enumerate([5, 5, 6, 5, 5]):
            y = (i + 0.5) / 5.0
            for j in range(count):
                x = (j + 0.5) / count
                centers[b, idx] = [x, y]
                idx += 1

    # 2, 3: Pure symmetric hexagonal layout estimation
    for b in [2, 3]:
        idx = 0
        for i, count in enumerate([5, 6, 4, 6, 5]):
            y = (i + 0.5) / 5.0
            for j in range(count):
                x = (j + 0.5) / count
                centers[b, idx] = [x, y]
                idx += 1

    # 4, 5: Concentric rings targeting heavy dense packings 
    for b in [4, 5]:
        centers[b, 0] = [0.5, 0.5]
        idx = 1
        for count, r in [(6, 0.18), (11, 0.33), (8, 0.44)]:
            for j in range(count):
                phase_offset = 0.5 if idx % 2 == 0 else 0
                angle = 2 * np.pi * (j + phase_offset) / count
                centers[b, idx] = [0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)]
                idx += 1

    # 6, 7: Sunflower seed spiral distribution
    for b in [6, 7]:
        phi = (1 + np.sqrt(5)) / 2
        for i in range(N):
            r = 0.45 * np.sqrt((i + 0.5) / N)
            theta = 2 * np.pi * i * phi
            centers[b, i] = [0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)]

    # 8, 9: Flat brick stacking model 
    for b in [8, 9]:
        idx = 0
        for i, count in enumerate([8, 10, 8]):
            y = (i + 0.5) / 3.0
            for j in range(count):
                x = (j + 0.5) / count
                centers[b, idx] = [x, y]
                idx += 1

    # 10, 11: Constrained tight hexagonal edge boundary limits 
    for b in [10, 11]:
        idx = 0
        for i, count in enumerate([4, 6, 6, 6, 4]):
            y = (i + 0.5) / 5.0
            for j in range(count):
                x = (j + 0.5) / count
                centers[b, idx] = [x, y]
                idx += 1

    return centers


def finalize_and_select(batched_centers, batched_radii):
    """
    Mathematically shrinks valid collisions completely and sequentially for precision.
    Scans entire batched operations selecting explicitly the top yielding configuration.
    """
    best_sum = -1.0
    best_c = None
    best_r = None
    B, N = batched_centers.shape[:2]

    for b in range(B):
        C = batched_centers[b].copy()
        R = batched_radii[b].copy()

        # Step limits inside square exact boundaries precisely
        for i in range(N):
            C[i, 0] = np.clip(C[i, 0], 1e-4, 1.0 - 1e-4)
            C[i, 1] = np.clip(C[i, 1], 1e-4, 1.0 - 1e-4)
            x, y = C[i]
            R[i] = min(R[i], x, y, 1.0 - x, 1.0 - y)

        # Iterative proportional sequential resolving guaranteeing zero overlaps formed backward 
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((C[i] - C[j]) ** 2))
                if R[i] + R[j] > dist:
                    scale = dist / (R[i] + R[j])
                    R[i] *= scale
                    R[j] *= scale

        S = np.sum(R)
        if S > best_sum:
            best_sum = S
            best_c = C
            best_r = R

    # Absolute safe floating point multiplication guaranteeing zero boundary strict evaluations logic triggers
    best_r *= 0.9999999

    return best_c, best_r, np.sum(best_r)


def construct_packing():
    """
    Primary executor dispatching robust multi-stream Adam optimizations.
    """
    B, N = 12, 26
    centers = generate_initial_layouts()

    # Broad seed distribution handling geometric uniformity offsets successfully
    np.random.seed(1337)
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.02, 0.98)

    # Initialize diverse range radii
    radii = np.ones((B, N)) * 0.01
    radii += np.random.uniform(0.002, 0.025, size=radii.shape)

    steps = 10000
    lr_c_start, lr_c_end = 0.008, 0.0001
    lr_r_start, lr_r_end = 0.004, 0.00005

    m_c, v_c = np.zeros_like(centers), np.zeros_like(centers)
    m_r, v_r = np.zeros_like(radii), np.zeros_like(radii)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    idx_diag = np.arange(N)

    for step in range(1, steps + 1):
        progress = step / steps
        lr_c = lr_c_start * ((lr_c_end / lr_c_start) ** progress)
        lr_r = lr_r_start * ((lr_r_end / lr_r_start) ** progress)

        # Scale penalty exponentially resolving boundary overlaps fully explicitly completely 
        w = 10.0 * (3000.0 ** progress)

        # Batch accelerated spatial calculations 
        diff = centers[:, :, None, :] - centers[:, None, :, :]
        d2 = np.sum(diff ** 2, axis=3)
        d2[:, idx_diag, idx_diag] = 1.0
        d = np.sqrt(d2)
        d[:, idx_diag, idx_diag] = np.inf

        R_sum = radii[:, :, None] + radii[:, None, :]
        O = np.maximum(0, R_sum - d)

        bx_min = np.maximum(0, radii - centers[:, :, 0])
        bx_max = np.maximum(0, radii + centers[:, :, 0] - 1.0)
        by_min = np.maximum(0, radii - centers[:, :, 1])
        by_max = np.maximum(0, radii + centers[:, :, 1] - 1.0)

        # Evaluation derivation constraints explicitly strictly 
        grad_r = -1.0 + w * (np.sum(O, axis=2) + bx_min + bx_max + by_min + by_max)

        dir_vec = diff / (d[:, :, :, None] + eps)
        grad_c = np.zeros_like(centers)
        grad_c += w * np.sum(O[:, :, :, None] * (-dir_vec), axis=2)
        grad_c[:, :, 0] += w * (bx_max - bx_min)
        grad_c[:, :, 1] += w * (by_max - by_min)

        # Noise decay ensuring positional structure settles efficiently securely properly nicely 
        if progress < 0.3:
            scale = (0.3 - progress)
            grad_c += np.random.normal(0, 0.05 * scale, size=centers.shape)
            grad_r += np.random.normal(0, 0.02 * scale, size=radii.shape)

        m_c = beta1 * m_c + (1 - beta1) * grad_c
        v_c = beta2 * v_c + (1 - beta2) * (grad_c ** 2)
        m_c_hat = m_c / (1 - beta1 ** step)
        v_c_hat = v_c / (1 - beta2 ** step)
        centers -= lr_c * m_c_hat / (np.sqrt(v_c_hat) + eps)

        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r ** 2)
        m_r_hat = m_r / (1 - beta1 ** step)
        v_r_hat = v_r / (1 - beta2 ** step)
        radii -= lr_r * m_r_hat / (np.sqrt(v_r_hat) + eps)

        radii = np.maximum(0.001, radii)
        centers = np.clip(centers, 0.001, 0.999)

    return finalize_and_select(centers, radii)


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