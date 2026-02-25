# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


class CircleOptimizer:
    """
    Iterative physics-based optimization algorithm to find an extremely tight
    arrangement of n circles in a 1x1 unit square to maximize the sum of radii.
    """

    def __init__(self, n=26):
        self.n = n
        
        # State vector [x, y, radius]
        self.X = np.zeros((n, 3))
        
        # 1. Strategic seeding with size bias:
        # Placement by size heuristic expects a larger circle around the center
        # and subsequent rings distributed appropriately filling toward boundaries.
        self.X[0, 0:2] = [0.5, 0.5]
        self.X[0, 2] = 0.2
        
        # Inner Ring
        for i in range(8):
            angle = 2 * np.pi * i / 8
            self.X[i + 1, 0] = 0.5 + 0.25 * np.cos(angle)
            self.X[i + 1, 1] = 0.5 + 0.25 * np.sin(angle)
            self.X[i + 1, 2] = 0.1
            
        # Outer Ring biased towards the corners and edges to maximize space utility
        for i in range(17):
            angle = 2 * np.pi * i / 17
            # Denominator projects uniform polar spacing outwards into a square box mapping
            denominator = max(max(abs(np.cos(angle)), abs(np.sin(angle))), 1e-5)
            r_dist = 0.45 / denominator
            r_dist = min(r_dist, 0.46)
            self.X[i + 9, 0] = 0.5 + r_dist * np.cos(angle)
            self.X[i + 9, 1] = 0.5 + r_dist * np.sin(angle)
            self.X[i + 9, 2] = 0.08
            
        # 2. Break perfect symmetry:
        # Introduction of structured but randomized initial states breaks the saddle 
        # point guarantees, avoiding premature saturation against bounding box edges.
        np.random.seed(42)  # Maintain deterministic path, perfectly escaping geometry ties
        self.X[:, 0:2] += np.random.uniform(-0.015, 0.015, size=(n, 2))
        
        # Initialize safe positions safely in frame
        self.X[:, 0:2] = np.clip(self.X[:, 0:2], 0.05, 0.95)
        
        self.lr = 0.03

    def optimize(self):
        """
        Executes gradient ascent on summed radii constrained by squared error penalties,
        wrapped entirely in an Adaptive Moment Estimation (Adam) vector scheme heavily
        laced with Langevin noise simulation to provide Simulated Annealing.
        """
        m = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        epochs = 10000
        lambda_reg = 5.0
        
        # Phase constraints - penalty factors increase dramatically to lock in coordinates
        for t in range(1, epochs + 1):
            if t == 2000:
                lambda_reg = 20.0
            elif t == 4000:
                lambda_reg = 100.0
            elif t == 6000:
                lambda_reg = 500.0
            elif t == 8000:
                lambda_reg = 2500.0
            elif t == 9000:
                lambda_reg = 10000.0
            
            # Smooth exponential decay schedule ensures convergence stability
            lr = self.lr * (0.9996 ** t)
            
            pos = self.X[:, 0:2]
            r = self.X[:, 2]
            
            # 3. Formulate loss structure and derivatives.
            # Objective grad: moving backwards minimizing E equals growing overall r 
            grad = np.zeros_like(self.X)
            grad[:, 2] = -1.0
            
            # Calculate gradient contributions ensuring points safely rest inside geometry bounding box
            err_L = np.maximum(0, r - pos[:, 0])
            grad[:, 0] -= lambda_reg * 2 * err_L
            grad[:, 2] += lambda_reg * 2 * err_L
            
            err_R = np.maximum(0, r + pos[:, 0] - 1)
            grad[:, 0] += lambda_reg * 2 * err_R
            grad[:, 2] += lambda_reg * 2 * err_R
            
            err_B = np.maximum(0, r - pos[:, 1])
            grad[:, 1] -= lambda_reg * 2 * err_B
            grad[:, 2] += lambda_reg * 2 * err_B
            
            err_T = np.maximum(0, r + pos[:, 1] - 1)
            grad[:, 1] += lambda_reg * 2 * err_T
            grad[:, 2] += lambda_reg * 2 * err_T
            
            # Rapid vectorized intersection test to resolve circle overlapping limits
            pos_exp_1 = pos[:, np.newaxis, :]
            pos_exp_2 = pos[np.newaxis, :, :]
            dpos = pos_exp_1 - pos_exp_2
            
            dist2 = np.sum(dpos**2, axis=-1)
            np.fill_diagonal(dist2, 1.0)  # Stop zeroes from breaking sqrt
            dist = np.sqrt(dist2 + 1e-12)
            
            req_dist = r[:, np.newaxis] + r[np.newaxis, :]
            err_ov = np.maximum(0, req_dist - dist)
            np.fill_diagonal(err_ov, 0)
            
            grad[:, 2] += lambda_reg * 2.0 * np.sum(err_ov, axis=1)
            
            factor = lambda_reg * 2.0 * err_ov / dist
            grad[:, 0] -= np.sum(factor * dpos[..., 0], axis=1)
            grad[:, 1] -= np.sum(factor * dpos[..., 1], axis=1)
            
            # Commit adjustments via standard momentum
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # 4. Integrate physical cooling
            # Simulate decaying Brownian momentum resolving structural dead-locks cleanly
            noise_scale = 0.002 * (0.9995 ** t)
            noise = np.random.normal(0, noise_scale, size=self.X.shape)
            noise[:, 2] *= 0.1  # Stabilize radius, target physical offsets strictly
            
            self.X -= lr * m_hat / (np.sqrt(v_hat) + eps) + noise
            self.X[:, 2] = np.clip(self.X[:, 2], 0.001, 1.0)
            self.X[:, 0:2] = np.clip(self.X[:, 0:2], 0.001, 0.999)

        # Force mathematically uncompromising limits on output formats and radii sets
        centers = self.X[:, 0:2]
        radii = self.X[:, 2]
        radii = self.enforce_validity(centers, radii)
        
        return centers, radii, np.sum(radii)

    def enforce_validity(self, centers, radii):
        """Mathematically shrink sizes eliminating even float collisions securely"""
        radii = np.copy(radii)
        for i in range(self.n):
            radii[i] = min(
                radii[i],
                centers[i, 0],
                1 - centers[i, 0],
                centers[i, 1],
                1 - centers[i, 1]
            )
            
        # Shrink proportionally to absolute boundaries
        for _ in range(200):
            overlap_found = False
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if radii[i] + radii[j] > dist:
                        overlap_found = True
                        scale = dist / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale
            if not overlap_found:
                break
                
        # Sub-scale cleanly resolving evaluation pipeline checks limits
        radii *= 0.99999
        return radii


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    maximizing sum radius capacity through dense iteration strategies.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    optimizer = CircleOptimizer(n=26)
    centers, radii, sum_radii = optimizer.optimize()
    
    return centers, radii, sum_radii

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