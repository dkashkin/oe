# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


class AdamOptimizer:
    """A standard Adam optimizer for gradient descent over parameters."""
    def __init__(self, shape_X, shape_R, lr=0.01):
        self.lr = lr
        self.m_X = np.zeros(shape_X)
        self.v_X = np.zeros(shape_X)
        self.m_R = np.zeros(shape_R)
        self.v_R = np.zeros(shape_R)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        
    def step(self, X, R, grad_X, grad_R):
        self.t += 1
        
        self.m_X = self.beta1 * self.m_X + (1 - self.beta1) * grad_X
        self.v_X = self.beta2 * self.v_X + (1 - self.beta2) * (grad_X**2)
        m_X_hat = self.m_X / (1 - self.beta1**self.t)
        v_X_hat = self.v_X / (1 - self.beta2**self.t)
        X_new = X - self.lr * m_X_hat / (np.sqrt(v_X_hat) + self.eps)
        
        self.m_R = self.beta1 * self.m_R + (1 - self.beta1) * grad_R
        self.v_R = self.beta2 * self.v_R + (1 - self.beta2) * (grad_R**2)
        m_R_hat = self.m_R / (1 - self.beta1**self.t)
        v_R_hat = self.v_R / (1 - self.beta2**self.t)
        R_new = R - self.lr * m_R_hat / (np.sqrt(v_R_hat) + self.eps)
        
        return X_new, R_new


def make_valid(X, R):
    """Ensure fully rigid geometric compliance by resolving overlaps, bounding dynamically,
    and pushing coordinates precisely towards mathematical kissing constraints to inflate bounds.
    """
    X = np.clip(X, 0.0, 1.0)
    R = np.clip(R, 0.0, 1.0)
    R = np.minimum.reduce([R, X[:, 0], 1.0 - X[:, 0], X[:, 1], 1.0 - X[:, 1]])
    n = len(R)
    
    # Resolving intersecting structural bounds loop safely geometrically
    for _ in range(4000):
        violation = False
        dx = X[:, 0].reshape(-1, 1) - X[:, 0]
        dy = X[:, 1].reshape(-1, 1) - X[:, 1]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        sum_R = R.reshape(-1, 1) + R
        overlap = sum_R - dist
        if np.max(overlap) > 1e-12:
            violation = True
            for i in range(n):
                for j in range(i + 1, n):
                    if overlap[i, j] > 1e-12:
                        scale = dist[i, j] / (R[i] + R[j] + 1e-16)
                        R[i] *= scale
                        R[j] *= scale
        if not violation:
            break
            
    # Iterative aggressive local block-coordinate space nudging & optimization
    # Expands inner spaces by sliding circles mathematically along multiple constraint interfaces
    for pass_idx in range(80):
        order = np.random.permutation(n)
        expanded = False
        
        for i in order:
            pos = X[i].copy()
            lr = 0.02 * (0.95 ** pass_idx)
            best_pos = pos.copy()
            best_r = R[i]
            
            # Active microscopic coordinate drift gradient against closest topological barriers
            for _ in range(15):
                dx = pos[0] - X[:, 0]
                dy = pos[1] - X[:, 1]
                dists = np.sqrt(dx**2 + dy**2)
                dists[i] = np.inf
                
                m_circ = dists - R
                m_bnd = np.array([pos[0], 1.0 - pos[0], pos[1], 1.0 - pos[1]])
                min_m = min(np.min(m_circ), np.min(m_bnd))
                
                if min_m > best_r:
                    best_r = min_m
                    best_pos = pos.copy()
                
                # Temperature based vector formulation directing into deepest space pools safely
                u_circ_x = dx / (dists + 1e-12)
                u_circ_y = dy / (dists + 1e-12)
                
                w_circ = np.exp(-200.0 * np.maximum(0, m_circ - min_m))
                w_circ[i] = 0.0
                
                fx = np.sum(u_circ_x * w_circ)
                fy = np.sum(u_circ_y * w_circ)
                
                w_bnd = np.exp(-200.0 * np.maximum(0, m_bnd - min_m))
                fx += w_bnd[0] * 1.0 - w_bnd[1] * 1.0
                fy += w_bnd[2] * 1.0 - w_bnd[3] * 1.0
                
                norm = np.sqrt(fx**2 + fy**2)
                if norm > 1e-12:
                    pos[0] += lr * fx / norm
                    pos[1] += lr * fy / norm
                    pos = np.clip(pos, 0.0, 1.0)
                else:
                    break

            # Confirm and capture safe bounds limits improvements continuously
            if best_r > R[i] + 1e-9:
                X[i] = best_pos
                R[i] = best_r
                expanded = True
                
        if not expanded and pass_idx > 10:
            break
            
    R = np.minimum.reduce([R, X[:, 0], 1.0 - X[:, 0], X[:, 1], 1.0 - X[:, 1]])
    R = np.maximum(R, 0.0)
    return X, R


def solve_packing(n=26, iterations=7000, restarts=12, lr_start=0.015):
    """Execute dynamic penalty-scaled Adam physics with targeted varied structured seeds."""
    best_X = None
    best_R = None
    best_score = -1

    for restart in range(restarts):
        np.random.seed(1337 + restart)
        
        # Heterogeneous structured parameter placements balancing varied densities intelligently
        if restart < 3:
            X = np.random.rand(n, 2) * 0.9 + 0.05
            R = np.random.rand(n) * 0.05 + 0.01
            
        elif restart < 6:
            X = np.zeros((n, 2))
            if n > 0: X[0] = [0.5, 0.5]
            n_inner = min((n - 1) // 3, 8) if n > 1 else 0
            n_outer = max(0, n - 1 - n_inner)
            
            for i in range(n_inner):
                angle = 2 * np.pi * i / max(1, n_inner) + np.random.randn() * 0.1
                X[i+1] = [0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)]
            for i in range(n_outer):
                angle = 2 * np.pi * i / max(1, n_outer) + np.random.randn() * 0.1
                X[i + 1 + n_inner] = [0.5 + 0.45 * np.cos(angle), 0.5 + 0.45 * np.sin(angle)]
                
            X = np.clip(X, 0.05, 0.95)
            R = np.ones(n) * 0.05
            if n > 0: R[0] = 0.1
            
        elif restart < 9:
            X = np.random.rand(n, 2) * 0.8 + 0.1
            R = np.random.rand(n) * 0.04 + 0.01
            max_c = min(n, 4)
            bases = [[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]]
            for i in range(max_c):
                X[i] = bases[i]
                R[i] = 0.15
            if n > 4:
                X[4] = [0.5, 0.5]
                R[4] = 0.15
                
        else:
            X = np.random.rand(n, 2)
            edge_x_mask = np.random.rand(n) > 0.5
            edge_y_mask = np.random.rand(n) > 0.5
            X[edge_x_mask, 0] = np.where(np.random.rand(np.sum(edge_x_mask)) > 0.5, 0.05, 0.95)
            X[edge_y_mask, 1] = np.where(np.random.rand(np.sum(edge_y_mask)) > 0.5, 0.05, 0.95)
            X = np.clip(X + np.random.randn(n, 2) * 0.02, 0.05, 0.95)
            R = np.ones(n) * 0.03
            
        opt = AdamOptimizer((n, 2), n, lr=lr_start)
        
        # Annealing engine parameters safely integrating complex gradient intersections
        for step in range(iterations):
            progress = step / iterations
            opt.lr = lr_start * (1 - progress) ** 2 + 1e-5
            k = 10 * (100000 ** progress)
            
            dx = X[:, 0].reshape(-1, 1) - X[:, 0]
            dy = X[:, 1].reshape(-1, 1) - X[:, 1]
            eye = np.eye(n, dtype=bool)
            
            dist = np.sqrt(dx**2 + dy**2)
            dist[eye] = 1.0  
            safe_dist = np.maximum(dist, 1e-10)
            
            dir_x = dx / safe_dist
            dir_y = dy / safe_dist
            dist[eye] = np.inf
            
            sum_R = R.reshape(-1, 1) + R
            C_ij = np.maximum(0, sum_R - dist)
            np.fill_diagonal(C_ij, 0)
            
            C_x0 = np.maximum(0, R - X[:, 0])
            C_x1 = np.maximum(0, R - (1 - X[:, 0]))
            C_y0 = np.maximum(0, R - X[:, 1])
            C_y1 = np.maximum(0, R - (1 - X[:, 1]))
            
            dR = -np.ones(n) + k * (np.sum(C_ij, axis=1) + C_x0 + C_x1 + C_y0 + C_y1)
            grad_X_x = k * (-np.sum(C_ij * dir_x, axis=1) - C_x0 + C_x1)
            grad_X_y = k * (-np.sum(C_ij * dir_y, axis=1) - C_y0 + C_y1)
            dX = np.column_stack((grad_X_x, grad_X_y))
            
            # Injection bounds preventing symmetrically stalled forces mathematically resolving space locks
            if progress < 0.6:
                noise_scale = 0.002 * (0.6 - progress)
                dX += np.random.randn(*dX.shape) * noise_scale
            
            X, R = opt.step(X, R, dX, dR)
            
            X = np.clip(X, 0.0, 1.0)
            R = np.clip(R, 0.0, 1.0)
            
        # Post evaluate strict limits to bounds ensuring safe returns globally scaling perfectly 
        val_X, val_R = make_valid(X, R)
        score = np.sum(val_R)
        
        if score > best_score:
            best_score = score
            best_X = val_X.copy()
            best_R = val_R.copy()
            
    return best_X, best_R, best_score


def construct_packing():
    """Construct highly optimized mathematically tight layouts evaluated inside strictly rigorous boundaries."""
    centers, radii, sum_radii = solve_packing(n=26)
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
    # visualize(centers, radii)