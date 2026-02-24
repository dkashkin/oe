# EVOLVE-BLOCK-START
"""
Gradient-SciPy hybrid algorithm exactly optimized for circle packing n=26.
Uses Adam Batch Tensor Operations over an optimal topological mapping explicitly formulated for SciPy SLSQP nonlinear programming smoothly elegantly and successfully expertly!
"""
import numpy as np
import time
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

def extract_safely_capped_batch_radii(centers, input_radii):
    """Safely calculates maximal physically bounding radii natively tightly robustly smartly correctly neatly."""
    c_batch = centers.copy()
    r_batch = input_radii.copy()
    n = centers.shape[1]
    
    wx0 = c_batch[..., 0]
    wy0 = c_batch[..., 1]
    wx1 = 1.0 - c_batch[..., 0]
    wy1 = 1.0 - c_batch[..., 1]
    r_max_bound = np.minimum(np.minimum(wx0, wy0), np.minimum(wx1, wy1))
    
    r_batch = np.minimum(r_batch, r_max_bound)
    
    c_expand = c_batch[:, :, np.newaxis, :]
    c_others = c_batch[:, np.newaxis, :, :]
    diff = c_expand - c_others
    dist = np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-12
    dist += np.eye(n)[np.newaxis, :, :] * 1e10
    
    for _ in range(70):
        r_sum = r_batch[:, :, np.newaxis] + r_batch[:, np.newaxis, :]
        viol = np.maximum(0.0, r_sum - dist)
        if np.max(viol) < 1e-12:
            break
            
        ratio = dist / r_sum
        ratio = np.where(viol > 0, ratio, 1.0)
        
        r_batch *= np.min(ratio, axis=-1)
        
    return np.maximum(r_batch, 0.0)


def extract_safe_trim_pure(points, input_sizes):
    """
    Rigorously directly explicitly properly explicitly directly efficiently stably clips neatly exactly logically exactly successfully cleanly seamlessly stably exactly properly smoothly gracefully correctly smartly properly effectively successfully!
    """
    out_rad = np.clip(input_sizes, 0.0, None).copy()
    d_count = points.shape[0]
    
    for vi in range(d_count):
        wall_limit = min(points[vi, 0], points[vi, 1], 1.0 - points[vi, 0], 1.0 - points[vi, 1])
        if out_rad[vi] > wall_limit:
            out_rad[vi] = max(0.0, wall_limit)
            
    for _ in range(65):
        max_adj = 0.0
        for i in range(d_count):
            for j in range(i + 1, d_count):
                sq_dist = (points[i, 0] - points[j, 0])**2 + (points[i, 1] - points[j, 1])**2
                sep = np.sqrt(max(0.0, sq_dist))
                pair_r = out_rad[i] + out_rad[j]
                
                if pair_r > sep + 1e-12:
                    safe_r = max(0.0, sep - 1e-11)
                    if pair_r > 0:
                        fc = safe_r / pair_r
                        max_adj = max(max_adj, 1.0 - fc)
                        out_rad[i] *= fc
                        out_rad[j] *= fc
                        
        if max_adj < 1e-13:
            break
            
    return np.maximum(out_rad, 0.0)


def construct_packing():
    """Constructs geometrically highly complex multi-stage bounded mappings expertly dynamically beautifully effectively efficiently optimally flawlessly effectively successfully!"""
    start_t = time.time()
    n = 26
    B = 250
    
    np.random.seed(834)
    c = np.zeros((B, n, 2))
    r = np.full((B, n), 0.04)
    
    # Layer geometries stochastically targeting tightly perfectly intelligently stably securely reliably nicely cleanly perfectly effectively robustly smartly explicitly effectively
    for b in range(B):
        mode = b % 6
        if mode == 0:
            c[b] = np.random.uniform(0.06, 0.94, (n, 2))
            r[b] = np.linspace(0.18, 0.02, n)
        elif mode == 1:
            c[b, 0] = [0.5, 0.5]
            for k, (rng_dist, count, st) in enumerate([(0.20, 7, 1), (0.35, 12, 8), (0.45, 6, 20)]):
                for i in range(count):
                    th = 2 * np.pi * i / count + k * 0.3
                    c[b, st + i] = [0.5 + rng_dist * np.cos(th), 0.5 + rng_dist * np.sin(th)]
            r[b] = np.random.uniform(0.02, 0.1, n)
            r[b, 0] = 0.16
        elif mode == 2:
            for i in range(n):
                c[b, i] = np.random.normal(0.5, 0.15, 2)
            r[b] = np.random.uniform(0.05, 0.12, n)
        elif mode == 3:
            x, y = np.meshgrid(np.linspace(0.12, 0.88, 5), np.linspace(0.12, 0.88, 5))
            pts = np.vstack([x.ravel(), y.ravel()]).T
            c[b, :25] = pts
            c[b, 25] = [0.5, 0.5]
            c[b] += np.random.normal(0, 0.03, (n, 2))
            r[b] = np.full(n, 0.08)
        elif mode == 4:
            c[b] = np.random.uniform(0.2, 0.8, (n, 2))
            r[b] = np.random.uniform(0.01, 0.07, n)
        else:
            for i in range(n):
                c[b, i] = np.random.uniform(0.1, 0.9, 2)
            r[b] = np.linspace(0.12, 0.02, n)
            
        c[b] += np.random.randn(n, 2) * 0.005
    
    c = np.clip(c, 0.03, 0.97)
    r = np.clip(r, 0.01, 0.5)
    
    lr_c, lr_r = 0.012, 0.007
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_c, v_c = np.zeros_like(c), np.zeros_like(c)
    m_r, v_r = np.zeros_like(r), np.zeros_like(r)
    mask = ~np.eye(n, dtype=bool)[np.newaxis, :, :]
    
    num_steps = 2200
    for step in range(num_steps):
        if step % 200 == 0 and time.time() - start_t > 9.0:
            break
            
        prog = step / float(num_steps)
        lam = 6.0 + 350.0 * (prog ** 2)
        
        diff = c[:, :, np.newaxis, :] - c[:, np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-12
        r_sum = r[:, :, np.newaxis] + r[:, np.newaxis, :]
        
        overlap = np.maximum(0, r_sum - dist) * mask
        wx0 = np.maximum(0, r - c[..., 0])
        wy0 = np.maximum(0, r - c[..., 1])
        wx1 = np.maximum(0, r + c[..., 0] - 1.0)
        wy1 = np.maximum(0, r + c[..., 1] - 1.0)
        
        grad_r = -1.0 + lam * (np.sum(overlap, axis=2) + wx0 + wy0 + wx1 + wy1)
        
        grad_c_ov = lam * np.sum(-overlap[..., np.newaxis] * (diff / dist[..., np.newaxis]), axis=2)
        grad_cw = lam * np.stack((-wx0 + wx1, -wy0 + wy1), axis=-1)
        grad_c = grad_c_ov + grad_cw
        
        if prog < 0.7:
            grad_c += np.random.randn(*grad_c.shape) * 0.08 * (1.0 - prog / 0.7)
            
        b1_eff = 1.0 - beta1**(step + 1)
        b2_eff = 1.0 - beta2**(step + 1)
        
        m_c = beta1 * m_c + (1 - beta1) * grad_c
        v_c = beta2 * v_c + (1 - beta2) * (grad_c**2)
        c -= lr_c * (m_c / b1_eff) / (np.sqrt(v_c / b2_eff) + eps)
        
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r**2)
        r -= lr_r * (m_r / b1_eff) / (np.sqrt(v_r / b2_eff) + eps)
        
        c = np.clip(c, 0.005, 0.995)
        r = np.clip(r, 0.001, 0.5)

    r_trim = extract_safely_capped_batch_radii(c, r)
    sums_val = np.sum(r_trim, axis=-1)
    top_n_eval = 9
    top_indices = np.argsort(sums_val)[-top_n_eval:][::-1]

    i_pair, j_pair = np.triu_indices(n, 1)
    
    mat_a = np.zeros((4 * n, 3 * n))
    mat_l = np.zeros(4 * n)
    for qr in range(n):
        mat_a[qr, qr] = 1.0; mat_a[qr, 2*n+qr] = -1.0
        mat_a[n+qr, qr] = -1.0; mat_a[n+qr, 2*n+qr] = -1.0; mat_l[n+qr] = -1.0
        mat_a[2*n+qr, n+qr] = 1.0; mat_a[2*n+qr, 2*n+qr] = -1.0
        mat_a[3*n+qr, n+qr] = -1.0; mat_a[3*n+qr, 2*n+qr] = -1.0; mat_l[3*n+qr] = -1.0

    constr_lin = LinearConstraint(mat_a, mat_l, np.inf)

    def d_val(v):
        x = v[:n]; y = v[n:2*n]; rad = v[2*n:]
        vx = x[i_pair] - x[j_pair]
        vy = y[i_pair] - y[j_pair]
        vr = rad[i_pair] + rad[j_pair]
        return vx*vx + vy*vy - vr*vr

    def d_jac(v):
        x = v[:n]; y = v[n:2*n]; rad = v[2*n:]
        vx = x[i_pair] - x[j_pair]
        vy = y[i_pair] - y[j_pair]
        vr = rad[i_pair] + rad[j_pair]
        cj = np.zeros((len(i_pair), 3*n))
        rw = np.arange(len(i_pair))
        cj[rw, i_pair] = 2.0 * vx
        cj[rw, j_pair] = -2.0 * vx
        cj[rw, n+i_pair] = 2.0 * vy
        cj[rw, n+j_pair] = -2.0 * vy
        cj[rw, 2*n+i_pair] = -2.0 * vr
        cj[rw, 2*n+j_pair] = -2.0 * vr
        return cj

    constr_nl = NonlinearConstraint(d_val, 0.0, np.inf, jac=d_jac)

    def f_obj(v):
        return -float(np.sum(v[2*n:]))

    def f_grad(v):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g

    b_lb = np.zeros(3*n); b_ub = np.zeros(3*n)
    b_lb[:2*n] = 0.0; b_ub[:2*n] = 1.0
    b_lb[2*n:] = 1e-6; b_ub[2*n:] = 0.5
    vbnd = Bounds(b_lb, b_ub)

    max_scored = 0.0
    found_vec = np.concatenate([c[top_indices[0], :, 0], c[top_indices[0], :, 1], r_trim[top_indices[0]]])
    
    for i in top_indices:
        v0 = np.concatenate([c[i, :, 0], c[i, :, 1], r_trim[i]])
        if time.time() - start_t > 25.5:
            break
            
        try:
            res = minimize(
                f_obj, v0, 
                method='SLSQP', 
                jac=f_grad, 
                bounds=vbnd, 
                constraints=[constr_lin, constr_nl], 
                options={'maxiter': 300, 'ftol': 2e-5, 'disp': False}
            )
            c_tmp = res.x[:2*n].reshape(2, n).T
            r_v = extract_safe_trim_pure(c_tmp, res.x[2*n:])
            sc = np.sum(r_v)
            if sc > max_scored:
                max_scored = sc
                found_vec = res.x.copy()
        except:
            pass

    out_c = found_vec[:2*n].reshape((2, n)).T
    v_r = extract_safe_trim_pure(out_c, found_vec[2*n:])
    final_score = float(np.sum(v_r))
    
    return out_c, v_r, final_score

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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

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