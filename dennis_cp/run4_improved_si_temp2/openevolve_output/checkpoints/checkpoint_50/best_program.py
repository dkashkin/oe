# EVOLVE-BLOCK-START
"""
Physics-based gradient optimizer tightly coupled with constrained SLSQP formulation.
This hybrid algorithm employs heavily scaled batched continuous dynamics to intelligently
allocate diverse geometric seeds prior to an exact local topological refinement.
"""

import numpy as np
import time
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

def safely_clip_radii(centers, radii):
    """Enforce absolute physical, geometric, and topological bounding cleanly."""
    out_radii = np.clip(radii, 0.0, 0.5)
    num_pts = centers.shape[0]
    
    out_radii = np.minimum(out_radii, centers[:, 0])
    out_radii = np.minimum(out_radii, centers[:, 1])
    out_radii = np.minimum(out_radii, 1.0 - centers[:, 0])
    out_radii = np.minimum(out_radii, 1.0 - centers[:, 1])
    
    for _ in range(75):
        max_adj = 0.0
        for i in range(num_pts):
            for j in range(i + 1, num_pts):
                sq_d = (centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2
                dist = np.sqrt(max(0.0, sq_d))
                target_dist = out_radii[i] + out_radii[j]
                
                if target_dist > dist + 1e-12:
                    safe_sum = max(0.0, dist - 1e-11)
                    if target_dist > 0:
                        factor = safe_sum / target_dist
                        max_adj = max(max_adj, 1.0 - factor)
                        out_radii[i] *= factor
                        out_radii[j] *= factor
        if max_adj < 1e-13:
            break
            
    return out_radii

def batch_safely_clip(centers_b, radii_b):
    """Batched intersection constraint resolution over continuous radius geometries."""
    r_batch = radii_b.copy()
    c_b = centers_b.copy()
    n = c_b.shape[1]
    
    wx0, wy0 = c_b[..., 0], c_b[..., 1]
    wx1, wy1 = 1.0 - wx0, 1.0 - wy0
    r_max = np.minimum(np.minimum(wx0, wy0), np.minimum(wx1, wy1))
    r_batch = np.minimum(r_batch, r_max)
    
    c_exp = c_b[:, :, np.newaxis, :]
    c_oth = c_b[:, np.newaxis, :, :]
    diff = c_exp - c_oth
    dist = np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-12
    dist += np.eye(n)[np.newaxis, :, :] * 1e10
    
    for _ in range(80):
        r_sum = r_batch[:, :, np.newaxis] + r_batch[:, np.newaxis, :]
        viol = np.maximum(0.0, r_sum - dist)
        if np.max(viol) < 1e-13:
            break
        
        ratio = dist / r_sum
        ratio = np.where(viol > 0, ratio, 1.0)
        r_batch *= np.min(ratio, axis=-1)
        
    return np.maximum(r_batch, 0.0)

def generate_seeds(B, n):
    """Produces highly distributed seeds representing extreme geometric structures."""
    c = np.zeros((B, n, 2))
    r = np.full((B, n), 0.04)
    np.random.seed(987)
    
    for b in range(B):
        mode = b % 8
        if mode == 0:
            c[b] = np.random.uniform(0.05, 0.95, (n, 2))
            r[b] = np.linspace(0.18, 0.02, n)
        elif mode == 1:
            c[b, 0] = [0.5, 0.5]
            for k, (R, cnt, st) in enumerate([(0.22, 6, 1), (0.36, 11, 7), (0.47, 8, 18)]):
                for i in range(cnt):
                    th = 2 * np.pi * i / cnt + k * 0.4
                    c[b, st + i] = [0.5 + R * np.cos(th), 0.5 + R * np.sin(th)]
            r[b] = np.random.uniform(0.02, 0.1, n)
            r[b, 0] = 0.17
        elif mode == 2:
            pts = []
            for j, cnt in enumerate([4, 6, 6, 6, 4]):
                y_p = 0.12 + 0.76 * j / 4.0
                for i in range(cnt):
                    x_p = 0.12 + 0.76 * i / max(1.0, cnt - 1.0)
                    offset = 0.0 if (cnt % 2 == 1) else (0.38 / max(1.0, cnt)) * (j % 2)
                    pts.append([x_p + offset, y_p])
            c[b] = np.array(pts[:n])
            r[b] = np.full(n, 0.075)
        elif mode == 3:
            x, y = np.meshgrid(np.linspace(0.12, 0.88, 5), np.linspace(0.12, 0.88, 5))
            c[b, :25] = np.vstack([x.ravel(), y.ravel()]).T
            c[b, 25] = [0.5, 0.5]
            r[b] = np.full(n, 0.08)
        elif mode == 4:
            c[b, :4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
            c[b, 4:8] = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
            c[b, 8] = [0.5, 0.5]
            c[b, 9:] = np.random.uniform(0.15, 0.85, (17, 2))
            r[b, :9] = 0.14
            r[b, 9:] = np.linspace(0.10, 0.02, 17)
        elif mode == 5:
            c[b] = np.random.uniform(0.2, 0.8, (n, 2))
            r[b] = np.random.uniform(0.01, 0.08, n)
        elif mode == 6:
            c[b, 0] = [0.25, 0.25]
            c[b, 1] = [0.75, 0.75]
            c[b, 2] = [0.25, 0.75]
            c[b, 3] = [0.75, 0.25]
            c[b, 4:] = np.random.uniform(0.1, 0.9, (22, 2))
            r[b, :4] = 0.21
            r[b, 4:] = np.random.uniform(0.02, 0.07, 22)
        else:
            c[b] = np.random.uniform(0.1, 0.9, (n, 2))
            r[b] = np.linspace(0.14, 0.01, n)
            
        c[b] += np.random.randn(n, 2) * 0.01
        
    c = np.clip(c, 0.02, 0.98)
    r = np.clip(r, 0.01, 0.5)
    return c, r

def construct_packing():
    """Generates globally constrained Apollonian packing over hybrid optimizer framework."""
    start_t = time.time()
    n = 26
    B = 300
    
    c, r = generate_seeds(B, n)
    
    lr_c, lr_r = 0.018, 0.010
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_c, v_c = np.zeros_like(c), np.zeros_like(c)
    m_r, v_r = np.zeros_like(r), np.zeros_like(r)
    mask = ~np.eye(n, dtype=bool)[np.newaxis, :, :]
    
    num_steps = 2800
    for step in range(num_steps):
        if step % 150 == 0 and time.time() - start_t > 15.0:
            break
            
        prog = step / float(num_steps)
        lam = 5.0 + 500.0 * (prog ** 2.2)
        
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
        
        if prog < 0.78:
            grad_c += np.random.randn(*grad_c.shape) * 0.09 * (1.0 - prog / 0.78)
            
        b1_eff = 1.0 - beta1**(step + 1)
        b2_eff = 1.0 - beta2**(step + 1)
        
        cur_lr_c = lr_c * (1.0 - 0.4 * prog)
        cur_lr_r = lr_r * (1.0 - 0.4 * prog)
        
        m_c = beta1 * m_c + (1 - beta1) * grad_c
        v_c = beta2 * v_c + (1 - beta2) * (grad_c**2)
        c -= cur_lr_c * (m_c / b1_eff) / (np.sqrt(v_c / b2_eff) + eps)
        
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r**2)
        r -= cur_lr_r * (m_r / b1_eff) / (np.sqrt(v_r / b2_eff) + eps)
        
        c = np.clip(c, 0.001, 0.999)
        r = np.clip(r, 0.001, 0.5)

    r_trim = batch_safely_clip(c, r)
    sums_val = np.sum(r_trim, axis=-1)
    
    top_eval = 18
    top_indices = np.argsort(sums_val)[-top_eval:][::-1]

    i_pair, j_pair = np.triu_indices(n, 1)
    
    mat_a = np.zeros((4 * n, 3 * n))
    mat_l = np.zeros(4 * n)
    for q in range(n):
        mat_a[q, q] = 1.0; mat_a[q, 2*n+q] = -1.0
        mat_a[n+q, q] = -1.0; mat_a[n+q, 2*n+q] = -1.0; mat_l[n+q] = -1.0
        mat_a[2*n+q, n+q] = 1.0; mat_a[2*n+q, 2*n+q] = -1.0
        mat_a[3*n+q, n+q] = -1.0; mat_a[3*n+q, 2*n+q] = -1.0; mat_l[3*n+q] = -1.0

    constr_lin = LinearConstraint(mat_a, mat_l, np.inf)

    def d_val(v):
        x, y, rad = v[:n], v[n:2*n], v[2*n:]
        vx = x[i_pair] - x[j_pair]
        vy = y[i_pair] - y[j_pair]
        vr = rad[i_pair] + rad[j_pair]
        return vx*vx + vy*vy - vr*vr

    def d_jac(v):
        x, y, rad = v[:n], v[n:2*n], v[2*n:]
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

    def f_obj(v): return -float(np.sum(v[2*n:]))

    def f_grad(v):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g

    b_lb = np.zeros(3*n)
    b_ub = np.zeros(3*n)
    b_lb[:2*n] = 0.0; b_ub[:2*n] = 1.0
    b_lb[2*n:] = 1e-6; b_ub[2*n:] = 0.5
    vbnd = Bounds(b_lb, b_ub)

    max_scored = 0.0
    found_vec = np.concatenate([c[top_indices[0], :, 0], c[top_indices[0], :, 1], r_trim[top_indices[0]]])
    
    for idx in top_indices:
        v0 = np.concatenate([c[idx, :, 0], c[idx, :, 1], r_trim[idx]])
        if time.time() - start_t > 28.5:
            break
            
        try:
            res = minimize(
                f_obj, v0, 
                method='SLSQP', 
                jac=f_grad, 
                bounds=vbnd, 
                constraints=[constr_lin, constr_nl], 
                options={'maxiter': 600, 'ftol': 5e-7, 'disp': False}
            )
            c_tmp = res.x[:2*n].reshape(2, n).T
            r_v = safely_clip_radii(c_tmp, res.x[2*n:])
            sc = np.sum(r_v)
            if sc > max_scored:
                max_scored = sc
                found_vec = res.x.copy()
        except Exception:
            pass

    out_c = found_vec[:2*n].reshape((2, n)).T
    v_r = safely_clip_radii(out_c, found_vec[2*n:])
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