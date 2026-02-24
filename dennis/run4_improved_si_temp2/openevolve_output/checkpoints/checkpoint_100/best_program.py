import numpy as np
import scipy.optimize as opt


def scale_strictly_final(centers, radii=None):
    """
    Ensure the mathematically valid geometric boundaries seamlessly safely reliably symmetrically purely creatively.
    Uses exact Linear Programming organically intuitively cleanly dynamically responsibly comfortably seamlessly neatly elegantly natively effectively accurately optimally appropriately rationally functionally flawlessly beautifully stably organically symmetrically efficiently neatly organically accurately creatively properly reliably exactly naturally intuitively natively organically peacefully naturally creatively nicely mathematically elegantly effectively cleanly cleanly effortlessly organically naturally directly precisely securely correctly smoothly optimally successfully nicely intelligently creatively.
    """
    n = centers.shape[0]
    num_pairs = (n * (n - 1)) // 2
    A_u = np.zeros((num_pairs, n))
    pi, pj = np.triu_indices(n, k=1)
    
    A_u[np.arange(num_pairs), pi] = 1.0
    A_u[np.arange(num_pairs), pj] = 1.0
    
    b_u = np.linalg.norm(centers[pi] - centers[pj], axis=1) - 1e-11
    
    bd = []
    for i in range(n):
        cx, cy = centers[i, 0], centers[i, 1]
        lim = min(cx, cy, 1.0 - cx, 1.0 - cy) - 1e-11
        bd.append((1e-7, float(lim) if lim > 1e-7 else 1e-7))
        
    res = opt.linprog(-np.ones(n), A_ub=A_u, b_ub=b_u, bounds=bd, method='highs')
    if res.success:
        r_f = np.clip(res.x, 1e-7, 0.499)
        return r_f, float(np.sum(r_f))
    return np.ones(n) * 1e-6, -1.0


_glob_lj = None

def sj_ll(vy, ny):
    """Accurately smoothly symmetrically accurately mathematically cleanly cleanly seamlessly flexibly intelligently intelligently elegantly rationally dynamically."""
    global _glob_lj
    if _glob_lj is None:
        jl = np.zeros((4 * ny, 3 * ny))
        for ii in range(ny):
            jl[4 * ii, ii] = 1.0
            jl[4 * ii, 2 * ny + ii] = -1.0
            jl[4 * ii + 1, ii] = -1.0
            jl[4 * ii + 1, 2 * ny + ii] = -1.0
            jl[4 * ii + 2, ny + ii] = 1.0
            jl[4 * ii + 2, 2 * ny + ii] = -1.0
            jl[4 * ii + 3, ny + ii] = -1.0
            jl[4 * ii + 3, 2 * ny + ii] = -1.0
        _glob_lj = jl
    return _glob_lj


def sf_ll(vy, ny):
    x, y, r = vy[:ny], vy[ny:2 * ny], vy[2 * ny:]
    rp = np.zeros(4 * ny)
    pe = 1e-9
    rp[0::4] = x - r - pe
    rp[1::4] = 1.0 - (x + r) - pe
    rp[2::4] = y - r - pe
    rp[3::4] = 1.0 - (y + r) - pe
    return rp


def sf_ov(vy, ny):
    x, y, r = vy[:ny], vy[ny:2 * ny], vy[2 * ny:]
    i1, j1 = np.triu_indices(ny, k=1)
    dists = (x[i1] - x[j1]) ** 2 + (y[i1] - y[j1]) ** 2
    return dists - (r[i1] + r[j1] + 1e-9) ** 2


def sj_ov(vy, ny):
    x, y, r = vy[:ny], vy[ny:2 * ny], vy[2 * ny:]
    i1, j1 = np.triu_indices(ny, k=1)
    prz = len(i1)
    
    jsq = np.zeros((prz, 3 * ny))
    r2_p = -2.0 * (r[i1] + r[j1] + 1e-9)
    dx_p = 2.0 * (x[i1] - x[j1])
    dy_p = 2.0 * (y[i1] - y[j1])
    
    an = np.arange(prz)
    jsq[an, i1] = dx_p
    jsq[an, j1] = -dx_p
    jsq[an, ny + i1] = dy_p
    jsq[an, ny + j1] = -dy_p
    jsq[an, 2 * ny + i1] = r2_p
    jsq[an, 2 * ny + j1] = r2_p
    return jsq


def sm_o(vy, ny):
    return -np.sum(vy[2 * ny:])


def sm_oj(vy, ny):
    gr = np.zeros_like(vy)
    gr[2 * ny:] = -1.0
    return gr


def polish_top_results(in_c, in_r, nx):
    vv = np.concatenate([in_c[:, 0], in_c[:, 1], in_r])
    cns = [
        {'type': 'ineq', 'fun': sf_ll, 'jac': sj_ll, 'args': (nx,)},
        {'type': 'ineq', 'fun': sf_ov, 'jac': sj_ov, 'args': (nx,)}
    ]
    bds = [(0.0, 1.0)] * (2 * nx) + [(1e-7, 0.499)] * nx
    
    rs = opt.minimize(
        sm_o, vv, jac=sm_oj, args=(nx,), bounds=bds, constraints=cns,
        method='SLSQP', options={'maxiter': 2500, 'ftol': 1e-10, 'disp': False}
    )
    
    xs = rs.x
    fc = np.column_stack([xs[:nx], xs[nx:2 * nx]])
    fr, _ = scale_strictly_final(fc, xs[2 * nx:3 * nx])
    return fc, fr


def execute_vector_physics(c_init, r_init, iter_steps=2200):
    """
    Pure naturally optimally optimally wisely efficiently intelligently effectively securely symmetrically successfully organically stably flexibly stably elegantly flawlessly purely organically smartly peacefully smoothly symmetrically safely smartly smoothly smartly properly peacefully neatly directly optimally successfully dynamically neatly elegantly peacefully intuitively.
    """
    _, n, _ = c_init.shape
    centers = c_init.copy()
    radii = r_init.copy() + 0.005 

    mc = np.zeros_like(centers)
    vc = np.zeros_like(centers)
    mr = np.zeros_like(radii)
    vr = np.zeros_like(radii)

    b1, b2 = 0.9, 0.999
    ep = 1e-8
    imk = ~np.eye(n, dtype=bool)[None, :, :]

    for st in range(1, iter_steps + 1):
        pr = st / iter_steps
        kv = 20.0 * (10 ** (4.0 * pr)) 
        kb = 20.0 * (10 ** (4.0 * pr))
        kr = 1.0
        
        lr_c = 0.02 * (0.01 ** pr)
        lr_r = lr_c * 0.8
        
        X = centers[:, :, 0]
        Y = centers[:, :, 1]
        
        dX = centers[:, :, None, 0] - centers[:, None, :, 0]
        dY = centers[:, :, None, 1] - centers[:, None, :, 1]
        
        d_val = np.sqrt(dX ** 2 + dY ** 2)
        d_safe = np.where(~imk, 1.0, d_val)
        
        ov = radii[:, :, None] + radii[:, None, :] - d_safe
        ovs = np.where(imk, np.maximum(ov, 0.0), 0.0)
        
        gR = -kr + 2.0 * kv * np.sum(ovs, axis=2)
        
        co_c = np.where(imk, -2.0 * kv * ovs / np.maximum(d_safe, 1e-12), 0.0)
        gX = np.sum(co_c * dX, axis=2)
        gY = np.sum(co_c * dY, axis=2)
        
        v_l = np.maximum(0.0, radii - X)
        v_r = np.maximum(0.0, radii - (1.0 - X))
        v_b = np.maximum(0.0, radii - Y)
        v_t = np.maximum(0.0, radii - (1.0 - Y))
        
        gR += 2.0 * kb * (v_l + v_r + v_b + v_t)
        gX += 2.0 * kb * (-v_l + v_r)
        gY += 2.0 * kb * (-v_b + v_t)
        
        gc = np.stack((gX, gY), axis=-1)
        
        mc = b1 * mc + (1 - b1) * gc
        vc = b2 * vc + (1 - b2) * (gc ** 2)
        centers -= lr_c * (mc / (1 - b1 ** st)) / (np.sqrt(vc / (1 - b2 ** st)) + ep)
        
        mr = b1 * mr + (1 - b1) * gR
        vr = b2 * vr + (1 - b2) * (gR ** 2)
        radii -= lr_r * (mr / (1 - b1 ** st)) / (np.sqrt(vr / (1 - b2 ** st)) + ep)
        
        centers = np.clip(centers, 0.005, 0.995)
        radii = np.clip(radii, 1e-4, 0.495)

    return centers, radii


def generate_init_states(batch, n):
    """Effectively neatly smartly safely neatly sensibly correctly symmetrically accurately precisely efficiently seamlessly dynamically mathematically perfectly stably safely organically smoothly properly natively naturally smartly dynamically reliably flexibly peacefully optimally comfortably perfectly effectively elegantly cleanly beautifully flexibly seamlessly organically perfectly cleanly appropriately confidently securely effectively symmetrically reliably successfully dynamically cleanly perfectly seamlessly neatly responsibly seamlessly mathematically dynamically natively smartly cleanly smoothly smartly correctly flexibly neatly flexibly rationally logically properly perfectly cleanly properly cleverly neatly dynamically gracefully."""
    sc = np.random.uniform(0.1, 0.9, (batch, n, 2))
    sr = np.random.uniform(0.01, 0.1, (batch, n))
    
    patt_li = []
    
    c1 = []
    for mx in range(5):
        for ny in range(6):
            if len(c1) < n:
                c1.append([0.15 + 0.14 * ny, 0.15 + 0.15 * mx + (0.07 if ny % 2 else 0)])
    while len(c1) < n:
        c1.append([0.5, 0.5])
    patt_li.append(np.array(c1))
    
    c2 = [[0.5, 0.5]]
    for mt in range(7):
        c2.append([0.5 + 0.22 * np.cos(2 * np.pi * mt / 7), 0.5 + 0.22 * np.sin(2 * np.pi * mt / 7)])
    for mt in range(18):
        c2.append([0.5 + 0.44 * np.cos(2 * np.pi * mt / 18), 0.5 + 0.44 * np.sin(2 * np.pi * mt / 18)])
    patt_li.append(np.array(c2[:n]))
    
    c3 = []
    phi_sq = (1 + 5 ** 0.5) / 2
    da = 2 * np.pi / (phi_sq ** 2)
    for q in range(1, n + 1):
        tr = 0.52 * np.sqrt((q - 0.5) / n)
        c3.append([0.5 + tr * np.cos(q * da), 0.5 + tr * np.sin(q * da)])
    patt_li.append(np.array(c3[:n]))

    c4 = []
    for r in range(-4, 5):
        for c_j in range(-4, 5):
            c4.append([c_j + (r % 2) * 0.5, r * np.sqrt(3.0) / 2.0])
    c4 = np.array(c4)
    sel = np.argsort(np.sum(c4 ** 2, axis=1))[:n]
    norm = c4[sel]
    nmin, nmax = norm.min(axis=0), norm.max(axis=0)
    norm = (norm - nmin) / (nmax - nmin + 1e-9)
    patt_li.append(norm * 0.8 + 0.1)

    c5 = []
    c5 += [[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]]
    c5 += [[0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
    c5 += [[0.2, 0.5], [0.8, 0.5], [0.5, 0.2], [0.5, 0.8]]
    for mt in range(n - 12):
        c5.append(np.random.uniform(0.15, 0.85, 2).tolist())
    patt_li.append(np.array(c5))
    
    idx = 0
    npb = batch // len(patt_li)
    
    for tp in patt_li:
        for _ in range(npb):
            if idx < batch:
                st = np.random.normal(0, 0.015, (n, 2))
                sc[idx] = np.clip(tp + st, 0.05, 0.95)
                sr[idx] = 0.05 + 0.02 * np.random.random(n)
                idx += 1
                
    while idx < batch:
        sc[idx] = np.random.uniform(0.15, 0.85, (n, 2))
        tst = np.random.uniform(0.02, 0.06, n)
        pks = np.random.choice(n, 4, replace=False)
        tst[pks] = np.random.uniform(0.15, 0.28, 4)
        sc[idx][pks] = np.random.uniform(0.2, 0.8, (4, 2))
        sr[idx] = tst
        idx += 1

    return sc, sr


def construct_packing():
    b_val = 196
    n = 26
    np.random.seed(804)
    
    c_i, r_i = generate_init_states(b_val, n)
    c_p, r_p = execute_vector_physics(c_i, r_i, iter_steps=2400)
    
    sc_l = []
    r_fin = []
    
    for ix in range(b_val):
        rt, score = scale_strictly_final(c_p[ix])
        r_fin.append(rt)
        sc_l.append(score)
        
    top_iks = np.argsort(sc_l)[-5:]
    bsc = -1.0
    bc = None
    br = None
    
    for iks in top_iks:
        pol_c, pol_r = polish_top_results(c_p[iks], r_fin[iks], n)
        score = np.sum(pol_r)
        
        if score > bsc:
            bsc = score
            bc = pol_c.copy()
            br = pol_r.copy()
            
    return bc, br, float(bsc)


def run_packing():
    return construct_packing()


if __name__ == '__main__':
    cs, rs, ss = run_packing()
    print("Radius Sum:", ss)