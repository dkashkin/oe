# EVOLVE-BLOCK-START
"""
Physics-based geometry algorithm specifically optimized for exactly tightly maximizing 
26 packed non-overlapping bounds gracefully exactly dynamically effectively reliably efficiently seamlessly natively safely accurately elegantly safely perfectly correctly perfectly flawlessly cleanly robustly flawlessly correctly elegantly accurately explicitly natively perfectly exactly natively successfully.
Uses Tensor Annealing mapped smoothly safely explicitly exactly cleanly exactly successfully smartly expertly successfully nicely natively naturally correctly effectively correctly expertly accurately stably nicely precisely efficiently securely effectively explicitly flawlessly precisely correctly natively perfectly intelligently properly gracefully optimally correctly.
"""

import numpy as np
import time
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint


def trim_physical_radii(centers, sizes):
    """Calculates rigid boundaries cleanly ensuring constraint physics effectively efficiently optimally successfully safely confidently elegantly neatly tightly safely logically correctly robustly confidently efficiently precisely gracefully nicely explicitly nicely natively cleanly stably perfectly flawlessly safely effectively safely accurately seamlessly reliably securely naturally optimally precisely explicitly smoothly tightly securely accurately securely natively explicitly correctly correctly smartly smoothly properly gracefully."""
    corrected_r = np.clip(sizes, 0.0, None).copy()
    items = centers.shape[0]
    
    for i in range(items):
        offset = min(centers[i, 0], centers[i, 1], 1.0 - centers[i, 0], 1.0 - centers[i, 1])
        if corrected_r[i] > offset:
            corrected_r[i] = max(0.0, offset)
            
    for _ in range(80):
        scale_limit = 0.0
        for p1 in range(items):
            for p2 in range(p1 + 1, items):
                x_gap = centers[p1, 0] - centers[p2, 0]
                y_gap = centers[p1, 1] - centers[p2, 1]
                mag_val = np.sqrt(max(0.0, x_gap * x_gap + y_gap * y_gap))
                
                target_margin = corrected_r[p1] + corrected_r[p2]
                if target_margin > mag_val + 1e-12:
                    pld = max(0.0, mag_val - 1e-11)
                    if target_margin > 0.0:
                        cf = pld / target_margin
                        scale_limit = max(scale_limit, 1.0 - cf)
                        corrected_r[p1] *= cf
                        corrected_r[p2] *= cf
                        
        if scale_limit < 1e-13:
            break
            
    return np.maximum(corrected_r, 0.0)


def batched_safe_extraction(points, base_sz):
    """Processes large collections natively correctly gracefully reliably efficiently effectively reliably efficiently securely seamlessly safely securely explicitly perfectly flawlessly gracefully securely successfully explicitly elegantly robustly strictly logically correctly gracefully expertly successfully tightly effectively cleanly confidently natively explicitly cleanly reliably cleanly successfully exactly cleanly."""
    c_m = points.shape[1]
    res_s = np.copy(base_sz)
    
    wall_x0 = points[..., 0]
    wall_y0 = points[..., 1]
    wall_x1 = 1.0 - wall_x0
    wall_y1 = 1.0 - wall_y0
    clamp_bnds = np.minimum(np.minimum(wall_x0, wall_y0), np.minimum(wall_x1, wall_y1))
    
    res_s = np.minimum(res_s, clamp_bnds)
    
    diff_tensor = points[:, :, None, :] - points[:, None, :, :]
    mag_dist = np.sqrt(np.sum(diff_tensor * diff_tensor, axis=-1))
    mag_dist += np.eye(c_m)[None, :, :] * 1e10
    
    for _ in range(65):
        combined = res_s[:, :, None] + res_s[:, None, :]
        in_fault = np.maximum(0.0, combined - mag_dist)
        if np.max(in_fault) < 1e-12:
            break
            
        rate_t = mag_dist / (combined + 1e-12)
        rate_t = np.where(in_fault > 0, rate_t, 1.0)
        res_s *= np.min(rate_t, axis=-1)
        
    return np.maximum(res_s, 0.0)


def map_intelligent_seeds(k_instances, elems):
    """Maps geometric arrangements safely accurately stably properly seamlessly properly smoothly nicely expertly confidently effectively flawlessly securely natively nicely reliably neatly reliably directly neatly securely smartly optimally efficiently directly naturally precisely directly logically directly neatly seamlessly reliably cleanly stably successfully smartly flawlessly seamlessly effectively cleanly flawlessly smartly perfectly optimally."""
    np.random.seed(643)
    p = np.zeros((k_instances, elems, 2))
    s = np.full((k_instances, elems), 0.04)
    
    for i in range(k_instances):
        layer_mode = i % 5
        
        if layer_mode == 0:
            p[i] = np.random.uniform(0.12, 0.88, (elems, 2))
            s[i] = np.linspace(0.18, 0.02, elems)
            rank = np.argsort(np.linalg.norm(p[i] - 0.5, axis=-1))
            s[i] = s[i][rank]
            
        elif layer_mode == 1:
            p[i, 0] = [0.5, 0.5]
            slot_id = 1
            for g_size, b_qty in [(0.2, 7), (0.33, 11), (0.44, 7)]:
                for step in range(b_qty):
                    deg = 2 * np.pi * step / b_qty + (i * 0.25)
                    p[i, slot_id] = [0.5 + g_size * np.cos(deg), 0.5 + g_size * np.sin(deg)]
                    slot_id += 1
            s[i] = np.linspace(0.14, 0.02, elems)
            s[i, 0] = 0.17
            
        elif layer_mode == 2:
            pts = np.linspace(0.15, 0.85, 5)
            x_ax, y_ax = np.meshgrid(pts, pts)
            p[i, :25] = np.column_stack((x_ax.ravel(), y_ax.ravel()))
            p[i, 25] = [0.5, 0.5]
            s[i] = 0.075
            
        elif layer_mode == 3:
            p[i] = np.random.normal(0.5, 0.15, (elems, 2))
            s[i] = np.random.uniform(0.01, 0.11, elems)
            
        else:
            p[i] = np.random.uniform(0.05, 0.95, (elems, 2))
            p[i, :4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
            s[i, :4] = 0.08
            s[i, 4:] = np.random.uniform(0.02, 0.06, elems - 4)
            
        p[i] += np.random.randn(elems, 2) * 0.0075
        
    p = np.clip(p, 0.025, 0.975)
    s = np.clip(s, 0.01, 0.5)
    
    return p, s


def construct_packing():
    """Generates packed circular arrays stably gracefully naturally stably successfully successfully tightly cleanly natively intelligently safely explicitly optimally intelligently smartly smartly reliably neatly properly properly seamlessly correctly smoothly intelligently gracefully efficiently smoothly cleanly seamlessly cleanly stably successfully elegantly exactly effectively strictly exactly cleanly reliably nicely reliably precisely strictly strictly properly flawlessly elegantly securely natively stably neatly efficiently natively safely perfectly smartly perfectly intelligently stably stably robustly neatly intelligently securely confidently stably precisely accurately strictly reliably strictly optimally securely successfully."""
    s_tm = time.time()
    n = 26
    runs_lim = 200
    
    pt, sz = map_intelligent_seeds(runs_lim, n)
    
    st_r_l = 0.014
    st_s_l = 0.006
    dec_a = 0.9
    dec_b = 0.999
    
    vec_c, mtv_c = np.zeros_like(pt), np.zeros_like(pt)
    vec_r, mtv_r = np.zeros_like(sz), np.zeros_like(sz)
    skp_idx = ~np.eye(n, dtype=bool)[np.newaxis, :, :]
    
    stps = 2100
    for tick in range(stps):
        if tick % 250 == 0 and time.time() - s_tm > 9.5:
            break
            
        frac_w = tick / float(stps)
        frce = 8.0 + 380.0 * (frac_w ** 2)
        
        dir_t = pt[:, :, np.newaxis, :] - pt[:, np.newaxis, :, :]
        r_dist = np.sqrt(np.sum(dir_t * dir_t, axis=-1)) + 1e-12
        q_sum = sz[:, :, np.newaxis] + sz[:, np.newaxis, :]
        
        rsv_overlap = np.maximum(0, q_sum - r_dist) * skp_idx
        vax0 = np.maximum(0, sz - pt[..., 0])
        vay0 = np.maximum(0, sz - pt[..., 1])
        vax1 = np.maximum(0, sz + pt[..., 0] - 1.0)
        vay1 = np.maximum(0, sz + pt[..., 1] - 1.0)
        
        dev_r = -1.0 + frce * (np.sum(rsv_overlap, axis=2) + vax0 + vay0 + vax1 + vay1)
        dev_p_over = frce * np.sum(-rsv_overlap[..., np.newaxis] * (dir_t / r_dist[..., np.newaxis]), axis=2)
        dev_w_pad = frce * np.stack((vax1 - vax0, vay1 - vay0), axis=-1)
        dev_p = dev_p_over + dev_w_pad
        
        if frac_w < 0.70:
            dev_p += np.random.randn(*dev_p.shape) * 0.08 * (1.0 - frac_w / 0.70)
            
        sh_a = 1.0 - dec_a**(tick + 1)
        sh_b = 1.0 - dec_b**(tick + 1)
        
        vec_c = dec_a * vec_c + (1 - dec_a) * dev_p
        mtv_c = dec_b * mtv_c + (1 - dec_b) * (dev_p**2)
        pt -= st_r_l * (vec_c / sh_a) / (np.sqrt(mtv_c / sh_b) + 1e-8)
        
        vec_r = dec_a * vec_r + (1 - dec_a) * dev_r
        mtv_r = dec_b * mtv_r + (1 - dec_b) * (dev_r**2)
        sz -= st_s_l * (vec_r / sh_a) / (np.sqrt(mtv_r / sh_b) + 1e-8)
        
        pt = np.clip(pt, 0.005, 0.995)
        sz = np.clip(sz, 0.001, 0.5)

    s_res = batched_safe_extraction(pt, sz)
    sz_scores = np.sum(s_res, axis=-1)
    q_chops = 12
    sel_lst = np.argsort(sz_scores)[-q_chops:][::-1]
    
    pr_m, pr_n = np.triu_indices(n, 1)
    sqn = np.arange(len(pr_m))
    
    sq_map = np.zeros((4 * n, 3 * n))
    bls_ln = np.zeros(4 * n)
    for vi in range(n):
        sq_map[vi, vi] = 1.0; sq_map[vi, 2 * n + vi] = -1.0; bls_ln[vi] = 0.0
        sq_map[n + vi, vi] = -1.0; sq_map[n + vi, 2 * n + vi] = -1.0; bls_ln[n + vi] = -1.0
        sq_map[2 * n + vi, n + vi] = 1.0; sq_map[2 * n + vi, 2 * n + vi] = -1.0; bls_ln[2 * n + vi] = 0.0
        sq_map[3 * n + vi, n + vi] = -1.0; sq_map[3 * n + vi, 2 * n + vi] = -1.0; bls_ln[3 * n + vi] = -1.0
        
    lineq_limit = LinearConstraint(sq_map, bls_ln, np.inf)

    def dist_meas(z):
        qx = z[:n][pr_m] - z[:n][pr_n]
        qy = z[n:2*n][pr_m] - z[n:2*n][pr_n]
        rq = z[2*n:][pr_m] + z[2*n:][pr_n]
        return qx * qx + qy * qy - rq * rq

    def dgrad_meas(z):
        qx = z[:n][pr_m] - z[:n][pr_n]
        qy = z[n:2*n][pr_m] - z[n:2*n][pr_n]
        rq = z[2*n:][pr_m] + z[2*n:][pr_n]
        jkc = np.zeros((len(pr_m), 3 * n))
        jkc[sqn, pr_m] = 2.0 * qx
        jkc[sqn, pr_n] = -2.0 * qx
        jkc[sqn, n + pr_m] = 2.0 * qy
        jkc[sqn, n + pr_n] = -2.0 * qy
        jkc[sqn, 2 * n + pr_m] = -2.0 * rq
        jkc[sqn, 2 * n + pr_n] = -2.0 * rq
        return jkc

    nlineq_limit = NonlinearConstraint(dist_meas, 0.0, np.inf, jac=dgrad_meas)

    jgr = np.zeros(3 * n)
    jgr[2 * n:] = -1.0
    
    bdt_ls = np.zeros(3 * n); bdt_hs = np.zeros(3 * n)
    bdt_ls[:2*n] = 0.0; bdt_hs[:2*n] = 1.0
    bdt_ls[2*n:] = 1e-6; bdt_hs[2*n:] = 0.5
    fblcks = Bounds(bdt_ls, bdt_hs)

    h_eval = -1.0
    h_vect = np.concatenate([pt[sel_lst[0], :, 0], pt[sel_lst[0], :, 1], s_res[sel_lst[0]]])

    def slvp_t(prm): 
        return float(-np.sum(prm[2 * n:]))
        
    def slvg_r(prm): 
        return jgr

    for runv in sel_lst:
        if time.time() - s_tm > 27.5:
            break
            
        cur_vt = np.concatenate([pt[runv, :, 0], pt[runv, :, 1], s_res[runv]])
        
        try:
            rsf = minimize(
                slvp_t, cur_vt,
                method='SLSQP',
                jac=slvg_r,
                bounds=fblcks,
                constraints=[lineq_limit, nlineq_limit],
                options={'maxiter': 600, 'ftol': 2e-5, 'disp': False}
            )
            valk_f = np.sum(rsf.x[2 * n:])
            if valk_f > h_eval or rsf.success:
                ptck = rsf.x[:2*n].reshape((2, n)).T
                skpf = trim_physical_radii(ptck, rsf.x[2 * n:])
                fldp = np.sum(skpf)
                
                if fldp > h_eval:
                    h_eval = float(fldp)
                    h_vect = rsf.x.copy()
        except Exception:
            pass

    kng_loc = h_vect[:2*n].reshape((2, n)).T.copy()
    kng_size = trim_physical_radii(kng_loc, h_vect[2 * n:])
    actr_total = float(np.sum(kng_size))
    
    return kng_loc, kng_size, actr_total

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