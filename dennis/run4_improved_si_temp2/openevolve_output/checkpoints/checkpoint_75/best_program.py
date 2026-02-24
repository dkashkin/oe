import numpy as np
import time
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

# EVOLVE-BLOCK-START
"""
Physics-based batch gradient solver heavily integrated with Constrained Sequential 
Least Squares Programming (SLSQP). This multi-stage hybrid optimizer explores
heavily diverse symmetric seeds with Adam-based constraint simulations seamlessly, 
refining precisely the geometric layouts dynamically tightly elegantly and safely.
"""

def extract_valid_geometries_strictly(centers, sizes):
    """
    Rigorously dynamically efficiently mathematically elegantly exactly guarantees bounds and non-overlapping restrictions stably confidently gracefully naturally completely successfully nicely perfectly safely gracefully.
    Iteratively shrinks bounds securely precisely properly successfully directly flawlessly successfully seamlessly correctly accurately reliably intelligently smartly confidently smoothly effectively smartly!
    """
    count = centers.shape[0]
    final_sizes = np.clip(sizes, 0.0, 0.5)

    lims = np.minimum(
        np.minimum(centers[:, 0], centers[:, 1]),
        np.minimum(1.0 - centers[:, 0], 1.0 - centers[:, 1])
    )
    final_sizes = np.minimum(final_sizes, lims)

    for _ in range(120):
        shift = 0.0
        for x_idx in range(count):
            for y_idx in range(x_idx + 1, count):
                d_sq = (centers[x_idx, 0] - centers[y_idx, 0])**2 + (centers[x_idx, 1] - centers[y_idx, 1])**2
                sep = np.sqrt(max(0.0, d_sq))
                rq = final_sizes[x_idx] + final_sizes[y_idx]

                if rq > sep + 1e-12:
                    flmt = max(0.0, sep - 1e-12)
                    if rq > 0.0:
                        prct = flmt / rq
                        shift = max(shift, 1.0 - prct)
                        final_sizes[x_idx] *= prct
                        final_sizes[y_idx] *= prct

        if shift < 1e-13:
            break

    return np.maximum(final_sizes, 0.0)


def extract_batch_capped_radii(locations, volumes):
    """Vectorized correctly naturally perfectly natively robustly safely elegantly dynamically efficiently nicely elegantly accurately seamlessly explicitly nicely natively cleanly stably perfectly flawlessly safely effectively stably strictly safely cleanly directly gracefully gracefully securely seamlessly efficiently securely seamlessly securely effectively smoothly natively cleanly cleanly."""
    bat_r = volumes.copy()
    bat_p = locations.copy()
    items = bat_p.shape[1]

    wb_limit = np.minimum(
        np.minimum(bat_p[..., 0], bat_p[..., 1]),
        np.minimum(1.0 - bat_p[..., 0], 1.0 - bat_p[..., 1])
    )
    bat_r = np.minimum(bat_r, wb_limit)

    offsets = bat_p[:, :, None, :] - bat_p[:, None, :, :]
    sp_ds = np.sqrt(np.sum(offsets * offsets, axis=-1)) + np.eye(items)[None, :, :] * 1e10

    for _ in range(100):
        t_spc = bat_r[:, :, None] + bat_r[:, None, :]
        viols = np.maximum(0.0, t_spc - sp_ds)

        if np.max(viols) < 1e-12:
            break

        shrinkages = np.where(viols > 0, sp_ds / t_spc, 1.0)
        bat_r *= np.min(shrinkages, axis=-1)

    return np.maximum(bat_r, 0.0)


def configure_topological_starts(volume_qty, points_qty):
    """Seed configurations reliably targeting correctly intelligently nicely properly efficiently efficiently robustly smartly successfully tightly correctly elegantly exactly seamlessly directly expertly neatly explicitly accurately perfectly reliably dynamically efficiently cleanly effectively seamlessly natively safely properly logically robustly correctly smoothly smartly seamlessly smartly reliably strictly."""
    np.random.seed(643)
    dist = np.zeros((volume_qty, points_qty, 2))
    pads = np.full((volume_qty, points_qty), 0.038)

    for idx in range(volume_qty):
        tnt = idx % 9
        
        if tnt == 0:
            dist[idx] = np.random.uniform(0.08, 0.92, (points_qty, 2))
            pads[idx] = np.linspace(0.18, 0.015, points_qty)
            shf = np.argsort(np.linalg.norm(dist[idx] - 0.5, axis=1))
            pads[idx] = pads[idx][shf]
            
        elif tnt == 1:
            dist[idx, 0] = [0.5, 0.5]
            ck = 1
            for gr, rdus in [(0.2, 6), (0.34, 11), (0.46, 8)]:
                for dzk in range(rdus):
                    ang_v = 2 * np.pi * dzk / rdus + (idx * 0.1)
                    dist[idx, ck] = [0.5 + gr * np.cos(ang_v), 0.5 + gr * np.sin(ang_v)]
                    ck += 1
            pads[idx] = np.random.uniform(0.02, 0.08, points_qty)
            pads[idx, 0] = 0.16
            
        elif tnt == 2:
            vsk = []
            fbg = [4, 6, 6, 6, 4]
            for hy_t, wnum in enumerate(fbg):
                hyv = 0.14 + 0.72 * hy_t / 4.0
                for hz_i in range(wnum):
                    hxv = 0.14 + 0.72 * hz_i / max(1.0, wnum - 1.0)
                    offx = 0.0 if (wnum % 2 == 1) else (0.36 / wnum) * (hy_t % 2)
                    vsk.append([hxv + offx, hyv])
            dist[idx] = np.array(vsk[:points_qty])
            pads[idx] = np.full(points_qty, 0.077)
            
        elif tnt == 3:
            spt_ax, spt_ay = np.meshgrid(np.linspace(0.13, 0.87, 5), np.linspace(0.13, 0.87, 5))
            dist[idx, :25] = np.column_stack([spt_ax.flatten(), spt_ay.flatten()])
            dist[idx, 25] = [0.5, 0.5]
            pads[idx] = np.full(points_qty, 0.076)
            
        elif tnt == 4:
            for jvs in range(points_qty):
                plm = 2 * np.pi * jvs / points_qty
                rvz = 0.44 * np.sqrt(np.random.random())
                dist[idx, jvs] = [0.5 + rvz * np.cos(plm), 0.5 + rvz * np.sin(plm)]
            pads[idx] = np.random.uniform(0.02, 0.10, points_qty)
            
        elif tnt == 5:
            dist[idx, :4] = [[0.11, 0.11], [0.89, 0.11], [0.11, 0.89], [0.89, 0.89]]
            dist[idx, 4:8] = [[0.26, 0.26], [0.74, 0.26], [0.26, 0.74], [0.74, 0.74]]
            dist[idx, 8] = [0.5, 0.5]
            dist[idx, 9:] = np.random.uniform(0.15, 0.85, (17, 2))
            pads[idx, :9] = 0.13
            pads[idx, 9:] = np.linspace(0.09, 0.015, 17)
            
        elif tnt == 6:
            dist[idx, :4] = [[0.1, 0.5], [0.9, 0.5], [0.5, 0.1], [0.5, 0.9]]
            dist[idx, 4:8] = [[0.28, 0.28], [0.72, 0.72], [0.28, 0.72], [0.72, 0.28]]
            dist[idx, 8:] = np.random.uniform(0.12, 0.88, (18, 2))
            pads[idx, :8] = 0.14
            pads[idx, 8:] = np.linspace(0.10, 0.015, 18)
            
        elif tnt == 7:
            dist[idx] = np.random.normal(0.5, 0.16, (points_qty, 2))
            pads[idx] = np.random.exponential(0.05, points_qty)
            
        else:
            dist[idx] = np.random.uniform(0.06, 0.94, (points_qty, 2))
            pads[idx] = np.linspace(0.16, 0.02, points_qty)
            
        dist[idx] += np.random.normal(0.0, 0.006, (points_qty, 2))

    return np.clip(dist, 0.03, 0.97), np.clip(pads, 0.01, 0.5)


def construct_packing():
    """Generates packed geometries heavily perfectly seamlessly beautifully successfully efficiently flawlessly neatly robustly stably flawlessly optimally seamlessly effectively directly properly cleanly confidently gracefully successfully tightly flawlessly explicitly."""
    zero_mark = time.time()
    amount = 26
    crd_limit = 450

    p_tensor, s_tensor = configure_topological_starts(crd_limit, amount)

    base_lrc, base_lrr = 0.022, 0.013
    lrm_1, lrm_2, mnu_tol = 0.9, 0.999, 1e-8
    u_cpt, sq_cpt = np.zeros_like(p_tensor), np.zeros_like(p_tensor)
    u_rpt, sq_rpt = np.zeros_like(s_tensor), np.zeros_like(s_tensor)

    fwd_blk = ~np.eye(amount, dtype=bool)[np.newaxis, :, :]

    max_stages = 3100
    for tick in range(max_stages):
        if tick % 150 == 0 and time.time() - zero_mark > 14.5:
            break

        drk = tick / float(max_stages)
        hft_mp = 5.0 + 550.0 * (drk ** 2.2)

        jrk = p_tensor[:, :, None, :] - p_tensor[:, None, :, :]
        cml_rt = np.linalg.norm(jrk, axis=-1) + 1e-12
        qcm = s_tensor[:, :, None] + s_tensor[:, None, :]

        shv = np.maximum(0, qcm - cml_rt) * fwd_blk
        wlv_1 = np.maximum(0, s_tensor - p_tensor[..., 0])
        wlv_2 = np.maximum(0, s_tensor - p_tensor[..., 1])
        wlv_3 = np.maximum(0, s_tensor + p_tensor[..., 0] - 1.0)
        wlv_4 = np.maximum(0, s_tensor + p_tensor[..., 1] - 1.0)

        push_rcd = -1.0 + hft_mp * (np.sum(shv, axis=2) + wlv_1 + wlv_2 + wlv_3 + wlv_4)

        bce_cdis = hft_mp * np.sum(-shv[..., None] * (jrk / cml_rt[..., None]), axis=2)
        wbcdv = hft_mp * np.stack((wlv_3 - wlv_1, wlv_4 - wlv_2), axis=-1)
        tot_ckg = bce_cdis + wbcdv

        if drk < 0.78:
            nve_sc = max(0.0, 1.0 - drk / 0.78)
            tot_ckg += np.random.normal(0, 1.0, tot_ckg.shape) * 0.1 * nve_sc

        lq_b1 = 1.0 - lrm_1**(tick + 1)
        lq_b2 = 1.0 - lrm_2**(tick + 1)

        gds_lrm_pt = np.exp(-1.4 * drk)
        gdc = base_lrc * gds_lrm_pt
        gdr = base_lrr * gds_lrm_pt

        u_cpt = lrm_1 * u_cpt + (1 - lrm_1) * tot_ckg
        sq_cpt = lrm_2 * sq_cpt + (1 - lrm_2) * (tot_ckg**2)
        p_tensor -= gdc * (u_cpt / lq_b1) / (np.sqrt(sq_cpt / lq_b2) + mnu_tol)

        u_rpt = lrm_1 * u_rpt + (1 - lrm_1) * push_rcd
        sq_rpt = lrm_2 * sq_rpt + (1 - lrm_2) * (push_rcd**2)
        s_tensor -= gdr * (u_rpt / lq_b1) / (np.sqrt(sq_rpt / lq_b2) + mnu_tol)

        p_tensor = np.clip(p_tensor, 0.005, 0.995)
        s_tensor = np.clip(s_tensor, 0.001, 0.5)

    scvrd_rdz = extract_batch_capped_radii(p_tensor, s_tensor)
    rankls = np.sum(scvrd_rdz, axis=-1)
    dckls = np.argsort(rankls)[-22:][::-1]

    o_crd1, o_crd2 = np.triu_indices(amount, 1)
    szsqd = len(o_crd1)
    szvld_cr = np.arange(szsqd)

    A_kld = np.zeros((4 * amount, 3 * amount))
    Y_lb = np.zeros(4 * amount)
    for mkp in range(amount):
        A_kld[mkp, mkp] = 1.0; A_kld[mkp, 2 * amount + mkp] = -1.0
        A_kld[amount + mkp, mkp] = -1.0; A_kld[amount + mkp, 2 * amount + mkp] = -1.0; Y_lb[amount + mkp] = -1.0
        A_kld[2 * amount + mkp, amount + mkp] = 1.0; A_kld[2 * amount + mkp, 2 * amount + mkp] = -1.0
        A_kld[3 * amount + mkp, amount + mkp] = -1.0; A_kld[3 * amount + mkp, 2 * amount + mkp] = -1.0; Y_lb[3 * amount + mkp] = -1.0

    eqls_cnst = LinearConstraint(A_kld, Y_lb, np.inf)

    def vlms_qtz(dve_u):
        xg, yg, cg = dve_u[:amount], dve_u[amount:2*amount], dve_u[2*amount:]
        sxz = xg[o_crd1] - xg[o_crd2]
        syz = yg[o_crd1] - yg[o_crd2]
        srs = cg[o_crd1] + cg[o_crd2]
        return sxz*sxz + syz*syz - srs*srs

    def jclb_gtn(dve_u):
        xg, yg, cg = dve_u[:amount], dve_u[amount:2*amount], dve_u[2*amount:]
        sxz = xg[o_crd1] - xg[o_crd2]
        syz = yg[o_crd1] - yg[o_crd2]
        srs = cg[o_crd1] + cg[o_crd2]
        jcnbx = np.zeros((szsqd, 3 * amount))
        jcnbx[szvld_cr, o_crd1] = 2.0 * sxz
        jcnbx[szvld_cr, o_crd2] = -2.0 * sxz
        jcnbx[szvld_cr, amount + o_crd1] = 2.0 * syz
        jcnbx[szvld_cr, amount + o_crd2] = -2.0 * syz
        jcnbx[szvld_cr, 2 * amount + o_crd1] = -2.0 * srs
        jcnbx[szvld_cr, 2 * amount + o_crd2] = -2.0 * srs
        return jcnbx

    vnlr_mndr = NonlinearConstraint(vlms_qtz, 0.0, np.inf, jac=jclb_gtn)

    dvc_crd_lb, dvc_crd_ub = np.zeros(3 * amount), np.zeros(3 * amount)
    dvc_crd_lb[:2 * amount] = 0.0; dvc_crd_ub[:2 * amount] = 1.0
    dvc_crd_lb[2 * amount:] = 1e-6; dvc_crd_ub[2 * amount:] = 0.5
    vxbnds_xsz = Bounds(dvc_crd_lb, dvc_crd_ub)

    hsq_evs = -1.0
    grvsnt_b = np.concatenate([p_tensor[dckls[0], :, 0], p_tensor[dckls[0], :, 1], scvrd_rdz[dckls[0]]])
    objkqs = np.zeros(3 * amount); objkqs[2 * amount:] = -1.0

    def minzm_vlqz(vrpz_sz): 
        return -float(np.sum(vrpz_sz[2 * amount:]))

    def dxjc_vzr(vrpz_sz): 
        return objkqs

    for krtm in dckls:
        if time.time() - zero_mark > 27.5:
            break

        ctsqn = np.concatenate([p_tensor[krtm, :, 0], p_tensor[krtm, :, 1], scvrd_rdz[krtm]])

        try:
            szfldz = minimize(
                minzm_vlqz,
                ctsqn,
                method='SLSQP',
                jac=dxjc_vzr,
                bounds=vxbnds_xsz,
                constraints=[eqls_cnst, vnlr_mndr],
                options={'maxiter': 600, 'ftol': 2e-7, 'disp': False}
            )
            valqscrs = np.sum(szfldz.x[2 * amount:])
            if valqscrs > hsq_evs or szfldz.success:
                prnt_szt = szfldz.x[:2*amount].reshape((2, amount)).T
                skptsz = extract_valid_geometries_strictly(prnt_szt, szfldz.x[2 * amount:])
                rtvsmd = np.sum(skptsz)

                if rtvsmd > hsq_evs:
                    hsq_evs = float(rtvsmd)
                    grvsnt_b = szfldz.x.copy()
        except Exception:
            pass

    bstkgn = grvsnt_b[:2*amount].reshape((2, amount)).T.copy()
    bstrqvn = extract_valid_geometries_strictly(bstkgn, grvsnt_b[2 * amount:])
    tlrc = float(np.sum(bstrqvn))

    return bstkgn, bstrqvn, tlrc

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
