"""Can a link pair collide at all within the VAMP DOF? For each pair the random
census never saw collide, minimise the exact hull distance (Coal) over the
joints on the kinematic path between the two links (master limits; head and
fingers as VAMP fixes them), from the census's closest confs (bounded
Nelder-Mead + L-BFGS-B, then a dense grid when the path has <= 2 joints).
    pair_minimize.py CENSUS.npz OUT.json [WORKERS]"""
import json
import os
import sys

import numpy as np

NSEED = int(os.environ.get('NSEED', 4))
NRAND = int(os.environ.get('NRAND', 0))


def path_joints(fk, a, b):
    """Movable VAMP joints on the tree path between links a and b."""
    from rb_common import VAMP_JOINTS

    parent = {j["child"]: j for j in fk.joints}

    def chain(l):
        out = []
        while l in parent:
            j = parent[l]
            out.append(j)
            l = j["parent"]
        return out

    ca, cb = chain(a), chain(b)
    na, nb = {j["name"] for j in ca}, {j["name"] for j in cb}
    path = (na ^ nb) & set(VAMP_JOINTS) - {"x", "y", "theta"}
    return [n for n in VAMP_JOINTS if n in path]


def work(args):
    pairs, seeds = args
    import coal
    from scipy.optimize import minimize

    from rb_common import FK, MASTER, OPEN_FINGERS, VAMP_JOINTS, link_hulls, master_bounds, vamp_values

    H = link_hulls(MASTER)
    fk = FK(MASTER)
    lo, hi = master_bounds()
    cvx = {}
    for l, hs in H.items():
        pts = coal.StdVec_Vec3s()
        pts.extend(list(np.asarray(hs[0].vertices, float)))
        cvx[l] = coal.Convex.convexHull(pts, False, "Qt")
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    out = {}
    for (a, b), Q0 in zip(pairs, seeds):
        idx = [VAMP_JOINTS.index(n) for n in path_joints(fk, a, b)]

        def dist(qsub, q0):
            q = q0.copy()
            q[idx] = np.clip(qsub, lo[idx], hi[idx])
            T = fk({**vamp_values(q), **OPEN_FINGERS})
            res.clear()
            return coal.distance(
                cvx[a], coal.Transform3s(T[a][:3, :3], T[a][:3, 3]), cvx[b], coal.Transform3s(T[b][:3, :3], T[b][:3, 3]), req, res
            )

        best = np.inf
        bestq = None
        if not idx:  # rigid in VAMP's model: one value
            best = dist(np.zeros(0), Q0[0])
        else:
            for q0 in Q0:
                x0 = q0[idx]
                r1 = minimize(lambda s: dist(s, q0), x0, method="Nelder-Mead", options={"maxfev": 600, "xatol": 1e-4, "fatol": 1e-5})
                r2 = minimize(lambda s: dist(s, q0), r1.x, method="L-BFGS-B", bounds=list(zip(lo[idx], hi[idx])), options={"maxfun": 400})
                for r in (r1, r2):
                    if r.fun < best:
                        best, bestq = float(r.fun), np.clip(r.x, lo[idx], hi[idx])
            if len(idx) <= 2:  # exhaustive grid at 1 deg
                axes = [np.linspace(lo[i], hi[i], max(2, int(np.degrees(hi[i] - lo[i])) + 1)) for i in idx]
                G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(idx))
                for g in G:
                    d = dist(g, Q0[0])
                    if d < best:
                        best, bestq = d, g
        out[f"{a}|{b}"] = dict(min=best, joints=[VAMP_JOINTS[i] for i in idx], q=None if bestq is None else list(map(float, bestq)))
    return out


def main(census, out, workers=8):
    from multiprocessing import get_context

    d = np.load(census, allow_pickle=True)
    D, Q, pairs = d["D"], d["Q"], [tuple(p) for p in d["pairs"]]
    todo, seeds = [], []
    for k, p in enumerate(pairs):
        if D[:, k].min() < 0:
            continue  # already seen colliding
        order = np.argsort(D[:, k])[:NSEED]
        rnd = np.random.default_rng(k).choice(len(Q), NRAND, replace=False)
        todo.append(p)
        seeds.append(np.vstack([Q[order], Q[rnd]]))
    chunks = [(todo[i::workers * 2], seeds[i::workers * 2]) for i in range(workers * 2)]
    with get_context("spawn").Pool(workers) as pool:
        parts = pool.map(work, chunks)
    res = {}
    for p in parts:
        res.update(p)
    json.dump(res, open(out, "w"), indent=1)
    rows = sorted(res.items(), key=lambda kv: kv[1]["min"])
    print(f"{len(rows)} pairs never seen colliding; minimised exact distance (mm):")
    for k, v in rows:
        print(f"  {k:52s} {1000 * v['min']:8.1f}   path joints {v['joints']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
