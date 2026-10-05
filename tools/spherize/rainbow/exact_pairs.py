"""Exact self-collision census of Rainbow's master hulls (Coal convex hulls,
qhull), over random confs of the 20 VAMP joint DOF (torso + both arms, master
limits), the head at pan = tilt = 0 and the fingers open (as VAMP models them).
Every link pair NOT in the derived exclusions
(qaa.robot_description.derived(PROFILE).self_collision_exclusions) is scored:
collision count, min distance (+ its conf), and how often the pair sits within
5/10/20/30 mm (sphere covers protrude up to delta + pad, so a pair closer than
twice that can overlap in spheres while the hulls are apart).
    exact_pairs.py N_CONF OUT.npz [SEED] [WORKERS]"""
import itertools
import sys
import time

import numpy as np

THRESH = (0.0, 0.005, 0.010, 0.020, 0.030)


def setup():
    import coal

    from rb_common import MASTER, link_hulls

    H = link_hulls(MASTER)
    cvx, bs = {}, {}
    for l, hs in H.items():
        h = hs[0]
        pts = coal.StdVec_Vec3s()
        pts.extend(list(np.asarray(h.vertices, dtype=float)))
        cvx[l] = coal.Convex.convexHull(pts, False, "Qt")
        c = h.vertices.mean(axis=0)
        bs[l] = (c, np.linalg.norm(h.vertices - c, axis=1).max())
    return cvx, bs


def pairs_to_check():
    from qaa.robot_description import derived

    from rb_common import PROFILE

    from rb_common import MASTER, link_hulls

    links = list(link_hulls(MASTER))
    ex = {frozenset(p) for p in derived(PROFILE).self_collision_exclusions}
    return [(a, b) for a, b in itertools.combinations(links, 2) if frozenset((a, b)) not in ex]


def work(args):
    Q, pairs = args
    import coal

    from rb_common import FK, MASTER, OPEN_FINGERS, vamp_values

    cvx, bs = setup()
    fk = FK(MASTER)
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    P = len(pairs)
    D = np.full((len(Q), P), np.inf, dtype=np.float32)
    for i, q in enumerate(Q):
        T = fk({**vamp_values(q), **OPEN_FINGERS})
        tf = {l: coal.Transform3s(T[l][:3, :3], T[l][:3, 3]) for l in cvx}
        wc = {l: T[l][:3, :3] @ bs[l][0] + T[l][:3, 3] for l in cvx}
        for k, (a, b) in enumerate(pairs):
            lb = np.linalg.norm(wc[a] - wc[b]) - bs[a][1] - bs[b][1]
            if lb > 0.05:
                D[i, k] = lb
                continue
            res.clear()
            D[i, k] = coal.distance(cvx[a], tf[a], cvx[b], tf[b], req, res)
    return D


def main(n, out, seed=0, workers=6):
    from multiprocessing import get_context

    from rb_common import master_bounds, random_confs

    pairs = pairs_to_check()
    Q = random_confs(n, seed=seed, bounds=master_bounds())
    chunks = np.array_split(Q, workers * 4)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        Ds = pool.map(work, [(c, pairs) for c in chunks])
    D = np.vstack(Ds)
    print(f"{n} confs x {len(pairs)} pairs in {time.time() - t0:.0f}s")
    np.savez(out, D=D, Q=Q, pairs=np.array(pairs))
    summarize(D, pairs)


def summarize(D, pairs, top=None):
    n = len(D)
    rows = []
    for k, (a, b) in enumerate(pairs):
        d = D[:, k]
        rows.append((d.min(), a, b, [(d < t).mean() for t in THRESH], int(np.argmin(d))))
    rows.sort()
    print(f"{'pair':52s} {'min mm':>8s} " + " ".join(f"{'<' + str(int(1000 * t)) + 'mm%':>8s}" for t in THRESH))
    for dmin, a, b, fr, _ in rows[: top or len(rows)]:
        if dmin > 0.06:
            break
        print(f"{a + ' -- ' + b:52s} {1000 * dmin:8.1f} " + " ".join(f"{100 * f:8.3f}" for f in fr))
    print("pairs with min distance > 60 mm:", sum(r[0] > 0.06 for r in rows), "of", len(rows))


if __name__ == "__main__":
    main(
        int(sys.argv[1]),
        sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
        int(sys.argv[4]) if len(sys.argv) > 4 else 6,
    )
