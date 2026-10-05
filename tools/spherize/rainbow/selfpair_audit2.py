"""Self-collision audit of a spherized Rainbow model against the master hulls.

Over random confs of the 20 VAMP DOF (torso + both arms; master limits, head at
0, fingers open), for every link pair NOT in the derived exclusions: how often
the SPHERES overlap (VAMP's verdict if the pair is enabled) and how often the
exact HULLS do (Coal), so: spurious hits (spheres only) and misses (hulls only:
impossible for a conservative cover; reported as a check).
Also scores the named postures (home, default, stow ...).

    selfpair_audit2.py SPHERIZED.urdf N OUT.npz [SEED] [WORKERS]"""
import itertools
import sys
import time

import numpy as np


def named_confs():
    """23-vecs (base 0) of the postures the planner starts from / returns to."""
    d = np.deg2rad
    torso = {"forward": d([0.0, 30.0, -60.0, 45.0, 0.0, 0.0]), "back": d([0.0, -30.0, 60.0, -15.0, 0.0, 0.0]), "zero": np.zeros(6)}
    right_home = d([60.0, -15.0, 0.0, -90.0, 0.0, -90.0, 0.0])
    left_home_mirror = d([60.0, 15.0, 0.0, -90.0, 0.0, -90.0, 0.0])
    out = {}
    for tk, tv in torso.items():
        out[f"default_{tk}"] = np.hstack([np.zeros(3), tv, np.zeros(7), np.zeros(7)])
        out[f"home_{tk}"] = np.hstack([np.zeros(3), tv, np.zeros(7), right_home])
        out[f"home_both_{tk}"] = np.hstack([np.zeros(3), tv, left_home_mirror, right_home])
    return out


def setup(sph_urdf):
    import coal

    from rb_common import MASTER, link_hulls, link_spheres

    H = link_hulls(MASTER)
    S = {l: s for l, s in link_spheres(sph_urdf, merge_parts=True).items() if s}
    cvx = {}
    for l, hs in H.items():
        pts = coal.StdVec_Vec3s()
        pts.extend(list(np.asarray(hs[0].vertices, dtype=float)))
        cvx[l] = coal.Convex.convexHull(pts, False, "Qt")
    SC = {l: np.array([c for c, _ in s]) for l, s in S.items()}
    SR = {l: np.array([r for _, r in s]) for l, s in S.items()}
    # link bounding spheres (loose: centroid + max reach)
    BS = {}
    for l in S:
        c = SC[l].mean(axis=0)
        BS[l] = (c, (np.linalg.norm(SC[l] - c, axis=1) + SR[l]).max())
    return cvx, SC, SR, BS


def pairs_to_check(sph_urdf):
    from qaa.robot_description import derived

    from rb_common import PROFILE

    from rb_common import MASTER, link_hulls, link_spheres

    S = {l for l, s in link_spheres(sph_urdf, merge_parts=True).items() if s}
    links = [l for l in link_hulls(MASTER) if l in S]
    ex = {frozenset(p) for p in derived(PROFILE).self_collision_exclusions}
    return [(a, b) for a, b in itertools.combinations(links, 2) if frozenset((a, b)) not in ex]


def score(Q, sph_urdf, pairs):
    """(sphere gap, exact distance) per conf x pair; gaps clipped at +0.05."""
    import coal

    from rb_common import FK, MASTER, OPEN_FINGERS, vamp_values

    cvx, SC, SR, BS = setup(sph_urdf)
    fk = FK(MASTER)
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    G = np.full((len(Q), len(pairs)), 0.05, dtype=np.float32)
    D = np.full((len(Q), len(pairs)), 0.05, dtype=np.float32)
    for i, q in enumerate(Q):
        T = fk({**vamp_values(q), **OPEN_FINGERS})
        W = {l: SC[l] @ T[l][:3, :3].T + T[l][:3, 3] for l in SC}
        B = {l: T[l][:3, :3] @ BS[l][0] + T[l][:3, 3] for l in SC}
        for k, (a, b) in enumerate(pairs):
            lb = np.linalg.norm(B[a] - B[b]) - BS[a][1] - BS[b][1]
            if lb > 0.05:
                continue
            g = (np.linalg.norm(W[a][:, None] - W[b][None], axis=2) - SR[a][:, None] - SR[b][None]).min()
            G[i, k] = min(g, 0.05)
            res.clear()
            ta = coal.Transform3s(T[a][:3, :3], T[a][:3, 3])
            tb = coal.Transform3s(T[b][:3, :3], T[b][:3, 3])
            D[i, k] = min(coal.distance(cvx[a], ta, cvx[b], tb, req, res), 0.05)
    return G, D


def _work(args):
    return score(*args)


def main(sph_urdf, n, out, seed=0, workers=8):
    from multiprocessing import get_context

    from rb_common import master_bounds, random_confs

    pairs = pairs_to_check(sph_urdf)
    Q = random_confs(n, seed=seed, bounds=master_bounds())
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_work, [(c, sph_urdf, pairs) for c in np.array_split(Q, workers * 4)])
    G = np.vstack([p[0] for p in parts])
    D = np.vstack([p[1] for p in parts])
    names = named_confs()
    NG, ND = score(np.array(list(names.values())), sph_urdf, pairs)
    print(f"{n} confs x {len(pairs)} pairs in {time.time() - t0:.0f}s")
    np.savez(out, G=G, D=D, Q=Q, pairs=np.array(pairs), NG=NG, ND=ND, names=np.array(list(names)))
    report(G, D, pairs, NG, ND, list(names))


def report(G, D, pairs, NG, ND, names, show=60):
    n = len(G)
    sph, ex = G < 0, D < 0
    rows = []
    for k, (a, b) in enumerate(pairs):
        rows.append((sph[:, k].mean(), ex[:, k].mean(), (sph[:, k] & ~ex[:, k]).mean(), int((ex[:, k] & ~sph[:, k]).sum()), a, b, D[:, k].min()))
    rows.sort(key=lambda r: -r[2])
    print(f"{'pair':52s} {'sph%':>6s} {'exact%':>7s} {'spur%':>6s} {'missed':>6s} {'exact min mm':>12s}")
    for s, e, sp, m, a, b, dmin in rows[:show]:
        print(f"{a + ' -- ' + b:52s} {100 * s:6.2f} {100 * e:7.2f} {100 * sp:6.2f} {m:6d} {1000 * dmin:12.1f}")
    anyx = ex.any(axis=1)
    anys = sph.any(axis=1)
    print(f"confs with any exact self-collision {100 * anyx.mean():.1f}%, any sphere hit {100 * anys.mean():.1f}%, "
          f"sphere-only (false-invalid) {100 * (anys & ~anyx).mean():.1f}%, exact-only (false-valid) {int((anyx & ~anys).sum())}")
    print("named postures: sphere hits where the hulls are clear (pair: sphere gap / exact distance, mm)")
    for i, nm in enumerate(names):
        bad = [(pairs[k], round(1000 * float(NG[i, k]), 1), round(1000 * float(ND[i, k]), 1)) for k in range(len(pairs)) if NG[i, k] < 0]
        print(f"  {nm:18s}", bad if bad else "clean")


if __name__ == "__main__":
    main(
        sys.argv[1],
        int(sys.argv[2]),
        sys.argv[3],
        int(sys.argv[4]) if len(sys.argv) > 4 else 0,
        int(sys.argv[5]) if len(sys.argv) > 5 else 8,
    )
