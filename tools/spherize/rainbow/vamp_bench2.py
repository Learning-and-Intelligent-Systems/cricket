"""Rainbow VAMP kernels vs the master's exact hulls, in a table scene.

  vamp_bench2.py make OUT.npz [N]                 confs + scene (fixed seed)
  vamp_bench2.py exact SET.npz SRDF OUT.npz       exact verdicts (Coal): env, self on
                                                  the SRDF's checked pairs, self on every
                                                  pair the sims check (derived exclusions)
  vamp_bench2.py spheres SET.npz URDF SRDF OUT.npz  the sphere model's verdicts in Python
                                                  (what the kernel must reproduce)
  vamp_bench2.py run VAMPDIR SET.npz OUT.npz      kernel verdicts + timing (VAMPDIR first
                                                  on sys.path; '-' = the installed vamp)
  vamp_bench2.py score SET.npz EXACT.npz RUN.npz ...

Confs, half each: "random" (every joint uniform in the master limits) and
"typical" (torso at a home knee, left arm at rest, right arm uniform); the
base (VAMP's base-link frame) uniform around the table, heading uniform.
Scene: the push_or_pick table (1.0 x 0.6 x 0.7 m at (0.75, 0, 0.35)), ten
boxes on it, a post, and a shelf board at 1.1 m."""
import itertools
import sys
import time

import numpy as np

from rb_common import FK, MASTER, OPEN_FINGERS, VAMP_JOINTS, link_geoms, link_spheres, master_bounds, read_disabled


def make(out, n=20000, seed=3):
    rng = np.random.default_rng(seed)
    lo, hi = master_bounds()
    Q = rng.uniform(lo, hi, size=(n, 23))
    half = n // 2
    knees = [np.deg2rad([0.0, -30.0, 60.0, -15.0, 0.0, 0.0]), np.deg2rad([0.0, 30.0, -60.0, 45.0, 0.0, 0.0])]
    for k in range(half, n):
        Q[k, 3:9] = knees[k % 2]
        Q[k, 9:16] = 0.0
    Q[:, 0] = rng.uniform(-0.4, 1.9, n)
    Q[:, 1] = rng.uniform(-1.2, 1.2, n)
    Q[:, 2] = rng.uniform(-np.pi, np.pi, n)
    boxes = [((0.75, 0.0, 0.35), (0, 0, 0), (0.5, 0.3, 0.35))]
    for _ in range(10):
        hx, hy, hz = rng.uniform(0.02, 0.07, 3)
        boxes.append(((rng.uniform(0.3, 1.2), rng.uniform(-0.25, 0.25), 0.7 + hz), (0, 0, rng.uniform(-np.pi, np.pi)), (hx, hy, hz)))
    boxes.append(((0.1, 0.6, 0.6), (0, 0, 0), (0.03, 0.03, 0.6)))  # a post
    boxes.append(((0.75, -0.75, 1.1), (0, 0, 0), (0.4, 0.15, 0.01)))  # a shelf board
    np.savez(out, Q=Q, n_random=half, centers=np.array([b[0] for b in boxes], float),
             eulers=np.array([b[1] for b in boxes], float), halfs=np.array([b[2] for b in boxes], float))


def _base_T(q):
    W = np.eye(4)
    c, s = np.cos(q[2]), np.sin(q[2])
    W[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    W[:2, 3] = q[:2]
    return W


def _exact_chunk(args):
    import coal
    import trimesh
    from qaa.robot_description import derived

    from rb_common import PROFILE

    Q, centers, eulers, halfs, srdf = args
    fk = FK(MASTER)
    G = link_geoms(MASTER)
    cvx, bs = {}, {}
    for l, ps in G.items():
        p = ps[0]
        if p[0] == "sphere":
            cvx[l] = (coal.Sphere(p[2]), p[1])
            bs[l] = (p[1], p[2])
        else:
            pts = coal.StdVec_Vec3s()
            pts.extend(list(np.asarray(p[1].vertices, float)))
            cvx[l] = (coal.Convex.convexHull(pts, False, "Qt"), None)
            c = p[1].vertices.mean(axis=0)
            bs[l] = (c, np.linalg.norm(p[1].vertices - c, axis=1).max())
    links = list(cvx)
    boxes = []
    for c, e, h in zip(centers, eulers, halfs):
        T = trimesh.transformations.euler_matrix(*e, axes="sxyz")
        boxes.append((coal.Box(*(2 * np.asarray(h))), coal.Transform3s(T[:3, :3], np.asarray(c, float)), np.asarray(c, float), np.linalg.norm(h)))
    dis = read_disabled(srdf)
    ex = {frozenset(p) for p in derived(PROFILE).self_collision_exclusions}
    allp = [(a, b) for a, b in itertools.combinations(links, 2) if frozenset((a, b)) not in ex]
    chk = {p for p in allp if frozenset(p) not in dis}
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    env_clear, self_chk, self_all = [], [], []
    for q in Q:
        T = fk({**dict(zip(VAMP_JOINTS, map(float, q))), **OPEN_FINGERS})
        M = _base_T(q) @ np.linalg.inv(T["base"])
        tf, wc = {}, {}
        for l in links:
            W = M @ T[l]
            g, ctr = cvx[l]
            wc[l] = W[:3, :3] @ bs[l][0] + W[:3, 3]
            if ctr is not None:
                W = W.copy()
                W[:3, 3] = W[:3, :3] @ ctr + W[:3, 3]
            tf[l] = coal.Transform3s(W[:3, :3], W[:3, 3])
        dmin = np.inf
        for l in links:
            for g, gt, gc, gr in boxes:
                lb = np.linalg.norm(wc[l] - gc) - bs[l][1] - gr
                if lb > 0.06:
                    dmin = min(dmin, lb)
                    continue
                res.clear()
                dmin = min(dmin, coal.distance(cvx[l][0], tf[l], g, gt, req, res))
        env_clear.append(dmin)
        sc = sa = False
        for a, b in allp:
            if np.linalg.norm(wc[a] - wc[b]) > bs[a][1] + bs[b][1]:
                continue
            res.clear()
            if coal.distance(cvx[a][0], tf[a], cvx[b][0], tf[b], req, res) < 0:
                sa = True
                if (a, b) in chk:
                    sc = True
                    break
        self_chk.append(sc)
        self_all.append(sa or sc)
    return np.array(env_clear), np.array(self_chk), np.array(self_all)


def exact(setf, srdf, out, workers=8):
    """Exact verdicts. self_all: some pair the sims check collides (the scan
    stops at the first CHECKED hit, so self_all is exact only where
    self_chk is False -- which is where it matters)."""
    from multiprocessing import get_context

    d = np.load(setf)
    chunks = np.array_split(d["Q"], workers * 4)
    with get_context("spawn").Pool(int(workers)) as pool:
        parts = pool.map(_exact_chunk, [(c, d["centers"], d["eulers"], d["halfs"], srdf) for c in chunks])
    env_clear = np.concatenate([p[0] for p in parts])
    self_chk = np.concatenate([p[1] for p in parts])
    self_all = np.concatenate([p[2] for p in parts])
    np.savez(out, env_clear=env_clear, self_chk=self_chk, self_all=self_all)
    print("exact: env-colliding", np.mean(env_clear < 0), "self (checked pairs)", np.mean(self_chk), "self (all sim pairs)", np.mean(self_all))


def spheres(setf, urdf, srdf, out):
    """The sphere model's verdict, computed directly (env: sphere vs box; self:
    every checked pair's sphere pairs)."""
    import trimesh

    d = np.load(setf)
    fk = FK(urdf)
    S = {l: s for l, s in link_spheres(urdf, merge_parts=False).items() if s}
    dis = read_disabled(srdf)
    links = list(S)
    pairs = [(a, b) for a, b in itertools.combinations(links, 2) if frozenset((a, b)) not in dis]
    boxes = []
    for c, e, h in zip(d["centers"], d["eulers"], d["halfs"]):
        T = trimesh.transformations.euler_matrix(*e, axes="sxyz")
        boxes.append((T[:3, :3], np.asarray(c, float), np.asarray(h, float)))
    SC = {l: np.array([c for c, _ in S[l]]) for l in links}
    SR = {l: np.array([r for _, r in S[l]]) for l in links}
    v_env, v_self = [], []
    for q in d["Q"]:
        T = fk(dict(zip(VAMP_JOINTS, map(float, q))))
        W = {l: SC[l] @ T[l][:3, :3].T + T[l][:3, 3] for l in links}
        C = np.vstack([W[l] for l in links])
        R = np.concatenate([SR[l] for l in links])
        env_hit = False
        for Rb, cb, hb in boxes:
            P = (C - cb) @ Rb  # into the box frame
            qd = np.abs(P) - hb
            dist = np.linalg.norm(np.maximum(qd, 0), axis=1) + np.minimum(qd.max(axis=1), 0)
            if (dist < R).any():
                env_hit = True
                break
        self_hit = False
        for a, b in pairs:
            if (np.linalg.norm(W[a][:, None] - W[b][None], axis=2) < SR[a][:, None] + SR[b][None]).any():
                self_hit = True
                break
        v_env.append(not env_hit)
        v_self.append(not self_hit)
    v_env, v_self = np.array(v_env), np.array(v_self)
    np.savez(out, v=v_env & v_self, v_selfonly=v_self)
    print(f"sphere model: valid {np.mean(v_env & v_self):.3f}  (self-valid {v_self.mean():.3f})")


def run(vdir, setf, out, module="rainbow"):
    if vdir != "-":
        sys.path.insert(0, vdir)
    import vamp

    if vdir != "-":
        assert vamp.__file__.startswith(vdir), vamp.__file__
    d = np.load(setf)
    env = vamp.Environment()
    for c, e, h in zip(d["centers"], d["eulers"], d["halfs"]):
        env.add_cuboid(vamp.Cuboid(list(map(float, c)), list(map(float, e)), list(map(float, h))))
    empty = vamp.Environment()
    Q = d["Q"].astype(np.float32)
    r = getattr(vamp, module)
    for q in Q[:500]:
        r.validate(q, env)
    t0 = time.perf_counter()
    v_env = np.array([r.validate(q, env) for q in Q])
    t_env = time.perf_counter() - t0
    t0 = time.perf_counter()
    v_self = np.array([r.validate(q, empty) for q in Q])
    t_self = time.perf_counter() - t0
    # motion validation: short straight moves between consecutive confs' arms
    nm = min(2000, len(Q) - 1)
    t0 = time.perf_counter()
    for k in range(nm):
        a = Q[k].copy()
        b = a.copy()
        b[16:23] = a[16:23] + 0.3 * np.sign(np.random.default_rng(k).standard_normal(7))
        r.validate_motion(a, b, empty)
    t_mot = time.perf_counter() - t0
    t0 = time.perf_counter()
    for q in Q[:2000]:
        _ = np.array([(p.x, p.y, p.z) for p in r.fk(q)], dtype=np.float32)
    t_fk = time.perf_counter() - t0
    np.savez(out, v_env=v_env, v_self=v_self, t_env=t_env, t_self=t_self, t_mot=t_mot / nm, t_fk=t_fk / 2000, n_spheres=r.n_spheres())
    print(f"{vdir}: n_spheres {r.n_spheres()}  validate(scene) {1e6 * t_env / len(Q):.2f} us  validate(empty) {1e6 * t_self / len(Q):.2f} us  "
          f"validate_motion(0.3 rad x7, empty) {1e6 * t_mot / nm:.1f} us  fk->numpy {1e6 * t_fk / 2000:.0f} us  valid {v_env.mean():.3f}")


def score(setf, exf, *runs):
    d = np.load(setf)
    nr = int(d["n_random"])
    e = np.load(exf)
    clear = e["env_clear"]
    for label, sl in (("random", slice(0, nr)), ("typical", slice(nr, None)), ("all", slice(None))):
        ok_chk = ~((clear < 0) | e["self_chk"])[sl]
        ok_all = ~((clear < 0) | e["self_all"])[sl]
        cl = clear[sl]
        print(f"[{label}] exact-valid {ok_chk.mean():.3f} (vs every sim-checked pair {ok_all.mean():.3f})")
        for rf in runs:
            r = np.load(rf)
            v = (r["v_env"] if "v_env" in r else r["v"])[sl]
            fi = (~v & ok_chk).mean()
            fv = (v & ~ok_chk).sum()
            fv_all = (v & ~ok_all).sum()
            near = [(~v & ok_chk & (cl < t)).sum() / max(1, (ok_chk & (cl < t)).sum()) for t in (0.01, 0.02, 0.05)]
            print(f"   {rf:28s} valid {v.mean():.3f}  false-valid {fv} (vs every sim pair {fv_all})  false-invalid {100 * fi:.1f}% of confs = "
                  f"{100 * fi / max(1e-9, ok_chk.mean()):.1f}% of exact-valid; of exact-valid within 1/2/5 cm of the scene: "
                  + ", ".join(f"{100 * x:.0f}%" for x in near))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20000)
    elif cmd == "exact":
        exact(*sys.argv[2:5])
    elif cmd == "spheres":
        spheres(*sys.argv[2:6])
    elif cmd == "run":
        run(*sys.argv[2:6])
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3], *sys.argv[4:])
