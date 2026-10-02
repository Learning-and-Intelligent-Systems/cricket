"""VAMP droid sphere models vs the exact hulls.

  vamp_bench.py make OUT.npz            fixed random confs + table-scene cuboids
  vamp_bench.py run  VAMPDIR SET.npz OUT.npz   (PYTHONPATH-free: inserts VAMPDIR)
  vamp_bench.py exact SET.npz SRDF OUT.npz     exact verdicts (Coal, master hulls)
  vamp_bench.py score SET.npz EXACT.npz RUN.npz ...
"""
import sys
import time

import numpy as np

LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
HI = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def make(out, n=20000, seed=3):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(LO, HI, size=(n, 7))
    # the droid scenes' table (top z = -0.02, centred x = 0.5) and clutter on it
    boxes = [((0.5, 0.0, -0.395), (0, 0, 0), (0.4, 0.6, 0.375))]
    for _ in range(10):
        hx, hy, hz = rng.uniform(0.02, 0.08, 3)
        c = (rng.uniform(0.25, 0.85), rng.uniform(-0.5, 0.5), -0.02 + hz)
        boxes.append((c, (0, 0, rng.uniform(-np.pi, np.pi)), (hx, hy, hz)))
    # a camera-stand-like post beside the robot
    boxes.append(((0.15, 0.45, 0.4), (0, 0, 0), (0.02, 0.02, 0.4)))
    np.savez(out, Q=Q, centers=np.array([b[0] for b in boxes]), eulers=np.array([b[1] for b in boxes]),
             halfs=np.array([b[2] for b in boxes]))


def run(vdir, setf, out):
    sys.path.insert(0, vdir)
    import vamp

    assert vamp.__file__.startswith(vdir), vamp.__file__
    d = np.load(setf)
    env = vamp.Environment()
    for c, e, h in zip(d["centers"], d["eulers"], d["halfs"]):
        env.add_cuboid(vamp.Cuboid(list(map(float, c)), list(map(float, e)), list(map(float, h))))
    empty = vamp.Environment()
    Q = d["Q"].astype(np.float32)
    r = vamp.droid
    for q in Q[:500]:  # warm-up
        r.validate(q, env)
    t0 = time.perf_counter()
    v_env = np.array([r.validate(q, env) for q in Q])
    t_env = time.perf_counter() - t0
    t0 = time.perf_counter()
    v_self = np.array([r.validate(q, empty) for q in Q])
    t_self = time.perf_counter() - t0
    np.savez(out, v_env=v_env, v_self=v_self, t_env=t_env, t_self=t_self, n_spheres=r.n_spheres())
    print(f"{vdir}: n_spheres {r.n_spheres()}  validate(env) {1e6 * t_env / len(Q):.2f} us/conf  "
          f"validate(no env) {1e6 * t_self / len(Q):.2f} us/conf  valid {v_env.mean():.3f}")


def exact(setf, srdf, out, sph_urdf="droid_spherized_cand2.urdf"):
    import coal
    from lxml import etree

    from fit_report import link_hulls
    from selfpair_audit import FK

    d = np.load(setf)
    fk = FK(sph_urdf)
    H = link_hulls()
    links = list(H)
    cvx = {}
    for l in links:
        pts = coal.StdVec_Vec3s()
        pts.extend(list(np.asarray(H[l][0].vertices, float)))
        cvx[l] = coal.Convex.convexHull(pts, False, "")
    import trimesh

    boxes = []
    for c, e, h in zip(d["centers"], d["eulers"], d["halfs"]):
        T = trimesh.transformations.euler_matrix(*e, axes="sxyz")
        boxes.append((coal.Box(*(2 * np.asarray(h))), coal.Transform3s(T[:3, :3], np.asarray(c, float))))
    dis = {frozenset((x.get("link1"), x.get("link2"))) for x in etree.parse(srdf).getroot().findall("disable_collisions")}
    pairs = [(a, b) for i, a in enumerate(links) for b in links[i + 1:] if frozenset((a, b)) not in dis]
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    env_hit, self_hit, env_clear = [], [], []
    for q in d["Q"]:
        T = fk(q)
        tf = {l: coal.Transform3s(T[l][:3, :3], T[l][:3, 3]) for l in links}
        dmin = np.inf
        for l in links:
            for g, gt in boxes:
                res.clear()
                dmin = min(dmin, coal.distance(cvx[l], tf[l], g, gt, req, res))
        sh = False
        for a, b in pairs:
            res.clear()
            if coal.distance(cvx[a], tf[a], cvx[b], tf[b], req, res) < 0:
                sh = True
                break
        env_hit.append(dmin < 0)
        env_clear.append(dmin)
        self_hit.append(sh)
    np.savez(out, env_hit=np.array(env_hit), self_hit=np.array(self_hit), env_clear=np.array(env_clear))
    print("exact: env-colliding", np.mean(env_hit), "self-colliding", np.mean(self_hit))


def score(setf, exf, *runs):
    e = np.load(exf)
    ok_exact = ~(e["env_hit"] | e["self_hit"])
    clear = e["env_clear"]
    for rf in runs:
        r = np.load(rf)
        v = r["v_env"]
        fp = (~v & ok_exact).mean()
        fn = (v & ~ok_exact).sum()
        fp_near = [(~v & ok_exact & (clear < t)).sum() / max(1, (ok_exact & (clear < t)).sum()) for t in (0.01, 0.02, 0.05)]
        print(f"{rf}: spheres {int(r['n_spheres'])}  valid {v.mean():.3f} (exact {ok_exact.mean():.3f})  "
              f"false-invalid {100 * fp:.2f}% of all  false-valid {fn}  "
              f"false-invalid among exact-valid within 1/2/5 cm: {', '.join(f'{100 * x:.0f}%' for x in fp_near)}  "
              f"{1e6 * float(r['t_env']) / len(v):.2f} us/conf")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "exact":
        exact(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3], *sys.argv[4:])
