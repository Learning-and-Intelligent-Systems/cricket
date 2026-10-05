"""Sphere-model vs exact-hull verdicts for whole-robot confs in a box scene,
in Python (no kernel build): the acceptance check on the push_or_pick edge
confs, and the stowed default conf at those bases.

VAMP's env for that run: the table is a PERMANENT body > 0.3 m, so it rides as
its axis-aligned bounds cuboid with x/y half-extents grown by distance_thr
(z not grown; cm_vamp.add_permanent_to_vamp_env); VAMP's default thr is
robo_config.COLLISION_DISTANCE_THR = 1 mm.
    scene_check.py SPHERIZED.urdf [THR_MM]"""
import pickle
import sys

import numpy as np

from rb_common import FK, MASTER, OPEN_FINGERS, link_geoms, link_spheres

EDGE_PKL = (
    "/private/tmp/claude-501/-Users-tlp-Documents-GitHub-NQR/"
    "b0776b8f-eaf6-43b5-af0b-acc04ba8d6b1/scratchpad/edge_confs.pkl"
)
TABLE = (np.array([0.75, 0.0, 0.35]), np.array([0.5, 0.3, 0.35]))  # centre, half extents
AXLE = 0.228


def vamp_coords(conf):
    b = np.asarray(conf["base"], float).copy()
    b[2] = (b[2] + np.pi) % (2 * np.pi) - np.pi
    b[0] -= AXLE * np.cos(b[2])
    b[1] -= AXLE * np.sin(b[2])
    return np.hstack([b, conf["torso"], conf["left"], conf["right"]])


def box_sd(P, c, h):
    """Signed distance from points to an axis-aligned box (negative inside)."""
    q = np.abs(P - c) - h
    out = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(axis=1), 0)
    return out + inside


class SphereModel:
    def __init__(self, urdf):
        from rb_common import VAMP_JOINTS

        self.fk = FK(urdf)
        self.S = {l: s for l, s in link_spheres(urdf, merge_parts=False).items() if s}
        self.joints = VAMP_JOINTS

    def world(self, q23):
        T = self.fk(dict(zip(self.joints, map(float, q23))))
        out = {}
        for l, s in self.S.items():
            C = np.array([c for c, _ in s]) @ T[l][:3, :3].T + T[l][:3, 3]
            out[l] = (C, np.array([r for _, r in s]))
        return out

    def box_clearance(self, q23, c, h):
        """{link: min(sphere-to-box distance - r)} (negative = collides)."""
        return {l.split("__s")[0]: (box_sd(C, c, h) - R).min() for l, (C, R) in self.world(q23).items()}


class ExactModel:
    """Master hulls placed at the HPN conf (base via the VAMP base frame), Coal."""

    def __init__(self):
        import coal

        self.coal = coal
        self.fk = FK(MASTER)
        self.cvx = {}
        for l, ps in link_geoms(MASTER).items():
            cs = []
            for p in ps:
                if p[0] == "sphere":
                    cs.append(("sphere", coal.Sphere(p[2]), p[1]))
                else:
                    h = p[1]
                    pts = coal.StdVec_Vec3s()
                    pts.extend(list(np.asarray(h.vertices, float)))
                    cs.append(("cvx", coal.Convex.convexHull(pts, False, "Qt"), None))
            self.cvx[l] = cs

    def link_T(self, conf, head=None):
        vals = {}
        for i, n in enumerate([f"torso_{k}" for k in range(6)]):
            vals[n] = float(conf["torso"][i])
        for i in range(7):
            vals[f"left_arm_{i}"] = float(conf["left"][i])
            vals[f"right_arm_{i}"] = float(conf["right"][i])
        hd = conf.get("head", [0.0, 0.0]) if head is None else head
        vals["head_0"], vals["head_1"] = float(hd[0]), float(hd[1])
        vals.update(OPEN_FINGERS)
        T = self.fk(vals)
        Tb = np.linalg.inv(T["base"])
        vc = vamp_coords(conf)
        W = np.eye(4)
        W[:3, :3] = [[np.cos(vc[2]), -np.sin(vc[2]), 0], [np.sin(vc[2]), np.cos(vc[2]), 0], [0, 0, 1]]
        W[:2, 3] = vc[:2]
        return {l: W @ Tb @ T[l] for l in T}

    def box_clearance(self, conf, c, h, head=None):
        coal = self.coal
        T = self.link_T(conf, head)
        box = coal.Box(*(2 * np.asarray(h, float)))
        btf = coal.Transform3s(np.eye(3), np.asarray(c, float))
        req, res = coal.DistanceRequest(), coal.DistanceResult()
        out = {}
        for l, cs in self.cvx.items():
            d = np.inf
            for kind, g, ctr in cs:
                M = T[l].copy()
                if kind == "sphere":
                    M[:3, 3] = M[:3, :3] @ ctr + M[:3, 3]
                res.clear()
                d = min(d, coal.distance(g, coal.Transform3s(M[:3, :3], M[:3, 3]), box, btf, req, res))
            out[l] = d
        return out


def default_conf(base, knee="back"):
    torso = {"forward": np.deg2rad([0.0, 30.0, -60.0, 45.0, 0.0, 0.0]), "back": np.deg2rad([0.0, -30.0, 60.0, -15.0, 0.0, 0.0])}[knee]
    return {"base": np.asarray(base, float), "torso": torso, "left": np.zeros(7), "right": np.zeros(7), "head": np.array([0.0, np.deg2rad(40)])}


def main(urdf, thr=0.001, verbose=True):
    found = pickle.load(open(EDGE_PKL, "rb"))
    sm, ex = SphereModel(urdf), ExactModel()
    c, h = TABLE
    hv = h + np.array([thr, thr, 0.0])
    rows, n_ok, n_all = [], 0, 0
    worst_exact = np.inf
    confs = [(e, i, q) for e, qs in found for i, q in enumerate(qs)]
    bases = {}
    for e, i, q in confs:
        bases.setdefault(tuple(np.round(np.asarray(q["base"], float), 3)), q)
    for e, i, q in confs:
        vc = vamp_coords(q)
        sc = sm.box_clearance(vc, c, hv)
        xc = ex.box_clearance(q, c, h)
        worst_exact = min(worst_exact, min(xc.values()))
        ok = min(sc.values()) > 0
        n_ok += ok
        n_all += 1
        rows.append((e, i, ok, min(xc.values()), sorted(sc.items(), key=lambda kv: kv[1])[:3]))
    stow_ok = 0
    stow_rows = []
    for b, q in bases.items():
        dq = default_conf(q["base"])
        sc = sm.box_clearance(vamp_coords(dq), c, hv)
        xc = ex.box_clearance(dq, c, h)
        ok = min(sc.values()) > 0
        stow_ok += ok
        stow_rows.append((b, ok, min(xc.values()), sorted(sc.items(), key=lambda kv: kv[1])[:2], min(xc, key=xc.get)))
    if verbose:
        for e, i, ok, xmin, worst in rows:
            if not ok:
                print("REJECT", e, i, f"exact min clearance {1000 * xmin:.1f} mm;", [(l, round(1000 * v, 1)) for l, v in worst])
        for b, ok, xmin, worst, xl in stow_rows:
            if not ok:
                print("STOW REJECT", b, f"exact min {1000 * xmin:.1f} mm ({xl});", [(l, round(1000 * v, 1)) for l, v in worst])
    print(f"{urdf}: thr {1000 * thr:.0f} mm  edge confs accepted {n_ok}/{n_all}  stow confs accepted {stow_ok}/{len(stow_rows)}  "
          f"(exact min clearance over the edge confs {1000 * worst_exact:.1f} mm)")
    return n_ok, n_all, stow_ok, len(stow_rows)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.001)
