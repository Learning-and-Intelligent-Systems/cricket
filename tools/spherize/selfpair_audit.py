"""Self-collision pair audit of a spherized VAMP model against the exact hulls:
over random arm confs, per link pair, how often the SPHERES overlap and how
often the exact HULLS do (Coal convex-convex), so: structural pairs (always
overlapping: must be disabled), spurious hits (spheres only), and blind spots
(a disabled pair whose hulls really collide)."""
import sys

import coal
import numpy as np
from lxml import etree

from fit_report import link_hulls, link_spheres


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_rot(ax, t):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


class FK:
    def __init__(self, urdf):
        r = etree.parse(urdf).getroot()
        self.joints = []
        for j in r.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rpy_mat(*rpy), xyz
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                                    child=j.find("child").get("link"), T=T,
                                    axis=[float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1],
                                    lo=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                                    hi=float(lim.get("upper")) if lim is not None and lim.get("upper") else None))
        self.active = [j for j in self.joints if j["type"] in ("revolute", "continuous", "prismatic")]
        children = {j["child"] for j in self.joints}
        self.root = [l.get("name") for l in r.findall("link") if l.get("name") not in children][0]

    def __call__(self, q):
        qi = {j["name"]: v for j, v in zip(self.active, q)}
        out = {self.root: np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for j in pending:
                if j["parent"] not in out:
                    rest.append(j)
                    continue
                T = out[j["parent"]] @ j["T"]
                if j["name"] in qi:
                    M = np.eye(4)
                    M[:3, :3] = axis_rot(j["axis"], qi[j["name"]])
                    T = T @ M
                out[j["child"]] = T
            pending = rest
        return out


def main(sph_urdf, srdf, n=4000, seed=0):
    fk = FK(sph_urdf)
    S = {k: v for k, v in link_spheres(sph_urdf, merge_parts=True).items() if v}
    H = link_hulls()
    links = [l for l in S if l in H]
    dis = {frozenset((d.get("link1"), d.get("link2"))) for d in etree.parse(srdf).getroot().findall("disable_collisions")}
    rng = np.random.default_rng(seed)
    act = fk.active  # in the spherized model only the 7 arm joints move
    lo = np.array([j["lo"] for j in act]); hi = np.array([j["hi"] for j in act])
    Q = rng.uniform(lo, hi, size=(n, len(act)))
    cvx = {}
    for l in links:
        h = H[l][0]
        pts = coal.StdVec_Vec3s()
        pts.extend(list(np.asarray(h.vertices, dtype=float)))
        cvx[l] = coal.Convex.convexHull(pts, False, "")
    pairs = [(a, b) for i, a in enumerate(links) for b in links[i + 1:]]
    stats = {p: [0, 0, 0, 0] for p in pairs}  # sphere hit, exact hit, spurious, missed
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    for q in Q:
        T = fk(q)
        W = {l: (np.array([c for c, _ in S[l]]) @ T[l][:3, :3].T + T[l][:3, 3], np.array([r for _, r in S[l]])) for l in links}
        for a, b in pairs:
            (Ca, Ra), (Cb, Rb) = W[a], W[b]
            sph = (np.linalg.norm(Ca[:, None] - Cb[None], axis=2) - Ra[:, None] - Rb[None]).min() < 0
            ta, tb = coal.Transform3s(T[a][:3, :3], T[a][:3, 3]), coal.Transform3s(T[b][:3, :3], T[b][:3, 3])
            res.clear()
            d = coal.distance(cvx[a], ta, cvx[b], tb, req, res)
            ex = d < 0
            st = stats[(a, b)]
            st[0] += sph; st[1] += ex; st[2] += sph and not ex; st[3] += ex and not sph
    print(f"{'pair':66s} {'srdf':8s} {'sph%':>6s} {'exact%':>7s} {'spur%':>6s} {'missed':>6s}")
    for (a, b), (s, e, sp, m) in stats.items():
        tag = "disabled" if frozenset((a, b)) in dis else "ENABLED"
        if s == 0 and e == 0:
            continue
        print(f"{a + ' -- ' + b:66s} {tag:8s} {100*s/n:6.1f} {100*e/n:7.2f} {100*sp/n:6.1f} {m:6d}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4000)
