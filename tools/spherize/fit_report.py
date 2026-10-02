"""Per link: how a sphere set fits the planner's own geometry (the convex hull of
the master URDF's collision mesh). coverage = deepest hull point outside every
sphere (must be <= 0 for a conservative model); protrusion = how far the
union's surface reaches beyond the hull."""
import os
import sys

import numpy as np
import trimesh
from lxml import etree

try:  # the QAA robot assets (the master URDFs and their meshes)
    from qaa.asset_paths import qr_assets_path

    QAA = os.path.join(qr_assets_path(), "robots")
except ImportError:
    QAA = os.environ.get("QAA_ROBOTS", "")
# the master URDF whose collision hulls the spheres must contain
MASTER = os.environ.get(
    "SPHERIZE_MASTER", os.path.join(QAA, "franka_description", "fr3_robotiq_2f_85.urdf")
)


def resolve(fn, urdf_dir):
    if fn.startswith("package://"):
        rel = fn[len("package://"):]
        for base in (QAA, os.path.dirname(QAA)):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    p = os.path.join(urdf_dir, fn)
    return p


def link_hulls(urdf=MASTER):
    t = etree.parse(urdf).getroot()
    out = {}
    for link in t.findall("link"):
        parts = []
        for c in link.findall("collision"):
            g = c.find("geometry")[0]
            if g.tag != "mesh":
                continue
            m = trimesh.load(resolve(g.get("filename"), os.path.dirname(urdf)), force="mesh")
            sc = g.get("scale")
            if sc:
                m.apply_scale([float(v) for v in sc.split()])
            o = c.find("origin")
            if o is not None:
                xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
                rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
                T = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
                T[:3, 3] = xyz
                m.apply_transform(T)
            parts.append(m.convex_hull)
        if parts:
            out[link.get("name")] = parts
    return out


def link_spheres(urdf, merge_parts=False):
    """{link: [(centre, radius)]}. merge_parts folds split_urdf.py's fixed
    sub-links (``L__sN``, identity joints, so the same frame) back into L."""
    t = etree.parse(urdf).getroot()
    out = {}
    for link in t.findall("link"):
        if merge_parts and "__s" in link.get("name"):
            base = link.get("name").split("__s")[0]
            for c in link.findall("collision"):
                g = c.find("geometry")[0]
                if g.tag == "sphere":
                    o = c.find("origin")
                    xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
                    out.setdefault(base, []).append((np.array(xyz), float(g.get("radius"))))
            continue
        s = []
        for c in link.findall("collision"):
            g = c.find("geometry")[0]
            if g.tag != "sphere":
                continue
            o = c.find("origin")
            xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
            s.append((np.array(xyz), float(g.get("radius"))))
        out.setdefault(link.get("name"), []).extend(s)
    return out


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def hull_samples(hulls, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for h in hulls:
        surf, _ = trimesh.sample.sample_surface(h, n, seed=int(rng.integers(1 << 30)))
        pts.append(surf)
        pts.append(h.vertices)
    return np.vstack(pts)


def uncovered_depth(pts, spheres):
    if not spheres:
        return np.full(len(pts), np.inf)
    C = np.array([c for c, _ in spheres])
    R = np.array([r for _, r in spheres])
    d = np.linalg.norm(pts[:, None, :] - C[None], axis=2) - R[None]
    return d.min(axis=1)  # > 0: outside every sphere by this much


def protrusion(spheres, hulls, n=400):
    if not spheres:
        return np.zeros(0)
    C = np.array([c for c, _ in spheres])
    R = np.array([r for _, r in spheres])
    u = fib_sphere(n)
    P = (C[:, None, :] + R[:, None, None] * u[None]).reshape(-1, 3)
    # keep the union's boundary: points not strictly inside another sphere
    d = np.linalg.norm(P[:, None, :] - C[None], axis=2) - R[None]
    P = P[(d < -1e-6).sum(axis=1) == 0]
    out = np.full(len(P), np.inf)
    for h in hulls:
        _, dist, _ = trimesh.proximity.closest_point(h, P)
        inside = h.contains(P)
        out = np.minimum(out, np.where(inside, 0.0, dist))
    return out


def report(sph_urdf, links=None, master=None):
    hulls = link_hulls(master or MASTER)
    spheres = link_spheres(sph_urdf, merge_parts=True)
    rows = []
    for name, hs in hulls.items():
        if links and name not in links:
            continue
        s = spheres.get(name, [])
        pts = hull_samples(hs)
        unc = uncovered_depth(pts, s)
        pr = protrusion(s, hs)
        vol_h = sum(h.volume for h in hs)
        rows.append((name, len(s), 1000 * max(0.0, unc.max()), 100 * (unc > 1e-4).mean(),
                     1000 * (pr.max() if len(pr) else 0), 1000 * (pr.mean() if len(pr) else 0), vol_h))
    print(f"{'link':38s} {'n':>3s} {'uncov max mm':>12s} {'uncov %':>8s} {'protr max mm':>12s} {'protr mean':>10s}")
    for r in rows:
        print(f"{r[0]:38s} {r[1]:3d} {r[2]:12.1f} {r[3]:8.2f} {r[4]:12.1f} {r[5]:10.1f}")
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    master = None
    if args and args[0].startswith("--master="):
        master = args.pop(0).split("=", 1)[1]
    report(args[0], args[1:] or None, master)
