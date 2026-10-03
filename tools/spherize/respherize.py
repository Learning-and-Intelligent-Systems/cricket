"""Tight conservative re-spherization of a VAMP robot model.

For every link of the spherized TEMPLATE (the cricket input URDF) that has a
collision mesh in the MASTER URDF (the planner's own geometry, hulled as every
consumer hulls it), replace its spheres by a cover of the hull that
  - contains it: hull surface, edges, vertices and an interior grid, verified
    on an independent dense sample, with a radius pad for the gaps between
    cover samples;
  - stays within `delta` of it (each sphere inside the hull's delta-offset
    polytope), and above `floor` for a mounted base link.
Usage: respherize.py TEMPLATE OUT DELTA_MM [link=delta_mm ...]"""
import sys

import numpy as np
import trimesh
from lxml import etree

from fit_report import MASTER, hull_samples, link_hulls, protrusion, uncovered_depth
from tight_cover import cover

PAD = 0.002
FLOORS = {"panda_link0": (0.0005, 0.012)}  # (sphere floor z, cover points above z): spheres stay above the base plane z = 0 -- a PO world has a permanent "floor" body under the base with its top at z = -0.013, which VAMP checks (the exact checker skips it by name)


def edge_samples(h, step=0.002):
    pts = []
    for a, b in h.vertices[h.edges_unique]:
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        t = np.linspace(0, 1, n)[:, None]
        pts.append(a + t * (b - a))
    return np.vstack(pts)


def verify_points(h, floor=None):
    pts = np.vstack([hull_samples([h], 60000, seed=11), edge_samples(h)])
    if floor is not None:
        pts = pts[pts[:, 2] >= floor[1]]
    return pts


def respherize(template, out, delta, per_link=None):
    per_link = per_link or {}
    hulls = link_hulls(MASTER)
    tree = etree.parse(template)
    root = tree.getroot()
    total, rows = 0, []
    for link in root.findall("link"):
        name = link.get("name")
        if name not in hulls:
            total += len(link.findall("collision"))
            continue
        h = hulls[name][0]
        d = per_link.get(name, delta)
        fl = FLOORS.get(name)
        sph, _ = cover_floor(h, d, fl)
        sph = [(c, r + PAD) for c, r in sph]
        if fl is not None:  # keep the pad above the floor too
            sph = [(c, min(r, c[2] - fl[0]) if c[2] - fl[0] > 0 else r) for c, r in sph]
        unc = uncovered_depth(verify_points(h, fl), sph)
        pr = protrusion(sph, [h])
        rows.append((name, len(sph), 1000 * d, 1000 * max(0.0, unc.max()), 1000 * pr.max(), 1000 * pr.mean()))
        for c in link.findall("collision"):
            link.remove(c)
        for c, r in sph:
            col = etree.SubElement(link, "collision")
            org = etree.SubElement(col, "origin")
            org.set("xyz", " ".join(f"{v:.6f}" for v in c))
            org.set("rpy", "0 0 0")
            geo = etree.SubElement(col, "geometry")
            etree.SubElement(geo, "sphere").set("radius", f"{r:.6f}")
        total += len(sph)
    tree.write(out, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"{'link':38s} {'n':>3s} {'delta':>5s} {'uncov max mm':>12s} {'protr max':>9s} {'mean':>6s}")
    for r in rows:
        print(f"{r[0]:38s} {r[1]:3d} {r[2]:5.0f} {r[3]:12.2f} {r[4]:9.1f} {r[5]:6.1f}")
    print(f"total spheres {total} -> {out}")


def cover_floor(h, d, fl):
    if fl is None:
        return cover(h, d)
    # cover only points above fl[1]; spheres stay above fl[0]
    import tight_cover as tc

    orig = tc.trimesh.sample.sample_surface_even

    return tc_cover_clipped(h, d, fl)


def tc_cover_clipped(h, delta, fl):
    import tight_cover as tc

    n, b = tc.hull_planes(h)
    grid = max(0.5 * delta, 0.004)
    surf, _ = trimesh.sample.sample_surface_even(h, int(max(2000, h.area / 0.004**2)), seed=0)
    lo, hi = h.bounds
    ax = [np.arange(lo[i] + grid / 2, hi[i], grid) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    G = G[(G @ n.T <= b + 1e-9).all(axis=1)]
    U = np.vstack([surf, h.vertices, G, edge_samples(h, 0.004)])
    U = U[U[:, 2] >= fl[1]]
    C = np.vstack([G, surf[:: max(1, len(surf) // 3000)], h.vertices])
    R = (b[None, :] + delta - C @ n.T).min(axis=1)
    R = np.minimum(R, C[:, 2] - fl[0] - PAD)  # room for the pad above the floor
    keep = R > 1e-4
    C, R = C[keep], R[keep]
    cov = np.vstack([
        np.linalg.norm(U[i : i + 4000, None, :] - C[None], axis=2) <= R[None] + 1e-9
        for i in range(0, len(U), 4000)
    ])
    if not cov.any(axis=1).all():
        raise RuntimeError(f"{(~cov.any(axis=1)).sum()} points uncoverable")
    unc, chosen = np.ones(len(U), bool), []
    while unc.any():
        gain = cov[unc].sum(axis=0)
        j = int(np.argmax(gain))
        chosen.append(j)
        unc &= ~cov[:, j]
    Cc, Rc = C[chosen], R[chosen]
    d = np.linalg.norm(U[:, None, :] - Cc[None], axis=2)
    owner = np.argmin(np.where(d <= Rc[None] + 1e-9, d, np.inf), axis=1)
    newR = np.array([d[owner == k, k].max() if (owner == k).any() else 0 for k in range(len(Cc))])
    used = newR > 0
    return [(c, float(r)) for c, r in zip(Cc[used], newR[used])], 0


if __name__ == "__main__":
    template, out, dmm = sys.argv[1], sys.argv[2], float(sys.argv[3])
    per = {}
    for a in sys.argv[4:]:
        k, v = a.split("=")
        per[k] = float(v) / 1000
    respherize(template, out, dmm / 1000, per)
