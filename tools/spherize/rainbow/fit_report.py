"""Per link: how a sphere set fits the master's hulls (link frame).
uncov = deepest hull point outside every sphere (must be <= 0 for a
conservative model); protr = how far the union's surface reaches beyond the
hull.  The lowest sphere bottom is reported in the BASE frame for the base
links (the floor is z = 0 there).
    fit_report.py SPHERIZED.urdf [link ...]"""
import sys

import numpy as np

from rb_common import FK, MASTER, hull_samples, link_hulls, link_spheres, protrusion, uncovered_depth


def report(sph_urdf, links=None, master=MASTER):
    hulls = link_hulls(master)
    spheres = link_spheres(sph_urdf, merge_parts=True)
    T0 = FK(sph_urdf)({})
    rows = []
    for name, hs in hulls.items():
        if links and name not in links:
            continue
        s = spheres.get(name, [])
        pts = hull_samples(hs)
        unc = uncovered_depth(pts, s)
        pr = protrusion(s, hs) if s else np.zeros(0)
        zbot = np.nan
        if s and name in T0:
            T = np.linalg.inv(T0["base"]) @ T0[name]
            zbot = min((T[:3, :3] @ c + T[:3, 3])[2] - r for c, r in s)
        rows.append(
            (
                name,
                len(s),
                1000 * max(0.0, unc.max()),
                100 * (unc > 1e-4).mean(),
                1000 * (pr.max() if len(pr) else 0),
                1000 * (pr.mean() if len(pr) else 0),
                1000 * zbot,
            )
        )
    print(f"{'link':22s} {'n':>4s} {'uncov max mm':>12s} {'uncov %':>8s} {'protr max mm':>12s} {'protr mean':>10s} {'min z mm':>8s}")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:4d} {r[2]:12.1f} {r[3]:8.2f} {r[4]:12.1f} {r[5]:10.1f} {r[6]:8.1f}")
    print("total spheres", sum(len(v) for v in spheres.values()))
    return rows


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2:] or None)
