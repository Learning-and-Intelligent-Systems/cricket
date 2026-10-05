"""Tight conservative re-spherization of Rainbow's VAMP model against the master.

For every link of the TEMPLATE (the cricket input URDF) with collision geometry
in the MASTER (rainbow_seiji.urdf, hulled as every consumer hulls it), replace
its spheres by a cover of each hull that
  - contains it (surface, edges, vertices, interior), verified on an
    independent dense sample, with a PAD on every radius for the gaps between
    cover samples;
  - stays within `delta` of it (+ PAD): each sphere inside the hull's
    delta-offset polytope;
  - for the links standing on the floor (base, wheels, lidar): every sphere
    bottom at world z >= 0.5 mm, covering the hull from FLOOR_COVER up
    (PO worlds carry a permanent floor body that VAMP checks, see the droid's
    deployment lesson).
A master SPHERE primitive (rby1_camera_mount) is copied exactly.
The lidar (master link lidar_mid360, a box on the base, not in the old model)
is added as its own fixed link. The 23 DOF's joint limits are synced to the
master (torso_2: [-140, +80] deg).

Usage: respherize.py TEMPLATE OUT DELTA_MM [link=delta_mm ...] [--pad=MM] [--floor_cover=MM]
"""
import copy
import sys

import numpy as np
import trimesh
from lxml import etree

from rb_common import FK, MASTER, VAMP_JOINTS, edge_samples_for, fib_sphere, link_geoms, protrusion, uncovered_depth
from tight_cover2 import cover

FLOOR_LINKS = ("base", "wheel_r", "wheel_l", "lidar_mid360")
FLOOR_EPS = 0.0005


def verify_points(h, floor_cover=None, seed=11):
    # independent of the cover's targets; ~1.5 mm surface spacing at least
    surf, _ = trimesh.sample.sample_surface(h, int(max(60000, h.area / 0.0015**2)), seed=seed)
    rng = np.random.default_rng(seed)
    lo, hi = h.bounds
    X = rng.uniform(lo, hi, size=(60000, 3))
    X = X[h.contains(X)][:20000]
    pts = np.vstack([surf, h.vertices, edge_samples_for(h, 0.002), X])
    if floor_cover is not None:
        pts = pts[pts[:, 2] >= floor_cover]
    return pts


def add_link(root, master_root, link):
    """Copy a master link (empty, to be covered) + the fixed joint that
    carries it into the template, if the template lacks it."""
    names = {l.get("name") for l in root.findall("link")}
    if link in names:
        return
    mj = [j for j in master_root.findall("joint") if j.find("child").get("link") == link][0]
    nl = etree.SubElement(root, "link", name=link)
    root.append(copy.deepcopy(mj))
    return nl


def add_lidar(root, master_root):
    """The master's lidar (a box on the base, not in the old model)."""
    return add_link(root, master_root, "lidar_mid360")


def sync_limits(root, master_root):
    mlim = {}
    for j in master_root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            mlim[j.get("name")] = (lim.get("lower"), lim.get("upper"))
    changed = []
    for j in root.findall("joint"):
        n = j.get("name")
        if n not in VAMP_JOINTS or n in ("x", "y", "theta") or n not in mlim:
            continue
        lim = j.find("limit")
        if (lim.get("lower"), lim.get("upper")) != mlim[n]:
            changed.append((n, (lim.get("lower"), lim.get("upper")), mlim[n]))
            lim.set("lower", mlim[n][0])
            lim.set("upper", mlim[n][1])
    return changed


# Per-plane tightening: planes of the link's hull whose normal n has
# n . dir > thresh get delta_t instead of the link's delta. Where two hulls sit
# a few mm apart at the robot's own rest postures (arms hanging at the torso's
# sides), a uniform delta would make the spheres overlap there.
TIGHT = {
    # torso_5's side faces (normal ~ (0, -+0.92, -0.4)) sit 15 mm from the
    # hanging upper arms (arm_2) at the default / stow posture (arms at 0)
    # (only the facets within 14 cm of the shoulder: where arm_2's upper end
    # swings as joint 0 turns)
    "link_torso_5": [((0, 1, 0), 0.6, 0.004, (0, 0.20, 0.03), 0.14), ((0, -1, 0), 0.6, 0.004, (0, -0.20, 0.03), 0.14)],
    # ... and each upper arm's inner face (toward the torso at joint 0)
    "link_right_arm_2": [((0, 1, 0), 0.6, 0.005)],
    "link_left_arm_2": [((0, -1, 0), 0.6, 0.005)],
}


# Links whose master geometry MOVES with joints VAMP bakes (the head's pan and
# tilt, fixed at 0 in the 23-DOF model): the link's spheres cover the convex
# hull of the listed parts swept over the joint ranges, expressed in the
# link's frame at the baked pose. The head camera (a 9.5 cm sphere 13 cm above
# the tilt axis) swings up to ~15 cm with tilt; a cover at tilt 0 alone misses
# the arm hitting it at the head's real tilt (log_2026-10-04_19_15_15 p68484:
# FT_sensor_R 72 mm into the mount at tilt 30 deg).
HEAD_RANGE = {"head_0": (-0.523, 0.523), "head_1": (-0.35, 1.57)}
SWEEP = {
    "rby1_camera_mount": dict(parts=["rby1_camera_mount", "link_head_2", "link_head_1"], joints=HEAD_RANGE),
}


def swept_hull(link, spec, n=31):
    """Convex hull, in `link`'s frame at the baked pose (head joints 0), of the
    spec's parts swept over a grid of the spec's joint ranges."""
    geoms = link_geoms(MASTER)
    fk = FK(MASTER)
    T0inv = np.linalg.inv(fk({})[link])
    names = list(spec["joints"])
    grids = np.meshgrid(*[np.linspace(*spec["joints"][j], n) for j in names], indexing="ij")
    combos = np.stack([g.ravel() for g in grids], -1)
    local = []
    for part in spec["parts"]:
        for g in geoms[part]:
            if g[0] == "sphere":
                local.append((part, g[1] + g[2] * fib_sphere(400)))
            else:
                local.append((part, np.asarray(g[1].vertices)))
    P = []
    for vals in combos:
        T = fk(dict(zip(names, vals)))
        for part, V in local:
            M = T0inv @ T[part]
            P.append(V @ M[:3, :3].T + M[:3, 3])
    return trimesh.convex.convex_hull(np.vstack(P))


def respherize(template, out, delta, per_link=None, pad=0.002, floor_cover=0.012, links=None, lidar=True, spacing=0.004, tight=None):
    per_link = per_link or {}
    tight = TIGHT if tight is None else tight
    geoms = link_geoms(MASTER)
    tree = etree.parse(template)
    root = tree.getroot()
    master_root = etree.parse(MASTER).getroot()
    if lidar:
        add_lidar(root, master_root)
    for extra in ("link_right_arm_camera", "link_left_arm_camera"):  # RainbowCam's wrist cameras
        if extra in geoms:
            add_link(root, master_root, extra)
    changed = sync_limits(root, master_root)
    T0 = FK_of(tree)
    total, rows = 0, []
    for link in root.findall("link"):
        name = link.get("name")
        if name not in geoms or (links and name not in links):
            total += len(link.findall("collision"))
            continue
        d = per_link.get(name, delta)
        floor = fc = None
        if name in FLOOR_LINKS:
            floor = -T0[name][2, 3]  # world z = 0 in the link frame (frames axis-aligned at q = 0)
            fc = floor + floor_cover
        sph, unc_max, pr = [], 0.0, []
        parts = geoms[name]
        if name in SWEEP:
            parts = [("mesh", swept_hull(name, SWEEP[name]))]
        for part in parts:
            if part[0] == "sphere":
                sph.append((part[1], part[2]))
                continue
            h = part[1]
            s = cover(h, d, floor=floor, floor_cover=fc, pad=pad, surf_spacing=spacing, tight=tight.get(name))
            s = [(c, r + pad) for c, r in s]
            if floor is not None:
                assert min(c[2] - r for c, r in s) >= floor + FLOOR_EPS - 1e-9
            vp = verify_points(h, fc)
            unc = uncovered_depth(vp, s)
            miss = np.zeros((0, 3))
            for _ in range(6):  # re-cover with the missed verification points as targets
                if unc.max() <= 0:
                    break
                miss = np.vstack([miss, vp[unc > 0]])
                s = cover(h, d, floor=floor, floor_cover=fc, pad=pad, extra_targets=miss, surf_spacing=spacing, tight=tight.get(name))
                s = [(c, r + pad) for c, r in s]
                unc = uncovered_depth(vp, s)
            if unc.max() > 0:
                print(f"WARNING {name}: {int((unc > 0).sum())} verification points uncovered, max {1000 * unc.max():.2f} mm")
            # further independent verification samples
            for sd in (29, 41, 53):
                unc = np.maximum(unc.max(), uncovered_depth(verify_points(h, fc, seed=sd), s).max())
            unc_max = max(unc_max, unc.max())
            pr.append(protrusion(s, [h]))
            sph.extend(s)
        pr = np.concatenate(pr) if pr else np.zeros(1)
        zb = min((T0[name][:3, :3] @ c + T0[name][:3, 3])[2] - r for c, r in sph)
        rows.append((name, len(sph), 1000 * d, 1000 * max(0.0, unc_max), 1000 * pr.max(), 1000 * pr.mean(), 1000 * zb))
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
    etree.indent(tree, space="  ")
    tree.write(out, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"{'link':22s} {'n':>4s} {'delta':>5s} {'uncov max mm':>12s} {'protr max':>9s} {'mean':>6s} {'min z mm':>8s}")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:4d} {r[2]:5.0f} {r[3]:12.2f} {r[4]:9.1f} {r[5]:6.1f} {r[6]:8.1f}")
    print("limits synced to the master:", changed)
    print(f"total spheres {total} -> {out}")


def FK_of(tree):
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("wb", suffix=".urdf", delete=False) as f:
        tree.write(f)
        p = f.name
    try:
        return FK(p)({})
    finally:
        os.unlink(p)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
    template, out, dmm = args[0], args[1], float(args[2])
    per = {}
    for a in args[3:]:
        k, v = a.split("=")
        per[k] = float(v) / 1000
    respherize(
        template,
        out,
        dmm / 1000,
        per,
        pad=float(opts.get("pad", 2)) / 1000,
        floor_cover=float(opts.get("floor_cover", 12)) / 1000,
        links=opts.get("links", "").split(",") if opts.get("links") else None,
        spacing=float(opts.get("spacing", 4)) / 1000,
    )
