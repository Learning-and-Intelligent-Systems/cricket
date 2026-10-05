"""Shared pieces for re-spherizing Rainbow (RB-Y1) against its master URDF.

The master (QAA rainbow_description/rainbow_seiji.urdf) is the planner's own
geometry: every consumer hulls its collision meshes (HPN PinRainbowFK,
kinsim, the sims), so the hull of each link's collision geometry is the
reference here. Primitive geometry is kept exact: rby1_camera_mount is a
sphere, lidar_mid360 a box.

The VAMP model (cricket input URDF) shares joint names with the master for its
23 DOF [x, y, theta, torso_0..5, left_arm_0..6, right_arm_0..6]; it fixes the
wheels, the head (pan = tilt = 0) and the fingers (fully open), and its root
has no -0.228 m theta offset (HPN's get_vamp_coords applies it), so world
comparisons align a reference link (base) between the two FKs.
"""

import os
import re

import numpy as np
import trimesh
from lxml import etree

try:
    from qaa.asset_paths import qr_assets_path

    QAA = os.path.join(qr_assets_path(), "robots")
except ImportError:  # pragma: no cover
    QAA = os.environ.get("QAA_ROBOTS", "")

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.environ.get(
    "SPHERIZE_MASTER", os.path.join(QAA, "rainbow_description", "rainbow_seiji.urdf")
)
PROFILE = os.environ.get("SPHERIZE_PROFILE", "rainbow")  # qaa.robot_description profile (derived exclusions)
TEMPLATE = os.path.join(HERE, "template_plus.urdf")  # cricket resources/rainbow @ 30eed3e
OLD_SRDF = os.path.join(HERE, "old_rainbow.srdf")

# VAMP's 23-vector order (vamp.rainbow.joint_names())
VAMP_JOINTS = (
    ["x", "y", "theta"]
    + [f"torso_{i}" for i in range(6)]
    + [f"left_arm_{i}" for i in range(7)]
    + [f"right_arm_{i}" for i in range(7)]
)
# the fingers as the VAMP model bakes them (fully open: 0.003 + 0.05 off ee)
OPEN_FINGERS = {
    "gripper_finger_r1": -0.05,
    "gripper_finger_r2": 0.05,
    "gripper_finger_l1": -0.05,
    "gripper_finger_l2": 0.05,
}


def resolve(fn, urdf_dir):
    if fn.startswith("package://"):
        rel = fn[len("package://") :]
        for base in (QAA, os.path.dirname(QAA)):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    return os.path.join(urdf_dir, fn)


def _origin(c):
    o = c.find("origin")
    xyz = [float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()]
    rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
    T = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    T[:3, 3] = xyz
    return T


def link_geoms(urdf=MASTER):
    """{link: [("mesh", hull) | ("sphere", centre, r, hull) | ("box", hull)]},
    each in the LINK frame. hull is the trimesh convex hull every consumer
    uses (for a sphere: a fine icosphere, used only for sampling/plots)."""
    t = etree.parse(urdf).getroot()
    out = {}
    for link in t.findall("link"):
        parts = []
        for c in link.findall("collision"):
            g = c.find("geometry")[0]
            T = _origin(c)
            if g.tag == "mesh":
                m = trimesh.load(resolve(g.get("filename"), os.path.dirname(urdf)), force="mesh")
                sc = g.get("scale")
                if sc:
                    m.apply_scale([float(v) for v in sc.split()])
                m.apply_transform(T)
                parts.append(("mesh", m.convex_hull))
            elif g.tag == "box":
                ext = [float(v) for v in g.get("size").split()]
                m = trimesh.creation.box(extents=ext)
                m.apply_transform(T)
                parts.append(("box", m))
            elif g.tag == "sphere":
                r = float(g.get("radius"))
                m = trimesh.creation.icosphere(subdivisions=4, radius=r)
                m.apply_translation(T[:3, 3])
                parts.append(("sphere", T[:3, 3].copy(), r, m))
            elif g.tag == "cylinder":
                m = trimesh.creation.cylinder(radius=float(g.get("radius")), height=float(g.get("length")))
                m.apply_transform(T)
                parts.append(("mesh", m.convex_hull))
        if parts:
            out[link.get("name")] = parts
    return out


def link_hulls(urdf=MASTER):
    """{link: [hull trimesh]} (spheres as fine icospheres)."""
    return {l: [p[-1] if p[0] == "sphere" else p[1] for p in ps] for l, ps in link_geoms(urdf).items()}


def link_spheres(urdf, merge_parts=True):
    """{link: [(centre, radius)]} of a spherized URDF; merge_parts folds the
    L__sN sub-links of split_urdf.py (identity fixed joints) back into L."""
    t = etree.parse(urdf).getroot()
    out = {}
    for link in t.findall("link"):
        name = link.get("name")
        key = name.split("__s")[0] if merge_parts else name
        s = []
        for c in link.findall("collision"):
            g = c.find("geometry")[0]
            if g.tag != "sphere":
                continue
            s.append((_origin(c)[:3, 3].copy(), float(g.get("radius"))))
        out.setdefault(key, []).extend(s)
    return out


def read_disabled(srdf):
    """Disabled pairs of an SRDF (regex: cricket's SRDFs carry '--' inside
    comments, which strict XML parsers reject)."""
    txt = open(srdf).read()
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.S)
    pat = re.compile(r'<disable_collisions\s+link1="([^"]+)"\s+link2="([^"]+)"')
    return {frozenset(m.groups()) for m in pat.finditer(txt)}


def rpy_mat(r, p, y):
    return trimesh.transformations.euler_matrix(r, p, y, axes="sxyz")[:3, :3]


def axis_rot(ax, t):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


class FK:
    """URDF forward kinematics (revolute / continuous / prismatic / fixed).
    __call__(values: {joint: value}) -> {link: 4x4 world}; joints not given
    are 0 (except `defaults`)."""

    def __init__(self, urdf, defaults=None):
        r = etree.parse(urdf).getroot()
        self.joints = []
        for j in r.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()])
            rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rpy_mat(*rpy), xyz
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints.append(
                dict(
                    name=j.get("name"),
                    type=j.get("type"),
                    parent=j.find("parent").get("link"),
                    child=j.find("child").get("link"),
                    T=T,
                    axis=[float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0],
                    lo=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                    hi=float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
                )
            )
        self.by_name = {j["name"]: j for j in self.joints}
        self.active = [j for j in self.joints if j["type"] in ("revolute", "continuous", "prismatic")]
        children = {j["child"] for j in self.joints}
        self.root = [l.get("name") for l in r.findall("link") if l.get("name") not in children][0]
        self.defaults = dict(defaults or {})
        # topological order
        order, have, pending = [], {self.root}, list(self.joints)
        while pending:
            rest = []
            for j in pending:
                if j["parent"] in have:
                    order.append(j)
                    have.add(j["child"])
                else:
                    rest.append(j)
            if len(rest) == len(pending):
                raise RuntimeError("disconnected URDF")
            pending = rest
        self.order = order

    def __call__(self, values):
        vals = dict(self.defaults)
        vals.update(values)
        out = {self.root: np.eye(4)}
        for j in self.order:
            T = out[j["parent"]] @ j["T"]
            q = vals.get(j["name"])
            if q is not None and j["type"] != "fixed":
                M = np.eye(4)
                if j["type"] == "prismatic":
                    M[:3, 3] = np.asarray(j["axis"], float) * q
                else:
                    M[:3, :3] = axis_rot(j["axis"], q)
                T = T @ M
            out[j["child"]] = T
        return out


def vamp_values(q23):
    return {n: float(v) for n, v in zip(VAMP_JOINTS, q23)}


def vamp_bounds(urdf=TEMPLATE):
    fk = FK(urdf)
    lo = np.array([fk.by_name[n]["lo"] for n in VAMP_JOINTS], float)
    hi = np.array([fk.by_name[n]["hi"] for n in VAMP_JOINTS], float)
    return lo, hi


def master_bounds(urdf=MASTER):
    """Master joint limits for the 23 VAMP DOF (base x/y +-2.5, theta +-pi)."""
    fk = FK(urdf)
    lo, hi = [], []
    for n in VAMP_JOINTS:
        j = fk.by_name[n]
        a = j["lo"] if j["lo"] is not None else -np.pi
        b = j["hi"] if j["hi"] is not None else np.pi
        lo.append(a)
        hi.append(b)
    return np.array(lo), np.array(hi)


def random_confs(n, seed=0, bounds=None, base_zero=True):
    lo, hi = bounds if bounds is not None else master_bounds()
    rng = np.random.default_rng(seed)
    Q = rng.uniform(lo, hi, size=(n, len(lo)))
    if base_zero:
        Q[:, :3] = 0.0
    return Q


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


def uncovered_depth(pts, spheres, chunk=20000):
    if not spheres:
        return np.full(len(pts), np.inf)
    C = np.array([c for c, _ in spheres])
    R = np.array([r for _, r in spheres])
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        d = np.linalg.norm(pts[i : i + chunk, None, :] - C[None], axis=2) - R[None]
        out[i : i + chunk] = d.min(axis=1)
    return out  # > 0: outside every sphere by this much


def protrusion(spheres, hulls, n=400):
    """Distance from the union's boundary points to the hull (0 inside)."""
    if not spheres:
        return np.zeros(0)
    C = np.array([c for c, _ in spheres])
    R = np.array([r for _, r in spheres])
    u = fib_sphere(n)
    P = (C[:, None, :] + R[:, None, None] * u[None]).reshape(-1, 3)
    keep = np.ones(len(P), bool)
    for i in range(0, len(P), 20000):
        d = np.linalg.norm(P[i : i + 20000, None, :] - C[None], axis=2) - R[None]
        keep[i : i + 20000] = (d < -1e-6).sum(axis=1) == 0
    P = P[keep]
    out = np.full(len(P), np.inf)
    for h in hulls:
        _, dist, _ = trimesh.proximity.closest_point(h, P)
        inside = h.contains(P)
        out = np.minimum(out, np.where(inside, 0.0, dist))
    return out


def edge_samples_for(h, step=0.002):
    pts = []
    for a, b in h.vertices[h.edges_unique]:
        k = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts.append(a + np.linspace(0, 1, k)[:, None] * (b - a))
    return np.vstack(pts)
