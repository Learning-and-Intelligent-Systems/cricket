"""Cost model for splitting big links into fixed sub-links (cricket gives each
link its own bounding sphere): expected sphere tests per conf."""
import sys

import numpy as np
from lxml import etree

from fit_report import link_spheres
from self_cost import bounding_sphere
from selfpair_audit import FK

urdf, srdf = sys.argv[1], sys.argv[2]
split = {a.split("=")[0]: int(a.split("=")[1]) for a in sys.argv[3:]}
S = {k: v for k, v in link_spheres(urdf).items() if v}
fk = FK(urdf)
dis = {frozenset((d.get("link1"), d.get("link2"))) for d in etree.parse(srdf).getroot().findall("disable_collisions")}
pairs = [(a, b) for i, a in enumerate(S) for b in list(S)[i + 1:] if frozenset((a, b)) not in dis]
parts = {}
for l, sph in S.items():
    k = split.get(l, 1)
    C = np.array([c for c, _ in sph])
    if k == 1:
        parts[l] = [sph]
        continue
    u = np.linalg.svd(C - C.mean(0))[2][0]  # principal axis
    t = (C - C.mean(0)) @ u
    order = np.argsort(t)
    parts[l] = [[sph[i] for i in chunk] for chunk in np.array_split(order, k)]
B = {l: [bounding_sphere(p) for p in parts[l]] for l in S}
rng = np.random.default_rng(0)
lo = np.array([j["lo"] for j in fk.active]); hi = np.array([j["hi"] for j in fk.active])
tot = 0.0
N = 2000
for q in rng.uniform(lo, hi, size=(N, len(lo))):
    T = fk(q)
    for a, b in pairs:
        for pa, (ca, ra) in zip(parts[a], B[a]):
            wa = T[a][:3, :3] @ ca + T[a][:3, 3]
            for pb, (cb, rb) in zip(parts[b], B[b]):
                wb = T[b][:3, :3] @ cb + T[b][:3, 3]
                tot += 1  # the bounding test itself
                if np.linalg.norm(wa - wb) < ra + rb:
                    tot += len(pa) * len(pb)
print(f"split {split}: expected tests per conf {tot / N:.0f}")
