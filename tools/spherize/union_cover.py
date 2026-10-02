"""World-frame union coverage: hull samples of every master link, posed by
the master FK, against ALL spheres of the VAMP model, posed by its FK (root
offsets removed by aligning one reference link). A point outside every sphere
is a VAMP blind spot for the planner's own geometry."""
import sys

import numpy as np

from fit_report import hull_samples, link_hulls, link_spheres
from selfpair_audit import FK

master, sph, ref = sys.argv[1], sys.argv[2], sys.argv[3]
n_conf = int(sys.argv[4]) if len(sys.argv) > 4 else 30
skip = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else set()
M, V = FK(master), FK(sph)
H, S = link_hulls(master), link_spheres(sph, merge_parts=True)
mj = {j["name"]: j for j in M.active}
rng = np.random.default_rng(0)
pts = {l: hull_samples(h, 3000, seed=1) for l, h in H.items() if l not in skip}
worst, frac = {}, {}
for t in range(n_conf):
    vals = {}
    for j in V.active:
        lo = j["lo"] if j["lo"] is not None else -np.pi
        hi = j["hi"] if j["hi"] is not None else np.pi
        vals[j["name"]] = rng.uniform(lo, hi)
    TV = V([vals[j["name"]] for j in V.active])
    TM = M([vals.get(j["name"], 0.0) for j in M.active])
    A = TV[ref] @ np.linalg.inv(TM[ref])  # master world -> vamp world
    C = np.vstack([np.array([c for c, _ in s]) @ TV[l][:3, :3].T + TV[l][:3, 3] for l, s in S.items() if s and l in TV])
    R = np.concatenate([[r for _, r in s] for l, s in S.items() if s and l in TV])
    for l, P in pts.items():
        if l not in TM:
            continue
        W = (P @ TM[l][:3, :3].T + TM[l][:3, 3]) @ A[:3, :3].T + A[:3, 3]
        d = (np.linalg.norm(W[:, None] - C[None], axis=2) - R[None]).min(axis=1)
        worst[l] = max(worst.get(l, 0.0), d.max())
        frac[l] = frac.get(l, 0.0) + (d > 0.002).mean() / n_conf
print(f"{'link':28s} {'worst gap mm':>12s} {'% pts > 2mm out':>16s}")
for l in worst:
    if worst[l] > 0.002:
        print(f"{l:28s} {1000 * worst[l]:12.1f} {100 * frac[l]:16.2f}")
print("links checked:", len(worst))
