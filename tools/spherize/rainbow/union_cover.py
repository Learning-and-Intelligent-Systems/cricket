"""World-frame union coverage: hull samples of every master link, posed by the
master FK, against ALL spheres of the VAMP model, posed by its FK (root offset
removed by aligning the base link). A point outside every sphere is a VAMP
blind spot for the planner's own geometry.
    union_cover.py SPHERIZED.urdf [N_CONF] [skip,links]"""
import sys

import numpy as np

from rb_common import FK, MASTER, OPEN_FINGERS, hull_samples, link_hulls, link_spheres, random_confs, vamp_values


def main(sph, n_conf=30, skip=()):
    M, V = FK(MASTER), FK(sph)
    H, S = link_hulls(MASTER), link_spheres(sph, merge_parts=False)
    pts = {l: hull_samples(h, 3000, seed=1) for l, h in H.items() if l not in skip}
    worst, frac = {}, {}
    for q in random_confs(n_conf, seed=0):
        vals = vamp_values(q)
        TV = V(vals)
        TM = M({**vals, **OPEN_FINGERS})
        A = TV["base"] @ np.linalg.inv(TM["base"])  # master world -> vamp world
        C = np.vstack([np.array([c for c, _ in s]) @ TV[l][:3, :3].T + TV[l][:3, 3] for l, s in S.items() if s])
        R = np.concatenate([[r for _, r in s] for l, s in S.items() if s])
        for l, P in pts.items():
            W = (P @ TM[l][:3, :3].T + TM[l][:3, 3]) @ A[:3, :3].T + A[:3, 3]
            d = np.full(len(W), np.inf)
            for i in range(0, len(W), 4000):
                d[i : i + 4000] = (np.linalg.norm(W[i : i + 4000, None] - C[None], axis=2) - R[None]).min(axis=1)
            worst[l] = max(worst.get(l, 0.0), d.max())
            frac[l] = frac.get(l, 0.0) + (d > 0.002).mean() / n_conf
    print(f"{'link':24s} {'worst gap mm':>12s} {'% pts > 2mm out':>16s}")
    for l in worst:
        if worst[l] > 0.0005:
            print(f"{l:24s} {1000 * worst[l]:12.1f} {100 * frac[l]:16.2f}")
    print("links checked:", len(worst), " worst overall mm:", round(1000 * max(worst.values()), 2))
    return worst, frac


if __name__ == "__main__":
    main(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 30,
        set(sys.argv[3].split(",")) if len(sys.argv) > 3 else (),
    )
