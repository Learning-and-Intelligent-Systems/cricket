"""Expected self-collision work per validate() of a cricket kernel: per checked
link pair, one bounding-sphere test, plus n_a x n_b sphere tests when the two
links' bounding spheres overlap (cricket's ccfk order; the kernel returns at
the first hit, so collision-free confs pay the full count).
Bounding spheres: the minimal ball of each link's spheres (cricket uses CGAL;
this is a Badoiu-Clarkson approximation, slightly larger).
    cost_model.py SPHERIZED.urdf SRDF [N]"""
import itertools
import sys

import numpy as np

from rb_common import FK, VAMP_JOINTS, link_spheres, master_bounds, read_disabled


def min_ball(C, R, iters=4000):
    """Approximate minimal ball enclosing balls (Badoiu-Clarkson)."""
    c = C.mean(axis=0)
    for i in range(iters):
        d = C - c
        dist = np.linalg.norm(d, axis=1)
        far = np.argmax(dist + R)
        u = d[far] / dist[far] if dist[far] > 1e-12 else np.zeros(3)
        p = C[far] + R[far] * u  # farthest point of the farthest ball
        c = c + (p - c) / (i + 2)
    return c, (np.linalg.norm(C - c, axis=1) + R).max()


def typical_confs(n, seed=0, free_only=False):
    """Manipulation-like confs: torso at a home knee, left arm at rest (zeros
    or a random small offset), right arm uniform in its limits."""
    lo, hi = master_bounds()
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, 23))
    knees = [np.deg2rad([0.0, -30.0, 60.0, -15.0, 0.0, 0.0]), np.deg2rad([0.0, 30.0, -60.0, 45.0, 0.0, 0.0])]
    ri = [VAMP_JOINTS.index(f"right_arm_{i}") for i in range(7)]
    for k in range(n):
        Q[k, 3:9] = knees[k % 2]
        Q[k, ri] = rng.uniform(lo[ri], hi[ri])
    return Q


def main(urdf, srdf, n=2000):
    S = {l: s for l, s in link_spheres(urdf, merge_parts=False).items() if s}
    dis = read_disabled(srdf)
    links = list(S)
    pairs = [(a, b) for a, b in itertools.combinations(links, 2) if frozenset((a, b)) not in dis]
    SC = {l: np.array([c for c, _ in S[l]]) for l in links}
    SR = {l: np.array([r for _, r in S[l]]) for l in links}
    B = {l: min_ball(SC[l], SR[l]) for l in links}
    fk = FK(urdf)
    from rb_common import random_confs

    for label, Q in (("random", random_confs(n, seed=5)), ("typical", typical_confs(n, seed=5))):
        tot, bt, hits = 0.0, 0.0, {}
        for q in Q:
            T = fk(dict(zip(VAMP_JOINTS, map(float, q))))
            W = {l: (T[l][:3, :3] @ B[l][0] + T[l][:3, 3], B[l][1]) for l in links}
            for a, b in pairs:
                bt += 1
                if np.linalg.norm(W[a][0] - W[b][0]) < W[a][1] + W[b][1]:
                    c = len(SR[a]) * len(SR[b])
                    tot += c
                    hits[(a, b)] = hits.get((a, b), 0) + c
        print(f"{label:8s}: {sum(len(v) for v in S.values())} spheres, {len(links)} links, {len(pairs)} checked pairs; "
              f"per conf: bounding tests {bt / n:.0f}, sphere tests {tot / n:.0f}")
        for (a, b), c in sorted(hits.items(), key=lambda kv: -kv[1])[:10]:
            print(f"     {a:24s} {b:24s} {c / n:8.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
