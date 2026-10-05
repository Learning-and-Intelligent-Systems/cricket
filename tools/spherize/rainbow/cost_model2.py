"""cost_model.py with the cluster level: per checked link pair one bounding
test; if the link balls overlap, k_a x k_b cluster-ball tests; if a cluster
pair overlaps, its n x m sphere tests.
    cost_model2.py SPHERIZED.urdf SRDF [CLUSTER_SIZE] [N]"""
import itertools
import sys

import numpy as np

from clusters import link_clusters
from cost_model import min_ball, typical_confs
from rb_common import FK, VAMP_JOINTS, link_spheres, random_confs, read_disabled


def main(urdf, srdf, size=8, n=1000):
    S = {l: s for l, s in link_spheres(urdf, merge_parts=False).items() if s}
    dis = read_disabled(srdf)
    links = list(S)
    pairs = [(a, b) for a, b in itertools.combinations(links, 2) if frozenset((a, b)) not in dis]
    SC = {l: np.array([c for c, _ in S[l]]) for l in links}
    SR = {l: np.array([r for _, r in S[l]]) for l in links}
    B = {l: min_ball(SC[l], SR[l]) for l in links}
    CL = {l: link_clusters(SC[l], SR[l], size) for l in links}
    nclu = sum(len(v) for v in CL.values())
    fk = FK(urdf)
    for label, Q in (("random", random_confs(n, seed=5)), ("typical", typical_confs(n, seed=5))):
        bt = ct = st = 0.0
        for q in Q:
            T = fk(dict(zip(VAMP_JOINTS, map(float, q))))
            W = {l: (T[l][:3, :3] @ B[l][0] + T[l][:3, 3], B[l][1]) for l in links}
            A = {l: [(T[l][:3, :3] @ SC[l][a] + T[l][:3, 3], r, m) for m, a, r in CL[l]] for l in links}
            for a, b in pairs:
                bt += 1
                if np.linalg.norm(W[a][0] - W[b][0]) >= W[a][1] + W[b][1]:
                    continue
                for ca, ra, ma in A[a]:
                    for cb, rb, mb in A[b]:
                        ct += 1
                        if np.linalg.norm(ca - cb) < ra + rb:
                            st += len(ma) * len(mb)
        print(f"{label:8s}: {sum(len(v) for v in S.values())} spheres in {nclu} clusters (size {size}); per conf: "
              f"link tests {bt / n:.0f}, cluster tests {ct / n:.0f}, sphere tests {st / n:.0f}, total {(bt + ct + st) / n:.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8, int(sys.argv[4]) if len(sys.argv) > 4 else 1000)
