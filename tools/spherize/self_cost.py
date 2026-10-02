"""Expected self-collision work per conf for a cricket kernel: per enabled
pair, P(link bounding spheres overlap) x n_i x n_j sphere tests (the generated
fkcc checks a pair's bounding spheres first, then every sphere pair)."""
import sys

import numpy as np
from lxml import etree

from fit_report import link_spheres
from selfpair_audit import FK


def bounding_sphere(spheres, iters=2000):
    C = np.array([c for c, _ in spheres]); R = np.array([r for _, r in spheres])
    c = C.mean(axis=0)
    for i in range(iters):  # Badoiu-Clarkson style for balls
        far = np.argmax(np.linalg.norm(C - c, axis=1) + R)
        d = C[far] - c
        n = np.linalg.norm(d)
        if n > 1e-12:
            c = c + d / n * ((n + R[far]) - (np.linalg.norm(C - c, axis=1) + R).max() * 0) / (i + 2)
    r = (np.linalg.norm(C - c, axis=1) + R).max()
    return c, r


def main(urdf, srdf, n=4000):
    S = {k: v for k, v in link_spheres(urdf).items() if v}
    fk = FK(urdf)
    dis = {frozenset((d.get("link1"), d.get("link2"))) for d in etree.parse(srdf).getroot().findall("disable_collisions")}
    links = list(S)
    pairs = [(a, b) for i, a in enumerate(links) for b in links[i + 1:] if frozenset((a, b)) not in dis]
    B = {l: bounding_sphere(S[l]) for l in links}
    rng = np.random.default_rng(0)
    lo = np.array([j["lo"] for j in fk.active]); hi = np.array([j["hi"] for j in fk.active])
    hits = {p: 0 for p in pairs}
    for q in rng.uniform(lo, hi, size=(n, len(lo))):
        T = fk(q)
        W = {l: (T[l][:3, :3] @ B[l][0] + T[l][:3, 3], B[l][1]) for l in links}
        for a, b in pairs:
            if np.linalg.norm(W[a][0] - W[b][0]) < W[a][1] + W[b][1]:
                hits[(a, b)] += 1
    rows = sorted(((hits[p] / n * len(S[p[0]]) * len(S[p[1]]), hits[p] / n, p) for p in pairs), reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"{urdf}: {sum(len(v) for v in S.values())} spheres, {len(pairs)} enabled pairs, "
          f"expected sphere tests per conf {tot:.0f} (static max {sum(len(S[a]) * len(S[b]) for a, b in pairs)})")
    for cost, p, (a, b) in rows[:8]:
        print(f"   {a:28s} {b:34s} P(bound overlap) {p:5.2f}  n {len(S[a])}x{len(S[b])}  exp {cost:7.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3000)
