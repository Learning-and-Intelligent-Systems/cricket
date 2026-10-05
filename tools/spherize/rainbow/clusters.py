"""Spatial clusters of each link's spheres for the kernel's middle screening
level: k-means on the sphere centres (k = ceil(n / SIZE)); each cluster is
screened by a ball centred on one of its own spheres (the anchor, so the
kernel already has its world position) with a constant radius."""
import numpy as np


def kmeans(C, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    # k-means++ init
    cent = [C[rng.integers(len(C))]]
    for _ in range(1, k):
        d = np.min(np.linalg.norm(C[:, None] - np.array(cent)[None], axis=2), axis=1) ** 2
        cent.append(C[rng.choice(len(C), p=d / d.sum())] if d.sum() > 0 else C[rng.integers(len(C))])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(C[:, None] - cent[None], axis=2), axis=1)
        new = np.array([C[lab == j].mean(axis=0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return lab


def anchor_ball(C, R):
    """(anchor index, radius): the member sphere whose centre gives the
    smallest enclosing ball of all member spheres."""
    D = np.linalg.norm(C[:, None] - C[None], axis=2) + R[None, :]
    reach = D.max(axis=1)
    a = int(np.argmin(reach))
    return a, float(reach[a])


def link_clusters(C, R, size=8, seed=0):
    """[(member indices, anchor index, radius)]"""
    n = len(C)
    k = int(np.ceil(n / size))
    if k <= 1:
        a, r = anchor_ball(C, R)
        return [(np.arange(n), a, r)]
    best = None
    for s in range(4):  # a few restarts; keep the smallest total screening volume
        lab = kmeans(C, k, seed=seed + s)
        cl = []
        for j in range(k):
            m = np.where(lab == j)[0]
            if len(m) == 0:
                continue
            a, r = anchor_ball(C[m], R[m])
            cl.append((m, int(m[a]), r))
        score = sum(r**3 for _, _, r in cl)
        if best is None or score < best[0]:
            best = (score, cl)
    return best[1]
