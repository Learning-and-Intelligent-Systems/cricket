"""Conservative tight sphere cover of a convex hull (sparse lazy-greedy set
cover, for hulls too big for tight_cover.py's dense coverage matrix).

Every target point (surface samples, vertices, edge samples, an interior grid)
lies in some sphere; every sphere lies inside the hull's delta-offset polytope
(n_i . x <= b_i + delta) and, for a link standing on the floor, above the
floor (bottom >= floor_z + pad + 0.5 mm, targets only above floor_cover_z).
After the greedy pick, each sphere shrinks to the farthest target it owns;
the caller adds the radius pad and verifies on an independent dense sample.
"""
import heapq

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def hull_planes(h):
    n = h.face_normals
    b = np.einsum("ij,ij->i", n, h.triangles_center)
    P = np.unique(np.round(np.column_stack([n, b]), 9), axis=0)
    return P[:, :3], P[:, 3]


def edge_samples(h, step=0.002):
    pts = []
    for a, b in h.vertices[h.edges_unique]:
        k = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts.append(a + np.linspace(0, 1, k)[:, None] * (b - a))
    return np.vstack(pts)


def interior_grid(h, spacing, n, b):
    lo, hi = h.bounds
    ax = [np.arange(lo[i] + spacing / 2, hi[i], spacing) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return G[(G @ n.T <= b + 1e-9).all(axis=1)]


def targets(h, surf_spacing=0.004, int_spacing=None, seed=0, floor_cover_z=None):
    n, b = hull_planes(h)
    ns = int(max(2000, h.area / surf_spacing**2))
    surf, _ = trimesh.sample.sample_surface_even(h, ns, seed=seed)
    ext = (h.bounds[1] - h.bounds[0]).min()
    int_spacing = int_spacing or min(0.012, max(0.004, ext / 6))
    G = interior_grid(h, int_spacing, n, b)
    U = np.vstack([surf, h.vertices, edge_samples(h, 0.004), G])
    if floor_cover_z is not None:
        U = U[U[:, 2] >= floor_cover_z]
        # the hardest slab: just above the floor-cover plane, spheres are floor-capped
        lo, hi = h.bounds
        ax = [np.arange(lo[i], hi[i], 0.002) for i in range(2)]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 2)
        slab = []
        for z in (floor_cover_z, floor_cover_z + 0.002, floor_cover_z + 0.005):
            Z = np.column_stack([P, np.full(len(P), z)])
            slab.append(Z[(Z @ n.T <= b + 1e-9).all(axis=1)])
        U = np.vstack([U] + slab)
    return U


def plane_deltas(n, delta, tight=None, h=None, b=None):
    """Per-plane offsets: `delta`, except planes whose normal has
    n . dir > thresh for a (dir, thresh, delta_t[, centre, radius]) in
    `tight`; with a centre, only planes with a hull face whose centroid lies
    within radius of it (link frame)."""
    d = np.full(len(n), float(delta))
    for t in tight or ():
        direction, thresh, dt = t[:3]
        m = n @ np.asarray(direction, float) > thresh
        if len(t) > 3:
            centre, radius = np.asarray(t[3], float), float(t[4])
            fn = h.face_normals
            fb = np.einsum("ij,ij->i", fn, h.triangles_center)
            near = np.linalg.norm(h.triangles_center - centre, axis=1) <= radius
            key = {tuple(np.round(np.r_[a, c], 9)) for a, c in zip(fn[near], fb[near])}
            m &= np.array([tuple(np.round(np.r_[a, c], 9)) in key for a, c in zip(n, b)])
        d[m] = np.minimum(d[m], dt)
    return d


def candidates(h, delta, n_int=20000, seed=0, floor=None, pad=0.0, tight=None):
    """Centres (random interior + hull-surface samples + vertices) and their
    largest admissible radius (delta polytope, floor)."""
    n, b = hull_planes(h)
    delta = plane_deltas(n, delta, tight, h, b)
    rng = np.random.default_rng(seed)
    lo, hi = h.bounds
    lo = lo - delta.max()
    hi = hi + delta.max()
    # interior of the delta-offset polytope (centres may sit just outside the hull)
    P = []
    need = n_int
    while need > 0:
        X = rng.uniform(lo, hi, size=(4 * n_int, 3))
        X = X[(X @ n.T <= b + 0.5 * delta[None, :]).all(axis=1)]
        P.append(X[:need])
        need -= len(X[:need])
    surf, _ = trimesh.sample.sample_surface_even(h, 4000, seed=seed + 1)
    C = np.vstack(P + [surf, h.vertices])
    R = (b[None, :] + delta[None, :] - C @ n.T).min(axis=1)
    if floor is not None:
        R = np.minimum(R, C[:, 2] - floor - pad - 0.0005 - 1.5e-4)  # the +1e-4 shrink slack and the pad stay above the floor
    keep = R > 1e-4
    return C[keep], R[keep]


def lazy_greedy(U, C, R):
    tree = cKDTree(U)
    sets = tree.query_ball_point(C, R + 1e-9)
    sets = [np.asarray(s, dtype=np.int64) for s in sets]
    unc = np.ones(len(U), bool)
    heap = [(-len(s), j) for j, s in enumerate(sets) if len(s)]
    heapq.heapify(heap)
    chosen = []
    left = len(U)
    while left > 0 and heap:
        g, j = heapq.heappop(heap)
        gain = int(unc[sets[j]].sum())
        if gain == 0:
            continue
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-gain, j))
            continue
        chosen.append(j)
        unc[sets[j]] = False
        left -= gain
    return chosen, unc


def cover(h, delta, floor=None, floor_cover=None, pad=0.0, seed=0, n_int=20000, extra_targets=None, surf_spacing=0.004, tight=None):
    """[(centre, radius)] before the pad; raises if a target is uncoverable."""
    U = targets(h, surf_spacing=surf_spacing, seed=seed, floor_cover_z=floor_cover)
    if extra_targets is not None and len(extra_targets):
        U = np.vstack([U, extra_targets])
    C, R = candidates(h, delta, n_int=n_int, seed=seed, floor=floor, pad=pad, tight=tight)
    chosen, unc = lazy_greedy(U, C, R)
    if unc.any():
        # targets no random centre reaches (tight spots by the floor): offer the
        # targets themselves, and points pulled toward the hull centroid, as centres
        n, b = hull_planes(h)
        dl = plane_deltas(n, delta, tight, h, b)
        P = U[unc]
        extra = [P] + [P + t * (h.centroid - P) for t in (0.02, 0.05, 0.1, 0.2)]
        X = np.vstack(extra)
        RX = (b[None, :] + dl[None, :] - X @ n.T).min(axis=1)
        if floor is not None:
            RX = np.minimum(RX, X[:, 2] - floor - pad - 0.0005 - 1.5e-4)
        keep = RX > 1e-4
        C, R = np.vstack([C, X[keep]]), np.concatenate([R, RX[keep]])
        chosen, unc = lazy_greedy(U, C, R)
    if unc.any():
        raise RuntimeError(f"{unc.sum()} target points uncoverable (delta {delta}, floor {floor})")
    Cc, Rc = C[chosen], R[chosen]
    # shrink: each target to its nearest covering chosen centre; radius = farthest owned
    tree = cKDTree(Cc)
    k = min(len(Cc), 16)
    d, idx = tree.query(U, k=k)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    ok = d <= Rc[idx] + 1e-9
    d_masked = np.where(ok, d, np.inf)
    col = np.argmin(d_masked, axis=1)
    owner = idx[np.arange(len(U)), col]
    dist = d[np.arange(len(U)), col]
    bad = ~np.isfinite(d_masked[np.arange(len(U)), col])
    if bad.any():  # owner beyond the 16 nearest: fall back to a brute-force pass
        for i in np.where(bad)[0]:
            dd = np.linalg.norm(Cc - U[i], axis=1)
            m = dd <= Rc + 1e-9
            jj = np.where(m)[0][np.argmin(dd[m])]
            owner[i], dist[i] = jj, dd[jj]
    newR = np.zeros(len(Cc))
    np.maximum.at(newR, owner, dist)
    used = newR > 0
    return [(c, float(r) + 1e-4) for c, r in zip(Cc[used], newR[used])]
