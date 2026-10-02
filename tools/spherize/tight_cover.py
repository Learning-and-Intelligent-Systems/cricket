"""Conservative tight sphere cover of a convex hull: every hull point (surface
samples, vertices, interior grid) inside some sphere, every sphere inside the
hull's delta-offset polytope (n_i . x <= b_i + delta), optionally above a
ground plane z >= z_floor (tangency, for a mounted base). Greedy set cover over
candidate centres (an interior grid plus the surface samples themselves, whose
radius is at least delta, so every point is coverable), then each chosen
sphere shrunk to the farthest point it is responsible for."""
import numpy as np
import trimesh


def hull_planes(h):
    n = h.face_normals
    b = np.einsum("ij,ij->i", n, h.triangles_center)
    P = np.unique(np.round(np.column_stack([n, b]), 9), axis=0)
    return P[:, :3], P[:, 3]


def cover(h, delta, surf_spacing=0.004, grid=None, z_floor=None, seed=0, max_spheres=400):
    n, b = hull_planes(h)
    grid = grid or max(0.5 * delta, 0.004)
    # points to cover
    area = h.area
    ns = int(max(2000, area / surf_spacing**2))
    surf, _ = trimesh.sample.sample_surface_even(h, ns, seed=seed)
    lo, hi = h.bounds
    ax = [np.arange(lo[i] + grid / 2, hi[i], grid) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    inside = (G @ n.T <= b + 1e-9).all(axis=1)
    G = G[inside]
    E = []
    for a, b2 in h.vertices[h.edges_unique]:
        k = max(2, int(np.linalg.norm(b2 - a) / 0.004) + 1)
        E.append(a + np.linspace(0, 1, k)[:, None] * (b2 - a))
    U = np.vstack([surf, h.vertices, G, np.vstack(E)])
    if z_floor is not None:
        U = U[U[:, 2] >= z_floor + 1e-9]
    # candidate centres: interior grid + surface samples + vertices
    C = np.vstack([G, surf[:: max(1, len(surf) // 3000)], h.vertices])
    R = (b[None, :] + delta - C @ n.T).min(axis=1)  # stay within the delta polytope
    if z_floor is not None:
        R = np.minimum(R, C[:, 2] - z_floor)  # no sphere below the floor
    keep = R > 1e-4
    C, R = C[keep], R[keep]
    # coverage matrix (points x candidates), chunked
    cov = []
    for i in range(0, len(U), 4000):
        d = np.linalg.norm(U[i : i + 4000, None, :] - C[None], axis=2)
        cov.append(d <= R[None] + 1e-9)
    cov = np.vstack(cov)
    uncovered = np.ones(len(U), bool)
    if not cov.any(axis=1).all():
        bad = (~cov.any(axis=1)).sum()
        raise RuntimeError(f"{bad} points no candidate can cover (delta too small or floor)")
    chosen = []
    while uncovered.any() and len(chosen) < max_spheres:
        gain = cov[uncovered].sum(axis=0)
        j = int(np.argmax(gain))
        if gain[j] == 0:
            break
        chosen.append(j)
        uncovered &= ~cov[:, j]
    # shrink: each point to its nearest chosen centre that covers it; radius = farthest
    Cc, Rc = C[chosen], R[chosen]
    d = np.linalg.norm(U[:, None, :] - Cc[None], axis=2)
    ok = d <= Rc[None] + 1e-9
    d_masked = np.where(ok, d, np.inf)
    owner = np.argmin(d_masked, axis=1)
    newR = np.zeros(len(Cc))
    for k in range(len(Cc)):
        m = owner == k
        if m.any():
            newR[k] = d[m, k].max()
    used = newR > 0
    return [(c, float(r) + 1e-4) for c, r in zip(Cc[used], newR[used])], int(uncovered.sum())
