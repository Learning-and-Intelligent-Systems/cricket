"""Arm links vs the head camera mount (rby1_camera_mount, the master's 9.5 cm
ZED sphere on link_head_2) -- exact (Coal, the master's sphere at the ACTUAL
head pan/tilt) vs a VAMP sphere model (whose head is baked at pan = tilt = 0,
or whatever the model's camera spheres cover).

Confs: torso at a home knee, left arm at rest, right arm uniform in the
master limits (and the mirror for the left arm). For each head tilt, the
fraction of confs where an arm link really hits the camera mount, and how
many of those the sphere model flags (arm spheres vs camera spheres).
With an SRDF, only the arm/head pairs it leaves enabled count.
    camera_audit.py SPHERIZED.urdf [SRDF|-] [N]"""
import sys

import coal
import numpy as np

from rb_common import FK, MASTER, OPEN_FINGERS, VAMP_JOINTS, link_geoms, link_hulls, link_spheres, master_bounds, read_disabled, vamp_values

ARM = {
    "right": [f"link_right_arm_{i}" for i in range(7)] + ["FT_sensor_R", "ee_right", "ee_finger_r1", "ee_finger_r2"],
    "left": [f"link_left_arm_{i}" for i in range(7)] + ["FT_sensor_L", "ee_left", "ee_finger_l1", "ee_finger_l2"],
}
CAMERA = "rby1_camera_mount"
HEAD_LINKS = ["link_head_0", "link_head_1", "link_head_2", CAMERA]


def confs(n, side, seed=0):
    lo, hi = master_bounds()
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, 23))
    knees = [np.deg2rad([0.0, -30.0, 60.0, -15.0, 0.0, 0.0]), np.deg2rad([0.0, 30.0, -60.0, 45.0, 0.0, 0.0])]
    idx = [VAMP_JOINTS.index(f"{side}_arm_{i}") for i in range(7)]
    for k in range(n):
        Q[k, 3:9] = knees[k % 2]
        Q[k, idx] = rng.uniform(lo[idx], hi[idx])
    return Q


def main(urdf, srdf=None, n=6000, tilts=(0.0, 0.5236, 0.6981, 1.2), pans=(0.0,)):
    dis = read_disabled(srdf) if srdf else set()
    G = link_geoms(MASTER)
    H = link_hulls(MASTER)
    fk = FK(MASTER)
    sfk = FK(urdf)
    S = link_spheres(urdf, merge_parts=False)
    cam = [p for p in G[CAMERA] if p[0] == "sphere"][0]
    cam_sph = coal.Sphere(cam[2])
    cvx = {}
    for l in ARM["right"] + ARM["left"] + HEAD_LINKS[:3]:
        pts = coal.StdVec_Vec3s()
        pts.extend(list(np.asarray(H[l][0].vertices, float)))
        cvx[l] = coal.Convex.convexHull(pts, False, "Qt")
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    cam_links = [l for l in S if l.split("__s")[0] in HEAD_LINKS and S[l]]
    for side in ("right", "left"):
        Q = confs(n, side, seed=3)
        real = {(p, t): np.zeros(n, bool) for p in pans for t in tilts}
        sph = np.zeros(n, bool)
        for k, q in enumerate(Q):
            vals = vamp_values(q)
            # sphere model: arm spheres vs the model's head/camera spheres
            TS = sfk(vals)
            W = {l: (np.array([c for c, _ in S[l]]) @ TS[l][:3, :3].T + TS[l][:3, 3], np.array([r for _, r in S[l]])) for l in S if S[l]}
            hitk = False
            for la in [l for l in W if l.split("__s")[0] in ARM[side]]:
                for lc in cam_links:
                    if frozenset((la, lc)) in dis:
                        continue
                    (CA, RA), (CC, RC) = W[la], W[lc]
                    if (np.linalg.norm(CA[:, None] - CC[None], axis=2) - RA[:, None] - RC[None]).min() < 0:
                        hitk = True
                        break
                if hitk:
                    break
            sph[k] = hitk
            for p in pans:
                for t in tilts:
                    T = fk({**vals, **OPEN_FINGERS, "head_0": p, "head_1": t})
                    Mc = T[CAMERA]
                    ctr = Mc[:3, :3] @ cam[1] + Mc[:3, 3]
                    hit = False
                    for l in ARM[side]:
                        res.clear()
                        d = coal.distance(cvx[l], coal.Transform3s(T[l][:3, :3], T[l][:3, 3]), cam_sph, coal.Transform3s(np.eye(3), ctr), req, res)
                        if d < 0:
                            hit = True
                            break
                    real[(p, t)][k] = hit
        print(f"[{side} arm, {n} confs] sphere model flags arm-vs-head/camera in {100 * sph.mean():.2f}% of confs")
        for (p, t), r in real.items():
            print(f"   head pan {np.degrees(p):5.1f} tilt {np.degrees(t):5.1f} deg: arm really hits the camera mount in {100 * r.mean():.2f}%"
                  f"  -> caught by the sphere model {int((r & sph).sum())}/{int(r.sum())}")


if __name__ == "__main__":
    main(sys.argv[1], None if len(sys.argv) < 3 or sys.argv[2] == "-" else sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6000)
