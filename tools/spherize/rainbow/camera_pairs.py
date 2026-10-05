"""Collidability of every link vs the SWEPT head-camera hull (respherize.SWEEP:
the camera sphere + head_1/head_2 over the head's pan/tilt range, in the camera
frame at VAMP's baked head pose). Census over random confs, then distance
minimisation for the pairs never seen colliding (as pair_minimize.py).
    camera_pairs.py OUT.json [N]"""
import json
import sys

import coal
import numpy as np
from scipy.optimize import minimize

from pair_minimize import path_joints
from respherize import SWEEP, swept_hull
from rb_common import FK, MASTER, OPEN_FINGERS, VAMP_JOINTS, link_hulls, master_bounds, random_confs, vamp_values

CAM = "rby1_camera_mount"


def main(out, n=40000):
    from qaa.robot_description import derived

    from rb_common import PROFILE

    H = link_hulls(MASTER)
    sw = swept_hull(CAM, SWEEP[CAM])
    pts = coal.StdVec_Vec3s(); pts.extend(list(np.asarray(sw.vertices, float)))
    cam = coal.Convex.convexHull(pts, False, "Qt")
    cvx = {}
    for l, hs in H.items():
        p = coal.StdVec_Vec3s(); p.extend(list(np.asarray(hs[0].vertices, float)))
        cvx[l] = coal.Convex.convexHull(p, False, "Qt")
    ex = {frozenset(p) for p in derived(PROFILE).self_collision_exclusions}
    rigid = {"link_torso_5", "link_head_0", "link_head_1", "link_head_2", CAM}
    others = [l for l in H if l not in rigid and frozenset((l, CAM)) not in ex]
    fk = FK(MASTER)
    req, res = coal.DistanceRequest(), coal.DistanceResult()
    Q = random_confs(n, seed=21)
    D = np.full((n, len(others)), np.inf)
    for i, q in enumerate(Q):
        T = fk({**vamp_values(q), **OPEN_FINGERS})
        tc = coal.Transform3s(T[CAM][:3, :3], T[CAM][:3, 3])
        for k, l in enumerate(others):
            res.clear()
            D[i, k] = coal.distance(cvx[l], coal.Transform3s(T[l][:3, :3], T[l][:3, 3]), cam, tc, req, res)
    lo, hi = master_bounds()
    result = {}
    for k, l in enumerate(others):
        m = float(D[:, k].min())
        if m > 0:
            idx = [VAMP_JOINTS.index(j) for j in path_joints(fk, l, CAM)]
            for q0 in Q[np.argsort(D[:, k])[:6]]:
                def f(s):
                    q = q0.copy(); q[idx] = np.clip(s, lo[idx], hi[idx])
                    T = fk({**vamp_values(q), **OPEN_FINGERS})
                    res.clear()
                    return coal.distance(cvx[l], coal.Transform3s(T[l][:3, :3], T[l][:3, 3]), cam, coal.Transform3s(T[CAM][:3, :3], T[CAM][:3, 3]), req, res)
                r = minimize(f, q0[idx], method="Nelder-Mead", options={"maxfev": 500})
                m = min(m, float(r.fun))
        result[f"{l}|{CAM}"] = dict(min=m, census_hit=float((D[:, k] < 0).mean()))
        print(f"{l:22s} vs swept camera: census hit {100 * (D[:, k] < 0).mean():6.2f}%  min {1000 * m:8.1f} mm")
    json.dump(result, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40000)
