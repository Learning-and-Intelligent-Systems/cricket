"""The built kernel's attachment path (fkcc_attach) vs a Python reference:
a held box (spheres at the TCP) checked against the scene and against the
robot links cricket lists as able to touch the hand (end_effector_collisions),
on top of the robot's own env + self verdict.
    attach_check.py VAMPDIR SET.npz URDF SRDF CRICKET_OUT.json [N]"""
import itertools
import json
import sys

import numpy as np
import trimesh

from rb_common import FK, VAMP_JOINTS, link_spheres, read_disabled


def main(vdir, setf, urdf, srdf, cout, n=4000, module="rainbow", tcp="tcp_right"):
    sys.path.insert(0, vdir)
    import vamp

    assert vamp.__file__.startswith(vdir)
    d = np.load(setf)
    Q = d["Q"][:n].astype(np.float32)
    out = json.load(open(cout))
    names = out["link_names"]
    ee_links = {names[i] for i in out["end_effector_collisions"]}
    # held object: a 6 x 6 x 20 cm box 12 cm out along the TCP approach axis, as spheres
    att_local = [((0.0, 0.0, 0.10 + 0.03 * k), 0.035) for k in range(5)]
    env = vamp.Environment()
    boxes = []
    for c, e, h in zip(d["centers"], d["eulers"], d["halfs"]):
        env.add_cuboid(vamp.Cuboid(list(map(float, c)), list(map(float, e)), list(map(float, h))))
        T = trimesh.transformations.euler_matrix(*e, axes="sxyz")
        boxes.append((T[:3, :3], np.asarray(c, float), np.asarray(h, float)))
    att = vamp.Attachment(np.eye(4, dtype=np.float32))
    att.add_spheres([vamp.Sphere(list(map(float, c)), float(r)) for c, r in att_local])
    env.attach(att)
    fk = FK(urdf)
    S = {l: s for l, s in link_spheres(urdf, merge_parts=False).items() if s}
    dis = read_disabled(srdf)
    pairs = [(a, b) for a, b in itertools.combinations(list(S), 2) if frozenset((a, b)) not in dis]
    SC = {l: np.array([c for c, _ in S[l]]) for l in S}
    SR = {l: np.array([r for _, r in S[l]]) for l in S}
    AC = np.array([c for c, _ in att_local]); AR = np.array([r for _, r in att_local])

    def box_hit(C, R):
        for Rb, cb, hb in boxes:
            P = (C - cb) @ Rb
            qd = np.abs(P) - hb
            dist = np.linalg.norm(np.maximum(qd, 0), axis=1) + np.minimum(qd.max(axis=1), 0)
            if (dist < R).any():
                return True
        return False

    dis_n = 0
    for q in Q:
        T = fk(dict(zip(VAMP_JOINTS, map(float, q))))
        W = {l: SC[l] @ T[l][:3, :3].T + T[l][:3, 3] for l in S}
        hit = box_hit(np.vstack([W[l] for l in S]), np.concatenate([SR[l] for l in S]))
        if not hit:
            for a, b in pairs:
                if (np.linalg.norm(W[a][:, None] - W[b][None], axis=2) < SR[a][:, None] + SR[b][None]).any():
                    hit = True
                    break
        A = AC @ T[tcp][:3, :3].T + T[tcp][:3, 3]
        if not hit:
            hit = box_hit(A, AR)
        if not hit:
            for l in ee_links:
                if (np.linalg.norm(A[:, None] - W[l][None], axis=2) < AR[:, None] + SR[l][None]).any():
                    hit = True
                    break
        dis_n += (not hit) != bool(getattr(vamp, module).validate(q, env))
    print(f"attachment path ({module}, {tcp}): kernel vs Python reference disagree on {dis_n} of {len(Q)} confs")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], a[4], int(a[5]) if len(a) > 5 else 4000, a[6] if len(a) > 6 else 'rainbow', a[7] if len(a) > 7 else 'tcp_right')
