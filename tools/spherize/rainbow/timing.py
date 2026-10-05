"""Kernel timing on collision-FREE work (no early exit): validate on confs
that are valid for both kernels, and validate_motion on short free right-arm
moves, in the bench's table scene and in an empty env.
    timing.py VAMPDIR|- SET.npz VALID_MASK.npz"""
import sys
import time

import numpy as np


def main(vdir, setf, maskf):
    if vdir != "-":
        sys.path.insert(0, vdir)
    import vamp

    r = vamp.rainbow
    d = np.load(setf)
    m = np.load(maskf)["mask"]
    Q = d["Q"][m].astype(np.float32)
    env = vamp.Environment()
    for c, e, h in zip(d["centers"], d["eulers"], d["halfs"]):
        env.add_cuboid(vamp.Cuboid(list(map(float, c)), list(map(float, e)), list(map(float, h))))
    empty = vamp.Environment()
    out = {}
    for name, E in (("scene", env), ("empty", empty)):
        for q in Q[:300]:
            r.validate(q, E)
        t0 = time.perf_counter()
        ok = [r.validate(q, E) for q in Q]
        out[f"validate_{name}"] = 1e6 * (time.perf_counter() - t0) / len(Q)
        assert all(ok), sum(not x for x in ok)
    # free motions: from each conf, a 0.05 rad step of the right arm that stays valid
    rng = np.random.default_rng(0)
    pairs = []
    for q in Q:
        b = q.copy()
        b[16:23] += rng.uniform(-0.05, 0.05, 7).astype(np.float32)
        if r.validate(b, env) and r.validate_motion(q, b, env, False):
            pairs.append((q, b))
    for name, E in (("scene", env), ("empty", empty)):
        t0 = time.perf_counter()
        for a, b in pairs:
            r.validate_motion(a, b, E, False)
        out[f"motion_{name}"] = 1e6 * (time.perf_counter() - t0) / len(pairs)
    print(f"{vamp.__file__.split('/')[-3]}: {len(Q)} free confs, {len(pairs)} free moves | " + "  ".join(f"{k} {v:.2f} us" for k, v in out.items()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
