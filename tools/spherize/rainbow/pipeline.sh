#!/bin/bash
# Rainbow VAMP refit, end to end (2026-10-04). Reproduces the candidate inputs
# and headers in this directory; the kernel build is the last step.
#
#   pipeline.sh rainbow      -> mapF.urdf/.srdf, cricket_run/rainbow{,l}_h.hh
#   pipeline.sh rainbowcam   -> mapFcam.urdf/.srdf, cricket_run/rainbowcam{,l}_h.hh
#
# Inputs: QAA's master (rainbow_seiji.urdf / rainbowcam_seiji.urdf) and the
# current cricket input template_plus.urdf (cricket 30eed3e
# resources/rainbow/rainbow_spherized_plus.urdf: kinematics, fixed head and
# fingers, VAMP's base convention). Python: NQR's nhpn-ik env.
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/tlp/Documents/GitHub/NQR/.pixi/envs/nhpn-ik/bin/python
QAA_R=/Users/tlp/Documents/GitHub/QAA/src/qaa/assets/robots/rainbow_description
which=${1:-rainbow}

# 1. the cover (per-link deltas in make_candidate.sh; TIGHT faces and the swept
#    head camera in respherize.py)
./make_candidate.sh mapE_tmp.urdf > resph_pipeline.out
$PY - <<'EOF'
import respherize as R
R.respherize("mapE_tmp.urdf", "mapF.urdf", 0.015, {}, pad=0.001, spacing=0.003, links=["rby1_camera_mount"])
EOF

# 2. which pairs can collide at all: random census + distance minimisation
#    (+ every link vs the SWEPT head camera)
$PY exact_pairs.py 40000 exact_pairs_40k.npz 0 8
$PY exact_pairs.py 200000 exact_pairs_200k.npz 1 8
$PY - <<'EOF'
import numpy as np
a = np.load("exact_pairs_40k.npz", allow_pickle=True); b = np.load("exact_pairs_200k.npz", allow_pickle=True)
np.savez("exact_pairs_240k.npz", D=np.vstack([a["D"], b["D"]]), Q=np.vstack([a["Q"], b["Q"]]), pairs=a["pairs"])
EOF
$PY pair_minimize.py exact_pairs_40k.npz pair_min.json 8
NSEED=8 NRAND=8 $PY pair_minimize.py exact_pairs_240k.npz pair_min2.json 10
$PY camera_pairs.py camera_pairs.json 20000
$PY make_srdf.py mapF.urdf mapF.srdf exact_pairs_240k.npz pair_min.json pair_min2.json camera_pairs.json

if [ "$which" = rainbowcam ]; then
  export SPHERIZE_MASTER=$QAA_R/rainbowcam_seiji.urdf SPHERIZE_PROFILE=rainbowcam
  $PY respherize.py mapF.urdf mapFcam.urdf 8 --pad=1 --spacing=3 --links=link_right_arm_camera,link_left_arm_camera
  $PY exact_pairs.py 100000 cam_exact_100k.npz 2 6
  NSEED=8 NRAND=4 $PY pair_minimize.py cam_exact_100k.npz cam_pair_min.json 8
  $PY camera_pairs.py cam_camera_pairs.json 20000
  $PY make_srdf.py mapFcam.urdf mapFcam.srdf cam_exact_100k.npz pair_min.json pair_min2.json cam_pair_min.json cam_camera_pairs.json
  URDF=mapFcam.urdf; SRDF=mapFcam.srdf; NAME=RainbowCam; NAMEL=RainbowCaml; STEM=rainbowcam
else
  URDF=mapF.urdf; SRDF=mapF.srdf; NAME=Rainbow; NAMEL=Rainbowl; STEM=rainbow
fi

# 3. cricket, two passes: pass 1 numbers the frames and spheres, pass 2 adds
#    the sphere clusters the kernel screens with (templates/*_h.hh)
cd cricket_run
cp ../$URDF rainbow/${STEM}_cand.urdf
cp ../$SRDF rainbow/${STEM}_cand.srdf
cat > ${STEM}_p1.json <<EOF
{"name": "$NAME", "urdf": "rainbow/${STEM}_cand.urdf", "srdf": "rainbow/${STEM}_cand.srdf",
 "end_effector": "tcp_right", "resolution": 32, "template": "templates/fk_template_loop.hh",
 "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template_loop.hh"}], "output": "${STEM}_p1.hh"}
EOF
docker run --rm -v "$PWD:/mount" --user "$(id -u):$(id -g)" cricket:latest /mount/${STEM}_p1.json -t /mount/${STEM}_p1.hh -o /mount/${STEM}_p1_out.json
$PY ../make_cricket_config.py ${STEM}_p1_out.json rainbow/${STEM}_cand.urdf $NAME tcp_right ${STEM}_h.json 8 rainbow/${STEM}_cand.urdf rainbow/${STEM}_cand.srdf
$PY - <<EOF
import json
d = json.load(open("${STEM}_h.json")); d["name"] = "$NAMEL"; d["end_effector"] = "tcp_left"; d["output"] = "${STEM}l_fk.hh"
json.dump(d, open("${STEM}l_h.json", "w"), indent=1)
EOF
for m in $STEM ${STEM}l; do
  docker run --rm -v "$PWD:/mount" --user "$(id -u):$(id -g)" cricket:latest /mount/${m}_h.json -t /mount/${m}_h.hh -o /mount/${m}_h_out.json
done
cd ..

# 4. a robot-only kernel build (vamp_src: a local clone, branch rainbow-refit-1004)
#   cp cricket_run/rainbow_h.hh vamp_src/src/impl/vamp/robots/rainbow.hh
#   $PY -m pip install -v --no-deps --target vamp_h1 \
#       -C cmake.define.VAMP_ROBOT_MODULES=rainbow -C cmake.define.VAMP_ROBOT_STRUCTS=Rainbow ./vamp_src
