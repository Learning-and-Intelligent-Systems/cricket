#!/bin/bash
# Rainbow's four VAMP kernels with the attachment hand's fingers compiled out
# (2026-10-12; vamp fingers-upstream-1012): rainbow and rainbowcam attach at
# tcp_right and leave out ee_finger_r1/r2, rainbowl and rainbowcaml attach at
# tcp_left and leave out ee_finger_l1/l2. The planner (HPN vamp_fingers) puts
# that hand's fingers on every check as the attachment, at its aperture.
#
# Inputs: INPUTS/{rainbow,rainbowcam}_spherized.urdf and .srdf, the 10-04
# refit's (vamp b239ec3 resources/rainbow, 1070/1134 spheres; this branch's
# resources/rainbow holds the older 197-sphere set); the finger links' collision spheres are dropped
# (strip_fingers.py), links and joints kept. The recipe is the 10-04 one:
# cricket in Docker, pass 1 (loop templates) numbers frames and spheres,
# ../make_cricket_config.py adds per-link sphere clusters (size 8), pass 2
# renders resources/templates/*_clustered.hh. With the fingers kept it
# reproduces vamp b239ec3's rainbow.hh byte for byte.
#
#   tools/spherize/rainbow/fingers/make_kernels.sh INPUTS OUTDIR
#   -> OUTDIR/{rainbow,rainbowl,rainbowcam,rainbowcaml}.hh for src/impl/vamp/robots/
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
IN=$(cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
PY=${PY:-/Users/tlp/Documents/GitHub/NQR/.pixi/envs/nhpn-ik/bin/python}
HERE=$ROOT/tools/spherize/rainbow
RUN=$OUT/run
mkdir -p $RUN/rainbow $RUN/templates
cp $ROOT/resources/templates/*.hh $RUN/templates/
for base in rainbow rainbowcam; do
  cp $IN/$base.srdf $RUN/rainbow/$base.srdf
  $PY $HERE/fingers/strip_fingers.py $IN/${base}_spherized.urdf $RUN/rainbow/${base}_R.urdf ee_finger_r1 ee_finger_r2
  $PY $HERE/fingers/strip_fingers.py $IN/${base}_spherized.urdf $RUN/rainbow/${base}_L.urdf ee_finger_l1 ee_finger_l2
done
DOCKER=(docker run --rm --cpus=2 --memory=4g -v "$RUN:/mount" --user "$(id -u):$(id -g)" cricket:latest)
kernel() {  # OUT URDF SRDF NAME EE
  local o=$1 urdf=$2 srdf=$3 name=$4 ee=$5
  cat > $RUN/${o}_p1.json <<J
{"name": "$name", "urdf": "$urdf", "srdf": "$srdf", "end_effector": "$ee", "resolution": 32,
 "template": "templates/fk_template_loop.hh", "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template_loop.hh"}],
 "output": "${o}_p1.hh"}
J
  "${DOCKER[@]}" /mount/${o}_p1.json -t /mount/${o}_p1.hh -o /mount/${o}_p1_out.json > $RUN/${o}_p1.log 2>&1
  (cd $RUN && PYTHONPATH=$HERE $PY $HERE/make_cricket_config.py ${o}_p1_out.json $urdf $name $ee ${o}_h.json 8 $urdf $srdf > /dev/null)
  $PY - <<P
import json
p = "$RUN/${o}_h.json"
d = json.load(open(p))
d["template"] = "templates/fk_template_clustered.hh"
d["subtemplates"] = [{"name": "ccfk", "template": "templates/ccfk_template_clustered.hh"}]
json.dump(d, open(p, "w"), indent=1)
P
  "${DOCKER[@]}" /mount/${o}_h.json -t /mount/${o}.hh -o /mount/${o}_out.json > $RUN/${o}_h.log 2>&1
  cp $RUN/${o}.hh $OUT/${o}.hh
  echo "$o: $(grep -m1 'n_spheres' $OUT/${o}.hh | tr -s ' ')"
}
kernel rainbow rainbow/rainbow_R.urdf rainbow/rainbow.srdf Rainbow tcp_right
kernel rainbowl rainbow/rainbow_L.urdf rainbow/rainbow.srdf Rainbowl tcp_left
kernel rainbowcam rainbow/rainbowcam_R.urdf rainbow/rainbowcam.srdf RainbowCam tcp_right
kernel rainbowcaml rainbow/rainbowcam_L.urdf rainbow/rainbowcam.srdf RainbowCaml tcp_left
