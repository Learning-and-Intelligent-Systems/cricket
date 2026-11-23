#!/bin/bash
# A robot's kernel in the clustered form, in two cricket passes:
#
#   gen_clustered.sh STEM NAME URDF SRDF EE [OUTDIR]
#   e.g. gen_clustered.sh droid Droid droid/droid_spherized.urdf droid/droid_partial.srdf grasp_frame
#
# URDF and SRDF are relative to resources/. Pass 1 (the _loop templates) numbers
# the frames and spheres; make_cricket_config.py groups each link's spheres into
# clusters; pass 2 (the _clustered templates) writes OUTDIR/STEM_fk.hh and
# resources/STEM_clustered.json. Needs build/fkcc_gen and a Python with numpy
# and lxml (PYTHON, default python3).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STEM=$1; NAME=$2; URDF=$3; SRDF=$4; EE=$5; OUT=${6:-$ROOT}
PY=${PYTHON:-python3}
GEN="$ROOT/build/fkcc_gen"
TMP=$(mktemp -d)
cd "$ROOT/resources"
cat > "${STEM}_p1.json" <<EOF
{"name": "$NAME", "urdf": "$URDF", "srdf": "$SRDF", "end_effector": "$EE", "resolution": 32,
 "template": "templates/fk_template_loop.hh",
 "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template_loop.hh"}], "output": "${STEM}_p1.hh"}
EOF
"$GEN" "${STEM}_p1.json" -t "$TMP/p1.hh" -o "$TMP/p1_out.json"
rm "${STEM}_p1.json"
(cd "$ROOT/tools/spherize/rainbow" && "$PY" make_cricket_config.py "$TMP/p1_out.json" "$ROOT/resources/$URDF" "$NAME" "$EE" "$ROOT/resources/${STEM}_clustered.json" 8 "$URDF" "$SRDF")
"$GEN" "${STEM}_clustered.json" -t "$OUT/${STEM}_fk.hh" -o "$TMP/p2_out.json"
rm -r "$TMP"
ls -la "$OUT/${STEM}_fk.hh"
