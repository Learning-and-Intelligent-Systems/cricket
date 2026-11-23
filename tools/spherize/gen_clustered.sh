#!/bin/bash
# gen_clustered.sh STEM NAME URDF SRDF EE: two cricket passes in run/, STEM_h.hh out
set -e
unset CONDA_BUILD_SYSROOT
R="$HOME/qr_rig/droidkernel1123"
STEM=$1; NAME=$2; URDF=$3; SRDF=$4; EE=$5
PY="$HOME/Documents/GitHub/NQR/.pixi/envs/nhpn-ik/bin/python"
GEN="pixi run --manifest-path $R/cricket/pixi.toml $R/cricket/build/fkcc_gen"
cd "$R/run"
cat > "${STEM}_p1.json" <<EOF
{"name": "$NAME", "urdf": "$URDF", "srdf": "$SRDF", "end_effector": "$EE", "resolution": 32,
 "template": "templates/fk_template_loop.hh",
 "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template_loop.hh"}], "output": "${STEM}_p1.hh"}
EOF
$GEN "${STEM}_p1.json" -t "${STEM}_p1.hh" -o "${STEM}_p1_out.json"
(cd "$R/cricket/tools/spherize/rainbow" && $PY make_cricket_config.py "$R/run/${STEM}_p1_out.json" "$R/run/$URDF" "$NAME" "$EE" "$R/run/${STEM}_h.json" 8 "$URDF" "$SRDF")
sed -i '' 's/template_h\.hh/template_clustered.hh/g' "${STEM}_h.json"
$GEN "${STEM}_h.json" -t "${STEM}_h.hh" -o "${STEM}_h_out.json"
ls -la "${STEM}_h.hh"
