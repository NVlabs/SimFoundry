#!/bin/bash
# CubePart (Roblox/cube cubepart) on the existing SAM 3D mesh.
# PartShapeDenoiserPipeline hardcodes num_parts = 8.
set -u
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="${CUDA_HOME}/bin:${PATH}"

FLOOR=8589934592
MIN_FREE=85899345920
MESH=/workspace/fruits-features/sam3d/0001.glb
PLY=/workspace/PartSAM/evaluation/results/0001.ply
OUT=/workspace/fruits-features/cubepart/0001
ROOT=/workspace/cube/cubepart
PY=/workspace/cubepart-env/bin/python
STATUS=/workspace/fruits-features/cubepart/status.json

free_bytes() {
  df -B1 / | awk 'NR==2 {print $4}'
}

ledger() {
  # $1 ok(true/false) $2 artifact $3 reason
  unset LD_LIBRARY_PATH
  unset LD_PRELOAD
  /opt/conda/bin/python3 - "$1" "$2" "$3" <<'PY'
import json, sys
sys.path.insert(0, "/workspace/sceneagent-simfoundry")
from sceneagent.agent_loop import append_step
ok = sys.argv[1] == "true"
artifact = sys.argv[2] or None
if artifact == "null":
    artifact = None
reason = sys.argv[3]
entry = append_step(
    "/workspace/fruits-agent",
    "cubepart",
    {"ok": ok, "artifact": artifact, "reason": reason, "detail": {}},
)
print(json.dumps(entry))
PY
}

echo "FREE_BEFORE $(free_bytes)"
FREE=$(free_bytes)
if [ "$FREE" -lt "$MIN_FREE" ]; then
  echo "STOP disk before download: $FREE bytes is under 80 GiB"
  exit 2
fi

mkdir -p /workspace/fruits-features/cubepart "$OUT"
if [ ! -f "$MESH" ] || [ ! -f "$PLY" ]; then
  ledger false null "CubePart was not run: SAM 3D mesh or PartSAM ply is missing."
  echo "MISSING_INPUT"
  exit 3
fi

echo "=== PART SCHEMA ==="
unset LD_LIBRARY_PATH
unset LD_PRELOAD
/workspace/sam3d-env/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
import trimesh
ply = "/workspace/PartSAM/evaluation/results/0001.ply"
mesh = trimesh.load(ply, force="mesh", process=False)
cols = np.asarray(mesh.visual.face_colors)[:, :3].astype(np.uint32)
packed = (cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]
uniq, counts = np.unique(packed, return_counts=True)
order = np.argsort(-counts)
# The released pipeline hardcodes num_parts = 8.
take = order[:8]
rows = []
for rank, idx in enumerate(take):
    color = int(uniq[idx])
    rows.append({
        "name": f"partsam_{rank:02d}",
        "faces": int(counts[idx]),
        "rgb": [(color >> 16) & 255, (color >> 8) & 255, color & 255],
    })
payload = {
    "source": ply,
    "distinct_face_colors": int(len(uniq)),
    "pipeline_num_parts": 8,
    "schema": rows,
    "note": "PartSAM wrote face colors and no part names. Names are ranks of those colors.",
}
Path("/workspace/fruits-features/cubepart/parts_schema.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
Path("/workspace/fruits-features/cubepart/parts.txt").write_text(
    ", ".join(row["name"] for row in rows), encoding="utf-8"
)
print("NCOLORS", len(uniq))
print("PARTS", ", ".join(row["name"] for row in rows))
PY

echo "=== CLONE ==="
if [ ! -f "$ROOT/examples/run_inference.py" ]; then
  rm -rf /workspace/cube
  git clone --depth 1 https://github.com/Roblox/cube.git /workspace/cube
fi
echo "CLONE_OK"

echo "=== VENV ==="
if [ ! -x "$PY" ]; then
  /workspace/sam3d-env/bin/python -m venv /workspace/cubepart-env
fi
"$PY" -m pip install -U pip wheel setuptools
if ! "$PY" -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121; then
  reason="CubePart install failed: torch 2.5.1 cu121 did not install. $(df -B1 / | awk 'NR==2{print \"free\", $4}')"
  echo "$reason"
  ledger false null "$reason"
  printf '%s\n' '{"status":"install_fail","stage":"torch"}' > "$STATUS"
  exit 4
fi
if ! "$PY" -m pip install -e "$ROOT"; then
  reason="CubePart install failed: pip install -e cubepart exited non-zero. $(df -B1 / | awk 'NR==2{print \"free\", $4}')"
  echo "$reason"
  ledger false null "$reason"
  printf '%s\n' '{"status":"install_fail","stage":"cube_part"}' > "$STATUS"
  exit 4
fi
echo "PIP_OK"
"$PY" -c 'import torch,cube_part; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

echo "=== WEIGHT GATE ==="
"$PY" - <<'PY'
import json, shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
token = Path("/root/.config/hf_token").read_text(encoding="utf-8").strip()
api = HfApi(token=token)
info = api.model_info("Roblox/cubepart", files_metadata=True)
want = {"multi_part_dit.safetensors", "vae.safetensors"}
sizes = {}
for sib in info.siblings:
    if sib.rfilename in want:
        sizes[sib.rfilename] = int(sib.size or 0)
        print("FILE", sib.rfilename, sib.size)
missing = sorted(want - set(sizes))
if missing:
    raise SystemExit("missing files " + ",".join(missing))
need = sum(sizes.values())
free = int(shutil.disk_usage("/").free)
floor = 8589934592
print("NEED", need)
print("FREE", free)
print("AFTER", free - need)
Path("/tmp/cubepart_need.txt").write_text(str(need), encoding="utf-8")
if free - need < floor:
    Path("/tmp/cubepart_gate.txt").write_text("reject", encoding="utf-8")
else:
    Path("/tmp/cubepart_gate.txt").write_text("ok", encoding="utf-8")
PY
GATE=$(cat /tmp/cubepart_gate.txt)
NEED=$(cat /tmp/cubepart_need.txt)
if [ "$GATE" != "ok" ]; then
  FREE=$(free_bytes)
  reason="CubePart weights are ${NEED} bytes. Free space was ${FREE} bytes. After that download $((FREE - NEED)) bytes would remain, under the 8 GiB floor of ${FLOOR} bytes. Not downloaded."
  echo "$reason"
  ledger false null "$reason"
  printf '%s\n' '{"status":"disk_reject","stage":"weights"}' > "$STATUS"
  exit 2
fi

echo "=== DOWNLOAD WEIGHTS ==="
"$PY" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
token = Path("/root/.config/hf_token").read_text(encoding="utf-8").strip()
path = snapshot_download(
    repo_id="Roblox/cubepart",
    allow_patterns=["multi_part_dit.safetensors", "vae.safetensors"],
    local_dir="/workspace/cube/cubepart/weights",
    token=token,
)
print("WEIGHT_DIR", path)
for name in ("multi_part_dit.safetensors", "vae.safetensors"):
    p = Path("/workspace/cube/cubepart/weights") / name
    print("SAVED", name, p.stat().st_size)
PY
echo "FREE_AFTER_WEIGHTS $(free_bytes)"

echo "=== TEXT ENCODER GATE ==="
"$PY" - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
token = Path("/root/.config/hf_token").read_text(encoding="utf-8").strip()
repo = "Qwen/Qwen3-VL-4B-Instruct"
api = HfApi(token=token)
info = api.model_info(repo, files_metadata=True)
need = 0
for sib in info.siblings:
    if sib.size:
        need += int(sib.size)
free = int(shutil.disk_usage("/").free)
floor = 8589934592
print("QWEN_NEED", need)
print("FREE", free)
print("AFTER", free - need)
Path("/tmp/qwen_need.txt").write_text(str(need), encoding="utf-8")
if free - need < floor:
    raise SystemExit(2)
snapshot_download(repo_id=repo, token=token)
print("QWEN_OK")
PY
qwen_rc=$?
echo "FREE_AFTER_QWEN $(free_bytes)"
if [ "$qwen_rc" -ne 0 ]; then
  FREE=$(free_bytes)
  reason="CubePart text encoder Qwen/Qwen3-VL-4B-Instruct was not downloaded (exit ${qwen_rc}). Free space is ${FREE} bytes. The part pipeline was not started."
  echo "$reason"
  ledger false null "$reason"
  printf '%s\n' '{"status":"install_fail","stage":"qwen"}' > "$STATUS"
  if [ "$qwen_rc" -eq 2 ]; then
    exit 2
  fi
  exit 4
fi

PARTS=$(cat /workspace/fruits-features/cubepart/parts.txt)
echo "RUN_PARTS $PARTS"
cd "$ROOT"
export PYTHONPATH="$ROOT"
echo "=== INFERENCE ==="
set +e
"$PY" examples/run_inference.py \
  --config configs/shape_denoiser_multimesh.yaml \
  --checkpoint weights/multi_part_dit.safetensors \
  --vae-checkpoint weights/vae.safetensors \
  --mesh "$MESH" \
  --parts "$PARTS" \
  --output "$OUT" \
  --guidance-scale 7.5 \
  --num-inference-steps 50 \
  --resolution-base 8.5 \
  --seed 0 \
  > "$OUT/infer.log" 2>&1
rc=$?
set -e
echo "INFER_RC $rc"
tail -n 40 "$OUT/infer.log" || true

if [ "$rc" -ne 0 ] && grep -qi "out of memory" "$OUT/infer.log"; then
  echo "=== INFERENCE RETRY LOWER RESOLUTION ==="
  set +e
  "$PY" examples/run_inference.py \
    --config configs/shape_denoiser_multimesh.yaml \
    --checkpoint weights/multi_part_dit.safetensors \
    --vae-checkpoint weights/vae.safetensors \
    --mesh "$MESH" \
    --parts "$PARTS" \
    --output "$OUT" \
    --guidance-scale 7.5 \
    --num-inference-steps 25 \
    --resolution-base 5.0 \
    --seed 0 \
    > "$OUT/infer_retry.log" 2>&1
  rc=$?
  set -e
  echo "RETRY_RC $rc"
  tail -n 40 "$OUT/infer_retry.log" || true
fi

echo "=== OUTPUTS ==="
find "$OUT" -type f -printf '%s %p\n' | sort -n
echo "FREE_AFTER $(free_bytes)"

"$PY" - <<'PY'
import json
from pathlib import Path
out = Path("/workspace/fruits-features/cubepart/0001")
files = []
for p in sorted(out.rglob("*")):
    if p.is_file() and p.suffix.lower() in {".glb", ".obj", ".ply"} and "infer" not in p.name:
        files.append({"path": str(p), "bytes": p.stat().st_size})
real = [f for f in files if f["bytes"] > 1000]
status = {
    "status": "artifact" if real else "no_artifact",
    "files": files,
}
Path("/workspace/fruits-features/cubepart/status.json").write_text(
    json.dumps(status, indent=2) + "\n", encoding="utf-8"
)
print("REAL", len(real))
PY

REAL=$("$PY" -c 'import json; print(len([f for f in json.load(open("/workspace/fruits-features/cubepart/status.json"))["files"] if f["bytes"]>1000]))')
if [ "$REAL" != "0" ]; then
  ART=$("$PY" -c 'import json; files=[f for f in json.load(open("/workspace/fruits-features/cubepart/status.json"))["files"] if f["bytes"]>1000]; files.sort(key=lambda f: f["bytes"], reverse=True); print(files[0]["path"])')
  BYTES=$("$PY" -c 'import json; files=[f for f in json.load(open("/workspace/fruits-features/cubepart/status.json"))["files"] if f["bytes"]>1000]; files.sort(key=lambda f: f["bytes"], reverse=True); print(files[0]["bytes"])')
  reason="Section 3.4 CubePart (https://github.com/Roblox/cube cubepart, checkpoints Roblox/cubepart) ran examples/run_inference.py on /workspace/fruits-features/sam3d/0001.glb. PartShapeDenoiserPipeline hardcodes num_parts = 8. PartSAM /workspace/PartSAM/evaluation/results/0001.ply has face colors and no part names, so the schema is the 8 largest color components, partsam_00 through partsam_07. Artifact ${ART} (${BYTES} bytes). Combined and per-part meshes are under ${OUT}."
  ledger true "$ART" "$reason"
  echo "LEDGER_ACCEPT"
else
  tailtxt=$(tail -n 8 "$OUT/infer.log" 2>/dev/null | tr '\n' ' ' | cut -c1-500)
  reason="CubePart ran and wrote no part mesh. examples/run_inference.py exit ${rc}. ${tailtxt}"
  ledger false null "$reason"
  echo "LEDGER_REJECT"
fi
echo "DONE_CUBEPART"
exit 0
