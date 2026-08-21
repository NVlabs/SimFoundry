# SimFoundry — Agent Installation Guide

A procedural guide for a coding agent (Claude Code, Codex, etc.) installing SimFoundry
on a fresh Linux + NVIDIA machine.

**Read this whole file before running anything.** The install takes hours and several
steps are effectively irreversible once started.



## 0. Ground rules for agents

- **You cannot complete OAuth flows.** `gcloud auth application-default login` and
  `hf auth login` (interactive) open a browser. Use the non-interactive paths in §3.
  If credentials are absent and you cannot obtain them, **stop and ask the user** —
  do not attempt to work around auth.
- **Never delete a conda env without confirming it with the user first.** Env names
  are not reliably namespaced; unrelated projects live alongside SimFoundry.
- **Run the installer detached with a logfile.** It runs for hours; a dropped
  foreground process loses everything.
- **`install_everything.sh` skips envs that already exist.** A half-built env from a
  crashed run is silently treated as done. After any failure, delete the specific
  broken env before re-running (see §6).

---

## 1. Preflight

Run all of these and confirm before touching anything.

```bash
# GPU + driver
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Toolchain — all must resolve
command -v mamba git-lfs ffmpeg gcloud

# CUDA 12.8 must exist at this exact path (install_any6d.sh hard-codes it)
ls -d /usr/local/cuda-12.8

# Disk: need ~250 GB free for all 7 envs + deps + checkpoints
df -h .
```

**Hard requirements:**

| Requirement | Why | Failure if missing |
|---|---|---|
| `mamba` (Miniforge) | every installer calls it | exits 127 immediately |
| `/usr/local/cuda-12.8` | `install_any6d.sh` sets `CUDA_HOME` to it | any6d build fails |
| `git-lfs` | articulation repos store assets in LFS | corrupt checkouts |
| ~250 GB free disk | envs ≈ 100 GB, deps ≈ 82 GB, HF cache ≈ 12 GB | mid-build ENOSPC |
| ≥ 16 GiB VRAM | stage 5 peaks ~14 GiB; budget is 90% of total | stage 5 rejected pre-flight |

The repo has no git submodules. All dependencies are
cloned by the install scripts into `deps/`.
---

## 2. Choose the install scope

> Fresh clones of `deps/` repos are checked out to their pinned commits automatically —
> the guard in `git_safe.sh` compares a branch against *its own remote*, so only a dirty
> tree or genuinely unpushed local commits skip the pin (with a loud `NOTE:`). Set
> `SIMFOUNDRY_FORCE_DEP_CHECKOUT=1` only to overwrite such local state deliberately.
> See §6.1.

```bash
# All 7 envs (simfoundry, hunyuan, any6d, da3, void, nerfstudio_simfoundry, 3dgrut)
bash scripts/installation/install_everything.sh

# Core reconstruction only — skips the auto-background trio
bash scripts/installation/install_everything.sh --only "simfoundry hunyuan any6d da3"
```

The auto-background envs (`void`, `nerfstudio_simfoundry`, `3dgrut`) are only needed
for the Gaussian-splat background flow, which is **opt-in** (`--bg-splat`). Skip them
unless the user asked for backgrounds.

Always run detached with a log:

```bash
mkdir -p ~/simfoundry_logs
bash scripts/installation/install_everything.sh > ~/simfoundry_logs/install.log 2>&1
```

Then watch for phase transitions and failures:

```bash
tail -f ~/simfoundry_logs/install.log \
  | grep -E --line-buffered ">>> \[|DONE|Error occurred at line|^Error:|Traceback|No space left|Killed"
```

Each env begins with a `>>> [name] install_X.sh (env: name)` line. That is your
progress marker.

---

## 3. Authentication (do this before checkpoints)

Checkpoints are **opt-in** and require Hugging Face auth, so the order is:
install envs → authenticate → download checkpoints.

### 3a. Hugging Face


The repo docs demonstrate login, `hf auth login`.

Check existing auth first — it usually already exists and needs nothing:

```bash
hf auth whoami
```

If that prints a username, you are done. Otherwise the user must supply a token
(`HF_TOKEN`). Write it into the keys file without echoing it:

```bash
cd scripts/installation
cp api_keys.template.txt api_keys.txt
# then edit api_keys.txt — set HF_TOKEN and GCLOUD_PROJECT
chmod 600 api_keys.txt
```

**Gated models.** The user's HF account must have approved access to these, or later
stages fail with a 401/403 that does not name the cause:

```bash
for r in facebook/sam3 \
         facebook/dinov3-vitl16-pretrain-lvd1689m \
         briaai/RMBG-2.0 \
         netflix/void-model; do
  printf '%-50s ' "$r"
  hf download "$r" config.json --quiet >/dev/null 2>&1 && echo OK || echo DENIED
done
```

`black-forest-labs/FLUX.1-Kontext-dev` is optional — only needed if a config sets
`model: flux`. The default configs use Gemini.

### 3b. Google Cloud / Gemini

All VLM stages (A stages 3, 5, 6, 11 and the whole B pipeline) need Gemini. There are
two routes:

**Vertex AI (default).** Needs Application Default Credentials, which require an
interactive browser flow an agent cannot perform:

```bash
gcloud auth application-default login   # INTERACTIVE — user must run this
export GCLOUD_PROJECT=<project-id>
```

Verify without triggering the flow:

```bash
ls ~/.config/gcloud/application_default_credentials.json && gcloud config get-value project
```

**API key (agent-friendly fallback).** Generate a key at
<https://aistudio.google.com/api-keys>, then:

```bash
export GEMINI_API_KEY=<key>
```

`simfoundry/models/vlm.py::resolve_gemini_auth` prefers an API key when present and
falls back to Vertex+ADC otherwise. A file named `api_keys.txt` in the repo root (or
any parent dir) is auto-loaded into the environment by `load_api_keys()`, so
`GEMINI_API_KEY=...` in `<repo>/api_keys.txt` works without exporting anything.

Note this root `api_keys.txt` is a *different file* from
`scripts/installation/api_keys.txt` used by `login_services.sh`.

### 3c. Non-interactive login

```bash
bash scripts/installation/login_services.sh --default
```

Reads `scripts/installation/api_keys.txt`. Blank values are skipped, so a file with
only `HF_TOKEN` and `GCLOUD_PROJECT` is fine.

---

## 4. Checkpoints

```bash
bash scripts/installation/download_checkpoints.sh --default
```

**This is the flakiest step.** FoundationStereo and FoundationPose weights come from
Google Drive via `gdown --folder`, which is rate-limited and fails intermittently with
no useful error. If it fails, re-run it — it is idempotent and skips existing files.

If a machine already has a good copy:

```bash
bash scripts/installation/download_checkpoints.sh --default \
  --checkpoint-fallback-root /path/to/known-good/repo-copy
```

Weights land in `deps/void-model/` (VOID, ~41 GB) and `checkpoints/`. Do not move
them — several runners resolve paths against `VOID_ROOT=deps/void-model`.

---

## 5. Verification

```bash
# Each must print a path inside THIS checkout
for e in simfoundry any6d da3 hunyuan; do
  printf '%-12s ' "$e"
  mamba run -n "$e" python -c "import simfoundry; print(simfoundry.__file__)" 2>&1 | tail -1
done

# Catches both §6.1 failures at once. Expect: lerobot 0.3.4, numpy 1.26.4,
# torch 2.x+cu128, cuda True, and a simfoundry path inside this checkout.
mamba run -n simfoundry python -c "
import lerobot, omnigibson, simfoundry, torch, numpy
print('lerobot   ', lerobot.__version__)
print('numpy     ', numpy.__version__)
print('torch     ', torch.__version__, 'cuda', torch.cuda.is_available())
print('simfoundry', simfoundry.__file__)"

# Stage plans (executes nothing)
bash scripts/pipeline/A_reconstruction/run.sh --dry-run --include 1b,2
bash scripts/pipeline/B_augmentation/run.sh   --dry-run --include 1
bash scripts/pipeline/C_application/run.sh    --dry-run --mode smoke-random

# Tests
mamba run -n simfoundry python -m pytest -q
```

A path outside the checkout means a stale editable install is shadowing the package —
`pip uninstall simfoundry` in that env and re-run its installer.

Note: the four-file test subset in `docs/INSTALL.md` runs before any environment is
built, but needs `pytest` in whatever Python you invoke — it is not in a stock
Miniforge `base` (`pip install pytest` first, as the docs say).

---

## 6. Known failure modes



### `ERROR: Required OmniGibson robot asset is missing: .../franka_robotiq.usda`

The `franka_robotiq` end effector download failed. It comes from the **public Hugging
Face dataset** `behavior-1k/omnigibson-robot-assets` (a ~225 MB subtree via
`snapshot_download`, no token needed), fetched after OmniGibson's own public asset
download — which does *not* carry `franka_robotiq` on its own.

Causes:

- **HF unreachable / download failed.** Re-run the installer (the fetch is idempotent
  and skips when `franka_robotiq.usda` already exists), or point
  `OG_ROBOT_ASSETS_HF_REPO` at a mirror with the same dataset layout.
- **Fully offline machine with a local copy.** `--robot-asset-fallback-root` expects the
  layout `<root>/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/<rel_path>`; a copy
  in any other layout needs a symlink shim:

```bash
SHIM=/tmp/asset_fallback
mkdir -p "$SHIM/deps/BEHAVIOR-1K/datasets"
ln -sfn /path/to/omnigibson-robot-assets "$SHIM/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets"

bash scripts/installation/install_simfoundry.sh \
  --project-root "$PWD" --env-name simfoundry --default \
  --robot-asset-fallback-root "$SHIM"
```

**`install_everything.sh` does not accept or forward `--robot-asset-fallback-root`**, so
in the fallback case the `simfoundry` env must be built by calling
`install_simfoundry.sh` directly. Afterwards, re-run `install_everything.sh` normally —
it skips the existing `simfoundry` env and continues with the other six.

### Installer stops partway
`install_everything.sh` uses `set -euo pipefail`, so the first failure aborts every
remaining env. Because re-runs **skip existing envs**, a half-built env is treated as
complete. Always delete the broken env explicitly before re-running:

```bash
mamba env remove -n <broken-env> -y
rm -rf ~/miniforge3/envs/<broken-env>     # mamba sometimes leaves a stub
bash scripts/installation/install_everything.sh          # resumes at that env
```

### `ImportError: cannot import name 'BaseRobot' from 'omnigibson.robots'`
`deps/BEHAVIOR-1K` is on `main` instead of the pinned commit. OmniGibson main renamed
`BaseRobot`→`Robot` and dropped `FrankaPanda`. Fix:

```bash
grep -n 'BEHAVIOR1K_COMMIT=' scripts/installation/install_simfoundry.sh
git -C deps/BEHAVIOR-1K rev-parse HEAD    # must match
```

### `ImportError: cannot import name 'HF_LEROBOT_HOME'`
lerobot is too new. The pinned OmniGibson 3.8.0 needs lerobot 0.3.4:

```bash
mamba run -n simfoundry pip install --no-deps \
  "lerobot@git+https://github.com/huggingface/lerobot.git@577cd10974b84bea1f06b6472eb9e5e74e07f77a"
mamba run -n simfoundry python -c "import numpy; print(numpy.__version__)"   # expect 1.26.4
```

### Stage 2c fails with `ns-process-data: not found`
Stage 2c runs in `nerfstudio_simfoundry`, not `simfoundry`. It is opt-in — omit
`--bg-splat` if you did not build that env.

### Out of VRAM at stage 7 on a 24 GiB card

`s7_mesh.low_vram` defaults to `false`, which needs **~29 GB** for shape generation —
more than a 24 GiB card has (`docs/INSTALL.md` documents this). On 24 GiB, always pass:

```bash
-- s7_mesh.low_vram=true
```

**Do not add `--no-stream` reflexively.** With the `hunyuan` backend, `low_vram=true` alone
was sufficient on a 4090: stage 7 ran 9m 4s across 3 streamed calls with no OOM. Reach for
`--no-stream` only if the streaming scheduler actually stalls or rejects the stage — it
serialises stages 5-8 and reloads the model per object, which is much slower.

---

## 7. Running the pipeline

These commands were verified end-to-end on a 24 GiB card. The only override needed is
`s7_mesh.low_vram=true`, and only on ~24 GiB cards (see §6).

```bash
export GCLOUD_PROJECT=<project>   # config reads it; auth may still use GEMINI_API_KEY

# A — reconstruction (~20 min for a 3-object tabletop scene)
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <name> --video-fpath /path/to/video.mov \
  -- s7_mesh.low_vram=true

# B — augmentation
bash scripts/pipeline/B_augmentation/run.sh --scene-name <name>

# C — OmniGibson smoke test
bash scripts/pipeline/C_application/run.sh --scene-name <name> --mode smoke-random
```

| Override | Prevents |
|---|---|
| `s7_mesh.low_vram=true` (24 GiB cards) | stage 7 OOM — the default needs ~29 GB |

**Expect benign noise in the logs**, none of it fatal:
- `Error importing diffusers ... Requires Flash-Attention >=2.7.1,<=2.7.4 but got 2.8.3` —
  printed by every VLM stage; disables only the FLUX backend.
- `AttributeError: 'NoneType' object has no attribute 'GetCamera'` from
  `omni.kit.widget.viewport` — Isaac Sim's headless shutdown, fires repeatedly in stages
  11-14.

Filter both out when monitoring, or real failures get lost in them.

Useful flags: `--include`/`--exclude` to select stages, `--dry-run` to print the plan,
`--detect-articulation` for stage 9, `--bg-splat` for stage 2c.

**Articulation needs a minimum of 18 GiB VRAM.** The 16 GiB minimum covers only the
standard pipeline; stage 9's segmentation models allocate outside the VRAM scheduler.

**`--detect-articulation` degrades gracefully.** The availability check now verifies the
`deps/articulate-anything` checkout and the `articulate-anything-*` conda envs, not just
the stage script; if they are missing, the flag is ignored with a warning and the rest
of the pipeline runs. To confirm articulation will actually run:

```bash
mamba env list | grep articulate-anything
```

Env-name overrides, if yours differ from the defaults:
`--env-simfoundry`, `--env-da3`, `--env-mesh`, `--env-nerfstudio`, `--env-b1k`.
`--env-mesh` must match the backend in `s7_mesh.shape_model` (`hunyuan` → `hunyuan` env).

---

## 8. Quick reference

| Env | Built by | Used for |
|---|---|---|
| `simfoundry` | `install_simfoundry.sh` | most stages, VLM calls, OmniGibson |
| `hunyuan` | `install_hunyuan.sh` | stage 7 / B stage 3 mesh generation |
| `any6d` | `install_any6d.sh` | stage 8 pose matching |
| `da3` | `install_da3.sh` | stage 2 depth |
| `void` | `install_void.sh` | auto-BG inpainting (optional) |
| `nerfstudio_simfoundry` | `install_nerfstudio.sh` | stage 2c splat training (optional) |
| `3dgrut` | `install_3dgrut.sh` | PLY → USDZ (optional) |
| `articulate-anything-{hunyuan,partfield}` | `install_articulate.sh` | stage 9 (optional) |

There is **no** `b1k` env, and none is needed — `--env-b1k` defaults to `simfoundry`
in all three pipelines. Pass it only if you keep OmniGibson in a separate environment.
