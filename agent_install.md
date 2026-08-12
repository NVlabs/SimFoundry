# SimFoundry — Agent Installation Guide

A procedural guide for a coding agent (Claude Code, Codex, etc.) installing SimFoundry
on a fresh Linux + NVIDIA machine. Written from an observed clean-room reinstall of commit
`b71bdb6` on an RTX 4090 / CUDA 12.8 box, where all 7 environments were built and verified.

**Read this whole file before running anything.** The install takes hours and several
steps are effectively irreversible once started.

**A from-scratch install fails four to five times before it succeeds.** Each failure is
documented in §6 with its fix. Budget for them — verified across two independent clean-room
installs on the same machine:

| # | Fails at | Fix |
|---|---|---|
| §6.1 | `simfoundry`, ~40 min in | `SIMFOUNDRY_FORCE_DEP_CHECKOUT=1` |
| §6.2 | `simfoundry`, ~50 min in | `--robot-asset-fallback-root` — **needs a copy of a non-public asset bundle; you may be hard-blocked here** |
| §6.5 | `simfoundry`, mesh/input step | build `evdev` with the system compiler |
| §6.6 | `simfoundry`, final FAISS step | kill and redo with `timeout` — it hangs silently, forever |
| §6.3 | `3dgrut` (last, optional) | patch conda hooks to be `nounset`-safe |

**Two operational rules that would have saved hours here:**

1. **Wrap every long step in `timeout`.** Two of this install's three multi-hour delays were
   silent hangs, not slow work.
2. **Never trust "a process exists" as progress.** Check CPU time (§6.6). Also confirm an
   installer is *actually running* after any manual intervention — it is easy to fix one
   failure and forget to restart the chain.

> **Shell note for agents:** in `zsh` (the default on many dev machines and on macOS),
> unquoted parameter expansion does **not** word-split. `bash run.sh $ARGS` passes the whole
> string as one argument, so flags silently become a single Hydra override and the stage
> fails with a confusing error. Always write flags literally, or use `bash -c`.

---

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
command -v mamba uv git-lfs ffmpeg gcloud

# CUDA 12.8 must exist at this exact path (install_any6d.sh hard-codes it)
ls -d /usr/local/cuda-12.8

# Disk: need ~250 GB free for all 7 envs + deps + checkpoints
df -h .
```

**Hard requirements:**

| Requirement | Why | Failure if missing |
|---|---|---|
| `mamba` (Miniforge) | every installer calls it | exits 127 immediately |
| `uv` | `install_3dgrut.sh` only (verified: no other installer references it) | 3dgrut env fails, others fine |
| `/usr/local/cuda-12.8` | `CUDA_HOME` in **`install_simfoundry.sh:82`** *and* `install_any6d.sh:75` | **core env fails**, not just any6d |
| `git-lfs` | `install_articulate.sh` only — optional articulation component | articulation checkouts corrupt; core install unaffected |
| ~170 GB free disk | **measured**: 160 GB for 9 envs + `deps/` + `checkpoints/`. Without articulation, ~135 GB. Excludes the HF cache (~23 GB), which is shared and usually already present. | mid-build ENOSPC |
| 24 GiB VRAM | the standard video pipeline | OOM at stage 7 |

**Do not** rely on `git submodule update --init --recursive` from the docs — this repo
has no `.gitmodules`. All dependencies are cloned by the install scripts into `deps/`.

---

## 2. Choose the install scope

> **Two things to settle BEFORE running anything in this section.** Getting either wrong
> costs ~50 minutes before the failure appears.
>
> **(a) Always set `SIMFOUNDRY_FORCE_DEP_CHECKOUT=1`.** Without it, a *fresh clone* of
> `deps/BEHAVIOR-1K` is misdetected as local development and the pinned commit is never
> checked out. See §6.1.
>
> **(b) Check whether `og_cdc_assets` is reachable first** — if it is not, the command below
> **cannot build the `simfoundry` env at all**, because `install_everything.sh` does not
> accept or forward `--robot-asset-fallback-root`:
>
> ```bash
> GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/cremebrule/og_cdc_assets.git HEAD
> ```
>
> If that fails, go to **§6.2 now** and build `simfoundry` with `install_simfoundry.sh`
> directly, then return here and run `install_everything.sh` for the remaining envs (it skips
> the one already built). Do not discover this the slow way.

```bash
# All 7 envs (simfoundry, hunyuan, any6d, da3, void, nerfstudio_simfoundry, 3dgrut)
SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 bash scripts/installation/install_everything.sh

# Core reconstruction only — skips the auto-background trio
SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 \
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

`huggingface-cli` is **removed** — it now hard-errors with
`huggingface-cli is deprecated and no longer works. Use hf instead.`
Ignore the `huggingface-cli login` line in `README.md` and `docs/INSTALL.md`.

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

Do **not** probe a fixed filename like `config.json` — not every repo has one, and a missing
file is indistinguishable from a permissions failure. `netflix/void-model`, for example,
contains only `README.md` and two `.safetensors`, so a `config.json` probe reports a false
`DENIED`. Ask the API whether the repo is listable instead:

```bash
python - <<'PY'
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
api = HfApi()
for r in ["facebook/sam3",
          "facebook/dinov3-vitl16-pretrain-lvd1689m",
          "briaai/RMBG-2.0",
          "netflix/void-model",
          "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"]:
    try:
        api.list_repo_files(r)
        print(f"{r:<50} OK")
    except GatedRepoError:
        print(f"{r:<50} DENIED - request access on the model page")
    except RepositoryNotFoundError:
        print(f"{r:<50} NOT FOUND / no access")
    except Exception as e:
        print(f"{r:<50} ERROR {type(e).__name__}")
PY
```

`black-forest-labs/FLUX.1-Kontext-dev` is optional — only needed if a config sets
`model: flux`. The default configs use Gemini.

### 3b. Google Cloud / Gemini

All VLM stages (A stages 3, 5, 6, 10 and the whole B pipeline) need Gemini. There are
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

Note: the docs claim a four-file test subset "works before any environment is built."
That is only true if `pytest` is available in whatever Python you invoke; it is not in
a stock Miniforge `base`.

---

## 6. Known failure modes

### 6.1 `ImportError: cannot import name 'DepthEncoderConfig' from 'lerobot.configs'`

**This is the default outcome of a from-scratch install.** It aborts
`install_simfoundry.sh` at the `download_omnigibson_robot_assets()` step.

Cause: `git_safe.sh::git_safe_checkout_detached` refuses to move a repo whose branch has
commits the pinned SHA lacks, to protect local work. A *freshly cloned* `deps/BEHAVIOR-1K`
is on `main`, which is hundreds of commits ahead of the pin — so a pristine clone is
misread as local development and the pin is silently skipped:

```
NOTE: leaving deps/BEHAVIOR-1K as-is (branch 'main' has local commits).
      Not checking out d89aae4e...
```

You then get OmniGibson `main` (which needs the `wensi-ai/lerobot@release/b1k` fork) while
the installer force-installs upstream `lerobot@577cd109` — an incompatible pair.

Fix — force the pinned checkout and rebuild the env:

```bash
mamba env remove -n simfoundry -y && rm -rf ~/miniforge3/envs/simfoundry
SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 bash scripts/installation/install_everything.sh
```

Verify the pin took:

```bash
git -C deps/BEHAVIOR-1K rev-parse HEAD          # must equal BEHAVIOR1K_COMMIT
grep -n 'BEHAVIOR1K_COMMIT=' scripts/installation/install_simfoundry.sh
```

The same trap applies to `deps/Hunyuan3D-2.1` (the only other
`git_safe_checkout_detached` call site), so the env var protects both.

### 6.2 `ERROR: Required OmniGibson robot asset is missing: .../franka_robotiq.usda`

**The second guaranteed failure of a from-scratch install**, right after 6.1 is fixed.
Preceded a few lines earlier by the actual cause:

```
Fetching SimFoundry OmniGibson robot assets from https://github.com/cremebrule/og_cdc_assets.git...
fatal: could not read Username for 'https://github.com': No such device or address
WARNING: could not fetch SimFoundry robot assets; falling back to --robot-asset-fallback-root.
```

`github.com/cremebrule/og_cdc_assets` is not publicly accessible (anonymous HTTPS prompts for
credentials; SSH returns `Repository not found`). It supplies the `franka_robotiq` end
effector, which the public OmniGibson download does **not** carry and which
`install_simfoundry.sh:591` marks `required`.

Ignore the error text's claim that "the public OmniGibson robot asset download should provide
this file" — it does not, and re-running the public download will not help.

**You cannot fix this by yourself.** Ask the user whether they have a copy of the bundle or
access to the repo. Look for one before asking:

```bash
find "$HOME" -maxdepth 8 -type d -name franka_robotiq 2>/dev/null
```

If a copy exists, `--robot-asset-fallback-root` expects the layout
`<root>/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/<rel_path>`. A copy in any other
layout needs a symlink shim:

```bash
SHIM=/tmp/asset_fallback
mkdir -p "$SHIM/deps/BEHAVIOR-1K/datasets"
ln -sfn /path/to/omnigibson-robot-assets "$SHIM/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets"

SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 bash scripts/installation/install_simfoundry.sh \
  --project-root "$PWD" --env-name simfoundry --default \
  --robot-asset-fallback-root "$SHIM"
```

**`install_everything.sh` does not accept or forward `--robot-asset-fallback-root`**, so the
`simfoundry` env must be built by calling `install_simfoundry.sh` directly. Afterwards,
re-run `install_everything.sh` normally — it skips the existing `simfoundry` env and
continues with the other six.

Confirm success:

```
Copying optional robot asset from fallback: models/franka/franka_robotiq
Completed installation of SimFoundry environment: simfoundry
```

If the repo becomes public, or you set `OG_SIMFOUNDRY_ASSETS_REPO` to a reachable mirror
whose **root** contains `models/`, none of this is needed.

### 6.3 `CONDA_BACKUP_CXX: unbound variable` during `install_3dgrut.sh`

The third guaranteed failure. `3dgrut` is the **last and optional** env (PLY → USDZ for
auto-backgrounds), so skip it entirely unless the user needs backgrounds.

```
.../envs/3dgrut/etc/conda/deactivate.d/deactivate-gxx_linux-64.sh: line 68:
  CONDA_BACKUP_CXX: unbound variable
```

Upstream `create_conda.sh` runs `conda install`, which triggers a conda reactivation that
sources hooks dereferencing `CONDA_BACKUP_*` with no default, under `nounset`.

**Do not** try to fix this by editing `set -euo pipefail` in `create_conda.sh` — the nounset
state comes from conda's activation machinery, not that script. Patch the hooks instead:

```bash
cd ~/miniforge3/envs/3dgrut/etc/conda
for f in activate.d/*.sh deactivate.d/*.sh; do
  cp -n "$f" "$f.orig"; printf 'set +u\n' | cat - "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
cd -
bash scripts/installation/install_3dgrut.sh --project-root "$PWD" --env-name 3dgrut --default
```

The env already exists at that point and `create_conda.sh` skips creation, so the re-run
resumes and completes. Verify with `mamba run -n 3dgrut python -c "import threedgrut"`.

### 6.4 `install_articulate.sh` blocks on a sudo password — you cannot complete it

Articulation is the one component an agent **cannot install unattended**.
[`install_articulate.sh:45`](scripts/installation/install_articulate.sh#L45) runs `sudo -v`,
and line 55 runs `sudo apt-get install`, both unconditionally — even when every package is
already present. There is no `--skip-system-deps` flag.

Check whether the system deps are actually needed before involving the user:

```bash
blender --version | head -1        # must match BLENDER_VERSION (default 4.2.3)
git lfs version
for p in ffmpeg libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libgl1; do
  printf '%-16s ' "$p"; dpkg -s "$p" 2>/dev/null | grep -q "^Status: install ok installed" \
    && echo installed || echo MISSING
done
```

If all are present, the sudo block is a no-op and you have three options — **ask the user
which**, do not pick for them:

1. **They run it** (safest): `bash scripts/installation/install_articulate.sh --default`,
   typing the password once. Suggest `tmux`, since it runs 1-2h and an SSH drop kills it.
2. **You run a copy with lines 43-79 stripped** — only propose this after verifying every
   dependency above is present.
3. **Passwordless sudo** — the broadest change; only if they want it for other reasons.

Budget ~25 GB: both `articulate-anything-hunyuan` and `articulate-anything-partfield` are
built unconditionally, with no flag to select one.

### 6.5 `ERROR: Failed building wheel for evdev` — `KEY_LINK_PHONE` undeclared

Hits **any machine without a pre-existing pip cache**, i.e. every genuinely new install.

```
src/evdev/ecodes.c:541:29: error: 'KEY_LINK_PHONE' undeclared
ERROR: Mesh/input package installation failed with exit code 1.
```

`evdev` ships no manylinux wheel, so it always compiles. The installer has already put conda's
`gcc_linux-64` on PATH (for BEHAVIOR-1K), so pip builds against conda's **sysroot** headers,
which lack the newer key codes the system headers have.

```bash
mamba activate simfoundry
CC=/usr/bin/gcc CXX=/usr/bin/g++ python -m pip install "evdev==1.9.3"
```

Then re-run `install_simfoundry.sh` — its `python -c 'import evdev'` guard makes it skip the
step. **Warning:** re-running restarts the whole script, including BEHAVIOR-1K's `setup.sh`;
budget ~30 minutes.

### 6.6 The install appears to hang with no output (usually FAISS)

`faiss_gpu.sh:12` runs `mamba install -c pytorch faiss-gpu -y` with **no timeout and no
retry**, and it is the *last* step of the *first* env. A stalled socket wedges it forever
while the log's last line looks like normal conda output.

**Always distinguish hung from slow before waiting:**

```bash
ps -o pid,stat,etime,time,cmd -p <pid>
```

`ELAPSED 51:39` with `TIME 00:00:00` means it has used no CPU and is **hung**, not working.
Confirm with `cat /proc/<pid>/wchan` (`do_poll` = blocked on I/O).

Recovery — kill it and redo that step with a cap, which succeeds immediately:

```bash
kill -9 <mamba pid>; pkill -f install_simfoundry.sh
timeout 600s mamba install -n simfoundry -c pytorch faiss-gpu=1.12 -y
mamba run -n simfoundry python -c "import faiss; print(faiss.get_num_gpus())"
```

Since faiss is the final step, the env is otherwise complete — no full rebuild needed.

**Generally: wrap every long install step in `timeout` and watch for stale logs.** Two of the
three multi-hour delays in this project's install were silent hangs, not slow work.

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

### B stage 6 crashes with `FileNotFoundError: gaussian_da3.usdz` (then exit 139)
`s13_og.include_gs` defaults to `true` but the splat is only produced by the opt-in
auto-background flow. Pass `s13_og.include_gs=false` to **both** A and B.

### `Unknown Gemini model 'gemini-3-pro-image-preview'`
Several configs use a `-preview` suffix that `vlm.py` does not register. Override with
the GA name: `prompt_cousin_structured.image_model=gemini-3-pro-image`.

### Out of VRAM at stage 7 on a 24 GiB card

`s7_mesh.low_vram` defaults to `false`, which needs **~29 GB** for shape generation
(`real2sim_cfg.yaml:166`) — more than a 24 GiB card has, despite `docs/INSTALL.md` saying
"24 GiB works for the standard video pipeline". Always pass:

```bash
-- s7_mesh.low_vram=true
```

**Do not add `--no-stream` reflexively.** With the `hunyuan` backend, `low_vram=true` alone
was sufficient on a 4090: stage 7 ran 9m 4s across 3 streamed calls with no OOM. Reach for
`--no-stream` only if the streaming scheduler actually stalls or rejects the stage — it
serialises stages 5-8 and reloads the model per object, which is much slower.

---

## 7. Running the pipeline

These are the exact commands verified end-to-end on a 24 GiB card. Every override is
required — each one avoids a defect documented in `summary.md`.

```bash
export GCLOUD_PROJECT=<project>   # config reads it; auth may still use GEMINI_API_KEY

# A — reconstruction (~20 min for a 3-object tabletop scene)
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <name> --video-fpath /path/to/video.mov \
  -- s13_og.include_gs=false s7_mesh.low_vram=true

# B — augmentation
bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name <name> \
  -- s13_og.include_gs=false \
     prompt_cousin_structured.image_model=gemini-3-pro-image

# C — OmniGibson smoke test
bash scripts/pipeline/C_application/run.sh --scene-name <name> --mode smoke-random
```

| Override | Prevents |
|---|---|
| `s13_og.include_gs=false` (**both** A and B) | B stage 6 `FileNotFoundError` on `gaussian_da3.usdz`, then exit 139 |
| `s7_mesh.low_vram=true` | stage 7 OOM — the default needs ~29 GB |
| `image_model=gemini-3-pro-image` | `assert_valid_key` rejecting the stale `-preview` name |

**Expect benign noise in the logs**, none of it fatal:
- `Error importing diffusers ... Requires Flash-Attention >=2.7.1,<=2.7.4 but got 2.8.3` —
  printed by every VLM stage; disables only the FLUX backend.
- `AttributeError: 'NoneType' object has no attribute 'GetCamera'` from
  `omni.kit.widget.viewport` — Isaac Sim's headless shutdown, fires repeatedly in stages
  10-13.

Filter both out when monitoring, or real failures get lost in them.

Useful flags: `--include`/`--exclude` to select stages, `--dry-run` to print the plan,
`--detect-articulation` for stage 8b, `--bg-splat` for stage 2c.

**`--detect-articulation` does not degrade gracefully.** The availability check tests
only whether the stage *script* exists, not whether the `articulate-anything-*` conda
envs do. Confirm the envs exist before passing the flag:

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
| `articulate-anything-{hunyuan,partfield}` | `install_articulate.sh` | stage 8b (optional) |

There is **no** `b1k` env — despite `--env-b1k` defaulting to `b1k` in
`A_reconstruction/run.sh`. B and C correctly default it to `simfoundry`, and no A stage
uses that role, so the stale default is harmless in practice.
