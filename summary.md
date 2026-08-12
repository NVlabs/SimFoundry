# SimFoundry Beta Test — Findings Summary

**Date:** 2026-08-11
**Commit tested:** `b71bdb6` ("Integrate fixes for release"), `origin/main`
**Machine:** Linux 6.8.0, RTX 4090 (24 GiB), driver 580.173.02, CUDA 12.8, Miniforge
**Test performed:** full clean-room reinstall — deleted all 5 SimFoundry conda envs
(`simfoundry`, `hunyuan`, `da3`, `pixal3d`, `articulate-anything-hunyuan`), deleted
`deps/` (82 GB) and `checkpoints/`, then ran `install_everything.sh` from scratch.

---

## Verdict

**A from-scratch install cannot succeed by following the documented steps.** It fails twice
inside the very first environment, each time with an error that names neither the real cause
nor the fix:

1. **B1** — the pinned dependency commit is silently skipped on a fresh clone, producing an
   incompatible OmniGibson/lerobot pair ~40 min in. Worked around with
   `SIMFOUNDRY_FORCE_DEP_CHECKOUT=1`, which nothing in the docs suggests setting.
2. **B2** — a *required* robot asset is fetched from a GitHub repo that is not publicly
   accessible. **There is no workaround without a pre-existing copy of that bundle**, so an
   external beta tester is hard-blocked here.

Both are in `install_simfoundry.sh`, the first of seven environments. A third failure (**M4**)
stops the last environment, `3dgrut`.

**Final outcome: with three workarounds applied, all 7 environments build and verify clean**,
checkpoints download without error, and **all three pipelines (A, B, C) run end-to-end on a
real video in ~33 minutes total**. So the system works — but the install is not achievable by
following the documentation, and not at all without a copy of an asset bundle that is not
publicly available.

| Env | Status | Verified |
|---|---|---|
| `simfoundry` | OK *(needed workarounds B1 + B2)* | lerobot 0.3.4, numpy 1.26.4, torch 2.7.0+cu128, cuda True, `import omnigibson` OK |
| `hunyuan` | OK | pin `82920d64` applied, torch 2.7.0+cu128, cuda True |
| `any6d` | OK | torch 2.7.0+cu128, cuda True |
| `da3` | OK | torch 2.7.0+cu128, cuda True |
| `void` | OK | torch 2.7.1+cu128, av 12.3.0, numpy 1.26.4 |
| `nerfstudio_simfoundry` | OK | gsplat 1.5.3, hydra 1.3.5, `ns-process-data` present |
| `3dgrut` | OK *(needed workaround M4)* | `import threedgrut` OK, torch 2.8.0+cu128, cuda True |

All four envs expose the editable `simfoundry` package resolving into this checkout. All three
documented `--dry-run` stage plans pass (exit 0) with correct env routing. `pytest` reports
7 failures, all stale tests rather than install defects (**H4**).

Separately, the release commit introduced a config/default inconsistency that makes the
default A→B pipeline path crash (**H1**).

---

## Scope — what this test did and did not cover

**Covered:** a full clean-room install of all 7 environments, checkpoint download, service
login, the documented verification steps (imports, three `--dry-run` stage plans, `pytest`),
and one complete A → B → C run on a real video.

**Not covered** — findings below say nothing about these areas:

- **Articulation (stage 8b) — now fully covered.** `install_articulate.sh` was run to
  completion and stage 8b was driven through all 5 steps on two scenes (a 25-object cluttered
  desk and a 2-object oven scene). **No articulated asset could be produced** — see **B3**.
  Steps 1-4 work; step 5 is broken by two deterministic bugs. This exercise also produced
  **H3** (guard), **H6** (OOM), **H7** (headless render), **H8** (repeated parts), **H9**
  (poisoned cache), **M8** and **M9** (installer).
- **The auto-background / Gaussian-splat flow.** `void`, `nerfstudio_simfoundry`, and `3dgrut`
  were built and import-verified, but stage 2c and the 7-step `auto_bg_reconstruction`
  pipeline were never executed. `--bg-splat` was not exercised.
- **B at full scale.** Because of **M6**, the run used the wrapper's demo preset (2 of 3
  objects, 2 images each). Its 12m 15s is a floor, not a representative time.
- **C beyond `smoke-random`.** The teleop modes (stages 2/2b) need TeleMoMa, which SimFoundry
  deliberately does not install; policy evaluation and demo generation were not run.
- **Stereo/ZED input.** Only the `video` pipeline mode was tested.
- **Breadth.** Two scenes (3-object and 25-object tabletop), both video-mode, on one machine
  (RTX 4090 / CUDA 12.8 / Miniforge). Findings that depend on GPU size or CUDA version may
  differ elsewhere — notably **L6**, which is specific to 24 GiB cards.

---

## Blockers

### B1 — A fresh clone silently ignores the pinned dependency commit

| | |
|---|---|
| **Severity** | Blocker — every from-scratch install |
| **Where** | [`scripts/installation/git_safe.sh:54-58`](scripts/installation/git_safe.sh#L54-L58) |
| **Symptom** | `ImportError: cannot import name 'DepthEncoderConfig' from 'lerobot.configs'`, aborting `install_simfoundry.sh:554` |

`git_safe_checkout_detached` declines to move a repo when its current branch has commits
the target SHA lacks, on the theory that this indicates local development worth preserving:

```bash
if [[ -n "${branch}" && -n "${target_sha}" ]] \
   && [[ "$(git -C "${repo_dir}" rev-list --count "${target_sha}..HEAD")" != "0" ]]; then
  _git_safe_skip_notice "${label}" "branch '${branch}' has local commits" "${target}"
  return
fi
```

A **freshly cloned** repo trivially satisfies that condition: it lands on `main`, which is
359 commits ahead of the pinned `d89aae4e`. The guard cannot distinguish "user has local
work" from "clone is simply newer than the pin" — and the latter is the normal case.

The install continues, because the skip is a `NOTE:` on stdout, not a non-zero exit:

```
NOTE: leaving deps/BEHAVIOR-1K as-is (branch 'main' has local commits).
      Not checking out d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864, so your local work is preserved.
      Commit/stash your changes, or set SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 to overwrite.
```

The failure surfaces ~40 minutes later. `deps/BEHAVIOR-1K` is left on OmniGibson `main`,
whose `OmniGibson/setup.py:58` requires the `wensi-ai/lerobot@release/b1k` fork, while
[`install_simfoundry.sh:501`](scripts/installation/install_simfoundry.sh#L501) force-installs
upstream `lerobot@577cd109` with `--no-deps`. That pair is incompatible.

**The pins themselves are correct.** At the pinned commit, `OmniGibson/setup.py:85`
requires exactly `huggingface/lerobot@577cd10974b84bea1f06b6472eb9e5e74e07f77a` — the same
SHA the installer installs. Only the skipped checkout breaks it.

`deps/Hunyuan3D-2.1` ([`install_hunyuan.sh:122`](scripts/installation/install_hunyuan.sh#L122))
is the only other call site and has the same exposure.

**Workaround:** `SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 bash scripts/installation/install_everything.sh`

**Suggested fixes** (any one closes it):
- Skip the "local commits" heuristic when the repo is on an untouched default branch with
  no local commits *relative to its own remote* — i.e. compare against `origin/HEAD`, not
  against the pin.
- Only apply the guard when the worktree is dirty or HEAD is not an ancestor of `origin/*`.
- Default `SIMFOUNDRY_FORCE_DEP_CHECKOUT=1` for a directory the installer itself just cloned.
- At minimum, make the skip fail loudly rather than print a `NOTE:` and continue.

### B2 — Install requires an asset bundle from a repo that is not publicly accessible

| | |
|---|---|
| **Severity** | Blocker — no external user can complete the install |
| **Where** | [`scripts/installation/install_simfoundry.sh:561`](scripts/installation/install_simfoundry.sh#L561), required at [`:591`](scripts/installation/install_simfoundry.sh#L591) |
| **Symptom** | `fatal: could not read Username for 'https://github.com'` → `ERROR: Required OmniGibson robot asset is missing: .../franka_robotiq/usd/franka_robotiq.usda`, exit 1 |

`install_simfoundry.sh` clones `https://github.com/cremebrule/og_cdc_assets.git` to obtain the
`franka_robotiq` end effector, which `validate_robot_asset_file(... required)` treats as
mandatory. That repository is not reachable:

| Access path | Result |
|---|---|
| Anonymous HTTPS | prompts for a username — i.e. not public |
| Beta tester's SSH key | `ERROR: Repository not found.` |
| *Same* SSH key vs `NVlabs/SimFoundry` | resolves normally (auth is fine) |

The public OmniGibson robot-asset download supplies `franka_panda`, `franka_mounted`, and
`franka_dexhand` — but **not** `franka_robotiq`, exactly as `docs/INSTALL.md` states. So there
is no supported way for an external user to satisfy a required dependency.

Two things make it harder to diagnose than it needs to be:

- **The error message points at the wrong source.** It says *"The public OmniGibson robot
  asset download should provide this file"* — but per `docs/INSTALL.md`, `franka_robotiq`
  comes from the `og_cdc_assets` bundle, not the public download. A tester will go and
  re-debug the public asset download, which is working correctly.
- **The real cause is only a `WARNING:`.** The clone failure degrades to *"falling back to
  --robot-asset-fallback-root"* and the script continues, failing three lines later. The
  actionable message is not the one that stops the build.

**Suggested fixes** (in preference order):
1. Make `og_cdc_assets` public, or mirror `franka_robotiq` into the SimFoundry release
   artifacts / Hugging Face.
2. If it must stay private, fail immediately at the clone with an actionable message naming
   the repo and `--robot-asset-fallback-root`, and document how to obtain the bundle.
3. Correct the error text so it names `og_cdc_assets` rather than the public download.

**Workaround (requires a pre-existing copy of the bundle).** `--robot-asset-fallback-root`
expects the layout `<root>/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/<rel_path>`, so
a copy stored in any other layout needs a shim:

```bash
SHIM=/tmp/asset_fallback
mkdir -p "$SHIM/deps/BEHAVIOR-1K/datasets"
ln -sfn /path/to/omnigibson-robot-assets "$SHIM/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets"

SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 bash scripts/installation/install_simfoundry.sh \
  --project-root "$PWD" --env-name simfoundry --default \
  --robot-asset-fallback-root "$SHIM"
```

Note `install_everything.sh` does **not** accept or forward `--robot-asset-fallback-root`, so
this environment must be built by invoking `install_simfoundry.sh` directly.

Verified working on this machine: a copy existed at `/home/gear/Projects/real2sim-assets/`,
and with the shim the installer logged `Copying optional robot asset from fallback:
models/franka/franka_robotiq` and completed.

### B3 — Articulation (stage 8b) cannot produce a URDF: two deterministic bugs in step 5

| | |
|---|---|
| **Severity** | Blocker — articulation is a headline release feature (`README.md:35`, "V0 rigid-body and articulation generation") and cannot complete for any object |
| **Where** | `deps/articulate-anything/` — `articulate_joint.py:179` vs `joint_actor.py:373`, and `cotracker_utils.py:97` |

Stage 8b was run to step 5 on a clean single-door object (`toaster oven`), after fixing
**H6**, **H7** and **H9**. Steps 1-4 all succeeded — classification, tree, segmentation, and
merge (`door: [3,4,5]`, `toaster_oven_base: [0,1]`). Step 5 then failed **all 5 actor-critic
attempts**, with two distinct deterministic errors and **no URDF produced**.

**Bug 1 — CoTracker checkpoint does not match the model architecture** (attempt `iter_0`):

```
RuntimeError: Error(s) in loading state_dict for CoTrackerThreeOffline:
  Missing key(s):    updateformer.vis_conf_head.weight, corr_mlp.fc1.weight, ...
  Unexpected key(s): pos_emb, norm.weight, track_feat_updater.0.weight, ...
  size mismatch for time_emb: checkpoint [1,16,456] vs model [1,16,1110]
  size mismatch for updateformer.flow_head.weight: checkpoint [130,384] vs model [2,384]
```

The pinned `deps/co-tracker` code and the downloaded weights are different CoTracker
generations. `joint_actor.py:587 -> load_predicted_rendering() -> make_cotracker()` cannot
load the model at all, so the video-modality path is dead on arrival.

**Bug 2 — API signature mismatch** (attempts `iter_1` through `iter_4`):

```
TypeError: JointPredictionActor._make_prompt_parts() missing 1 required
positional argument: 'link_placement_path'
```

`joint_actor.py:373-374` declares `link_placement_path` as the **first required positional
parameter**, but the only caller, `articulate_joint.py:179`, invokes:

```python
joint_actor.generate_prediction(gt_input=cfg.prompt, **retry_kwargs, **cfg.gen_config)
```

which never supplies it. This is a plain Python API break — it cannot succeed for any object,
scene, model, or configuration. It is not VLM-dependent and not stochastic.

**Why this outranks H8.** H8 (repeated-part naming) blocks multi-drawer objects at step 4.
B3 blocks *every* object at step 5, including the ideal single-door case that cleanly passes
step 4. Even with H8 fixed, no articulated asset can be produced.

**Evidence that steps 1-4 are healthy** — the oven run, with H6/H7/H9 worked around:

| Step | Result |
|---|---|
| classify | `Articulated: ['toaster oven']` — correct |
| s2 tree | `['toaster_oven_base', 'door']`, joint type `revolute` — correct |
| s3 segment | P3-SAM segmented the mesh, renders produced under EGL |
| s4 merge | `door: [3,4,5]`, `toaster_oven_base: [0,1]` — correct, no name mismatch |
| **s5 articulate** | **5/5 attempts failed; no `mobility.urdf`** |

**Suggested fix:** pass `link_placement_path` at `articulate_joint.py:179` (or give it a
default in `joint_actor.py:373`), and re-pin `deps/co-tracker` to the commit matching the
downloaded checkpoint. Both are small, but until they land `--detect-articulation` cannot
deliver its advertised output.

---

## High severity

### H1 — Default `include_gs: true` crashes B stage 6 (regression in `b71bdb6`)

| | |
|---|---|
| **Where** | [`scripts/cfg/real2sim_cfg.yaml:330`](scripts/cfg/real2sim_cfg.yaml#L330), [`6_sample_reconstructed_scene.py:1113`](scripts/pipeline/B_augmentation/stages/6_sample_reconstructed_scene.py#L1113) |
| **Symptom** | `FileNotFoundError` on `gaussian_da3.usdz`, then segfault (exit 139) |

Commit `b71bdb6` correctly made stage 2c (the splat producer) **opt-in** via `--bg-splat`,
defaulting off. But `s13_og.include_gs` still defaults to `true`. So the default
configuration now *consumes* an artifact the default pipeline no longer *produces*.

The two consumers disagree on how to handle the missing file:

- [`13_create_og_scene.py:387`](scripts/pipeline/A_reconstruction/stages/13_create_og_scene.py#L387) — checks `os.path.exists`, warns, and skips. Graceful.
- [`6_sample_reconstructed_scene.py:1113`](scripts/pipeline/B_augmentation/stages/6_sample_reconstructed_scene.py#L1113) — constructs `USDObject(usd_path=gs_path_da3)` with **no existence check and no try/except**. Hard crash.

The assertion that would have caught this is commented out at line 858:

```python
# TODO: Remove this once include_gs is fully validated
# assert not include_gs, "TODO: Gaussian background still needs to be validated!"
```

There is a second, quieter effect: both stages set `"floor_plane_visible": not include_gs`
and `"use_skybox": not include_gs`. So even on stage 13's graceful path, the default
produces a scene with an invisible floor and no skybox.

`include_gs` is documented in **no** README, INSTALL, or pipeline doc.

**Suggested fix:** default `include_gs: false`, or derive it from whether `--bg-splat` ran.
Add the same `os.path.exists` guard to B stage 6 that stage 13 already has.

**Workaround:** pass `s13_og.include_gs=false` to **both** A and B.

### H2 — `gemini-3-pro-image-preview` is not a registered model

| | |
|---|---|
| **Where** | [`simfoundry/models/vlm.py:560`](simfoundry/models/vlm.py#L560) + 6 config files |

`Gemini.VERSIONS` registers only the GA name `gemini-3-pro-image`
([`vlm.py:547`](simfoundry/models/vlm.py#L547)), and `assert_valid_key` rejects anything
else. Still referencing the non-existent `-preview` name:

- `simfoundry/models/vlm.py:560` — the constructor default
- `scripts/cfg/real2sim_cfg.yaml:617` — `prompt_cousin_structured.image_model`
- `scripts/cfg/auto_bg.yaml:71` — `gemini_model`
- `scripts/cfg/stack_dishware.yaml:105`, `stack_dishware_easy.yaml:105`, `serve_banana.yaml:105`

`real2sim_cfg.yaml:169` and `:180` were fixed to the GA name; the cousin-generation and
auto-background paths were missed, so B augmentation still breaks.

**Workaround:** `prompt_cousin_structured.image_model=gemini-3-pro-image`

### H3 — `--detect-articulation` does not degrade gracefully, contrary to docs

| | |
|---|---|
| **Where** | [`simfoundry/pipeline/orchestrator.py:55`](simfoundry/pipeline/orchestrator.py#L55) |

Both [README.md:116-118](README.md#L116-L118) and [docs/INSTALL.md:227-228](docs/INSTALL.md#L227-L228)
state that if the articulation environments are not installed, the flag "is ignored with a
warning and the rest of the reconstruction pipeline runs normally."

The actual check is:

```python
def articulation_available() -> bool:
    """Whether the optional articulation stage (8b) is present in this checkout."""
    return (REPO_ROOT / ARTICULATION_STAGE_SCRIPT).is_file()
```

That tests only whether the *stage script* exists — which it always does, since it ships in
the repo. It never inspects the `articulate-anything-{hunyuan,partfield}` conda envs that
stage 8b actually needs. So the promised graceful degradation never triggers; stage 8b is
scheduled and fails at runtime.

**Observed directly** on the 25-object `ClutteredScene`, with the articulation envs absent
and `deps/articulate-anything` not cloned. `--detect-articulation` was accepted without
warning, stage 8b ran, made a **live Gemini call** that correctly classified `bamboo organizer`
as articulated, then died:

```
[8b_articulate_objects.py:305] VLM Classification:
  Articulated: ['bamboo organizer']
  Non-articulated: ['black eraser', 'hamburger toy', 'red marker', ...]
[8b_articulate_objects.py:385] Processing 1 objects: ['bamboo_organizer']
FileNotFoundError: Template not found:
  /home/gear/SimFoundry/deps/articulate-anything/simfoundry/cfg/hunyuan_template.yaml
```

`subprocess.CalledProcessError` then propagated and took the whole pipeline down with a
non-zero exit. Three things make this worse than the docs imply:

- It **burns paid VLM quota** before discovering a dependency that could have been checked
  in microseconds at plan time.
- In a full run it aborts at stage 8b, **discarding ~1 hour** of completed stages 1-8.
- The error names a missing YAML template, giving no hint that the real problem is an
  uninstalled optional component.

**Suggested fix:** have `articulation_available()` also check that `deps/articulate-anything`
exists and that the env in `CONDA_ENVS` (`8b_articulate_objects.py:37`) resolves. That would
make the documented "ignored with a warning" behaviour real.

**Suggested fix:** also check that the env in `CONDA_ENVS` (`8b_articulate_objects.py:37`)
resolves, and that `deps/articulate-anything/` exists.

### H4 — The shipped test suite fails 7/149 on a clean install

`docs/INSTALL.md` §5 presents `pytest` as the way to verify an install. On a freshly built,
known-good `simfoundry` env it reports **7 failed, 142 passed, 23 skipped**. A beta tester
cannot distinguish "my install is broken" from "these tests are stale" — and all 7 are stale.

**Group 1 — three tests contradict the `--bg-splat` change made in the same commit.**

| Test | Error |
|---|---|
| `test_pipeline_orchestrator.py::test_named_stage_plans_use_new_subdirectories` | `StopIteration` |
| `test_pipeline_orchestrator.py::test_stage_2c_uses_dedicated_nerfstudio_environment` | `StopIteration` |
| `test_run_pipeline_cli_e2e.py::test_reconstruction_cli_routes_stage_2c_to_nerfstudio_env` | `AssertionError: assert 'mamba run -n custom-nerfstudio python' in '...'` |

All three assume stage 2c is unconditionally in the plan:

```python
stage_2c = next(
    spec for spec in get_stage_plan("video", pipeline_name="reconstruction")
    if spec.stage_id == "2c"
)
```

But `b71bdb6` made 2c conditional on `bg_splat`, which defaults to `False`. Verified
directly against the installed package:

```
bg_splat=False: 2c present=False
bg_splat=True:  2c present=True  env=nerfstudio
```

The routing feature works correctly; the tests simply never pass `bg_splat=True` (and the
CLI test never passes `--bg-splat`). **Fix:** add `bg_splat=True` to the three call sites.

**Group 2 — four tests reference functions that do not exist in the stage scripts.**

| Test | Missing attribute |
|---|---|
| `test_stage10_sim_ready.py::test_discover_sim_ready_jobs_filters_requested_indices` | `discover_sim_ready_jobs` |
| `test_stage10_sim_ready.py::test_convert_objects_parallel_invokes_each_conversion_once` | `ConversionTask` |
| `test_stage12_usd_import.py::test_import_usd_asset_uses_configured_dataset_and_fails_fast` | `import_usd_asset` |
| `test_stage12_usd_import.py::test_import_usd_asset_raises_when_importer_creates_no_usd` | `import_usd_asset` |

`10_make_objects_sim_ready.py` defines none of `discover_sim_ready_jobs`,
`convert_objects_parallel`, or `ConversionTask`. `12_import_usd.py` defines only
`resolve_reparent_script` and `main` — no `import_usd_asset`. The tests appear to target a
refactor of these two stages that is not in this release.

**Suggested fix:** update or skip all 7 before release, so `pytest` is a usable install check.
If the stage-10/12 refactor is intentionally unreleased, mark those four `xfail`/`skip` with
a reason.

### H5 — `--scene-name` without `--video-fpath` silently points at another scene's video

| | |
|---|---|
| **Where** | [`A_reconstruction/run.sh:55-57`](scripts/pipeline/A_reconstruction/run.sh#L55-L57) |
| **Severity** | High — silent, and can overwrite one scene's data with another's |

`VIDEO_FPATH` is derived from `SCENE_NAME` **before** the argument-parsing loop runs:

```bash
55  SCENE_NAME="${SCENE_NAME:-home_coffee_4}"
57  VIDEO_FPATH="${VIDEO_FPATH:-${ROOT_DIR}/${SCENE_NAME}/s1_video/video/scene.mp4}"
...
91        SCENE_NAME="$2"      # --scene-name is parsed 34 lines too late
```

So `--scene-name` never affects the derived default. Reproduced directly:

```
$ bash scripts/pipeline/A_reconstruction/run.sh --scene-name ClutteredScene --dry-run --include 1b
[Stage 1b] cmd: ... scene_name=ClutteredScene \
    s1_video.video_fpath=/home/gear/SimFoundry/Data/home_coffee_4/s1_video/video/scene.mp4
```

The scene name is right and the video path is wrong — hardcoded to whatever
`SCENE_NAME` defaulted to (`home_coffee_4`).

**Why this matters:** re-running early stages on an existing scene is a normal operation
(`--include 1b,2` after tweaking a config). Doing so without re-passing `--video-fpath` will
process **the wrong scene's video into this scene's directory**, silently overwriting its
frames. If the default scene does not exist, it fails with a confusing missing-file error
instead.

It went unnoticed here because every full run passes `--video-fpath` explicitly. It surfaced
when re-running a single later stage, where the flag is not obviously needed.

**Suggested fix:** move the `VIDEO_FPATH` derivation to *after* the parsing loop. B and C do
not share this pattern — A is the only affected wrapper.

### H6 — Articulation ships `point_num` 5x upstream's default and cannot fit on a 24 GiB GPU

| | |
|---|---|
| **Where** | `deps/articulate-anything/simfoundry/cfg/hunyuan_template.yaml:72` (`s3_segment_mesh.point_num: 500000`) |
| **Severity** | High — stage 8b is unusable on 24 GiB at shipped defaults |

P3-SAM's own default is `point_num=100000`
(`deps/articulate-anything/deps/Hunyuan3D-Part/P3-SAM/demo/auto_mask.py:767`). SimFoundry's
template ships **500000**. In `auto_mask.py:163-169` the features are expanded to
`[point_num, prompt_bs, 512]` float32, so peak memory scales linearly with `point_num`:

| `point_num` | `feats[N, 32, 512]` fp32 | Fits in 23.5 GiB? |
|---|---|---|
| **500000 (shipped)** | **30.5 GiB** | **No — exceeds the whole card** |
| 200000 | 12.2 GiB | No (OOM'd at 22.95 GiB in use) |
| 100000 (upstream default) | 6.1 GiB | Yes |

A single intermediate tensor at the shipped value needs more memory than the entire GPU, so
this is not a tuning problem — **it cannot succeed on any 24 GiB card**.

Observed on `ClutteredScene` (bamboo organizer), GPU otherwise idle at 454 MiB:

```
point_num=500000 -> OOM: Tried to allocate 7.63 GiB, 5.05 GiB free, 18.02 GiB in use
point_num=200000 -> OOM: Tried to allocate 1.53 GiB, 115 MiB free, 22.95 GiB in use
```

Compounding problems:

- **No VRAM knob is exposed.** `s8b_articulate_objects` has no `low_vram` or `point_num`
  equivalent (compare `s7_mesh.low_vram`). The only tuning point is a YAML inside a *cloned
  dependency*, which is not obvious and is lost on a fresh clone.
- **The child process masks the OOM as success.** `run_articulation` reports
  `Articulation subprocess exited successfully but did not create expected URDF(s)`. Only
  SimFoundry's own post-check turns this into a visible failure — that guard is doing real
  work and should be kept.
- Combined with **L6** (stage 7 needing `low_vram=true`), this is the second place where
  `docs/INSTALL.md`'s "24 GiB works for the standard video pipeline" does not hold.

**Suggested fix:** set the template to upstream's `100000`, and expose
`s8b_articulate_objects.point_num` (or a `low_vram` flag) so it is tunable from
`real2sim_cfg.yaml` without editing a dependency.

### H7 — Articulation renders via pyrender, which cannot run headless

| | |
|---|---|
| **Where** | `deps/articulate-anything/simfoundry/cfg/hunyuan_template.yaml:41,58` (`renderer: pyrender`) |

With `DISPLAY` unset — i.e. any SSH session, cluster node, or container — pyrender's default
backend tries to open an X display and fails:

```
Pyrender offscreen failed: Cannot connect to "None"
Error processing bamboo_organizer: 'NoneType' object has no attribute 'save'
```

The helper returns `None` instead of raising, so the real cause surfaces several frames later
as an unrelated `AttributeError`. Nothing in the message mentions displays, EGL, or headless
rendering.

**Fix that worked:** `export PYOPENGL_PLATFORM=egl`. NVIDIA's EGL driver
(`libEGL_nvidia.so.0`, `10_nvidia.json`) was already installed, so nothing else was needed —
rendering then succeeded (`PyRender: Rendering 9 parts...`).

**Suggested fix:** export `PYOPENGL_PLATFORM=egl` from `8b_articulate_objects.py` (or the
installer) when `DISPLAY` is unset, and let the render helper raise instead of returning
`None`. Given that remote/headless is the normal way to run this, the current default fails
for most users.

### H8 — Articulation steps s2 and s4 can disagree on part cardinality, with no reconciliation

| | |
|---|---|
| **Where** | articulate-anything s2 (`generate_articulation_tree`) vs s4 (`merge_mesh_parts`) |

On the `bamboo organizer`, step 2 proposed a two-part tree:

```json
{"parts": [{"part_name": "drawer"}, {"part_name": "bamboo_organizer_base"}],
 "joints": [{"joint_name": "bamboo_organizer_base_drawer_joint", "joint_type": "prismatic"}]}
```

Step 4, looking at per-segment renders, found the object genuinely has **six** drawers and
returned `drawer_1 ... drawer_6`. Its own analysis is specific and plausible: *"drawers 6, 5
(with handle 8), and 7 on the left... 3 (with handle 0) and 4 on the right"*. The run then
aborts:

```
part merge returned unknown part name(s) ['drawer_1'...'drawer_6'];
expected a subset of ['drawer', 'bamboo_organizer_base'].
Refusing to guess: an unmatched name silently loses its mesh faces downstream.
```

s4 is arguably **more correct than s2** here — the object really does have six drawers. The
two steps view the object differently (whole-object renders vs per-segment renders) and there
is no reconciliation path: no re-prompt of s2 with s4's segment count, no mapping of
`drawer_N` onto a repeated child link.

**Confirmed deterministic across independent re-rolls** (s2 and s4 outputs deleted between
attempts, so both VLM calls were genuinely re-issued):

| Attempt | s2 parts | s4 returned | Result |
|---|---|---|---|
| 1 | `drawer`, `bamboo_organizer_base` | `drawer_1..drawer_6` | abort |
| 2 | `drawer`, `bamboo_organizer_base` | `drawer_1..drawer_3` | abort |

s2 is **stable** at exactly one `drawer` every time; s4 is stochastic in *count* (6 then 3)
but always returns **numbered plural** names. The two can therefore never agree on this
object. This is not a bad roll that retrying fixes — any multi-compartment object appears to
fail permanently, and no amount of re-running will produce a URDF.

The guard itself is good and should be kept — silently dropping unmatched names would corrupt
the mesh. The gap is that there is no recovery, only abort.

**Suggested fix:** feed the s3 segment count into s2's prompt, or allow s4 to expand a single
tree part into N sibling links sharing the parent joint type.

### H9 — Failed articulation steps leave cached state that silently poisons every retry

`hunyuan_template.yaml` sets `rerun: false` on s1-s4 (only `s5_articulate` is `true`), and the
cache key is *"does the output directory exist"*, not *"did the step succeed"*. Observed twice:

- After the pyrender failure, `s3_segment_mesh/rendered_parts/` contained `exploded_mesh.glb`
  and `face_ids.npy` but **no images**. The next run skipped re-rendering and failed further
  downstream with a different, misleading error (`No renders found`).
- After the s4 name mismatch, `merge_result.json` was cached. A retry replayed the identical
  VLM response verbatim, so re-running could never produce a different outcome despite the
  failure being a one-shot VLM disagreement.

Recovery requires knowing to delete specific step directories by hand — nothing in the error
messages suggests it.

**This is the same structural pattern as M1 and M4**: resume logic keyed on output existence
rather than success. Across the installer (`install_everything.sh` skipping half-built envs),
`3dgrut`, and articulation, it has now caused four separate misleading failures in this test.
Worth treating as one cross-cutting design issue rather than three unrelated bugs.

---

## Medium severity

### M1 — Re-running the installer silently skips half-built environments

[`install_everything.sh:104-108`](scripts/installation/install_everything.sh#L104-L108) treats
any env that exists as complete:

```bash
elif mamba run -n "${envn}" true 2>/dev/null; then
  echo "    env '${envn}' already exists — skipping (use --fresh to rebuild)."
```

Combined with `set -euo pipefail`, the first failure aborts every remaining env — but leaves
the failed env *created*. The natural next step (re-run the command) then skips the broken
env and proceeds to the next one, producing a superficially successful install with one
silently broken environment. This is exactly what happens after B1.

`--fresh` exists but rebuilds *everything*, discarding hours of correct work.

**Suggested fix:** record a per-env completion sentinel (e.g. `conda-meta/simfoundry.done`)
and treat "exists but no sentinel" as incomplete, or support `--fresh <env>` for one env.

### M2 — `huggingface-cli login` no longer works

[README.md:57](README.md#L57) and [docs/INSTALL.md:112](docs/INSTALL.md#L112) both instruct
`huggingface-cli login`. That command now hard-errors:

```
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
```

`login_services.sh:131` already uses the correct `hf auth login`. Only the docs are stale.
Correct command: `hf auth login`.

### M3 — Checkpoint download depends on `gdown` against Google Drive folders

[`download_checkpoints.sh:178/186/193`](scripts/installation/download_checkpoints.sh#L178)
fetches FoundationStereo and FoundationPose weights via `gdown --folder`. Google Drive
folder downloads are rate-limited and fail intermittently with unhelpful errors. The script
already anticipates this with `--checkpoint-fallback-root`, but the primary path remains
fragile and is the step most likely to fail for an external beta tester.

**This run it worked** — all three Drive downloads succeeded and `download_checkpoints.sh`
exited 0 with zero errors. Logged as a durability risk, not an observed failure.

**Suggested fix:** mirror these weights to Hugging Face or an NGC bucket.

### M4 — `install_3dgrut.sh` fails on a `nounset`-unsafe conda hook

| | |
|---|---|
| **Where** | [`install_3dgrut.sh:133`](scripts/installation/install_3dgrut.sh#L133) → upstream `deps/3dgrut/scripts/create_conda.sh` |
| **Symptom** | `.../envs/3dgrut/etc/conda/deactivate.d/deactivate-gxx_linux-64.sh: line 68: CONDA_BACKUP_CXX: unbound variable` |

The wrapper anticipates exactly this class of failure — lines 116-121 wrap its *own*
`conda deactivate` calls in `set +u`, commenting that "conda/cuda-nvcc (de)activate hooks
aren't `set -u` safe". But it then shells out to upstream `create_conda.sh`, whose
`conda install` calls (lines 117/119/126) trigger a conda **reactivation** that sources
`deactivate.d/*.sh` hooks dereferencing `CONDA_BACKUP_*` with no default.

The env is created before the failure, so this also leaves a **half-built env** that
`install_everything.sh` silently skips on re-run (M1 — observed twice during this test).
`threedgrut` is never installed, because Step 3 (`install_env_uv.sh`) never runs.

`patches/3dgrut.patch` does not help — it only sets `export_cameras=False` in `ply_to_usd.py`.

**What did *not* work:** removing `-u` from `create_conda.sh`'s own `set -euo pipefail`. The
nounset state comes from conda's activation machinery, not that script's shell options.

**What did work** — make the env's conda hooks nounset-safe, then re-run:

```bash
cd ~/miniforge3/envs/3dgrut/etc/conda
for f in activate.d/*.sh deactivate.d/*.sh; do
  cp -n "$f" "$f.orig"; printf 'set +u\n' | cat - "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
bash scripts/installation/install_3dgrut.sh --project-root "$PWD" --env-name 3dgrut --default
```

`create_conda.sh` skips creation when the env exists, so the re-run resumes cleanly and
completes (verified: `import threedgrut` OK, torch 2.8.0+cu128, CUDA available).

**Suggested fix:** sanitize the hooks right after `conda create` (or export `CONDA_BACKUP_*`
defaults) before any step that can trigger a conda reactivation.

Medium rather than blocker only because `3dgrut` is optional — PLY → USDZ for auto-backgrounds.

### M5 — The installer pins a `flash-attn` version that breaks the FLUX backend

| | |
|---|---|
| **Where** | [`install_simfoundry.sh:230`](scripts/installation/install_simfoundry.sh#L230) vs [`requirements.txt:21`](requirements.txt#L21) |
| **Symptom** | Printed at the start of *every* VLM stage: `Error importing diffusers ... Requires Flash-Attention version >=2.7.1,<=2.7.4 but got 2.8.3.` |

`install_simfoundry.sh:230` does `pip install flash-attn==2.8.3`, with `flash-attn==2.7.3`
commented out directly above at line 229. But `diffusers` (0.39.0, from `diffusers>=0.35.0`)
requires flash-attn `<=2.7.4` for its `loaders.ip_adapter` path, which
`pipelines.flux.pipeline_flux_kontext` depends on. The two pins are mutually incompatible.

Consequence: `import diffusers` succeeds, but `FluxKontextPipeline` does not, so
[`vlm.py`](simfoundry/models/vlm.py) sets `FluxKontextPipeline = None` and the **FLUX backend
is non-functional in the environment the installer produces** — independent of, and in
addition to, the HF gating on `black-forest-labs/FLUX.1-Kontext-dev`. `model: flux` is offered
as an option in `real2sim_cfg.yaml:169/180` and in README step 2.

The failure is silent by design: `vlm.py` catches `(ImportError, RuntimeError)` and continues.
The only visible symptom is a traceback-shaped message printed on every stage that imports
`vlm`, which reads like an error but is not fatal — noise that will send beta testers hunting.

**Not a blocker**: the default Gemini path is unaffected, and this run of A completed VLM
stages 3 and 5 normally.

**Suggested fix:** pin `flash-attn<=2.7.4`, or cap `diffusers` to a release whose
`ip_adapter` loader accepts 2.8.x, and confirm which consumer required the move to 2.8.3
(line 229 suggests 2.7.3 was tried first). Either way the two pins should be reconciled
rather than left to fail at import.

### M6 — `B_augmentation/run.sh` hardcodes a demo-scale preset that overrides the config

| | |
|---|---|
| **Where** | [`B_augmentation/run.sh:141-154`](scripts/pipeline/B_augmentation/run.sh#L141-L154) |

The wrapper unconditionally appends 13 Hydra overrides before the user's `--` overrides:

```bash
"prompt_cousin_structured.min_keep_per_dim=1"     # config says 2
"prompt_cousin_structured.max_objects=2"          # config says null (= all)
"prompt_cousin_structured.max_components=2"       # config says null
"prompt_cousin_structured.max_generated_images_per_object=2"   # config says null
"prompt_cousin_structured.text_model=gemini-2.5-flash"         # config says gemini-2.5-pro
"generate_cousins_combination.max_combinations=2"
"s13_og.auto_iter_num=2"
"propose_scene_task.num_tasks=2"
```

So the `prompt_cousin_structured` block in `real2sim_cfg.yaml` is **effectively dead** — editing
it changes nothing, because the wrapper's values always win. Users must pass `--` overrides
instead (those are appended last and do take precedence).

Observed effect, across two scenes:

| Scene | Objects reconstructed | Objects given cousins | Coverage |
|---|---|---|---|
| `PutMarkerInCup` | 3 | 2 | 67% |
| `ClutteredScene` | **25** | **2** | **8%** |

The cap is absolute, not proportional, so it degrades badly as scenes get richer — exactly the
case augmentation is most useful for. On the 25-object scene B finished in 14m and produced
cousins for a baseball and a tape dispenser only; the other 23 objects were silently skipped,
with nothing in the output indicating that a limit was applied.

This looks like a demo/smoke preset that was left switched on. **Suggested fix:** move these
behind an explicit `--quick` / `--demo` flag, or delete them and let the config defaults apply.

### M7 — `usd_cousins/` is always empty, though docs list it as a key output

| | |
|---|---|
| **Where** | [`5_import_cousin_usd.py:74-85`](scripts/pipeline/B_augmentation/stages/5_import_cousin_usd.py#L74) |

Stage 5 creates `usd_cousins/` and then never writes to it. It shells out to
`omnigibson.examples.objects.import_custom_object --dataset-name custom-assets`, so the USDs
land in `deps/BEHAVIOR-1K/datasets/custom-assets/objects/...`. `out_dir` is used only as one
of the paths in an already-exists check.

Both docs advertise the empty directory as an output:
- `README.md:159` — "`sim_cousins/` and `usd_cousins/`: simulation-ready cousin assets"
- `scripts/pipeline/README.md:134` — outputs "`usd_cousins/`, custom asset dataset entries"

The stage is otherwise fine: all 3 cousin USDs imported correctly on this run. But it emits
**zero stdout** (its Hydra log is 0 bytes) and takes ~50s, so an empty output directory plus a
silent success reads exactly like a no-op failure. That cost real debugging time here.

Related dead config: `usd.dataset_name: real2sim-assets`
([real2sim_cfg.yaml:647](scripts/cfg/real2sim_cfg.yaml#L647)) is never read — the script
hardcodes `"custom-assets"`.

**Suggested fix:** log each import, and either write to `usd_cousins/` or correct both docs to
point at the `custom-assets` dataset.

### M8 — `install_articulate.sh` requires sudo unconditionally, even when nothing needs installing

| | |
|---|---|
| **Where** | [`install_articulate.sh:45`](scripts/installation/install_articulate.sh#L45) and [`:55`](scripts/installation/install_articulate.sh#L55) |

The script calls `sudo -v` and then `sudo apt-get install -y ffmpeg libxrender1 libxi6
libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libgl1` with no guard. On this machine all 8
packages were already installed, Blender was already at the exact required 4.2.3, and git-lfs
was present — so the entire system-dependency block was a no-op, yet it still blocked on a
password prompt.

Consequences:
- **Cannot be run by an agent, in CI, or in a container without root**, which is squarely at
  odds with shipping an agent-oriented install guide. Every other SimFoundry installer runs
  unprivileged.
- No `--skip-system-deps` flag exists, so there is no supported way around it. The only
  options are an interactive human, passwordless sudo, or editing the script.

Note the Blender download at lines 66-78 is *already* correctly guarded by a version check —
the same treatment applied to the apt line and `sudo -v` would fix this.

**Suggested fix:** guard the apt call with a `dpkg -s` check (as the Blender block does), move
`sudo -v` inside the branch that actually needs it, and add `--skip-system-deps`.

### M9 — `install_articulate.sh` builds both segmentation backends with no way to choose

[Lines 100-101](scripts/installation/install_articulate.sh#L100) run `installation_hunyuan.sh`
and `installation_partfield.sh` unconditionally. Stage 8b only uses one backend per run
(`CONDA_ENVS` in `8b_articulate_objects.py:37`), but both envs — roughly 20-25 GB combined —
are always built. On a disk-constrained machine that is the difference between fitting and not.

**Suggested fix:** a `--backend hunyuan|partfield|both` flag defaulting to `hunyuan`.

---

## Low severity / documentation

| # | Issue | Where |
|---|---|---|
| L1 | `git submodule update --init --recursive` is a no-op — the repo has no `.gitmodules`, and "Git submodules enabled" is listed as a requirement. All deps are cloned by the install scripts. | [docs/INSTALL.md:11,26](docs/INSTALL.md#L26) |
| L2 | No disk-space requirement is documented. A full install needs **~250 GB** (envs ≈ 100 GB, `deps/` ≈ 82 GB of which `void-model` alone is 41 GB, HF cache ≈ 12 GB). Running out mid-build wastes hours. | docs/INSTALL.md "Requirements" |
| L3 | `--env-b1k` defaults to `b1k` in `A_reconstruction/run.sh`, but no installer ever creates a `b1k` env. Harmless in practice — no A stage uses the `b1k` role, and B/C correctly default to `simfoundry` — but it is confusing dead config. | [A_reconstruction/run.sh:64](scripts/pipeline/A_reconstruction/run.sh#L64) |
| L4 | "A four-file subset needs no runtime dependencies at all, so it works before any environment is built" — true only if `pytest` is importable; it is not present in a stock Miniforge `base`. | [docs/INSTALL.md:187-193](docs/INSTALL.md#L187) |
| L5 | `install_3dgrut.sh`'s header says it supersedes a version referencing `patches/3dgrut.patch` "that no longer exists" — but that patch does exist and is still applied. Stale comment. | [install_3dgrut.sh:12-14](scripts/installation/install_3dgrut.sh#L12) |
| L6 | `s7_mesh.low_vram` defaults to `false`, which the config itself says needs **~29 GB** for shape generation — but `docs/INSTALL.md` states "24 GiB works for the standard video pipeline". A 24 GiB user must pass `s7_mesh.low_vram=true` or hit OOM at stage 7. Either flip the default or qualify the requirement. | [real2sim_cfg.yaml:166](scripts/cfg/real2sim_cfg.yaml#L166), [docs/INSTALL.md:18](docs/INSTALL.md#L18) |
| L7 | Isaac Sim logs `AttributeError: 'NoneType' object has no attribute 'GetCamera'` from `omni.kit.widget.viewport` on every headless shutdown — many times during stages 10-13. Harmless, but traceback-shaped and easily mistaken for a real failure. Worth suppressing or documenting as expected. | Kit teardown, stages 12/13 |

---

## Environment-specific notes (not repo bugs)

- **Both Gemini routes work on this account.** `login_services.sh` warns that
  `[thtu@nvidia.com] does not have permission to access projects instance [simfoundry-gear]`,
  but that is `gcloud projects describe` (resource-manager listing), which is a *different*
  IAM permission from Vertex AI usage. Confirmed empirically: stage 8b runs with
  `vlm_backend: vertex` and its classification calls succeed. Meanwhile
  `resolve_gemini_auth()` returns `route: api_key` for the main pipeline, because
  `GEMINI_API_KEY` in the repo-root `api_keys.txt` takes precedence when present. So A/B run
  on the API key and articulation runs on Vertex — the warning is noise, not a blocker.
- **`GEMINI_API_KEY` is also exported in the shell**, and `load_api_keys()` only sets keys
  *not already* in `os.environ` — so an exported stale value silently overrides the file.
- **`black-forest-labs/FLUX.1-Kontext-dev` access is denied** for this HF account
  (`andytu101`). It is listed as optional in README step 2 and only matters if a config sets
  `model: flux`; the defaults use Gemini. Request access at
  <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>.
- **`flash-attn` is 2.8.3** in the `simfoundry` env, and something in the stack warns
  `Requires Flash-Attention version >=2.7.1,<=2.7.4 but got 2.8.3`. It is a warning, not an
  error — imports and a live Gemini call both succeeded — but it may matter for local
  attention-based models (SAM3 / DINOv3) and is worth a deliberate pin.
- All other gated models are accessible: `facebook/sam3`,
  `facebook/dinov3-vitl16-pretrain-lvd1689m`, `briaai/RMBG-2.0`, `netflix/void-model`,
  `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`.
- The `pixal3d` env that existed on this machine has **no references anywhere in the repo**
  and is not built by any installer — it was installed out-of-band. Commit `e26c7ad` reset
  `s7_mesh.shape_model`/`texture_model` from `trellis2` to `hunyuan`, so the default mesh
  backend is now `hunyuan`.

---

## Fixed since the last run on this machine

Worth noting, because these previously required manual workarounds:

- **Stage 2c is no longer unconditional.** It was previously in the stage plan with no
  config gate and required `--exclude 2c`. It is now opt-in via `--bg-splat` and correctly
  declared to run in `nerfstudio_simfoundry` rather than `simfoundry`
  ([orchestrator.py:136](simfoundry/pipeline/orchestrator.py#L136)).
- **`FluxPipeline` → `FluxKontextPipeline`**, with the import guard widened to catch
  `ImportError` as well as `RuntimeError`, and a clear error when the pipeline class is
  unavailable ([vlm.py](simfoundry/models/vlm.py)).
- **Canonical frame selection** (`s3_ground.img_idx: auto`) is new, documented, and the
  config matches the docs.

---

## Reproduction

```bash
# Blocker B1 — from a clean checkout with no deps/
bash scripts/installation/install_everything.sh
# ~40 min later: ImportError: cannot import name 'DepthEncoderConfig'
git -C deps/BEHAVIOR-1K rev-parse HEAD    # != BEHAVIOR1K_COMMIT

# H1
bash scripts/pipeline/A_reconstruction/run.sh --scene-name X --video-fpath V
bash scripts/pipeline/B_augmentation/run.sh --scene-name X
# B stage 6: FileNotFoundError on gaussian_da3.usdz, exit 139
```

## Working invocation on this machine

```bash
export GCLOUD_PROJECT=simfoundry-gear

SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 bash scripts/installation/install_everything.sh

bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name NAME --video-fpath /path/to/video.mov \
  -- s13_og.include_gs=false s7_mesh.low_vram=true

bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name NAME \
  -- s13_og.include_gs=false \
     prompt_cousin_structured.image_model=gemini-3-pro-image
```

### Verified end-to-end

**All three pipelines were run to completion on two scenes**, every stage exiting 0:

| | `PutMarkerInCup` (3 objects) | `ClutteredScene` (25 objects) |
|---|---|---|
| **A** reconstruction | 19m 32s | **1h 29m 19s** |
| **B** augmentation | 12m 15s | 14m 00s |
| **C** smoke-random | 1m 08s | 1m 05s |
| **Total** | **~33 min** | **~1h 45m** |
| Disk used | 1.2 GB | 1.5 GB |

Both produced correct-looking scenes. `ClutteredScene` reconstructed 25 objects — toolbox,
thermos, mug, bottles, bowl with fruit, travel pillow, tennis ball, phone, pen, and more —
with clean geometry and textures, and proposed `put_tennis_ball_in_bowl` and
`place_blue_marker_on_organizer`.

**Scaling behaviour (RTX 4090, 24 GiB):**

| Stage | 3 objects | 25 objects | Ratio |
|---|---|---|---|
| 5-8 streamed (incl. mesh gen) | 12m 35s | 59m 03s | 4.7x |
| 10 make sim-ready | 1m 44s | 19m 56s | **11.5x** |
| 12 import USD | 2m 35s | 7m 25s | 2.9x |
| 13 create OG scene | 1m 18s | 1m 15s | 1.0x |

8.3x more objects produced only 4.6x more wall time overall, so per-object cost falls with
batching. The exception is **stage 10, which scaled worse than linearly (11.5x)** and became
the second-largest cost after mesh generation — the most likely target if reconstruction
throughput needs improving.

B is flat across both scenes (12m vs 14m) *only* because of the **M6** cap; it is not doing
proportionally more work on the richer scene.

A's `s13_og/reconstructed_scene.png` shows the marker, cup, and scissors correctly placed on
the support plane with the Franka arm posed alongside. B produced 2 cup cousins and 1 scissors
cousin (the marker was skipped — see **M6**), imported all 3 USDs into the `custom-assets`
dataset, and proposed `put_marker_in_cup` and `put_scissors_in_cup` tasks. C loaded the
OmniGibson scene, ran random actions, and recorded a valid 3-camera h264 video
(3280x720, 21 frames, 2.1s).

Two observations from C, neither a failure:

- The rendered viewpoints are dominated by skybox — the support plane is visible only
  edge-on. The scene is correct, but the default `s13_og.auto_camera_pos`/`auto_camera_ori`
  make the smoke video hard to use for visual verification.
- The smoke clip is only 2.1s / 21 frames, which is enough to prove the scene loads and steps
  but not much else.

| Stage | Time | | Stage | Time |
|---|---|---|---|---|
| 1b process video | 8.70s | | 9 compile | 5.05s |
| 2 depth (`da3`) | 18.95s | | 10 sim-ready | 1m 44.36s |
| 3 ground plane | 35.51s | | 11 stabilize physics | 5.14s |
| 4 world frame | 5.14s | | 12 import USD | 2m 35.19s |
| 5-8 streamed | 12m 35.28s | | 13 create OG scene | 1m 18.20s |

Within the streamed block: stage 5 = 2m 4s, stage 6 = 2m 40s, **stage 7 (mesh) = 9m 4s**,
stage 8 = 1m 23s.

Two corrections to earlier guidance, established by this run:

- **`--no-stream` is not needed with the `hunyuan` backend.** `s7_mesh.low_vram=true` alone
  kept stage 7 inside a 24 GiB budget across 3 streamed calls. The earlier "`--no-stream` +
  `low_vram`" advice was specific to the previous mesh backend.
- **`s7_mesh.low_vram=true` is effectively mandatory on 24 GiB.** The default `false` needs
  ~29 GB for shape generation, which contradicts `docs/INSTALL.md`'s claim that "24 GiB works
  for the standard video pipeline". Either change the default or qualify the doc (**L6**).

See [agent_install.md](agent_install.md) for the full install procedure.
