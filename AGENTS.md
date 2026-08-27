<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SimFoundry — agent guide

Orientation for coding agents. This covers what the code does not say out loud: which
environment to run in, which trees are safe to edit, and which contracts break silently.
For what the project *is*, read [README.md](README.md).

## What this repo does

SimFoundry turns a short real-world video into a physics-ready OmniGibson scene: segment
the objects, reconstruct geometry, generate textured meshes, compile a scene with physical
parameters and task proposals. The pipeline runs in three stages under
[`scripts/pipeline/`](scripts/pipeline/):

| Stage | Directory | Produces |
|---|---|---|
| A — reconstruction | `A_reconstruction/stages/` | Depth, segmentation, meshes, `s11_sim/scene_objects_info.json`, `s12_physics/pb_scene_poses.json`, an OmniGibson scene |
| B — augmentation | `B_augmentation/stages/` | Digital-cousin variants, scene sampling, task proposals |
| C — application | `C_application/stages/` | Teleop, demo generation, policy evaluation |

Scene editing sits between B and C: a human corrects what the pipeline got wrong — object
poses, scales, cameras — before the scene is used for learning or evaluation. Everything
downstream inherits those corrections.

## Environments

There are many conda envs on a dev box (`3dgrut`, `da3`, `hunyuan`, …); most belong to a
single pipeline stage. For general work:

- **`simfoundry`** — the main env. OmniGibson, Isaac Sim, torch. Used by the pipeline, the
  OmniGibson editor, and `settle.py`. `find_settle_python` in `light_editor/server.py`
  checks that the env's `omnigibson` resolves *inside this repo*.
- **`simfoundry-editor`** — the light editor's env, no GPU. Created by
  `scripts/installation/install_light_editor.sh`; re-run that after a pull. See the trap below.

> **Trap: `omnigibson` can resolve outside this checkout.** An env may carry this repo's
> `simfoundry` package but a *different* clone's OmniGibson. The editor then runs against
> the wrong `deps/` tree and stamps the wrong OmniGibson version into saved scenes.
> `run_editor.sh` warns about this at startup; do not ignore it. Check with
> `python -c 'import omnigibson; print(omnigibson.__file__)'`.

> **Trap: the light editor needs its own venv.** It depends on standalone `usd-core`, while
> Isaac Sim ships its own `pxr` that is only importable once Kit is running. Installing
> `usd-core` into `simfoundry` risks shadowing that copy. Never merge
> `light_editor/requirements.txt` into the conda envs.

## Do not edit `deps/`

`deps/BEHAVIOR-1K/` holds OmniGibson and is **not tracked by this repo** — `git ls-files deps/`
is empty. An edit there exists on one machine and is erased by the next `install.sh` run.

Anything requiring an OmniGibson change must become an upstream PR or a patch file in
[`patches/`](patches/) (the mechanism exists for 3dgrut, Hunyuan, FoundationPose). Keep work
in `simfoundry/` and `scripts/`.

The dependency is pinned by `scripts/installation/install_simfoundry.sh`, not by this repo's
git. A `git pull` inside `deps/BEHAVIOR-1K` can break tooling with no SimFoundry commit —
suspect the pin first when something breaks for one person and not another. The editors reach
into OmniGibson internals through its `lazy` passthrough, which is not a stable public API.

## The saved-scene contract

A scene state JSON (`<scene>_scene_state_<tag>.json`) is the interchange format between the
editors and the rest of the pipeline. Top-level keys: `versions`, `metadata`, `state`,
`init_info`, `objects_info`, `ground_plane_info`, plus optional `viewer_camera_state`,
`lighting_state`, `mesh_background_state`.

The editable quantities live in predictable places:

```
geometry   objects_info.init_info[name].args.usd_path            (relative to the JSON)
scale      objects_info.init_info[name].args.scale
pose       state.registry.object_registry[name].root_link.pos / .ori
friction   objects_info.init_info[name].args.link_physics_materials
                                       {<link>: {static_friction, dynamic_friction}}
mass       objects_info.init_info[name].args.mass                (kg)
joints     state.registry.object_registry[name].joint_pos         (the DOF array)
limits     objects_info.init_info[name].args.joint_limits
                                       {<joint>: {lower, upper}}
```

`friction` and `mass` are written by the light editor's physics panel and are
**not symmetric**. `link_physics_materials` is a real OmniGibson constructor kwarg
(`object_base.py`) and is applied to the link's collision meshes at load, so a
friction written here works everywhere with no consumer change. `mass` is
recorded only: OmniGibson has no load-time mass kwarg — `RigidPrim` reads one
from its load_config, but `EntityPrim` builds each link's load_config from a
fixed whitelist that mass is not on, and `BaseObject.mass` raises on assignment.
PhysX reads `physics:mass` from the USD, which rule 1 below forbids editing (and
which is shared across scenes anyway). **A consumer that wants to honour a
scene's mass applies it after load with `obj.root_link.mass = args["mass"]`,
next to where `set_obj_materials` already applies frictions.** The `<link>` key
is the rigid body's name read out of the asset, never `base_link` by assumption:
reconstructed assets name it after the reconstruction (`link_iter_0_canonical_mesh_obj`),
and OmniGibson raises rather than warns when the name is not there.

`joints` and `limits` are the joints panel, and they are asymmetric for the same reason.
`joint_pos` is an **existing** key that OmniGibson restores on load, so a joint
value written there is real with no consumer change — and, by rule 3's logic,
writing one zeroes that object's `joint_vel`, or the drawer slides the moment
physics starts. `joint_limits` is **recorded only**: PhysX reads a joint's range
out of the USD, which rule 1 forbids editing and which is shared by every scene
using that model, so the range *this* scene wants is a fact about the scene. **A
consumer that wants to honour it applies it after load with
`obj.joints[<joint>].lower_limit = ...`,** beside the mass line above. It is
keyed by joint **name**, never by index: the index order in `joint_pos` is PhysX
metadata that this format does not record — `robot_pose.ROBOT_ARM_JOINTS` exists
because it is not derivable — while the name is what a consumer addresses. The
editor only offers named editing when the asset's degree-of-freedom count
matches the array's length, and shows the raw array otherwise.

**Who may edit what.** The light editor authorizes three nested sets, not one, because the
robot and the scanned room are each half-locked in a *different* half (`iter_objects`):

```
posable    position + orientation      every object
scalable   scale                       props and the room, never the robot
editable   everything else             props only — and this is also the set that
                                       Select All, Arrange, Duplicate and Remove use
```

A robot's scale is locked because resizing the mesh leaves its URDF-derived joint frames,
collision geometry and actuator limits at the authored size. A room's is not: a scan is
plain geometry. The room's *pose* is expected to need correcting — a registered room lands
at the right height but its yaw and origin within the room are arbitrary (`background_io`) —
and moving it moves nothing else, since props hold world poses and are not parented to it.

**Rules:**

1. **Never rewrite a USD asset to move or resize an object.** Edit the JSON. Each object
   carries an `expected_file_hash`; touching the USD invalidates it.
2. **New keys must be additive and optional.** At least six consumers read this format —
   policy evaluation (`C_application/stages/1_eval_policy_og_scene.py`), teleop
   (`2_teleop_og_scene.py`), task proposal (`B_augmentation/stages/7_propose_scene_tasks.py`),
   scene sampling (`6b_sample_asset_scene.py`), background alignment
   (`A_reconstruction/stages/auto_bg_reconstruction/7_build_og_scene_assets.py`), and
   `scripts/interactive/debug_task_creation.py`.
3. **Zero velocities on any object you reposition.** Saved states carry live `lin_vel`/`ang_vel`
   from whenever the sim was paused; replaying those against a hand-placed pose makes objects
   drift the moment physics starts.
4. **Do not overwrite `_latest.json` from an unverified edit.** Write a timestamped file and
   promote only after physics has confirmed the pose.

## Conventions

- **Coordinates:** metres, Z-up scene frame, quaternions in **XYZW** order (identity is
  `[0, 0, 0, 1]`). This matches three.js, so the browser editor needs no conversion.
- **`upAxis` is advisory.** Object USDs are Z-up and scanned mesh backgrounds are Y-up, but
  USD does not rotate on reference and neither does OmniGibson. Do **not** apply a browser-only
  correction — doing so visibly shattered the background once already. Re-check this whenever
  a new background type appears.
- **Backgrounds come in two kinds.** A *mesh* background (`--mesh_background`) is ordinary
  triangles and exports to glTF fine. A *GS* background (`--background`) is 3D Gaussian
  Splatting rendered through NuRec — it has no mesh prims, is loaded `visual_only=True`, and
  needs a skybox or it renders at ~4% brightness. Most active scenes use mesh. The browser
  editor draws both: `light_editor/splat_io.py` reads the gaussians out of the NuRec USDZ
  with no NuRec, no Isaac Sim and no GPU, and `web/splats.js` renders them.
- **A GS background is not collision geometry.** Being `visual_only`, it gives a scene
  *nothing* to rest on — props fall through the desk in the picture the moment physics
  starts. `ground_plane_info` in the saved scene is what fixes that, and
  `PickPlaceTask._apply_ground_plane_from_scene` is what reads it. The browser editor's
  **Ground plane** panel writes it.

  **`og.sim.restore()` does not read it.** Anything that restores a scene rather than building
  an `Environment` — `light_editor/settle.py`, `light_editor/parity_check.py` — has to apply it
  itself, through `simfoundry/utils/ground_plane_utils.apply_ground_plane_info`, which is the
  same call `PickPlaceTask` makes. A gate that models the contract differently from the
  consumer fails exports that would have run, and passes ones that would not on the day the
  offset happens to cancel.

- **`restore()` also resolves relative paths against the process cwd, not against the file.**
  So anything handing it a scene passes a pre-resolved copy — and anything folding the
  simulator's own serialization back into an authored document has to put the relative
  spellings back (`scene_io.merge_settled_scene`), or a portable scene directory becomes one
  that loads on exactly one machine.
- Google-style docstrings; SPDX headers on new files (copy from any existing file).
- Comments explain *why*, not *what*. Several in this tree record hard-won findings — treat
  them as load-bearing.

## The two editors

### Browser editor — `scripts/interactive/light_editor/`

The fast path for placement, and where new authoring work should go. Extracts glTF proxies
with standalone OpenUSD and edits in three.js: click-to-select, transform gizmos, numeric
fields, drop-to-surface, camera placement with look-through. **No Isaac Sim** — a scene loads
in under a second against ~90 s through OmniGibson.

Has its own [README](scripts/interactive/light_editor/README.md); read it before changing
anything here. Files: `extract.py` (USD→glTF), `scene_io.py` (JSON patch/write),
`camera_io.py` (external-sensor configs), `server.py` (HTTP + save), `settle.py` (headless
physics check), `parity_check.py`, `web/app.js` (viewer).

```bash
cd scripts/interactive/light_editor
mamba run -n simfoundry-editor python server.py --scene /abs/path/to/<scene>_scene_state_latest.json
```

**Single writer by design, but no longer silently.** Every save rebuilds from the immutable
startup scene plus one client's **complete** snapshot (`server.py`,
`copy.deepcopy(self.base_scene)`). Complete includes physics and joints, not just transforms:
a field the browser leaves out is not "unchanged", it is "put back to what the file held when
the editor opened".

Three guards, in three scopes, because they answer different questions:

- `scene_revision` is compared, published and incremented **inside** `WRITE_LOCK`, so two
  requests carrying the same revision cannot both pass.
- A client **never adopts a revision without the content it names**. It rehydrates from
  `/api/scene_state` when it has nothing unsaved, and latches read-only otherwise. Taking the
  number alone used to make the *next* save pass the check while still carrying the snapshot
  the page loaded with.
- Every publish states the digest it expects to replace and takes a cross-process lock:
  `scene_io.guarded_write_text`. `WRITE_LOCK` serialises this process and says nothing about a
  second editor server or a text editor, which is the only thing that can reach these files.

The session keeps two documents apart, and conflating them is the bug to avoid re-introducing:
`base_scene` is the **compilation origin** (immutable for a binding), `current_scene` is what
the scene **actually says now** (what a reload is served, and what `scene_revision` names).
Promotion moves the second and not the first.

### OmniGibson editor — `scripts/interactive/interactive_scene_editor.py`

3,500 lines, keyboard-driven, runs inside Isaac Sim. Still the only place to see the true
render (GS backgrounds, materials, lighting) and to run physics. ~90 s to load.

```bash
./scripts/interactive/run_editor.sh          # resume latest saved state
./scripts/interactive/run_editor.sh fresh    # build from pipeline outputs in Data/
```

`import omnigibson` boots the Kit app, so **this process *is* Isaac Sim** and its
`while True: render()` loop is the app's frame loop. There is no separate UI thread — a
blocking `input()` in a keyboard callback freezes rendering. Never add one.

**Keys are declared in one place:** the `BINDINGS` table in
[`scripts/interactive/editor_bindings.py`](scripts/interactive/editor_bindings.py). It drives
the registration loop, the HUD legend (`_hud_text`), and `print_controls()` via
`format_controls()`, and `validate()` fails at startup on a double-bound key or a missing
handler. Add or change a key there, not at the call site.

One hand-maintained copy remains: the **module docstring** of `interactive_scene_editor.py`
still restates the keymap, and still says "the authoritative list is `print_controls()`" —
which is now stale advice. That docstring has already drifted once and shipped wrong
instructions to users. Either regenerate it from `format_controls()` or replace it with a
pointer to `editor_bindings.py`.

`interactive_scene_editor_cousin_swap.py` is a 130 KB near-duplicate fork; check whether a fix
belongs in both.

## Current state of editor work

[docs/INSTRUCTIONS_SCENE_EDITOR.md](docs/INSTRUCTIONS_SCENE_EDITOR.md) is how both editors are
driven end to end — **read it before starting editor work**.

Implemented: on-screen HUD and selection outline; non-blocking dialogs; the browser editor
(which absorbed the mouse-selection and drag-gizmo phases as ordinary web work); camera
placement with look-through; the `editor_bindings.py` keymap table.

The main unbuilt feature is **auto-arranging props around a central object**. The placement
math already exists in `simfoundry/utils/` (see Working style below); the real gap is anchor
*detection* — nothing upstream marks which object a task is about, so start with a manual
"set anchor" and a largest-AABB-nearest-centroid heuristic. It belongs in the browser, where
re-rolling a layout is instant rather than a 90 s reload.

## Testing

```bash
pytest                                                              # repo tests
```

`pytest.ini` sets `--continue-on-collection-errors` because many tests need heavy optional
deps (torch, open3d, faiss) present in only some envs. A collection error still fails the run,
so genuine breakage is not masked.

The light editor ships without its own test suite; validate editor changes with
`settle.py` and `parity_check.py` against a real scene.

The OmniGibson editor has **no tests** — it all lives in `__main__` behind a 90 s boot. The
light editor is plain importable Python and is tested; prefer putting new logic there.

Install the whitespace hook once per clone:

```bash
ln -sf ../../scripts/hooks/pre-commit .git/hooks/pre-commit
```

It runs `git diff --cached --check`, which refuses trailing whitespace, a blank line at end of
file and conflict markers. All three are invisible in review and one of them has already
shipped. A symlink rather than `core.hooksPath`, which would take the whole hooks directory
over and silence the git-lfs hooks this repository depends on. `--no-verify` skips it.

## Working style

- Prefer reusing `simfoundry/utils/` over writing new geometry code. Placement helpers already
  exist: `placement_utils.place_with_predicate`, `separate_overlapping_objects`,
  `distractor_utils.compute_scene_centroid`, `place_distractor`,
  `scene_sampling_utils.compute_randomized_poses`.
- Cite `file:line` when reporting findings.
- Anything that boots Isaac Sim takes ~90 s and needs a GPU and a display — do not put it in a
  hot loop or a test.
- Verify before claiming. If you could not run it, say so.
