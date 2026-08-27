# SimFoundry light scene editor

A browser-based editor for existing OmniGibson scenes. It extracts lightweight
GLB proxies with standalone OpenUSD and lets you move, rotate, scale, add and
remove objects, place cameras, and edit task configs — all without launching
Isaac Sim or OmniGibson. Saving compiles your edits into a new timestamped copy
of the scene JSON; the source scene is never rewritten, and the browser never
touches USD assets.

The full user guide, including the complete keymap, lives in
[docs/INSTRUCTIONS_SCENE_EDITOR.md](../../../docs/INSTRUCTIONS_SCENE_EDITOR.md).
Press **? Help** in the editor for the authoritative key reference.

## Setup

```bash
bash scripts/installation/install_light_editor.sh
```

This creates a dedicated `simfoundry-editor` conda env. The editor must not
share an env with simfoundry: it uses standalone `usd-core`, which can
shadow Isaac Sim's own `pxr`. Nothing in the env needs a GPU or OmniGibson.

**Re-run the installer after a pull.** Optional dependencies degrade quietly
(without `ruamel.yaml`, task-config saves lose their comments; without
`google-genai`, the Generate task panel reports itself missing), so a stale env
looks like a working one.

## Launch

```bash
mamba run -n simfoundry-editor python scripts/interactive/light_editor/server.py \
  --scene /absolute/path/to/<scene>_scene_state_latest.json \
  [--cameras nv_franka_droid]
```

Open <http://localhost:8770>. Useful flags:

| Flag | What it does |
|---|---|
| `--cameras NAME\|PATH` | Load an `external_sensors` config; its cameras become editable and the Cameras tab turns on |
| `--task-cfg NAME` | Pin which task config decides the camera → policy-input mapping when configs disagree |
| `--scene-root DIR` | Extra directory the scene launcher searches |
| `--asset-root DIR` | Extra directory the asset importer searches (repeatable) |
| `--allow-incomplete` | Open even when some objects have no drawable proxy (they appear as error rows) |
| `--splat-budget N` | Cap Gaussian-splat rendering (default 1,000,000; lower it heavily on machines without a GPU) |
| `--host 0.0.0.0` | Bind to the LAN. **There is no authentication** — prefer an SSH tunnel: `ssh -N -L 8770:localhost:8770 you@host` |

Extraction is cached under `web/data/` and invalidates itself when the scene,
its USDs, or the extractor change; a re-extract costs a couple of seconds.

## Example scenes

Download example scenes from
[nadunRanawaka1/simfoundry-assets](https://huggingface.co/datasets/nadunRanawaka1/simfoundry-assets):

```bash
hf download nadunRanawaka1/simfoundry-assets --repo-type dataset --local-dir assets
```

`Open scene…` (below) lists every scene under `assets/scenes/` once downloaded.

## Opening and creating scenes

- **Open scene…** (`O`) lists every scene under the scene roots plus recent
  ones, with prop counts, the room, and missing-asset warnings. Switching
  scenes never silently discards unsaved work.
- **+ New scene…** composes a scene from a template known to load (its room,
  robot, ground plane and version block carry over; you pick which props).
  Composition is refused if any referenced asset would not resolve.
- **Pipeline scenes**: `s13_og/reconstructed_og_scene.json` loads directly.
  Objects from `behavior-1k-assets` are encrypted and show as error rows
  (OmniGibson still loads them fine). Pass `--cameras-out <scene>_cameras` so
  two pipeline scenes' camera placements do not collide.

### Rooms and the ground plane

A pipeline scene has no background. Attach a scanned mesh room with:

```bash
mamba run -n simfoundry-editor python scripts/interactive/light_editor/background_io.py \
  --scene <scene>.json --list                        # what is available
mamba run -n simfoundry-editor python scripts/interactive/light_editor/background_io.py \
  --scene <scene>.json --background droid_desk_mesh  # attach one
```

After attaching, the tool probes the surface under each prop and fails loudly
when the room is in the wrong place. Expect to nudge yaw/origin with
`--position`/`--orientation` and re-check.

**Gaussian-splat rooms are drawn** (extracted straight from the USDZ, thinned
to `--splat-budget`), but a splat has no collision geometry: anything placed on
it falls through when physics starts. Use **Ground plane** on the Objects tab
to write `ground_plane_info` — position, orientation, and a `visible` flag —
which policy evaluation, teleop, and the OmniGibson editor all read.

## Editing

- **Selection**: click; `Shift`+click adds/removes; `Ctrl`+`A` selects all
  props; `Esc` clears. Group translate/rotate is rigid (relative offsets are
  preserved); group scale is always uniform.
- **Transform inspector**: position in metres, rotation in degrees (the saved
  quaternion stays visible), per-row revert, **Upright**, **Copy**/**Paste**
  across scenes.
- **Size row**: type the real-world measurement (m/cm/mm) and the scale is
  solved from the asset's native size. `base` keeps the object's footprint and
  underside in place while resizing. The panel warns in amber when OmniGibson
  will not reproduce the resulting box exactly (per-link scaling quirks).
  Mass does **not** follow size — edit it too for dynamics-sensitive tasks.
- **Drop to surface** rests the selection on the table (only the table — never
  the floor or the robot). It is geometric contact, not a settled pose.
- `+` / `-` scale the selection 5% per press (`Alt` for 1%).
- **Joints**: articulated assets get an **Edit Articulated Object** window with
  a slider per degree of freedom and ghosts at the limits. Joint *values* are
  saved and restored by OmniGibson; typed *limits* are recorded only — PhysX
  still reads limits from the USD.
- **Undo/Redo** cover objects and cameras in one history and never switch
  tabs or enter a camera.

## Cameras

With `--cameras`, each exterior sensor gets a live preview pane and a frustum.
Selecting a camera enters it: the viewport becomes the sensor's exact frame and
`W`/`A`/`S`/`D` fly it (`Space`/`Q` up/down, drag to aim, `Esc` to leave).

> `Ctrl`+`W` closes the browser tab — use `Q` for down while holding `W`.

- **Saving cameras** writes `scripts/cfg/external_sensors/<background>_cameras.yaml`,
  keyed by the scanned room, so a placement is aimed once per room and shared
  by every scene built in it. Poses are relative to the robot base.
- **Which camera feeds `exterior_image_1_left` is decided by the task config**,
  not the rig, and this repo's configs do not all agree. The editor flags
  disagreement at startup and on the preview badges; pass `--task-cfg` to pin
  the mapping for the task you are building.
- The **wrist** camera comes from the robot USD itself. It is shown, previewed,
  and locked — its pose is the arm's.

Use a saved placement in evaluation:

```bash
bash scripts/pipeline/C_application/run.sh --mode eval --scene-name my_scene \
  -- s15_eval.external_sensors_cfg=<background>_cameras
```

## Task configs

The **Task** section attaches a task config to the scene and edits its reset
randomization in place. Fields are **± half-widths** (`±0.1 m` is a 20 cm
spread); rotation is shown in degrees, stored in radians. The panel shows what
each group binds in the open scene, disables X/Y where predicate placement
overwrites them, and warns when `workspace_bounds` would clamp a range. A
**Visualize Reset Ranges** toggle at the top of the section draws each bound
object's ±X/±Y/±Z reset box in the viewport, live with unsaved edits. Saves
go through `ruamel.yaml` so hand-written comments survive, and are refused if
the file changed on disk in between.

Every config the editor writes or exports is validated first: unknown predicate
states, missing groups, missing `type`/`activity_name`/`termination_config`,
and — the quietest failure in the format — an **empty goal, which makes every
episode report success on step 1**. Invalid saves are refused with the reasons
(or written explicitly as drafts). The same check runs from a shell:

```bash
mamba run -n simfoundry-editor python scripts/interactive/light_editor/task_semantics.py \
  scripts/cfg/task/droid/*.yaml
```

## Saving and evaluating

Each save sends a complete snapshot, validated server-side and compiled against
the immutable startup scene — repeated saves are cumulative, writes are atomic,
and concurrent writers (a second tab, a second server, a text editor) are
detected and refused rather than overwritten. The output is a timestamped file:

```text
my_scene_state_light_edit_20260813_141501_123456_a1b2c3d4.json
```

Leave **also overwrite `_latest.json`** unchecked until the scene has passed
OmniGibson validation. Try the result:

```bash
mamba run -n simfoundry python \
  scripts/pipeline/C_application/stages/0_smoke_random_actions.py \
  --scene-json /absolute/path/to/my_scene_state_light_edit_....json

bash scripts/pipeline/C_application/run.sh --mode eval --scene-name my_scene \
  -- s15_eval.scene_json=/absolute/path/to/my_scene_state_light_edit_....json
```

**Review & Export** writes the scene, camera config, task config and evaluation
command together in one transaction, snapshots the configs it names, records a
manifest with a SHA-256 for every file, and refuses to export a task config
that could not run (unless you confirm).

## Physics checks

The browser cannot tell "resting" from "floating 2 mm above the desk". Two
optional OmniGibson-side checks close the gap:

```bash
# Step physics and report how far each object moved; --promote updates _latest
# only when nothing drifted beyond tolerance.
mamba run -n simfoundry python scripts/interactive/light_editor/settle.py \
  --scene /absolute/path/to/my_scene_state_light_edit_....json \
  [--steps 240] [--promote] [--report out.json]

# Load an export in OmniGibson and compare poses at load and after resets.
mamba run -n simfoundry python scripts/interactive/light_editor/parity_check.py \
  --scene /absolute/path/to/my_scene_state_light_edit_....json \
  [--cameras nv_franka_droid] [--out parity_out/]
```

`--settle-after-save` on the server runs settling after every save instead.
Parity is necessary, not sufficient — it does not build the full evaluation
environment.

## Importing assets

The asset library lists deduplicated assets from the scene's own `objects/`
tree, sibling scenes, and any `--asset-root`. Three ways in: **Browse…** (the
server's filesystem), drag-and-drop (`.glb`/`.ply`/`.stl` only — formats that
reference sibling files are refused rather than imported untextured), or a
typed path. Meshes are converted to pipeline-shaped USDs (Z-up, colliders,
both preview and MDL shaders); USDs are copied with their full dependency
closure. Imported objects have no verified physics — settle before relying on
them.

## Scope

- **Supported**: visualize everything the scene serializes (including
  Gaussian-splat rooms), transform/add/remove objects, edit joints, place the
  ground plane and cameras, edit and validate task configs, save loadable
  scene copies, and export a reviewed evaluation bundle.
- **Approximate**: proxies are visual-only — collision geometry, materials,
  and unsupported prim types may differ from the simulator; splats render
  without view-dependent color.
- **Not implemented**: authoring sensors, physics validation in the browser,
  authentication. Missing proxies appear as error rows, never silently.
