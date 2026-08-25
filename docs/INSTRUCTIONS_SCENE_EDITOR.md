# Scene Editor Instructions

How to run the two scene editors, place cameras, verify a layout under physics,
and hand the result to evaluation.

There are **two editors**, and they are for different jobs:

| | [Light editor](#light-editor-browser) | [OmniGibson editor](#omnigibson-editor) |
|---|---|---|
| Starts in | ~2 seconds | ~90 seconds |
| Runs | standalone OpenUSD + browser | Isaac Sim |
| Good for | placing/scaling objects, aiming cameras | physics, lighting, high-fidelity inspection |
| Manipulation | mouse gizmos, click-to-select | keyboard only |
| Physics | none (verify separately) | yes |

Use the light editor for layout. Drop to the OmniGibson editor when you need
physics or photoreal rendering.

**In a hurry?** Open a scene, move things, press **Save scene JSON**, then
[settle it](#settlepy--is-anything-floating) before you tick *also overwrite
`_latest.json`*. Everything else here is detail you can come back for.

## Contents

| | |
|---|---|
| [Building a fresh scene from a video](#building-a-fresh-scene-from-a-video) | reconstruction → editor |
| [Environments](#environments) | which Python, and why it matters |
| **[Light editor](#light-editor-browser)** | [run it](#run-it) · [open another scene](#opening-another-scene) · [start a new one](#starting-a-new-scene) · [controls](#controls) |
| [Placing things](#the-transform-inspector) | [transform inspector](#the-transform-inspector) · [drop to surface](#drop-to-surface-and-re-seat) · [anchor, recenter, arrange](#layout-anchor-recenter-arrange) · [add and remove](#adding-and-removing-objects) |
| [Physical properties](#physics-mass-and-friction) | [mass and friction](#physics-mass-and-friction) · [joints](#joints-opening-a-drawer) · [the ground plane](#the-ground-plane) |
| [Cameras](#cameras) | [once per room](#do-i-set-this-up-once-or-per-scene) · [aiming](#aiming-them) · [previews](#seeing-what-the-cameras-see) · [which input is which](#which-camera-is-exterior_image_1_left) |
| [Tasks](#tasks) | [attach one](#attaching-a-task-config) · [ranges](#randomisation-ranges) · [generate one](#generating-a-task) |
| [Finishing](#checking-a-layout-before-you-commit-to-it) | [check layout](#checking-a-layout-before-you-commit-to-it) · [save](#saving) · [Review & Export](#review--export) |
| [Verifying](#verifying-an-edited-scene) | [settle](#settlepy--is-anything-floating) · [parity](#parity_checkpy--does-the-simulator-agree) |
| [OmniGibson editor](#omnigibson-editor) | the 90-second one |
| [Troubleshooting](#troubleshooting) | |

---

## Building a fresh scene from a video

```bash
cd /path/to/simfoundry
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name cluttered_scene \
  --video-fpath /path/to/ClutteredScene.MOV \
  -- s7_mesh.low_vram=true
```

`s7_mesh.low_vram=true` is required on a 24 GiB card such as the RTX 4090 —
mesh generation at stage 7 otherwise wants about 29 GiB. Add
`--detect-articulation` for scenes with hinges or drawers.

Output lands in `Data/<scene-name>/`, ending with `reconstructed_og_scene.json`
(stage 13). Smoke-test it, then open that JSON in the light editor:

```bash
bash scripts/pipeline/C_application/run.sh --scene-name cluttered_scene --mode smoke-random

cd scripts/interactive/light_editor
mamba run -n simfoundry-editor python server.py --scene ../../../Data/cluttered_scene/reconstructed_og_scene.json
```

### The layout already matches your video

This is the point of reconstruction, and worth being explicit about: stage 4
unifies the world frame, stage 8 matches object poses against the video, and
stage 11 settles them under physics. **Objects come out where they actually were
on your table, at true scale, with true spacing.** You do not need to rebuild
that by hand.

So for a video-derived scene, use **Recenter**, not **Arrange**:

| | keeps video layout | use when |
|---|---|---|
| **Recenter** | yes — rigid shift only | you want the real arrangement, positioned consistently for the robot |
| **Arrange** | **no** — re-scatters props | you want randomised variation, e.g. generating training diversity |

Reaching for Arrange on a freshly reconstructed scene throws away the exact
information the pipeline just spent an hour recovering.

## Environments

Two different Python environments are involved. Getting this wrong is the most
common way to waste an hour.

### `simfoundry` — for anything that imports OmniGibson

Beware: an env can carry this repo's `simfoundry` package but a *different*
checkout's OmniGibson (from another clone's `deps/BEHAVIOR-1K/`). Verify before
editing:

```bash
mamba run -n simfoundry python -c 'import omnigibson; print(omnigibson.__file__)'
```

The printed path must resolve inside **this** checkout's `deps/` tree. Running
against the wrong one stamps another tree's OmniGibson version into saved
scenes and silently ignores any `deps/` change made here. `run_editor.sh`
checks this and warns, but does not block.

### `simfoundry-editor` — for the browser editor

A separate conda env, because Isaac Sim ships its own `pxr` and installing
standalone `usd-core` alongside it risks shadowing that copy. Setup (re-run
after a pull to pick up new dependencies):

```bash
bash scripts/installation/install_light_editor.sh
```

No GPU, no Isaac Sim, no OmniGibson.

---

## Light editor (browser)

### Run it

```bash
cd /path/to/simfoundry/scripts/interactive/light_editor

mamba run -n simfoundry-editor python server.py \
  --scene /path/to/simfoundry/assets/scenes/droid_desk_put_away_trash/droid_desk_put_away_trash_scene_state_latest.json
```

Open <http://localhost:8770>. Loopback only.

Any scene works — `ls assets/scenes/*/[a-z]*_scene_state_latest.json` lists the
saved ones, and a raw pipeline scene (`Data/<name>/s13_og/reconstructed_og_scene.json`)
opens just as well. A file is a scene if it *contains* `objects_info` and
`state`, not if it is named like one, so both kinds appear in the launcher.

`--scene` only says which one to open **first**. After that you switch inside
the editor.

### Opening another scene

**Open scene…** in the panel, or `O`, lists every scene under the scene roots —
the folder holding the scene you opened, plus this checkout's `assets/scenes` —
with the ones you have opened before at the top. Click one and the editor
rebinds and reloads: about a quarter of a second for a scene it has extracted
before, about two seconds for a new one.

Each row carries what you need to recognise it: how many props, which scanned
room, and in amber how many of its assets are missing on this machine — so a
scene that will not open says so before you click. `N saves` expands that
scene's earlier state files (`light edit · 2026-08-14 09:14`, bare timestamps
from the OmniGibson editor, cousin combinations); any can be opened directly.

**Switching never discards work silently.** With edits pending you are asked,
and given all three answers rather than two:

| Answer | What happens |
|---|---|
| Save, then continue | Writes the scene JSON, and the camera YAML too if a camera moved, then switches |
| Discard and continue | Switches; the edits are gone |
| Stay here | Nothing happens |

Imported objects are guarded separately, by the server, because a second tab
could have made them and this page would not know. A switch is refused until the
browser confirms, naming the objects. The copied asset files stay on disk under
the scene's `objects/` directory either way, which is harmless.

The recents list lives at
`$XDG_STATE_HOME/simfoundry/light_editor/recent_scenes.json`, outside the
repository. Scenes that have moved or been deleted drop off it.

> **A scene can only be opened from the roots the launcher searches.** The
> endpoint takes an absolute path, so without that "open a scene" would be "read
> any JSON on this machine". Add `--scene-root DIR` for a library elsewhere.

### Starting a new scene

**+ New scene…** composes one from a template. It deliberately does *not* write
a blank document: a scene with no `versions` block, no `ground_plane_info`, no
robot and no background is not a scene — every pose in it is expressed against a
room that does not exist, and you find out minutes later inside Isaac Sim.

| Step | What it asks |
|---|---|
| 1 · Template | Which scene supplies the room, robot, ground plane and version block |
| 2 · What comes with it | Which of its props to carry over — **Room only** and **Keep everything** cover the usual cases. Robot and background always come |
| 3 · Name | Lower-case letters, digits and underscores; the panel shows the exact file it will write |

Kept props are copied with their whole bundle, so `usd_path` stays relative and
the scene still loads on another machine. The scanned room is referenced, not
copied — it is shared by every scene in that root. Velocities are zeroed, and
`metadata.composed_from` records what it came from.

The scene is created in the first scene root (`--compose-root` overrides), built
aside and renamed into place, and **refused if any asset it references is
missing**. Then it opens, and you fill it from the asset library.

"Missing" means the library is installed and does not have it. That is a
different answer from *unchecked* — no root was given to look in — and only the
first one blocks, because "the datasets are not fetched on this machine" is not
"this scene is broken". Assets named by class rather than by path count too: a
`DatasetObject` and the robot have no `usd_path`, and treating that as "nothing
to resolve" is how a scene missing every one of its props used to pass.

A template also has to *supply* what the wizard says it inherits — a `versions`
block, a robot and a room. Composing around a missing one produces a document
where every pose is expressed against a room that does not exist.

### Controls

| Input | Action |
|---|---|
| LMB drag | Orbit |
| RMB drag | Pan |
| Wheel | Zoom |
| Drag the panel edge | Trade width between the viewport and the panel · double-click it to go back to the default · arrow keys work too |
| Click object | Select |
| `Shift`+click | Add to the selection, or take it back out — in the viewport or the object list |
| `Ctrl+A` | Select every prop (not the robot or the room, which are not movable here) |
| `M` / `R` | Move / rotate gizmo. Scale has no key — `+`/`-` below already scale, and `W`..`D` are the axis nudges |
| `W` `A` `S` `D` | **Nothing selected:** walk the view, held — forward, left, back, right along the ground. `Space`/`E` up, `Q` down, `Shift` faster, `Alt` finer. **With a selection:** nudge it along the world axes instead |
| `F` / `Shift+F` | Frame the selection / the whole scene |
| `O` | Open another scene, or start a new one |
| `V` | Show/hide the exterior camera previews |
| Arrows | Nudge the selection 5 mm (Shift 5 cm, Alt 1 mm) |
| `+` / `-` | Scale the selection up or down 5% a press, uniformly (Alt 1%). Works whatever the gizmo is set to |
| `PgUp` / `PgDn` | Nudge up / down |
| `G` | Show/hide the safe-frame guides inside those previews |
| `Shift+G` | Show/hide the ground grid and the origin axes — a view setting, never saved |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo (100 steps). Restores state only — it never switches tabs or puts you inside a camera |
| `Ctrl+D` | Duplicate the selection (one object at a time) |
| `Del` / `Backspace` | Remove the selection |
| `Esc` | Deselect everything, in one press |

**Several objects at once.** `Shift`+click builds a set; the gizmo then moves,
turns or scales all of it together, and one `Ctrl+Z` unwinds the whole drag.
A group translate and a group rotate are rigid — every relative offset in the
set survives them — and a group scale is always **uniform**, because scaling a
set along world axes when its members are turned differently from one another
is a shear, and the scene JSON has only a position, a quaternion and a scale to
write it into. One object at a time can still be stretched per-axis.

Rotate and scale turn about the middle of the set, which the panel prints under
the object count. The transform fields show `mixed` where the members disagree,
and typing a number sets it on **every** member — five props and `0.752` in the
Z box is "put these all at desk height".

Panel: **Snap** (1 cm / 15°), **World/Local** gizmo space, **Hide bg**,
**Hide grid**, **Cam views**, **Guides**, the transform inspector ([below](#the-transform-inspector)),
**Drop to surface**, **Revert**, **+ Add object**, **Duplicate**, **Remove**.

The viewport's top-left pill also offers **Wrist**: the robot's own camera, the
image the policy receives as `wrist_image_left`. It comes out of the robot USD
rather than out of an `external_sensors` config, sits where the arm's saved joint
pose puts it, and is **read-only** — you can look from it and watch its preview,
but its pose is the arm's and nothing here writes it back.

**Undo** and **Redo** are in the viewport instead, top right, beside **? Help**,
with the number of steps still on the stack in front of them. They take back
what the mouse just did out there, so they sit where the mouse already is rather
than in a panel section that may have scrolled away.

#### The panel's width is yours

Drag the seam between the panel and the viewport. It opens at 360 px and will
not go below 345, which is where a rotation like `-179.999` stops losing its
minus sign off the left edge of its field — the fields are right-aligned, so the
digit a narrow panel drops first is the sign. Widen it to 720 px when you are
reading coordinates, and narrow it back when you would rather have the room for
the scene.

Double-click the seam to return to the default; the arrow keys move it 16 px at
a time (64 with `Shift`) once it has focus. The width you asked for is what is
remembered — for this editor, on this machine, across reloads and scene
switches.

On a window too narrow to give the panel 345 px *and* the viewport 360 px, the
**panel** gives way, down to a floor of 240 px. Below 345 a coordinate starts
losing its sign, which is bad; a viewport you cannot see the scene in is worse,
and it takes the camera previews with it.

#### Drop to surface, and re-seat

**Drop to surface** rests the selection on the **table** — never on the floor.
A prop that has ended up under the desk comes *up* onto it, and anything already
on the table counts, so a fruit dropped where a plate sits lands on the plate.
It only changes z; **Re-seat on table centre** only changes x and y.

### The transform inspector

The numbers under **Selected**, for when the gizmo is not precise enough:

| Row | Unit | Notes |
|---|---|---|
| `pos` | metres | Scene frame, Z-up. For a camera, relative to the robot base |
| `rot` | degrees | Extrinsic XYZ about the world axes |
| `size` | metres | The object's bounding box, along its own axes. Click the `m` to read it in cm or mm |
| `scale` | multiplier | 1 is the asset as modelled |
| `quat` | — | Read-only. This is what the scene JSON actually stores |

Columns are labelled **X, Y, Z once at the top, in the gizmo's own colours** —
X red, Y green, Z blue — and each field carries its column's colour on its left
edge, so the box and the arrow you would drag instead read as one thing.

**The degrees row is a view over the quaternion, not a second copy of it.**
Typing an angle recomputes the quaternion straight away and the `quat` line
shows the result. Degrees do not round-trip exactly — 190 comes back as −170,
and a gimbal-locked pose can redistribute between axes — which is why the
quaternion stays on show rather than being hidden behind the friendlier form.
That is the rotation being the same, not drifting.

All six numeric fields are steppers, and the arrow keys nudge the selection
along whichever world axis best matches the direction the arrow points on
screen. Holding an arrow is one undo step, not forty. A field that cannot read
what you typed puts the old value back and says which field it was — the editor
has one status line, so an unnamed refusal reads as being about whatever you did
last.

| Control | What it does |
|---|---|
| `⟲` on a row | Restores that row to the last saved value and leaves the other two alone — for when the rotation is right and only the position drifted. **Revert** still does all three |
| `Uniform` | Uniform scale: typing into any scale or size field sets all three, and a gizmo scale drag applies one ratio to every axis |
| `m` on the size row | Cycles the unit: metres → centimetres → millimetres. Always metres on a fresh load |
| `base` on the size row | On by default. Resizing keeps the bottom of the box on the surface it was resting on. Off, it resizes about the asset's own origin |
| **Upright** | Clears all rotation to identity — *not* the same as `⟲`, which restores the saved angle |
| **Copy** / **Paste** | Moves a whole transform from one object to another, as one undo step |

Copy survives a reload and a scene switch, so "put this where the one in the
other scene was" works. Pasting onto a camera applies position and rotation
only; a camera has no scale the config stores.

**Sizing to a real measurement.** Measure the real object, type the number. The
`size` row is the object's bounding box, so the editor solves the scale from it
(`scale = measured / native`) — measure a baseball at 73 mm and a scanned proxy
reconstructed at 69.0 mm takes a scale of 1.058. A cube's bounding box is the
cube. With `Uniform` on, one measured dimension sizes the whole object, which is
what you want when what you measured was a diameter. Nothing new is written to
the scene JSON: `scale` is still the only field that changes, and the size is
recomputed from it every time it is shown.

Two things it cannot promise, both said on screen when they apply. **Some assets
do not scale linearly in OmniGibson** — it scales each link separately, so an
object with a hinged lid keeps the gap between its parts at full size (about 8%
off at scale 0.8, 46% at 0.5), and an asset whose USD already scales its own root
has its scale discarded entirely. The panel warns in amber and names the object.
And **mass does not follow size** — these assets carry an explicit mass and
inertia that no resize touches, so a prop scaled for the camera is not
automatically right for dynamics.

### Seeing what the cameras see

Every exterior sensor renders live into its own window over the overview, so you
can arrange props and watch the shot the policy will actually receive instead of
placing by eye and discovering afterwards that something is out of frame.

| Part of the window | What it does |
|---|---|
| Caption bar | Drag to move; double-click to put it back where it started |
| The picture | Click to step into that camera — the same act as selecting it |
| Corner grip | Drag to resize; the sensor's aspect ratio is kept |
| `–` / `+` | Collapse to the caption bar and back; a collapsed pane is not rendered |

Where a window starts is the camera's own place in the room: the left-hand pane
is the sensor physically on the robot's left, worked out from its pose in the
robot's frame rather than from the order the config happens to list them. A rig
with three sensors spreads them along the same edge. Positions and sizes are
remembered per camera config, so a layout worked out once survives a reload.

Each pane wears **its frustum's colour** — one colour per camera, on the cone in
the viewport, on the camera body, on the pane border and on the row in the object
list. That matters here because **both DROID sensors are named `..._left`**: they
are the left eye of a stereo pair, not a left and a right camera, so the name
alone is easy to misread and the picture alone says nothing.

The caption also carries **which policy input this camera lands in** — `ext 1`
for `exterior_image_1_left` — and the footer its **resolution and field of
view**. The observation key is not recorded in the sensor config at all; see
[which camera is exterior_image_1_left](#which-camera-is-exterior_image_1_left).
When the task configs disagree about it, the badge turns amber and its tooltip
names which config says what.

**Guides**, or `G`, draws the ordinary framing aids inside each picture: 90%
action-safe and 80% title-safe rectangles, thirds, and a centre cross. These
sensors are matched to hand-aimed real ones, so a subject sitting against the
edge of the frame survives neither a small pose error nor the next scene built in
the same room.

**Cam views** in the Gizmo panel, or `V`, hides them all. They are not drawn while
you are flying a camera — you are already looking through one, and that view has
its own third-person inset.

**Nothing the editor draws for itself reaches these pictures.** The grid, the
axes, the transform gizmo, the camera bodies and every frustum sit on a separate
render layer that the sensor cameras do not draw, so a preview shows what
OmniGibson's sensor would record and nothing else.

Cost is nil in practice: the same scene renders at 60 fps with the previews on or
off on an RTX 4090, since the pictures are scissor rectangles in the same canvas
rather than separate WebGL contexts, and only the frames around them are DOM.

### Checking a layout before you commit to it

**Check layout** answers four questions that nothing used to ask:

| Check | Why it matters |
|---|---|
| In frame? | An object outside every policy camera wastes a whole evaluation run, and settling cannot see a frustum |
| Within reach? | Auto-arrange always respected the robot's ~0.95 m reach; hand placement was held to nothing |
| Overlapping? | Interpenetrating objects are what settling turns into an explosion rather than a correction |
| Task bindings intact? | Task YAMLs bind objects by name or category. Deleting a bound object leaves an empty group, and the run then produces episodes that **vacuously succeed** — indistinguishable from a policy result. Duplicating one can make a group ambiguous and trip an assertion 90 seconds into a run |

Marked rows carry the reason in their tooltip. **None of this ever blocks a
save** — the tool cannot know what you meant. The same summary is printed after
every save, so you see it whether or not you asked.

### Review & Export

Saving the scene and saving the cameras used to be unrelated acts. Nothing
recorded that a layout was authored against a particular set of camera poses,
and nothing checked that the task about to run still bound the objects in the
scene. Each piece was individually valid and the combination could be wrong —
which is the worst shape a mistake takes here, because the run completes,
produces plausible video, and reports numbers.

**Review & Export** decides all four together and writes them in one
transaction:

| Part | What the review tells you |
|---|---|
| Scene | What will be written, and the pending change counts |
| Cameras | Which `external_sensors` config, and whether these are still the **rig defaults** nobody has aimed for this room |
| Task | Which task config. It defaults to the one the **Task** panel has open; failing that, to the best-guess association, with how it was inferred — `certain` (a run config pairs them), `likely` (task name), `possible` (scene variant) |
| Warnings | Layout and task-binding problems, in one place |

Then the exact evaluation command, with **Copy command** and **Open output
folder**.

Two things it deliberately does:

- **The command pins an absolute path to the exported file.** Passing a scene
  *name* makes the eval stage resolve `_latest.json` at run time, so the layout
  evaluated would be whatever was promoted most recently rather than the one you
  just reviewed. Pinning the file is what makes the review mean anything.
- **The review offers no runnable command.** The exported filename carries a
  timestamp, so it cannot be known in advance, and showing the source path
  instead would hand you a command that evaluates the *unedited* scene. Copy and
  Open stay disabled until you export.

- **A task that could not run is refused**, ahead of writing anything. A config
  with no goal predicate does not fail at run time — `all([])` is `True`, so
  every episode reports success on step 1. The refusal names what is wrong and
  you can insist; see [tasks](#tasks).

Every export also writes `<scene>_export.json` beside the scene, recording all
four parts, the command, the warnings as they stood, **and a sha256 for every
file it names**.

**The files it names cannot change afterwards.** The `external_sensors` and
`task` configs are shared, hand-edited YAML, and an export that merely *pointed*
at them described a run that could quietly become a different run. So each is
copied into `exports/` under its own config group, with a name carrying the
export's id, and the command names the copy:

```text
scripts/cfg/external_sensors/exports/nv_franka_droid_20260820_143012_123456_a1b2c3d4.yaml
scripts/cfg/task/exports/droid_desk_serve_fruits_20260820_143012_123456_a1b2c3d4.yaml
```

Expect one pair per export in `git status`. They are the record of what was run;
send `snapshot_configs: false` if you are iterating against a config you are
still editing, and the manifest says which was done.

**It is one transaction.** Every file's contents are decided before any of them
is written. If a later write fails, the ones that landed are taken back and the
response says so — a checkout holding half an export, with camera poses
retargeted for a scene file that was never written, is the outcome this exists
to prevent.

> **One thing a snapshot cannot fix.** The eval stage resolves
> `task/<task_name>.yaml` *before* the group it was given, because the
> `task=load_scene task.task_name=…` indirection depends on that order. A flat
> config whose name matches therefore shadows the group the command names. The
> review warns when that would happen.

### Adding and removing objects

**Remove** leaves the selected object out of the next save. Nothing is deleted
from disk and the source scene is never rewritten — the row stays in the list,
struck through, and one click (or `Ctrl+Z`) puts it back. The robot and the
background cannot be removed: one comes from the rig and the other is the frame
every pose is expressed in.

**+ Add object** opens the asset library: every USD under this scene's own
`objects/` tree (including any cousins generated for it) plus every sibling
scene's. Filter by category, asset id or source scene, then click a row.

Clicking a row **selects** it — it does not import. The panel below then shows a
turning 3D preview and the facts you need before committing:

| | |
|---|---|
| **size** | in millimetres, so a model authored in centimetres is obvious immediately |
| **mass** | authored (USD) or estimated from volume (mesh) |
| **collision** | for a USD, *what it has* — "4 prim(s): convexHull", or "none in this USD"; for a mesh, *what conversion will make* |
| **caveats** | the small print, before it costs you an undo |

For a mesh, changing **scale**, **up axis** or **collide** re-reads it, so the
numbers always describe the settings currently selected.

Then **Add** puts a translucent ghost of the object under your pointer. It rests
on whatever surface is beneath — the status line names it — and a click commits.
`Esc` cancels, and nothing has been copied into the scene up to that point.

Each row is one *asset*, not one file. The pipeline copies every asset into each
scene that uses it, so one bowl modelled once exists as four files; those collapse
into a single row tagged `×4`. Rows carry the asset id (`orange_bowl hvrbyn`)
because several scenes model the same category differently — there are three
genuinely different `orange_bowl`s here, and the id is how you tell them apart. The
asset is copied into `<scene>/objects/<category>/<variant>/` and converted to a
browser proxy, and the new object appears on the anchor's support surface with a
generated name (`yellow_banana_0`). **Duplicate** does the same thing with the
selected object's own asset, so two apples are two real objects.

Assets are copied rather than referenced where they lie so that `usd_path` stays
relative and the saved scene still loads on another machine — the failure mode
the checked-in cousin-combination JSONs already have, where every path points
into a home directory that does not exist here.

Both operations are ordinary undo steps, and neither touches the scene file
until you press **Save scene JSON**. An import that you undo and never save
leaves its copied asset behind under `objects/`, which is harmless.

#### Importing something from your own machine

Under the library there is an **Import from this machine** field. Paste an
absolute path and press Enter:

| What you paste | What happens |
|---|---|
| `/path/to/thing.usd` (or `.usda`, `.usdc`, `.usdz`) | Copied in with everything it references — textures, sublayers, payloads — as authored |
| `/path/to/thing.glb` (or `.gltf`, `.obj`, `.ply`, `.stl`, `.dae`) | Converted to a USD shaped like the pipeline's own, with materials and a generated collider |
| `/path/to/a/folder` | Every asset in it joins the library, browsable like any other source |

The field takes `~`, surrounding quotes, and `file://` URLs, so whatever you
copied usually works unchanged. There is no file picker because a browser
deliberately withholds real paths from the page — and a path is what the server
needs, since it reads the file directly and a multi-file USD bundle cannot
travel through an upload as one file.

**Options** (mesh files only):

| Option | Default | Why you would change it |
|---|---|---|
| `scale` | 1.0 | The source was authored in centimetres (`0.01`) or millimetres (`0.001`). The panel reports the resulting size in metres, so you can check |
| `up axis` | auto | `auto` rotates glTF, which is Y-up by specification, and leaves OBJ/PLY/STL alone because they have no convention. Flip it if an object arrives on its side |
| `collide` | convex hull | A bowl or mug needs `convex decomposition` or it behaves as a solid lump |
| `mass` | estimated | Estimated from volume at 500 kg/m³. Manipulation cares about this, so set it if you know it |

After an import the panel reports the final size, mass, whether it was rotated,
and any caveats.

> **A converted mesh has an approximated collider,** generated here rather than
> cooked by the pipeline's USD-import stage. It is good enough to rest on a
> table — verified: an imported `.glb` and `.stl` both settled and then passed
> `parity_check.py` through load, two resets and 60 physics steps within 0.1 mm
> — but for anything precise, run the asset through the pipeline instead.

An imported object is placed by hand, so it has no physics behind it. Verify it
the same way as any other hand-placed pose:
[settle](#settlepy--is-anything-floating) it, then check parity. A freshly
imported banana in a test run dropped 7 cm on the first physics step and failed
`parity_check.py`; after one settle it passed load, both resets and 60 steps
within 0.1 mm.

The panel has two tabs — **Objects** and **Cameras** — because they write to
different files with different rules, and because they are edited in completely
different ways: objects with the gizmo above, cameras from inside the camera
([below](#cameras)). Switching tabs clears the selection.

### Layout: anchor, recenter, arrange

Both tools work off an **anchor** — the middle object. It is guessed on load as
the object nearest the XY centroid of the editable set (on a cluttered table the
subject usually sits amid the distractors, so "most central" beats "largest",
which can latch onto a big distractor). It shows as `anchor` in the object list;
select anything and hit **Set as anchor** to override.

#### Recenter — keeps the video's layout

Shifts the **whole** layout rigidly in XY so the anchor's bounding-box centre
lands a set distance in front of the robot base, along the robot's own forward
axis. Every relative offset is preserved exactly and Z is never touched, so
objects stay on their support surface.

Use this on any scene reconstructed from video: the arrangement stays as it was
on your real table, but every scene presents its subject to the robot the same
way, inside the reachable workspace.

#### Arrange — discards it

Scatters every non-anchor object around the anchor instead, for randomised
variation. Press **Arrange / re-roll** repeatedly until a layout looks right —
re-rolling is the point, and one `Ctrl+Z` unwinds an entire arrange rather than
one object at a time. `radius` is the distance from the anchor in metres.

Props are spread at even angles (with jitter), rested on the anchor's base
plane, kept within ~0.95 m of the robot base when a robot is present, and
rejected if their bounding box overlaps something already placed. If a prop
cannot be placed in 40 attempts it is **left where it was and named in the
status line** — a partial arrange never silently pretends to have succeeded.

Arranged poses have no physics behind them, so run a
[settle](#settlepy--is-anything-floating) before relying on the result.

### Physics: mass and friction

Under **Selected**, for one prop at a time. Not for the robot (its mass and
contact materials come from its rig) and not for the room.

| Field | Starts at | What it does |
|---|---|---|
| `mass` | the USD's own authored mass | An override in kilograms. Type the asset's own number back, or press `⟲`, and the override comes off rather than being written as the same value |
| `friction` | whatever the scene already carries | One coefficient for the whole object — it is applied to every rigid body the asset has |

**The two are not symmetric, and the panel says so.** Friction is a real
OmniGibson constructor argument (`link_physics_materials`), applied to the
link's collision meshes at load, so a number typed here works everywhere with no
change to any consumer. **Mass is recorded only.** OmniGibson has no load-time
mass argument, and PhysX reads `physics:mass` out of the USD — which is shared
by every scene using that model, so rewriting it there would change all of them.
A consumer that wants to honour a scene's mass applies it after load with one
line: `obj.root_link.mass = args["mass"]`.

The friction field is greyed out when the asset's links could not be read. There
is nothing to apply a coefficient *to* in that case, and a link name OmniGibson
cannot find raises on load rather than misbehaving quietly.

Neither field has anything to see in the viewport, so a retuned mass is unsaved
work you could otherwise lose by reloading. It counts on the Save button
(`1 retuned`, kept apart from `1 moved` — nothing was moved) and marks the row.

### Joints: opening a drawer

A prop with degrees of freedom gets a **Joints** section and an **Edit
Articulated Object** button. The window draws that object on its own and moves
the link each joint drives, so the number and the geometry are the same claim:

| | |
|---|---|
| `value` | Where the joint is. Metres for a slide, degrees for a hinge |
| `lower` / `upper` | This scene's override of the asset's range |
| **Preview range** | Runs the joint end to end and puts it back — a preview, not an edit |
| Ghosts | A translucent copy of the link at each limit |

Same asymmetry as above, for the same reason. **`joint_pos` is real** — it is an
existing key OmniGibson restores on load — so a drawer left open here is open
everywhere. **`joint_limits` is recorded only**: PhysX reads a joint's range out
of the USD, and a consumer applies the override after load with
`obj.joints[<name>].lower_limit = …`.

Values past the asset's authored range are flagged, never clamped. A real saved
scene already stores a drawer at 0.17 m against a USD limit of 0.15, and quietly
moving it to make the panel self-consistent would be lying about the file.

> **Gravity closes what you open.** A drawer left open with nothing to rest
> against is shut again by the first settle, which then reports 0.0000 m of root
> movement because the object's base never moved. `settle.py` reports joint
> drift separately for exactly this reason — see
> [verifying a layout](#settlepy--is-anything-floating).

### The ground plane

**A Gaussian-splat room has no collision geometry at all.** It is loaded
`visual_only` and has no mesh prims, so the desk you can see is a picture of a
desk: props placed on it fall through and keep falling the moment physics
starts. A mesh room does not have this problem — its triangles are colliders.

The **Ground plane** section writes `ground_plane_info`, which is what fixes
that. Put the plane at the height of the surface in the picture:

| Control | Effect |
|---|---|
| **Add** / **Remove** | Whether the scene states a floor position at all |
| `height` | Where it sits. The plane is infinite, so this is the only number that matters unless you tilt it |
| **Visible in the simulator** | Overrides the run config's `floor_plane_visible`. Under a splat room you usually want it *off* — the floor should be at the height of the desk in the picture, not drawn as a grey plane through it |
| **Ground plane** (Visibility) | A *view* setting. Whether the editor draws its stand-in, and nothing to do with what is saved |

> **The plane's existence is not the scene's to decide.** Whether OmniGibson
> creates a floor plane at all comes from `use_floor_plane` in the task YAML, at
> environment-construction time. Every task config here sets it true, so writing
> this block positions a plane that is already there — it cannot conjure one
> into a run configured without it.

### Saving

**Save scene JSON** writes a timestamped file beside the original:

```text
<scene>_scene_state_light_edit_<timestamp>_<hash>.json
```

`_scene_state_latest.json` is **not** touched unless you tick *unsafe: also
overwrite `_latest.json`*. That file is what every downstream stage loads by
default, and the browser has no physics to confirm a pose actually rests on a
surface. Leave it unticked until the scene has
[settled](#settlepy--is-anything-floating).

**A save carries everything, not just what you moved.** The server rebuilds each
write from the document it opened plus one complete snapshot from the page, so a
field the browser left out would be *reverted*, not left alone. Transforms,
mass, friction, joint values and limits all go every time. Sending an unchanged
value costs nothing — the server compares before it writes, so a save that
changes nothing writes nothing and reports nothing as changed.

**Promoting moves what the editor is serving.** Tick the box and `_latest.json`
is rewritten, which is the file the launch command opens — so the editor adopts
it: reloading the page shows the promoted layout rather than the one this
session started with.

#### Two tabs, or two people

One person editing at a time is still the intended way to use this, but a second
one no longer costs anybody their work silently:

| | What happens |
|---|---|
| Another tab saves while you have **nothing** pending | Your page quietly catches up — it fetches that revision's contents and adopts both together |
| Another tab saves while you **do** have edits pending | Your page latches: Save and Review & Export go grey, a banner explains, and your edits stay on the page to copy out |
| Another *process* changes a file this editor is about to write | The write is refused with what is on disk now, and nothing is overwritten |

The last one is the case no lock inside this server can help with — a second
editor, or a text editor, or a settle job that started ten minutes ago. Every
publish states the digest it expects to be replacing.

### Server flags

| Flag | Effect |
|---|---|
| `--port N` | Serve on another port (default 8770) |
| `--host ADDR` | Interface to bind. Loopback by default. `0.0.0.0` lets a teammate on the LAN in — there is **no authentication** |
| `--allow-hosts NAMES` | Extra `Host` header names to accept, for a non-loopback bind |
| `--no-extract` | Reuse the cached `web/data/`; refused if it does not match the scene and its SHA-256 |
| `--no_textures` | Geometry only — faster, flat shading |
| `--allow-incomplete` | Serve even when some objects have no usable visual proxy. Off by default: placing an object you cannot see is how a scene silently goes wrong |
| `--splat-budget N` | Most gaussians a splat room may keep (default 1,000,000; 0 keeps all). Lower it — 100000 is comfortable — when running headless without a GPU |
| `--dataset-dir DIR` | OmniGibson's `gm.DATA_PATH`, where `DatasetObject` entries are resolved. Defaults to `deps/BEHAVIOR-1K/datasets` |
| `--asset-root DIR` | Where to look for importable USDs. Repeatable. Defaults to the scene's own `objects/` tree plus its sibling scenes |
| `--cameras NAME` | Load an `external_sensors` config (see below) |
| `--cameras-out NAME` | Override the config name (default: `<background>_cameras`) |
| `--task-cfg NAME` | Task config (a stem from `scripts/cfg/`) that settles which camera is `exterior_image_1_left`. Only needed when the task configs disagree — the editor says so when they do |
| `--scene-root DIR` | Extra directory of scene directories for the launcher. Repeatable. Also bounds where a scene may be opened from at runtime |
| `--compose-root DIR` | Where **+ New scene…** creates scene directories (default: the first scene root) |
| `--recents-file PATH` | Where the recently-opened list is kept (default: `$XDG_STATE_HOME/simfoundry/light_editor/recent_scenes.json`) |
| `--settle-after-save` | Run physics verification after each save |
| `--settle-steps N` | Physics steps per settle (default 240) |
| `--settle-tolerance M` | Metres of movement counted as unsettled (default 0.005) |
| `--settle-python PATH` | Interpreter with OmniGibson (auto-detected otherwise) |

---

## Cameras

### Do I set this up once, or per scene?

**Once per room.**

Camera poses in `external_sensors` configs are relative to the robot base link
(`pose_frame: parent`, `panda_link0`), not to the world. So the same placement
frames the same shot in every scene built in the same scanned background with the
same rig — all sixteen `droid_desk_*` scenes share `droid_desk_mesh`, and one
placement covers all of them.

That is what the editor keys on. Aim the cameras, hit **Save cameras YAML**, and
the placement lands in a config named after the room:

```text
scripts/cfg/external_sensors/droid_desk_mesh_cameras.yaml
```

Open *any* other scene shot in that room and it picks up where you left off:

```bash
mamba run -n simfoundry-editor python server.py \
  --scene /absolute/path/to/droid_desk_serve_banana_scene_state_latest.json \
  --cameras nv_franka_droid
```

```text
Cameras: resuming the placement saved for background droid_desk_mesh (droid_desk_mesh_cameras.yaml)
```

`--cameras` is the starting point for a room you have never aimed in, not an
override. Pass `--cameras-out NAME`, or type a name into **Save as**, to key the
file by something else — a single scene that needs its own framing, or a name
shared across rooms. Configs written by an older version as
`<scene>_cameras.yaml` are still loaded and move to the room-keyed name on the
next save.

A named file sits *beside* the room default rather than replacing it, so only a
write that landed on `<background>_cameras.yaml` is the one other scenes in that
room pick up. The panel says which of the two just happened. Naming a file this
session did not write asks before overwriting it; saving repeatedly under a name
of your own does not.

The count beside the button is what a save would **write**. The robot's own
camera is on the tab so you can look through it, but it rides the arm and has no
`external_sensors` entry to write back to, so it is named separately rather than
counted in.

### Aiming them

Switch to the **Cameras** tab and select one. **You are now inside it**: the
viewport is that sensor's view at its true field of view, letterboxed to its
aspect ratio, so what you are looking at is the frame the policy receives.

| Input | Action |
|---|---|
| `W` / `S` | Walk forward / back along the view direction |
| `A` / `D` | Strafe left / right |
| `Q` / `E` | Down / up |
| Drag | Aim — swing and tilt, without moving the eye point |
| `Shift` / `Alt` | 4× faster / 5× finer while held |
| Wheel | Set the walking speed |
| `Esc` | Back to the overview |

A crosshair marks the sensor centre and the panel reads out how far away whatever
is under it sits — the number to match against a tape measure on the real rig.
The inset in the corner shows the camera and its frustum from behind, so being
inside it does not cost you your bearings. **Level** takes roll out of a shot
that has drifted off horizontal; nothing else touches roll, so a deliberately
tilted mount survives being flown. `pos` still accepts typed values for an offset
measured off the real robot.

Cameras are not moved with the gizmo, and scale and *Drop to surface* do not
apply to them.

For DROID, `nv_franka_droid` carries two exterior cameras — `external_cam0_left`
and `external_cam1_left` — both 320×180 at 104° × 71.5° field of view.

### Which camera is `exterior_image_1_left`?

**Not something the sensor config records.** The evaluation stage reads two
camera *names* out of the **task** config and looks them up in the observation
dict, falling back to config order when a name is absent
([`1_eval_policy_og_scene.py:603`](../scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py#L603)):

```python
if base_camera_1_name in external_obs:      # s15_eval.base_camera_1_name
    exterior_image_1_left = that camera
elif len(external_cams) >= 1:
    exterior_image_1_left = external_cams[0]
```

So the same rig feeds the two policy inputs in whichever order the task asks for,
and the task configs here do **not** all agree. Eight of them name
`external_cam0_left` as camera 1; `real2sim_cfg_trash_can.yaml` swaps the pair —
which matters, because that is the config for a `droid_desk_put_away_trash`
evaluation. Aiming a camera without knowing which input it lands in is how a
correctly framed shot reaches the wrong policy input.

The editor resolves this from the task configs and prints it at startup:

```text
Cameras:   external_cam0_left -> exterior_image_1_left  <-- task configs disagree
Cameras:   external_cam1_left -> exterior_image_2_left  <-- task configs disagree
Cameras: task configs disagree on external_cam0_left, external_cam1_left — pass --task-cfg <name> to pin one
```

Pass `--task-cfg <stem>` to settle it against the task you are building for. The
preview badges then state the mapping plainly instead of flagging it amber:

```bash
mamba run -n simfoundry-editor python server.py --scene ... \
  --cameras nv_franka_droid --task-cfg real2sim_cfg_trash_can
```

An unknown `--task-cfg` is a startup error listing the configs that do bind
cameras, rather than a confident mapping derived from nothing.

### Feeding it to evaluation

```bash
bash scripts/pipeline/C_application/run.sh \
  --mode eval \
  --scene-name my_scene \
  -- s15_eval.external_sensors_cfg=droid_desk_mesh_cameras
```

Only `position` and `orientation` are ever rewritten; optics, modalities, prim
paths and resolution pass through untouched. The source config is never modified.

---

## Tasks

A task config says what the run is *for*: which objects are the subject, what
counts as success, and what sentence the policy is given. The editor edits them
because a layout and a task that disagree still run, still record video, and
still report numbers.

### Attaching a task config

The **Task** panel picks the config this scene is for. The dropdown offers what
the server associates with the scene; **Browse…** reaches any YAML in the
checkout. The choice is remembered per scene, and it is what
[Review & Export](#review--export) defaults to.

### Randomisation ranges

Per group, the panel edits `group_xyz_randomization` and
`group_z_rot_randomization` — how far each reset may move that group from where
you placed it. The numbers are **± half-widths**: `0.05` means ±5 cm, a 10 cm
span.

A blank field is not zero. It means the group has no range at all, which is a
different statement to write into the YAML — so blank fields stay blank until
you type into one. X, Y and Z are a single 3-vector, though, so filling one
fills its siblings with explicit zeros rather than letting the file and the
panel disagree.

Saving rewrites only those two maps and leaves the rest of the file — including
its comments — alone, and it quotes back the digest it read so an edit made
while the panel was open cannot be lost.

**Visualize Reset Ranges**, at the top of the Task section, draws the panel's numbers in
the viewport: for every object a group binds, a translucent box (one colour per
group) marking the ±X/±Y/±Z region a reset may move that object's origin into,
centred on where it stands now and following it as you move it. The boxes read
the fields live — unsaved edits show immediately — X/Y are dropped for a
predicate-placed group exactly as the panel disables them, and the ±Z rotation
range is not drawn. A view toggle only: the sensor previews never see the
boxes, and nothing is written.

### Generating a task

**Generate task** describes the scene to Gemini and asks for a config. It is
model output, so two things are checked before it can be saved.

**Could it run at all?** Every predicate has to name a state OmniGibson has,
every `group` and `other_group` has to appear in `semantic_group_mapping`, a
binary state needs its `other_group`, a boolean state's `value` has to be a
boolean — and the config has to state a goal, name a `type`, an `activity_name`
and a `termination_config`. Each of those last three is a crash: a missing
`type` raises while the environment loads, `activity_name` is a positional
argument of `PickPlaceTask`, and the eval stage writes `max_steps` straight into
`termination_config`.

> An empty goal is the quietest way to be wrong in this format. The termination
> condition is `MultiPredicate([])`, and `all([])` is `True`, so a config with no
> goal predicate reports **every episode successful on step 1**.

**Is it right about *this* scene?** Each group is resolved against the objects
actually present — nothing matched, or several matched where the config asserts
exactly one.

A config that cannot run is refused, naming what is wrong; you can save it
anyway, and the panel then calls it a **draft** rather than "Saved". A config
whose `type` is not `PickPlaceTask` is reported as *unjudged* rather than as
fine, because "we did not check this" and "we checked this and it is fine" are
not the same answer.

And the model can now decline. Asked for something these objects cannot do it
answers `cannot: <reason>` instead of inventing a task — "put the blue stapler
in the microwave", in a scene with neither, used to come back as a valid task
about an apple and a tray.

---

## Verifying an edited scene

Two different questions, two tools, in this order. Run both before an edited
scene is used for evaluation:

| | Question | Tool |
|---|---|---|
| 1 · **Settling** | Is anything floating or intersecting? | `settle.py` |
| 2 · **Parity** | Does OmniGibson load the settled result the way the browser showed it? | `parity_check.py` |

Settle first: parity measures a scene against what the JSON says, so running it
on a layout that has not been settled tells you the file is self-consistent
without telling you the props are resting on anything.

### settle.py — is anything floating?

The browser cannot tell an object resting on the desk from one floating 2 mm
above it. `settle.py` loads a saved scene in headless OmniGibson, steps physics,
and reports how far each object moved.

#### After every save

```bash
mamba run -n simfoundry-editor python server.py \
  --scene /absolute/path/to/my_scene_state_latest.json \
  --settle-after-save
```

Saves return immediately and the status line updates as the job runs, so editing
stays responsive while Isaac Sim boots. If *overwrite `_latest.json`* is ticked,
promotion is deferred until after settling, so `_latest` receives the verified
file rather than the hand-placed one.

#### One-off

```bash
mamba run -n simfoundry python \
  scripts/interactive/light_editor/settle.py \
  --scene /absolute/path/to/my_scene_state_light_edit_....json \
  [--steps 240] [--promote] [--gui] [--report out.json]
```

Exit status is non-zero if anything moved beyond `--tolerance`, so it can gate a
pipeline step. Output looks like:

```text
object                        root (m)  joint drift
----------------------------------------------------------------------------
robot0                          0.0000  -
black_trash_can_0               0.0000  -
blue_cup_0                      0.0200                                <-- unsettled
wooden_organizer_0              0.0000  drawer_1 0.0421 m             <-- unsettled
mesh_background_0               0.0000
```

**Root movement and joint drift are reported apart**, because "the cup rolled"
and "the drawer shut" are different problems. A scene saved with a drawer open
settles it closed while the object's base never moves at all — which used to
read as 0.0000 m and a clean promotion with the edit silently gone.

Like the parity check, it applies the scene's ground plane before stepping, and
says what height it used.

**Promotion is guarded.** `--promote` writes `_latest.json` only if nothing
moved, *and* only if the file still holds what it held when the settle started —
a settle takes minutes, and `_latest` is the file every downstream stage opens.
A refusal says which of the two it was.

### parity_check.py — does the simulator agree?

The browser writes JSON that *looks* right, but "structurally valid" is not "the
simulator does the same thing with it". This loads an export in OmniGibson and
checks that every object lands at the pose the JSON specified **before any
physics runs**, that the scene steps without exploding, and that each external
camera renders:

```bash
mamba run -n simfoundry python \
  scripts/interactive/light_editor/parity_check.py \
  --scene /absolute/path/to/my_scene_state_light_edit_....json \
  [--cameras <config name>] [--out DIR] [--report out.json] [--tolerance 1e-4]
```

```text
object                        delta (m)
robot0                         0.000000
black_trash_can_0              0.000000
blue_cup_0                     0.000000
mesh_background_0              0.000000
PARITY OK: 4 object(s) within 0.0001 m; scene stepped
```

Exit status is non-zero on any mismatch, so it can gate a pipeline step.

With `--cameras`, it also writes a rendered frame per camera into `--out`.
**Compare those against the browser's camera view** — that is the only way to
check the frustum convention, which cannot be verified from the browser alone.

It puts the scene's own `ground_plane_info` under the props first, and again
after each reset, through the same helper `PickPlaceTask` uses. `og.sim.restore()`
does not read that block, so without it the floor sits at z=0 whatever the scene
says — and a gate simulating against a different surface from the run it is
vouching for fails exports that would have worked. The report says what height it
applied.

> **A pass is necessary, not sufficient.** It restores a scene and moves the
> viewer camera; it does not build the evaluation `Environment` with the selected
> task and its `external_sensors`, which is the path
> `1_eval_policy_og_scene.py` actually takes.

### Or just run it and look

```bash
bash scripts/pipeline/C_application/run.sh --mode eval --scene-name my_scene \
  -- s15_eval.scene_json=/absolute/path/to/my_scene_state_light_edit_....json
```

---

## OmniGibson editor

The high-fidelity editor. Boots Isaac Sim, so expect ~90 seconds.

```bash
cd /path/to/simfoundry
./scripts/interactive/run_editor.sh

# another scene
SCENE_NAME=droid_desk_stack_dishware ./scripts/interactive/run_editor.sh

# build from pipeline outputs in Data/ instead of resuming a saved state
./scripts/interactive/run_editor.sh fresh
```

Explicitly:

```bash
cd /path/to/simfoundry/scripts/interactive
mamba run -n simfoundry python -u interactive_scene_editor.py \
  --scene_name droid_desk_put_away_trash \
  --load_scene /path/to/simfoundry/assets/scenes/droid_desk_put_away_trash/droid_desk_put_away_trash_scene_state_latest.json
```

`DISPLAY` must point at a live X session (often `:1` on a box you log into
remotely) — non-login shells often have it unset. `-u` keeps Python's prints
interleaved with Isaac Sim's logging when piping to a file.

Add `--debug_shell` to enable the `B` IPython shell. It is off by default because
it freezes rendering until you exit it.

### Swapping in generated cousins

Once `B_augmentation` has produced cousins, the editor can hot-swap between
combinations without restarting:

```bash
mamba run -n simfoundry python -u interactive_scene_editor.py \
  --scene_name my_scene --load_scene <path> \
  --cousins_combinations Data/my_scene/<...>/combinations.json \
  [--cousins_dataset custom-assets] [--cousins_swap_key H] [--cousins_settle_steps 30]
```

Press the swap key to advance to the next combination. Without
`--cousins_combinations` the feature is entirely inert.

Inputs come from the pipeline: `combinations.json` from B_augmentation stage 2,
and the `custom-assets` dataset from stage 5. Pre-flight checks both and names
those stages if either is missing.

**Note the default key `H` is also skybox rotation** — the editor prints a
warning when the swap key overrides an existing binding. Pass a different
`--cousins_swap_key` to keep both.

> **Unverified.** This was ported from `interactive_scene_editor_cousin_swap.py`,
> which is kept until someone runs the pipeline stages and confirms a real swap.

### Controls

**There is no full keymap here on purpose.** It is generated from one table,
[`editor_bindings.py`](../scripts/interactive/editor_bindings.py), which also
drives the key registration and the on-screen HUD legend, and
`print_controls()` prints it at startup — so the list on your terminal is the
list that is actually bound. This repo has already shipped wrong instructions
from a keymap written out by hand in several places; a copy in this file would
be one more.

The few worth knowing before you start:

| Keys | Action |
|---|---|
| `[` / `]` | Cycle selection |
| `ENTER` | Save scene state |
| `D` | Delete (soft — hidden and stripped on save, so `U` brings it back) |
| `U` | Undo |
| `SPACE` / `O` / `P` | Sim toggle / stop / play |
| `F2` / `F3` | Toggle the HUD / the selection outline |
| `ESC` | Exit |

Saving writes `<scene>_scene_state_<timestamp>.json` **and** overwrites
`_latest.json`.

The HUD shows a `LOAD ERRS n` row if anything failed to load — a missing USD used
to print one warning into thousands of lines of Isaac Sim output and then simply
be absent from the scene. The detail is in the terminal; the row exists so you
notice at all.

## Troubleshooting

**"omnigibson resolves outside this repo"** — the env's OmniGibson came from a
different checkout. Use the env whose `omnigibson` resolves inside this repo
(`SIMFOUNDRY_ENV=<env>` overrides which one `run_editor.sh` uses), and verify
with `python -c 'import omnigibson; print(omnigibson.__file__)'`.

**The OmniGibson editor looks hung with no output** — Python's prints are block
buffered when piped. Use `python -u`; `run_editor.sh` already does.

**Isaac Sim takes much longer than 90 s** — check for other instances:
`pgrep -af interactive_scene_editor`. Three at once on one GPU turned a 90 s load
into 142 s. Kill strays with `pkill -f interactive_scene_editor.py`.

**"cached manifest is for … not …; remove --no-extract"** — `--no-extract` reuses
an extraction belonging to a different scene. Drop the flag.

**`ModuleNotFoundError: pxr` in the light editor** — you are using the wrong
interpreter. Use the `simfoundry-editor` env. Isaac Sim's `pxr` is only importable
inside a running Kit app; the light editor needs standalone `usd-core`.

**Camera edits do not appear in eval** — check you passed the config name, not a
path: `s15_eval.external_sensors_cfg=droid_desk_mesh_cameras`.

**"the scene changed in another tab since this page loaded"** — another tab (or
another editor server) wrote this scene while you had edits pending. Your edits
are still on the page; copy anything you need out of it, then reload. The page
will not write again until you do, deliberately — adopting the newer revision
without its contents is how the *next* save would silently revert the other
tab's work.

**"… was changed by something outside this editor"** — the file this write was
about to replace no longer holds what this session last put there. Nothing was
written. Look at what is on disk before retrying; a second editor server on the
same scene is the usual cause.

**"<task> would not run"** on export — the task config cannot run anywhere, not
just here. The message names the first problem and the dialog lists the rest;
see [tasks](#tasks). Export anyway only if you mean it.

**A prop falls through the desk under physics** — the room is a Gaussian splat,
which has no colliders. Give the scene a [ground plane](#the-ground-plane) at
the height of the surface in the picture.

**Objects render but the background is missing or wrong** — object USDs are Z-up
while scanned mesh backgrounds are Y-up. No correction is applied, deliberately:
USD treats `upAxis` as advisory and does not rotate on reference, so raw geometry
matches OmniGibson. If a new background type appears wrong, this is the first
thing to check.

---

## Known quirks in the current scenes

Found while testing; neither is caused by the editors.

- `droid_desk_put_away_trash` has `blue_cup_0` about **2 cm above the desk** — it
  drops that far under physics even with no edits.
- The same scene puts the robot base at `z = -0.08` where every other
  `droid_desk_*` scene uses `-0.04` (still true as of this writing). Since
  cameras are robot-relative, framing shifts ~4 cm between that scene and the
  others sharing one config.

## See also

- [scripts/interactive/light_editor/README.md](../scripts/interactive/light_editor/README.md) —
  light editor internals: the saved-scene contract, how a save is compiled, what
  the guards actually check
- [AGENTS.md](../AGENTS.md) — the repo-wide guide: environments, the six
  consumers of the scene format, and the rules an edit must not break
