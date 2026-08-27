// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Browser-side scene editor. Geometry and transforms come from extract.py's
// manifest; edits go back through /api/save into the OmniGibson scene JSON.
//
// Coordinates are OmniGibson's throughout: Z-up, metres, quaternions in
// (x, y, z, w) order — which is also three.js's order. Nothing is converted on
// the way in or out, so what the panel shows is what lands in the JSON.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatCloud } from './splats.js';

const viewport = document.getElementById('viewport');
const statusEl = document.getElementById('status');

// --- how wide the panel is ---------------------------------------------------
// Applied at the very top of the module so the first resize() measures the
// right viewport width. Stored in localStorage (already scoped per origin):
// a property of the monitor, not of the scene.
const PANEL_KEY = 'simfoundry.light-editor.panel-width';
const PANEL_DEFAULT_W = 360;
// Below this the right-aligned transform fields clip a coordinate's minus sign.
const PANEL_MIN_W = 345;
const PANEL_MAX_W = 720;
// What the viewport keeps whatever the panel asks for; also keeps the camera
// preview panes drawable.
const VIEWPORT_MIN_W = 360;
// Panel floor when the window cannot hold PANEL_MIN_W and VIEWPORT_MIN_W at once.
const PANEL_HARD_MIN_W = 240;
const PANEL_DRAG_SLOP = 4;   // px of movement that stops a press being a click
// The grip is a flex item, so its width comes out of the viewport's share.
const PANEL_GRIP_W = document.getElementById('panel-grip').offsetWidth;

function panelLimit() {
  // Re-read rather than cached: the window can be resized under a stored width.
  const room = window.innerWidth - VIEWPORT_MIN_W - PANEL_GRIP_W;
  if (room >= PANEL_MIN_W) return Math.min(PANEL_MAX_W, room);
  // Neither reserve can be honoured: the panel gives way, down to a hard floor.
  return Math.max(PANEL_HARD_MIN_W, Math.max(0, room));
}

function applyPanelWidth(px) {
  // NaN would reach the stylesheet as `NaNpx`, which CSS ignores.
  const want = Number.isFinite(px) ? px : PANEL_DEFAULT_W;
  const clamped = Math.round(Math.min(Math.max(want, PANEL_MIN_W), panelLimit()));
  document.documentElement.style.setProperty('--panel-w', `${clamped}px`);
  return clamped;
}

function readPanelWidth() {
  try {
    const raw = JSON.parse(localStorage.getItem(PANEL_KEY) || 'null');
    return Number.isFinite(raw) ? raw : PANEL_DEFAULT_W;
  } catch {
    return PANEL_DEFAULT_W;   // private browsing, or a value from an older format
  }
}

function savePanelWidth() {
  try {
    // The width asked for, not the clamped one a narrow window allowed.
    localStorage.setItem(PANEL_KEY, JSON.stringify(panelWantW));
  } catch { /* the panel still resizes; it just forgets. Not worth a message. */ }
}

// Nothing else may be called from up here: `camera`, `renderer` and the PIP_*
// constants are still in their temporal dead zone.
let panelWantW = readPanelWidth();
let panelWidth = applyPanelWidth(panelWantW);

// --- how tall the object list is ---------------------------------------------
// Stored in localStorage for the same reason as the panel width.
const OBJLIST_KEY = 'simfoundry.light-editor.objlist-height';
const OBJLIST_DEFAULT_H = 220;
const OBJLIST_MIN_H = 120;
// Caps the drag so the action row below the list stays reachable.
const OBJLIST_MAX_H = 600;

function applyObjlistHeight(px) {
  const want = Number.isFinite(px) ? px : OBJLIST_DEFAULT_H;
  const clamped = Math.round(Math.min(Math.max(want, OBJLIST_MIN_H), OBJLIST_MAX_H));
  document.documentElement.style.setProperty('--objlist-h', `${clamped}px`);
  return clamped;
}

function readObjlistHeight() {
  try {
    const raw = JSON.parse(localStorage.getItem(OBJLIST_KEY) || 'null');
    return Number.isFinite(raw) ? raw : OBJLIST_DEFAULT_H;
  } catch {
    return OBJLIST_DEFAULT_H;   // private browsing, or a value from an older format
  }
}

function saveObjlistHeight() {
  try {
    localStorage.setItem(OBJLIST_KEY, JSON.stringify(objlistHeight));
  } catch { /* the list still resizes; it just forgets. Not worth a message. */ }
}

let objlistHeight = applyObjlistHeight(readObjlistHeight());

// Registered before the resize listener further down so it runs first and
// resize() measures the re-clamped width.
window.addEventListener('resize', () => { panelWidth = applyPanelWidth(panelWantW); });

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 500);
camera.up.set(0, 0, 1);            // Z-up, matching OmniGibson
camera.position.set(1.6, -1.6, 1.2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.1;

const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setSpace('world');
scene.add(gizmo);
// Dragging the gizmo must not also orbit the camera.
gizmo.addEventListener('dragging-changed', (e) => {
  orbit.enabled = !e.value;
  // Latched for the pick handler: TransformControls clears `dragging` before
  // this page's pointerup, so a press-and-release on a handle would otherwise
  // read as a click on empty air.
  if (e.value) gizmoInteracted = true;
  // The table marker is editor furniture, not a scene object: it takes no undo
  // slot and none of the scale guards below.
  if (gizmo.object === tableMarker) return;
  if (e.value) beginGizmoDrag(); else endGizmoDrag();
});
gizmo.addEventListener('objectChange', () => {
  if (gizmo.object === tableMarker) { refreshTableReadout(); return; }
  // A group drag holds the pivot; its delta goes out to every member.
  if (gizmo.object === pivot) { applyPivotDelta(); return; }
  if (!selected) return;
  if (scaleLocked && gizmo.getMode() === 'scale') enforceUniformScale(selected.group);
  const scale = selected.group.scale;
  if (![scale.x, scale.y, scale.z].every((value) => Number.isFinite(value) && value > 0)) {
    selected.group.scale.copy(selected.lastValidScale);
    setStatus('Scale must stay positive on every axis.', 'err');
    refreshReadout();
    return;
  }
  selected.lastValidScale.copy(scale);
  markDirty();
  refreshReadout();
});

scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(2, -3, 4);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.5);
fill.position.set(-3, 2, 1);
scene.add(fill);

// Editor furniture (grid, axes, camera bodies, frustum lines) lives on an
// overlay layer so sensor views draw only real scene content.
const LAYER_CONTENT = 0;           // scene geometry: what a sensor actually sees
const LAYER_OVERLAY = 1;           // editor-only furniture

/** Move an object and everything under it onto the overlay layer. */
function markAsOverlay(object3d) {
  object3d.traverse((node) => node.layers.set(LAYER_OVERLAY));
  return object3d;
}

// The overview camera is the editor's own view and wants both.
camera.layers.enable(LAYER_OVERLAY);

const grid = new THREE.GridHelper(4, 40, 0x3a4150, 0x252a33);
grid.rotation.x = Math.PI / 2;     // GridHelper is XZ by default; make it XY
scene.add(markAsOverlay(grid));
// The origin cross toggles with the grid: it is the grid's zero.
const originAxes = markAsOverlay(new THREE.AxesHelper(0.25));
scene.add(originAxes);

// The gizmo is editor furniture too. Its own Raycaster tests layer 0 only, so
// it must be told about the overlay layer or the handles become unpickable.
gizmo.getRaycaster().layers.enable(LAYER_OVERLAY);
markAsOverlay(gizmo);

const objects = new Map();         // name -> { name, group, entry, initial }

// --- selection -------------------------------------------------------------
// `selection` is the set the group operations act on; `selected` is its most
// recent member and what every single-object panel control speaks for. Set
// iteration order is insertion order, so "the last one clicked" is defined.
const selection = new Set();
let selected = null;

// --- the group pivot -------------------------------------------------------
// An empty Object3D parked at the middle of the selection for TransformControls
// to hold; its frame-to-frame delta is handed out to every member. Nothing is
// ever re-parented into it: saved poses are world poses, so every record must
// stay a direct child of the scene root.
const pivot = new THREE.Object3D();
pivot.name = '__multiPivot';
// Not markAsOverlay (which traverses): the pivot must never take a child.
pivot.layers.set(LAYER_OVERLAY);
pivot.raycast = () => {};
scene.add(pivot);

const _lcPos = new THREE.Vector3();
const _lcQuat = new THREE.Quaternion();
const _lcScale = new THREE.Vector3();
const _lcBox = new THREE.Box3();

/**
 * The world box of what an object actually draws, measured off its vertices
 * (setFromObject's precise mode) rather than around transformed local boxes,
 * so a tilted mesh's box faces touch the geometry. Use wherever the faces are
 * the answer — resting on a surface, keeping a base planted, overlap tests.
 */
function contactBox(object, out = new THREE.Box3()) {
  return out.setFromObject(object, true);
}

/**
 * The centre of a record's bounding box measured in its own frame, cached.
 * Not the origin (which is wherever the asset author put it), and measured
 * with the transform set aside so the point moves with the object.
 */
function localCentre(rec) {
  if (rec.localCentre) return rec.localCentre;
  const g = rec.group;
  _lcPos.copy(g.position); _lcQuat.copy(g.quaternion); _lcScale.copy(g.scale);
  try {
    g.position.set(0, 0, 0);
    g.quaternion.identity();
    g.scale.set(1, 1, 1);
    _lcBox.setFromObject(g);
    // An empty box means a record whose visual proxy failed to load. Its origin
    // is the only point it has.
    rec.localCentre = _lcBox.isEmpty()
      ? new THREE.Vector3()
      : _lcBox.getCenter(new THREE.Vector3());
  } finally {
    // Restore even on a throw, or the object is left at the origin, unscaled.
    g.position.copy(_lcPos); g.quaternion.copy(_lcQuat); g.scale.copy(_lcScale);
    g.updateWorldMatrix(false, true);
  }
  return rec.localCentre;
}

/** Where a member's box centre sits in the world, at its current transform. */
function memberCentre(rec, out) {
  return out.copy(localCentre(rec))
    .multiply(rec.group.scale)
    .applyQuaternion(rec.group.quaternion)
    .add(rec.group.position);
}

const _memberCentre = new THREE.Vector3();

/**
 * Park the pivot at the mean of the members' box centres, unrotated, unscaled.
 * The mean is an exact fixed point of the group operations, so the pivot is
 * derived on demand rather than stored — nothing to go stale across undo.
 */
function rebuildPivot() {
  const recs = selectionRecords();
  if (recs.length < 2) return;
  pivot.position.set(0, 0, 0);
  for (const rec of recs) pivot.position.add(memberCentre(rec, _memberCentre));
  pivot.position.divideScalar(recs.length);
  pivot.quaternion.identity();
  pivot.scale.set(1, 1, 1);
  pivot.updateMatrixWorld(true);
  refreshPivotNote();
}

/** Say where the pivot is, because rotate and scale turn about a point the
 *  gizmo draws but does not name. */
function refreshPivotNote() {
  const note = document.getElementById('pivot-note');
  if (!note) return;
  const many = selection.size > 1;
  note.style.display = many ? '' : 'none';
  if (!many) return;
  const at = pivot.position.toArray().map((v) => v.toFixed(3)).join('  ');
  note.textContent = `pivot ${at} — rotate and scale turn about this point`;
}

const dirty = new Set();
// Declared up here rather than beside the physics panel: `pendingChanges`
// reads it during boot.
const physicsDirty = new Set();
// Same, for an articulated prop's joints; separate because the two panels
// revert independently.
const jointsDirty = new Set();
let activeManifest = null;

// --- ground-plane state, declared here rather than beside the rest of it -----
// The panel lives at the bottom of the file, but `pendingChanges` reads these
// up here, so they are declared early and only assigned down there.
let groundState = null;             // the last /api/ground_plane response
let groundSaved = null;             // the plane the saved scene has, or null
let groundLive = null;              // the plane the panel is describing, or null
let groundDirty = false;
let groundHiddenForView = true;     // a view setting; see m-floor
// The Ground plane section's collapse handle, assigned once /api/ground_plane
// answers.
let groundSection = null;

const FLAT_COLORS = { robot: 0x9aa3b0, background: 0x6b7280, object: 0x6fa8dc };

function setStatus(msg, cls = '') {
  statusEl.textContent = msg;
  statusEl.className = cls;
}

// Counts the unsaved edits a scene save would write, for the Save button label.
function pendingChanges() {
  let moved = 0, added = 0, removed = 0, physics = 0;
  for (const rec of objects.values()) {
    if (rec.isCamera || !rec.entry.posable) continue;
    if (rec.present === false) { if (rec.basePresent !== false) removed++; continue; }
    if (rec.basePresent === false) { added++; continue; }
    if (dirty.has(rec.name)) moved++;
    // A physics change has nothing to see in the viewport, so it reaches the
    // Save button on its own; counted apart from `moved`.
    else if (physicsDirty.has(rec.name) || jointsDirty.has(rec.name)) physics++;
  }
  // The ground plane is written by the same save, so it counts too.
  const ground = groundDirty ? 1 : 0;
  return { moved, added, removed, physics,
           total: moved + added + removed + physics + ground };
}

/**
 * Whether anything a scene save would write is currently unsaved. The one
 * predicate every caller shares, so all five kinds of scene edit count.
 * Cameras and task ranges are separate files with separate saves and are
 * deliberately not included.
 */
function hasUnsavedSceneEdits() {
  return pendingChanges().total > 0;
}

function refreshSaveButton() {
  const button = document.getElementById('btn-save');
  if (!button) return;
  const { moved, added, removed, physics } = pendingChanges();
  // Once the server has moved to another scene this page cannot write anywhere.
  button.disabled = !hasUnsavedSceneEdits() || sceneMoved || sceneStale;
  const parts = [];
  if (moved) parts.push(`${moved} moved`);
  if (added) parts.push(`${added} added`);
  if (removed) parts.push(`${removed} removed`);
  if (physics) parts.push(`${physics} retuned`);
  button.textContent = parts.length
    ? `Save scene JSON (${parts.join(', ')})`
    : 'Save scene JSON';
  refreshPromoteNote('promote', 'promote-note');
}

// One list rebuild for the whole set: a group drag runs this every frame.
function markDirtyAll(recs) {
  let cameras = false;
  for (const rec of recs) {
    if (rec.isCamera) { cameraDirty.add(rec.name); cameras = true; }
    else dirty.add(rec.name);
  }
  if (cameras) document.getElementById('btn-save-cameras').disabled = sceneMoved;
  refreshSaveButton();
  renderList();
}

function markDirty(rec = selected) {
  if (rec) markDirtyAll([rec]);
}

// --- undo / redo -----------------------------------------------------------
// One entry per user-visible operation, holding the transform as it was before
// it. A drag snapshots once on start, not per objectChange frame.

const undoStack = [];
const redoStack = [];
const MAX_HISTORY = 100;

// Per-process secret injected into index.html by the server; required on every
// mutation so an unrelated page cannot write here (DNS rebinding).
const EDITOR_TOKEN =
  document.querySelector('meta[name="editor-token"]')?.content || '';

// One write at a time per endpoint. Without this, two responses can land out of
// order and the older one adopts a stale baseline.
let saveInFlight = false;
let cameraSaveInFlight = false;
// Revision of the camera config this page loaded; echoed back on save so the
// server can refuse a stale client.
let cameraRevision = 0;

function snapshot(rec) {
  return {
    name: rec.name,
    position: rec.group.position.toArray(),
    orientation: rec.group.quaternion.toArray(),
    scale: rec.group.scale.toArray(),
    // Presence rides in the snapshot so add and remove unwind through the same
    // undo stack.
    present: rec.present !== false,
  };
}

// --- presence --------------------------------------------------------------
// A removed object stays in `objects` with present=false: undo can bring it
// back, and the save payload names it so the server can tell a deletion from a
// dropped record.

// Whether a record is drawn at all: present in the scene AND not hidden for
// viewing. The two flags are deliberately separate.
function updateVisibility(rec) {
  rec.group.visible = rec.present !== false && !rec.hiddenForView;
}

/** The eye icon's own toggle — hides one record from the viewport without
    touching selection, presence, or anything a save would write. */
function toggleRecordHidden(rec) {
  rec.hiddenForView = !rec.hiddenForView;
  updateVisibility(rec);
  // An invisible object must not stay selected under the gizmo.
  if (rec.hiddenForView && selection.has(rec)) {
    selection.delete(rec);
    setSelection(selectionRecords());
  }
  refreshBackgroundToggle();
  renderList();
}

/**
 * Light the Background button iff a background is being drawn. Two controls
 * write `hiddenForView` on the same records, so the lit state is read back
 * off them. Lit means shown.
 */
function refreshBackgroundToggle() {
  const backgrounds = [...objects.values()].filter((r) => r.entry.kind === 'background');
  if (!backgrounds.length) return;
  document.getElementById('m-bg')
    .classList.toggle('on', backgrounds.some((r) => !r.hiddenForView));
}

function setPresent(rec, present) {
  rec.present = present;
  updateVisibility(rec);
  // Only this record leaves the selection, so a group removal keeps going.
  if (!present && selection.has(rec)) {
    selection.delete(rec);
    setSelection(selectionRecords());
  }
  if (!present && anchorName === rec.name) setAnchor(heuristicAnchor());
  invalidateFlyTargets();
  refreshDirty(rec);
}

// Every record the scene currently contains (not "every record ever loaded").
function liveRecords() {
  return [...objects.values()].filter((r) => r.present !== false);
}

function removedRecords() {
  return [...objects.values()].filter((r) => r.present === false);
}

// Accepts one record or many: an arrange must unwind in a single Ctrl+Z.
function pushUndo(recs, label) {
  const list = (Array.isArray(recs) ? recs : [recs]).filter(Boolean);
  if (list.length === 0) return;
  undoStack.push({ label, snaps: list.map(snapshot) });
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  // A fresh edit invalidates any forward history.
  redoStack.length = 0;
  updateHistoryButtons();
}

const CLOSE = 1e-9;
const arraysClose = (a, b) => a.length === b.length
  && a.every((v, i) => Math.abs(v - b[i]) <= CLOSE);

// Dirtiness is recomputed from the current transform rather than stored, so
// history stays correct across a save (which moves the baseline).
function refreshDirty(rec) {
  const g = rec.group;
  const present = rec.present !== false;
  // Presence is compared against the same baseline as the transform: a fresh
  // import is a change by existing; one imported then undone is not.
  const clean = present === (rec.basePresent !== false) && (!present
    || (arraysClose(g.position.toArray(), rec.initial.position)
      && arraysClose(g.quaternion.toArray(), rec.initial.orientation)
      && arraysClose(g.scale.toArray(), rec.initial.scale)));
  const set = rec.isCamera ? cameraDirty : dirty;
  if (clean) set.delete(rec.name); else set.add(rec.name);
  if (rec.isCamera) {
    const btn = document.getElementById('btn-save-cameras');
    if (btn) btn.disabled = cameraDirty.size === 0 || sceneMoved;
  } else {
    refreshSaveButton();
  }
}

function applySnapshot(snap) {
  const rec = objects.get(snap.name);
  if (!rec) return null;
  rec.group.position.fromArray(snap.position);
  rec.group.quaternion.fromArray(snap.orientation);
  rec.group.scale.fromArray(snap.scale);
  rec.lastValidScale.copy(rec.group.scale);
  if ((rec.present !== false) !== snap.present) setPresent(rec, snap.present);
  refreshDirty(rec);
  // Keep the flight state in sync when the flown camera's pose is restored.
  if (rec === flying) syncFlyFromRecord();
  return rec;
}

function stepHistory(from, to, verb) {
  // Not mid-drag: the next pointermove would overwrite what undo restored.
  if (gizmo.dragging) return;
  // Close any open flight or arrow-key edit burst: their entries just changed
  // stacks, and reusing an open burst would leave edits nothing can unwind.
  closeFlyEdit();
  closeEditBurst();
  const entry = from.pop();
  if (!entry) return;
  const recs = entry.snaps.map((s) => objects.get(s.name)).filter(Boolean);
  if (recs.length === 0) { updateHistoryButtons(); return; }
  // Capture the current state on the opposite stack before overwriting it.
  to.push({ label: entry.label, snaps: recs.map(snapshot) });
  entry.snaps.forEach(applySnapshot);

  // Undo restores state; it must not change editor mode. Select the affected
  // record only when it costs nothing: never while flying, never across tabs,
  // never by entering a camera, and never collapsing a multi-selection.
  const only = recs.length === 1 && selection.size <= 1 && recs[0].present !== false
    ? recs[0] : null;
  let elsewhere = null;
  if (only && !flying) {
    if (!!only.isCamera === (activeTab === 'cameras')) {
      if (selected !== only) select(only, { enterCamera: false });
    } else {
      elsewhere = only.isCamera ? 'Cameras' : 'Objects';
    }
  }

  // The members are back where they were, so the middle of them has moved.
  rebuildPivot();
  renderList();
  refreshReadout();
  updateHistoryButtons();
  const what = recs.length === 1 ? recs[0].name : `${recs.length} objects`;
  setStatus(`${verb} ${entry.label} on ${what}.`
    + (elsewhere ? ` It is on the ${elsewhere} tab.` : ''));
}

const undo = () => stepHistory(undoStack, redoStack, 'Undid');
const redo = () => stepHistory(redoStack, undoStack, 'Redid');

function updateHistoryButtons() {
  document.getElementById('btn-undo').disabled = undoStack.length === 0;
  document.getElementById('btn-redo').disabled = redoStack.length === 0;
  const info = document.getElementById('histinfo');
  if (info) info.textContent = undoStack.length ? `${undoStack.length} step(s)` : '';
}

document.getElementById('btn-undo').onclick = undo;
document.getElementById('btn-redo').onclick = redo;

const gltfLoader = new GLTFLoader();

// Build one scene record from a manifest entry and wait for its proxy to load.
// Shared by the initial load and by imports.
async function instantiate(entry) {
  // The outer group carries the authored transform (what the gizmo moves and
  // what is exported); the inner group holds visual-only fix-ups.
  const group = new THREE.Group();
  group.name = entry.name;
  group.position.fromArray(entry.position);
  group.quaternion.fromArray(entry.orientation);   // (x, y, z, w)
  group.scale.fromArray(entry.scale);

  const inner = new THREE.Group();
  // No upAxis correction is applied (objects are Z-up, the scanned mesh
  // background Y-up): USD treats upAxis as advisory and OmniGibson places the
  // raw geometry, so a viewer-only rotation here would corrupt editing.
  group.add(inner);
  // Not added to the scene until the proxy is in: an empty group would already
  // be pickable and framed.
  const rec = {
    name: entry.name,
    group,
    entry,
    // `removed` marks a manifest entry the scene no longer has; the record must
    // still exist (saves send complete snapshots) but come back absent.
    present: !entry.removed,
    added: !!entry.added,
    // Whether the last scene this editor wrote contains it. An import does not
    // until it is saved, which is what makes it show as a pending change.
    basePresent: !entry.added && !entry.removed,
    loadError: entry.error || null,
    lastValidScale: group.scale.clone(),
    initial: {
      position: entry.position.slice(),
      orientation: entry.orientation.slice(),
      scale: entry.scale.slice(),
    },
  };

  // A Gaussian-splat room arrives as a `.splat` rather than a `.glb`; it goes
  // under the authored transform like any mesh proxy so the rest of the editor
  // treats it uniformly.
  if (entry.splat && entry.status !== 'error') {
    try {
      rec.splat = await SplatCloud.load(`./data/${entry.splat}`);
      rec.splat.object.userData.owner = entry.name;
      inner.add(rec.splat.object);
    } catch (err) {
      rec.loadError = err.message;
      console.warn(`failed to load ${entry.splat}`, err);
    }
    publish(rec);
    return rec;
  }

  if (!entry.glb || entry.status === 'error') {
    rec.loadError = rec.loadError || 'no visual proxy was extracted';
    publish(rec);
    return rec;
  }

  try {
    const gltf = await gltfLoader.loadAsync(`./data/${entry.glb}`);
    gltf.scene.traverse((child) => {
      if (!child.isMesh) return;
      child.userData.owner = entry.name;
      if (!entry.textured) {
        child.material = new THREE.MeshStandardMaterial({
          color: FLAT_COLORS[entry.kind] ?? 0x6fa8dc,
          roughness: 0.8, metalness: 0.05,
        });
      } else if (child.material) {
        // The texture already carries the colour; keep its multiplier neutral.
        if (child.material.color) child.material.color.setHex(0xffffff);
        child.material.roughness = 0.85;
        child.material.metalness = 0.0;
      }
    });
    inner.add(gltf.scene);
  } catch (err) {
    rec.loadError = err.message;
    console.warn(`failed to load ${entry.glb}`, err);
  }
  publish(rec);
  return rec;
}

// Make a record part of the editor, all at once, after its geometry arrived.
function publish(rec) {
  scene.add(rec.group);
  objects.set(rec.name, rec);
  // A record that arrives already removed must not be drawn.
  updateVisibility(rec);
  // Fly targets are cached per flight; a mid-flight import must invalidate them.
  invalidateFlyTargets();
}

/**
 * The loading line: names what is coming (gaussian count and megabytes for a
 * splat room) and counts proxies off as they land.
 *
 * @param {?number} landed How many proxies are in, or null before any are.
 */
function loadingLine(landed) {
  const el = document.getElementById('loading');
  if (!el) return;
  const entries = (activeManifest && activeManifest.objects) || [];
  const splat = entries.find((entry) => entry.splat);
  const what = splat
    ? `a Gaussian-splat room: ${(splat.splatCount || 0).toLocaleString()} gaussians, `
      + `${Math.max(1, Math.round((splat.splatBytes || 0) / 1e6))} MB`
    : `${entries.length} object(s)`;
  el.textContent = landed === null
    ? `Loading ${what}…`
    : `Loading ${what} — ${landed}/${entries.length} in.`;
}

async function boot() {
  try {
    const res = await fetch('./data/manifest.json');
    if (!res.ok) throw new Error(`manifest.json -> HTTP ${res.status}`);
    activeManifest = await res.json();
  } catch (err) {
    document.getElementById('loading').remove();
    setStatus(`No manifest — run extract.py first. (${err.message})`, 'err');
    return;
  }

  document.getElementById('scenename').textContent = activeManifest.scene_json;
  loadingLine(null);
  loadRecentScenes();
  initTaskSection();
  const failures = [];

  let landed = 0;
  const loads = activeManifest.objects.map(
    (entry) => instantiate(entry).catch((err) => {
      failures.push(`${entry.name}: ${err.message}`);
    }).finally(() => { landed += 1; loadingLine(landed); }),
  );

  await Promise.all(loads);
  for (const rec of objects.values()) {
    if (rec.loadError) failures.push(`${rec.name}: ${rec.loadError}`);
  }

  await loadCameras();
  await loadGroundPlane();

  // Seed the anchor so auto-arrange is usable without a setup step.
  setAnchor(heuristicAnchor());

  document.getElementById('loading').remove();
  renderList();
  frameScene();
  updateHint();
  maybeShowTour();
  const loaded = failures.length
    ? `${objects.size - failures.length}/${objects.size} visual proxies loaded. Missing items are marked.`
    : `${objects.size} objects loaded.`;
  if (failures.length) {
    setStatus(loaded, 'err');
  } else if (statusEl.className === 'err') {
    // A warning raised while loading (e.g. a splat room with no ground plane)
    // must not be erased; the count rides behind it.
    setStatus(`${statusEl.textContent} ${loaded}`, 'err');
  } else {
    setStatus(loaded);
  }
  // Deliberately not awaited into the ready flag: the editor must be usable
  // whether or not the launcher's room choice lands.
  applyChosenRoom();
}

// --- connection ------------------------------------------------------------
// The page cannot tell that the server was stopped — fetches just start
// failing. A persistent banner says "the server is not running".

// The scene revision this page is working from. Bumped by the server on every
// accepted mutation; a stale one is refused rather than overwriting another tab.
let sceneRevision = 0;
let serverDown = false;

function setServerDown(down, detail = '') {
  if (down === serverDown) return;
  serverDown = down;
  const banner = document.getElementById('conn-banner');
  if (!banner) return;
  banner.hidden = !down;
  if (down) {
    banner.textContent = `Lost contact with the editor server${detail ? ` (${detail})` : ''}. `
      + 'Your edits are still on this page, but reloading discards them — nothing '
      + 'about them is written down anywhere else. Restart the server and this '
      + 'page reconnects within a few seconds; then press Save.';
  }
}

// A revision this page has not caught up with, and cannot write under. Never
// adopt a revision without adopting the content it names: with nothing unsaved,
// rehydrate from /api/scene_state (content and revision together); with unsaved
// work, latch read-only until reload.
let sceneStale = false;
// Resolves once the scene's records exist (even after a failed load), created
// at module scope so an early heartbeat has something to wait on.
let bootDone;
const bootReady = new Promise((resolve) => { bootDone = resolve; });
// One catch-up at a time: two interleaved rehydrations would half-apply each
// other's state.
let catchingUp = false;
const STALE_MESSAGE = 'Another tab wrote this scene, so this page can no longer save. '
  + 'Your edits are still here — copy anything you need, then reload.';

/** Latch this page read-only against the scene, and say why. */
function setSceneStale(detail = '') {
  if (sceneStale) return;
  sceneStale = true;
  const banner = document.getElementById('conn-banner');
  if (banner) {
    banner.hidden = false;
    banner.textContent = `${detail ? `${detail} ` : ''}${STALE_MESSAGE}`;
  }
  // The Save buttons would 409 anyway; disabling them makes it visible sooner.
  // Cameras are a separate file with a separate revision, so left alone.
  for (const id of ['btn-save', 'btn-review']) {
    const button = document.getElementById(id);
    if (button) { button.disabled = true; button.title = STALE_MESSAGE; }
  }
  setStatus(STALE_MESSAGE, 'err');
}

/**
 * Take the server's current scene state, content and revision together. Only
 * called with nothing unsaved here, so every baseline moves to the server's.
 * Refuses rather than half-applies when the object sets disagree (this page
 * would have no proxy for a new object); a reload is the only answer there.
 *
 * @returns {Promise<boolean>} Whether the page caught up.
 */
async function rehydrateScene() {
  const res = await fetch('/api/scene_state', { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const body = await res.json();

  const mine = [...objects.values()]
    .filter((r) => !r.isCamera && r.entry.posable).map((r) => r.name);
  const theirs = Object.keys(body.objects || {});
  const unknown = theirs.filter((n) => !objects.has(n));
  const vanished = mine.filter((n) => !(n in (body.objects || {})));
  if (unknown.length || vanished.length) {
    throw new Error(`the scene now has a different set of objects (${
      [...unknown.map((n) => `+${n}`), ...vanished.map((n) => `-${n}`)].join(', ')})`);
  }

  for (const [name, state] of Object.entries(body.objects)) {
    const rec = objects.get(name);
    rec.group.position.fromArray(state.position);
    rec.group.quaternion.fromArray(state.orientation);
    if (rec.entry.scalable) rec.group.scale.fromArray(state.scale);
    rec.lastValidScale.copy(rec.group.scale);
    const present = state.present !== false;
    if ((rec.present !== false) !== present) setPresent(rec, present);
    rec.basePresent = present;
    rec.initial = {
      position: rec.group.position.toArray(),
      orientation: rec.group.quaternion.toArray(),
      scale: rec.group.scale.toArray(),
    };
    refreshDirty(rec);
    if (hasPhysics(rec)) {
      const live = physicsOf(rec);
      live.mass = state.mass != null ? state.mass : live.authoredMass;
      live.friction = state.friction;
      rec.physicsSaved = { mass: live.mass, friction: live.friction };
      markPhysicsDirty(rec);
    }
    if (hasJoints(rec)) {
      const live = jointsOf(rec);
      live.values = (state.joint_values || []).slice();
      live.limits = structuredClone(state.joint_limits || {});
      live.addressable = live.values.length === live.list.length;
      rec.jointsSaved = {
        values: live.values.slice(), limits: structuredClone(live.limits),
      };
      markJointsDirty(rec);
    }
  }

  adoptSavedGround(body.ground_plane);
  setGround(groundSaved === null ? null
    : { ...groundSaved, orientation: groundSaved.orientation.slice() });

  // Every history entry now describes a pose that is no longer on screen.
  undoStack.length = 0;
  redoStack.length = 0;
  updateHistoryButtons();

  sceneRevision = body.scene_revision;
  renderList();
  refreshPhysics();
  refreshJoints();
  refreshSaveButton();
  return true;
}

/** Catch up with a revision this page has not seen, or latch if it cannot. */
async function catchUpTo(revision) {
  if (sceneStale || catchingUp || revision === sceneRevision) return;
  catchingUp = true;
  try {
    // Rehydrating before boot finished would find no objects and latch the
    // page for good.
    await bootReady;
    if (sceneStale || revision === sceneRevision) return;
    if (hasUnsavedSceneEdits()) {
      setSceneStale('Another tab saved this scene while you had unsaved edits.');
      return;
    }
    await rehydrateScene();
    setStatus('Caught up with a save made in another tab.');
  } catch (err) {
    setSceneStale(`Could not catch up with the other tab's save (${err.message}).`);
  } finally {
    catchingUp = false;
  }
}

async function heartbeat() {
  try {
    const res = await fetch('/api/session', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    setServerDown(false);
    // A changed session revision means the server moved to another scene, and
    // this page's records come from a manifest it no longer serves. Checked
    // first: it subsumes every revision question below.
    if (sessionRevision === null) sessionRevision = body.session_revision;
    else if (body.session_revision !== sessionRevision) {
      setSceneMoved(body.scene_name, body.scene);
      return;
    }
    if (!sceneRevisionSeeded) {
      sceneRevisionSeeded = true;
      // Revision zero means nothing has been written since this scene was
      // bound, so adopting the number alone is safe; otherwise the content has
      // to come with the number.
      sceneRevision = 0;
      if (body.scene_revision !== 0) await catchUpTo(body.scene_revision);
      else sceneRevision = body.scene_revision;
      return;
    }
    await catchUpTo(body.scene_revision);
  } catch (err) {
    setServerDown(true, err.message);
  }
}

// The scene binding this page loaded against; null until the first beat so a
// page starting mid-switch adopts what it finds.
let sessionRevision = null;
let sceneMoved = false;

/** Say that the server is now on a different scene, and stop pretending. */
function setSceneMoved(name, path) {
  if (sceneMoved) return;
  sceneMoved = true;
  const banner = document.getElementById('conn-banner');
  if (banner) {
    banner.hidden = false;
    banner.textContent = `The editor was pointed at ${name || 'another scene'}`
      + `${path ? ` (${path.split('/').pop()})` : ''}. This page is still showing the `
      + 'previous one and can no longer save — reload to follow it.';
  }
  // The Save buttons would 409 anyway; disabling them makes it visible sooner.
  for (const id of ['btn-save', 'btn-save-cameras']) {
    const button = document.getElementById(id);
    if (button) { button.disabled = true; button.title = 'The server moved to another scene; reload.'; }
  }
  setStatus('The editor moved to another scene. Reload this page to follow it.', 'err');
}

let sceneRevisionSeeded = false;
// Beat once immediately: until the first one lands the page holds revision 0
// and any save before then would be refused as stale.
const heartbeatReady = heartbeat();
setInterval(heartbeat, 5000);

// --- adding and removing objects -------------------------------------------
// Importing needs the server (copy into the scene directory + glTF proxy);
// removing is a flag here and a name in the save payload, so both are undoable.

let library = null;          // discovered assets, fetched on first use
let libraryFilter = '';
let addInFlight = false;

// Where a newly imported object lands: on the anchor's support surface, offset
// enough not to be hidden inside whatever is already there.
function spawnPose() {
  const anchor = objects.get(anchorName);
  const reference = anchor && anchor.present !== false
    ? anchor
    : liveRecords().find((r) => r.entry.editable && !r.isCamera);
  if (!reference) return { position: [0, 0, 0.9], orientation: [0, 0, 0, 1] };

  const box = new THREE.Box3().setFromObject(reference.group);
  const centre = box.getCenter(new THREE.Vector3());
  // Fan successive imports so the second is not invisible inside the first.
  const n = [...objects.values()].filter((r) => r.added).length;
  const angle = n * 2.399963;                    // golden angle, so no two coincide
  return {
    position: [
      centre.x + Math.cos(angle) * 0.22,
      centre.y + Math.sin(angle) * 0.22,
      box.min.z,
    ],
    orientation: [0, 0, 0, 1],
  };
}

async function addAsset(key, { scale = [1, 1, 1], label = 'add object', pose = null } = {}) {
  if (addInFlight) { setStatus('An import is already in progress…'); return null; }
  // An explicit pose is how Duplicate puts the copy beside its original;
  // otherwise fan around the anchor.
  pose = pose || spawnPose();
  addInFlight = true;
  setStatus('Importing…');
  try {
    const res = await fetch('/api/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      // Mesh options ride along; raw library meshes are converted on import.
      body: JSON.stringify({ key, scale, ...pose, ...importOptions() }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    const rec = await instantiate(body.object);
    rec.added = true;
    // Pushed *before* the object is marked present, so undo restores the
    // absent state and redo brings it back — the same mechanism a removal uses.
    undoStack.push({ label, snaps: [{ ...snapshot(rec), present: false }] });
    if (undoStack.length > MAX_HISTORY) undoStack.shift();
    redoStack.length = 0;
    updateHistoryButtons();
    setPresent(rec, true);
    select(rec);
    renderList();
    refreshReadout();
    showImportReport(body.report, rec.name);
    revealImport(rec);
    setStatus(`Added ${rec.name}. Move it into place, then Save.`, 'ok');
    return rec;
  } catch (err) {
    setStatus(`Add failed: ${err.message}`, 'err');
    return null;
  } finally {
    addInFlight = false;
  }
}

// Frame and outline a fresh import, which regularly lands off-screen.
let selectionOutline = null;

function revealImport(rec) {
  if (!rec || rec.present === false) return;
  frameSelection();
  flashOutline(rec);
}

function flashOutline(rec) {
  if (selectionOutline) { scene.remove(selectionOutline); selectionOutline = null; }
  const box = contactBox(rec.group);
  if (box.isEmpty()) return;
  selectionOutline = new THREE.Box3Helper(box, 0x76b900);
  // Overlay layer: drawn over geometry, excluded from the sensor previews.
  markAsOverlay(selectionOutline);
  scene.add(selectionOutline);
  const mine = selectionOutline;
  setTimeout(() => {
    if (selectionOutline === mine) { scene.remove(mine); selectionOutline = null; }
  }, 2500);
}

function removeSelected() {
  // Not mid-drag: setPresent would empty the selection under a live gizmo.
  if (gizmo.dragging) return;
  // Snapshotted first: setPresent mutates the selection being walked.
  const recs = selectionRecords().filter((rec) => !rec.isCamera);
  if (!recs.length) { setStatus('Select an object first.', 'err'); return; }
  const removable = recs.filter((rec) => rec.entry.editable);
  if (!removable.length) { setStatus('That object cannot be removed.', 'err'); return; }
  pushUndo(removable, 'remove');
  for (const rec of removable) setPresent(rec, false);
  rebuildPivot();
  renderList();
  refreshReadout();
  setStatus(removable.length > 1
    ? `Removed ${removable.length} objects. Ctrl+Z restores them; `
      + 'Save writes the scene without them.'
    : `Removed ${removable[0].name}. Ctrl+Z restores it; `
      + 'Save writes the scene without it.');
}

function restoreRecord(rec) {
  pushUndo(rec, 'restore');
  // Clear both flags that keep an object off screen, or "Restored" can show
  // nothing.
  rec.hiddenForView = false;
  setPresent(rec, true);
  select(rec);
  renderList();
  refreshReadout();
  setStatus(`Restored ${rec.name}.`);
}

// Duplicating re-imports the selection's own USD, so the copy is a real second
// object with its own name and pose rather than a second reference to the first.
async function duplicateSelected() {
  // The button is disabled for a set; Ctrl+D has to agree with it.
  if (selection.size > 1) {
    setStatus('Duplicate works on one object at a time.', 'err');
    return;
  }
  if (!selected || selected.isCamera || !selected.entry.editable) {
    setStatus('Select an object to duplicate.', 'err');
    return;
  }
  const source = selected;
  let assets;
  try {
    assets = await loadLibrary();
  } catch (err) {
    setStatus(`Duplicate failed: ${err.message}`, 'err');
    return;
  }
  const match = assets.find((a) => a.usd === source.entry.sourceUsd);
  if (!match) {
    setStatus(`No library entry for ${source.name}'s asset; add it from the list instead.`, 'err');
    return;
  }
  const rec = await addAsset(match.key, {
    scale: source.group.scale.toArray(),
    label: 'duplicate',
    pose: besidePose(source),
  });
  if (!rec) return;
  // Orientation matches; position is offset so the copy is visible.
  rec.group.quaternion.copy(source.group.quaternion);
  refreshDirty(rec);
  select(rec);
  refreshReadout();
  setStatus(`Duplicated ${source.name} as ${rec.name}. Arrows nudge it into place.`, 'ok');
}

// Where a copy of *this* object should land: clear of its own bounding box,
// along the direction that reads as "to the right" from where you are looking.
const _besideRight = new THREE.Vector3();
const _besideFwd = new THREE.Vector3();
function besidePose(source) {
  const box = new THREE.Box3().setFromObject(source.group);
  const size = box.getSize(new THREE.Vector3());
  camera.getWorldDirection(_besideFwd);
  _besideFwd.z = 0;
  if (_besideFwd.lengthSq() < 1e-9) _besideFwd.set(1, 0, 0);
  _besideFwd.normalize();
  _besideRight.crossVectors(_besideFwd, WORLD_UP).normalize().negate();
  // A 20% gap so the two are visibly separate rather than touching.
  const gap = Math.max(size.x, size.y, 0.05) * 1.2;
  const p = source.group.position;
  return {
    position: [p.x + _besideRight.x * gap, p.y + _besideRight.y * gap, p.z],
    orientation: source.group.quaternion.toArray(),
  };
}

async function loadLibrary() {
  if (library) return library;
  const res = await fetch('/api/library');
  if (!res.ok) throw new Error(`library -> HTTP ${res.status}`);
  const body = await res.json();
  library = body.assets;
  return library;
}

function renderLibrary() {
  const list = document.getElementById('liblist');
  const count = document.getElementById('libcount');
  list.innerHTML = '';
  if (!library) { count.textContent = 'loading…'; return; }

  const needle = libraryFilter.trim().toLowerCase();
  const matches = library.filter((a) => !needle
    || a.category.includes(needle)
    || a.asset_id.includes(needle)
    || a.source.toLowerCase().includes(needle));

  count.textContent = needle
    ? `${matches.length} of ${library.length}`
    : `${library.length} asset(s)`;

  // Cap the rebuild so typing stays responsive.
  const LIMIT = 120;
  for (const asset of matches.slice(0, LIMIT)) {
    const row = document.createElement('div');
    row.className = 'asset';
    row.title = asset.usd;
    const label = document.createElement('span');
    label.textContent = asset.category;
    // Several scenes model the same category differently; the variant
    // disambiguates otherwise-identical rows.
    if (asset.kind !== 'mesh') {
      const variant = document.createElement('span');
      variant.className = 'variant';
      variant.textContent = ` ${asset.variant}`;
      label.appendChild(variant);
    }
    const tag = document.createElement('span');
    tag.className = 'tag';
    // A mesh is converted on import; say so.
    tag.textContent = asset.kind === 'mesh'
      ? `${asset.format} → usd`
      : (asset.local ? 'in scene' : asset.source);
    // Shared by several scenes, which is worth knowing when picking a prop.
    if (asset.copies > 1) tag.textContent += ` ×${asset.copies}`;
    row.append(label, tag);
    row.onclick = () => selectAsset(asset);
    if (selectedAsset && selectedAsset.key === asset.key) row.classList.add('sel');
    list.appendChild(row);
  }
  if (matches.length > LIMIT) {
    const more = document.createElement('div');
    more.className = 'hint';
    more.textContent = `…and ${matches.length - LIMIT} more — narrow the search.`;
    list.appendChild(more);
  }
}

// --- placing --------------------------------------------------------------
// A translucent stand-in follows the pointer across whatever surface is under
// it, and a click commits. Nothing is copied into the scene directory until
// commit.

let placing = null;      // {ghost, request, facts, label, valid}
const _placeRay = new THREE.Raycaster();
const _placePointer = new THREE.Vector2();
const _placeBox = new THREE.Box3();

function beginPlacement({ request, facts, glb, label }) {
  cancelPlacement();
  closeLibrary();
  gltfLoader.loadAsync(`./data/${glb}`).then((gltf) => {
    const ghost = gltf.scene;
    ghost.traverse((child) => {
      if (!child.isMesh) return;
      // Cloned, or the library preview would turn translucent too.
      child.material = child.material.clone();
      child.material.transparent = true;
      child.material.opacity = 0.55;
      child.material.depthWrite = false;
      child.raycast = () => {};   // the ghost must never be its own drop target
    });
    markAsOverlay(ghost);
    scene.add(ghost);
    placing = { ghost, request, facts, label, valid: false };
    document.getElementById('viewport').classList.add('placing');
    setStatus(`Placing ${label} — move over a surface and click. `
      + 'Esc cancels, Shift keeps it level with the last point.');
  }).catch((err) => {
    setStatus(`Could not load a preview to place: ${err.message}`, 'err');
  });
}

function cancelPlacement(message) {
  if (!placing) return;
  scene.remove(placing.ghost);
  placing = null;
  document.getElementById('viewport').classList.remove('placing');
  if (message) setStatus(message);
}

// Where the ghost sits for a pointer at (x, y) over the canvas.
function updatePlacement(event) {
  if (!placing) return;
  const rect = renderer.domElement.getBoundingClientRect();
  _placePointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  _placePointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  _placeRay.setFromCamera(_placePointer, camera);

  // Only real scene geometry is a support surface; the ghost has no raycast.
  const targets = [];
  for (const rec of objects.values()) {
    if (rec.isCamera || rec.present === false) continue;
    rec.group.traverse((child) => { if (child.isMesh) targets.push(child); });
  }
  const hit = _placeRay.intersectObjects(targets, false)[0];
  placing.valid = !!hit;
  if (!hit) {
    setStatus(`Placing ${placing.label} — no surface under the pointer.`, 'err');
    return;
  }

  // Rest the ghost's underside on the hit point rather than its origin: an
  // asset's pivot is wherever its author put it.
  _placeBox.setFromObject(placing.ghost);
  const centre = _placeBox.getCenter(new THREE.Vector3());
  placing.ghost.position.x += hit.point.x - centre.x;
  placing.ghost.position.y += hit.point.y - centre.y;
  placing.ghost.position.z += hit.point.z - _placeBox.min.z;
  placing.on = hit.object.userData.owner || 'the scene';
  setStatus(`Placing ${placing.label} on ${placing.on} — click to commit, Esc cancels.`);
}

async function commitPlacement() {
  if (!placing || !placing.valid) return;
  const { request, ghost, label } = placing;
  const pose = {
    position: ghost.position.toArray(),
    orientation: ghost.quaternion.toArray(),
  };
  cancelPlacement();
  setStatus(`Adding ${label}…`);
  // The import runs now, at the chosen pose.
  if (request.key) {
    await addAsset(request.key, { pose, label: 'add object' });
  } else {
    await importFromPath({ path: request.path, pose });
  }
}

// --- reviewing an asset before placing it ----------------------------------
// /api/inspect reports size, mass and collision before anything is imported.

let selectedAsset = null;      // the library row under review
let assetFacts = null;         // what /api/inspect said about it
let inspectSeq = 0;            // so a slow inspect cannot overwrite a newer one

async function selectAsset(asset) {
  selectedAsset = asset;
  assetFacts = null;
  renderLibrary();
  const detail = document.getElementById('asset-detail');
  detail.hidden = false;
  document.getElementById('detail-status').textContent = '';
  document.getElementById('btn-place').disabled = true;
  document.getElementById('detail-facts').innerHTML =
    `<div><span class="k">asset</span>${asset.category} ${asset.variant || ''}</div>`
    + '<div class="hint" style="margin:6px 0 0">reading…</div>';
  document.getElementById('detail-caveats').textContent = '';

  const seq = ++inspectSeq;
  try {
    const body = await inspectAsset({ key: asset.key });
    if (seq !== inspectSeq) return;      // a newer selection won
    assetFacts = body;
    renderAssetDetail(asset, body);
  } catch (err) {
    if (seq !== inspectSeq) return;
    document.getElementById('detail-facts').innerHTML =
      `<div class="warn">Could not read this asset: ${err.message}</div>`;
  }
}

async function inspectAsset(request) {
  const res = await fetch('/api/inspect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
    body: JSON.stringify({ ...request, ...importOptions() }),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

const mm = (metres) => `${(metres * 1000).toFixed(0)} mm`;

function renderAssetDetail(asset, body) {
  const f = body.facts;
  const rows = [];
  rows.push(`<div><span class="k">asset</span>${body.category} `
    + `<span class="dim">${asset.variant || body.asset_id}</span></div>`);
  if (f.size) {
    rows.push(`<div><span class="k">size</span>${f.size.map(mm).join(' × ')}</div>`);
  }
  rows.push(`<div><span class="k">mass</span>${
    f.mass != null ? `${f.mass} kg` : '<span class="dim">not authored</span>'}</div>`);

  // A USD reports the collision it *has*; a mesh what conversion *will make*.
  if (body.kind === 'usd') {
    rows.push(`<div><span class="k">collision</span>${
      f.collision_prims
        ? `${f.collision_prims} prim(s): ${f.collision.join(', ')}`
        : '<span class="warn">none in this USD</span>'}</div>`);
  } else {
    rows.push(`<div><span class="k">collision</span>${f.collision} `
      + '<span class="dim">(generated on import)</span></div>');
    if (f.rotated) {
      rows.push('<div><span class="k">up axis</span>rotated Y-up → Z-up</div>');
    }
  }
  if (f.verts) {
    rows.push(`<div><span class="k">geometry</span>${f.verts.toLocaleString()} verts`
      + `${f.textured ? ', textured' : ''}</div>`);
  }
  document.getElementById('detail-facts').innerHTML = rows.join('');
  document.getElementById('detail-caveats').innerHTML =
    (f.caveats || []).map((c) => `<div class="warn">⚠ ${c}</div>`).join('');
  document.getElementById('btn-place').disabled = false;
  showPreview(body.glb);
}

// --- the little preview renderer -------------------------------------------
// Its own GL context, created on first use and torn down with the modal.

let previewRenderer = null;
let previewScene = null;
let previewCamera = null;
let previewObject = null;
let previewSpin = 0;

function ensurePreview() {
  if (previewRenderer) return;
  const canvas = document.getElementById('detail-preview');
  previewRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  previewRenderer.setPixelRatio(window.devicePixelRatio || 1);
  previewRenderer.setSize(canvas.clientWidth || 220, canvas.clientHeight || 165, false);
  previewScene = new THREE.Scene();
  previewScene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(1, -1, 2);
  previewScene.add(key);
  previewCamera = new THREE.PerspectiveCamera(35, (canvas.clientWidth || 220) / (canvas.clientHeight || 165), 0.01, 50);
  previewCamera.up.copy(WORLD_UP);
}

async function showPreview(glbName) {
  ensurePreview();
  if (previewObject) { previewScene.remove(previewObject); previewObject = null; }
  try {
    const gltf = await gltfLoader.loadAsync(`./data/${glbName}`);
    previewObject = gltf.scene;
    previewScene.add(previewObject);
    // Frame the object so it fills the canvas whatever its scale.
    const box = new THREE.Box3().setFromObject(previewObject);
    const centre = box.getCenter(new THREE.Vector3());
    previewObject.position.sub(centre);
    const radius = Math.max(box.getSize(new THREE.Vector3()).length(), 1e-3);
    previewCamera.position.set(radius * 1.1, -radius * 1.1, radius * 0.7);
    previewCamera.lookAt(0, 0, 0);
    previewSpin = 0;
  } catch (err) {
    console.warn('preview failed', err);
  }
}

function renderPreviewFrame(dt) {
  if (!previewRenderer || !previewObject || !libraryIsOpen()) return;
  // A slow turn: a still image of an unfamiliar prop hides its depth.
  previewSpin += dt * 0.6;
  previewObject.rotation.z = previewSpin;
  previewRenderer.render(previewScene, previewCamera);
}

function disposePreview() {
  if (!previewRenderer) return;
  previewRenderer.dispose();
  previewRenderer = null;
  previewScene = previewObject = previewCamera = null;
}

function libraryIsOpen() {
  return !document.getElementById('library-modal').hidden;
}

function closeLibrary() {
  document.getElementById('library-modal').hidden = true;
  document.getElementById('btn-library').classList.remove('on');
  // The preview owns a GL context; release it with the modal.
  disposePreview();
}

async function openLibrary() {
  if (libraryIsOpen()) { closeLibrary(); return; }
  document.getElementById('library-modal').hidden = false;
  document.getElementById('btn-library').classList.add('on');
  // Typing is how you find anything in a 350-asset list, so start there.
  document.getElementById('libsearch').focus();
  renderLibrary();
  try {
    await loadLibrary();
    renderLibrary();
  } catch (err) {
    document.getElementById('libcount').textContent = `failed: ${err.message}`;
  }
}

document.getElementById('btn-library-close').onclick = closeLibrary;
document.getElementById('library-backdrop').onclick = closeLibrary;

// --- importing from anywhere on this machine -------------------------------
// The browser cannot hand a real filesystem path to the server, so the path is
// typed or pasted; the server reads it directly and copies the whole asset in.

function importOptions() {
  const num = (id, fallback) => {
    const raw = document.getElementById(id).value.trim();
    if (!raw) return fallback;
    const value = Number.parseFloat(raw);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    mesh_scale: num('imp-scale', 1.0),
    up_axis: document.getElementById('imp-up').value,
    collision: document.getElementById('imp-collision').value,
    mass: num('imp-mass', null),
  };
}

function showImportReport(report, source) {
  const el = document.getElementById('imp-report');
  el.innerHTML = '';
  if (!report) return;
  const lines = [];
  if (report.size) {
    lines.push(`${source}: ${report.size.map((v) => v.toFixed(3)).join(' × ')} m`
      + (report.mass ? `, ${report.mass} kg` : '')
      + (report.rotated ? ', rotated Y-up → Z-up' : ''));
  }
  for (const note of report.notes || []) lines.push(note);
  for (const caveat of report.caveats || []) lines.push(`⚠ ${caveat}`);
  for (const line of lines) {
    const div = document.createElement('div');
    div.textContent = line;
    if (line.startsWith('⚠')) div.className = 'warn';
    el.appendChild(div);
  }
}

// --- file picker -----------------------------------------------------------
// Browses the server's filesystem: the import endpoint needs an absolute path
// on the machine running the server, which a native file input never yields.

let pickerPath = null;
// Which input the picker is filling in, and what /api/browse should list;
// callers override both for the duration of one open.
let pickerTarget = 'imp-path';
let pickerFilter = null;
let pickerOnChoose = null;
// What kind of answer this open is for: 'file', 'directory', or 'any' (the
// importer's mode, where both are answers).
let pickerMode = 'any';
// Back/forward like a browser's: the folders visited this open. Reset each
// time the picker opens fresh.
let pickerHistory = [];
let pickerHistoryIndex = -1;

function updatePickerNavButtons() {
  document.getElementById('btn-picker-back').disabled = pickerHistoryIndex <= 0;
  document.getElementById('btn-picker-forward').disabled =
    pickerHistoryIndex >= pickerHistory.length - 1;
}

function pickerBack() {
  if (pickerHistoryIndex <= 0) return;
  pickerHistoryIndex -= 1;
  browseTo(pickerHistory[pickerHistoryIndex], { fromHistory: true });
}

function pickerForward() {
  if (pickerHistoryIndex >= pickerHistory.length - 1) return;
  pickerHistoryIndex += 1;
  browseTo(pickerHistory[pickerHistoryIndex], { fromHistory: true });
}

function pickerIsOpen() {
  return !document.getElementById('picker-modal').hidden;
}

function closePicker() {
  document.getElementById('picker-modal').hidden = true;
}

function formatSize(bytes) {
  if (bytes === null || bytes === undefined) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** True when this open accepts *kind* ('file' or 'directory') as its answer. */
function pickerAccepts(kind) {
  return pickerMode === 'any' || pickerMode === kind;
}

/**
 * Take *path* as the picker's answer, if it is the kind this open asked for.
 * A refusal leaves the picker open — the wanted answer is in the folder shown.
 */
function choosePath(path, kind = 'file') {
  if (!pickerAccepts(kind)) {
    document.getElementById('picker-note').textContent =
      pickerMode === 'directory'
        ? `${path.split('/').pop()} is a file — open a folder and press "Use this folder".`
        : `${path.split('/').pop()} is a folder — pick a file inside it.`;
    return;
  }
  if (pickerOnChoose) {
    pickerOnChoose(path);
  } else {
    document.getElementById(pickerTarget).value = path;
    setStatus(`Selected ${path.split('/').pop()} — press Import to bring it in.`);
  }
  closePicker();
}

/**
 * Navigate the picker to *path*, or — typed paths only — accept a typed file
 * path directly. `fromInput` keeps a listing's own entries from being
 * reinterpreted as "choose this"; the shortcut answers with a file, so an open
 * that wants a folder must not take it.
 */
async function browseTo(path, { fromInput = false, fromHistory = false } = {}) {
  const list = document.getElementById('picker-list');
  const note = document.getElementById('picker-note');
  list.innerHTML = '';
  note.textContent = 'Loading…';
  try {
    const res = await fetch('/api/browse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({ path: path ?? null, filter: pickerFilter }),
    });
    const body = await res.json();
    if (!res.ok) {
      if (fromInput && path && /not a directory/i.test(body.error || '')) {
        if (pickerAccepts('file')) {
          choosePath(path, 'file');
          return;
        }
        throw new Error(`${path} is a file — this needs a folder`);
      }
      throw new Error(body.error || res.statusText);
    }

    pickerPath = body.path;
    document.getElementById('picker-path').value = body.path;
    document.getElementById('btn-picker-up').disabled = !body.parent;
    document.getElementById('btn-picker-up').onclick = () => browseTo(body.parent);

    if (!fromHistory) {
      // A fresh navigation drops any forward history, like a browser tab.
      pickerHistory = pickerHistory.slice(0, pickerHistoryIndex + 1);
      if (pickerHistory[pickerHistoryIndex] !== body.path) {
        pickerHistory.push(body.path);
        pickerHistoryIndex = pickerHistory.length - 1;
      }
    }
    updatePickerNavButtons();

    const places = document.getElementById('picker-places');
    places.innerHTML = '';
    for (const shortcut of body.shortcuts || []) {
      const button = document.createElement('button');
      button.textContent = shortcut.label;
      button.title = shortcut.path;
      button.onclick = () => browseTo(shortcut.path);
      places.appendChild(button);
    }

    for (const entry of body.entries) {
      const row = document.createElement('div');
      const isDir = entry.kind === 'directory';
      row.className = `pick ${isDir ? 'dir' : 'file'}`;
      row.title = entry.path;

      const glyph = document.createElement('span');
      glyph.className = 'glyph';
      glyph.textContent = isDir ? '▸' : '·';
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = entry.name;
      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = isDir ? '' : `${entry.kind}  ${formatSize(entry.size)}`;

      // A file this open cannot accept is shown but must not look clickable.
      if (!isDir && !pickerAccepts('file')) row.classList.add('inert');

      row.append(glyph, name, meta);
      // A directory navigates; a file is the answer. "Use this folder" is the
      // separate, deliberate act of adding a library root.
      row.onclick = () => (isDir ? browseTo(entry.path)
                                 : choosePath(entry.path, 'file'));
      list.appendChild(row);
    }

    const files = body.entries.filter((e) => e.kind !== 'directory').length;
    const noun = pickerFilter === 'yaml' ? 'file(s)' : 'asset(s)';
    note.textContent = body.entries.length === 0
      ? 'Nothing to pick here.'
      : pickerMode === 'directory'
        ? `${body.entries.length - files} folder(s) — press "Use this folder" to choose this one`
        : `${files} ${noun}${body.truncated ? ' — list truncated' : ''}`;
  } catch (err) {
    note.textContent = `Cannot open: ${err.message}`;
  }
}

/**
 * Open the shared file-browser modal. `opts.target` names the input to fill
 * (default: the asset importer's path field); `opts.onChoose` replaces that.
 * `opts.filter` narrows what /api/browse lists ('yaml' for task configs).
 * `opts.mode` is what an answer has to be — 'file', 'directory', or 'any'
 * (the default). `opts.title` names what is being chosen.
 */
function openPicker(opts = {}) {
  pickerTarget = opts.target || 'imp-path';
  pickerFilter = opts.filter || null;
  pickerOnChoose = opts.onChoose || null;
  pickerMode = opts.mode || 'any';
  pickerHistory = [];
  pickerHistoryIndex = -1;
  updatePickerNavButtons();
  document.getElementById('picker-title').textContent = opts.title || ({
    file: 'Choose a file',
    directory: 'Choose a folder',
  }[pickerMode] || 'Choose a file or folder');
  // "Use this folder" answers with a directory; hidden when the mode cannot
  // accept one.
  const useDir = document.getElementById('btn-picker-usedir');
  useDir.hidden = !pickerAccepts('directory');
  useDir.classList.toggle('primary', pickerMode === 'directory');
  document.getElementById('picker-modal').hidden = false;
  // Start where the field already points, so re-opening resumes.
  const current = pickerOnChoose ? '' : document.getElementById(pickerTarget).value.trim();
  browseTo(current || pickerPath || null);
}

// --- drag and drop ---------------------------------------------------------
// A dropped file is staged on the server, then imported through the same
// endpoint the picker uses. Only self-contained formats: a lone .usd or .obj
// usually references textures that are not part of the drop.
const DROPPABLE = ['.glb', '.ply', '.stl'];

function dropHint(show, message) {
  const el = document.getElementById('drop-overlay');
  el.hidden = !show;
  if (message) el.querySelector('span').textContent = message;
}

async function uploadAndImport(file) {
  const dot = file.name.lastIndexOf('.');
  const suffix = dot < 0 ? '' : file.name.slice(dot).toLowerCase();
  if (!DROPPABLE.includes(suffix)) {
    setStatus(`${file.name}: drop takes ${DROPPABLE.join(', ')} — a .usd or .obj `
      + 'needs its textures alongside it, so use Browse… for those.', 'err');
    return;
  }

  setStatus(`Uploading ${file.name}…`);
  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Editor-Token': EDITOR_TOKEN,
        // Header values must be latin-1, and a filename need not be.
        'X-Filename': encodeURIComponent(file.name),
      },
      body: file,
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    // Hand the staged path to the ordinary import.
    document.getElementById('imp-path').value = body.path;
    await importFromPath({ label: file.name });
  } catch (err) {
    setStatus(`Upload failed: ${err.message}`, 'err');
  }
}

(function wireDragAndDrop() {
  const zone = document.body;
  let depth = 0;                      // dragenter/leave fire per child element

  const carriesFiles = (e) =>
    e.dataTransfer && [...e.dataTransfer.types].includes('Files');

  zone.addEventListener('dragenter', (e) => {
    if (!carriesFiles(e)) return;
    e.preventDefault();
    depth += 1;
    dropHint(true, `Drop to import  (${DROPPABLE.join(' · ')})`);
  });
  zone.addEventListener('dragover', (e) => {
    if (!carriesFiles(e)) return;
    // Without this the browser navigates to the file instead of dropping it.
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  zone.addEventListener('dragleave', (e) => {
    if (!carriesFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) dropHint(false);
  });
  zone.addEventListener('drop', async (e) => {
    if (!carriesFiles(e)) return;
    e.preventDefault();
    depth = 0;
    dropHint(false);
    const files = [...(e.dataTransfer.files || [])];
    if (files.length === 0) return;
    // Sequential: the server takes one write at a time anyway.
    for (const file of files) await uploadAndImport(file);
  });
})();

document.getElementById('btn-browse').onclick = () => openPicker();
document.getElementById('btn-picker-close').onclick = closePicker;
document.getElementById('picker-backdrop').onclick = closePicker;
document.getElementById('btn-picker-usedir').onclick = () => {
  // `pickerPath` is whatever /api/browse last listed — a directory by construction.
  if (pickerPath) choosePath(pickerPath, 'directory');
};
document.getElementById('btn-picker-back').onclick = pickerBack;
document.getElementById('btn-picker-forward').onclick = pickerForward;

{
  const goToTyped = () => {
    const text = document.getElementById('picker-path').value.trim();
    if (text) browseTo(text, { fromInput: true });
  };
  document.getElementById('btn-picker-go').onclick = goToTyped;
  document.getElementById('picker-path').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); goToTyped(); }
  });
}

async function importFromPath({ label, path: explicitPath = null, pose = null } = {}) {
  if (addInFlight) { setStatus('An import is already in progress…'); return; }
  const field = document.getElementById('imp-path');
  const path = explicitPath || field.value.trim();
  if (!path) { setStatus('Type or paste a path to import.', 'err'); return; }

  addInFlight = true;
  // A dropped file is staged under a temp name; say what the user dropped.
  setStatus(label ? `Importing ${label}…` : 'Importing…');
  try {
    const res = await fetch('/api/import_path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({ path, ...(pose || spawnPose()), ...importOptions() }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);

    if (body.kind === 'directory') {
      // A folder joins the library rather than becoming one object.
      library = body.assets;
      renderLibrary();
      showImportReport(null);
      setStatus(body.message, body.added_root ? 'ok' : '');
      field.value = '';
      return;
    }

    const rec = await instantiate(body.object);
    rec.added = true;
    undoStack.push({ label: 'import', snaps: [{ ...snapshot(rec), present: false }] });
    if (undoStack.length > MAX_HISTORY) undoStack.shift();
    redoStack.length = 0;
    updateHistoryButtons();
    setPresent(rec, true);
    select(rec);
    renderList();
    refreshReadout();
    showImportReport(body.report, rec.name);
    revealImport(rec);
    if (!explicitPath) field.value = '';
    // The library cache is stale now that the file has been copied in.
    library = null;
    setStatus(`Imported ${rec.name}. Move it into place, then Save.`, 'ok');
  } catch (err) {
    setStatus(`Import failed: ${err.message}`, 'err');
    showImportReport(null);
  } finally {
    addInFlight = false;
  }
}

document.getElementById('btn-import').onclick = () => importFromPath();
document.getElementById('imp-path').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); importFromPath(); }
});
document.getElementById('btn-import-opts').onclick = (e) => {
  const panel = document.getElementById('imp-options');
  const open = panel.style.display === 'none';
  panel.style.display = open ? '' : 'none';
  e.target.classList.toggle('on', open);
};

document.getElementById('btn-place').onclick = () => {
  if (!selectedAsset || !assetFacts) return;
  beginPlacement({
    request: { key: selectedAsset.key },
    facts: assetFacts.facts,
    glb: assetFacts.glb,
    label: `${assetFacts.category} ${selectedAsset.variant || ''}`.trim(),
  });
};

document.getElementById('btn-library').onclick = openLibrary;
document.getElementById('btn-remove').onclick = removeSelected;
document.getElementById('btn-duplicate').onclick = duplicateSelected;
for (const id of ['imp-scale', 'imp-up', 'imp-collision', 'imp-mass']) {
  const field = document.getElementById(id);
  // These change what the import would write, so re-inspect the selected asset.
  if (field) field.addEventListener('change', () => {
    if (selectedAsset) selectAsset(selectedAsset);
  });
}

document.getElementById('libsearch').addEventListener('input', (e) => {
  libraryFilter = e.target.value;
  renderLibrary();
});

// --- cameras ---------------------------------------------------------------
// external_sensors poses are relative to a robot link, not the world. Each
// camera is parented to that object, so its local transform writes back as-is.

let cameraConfig = null;
const cameraDirty = new Set();

// How far the drawn frustum extends, in metres. Display only.
const FRUSTUM_LENGTH_M = 1.2;

// One colour per camera, in config order, shared by its frustum, body and
// preview frame.
const CAMERA_COLOURS = [
  0x4ea1ff,   // blue
  0xff9d3d,   // orange
  0xc98bff,   // violet
  0xffd93d,   // yellow
  0xff6b8a,   // pink
  0x3ddbd9,   // cyan
];

const hexColour = (value) => `#${value.toString(16).padStart(6, '0')}`;

// Reads a frustum's colour from its vertex colours (CameraHelper stores colour
// per vertex, not on the material).
const _frustumColour = new THREE.Color();
function frustumColourHex(rec) {
  const attribute = rec.helper && rec.helper.geometry.getAttribute('color');
  if (!attribute) return null;
  return hexColour(_frustumColour.fromBufferAttribute(attribute, 0).getHex());
}

// Render clip planes, in metres. An explicit clipping_range in the config wins;
// the defaults keep a depth ratio WebGL can resolve.
const DEFAULT_SENSOR_NEAR_M = 0.01;
const DEFAULT_SENSOR_FAR_M = 100;
const MAX_SENSOR_FAR_M = 1000;      // beyond this, z-fighting is worse than clipping

/** Near/far for a sensor's render camera, honouring an explicit clipping_range. */
function sensorClipping(cam) {
  const range = cam.clipping_range;
  if (!Array.isArray(range) || range.length !== 2) {
    return [DEFAULT_SENSOR_NEAR_M, DEFAULT_SENSOR_FAR_M];
  }
  const [rawNear, rawFar] = range.map(Number);
  if (!Number.isFinite(rawNear) || !Number.isFinite(rawFar) || rawFar <= rawNear) {
    return [DEFAULT_SENSOR_NEAR_M, DEFAULT_SENSOR_FAR_M];
  }
  // Clamp: near kept off zero, far bounded, to preserve depth precision.
  return [Math.max(rawNear, 1e-3), Math.min(rawFar, MAX_SENSOR_FAR_M)];
}

async function loadCameras() {
  let payload;
  try {
    const res = await fetch('/api/cameras');
    payload = await res.json();
  } catch {
    return;                       // server started without --cameras
  }
  if (!payload || !payload.cameras || payload.cameras.length === 0) return;
  cameraConfig = payload;
  if (typeof payload.camera_revision === 'number') cameraRevision = payload.camera_revision;

  const parentRec = payload.parent_object ? objects.get(payload.parent_object) : null;
  const parent = parentRec ? parentRec.group : scene;
  if (!parentRec) {
    setStatus(`Cameras reference ${payload.parent_object || 'an unknown prim'}, `
      + 'which is not in this scene — poses shown in world frame.', 'err');
  }

  const observation = (payload.observation && payload.observation.cameras) || {};
  if (payload.observation && payload.observation.note) {
    document.getElementById('cam-observation').textContent = payload.observation.note;
  }

  payload.cameras.forEach((cam, camIndex) => {
    const colour = CAMERA_COLOURS[camIndex % CAMERA_COLOURS.length];
    const group = new THREE.Group();
    group.name = cam.name;
    group.position.fromArray(cam.position);
    group.quaternion.fromArray(cam.orientation);
    parent.add(group);

    // Frustum uses the config's real optics. USD and three.js both look down -Z
    // with +Y up.
    const vfov = cam.v_fov_deg || 60;
    // Aspect from pixel dimensions when known, else the config's aspect hint.
    const aspect = (cam.image_width && cam.image_height)
      ? cam.image_width / cam.image_height : (cam.aspect || 16 / 9);
    // A read-only camera belongs to the robot asset itself: it cannot be moved
    // here and has no external_sensors entry to write back to.
    const readOnly = cam.read_only === true;

    // Two cameras: the render needs the sensor's real far plane, the drawn
    // frustum a short one.
    const [near, far] = sensorClipping(cam);
    const proj = new THREE.PerspectiveCamera(vfov, aspect, near, far);
    // Content only: this is the frame the policy receives, so no editor
    // furniture may appear in it.
    proj.layers.set(LAYER_CONTENT);
    group.add(proj);

    // Display-only twin with a short far plane, so the frustum reads as a cone.
    const frustumCam = new THREE.PerspectiveCamera(vfov, aspect, near, FRUSTUM_LENGTH_M);
    group.add(frustumCam);
    const helper = new THREE.CameraHelper(frustumCam);
    helper.userData.owner = cam.name;
    // Per-camera colour, matching the body and the preview frame.
    const line = new THREE.Color(colour);
    const dim = line.clone().multiplyScalar(0.45);
    helper.setColors(line, line, line, line, dim);
    scene.add(markAsOverlay(helper));

    // A small solid body makes the camera clickable; helper lines are not.
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.05, 0.06),
      new THREE.MeshStandardMaterial({ color: colour, roughness: 0.6 }),
    );
    body.userData.owner = cam.name;
    group.add(markAsOverlay(body));

    objects.set(cam.name, {
      name: cam.name,
      group,
      isCamera: true,
      proj,
      helper,
      body,
      colour,
      // Config order; the evaluation's fallback when a task names no camera.
      configIndex: camIndex,
      optics: {
        width: cam.image_width || null,
        height: cam.image_height || null,
        hFov: cam.h_fov_deg || null,
        vFov: cam.v_fov_deg || null,
        modalities: cam.modalities || [],
      },
      observation: observation[cam.name] || null,
      readOnly,
      readOnlyWhy: cam.why || null,
      // Synthetic entry: a camera is not a scene object. Save paths skip camera
      // records on `isCamera` — cameras are written to the external_sensors
      // config, never to the scene JSON.
      entry: {
        name: cam.name, kind: 'camera',
        category: readOnly ? 'robot_sensor' : 'external_sensor',
        // All three off for a robot's own camera: it can only be looked through.
        editable: !readOnly, posable: !readOnly, scalable: !readOnly,
      },
      lastValidScale: group.scale.clone(),
      initial: {
        position: cam.position.slice(),
        orientation: cam.orientation.slice(),
        scale: [1, 1, 1],
      },
    });
  });

  document.getElementById('tab-cameras').disabled = false;
  document.getElementById('tab-cameras').title = '';
  // Only worth offering once there is something to hide.
  document.getElementById('m-cams').hidden = false;
  applyCameraOverlayVisibility();
  // Count only what a save writes: read-only robot cameras have no
  // external_sensors entry.
  const writable = payload.cameras.filter((cam) => cam.read_only !== true);
  const locked = payload.cameras.length - writable.length;
  document.getElementById('cam-target').textContent =
    `${writable.length} camera(s) → ${payload.cfg_name}.yaml`
    + (locked ? ` · ${locked} on the robot, not written` : '');
  const link = payload.parent_link ? ` (relative to ${payload.parent_link})` : '';
  document.getElementById('cam-source').textContent =
    `from ${payload.source.split('/').pop()}${link}`;
  const room = payload.background ? `room “${payload.background}”` : 'this scene';
  document.getElementById('cam-memory').textContent = payload.resumed
    ? `resumed the placement saved for ${room}`
    : `nothing saved for ${room} yet — this is the rig template`;
  // Before the pips: `defaultPipLayout` measures the switcher bar, which must
  // be unhidden by then.
  refreshViewSwitcher();
  buildPips();
}

// --- flying a camera -------------------------------------------------------
// Selecting a camera puts the viewport inside it: WASD walks the eye point,
// dragging aims. The pose is stored relative to the robot base but flown in
// world space and converted back on each commit. Yaw is applied in world space,
// pitch in the camera's frame; authored roll is kept until `Level` removes it.

const WORLD_UP = new THREE.Vector3(0, 0, 1);
const LOCAL_RIGHT = new THREE.Vector3(1, 0, 0);
const UNIT_SCALE = new THREE.Vector3(1, 1, 1);
// Pitch stops just short of straight up, where a yaw/pitch camera loses heading.
const MAX_PITCH_SIN = Math.sin(THREE.MathUtils.degToRad(89));
const LOOK_RADIANS_PER_PIXEL = 0.0025;      // ~0.14 deg, close to a game default
const SPEED_LIMITS = { min: 0.02, max: 5 };
// Caps dt so a backgrounded tab's first frame back cannot fling the camera.
const MAX_FRAME_SECONDS = 0.05;

let flying = null;                          // camera record being flown, or null
const flyPos = new THREE.Vector3();
const flyQuat = new THREE.Quaternion();
let flySpeed = 0.6;                         // metres per second
const flyKeys = new Set();
// Movement keys by physical code, so modifiers cannot desync keydown/keyup:
// WASD on the ground plane, Space/E up, Ctrl/Q down.
const FLY_CODES = {
  KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd',
  Space: 'up', KeyE: 'up',
  ControlLeft: 'down', ControlRight: 'down', KeyQ: 'down',
};
let flyFast = false;
let flySlow = false;
let flyTargets = [];                        // meshes the centre ray can hit
let flyTargetsStale = false;
const invalidateFlyTargets = () => { flyTargetsStale = true; };
let overviewOn = true;

// Scratch objects reused every frame to avoid allocation in the animation loop.
const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _move = new THREE.Vector3();
const _target = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _matrix = new THREE.Matrix4();
const _parentInverse = new THREE.Matrix4();
const _scratchScale = new THREE.Vector3();

const flyForward = () => _fwd.set(0, 0, -1).applyQuaternion(flyQuat);

// World up rather than the camera's own, so a rolled camera still strafes level.
function flyRight() {
  _right.crossVectors(flyForward(), WORLD_UP);
  if (_right.lengthSq() < 1e-8) _right.copy(LOCAL_RIGHT).applyQuaternion(flyQuat);
  return _right.normalize();
}

// Re-derives flight state from the record, the source of truth for undo,
// Revert and typed edits.
function syncFlyFromRecord() {
  if (!flying) return;
  flying.group.updateWorldMatrix(true, false);
  flyPos.setFromMatrixPosition(flying.group.matrixWorld);
  flying.group.getWorldQuaternion(flyQuat);
}

// World pose back to the camera's local transform, which is exactly what the
// external_sensors YAML stores.
function commitFly() {
  const parent = flying.group.parent;
  parent.updateWorldMatrix(true, false);
  _matrix.compose(flyPos, flyQuat, UNIT_SCALE);
  _matrix.premultiply(_parentInverse.copy(parent.matrixWorld).invert());
  _matrix.decompose(flying.group.position, flying.group.quaternion, _scratchScale);
  noteCameraMoved(flying);
  refreshReadout();
}

// Called per frame while flying; rebuilds the list only when the flag flips.
function noteCameraMoved(rec) {
  const was = cameraDirty.has(rec.name);
  refreshDirty(rec);
  if (was !== cameraDirty.has(rec.name)) renderList();
}

// One undo entry per burst of flying, not per frame: a two-second walk must
// unwind in one Ctrl+Z. The burst closes once the controls have been idle.
let flyEditOpen = false;
let flyEditTimer = null;

function beginFlyEdit(label) {
  clearTimeout(flyEditTimer);
  if (!flyEditOpen) {
    pushUndo(flying, label);
    flyEditOpen = true;
  }
}

function endFlyEdit() {
  clearTimeout(flyEditTimer);
  flyEditTimer = setTimeout(closeFlyEdit, 500);
}

// Close the burst immediately. Undo/redo call this so history cannot be stepped
// while a burst is still accumulating into the entry being undone.
function closeFlyEdit() {
  clearTimeout(flyEditTimer);
  flyEditOpen = false;
}

function look(deltaYaw, deltaPitch) {
  if (!flying) return;
  beginFlyEdit('aim');
  flyQuat.premultiply(_quat.setFromAxisAngle(WORLD_UP, deltaYaw));
  // Pitch in the camera's own frame. Steps past vertical are rejected (clamping
  // would creep in roll), except ones that bring an over-pitched camera back.
  const before = Math.abs(flyForward().z);
  const pitched = flyQuat.clone().multiply(_quat.setFromAxisAngle(LOCAL_RIGHT, deltaPitch));
  const after = Math.abs(_fwd.set(0, 0, -1).applyQuaternion(pitched).z);
  if (after < MAX_PITCH_SIN || after < before) flyQuat.copy(pitched);
  commitFly();
}

// --- walking the free view --------------------------------------------------
// Walks the editor's own camera: WASD on the ground plane, Space/E up, Q down.
// Active only with nothing selected — those keys nudge a selection otherwise.
// No undo entry: moving the view is not a scene edit.
const walkKeys = new Set();
// Metres per second; Shift and Alt scale it, same as the sensor flight.
const WALK_SPEED = 0.6;
// Keyed by physical code, like FLY_CODES. Ctrl is not a descend key here — it
// belongs to the Ctrl+Z/A/D shortcuts; Q descends instead.
const WALK_CODES = {
  KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd',
  Space: 'up', KeyE: 'up', KeyQ: 'down',
};

const _walkFwd = new THREE.Vector3();
const _walkRight = new THREE.Vector3();
const _walkMove = new THREE.Vector3();

/** Whether the movement keys steer the free view rather than anything else. */
function freeWalkArmed() {
  return !flying && selection.size === 0;
}

function updateFreeWalk(dt) {
  if (!walkKeys.size) return;
  // Selecting something mid-stride hands the keys back to the nudges.
  if (!freeWalkArmed()) { walkKeys.clear(); return; }

  _walkFwd.subVectors(orbit.target, camera.position);
  if (_walkFwd.lengthSq() < 1e-12) return;
  _walkFwd.normalize();
  // World up, so a pitched-down view still strafes level.
  _walkRight.crossVectors(_walkFwd, WORLD_UP);
  if (_walkRight.lengthSq() < 1e-8) _walkRight.copy(LOCAL_RIGHT);
  _walkRight.normalize();

  _walkMove.set(0, 0, 0);
  if (walkKeys.has('w')) _walkMove.add(_walkFwd);
  if (walkKeys.has('s')) _walkMove.sub(_walkFwd);
  if (walkKeys.has('d')) _walkMove.add(_walkRight);
  if (walkKeys.has('a')) _walkMove.sub(_walkRight);
  if (walkKeys.has('up')) _walkMove.add(WORLD_UP);
  if (walkKeys.has('down')) _walkMove.sub(WORLD_UP);
  if (_walkMove.lengthSq() < 1e-12) return;

  const scale = (flyFast ? 4 : 1) * (flySlow ? 0.2 : 1);
  _walkMove.normalize().multiplyScalar(WALK_SPEED * scale * Math.min(dt, MAX_FRAME_SECONDS));
  // Move the orbit target with the camera so the next drag pivots from here.
  camera.position.add(_walkMove);
  orbit.target.add(_walkMove);
  // Walking moves the view, so any standing-at-a-sensor claim is now stale.
  clearViewPreset();
}

function updateFly(dt) {
  if (!flying || flyKeys.size === 0) return;
  const forward = flyForward().clone();
  const right = flyRight();
  _move.set(0, 0, 0);
  if (flyKeys.has('w')) _move.add(forward);
  if (flyKeys.has('s')) _move.sub(forward);
  if (flyKeys.has('d')) _move.add(right);
  if (flyKeys.has('a')) _move.sub(right);
  if (flyKeys.has('up')) _move.add(WORLD_UP);
  if (flyKeys.has('down')) _move.sub(WORLD_UP);
  if (_move.lengthSq() < 1e-12) return;

  const scale = (flyFast ? 4 : 1) * (flySlow ? 0.2 : 1);
  _move.normalize().multiplyScalar(flySpeed * scale * Math.min(dt, MAX_FRAME_SECONDS));
  flyPos.add(_move);
  commitFly();
}

// Rebuild the camera's orientation from its heading alone, discarding roll.
function levelHorizon() {
  if (!flying) return;
  beginFlyEdit('level');
  const back = flyForward().clone().negate();     // three.js cameras look down -Z
  const right = flyRight().clone();
  _up.crossVectors(back, right).normalize();
  flyQuat.setFromRotationMatrix(_matrix.makeBasis(right, _up, back));
  commitFly();
  endFlyEdit();
  setStatus(`Levelled ${flying.name}.`);
}

function setFlySpeed(value) {
  flySpeed = THREE.MathUtils.clamp(value, SPEED_LIMITS.min, SPEED_LIMITS.max);
  const field = document.getElementById('fly-speed');
  if (document.activeElement !== field) field.value = flySpeed.toFixed(2);
}

/**
 * Updates the state-dependent hint line at the viewport's bottom-left. Reads
 * live state (`flying`, `selection`, `activeTab`) rather than taking arguments.
 */
function updateHint() {
  const hint = document.getElementById('hint');
  if (flying) {
    hint.textContent = 'WASD to walk, drag to aim · Esc exits · ? for every key';
  } else if (activeTab === 'cameras') {
    hint.textContent = 'Click a camera to look through it · WASD walks the view '
      + '· ? for every key';
  } else if (selection.size > 1) {
    hint.textContent = `${selection.size} selected — drag the gizmo to move them · Del removes · ? for every key`;
  } else if (selected && !selected.entry.editable) {
    // Posable but not editable: the room scales, the robot does not.
    hint.textContent = selected.entry.scalable
      ? 'Drag the gizmo to move it · M/R switch mode · +/− scale · Del is locked here · ? for every key'
      : 'Drag the gizmo to move it · M/R switch mode · scale and Del are locked here · ? for every key';
  } else if (selected) {
    hint.textContent = 'Drag the gizmo to move it · M/R switch mode · +/− scale · Del removes · ? for every key';
  } else {
    hint.textContent = 'WASD walks the view · click an object to select it, '
      + 'or drop a file in to import it · ? for every key';
  }
}

function enterFly(rec) {
  if (!rec || !rec.isCamera) {
    setStatus('Select a camera first.', 'err');
    return;
  }
  if (flying && flying !== rec) showCameraProxy(flying, true);
  // Entering a camera supersedes any view preset.
  viewPreset = null;
  freeViewHome = null;
  flying = rec;
  // Its own frustum and body sit at the eye point and would fill the frame.
  showCameraProxy(rec, false);
  gizmo.detach();
  gizmo.visible = false;
  orbit.enabled = false;                    // the mouse is aiming the sensor now
  // Blur any focused button so Space ascends instead of re-activating it.
  if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
  flyKeys.clear();
  flyEditOpen = false;
  syncFlyFromRecord();
  flyTargets = collectCenterRayTargets();
  document.getElementById('viewport').classList.add('flying');
  updateHint();
  updatePipVisibility();
  updateCameraButtons();
  setStatus(`Flying ${rec.name}. What you see is what it records.`);
  resize();
}

function exitFly(message = 'Back to the overview.') {
  if (!flying) return;
  showCameraProxy(flying, true);
  flying = null;
  flyKeys.clear();
  flyEditOpen = false;
  gizmo.visible = true;
  orbit.enabled = true;
  document.getElementById('viewport').classList.remove('flying');
  updateHint();
  document.getElementById('cam-center').textContent = '-';
  updatePipVisibility();
  updateCameraButtons();
  setStatus(message);
  resize();
}

// Global toggle for camera bodies and frustum lines.
let cameraOverlaysOn = true;

function showCameraProxy(rec, visible, { force = false } = {}) {
  // force: the fly-view inset draws the flown camera regardless of the toggle.
  const show = visible && (cameraOverlaysOn || force);
  rec.helper.visible = show;
  rec.body.visible = show;
}

function applyCameraOverlayVisibility() {
  for (const rec of objects.values()) {
    if (!rec.isCamera) continue;
    // The flown camera and the one the free view stands at remain hidden:
    // their proxies sit at the eye point.
    showCameraProxy(rec, rec !== flying && rec.name !== viewPreset);
  }
  // Lit means shown, same as every other button in the Visibility section.
  document.getElementById('m-cams')?.classList.toggle('on', cameraOverlaysOn);
}

function updateCameraButtons() {
  const button = document.getElementById('btn-look');
  const isCamera = !!(selected && selected.isCamera);
  button.disabled = !isCamera && !flying;
  button.textContent = flying ? 'Exit camera' : 'Enter camera';
  button.classList.toggle('on', !!flying);
  document.getElementById('btn-level').disabled = !flying;
  refreshViewSwitcherActive();
}

// --- the in-viewport view switcher -------------------------------------
// Moves the editor's own camera to stand where a sensor stands. Unlike entering
// a camera (a mode), the selection, gizmo and orbit all stay live.

/** The sensor the free view is standing at, by name; null for the free view. */
let viewPreset = null;
/** Where the free view was before the first jump, so Free view can go back. */
let freeViewHome = null;

/** The evaluation stage's observation key for the robot's wrist camera. */
const WRIST_KEY = 'wrist_image_left';

/** "wrist_cam_left" -> "Wrist L", when more than one wrist camera exists and
    they need telling apart; a single one is just "Wrist". */
function wristLabel(name) {
  if (/left/i.test(name)) return 'Wrist L';
  if (/right/i.test(name)) return 'Wrist R';
  return name;
}

// 20 m is past anything in a room; the overlay layer stays off so camera
// proxies are not hit.
const povRay = new THREE.Raycaster();
povRay.far = 20;

/** Put the free view where *rec* is, looking where it looks. */
function lookFrom(rec) {
  if (!rec || !rec.isCamera) return;
  // Leave any flight first: this view needs the mouse for orbit and gizmo.
  if (flying) exitFly(`Left ${flying.name}.`);
  if (!viewPreset) {
    freeViewHome = {
      position: camera.position.toArray(),
      target: orbit.target.toArray(),
    };
  }

  const eye = rec.group.getWorldPosition(new THREE.Vector3());
  const forward = new THREE.Vector3(0, 0, -1)      // three.js cameras look down -Z
    .applyQuaternion(rec.group.getWorldQuaternion(new THREE.Quaternion()));
  // Set the orbit target to whatever the sensor is pointed at.
  povRay.set(eye, forward);
  const hit = povRay.intersectObjects(collectCenterRayTargets(rec), true)[0];
  const standoff = hit ? Math.max(hit.distance, 0.3) : 2;

  camera.position.copy(eye);
  orbit.target.copy(eye).addScaledVector(forward, standoff);
  // Roll is dropped: OrbitControls re-derives orientation from position, target
  // and `up` every frame, so it would not survive the first drag anyway.
  camera.up.set(0, 0, 1);
  orbit.update();

  viewPreset = rec.name;
  applyCameraOverlayVisibility();
  refreshViewSwitcherActive();
  // A robot's own camera cannot be aimed, so the status says what it is instead.
  setStatus(rec.readOnly
    ? `Looking from ${rec.name}. It is the robot's own camera: it rides the arm `
      + 'at the pose this scene saved, so there is nothing here to aim.'
    : `Looking from ${rec.name}. The selection and the gizmo are still `
      + 'yours — Enter camera, on the Cameras tab, is the one that aims it.');
}

/** Back to the view this bar took you away from. */
function returnToFreeView() {
  if (flying) { exitFly(); return; }
  if (!viewPreset) { setStatus('Already in the free view.'); return; }
  const from = viewPreset;
  viewPreset = null;
  if (freeViewHome) {
    camera.position.fromArray(freeViewHome.position);
    orbit.target.fromArray(freeViewHome.target);
    orbit.update();
  }
  freeViewHome = null;
  applyCameraOverlayVisibility();
  refreshViewSwitcherActive();
  setStatus(`Back to the free view you were in before ${from}.`);
}

/** Drops the view preset; called whenever anything else moves the view. */
function clearViewPreset() {
  if (!viewPreset) return;
  viewPreset = null;
  freeViewHome = null;
  applyCameraOverlayVisibility();
  refreshViewSwitcherActive();
}

orbit.addEventListener('start', clearViewPreset);

function refreshViewSwitcher() {
  const bar = document.getElementById('view-switcher');
  const cams = [...objects.values()].filter((r) => r.isCamera);
  if (!cams.length) { bar.hidden = true; return; }
  // Drop a preset naming a camera that no longer exists (e.g. after a rig
  // import), so the Free view button lights correctly.
  if (viewPreset && !cams.some((r) => r.name === viewPreset)) {
    viewPreset = null;
    freeViewHome = null;
  }
  bar.hidden = false;
  bar.innerHTML = '';

  const addButton = (label, title, target, onClick) => {
    const button = document.createElement('button');
    button.textContent = label;
    button.title = title;
    // Sensor name that lights this button; empty for Free view.
    button.dataset.target = target || '';
    button.onclick = onClick;
    bar.appendChild(button);
  };

  addButton('Free view', 'Back to the view you were in before', '',
    () => returnToFreeView());

  const offered = new Set();
  const offer = (rec, label, note) => {
    if (offered.has(rec.name)) return;
    offered.add(rec.name);
    addButton(label,
      `Look from ${rec.name}${note ? ` (${note})` : ''} — the scene stays editable; `
      + 'this only moves you',
      rec.name, () => lookFrom(rec));
  };
  // Exteriors: found by policy input slot, labelled by the side of the room the
  // sensor stands on (the same lateral ordering the preview panes use). The
  // `_left` in a sensor's own name is its stereo eye, not a side.
  const exteriors = [];
  for (const slot of [1, 2]) {
    const wanted = `exterior_image_${slot}_left`;
    const rec = cams.find((r) => r.observation && r.observation.key === wanted);
    if (rec) exteriors.push({ rec, slot });
  }
  const robot = liveRecords().find((r) => r.entry.kind === 'robot');
  for (const ext of exteriors) ext.lateral = lateralOffset(ext.rec, robot);
  exteriors.sort((a, b) => (b.lateral - a.lateral) || (a.slot - b.slot));
  exteriors.forEach((ext, index) => {
    // A single exterior gets no side label.
    const side = exteriors.length > 1 ? (index === 0 ? ' left' : ' right') : '';
    offer(ext.rec, `Exterior${side}`, `exterior_image_${ext.slot}_left`);
  });
  // Wrist cameras: matched by observation key first, by name only as the
  // fallback for a rig that fills no observation map.
  const wrists = cams.filter((r) => (r.observation && r.observation.key === WRIST_KEY)
    || /wrist/i.test(r.name));
  for (const rec of wrists) {
    offer(rec, wrists.length > 1 ? wristLabel(rec.name) : 'Wrist',
      rec.observation ? rec.observation.key : null);
  }
  // With no observation map and no wrist, offer the first few cameras by name.
  if (!offered.size) for (const rec of cams.slice(0, 4)) offer(rec, rec.name);

  refreshViewSwitcherActive();
}

function refreshViewSwitcherActive() {
  const bar = document.getElementById('view-switcher');
  if (!bar || bar.hidden) return;
  for (const button of bar.querySelectorAll('button')) {
    const target = button.dataset.target;
    button.classList.toggle('on', target
      ? (!!flying && flying.name === target) || viewPreset === target
      : !flying && !viewPreset);
  }
}

/** Every mesh a viewport ray can hit; *exclude* skips the camera at the eye. */
function collectCenterRayTargets(exclude = flying) {
  const meshes = [];
  for (const rec of objects.values()) {
    // Presence, not visibility: a hidden background is still a real surface.
    if (rec === exclude || rec.present === false) continue;
    rec.group.traverse((child) => { if (child.isMesh) meshes.push(child); });
  }
  return meshes;
}

// Distance to whatever sits under the crosshair, for matching real standoff.
const centerRay = new THREE.Raycaster();
centerRay.far = 20;
let centerCheckedAt = 0;

function updateCenterDistance(now) {
  if (!flying || now - centerCheckedAt < 250) return;
  centerCheckedAt = now;
  // Re-collect targets only when the object set actually changed.
  if (flyTargetsStale) { flyTargets = collectCenterRayTargets(); flyTargetsStale = false; }
  centerRay.set(flyPos, flyForward());
  const hit = centerRay.intersectObjects(flyTargets, false)[0];
  document.getElementById('cam-center').textContent = hit
    ? `${hit.distance.toFixed(2)} m · ${hit.object.userData.owner || 'scene'}`
    : 'nothing within 20 m';
}

// Third-person inset shown while flying, so the flown camera's place in the
// room stays visible. A camera of its own; the orbit view is left untouched.
const overviewCam = new THREE.PerspectiveCamera(55, 16 / 9, 0.02, 200);
overviewCam.up.copy(WORLD_UP);
// The inset exists to show where you are standing, so it needs the furniture.
overviewCam.layers.enable(LAYER_OVERLAY);

function updateOverviewCam() {
  const forward = flyForward();
  overviewCam.position.copy(flyPos).addScaledVector(forward, -1.4).addScaledVector(WORLD_UP, 0.8);
  overviewCam.lookAt(_target.copy(flyPos).addScaledVector(forward, 1.2));
}

document.getElementById('btn-level').onclick = levelHorizon;

document.getElementById('btn-overview').onclick = (e) => {
  overviewOn = !overviewOn;
  e.target.classList.toggle('on', overviewOn);
};

const speedField = document.getElementById('fly-speed');
speedField.addEventListener('change', () => {
  const value = parseFloat(speedField.value);
  setFlySpeed(Number.isFinite(value) ? value : flySpeed);
  speedField.value = flySpeed.toFixed(2);
});

// --- auto-arrange ----------------------------------------------------------
// Scatters props around an anchor object on its support surface without
// intersecting; each press re-rolls the layout. Independent of the pipeline's
// distractor sampler, which needs live OmniGibson objects and cannot run here.

let anchorName = null;

// How far from its base the arm can usefully work, in metres.
const ROBOT_REACH_M = 0.95;

function propRecords() {
  return liveRecords().filter(
    (r) => !r.isCamera && r.entry.editable && r.name !== anchorName,
  );
}

// Default anchor: the object nearest the middle of the editable set's XY extent
// (mid-extent, not centroid, which biases towards wherever objects bunch up).
// Only a seed; one click overrides it.
function heuristicAnchor() {
  const candidates = liveRecords().filter(
    (r) => !r.isCamera && r.entry.editable);
  if (candidates.length === 0) return null;

  // Bounding-box centres, not origins: a pivot is rarely the middle of the mesh.
  const centres = candidates.map((r) => new THREE.Box3().setFromObject(r.group)
    .getCenter(new THREE.Vector3()));
  // XY only: a tall object is no less central than a flat one.
  const xs = centres.map((c) => c.x);
  const ys = centres.map((c) => c.y);
  const midX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const midY = (Math.min(...ys) + Math.max(...ys)) / 2;

  let best = null, bestDistance = Infinity;
  candidates.forEach((rec, i) => {
    const distance = Math.hypot(centres[i].x - midX, centres[i].y - midY);
    if (distance < bestDistance) { bestDistance = distance; best = rec; }
  });
  return best ? best.name : null;
}

function setAnchor(name) {
  anchorName = name;
  // The object list tags the anchor's row; there is no other readout.
  renderList();
}

const footprint = (rec) => {
  const box = new THREE.Box3().setFromObject(rec.group);
  return { box, size: box.getSize(new THREE.Vector3()) };
};

function arrangeProps({ radius, jitter = 0.35 }) {
  const anchor = objects.get(anchorName);
  if (!anchor) { setStatus('Set an anchor first.', 'err'); return; }
  const props = propRecords();
  if (props.length === 0) { setStatus('No props to arrange.', 'err'); return; }

  pushUndo([anchor, ...props], 'arrange');

  const anchorBox = new THREE.Box3().setFromObject(anchor.group);
  const supportZ = anchorBox.min.z;           // props sit level with the anchor's base
  const centre = anchor.group.position;

  // Keep props within the robot's reach when there is one.
  const robot = liveRecords().find((r) => r.entry.kind === 'robot');
  const reach = robot ? ROBOT_REACH_M : Infinity;

  const placed = [{ box: anchorBox }];
  const failed = [];
  // Even angular spread beats uniform-in-a-disc, which clumps.
  const step = (Math.PI * 2) / props.length;

  props.forEach((rec, index) => {
    const { box, size } = footprint(rec);
    const half = Math.max(size.x, size.y) / 2;
    let done = false;

    for (let attempt = 0; attempt < 40 && !done; attempt++) {
      const angle = step * index + (Math.random() - 0.5) * step * jitter * 2;
      const r = radius * (0.75 + Math.random() * 0.5) + half;
      const x = centre.x + Math.cos(angle) * r;
      const y = centre.y + Math.sin(angle) * r;
      if (robot && Math.hypot(x - robot.group.position.x, y - robot.group.position.y) > reach) continue;

      // Candidate AABB at the trial position, resting on the support plane.
      const cand = new THREE.Box3(
        new THREE.Vector3(x - size.x / 2, y - size.y / 2, supportZ),
        new THREE.Vector3(x + size.x / 2, y + size.y / 2, supportZ + size.z),
      );
      if (placed.some((p) => p.box.intersectsBox(cand))) continue;

      // Move by the delta between current and target box, so the object's own
      // offset from its origin is preserved.
      rec.group.position.x += x - box.getCenter(new THREE.Vector3()).x;
      rec.group.position.y += y - box.getCenter(new THREE.Vector3()).y;
      rec.group.position.z += supportZ - box.min.z;
      placed.push({ box: cand });
      refreshDirty(rec);
      done = true;
    }
    if (!done) {
      failed.push(rec.name);
      // An unplaced prop stays where it is, so it still occupies that space.
      placed.push({ box: new THREE.Box3().setFromObject(rec.group) });
    }
  });

  // Moved props may be selected; the group pivot must follow them.
  rebuildPivot();
  renderList();
  refreshReadout();
  const ok = props.length - failed.length;
  if (failed.length) {
    setStatus(`Arranged ${ok}/${props.length} around ${anchorName}. `
      + `No room for: ${failed.join(', ')}`, 'err');
  } else {
    setStatus(`Arranged ${ok} prop(s) around ${anchorName}. Press again to re-roll.`);
  }
}

// Rigidly shifts the whole layout so the anchor sits a fixed distance in front
// of the robot. XY translation only: relative offsets and Z stay untouched.
function recenterOnAnchor(distance) {
  const anchor = objects.get(anchorName);
  if (!anchor) { setStatus('Set an anchor first.', 'err'); return; }

  const robot = liveRecords().find((r) => r.entry.kind === 'robot');
  let target;
  if (robot) {
    // The robot's own +X, flattened into the ground plane.
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(robot.group.quaternion);
    forward.z = 0;
    if (forward.lengthSq() < 1e-9) forward.set(1, 0, 0);
    forward.normalize();
    target = robot.group.position.clone().addScaledVector(forward, distance);
  } else {
    target = new THREE.Vector3(0, 0, 0);
    setStatus('No robot in this scene — centring on the world origin instead.');
  }

  // Bounding-box centre, not origin: a pivot is rarely the middle of the mesh.
  const centre = new THREE.Box3().setFromObject(anchor.group).getCenter(new THREE.Vector3());
  const dx = target.x - centre.x;
  const dy = target.y - centre.y;

  const movers = [anchor, ...propRecords()];
  pushUndo(movers, 'recenter');
  movers.forEach((rec) => {
    rec.group.position.x += dx;
    rec.group.position.y += dy;
    refreshDirty(rec);
  });

  rebuildPivot();
  renderList();
  refreshReadout();
  setStatus(`Centred ${anchorName} ${distance.toFixed(2)} m in front of `
    + `${robot ? robot.name : 'the origin'}; ${movers.length} object(s) moved `
    + `by ${Math.hypot(dx, dy).toFixed(3)} m. Relative spacing unchanged.`);
}

// Strict bounded parse: values are metres and must be finite and in range.
function readDistance(inputId, { min, max, label }) {
  const raw = document.getElementById(inputId).value;
  const value = Number.parseFloat(raw);
  if (!Number.isFinite(value)) {
    setStatus(`${label} must be a number.`, 'err');
    return null;
  }
  if (value < min || value > max) {
    setStatus(`${label} must be between ${min} and ${max} m (got ${value}).`, 'err');
    return null;
  }
  return value;
}

document.getElementById('btn-recenter').onclick = () => {
  const distance = readDistance('recenter-dist', {
    min: 0.05, max: 5, label: 'Recenter distance',
  });
  if (distance === null) return;
  recenterOnAnchor(distance);
};

document.getElementById('btn-anchor').onclick = () => {
  // Only editable props may anchor: anchoring on the robot would let arrange
  // move the robot itself.
  if (!selected || selected.isCamera || !selected.entry.editable) {
    setStatus('Pick a prop to anchor on.', 'err');
    return;
  }
  setAnchor(selected.name);
  setStatus(`Anchor is now ${selected.name}.`);
};

document.getElementById('btn-arrange').onclick = () => {
  const radius = readDistance('arrange-radius', {
    min: 0.05, max: 3, label: 'Arrange radius',
  });
  if (radius === null) return;
  arrangeProps({ radius });
};

// --- layout warnings -------------------------------------------------------
// Geometry checks the server does not do: camera coverage, robot reach and
// prop overlap, computed against the config's real camera optics.

const _warnFrustum = new THREE.Frustum();
const _warnMatrix = new THREE.Matrix4();
const _warnBox = new THREE.Box3();
const _warnCentre = new THREE.Vector3();

function cameraCoverage(rec) {
  /** Which sensors see this object: 'all', 'some', or 'none'. */
  const cams = [...objects.values()].filter((r) => r.isCamera);
  if (cams.length === 0) return { seen: 'unknown', by: [] };
  _warnBox.setFromObject(rec.group);
  if (_warnBox.isEmpty()) return { seen: 'unknown', by: [] };
  const by = [];
  for (const cam of cams) {
    cam.proj.updateMatrixWorld();
    cam.proj.updateProjectionMatrix();
    _warnMatrix.multiplyMatrices(cam.proj.projectionMatrix, cam.proj.matrixWorldInverse);
    _warnFrustum.setFromProjectionMatrix(_warnMatrix);
    // Conservative test: "partly in frame" must not read as "out of frame".
    if (_warnFrustum.intersectsBox(_warnBox)) by.push(cam.name);
  }
  return { seen: by.length === cams.length ? 'all' : (by.length ? 'some' : 'none'), by };
}

function layoutWarnings() {
  const warnings = [];
  const props = liveRecords().filter((r) => !r.isCamera && r.entry.editable);
  const robot = liveRecords().find((r) => r.entry.kind === 'robot');

  for (const rec of props) {
    const coverage = cameraCoverage(rec);
    if (coverage.seen === 'none') {
      warnings.push({ name: rec.name, kind: 'unseen',
                      text: 'outside every camera frame' });
    } else if (coverage.seen === 'some') {
      warnings.push({ name: rec.name, kind: 'partly-seen',
                      text: `seen only by ${coverage.by.join(', ')}` });
    }
    if (robot) {
      _warnBox.setFromObject(rec.group).getCenter(_warnCentre);
      const distance = Math.hypot(_warnCentre.x - robot.group.position.x,
                                  _warnCentre.y - robot.group.position.y);
      if (distance > ROBOT_REACH_M) {
        warnings.push({ name: rec.name, kind: 'unreachable',
                        text: `${distance.toFixed(2)} m from the robot base `
                              + `(reach ~${ROBOT_REACH_M} m)` });
      }
    }
  }

  // Pairwise overlap between tight contact boxes (a world AABB on a turned prop
  // over-reports contact).
  const boxes = props.map((rec) => ({
    rec, box: contactBox(rec.group),
  })).filter((entry) => !entry.box.isEmpty());
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      if (boxes[i].box.intersectsBox(boxes[j].box)) {
        warnings.push({ name: boxes[i].rec.name, kind: 'overlap',
                        text: `overlaps ${boxes[j].rec.name}` });
      }
    }
  }
  return warnings;
}

// Keyed by object name for the list badges; recomputed on demand, not per frame.
let layoutIssues = new Map();

function refreshLayoutWarnings() {
  const warnings = layoutWarnings();
  layoutIssues = new Map();
  for (const w of warnings) {
    if (!layoutIssues.has(w.name)) layoutIssues.set(w.name, []);
    layoutIssues.get(w.name).push(w);
  }
  renderList();
  return warnings;
}

// --- tabs ------------------------------------------------------------------
// Objects and cameras write to different files, so they get separate tabs.

let activeTab = 'objects';

function setTab(tab) {
  if (tab === 'cameras' && !cameraConfig) return;
  activeTab = tab;
  document.getElementById('tab-objects').classList.toggle('on', tab === 'objects');
  document.getElementById('tab-cameras').classList.toggle('on', tab === 'cameras');
  const onCameras = tab === 'cameras';
  document.getElementById('camera-section').style.display = onCameras ? '' : 'none';
  document.getElementById('camera-save-section').style.display = onCameras ? '' : 'none';
  document.getElementById('scene-save-section').style.display = onCameras ? 'none' : '';
  document.getElementById('arrange-section').style.display = onCameras ? 'none' : '';
  // Add/Duplicate/Remove act on scene objects; cameras cannot be created or
  // deleted here.
  document.getElementById('object-actions').style.display = onCameras ? 'none' : '';
  // Update the label span, not the h2, which also carries the fold chevron.
  document.getElementById('objects-heading-label').textContent = onCameras ? 'Cameras' : 'Objects';
  // The library imports scene objects, which the cameras tab cannot hold.
  if (onCameras) closeLibrary();
  // Cameras are aimed from inside them, never with the gizmo.
  document.getElementById('gizmo-toolbar').style.display = onCameras ? 'none' : '';
  // Switching tabs drops the selection so the gizmo never holds a hidden record.
  exitFly();
  select(null);
  renderList();
}

document.getElementById('tab-objects').onclick = () => setTab('objects');
document.getElementById('tab-cameras').onclick = () => setTab('cameras');

function renderList() {
  const list = document.getElementById('objlist');
  list.innerHTML = '';
  // Sort: anchor first, plain objects, then robot/background; removed rows sink
  // to the bottom (a click restores them).
  const groupRank = (rec) => {
    if (rec.name === anchorName) return 0;
    const kind = rec.entry.kind;
    return (kind === 'robot' || kind === 'background') ? 2 : 1;
  };
  const rows = [...objects.values()].filter(
    (rec) => !!rec.isCamera === (activeTab === 'cameras'),
  ).sort((a, b) => (a.present === false) - (b.present === false)
    || groupRank(a) - groupRank(b));

  for (const rec of rows) {
    const gone = rec.present === false;
    const row = document.createElement('div');
    // `posable`, not `editable`: the robot is clickable and should not read as
    // locked.
    const posable = rec.entry.posable;
    // Every selection member is highlighted, not just the last one clicked.
    row.className = 'obj' + (selection.has(rec) ? ' sel' : '')
      + (posable ? '' : ' locked') + (gone ? ' gone' : '')
      + (rec.hiddenForView ? ' hidden-view' : '');
    // The name rides as data rather than being scraped from textContent.
    row.dataset.name = rec.name;
    if (rec.isCamera && rec.readOnly) {
      row.title = `Click to look from ${rec.name} — ${rec.readOnlyWhy
        || 'it belongs to the robot and cannot be moved here'}`;
    }
    if (rec.loadError) {
      row.classList.add('error');
      row.title = rec.loadError;
    }
    const label = document.createElement('span');
    if (dirty.has(rec.name) || cameraDirty.has(rec.name) || physicsDirty.has(rec.name)
        || jointsDirty.has(rec.name)) {
      const changed = document.createElement('span');
      changed.className = 'dirty';
      changed.textContent = '* ';
      label.appendChild(changed);
    }
    // Swatch matches the camera's frustum and preview-frame colour.
    if (rec.isCamera && rec.colour !== undefined) {
      const swatch = document.createElement('span');
      swatch.className = 'pip-dot';
      swatch.style.setProperty('--pip-color', hexColour(rec.colour));
      swatch.style.display = 'inline-block';
      swatch.style.marginRight = '6px';
      label.appendChild(swatch);
    }
    label.appendChild(document.createTextNode(rec.name));
    const tag = document.createElement('span');
    tag.className = 'tag';
    const kind = rec.name === anchorName ? 'anchor' : rec.entry.kind;
    if (gone) {
      tag.textContent = 'removed · restore';
      row.title = `${rec.name} will not be written. Click to restore it.`;
    } else {
      tag.textContent = rec.loadError ? `${kind} !` : (rec.added ? `${kind} +` : kind);
      // Layout warnings and settle drift badge the row directly.
      const issues = layoutIssues.get(rec.name);
      const drift = settleDrift.get(rec.name);
      const jointDrift = settleJointDrift.get(rec.name);
      const unverified = settleUnverified.get(rec.name);
      const notes = [];
      if (issues) notes.push(...issues.map((w) => w.text));
      if (drift !== undefined) notes.push(`settled ${drift.toFixed(3)} m away`);
      if (jointDrift) notes.push(`joints settled away: ${jointDrift}`);
      if (unverified) notes.push(`joints could not be verified: ${unverified}`);
      if (notes.length) {
        row.classList.add('warn');
        row.title = `${rec.name}: ${notes.join('; ')}`;
        tag.textContent = issues && issues.some((w) => w.kind === 'unseen')
          ? 'not in frame'
          : (drift !== undefined ? `moved ${drift.toFixed(2)}m`
            : (jointDrift ? 'joint moved' : (unverified ? 'unverified' : 'check')));
      }
    }
    // `.obj` is space-between on exactly two children; the tag and the eye ride
    // together as one.
    const right = document.createElement('span');
    right.className = 'obj-right';
    right.appendChild(tag);
    // View-only visibility toggle; it never reaches a save. Not for removed
    // rows or cameras, which have their own visibility controls.
    if (!gone && !rec.isCamera) {
      const eye = document.createElement('button');
      eye.className = 'eye';
      eye.textContent = rec.hiddenForView ? '🚫' : '👁';
      eye.title = rec.hiddenForView ? `Show ${rec.name} in the viewport` : `Hide ${rec.name} from the viewport`;
      eye.onclick = (event) => {
        event.stopPropagation();
        toggleRecordHidden(rec);
      };
      right.appendChild(eye);
    }
    row.append(label, right);
    if (gone) row.onclick = () => restoreRecord(rec);
    else if (rec.isCamera && rec.readOnly) row.onclick = () => lookFrom(rec);
    else if (posable) {
      row.onclick = (event) => {
        ctrlAlone = false;
        // The robot never joins a multi-select; setSelection would drop it.
        if (event.shiftKey && !rec.isCamera && rec.entry.editable) toggleSelected(rec);
        else select(rec);
      };
    }
    list.appendChild(row);
  }
}

// --- why a scale is refused -------------------------------------------------
// The one shared wording for why a robot's scale is locked. Declared before
// `setSelection`, which reads it.
const SCALE_LOCK_WHY = 'its joint frames, collision geometry and actuator '
  + 'limits are authored at this size';

// The scale controls' original tooltips from index.html, captured before
// anything rewrites them. `m-uniform` is absent: `setScaleLocked` owns its title.
const SCALE_CONTROL_TITLES = new Map(['s0', 's1', 's2', 'rst-scale', 'rst-size', 'm-scale']
  .map((id) => [id, document.getElementById(id).title]));

/** Records in the selection whose scale is server-locked — in practice, robots. */
function scaleLockedRecords() {
  return selectionRecords().filter((rec) => !rec.isCamera && !rec.entry.scalable);
}

/** The refusal to show for a scale nothing in the selection may take. */
function scaleRefusal() {
  const locked = scaleLockedRecords();
  return locked.length
    ? `${locked.map((rec) => rec.name).join(', ')}: scale is locked because `
      + `${SCALE_LOCK_WHY}.`
    : 'Select an object to scale.';
}

/** Whether a record is one the editor will let you take hold of at all. */
function canSelect(rec) {
  // `posable` is `editable`'s superset: every prop, plus the robot, which is
  // movable and rotatable but not scalable, duplicable or removable.
  return !!rec && rec.entry.posable && rec.present !== false;
}

/** The selection in the order it was built, which is the order it is shown. */
function selectionRecords() {
  return [...selection];
}

/**
 * Replace the selection wholesale. The sole writer of `selected` and sole
 * caller of gizmo.attach/detach for scene records. `enterCamera: false` lets
 * programmatic callers (undo, most obviously) select a camera without flying
 * it. Returns whether it acted.
 */
function setSelection(recs, { enterCamera = true } = {}) {
  // Never re-attach while a drag is live: TransformControls would teleport the
  // newcomer.
  if (gizmo.dragging) return false;
  const list = (Array.isArray(recs) ? recs : [recs]).filter(canSelect);
  // Any list that touches a camera collapses to that one camera.
  const firstCamera = list.find((rec) => rec.isCamera);
  // The robot never joins a group: a solo pick passes through, a multi-select
  // drops it.
  const members = firstCamera ? [firstCamera]
    : (list.length > 1 ? list.filter((rec) => rec.entry.editable) : list);

  // The table-placing marker yields the gizmo to a real selection.
  if (placingTable && members.length) setPlacingTable(false);

  // Close any open edit burst: its pending undo entry holds the old selection.
  closeEditBurst();

  selection.clear();
  for (const rec of members) selection.add(rec);
  selected = members.length ? members[members.length - 1] : null;

  if (members.length > 1) {
    rebuildPivot();
    gizmo.attach(pivot);
  } else if (selected && !selected.isCamera) {
    gizmo.attach(selected.group);
  } else {
    gizmo.detach();
  }

  const isCamera = !!(selected && selected.isCamera);
  // A deliberate camera pick enters the camera; programmatic selection does not.
  if (isCamera && enterCamera) enterFly(selected);
  else if (!isCamera && flying) exitFly();

  // A half-locked pick (the room or the robot) explains which half is locked.
  if (members.length === 1 && !isCamera && !selected.entry.editable) {
    setStatus(selected.entry.scalable
      // scalable-but-not-editable is the room; the other case is the robot.
      ? `${selected.name} is the room — move and scale it freely; the props `
        + 'stay where they are and it slides underneath them. It is left out '
        + 'of Select All, Duplicate and Remove.'
      : `${selected.name} is yours to place — its pose is a scene decision. `
        + `Scale, Duplicate and Remove stay locked: ${SCALE_LOCK_WHY}.`);
  }

  refreshSelectionUI();
  return true;
}

/** Make *rec* the whole selection; an unselectable *rec* changes nothing. */
function select(rec, opts) {
  if (rec && !canSelect(rec)) return false;
  return setSelection(rec ? [rec] : [], opts);
}

/** Select *rec* — except a read-only robot camera, which is looked from instead. */
function selectOrLookFrom(rec) {
  if (rec && rec.isCamera && rec.readOnly) { lookFrom(rec); return true; }
  return select(rec);
}

function clearSelection() {
  return setSelection([]);
}

/** Add a record to the selection, or take it back out. */
function toggleSelected(rec) {
  if (gizmo.dragging || !canSelect(rec) || rec.isCamera) return;
  // A camera selection cannot be extended; treat as a plain click.
  if (selected && selected.isCamera) { select(rec); return; }
  if (selection.has(rec)) {
    selection.delete(rec);
    setSelection(selectionRecords());
  } else {
    setSelection([...selection, rec]);
  }
}

/** Select every prop; the robot and the background are not `editable` and stay out. */
function selectAllProps() {
  const props = liveRecords().filter((rec) => !rec.isCamera && rec.entry.editable);
  // setSelection can refuse mid-drag; only report a change that happened.
  if (!setSelection(props)) return;
  setStatus(props.length
    ? `Selected ${props.length} objects — Esc to clear.`
    : 'Nothing to select.', props.length ? '' : 'err');
}

/** Updates everything the panel says about the current selection. */
function refreshSelectionUI() {
  const rec = selected;
  const many = selection.size > 1;
  const isCamera = !!(rec && rec.isCamera);
  // `full` is a prop (everything offered); `scalable` is a prop or the room.
  // The robot is neither. See iter_objects for the three-way split.
  const full = !!rec && !isCamera && !!rec.entry.editable;
  const scalable = !!rec && !isCamera && !!rec.entry.scalable;
  const off = (id, disabled) => { document.getElementById(id).disabled = disabled; };

  document.getElementById('selname').textContent = !rec
    ? '(none)'
    : (many ? `${selection.size} objects selected`
            : `${rec.name}  (${rec.entry.category})`);
  refreshPivotNote();
  updateHint();
  refreshPhysics();
  refreshJoints();

  off('btn-reset', !rec);
  // Drop only applies to fully editable props.
  off('btn-drop', !full);
  // Remove only applies to fully editable props.
  off('btn-remove', !full);
  // Duplicate works on one object at a time.
  const duplicate = document.getElementById('btn-duplicate');
  duplicate.disabled = !full || many;
  duplicate.title = many
    ? 'Duplicate works on one object at a time — click a row to select just it'
    : 'Import a second copy of the selected object (Ctrl+D)';
  // Blur a focused field before disabling it, or the caret sits in a dead box.
  if (!rec && isTyping()) document.activeElement.blur();
  for (const id of ['p0', 'p1', 'p2', 'bp0', 'bp1', 'bp2']) off(id, !rec);
  for (const id of ['s0', 's1', 's2']) off(id, !scalable);
  // Locked scale controls get a tooltip saying why; otherwise the original.
  const scaleWhy = rec && !isCamera && !scalable
    ? `${rec.name}: scale is locked because ${SCALE_LOCK_WHY}`
    : null;
  for (const [id, original] of SCALE_CONTROL_TITLES) {
    document.getElementById(id).title = scaleWhy || original;
  }
  // Size fields need a measurable box: an object whose proxy failed to load has
  // a scale but no known size.
  const measurable = selectionRecords().some(sizeable);
  for (const id of SIZE_FIELDS) off(id, !rec || isCamera || !measurable);
  off('m-size-base', !rec || isCamera || !measurable);
  // Copy size is one-object only, like the transform clipboard.
  off('btn-copy-size', !rec || isCamera || !measurable || many);
  refreshSizeClipboardNote();
  // The unit is a display setting; it stays usable with nothing selected.
  off('m-size-unit', false);
  // Rotation applies to cameras too: aiming one by typing a measured angle.
  for (const id of ['r0', 'r1', 'r2']) off(id, !rec);
  off('btn-upright', !rec);
  // Per-row resets and the scale lock follow the fields they belong to.
  off('rst-position', !rec);
  off('rst-robot', !rec);
  off('rst-rotation', !rec);
  off('rst-scale', !scalable);
  off('rst-size', !scalable || isCamera || !measurable);
  // Anchors are props only (see btn-anchor).
  off('btn-anchor', !full);
  // Group scale is always uniform (per-axis would shear; see applyPivotDelta).
  // Shown as on without touching the user's own setting.
  const uniform = document.getElementById('m-uniform');
  uniform.disabled = !scalable || many;
  uniform.classList.toggle('on', many || scaleLocked);
  uniform.title = many
    ? 'A group scale is always uniform — per-axis would shear the set. '
      + 'Select one object to stretch it.'
    : (scaleWhy || (scaleLocked
      ? 'Uniform scale on — typing or dragging one axis moves all three'
      : 'Uniform scale off — each axis is independent'));
  // Copy transform is one-object only.
  off('btn-copy-xform', !rec || many);
  refreshClipboardNote();
  // The gizmo's Scale mode writes straight to group.scale, so it is locked too;
  // if it was active, fall back to Move.
  off('m-scale', !scalable);
  // Only when something unscalable is actually selected; an empty selection
  // keeps the chosen mode.
  if (rec && !scalable && gizmo.getMode() === 'scale') setMode('translate');
  // A camera has no scale; its rows make room for the camera controls.
  document.getElementById('scale-row').style.display = isCamera ? 'none' : '';
  document.getElementById('size-row').style.display = isCamera ? 'none' : '';
  document.getElementById('sel-frame').style.display = isCamera ? '' : 'none';
  updateCameraButtons();
  renderList();
  refreshReadout();
}

// --- group drags -----------------------------------------------------------
// Each frame applies the delta measured from drag start to each member's
// start transform; deltas are never accumulated frame-to-frame.

let dragPivotStart = null;         // the pivot's frame when the drag began
let dragMembers = [];              // [{ rec, position, quaternion, scale }]

function beginGizmoDrag() {
  const recs = selectionRecords();
  if (!recs.length) return;
  // Snapshot once at drag start: one drag costs one undo entry.
  pushUndo(recs, `${gizmo.getMode()} drag`);
  if (gizmo.object === pivot) {
    dragPivotStart = {
      position: pivot.position.clone(),
      quaternion: pivot.quaternion.clone(),
      scale: pivot.scale.clone(),
    };
    dragMembers = recs.map((rec) => ({
      rec,
      position: rec.group.position.clone(),
      quaternion: rec.group.quaternion.clone(),
      scale: rec.group.scale.clone(),
    }));
  } else {
    // Where a locked scale drag has to measure its one ratio from.
    dragStartScale.copy(selected.group.scale);
  }
}

function endGizmoDrag() {
  dragPivotStart = null;
  dragMembers = [];
  // Recentre the pivot on the moved members.
  rebuildPivot();
}

const _deltaQuat = new THREE.Quaternion();
const _deltaOffset = new THREE.Vector3();

/**
 * Apply the pivot's delta since drag start to every selection member.
 *
 * Translate and rotate are rigid. Group scale is always uniform: a non-uniform
 * scale about a point on differently rotated members is a shear, which cannot
 * be expressed as position + quaternion + scale in the scene JSON.
 */
function applyPivotDelta() {
  if (!dragPivotStart || !dragMembers.length) return;
  const mode = gizmo.getMode();
  const start = dragPivotStart;

  if (mode === 'scale') enforceUniformScale(pivot, UNIT_SCALE);
  const k = pivot.scale.x / start.scale.x;
  // Check every component: dragging a Y or Z handle past the pivot can leave
  // X at 1 while another axis goes negative.
  const positive = [pivot.scale.x, pivot.scale.y, pivot.scale.z]
    .every((value) => Number.isFinite(value) && value > 0);
  if (mode === 'scale' && !(positive && Number.isFinite(k) && k > 0)) {
    // Skipping this frame leaves every member at the last good frame.
    setStatus('Scale must stay positive on every axis.', 'err');
    return;
  }
  _deltaQuat.copy(start.quaternion).invert().premultiply(pivot.quaternion);

  for (const m of dragMembers) {
    const g = m.rec.group;
    if (mode === 'translate') {
      g.position.copy(m.position).add(pivot.position).sub(start.position);
    } else if (mode === 'rotate') {
      _deltaOffset.copy(m.position).sub(start.position).applyQuaternion(_deltaQuat);
      g.position.copy(start.position).add(_deltaOffset);
      g.quaternion.copy(_deltaQuat).multiply(m.quaternion).normalize();
    } else {
      _deltaOffset.copy(m.position).sub(start.position).multiplyScalar(k);
      g.position.copy(start.position).add(_deltaOffset);
      g.scale.copy(m.scale).multiplyScalar(k);
    }
    m.rec.lastValidScale.copy(g.scale);
    // No syncFlyFromRecord: cameras never join a multi-selection.
  }

  markDirtyAll(dragMembers.map((m) => m.rec));
  refreshPivotNote();
  refreshReadout();
}

/** The tail every multi-record edit shares. */
function afterGroupEdit(recs, message) {
  markDirtyAll(recs);
  rebuildPivot();
  refreshReadout();
  if (message) setStatus(message);
}

const f3 = (v) => Array.from(v).map((x) => x.toFixed(3)).join('  ');

// Whether a keystroke belongs to a text field rather than to the editor.
function isTyping() {
  const active = document.activeElement;
  return !!active && !active.disabled
    && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA'
        || active.tagName === 'SELECT');
}

// Agreement between members is decided on the shown strings, not the numbers.
function setField(id, shown) {
  const el = document.getElementById(id);
  // Never overwrite a field the user is typing in.
  if (document.activeElement === el) return;
  if (shown.every((v) => v === shown[0])) {
    el.value = shown[0];
    el.placeholder = '';
    el.classList.remove('mixed');
  } else {
    // A number input discards a non-numeric .value, so the word goes in the
    // placeholder.
    el.value = '';
    el.placeholder = 'mixed';
    el.classList.add('mixed');
  }
}

const fixed4 = (v) => v.toFixed(4);

// --- the robot's frame -------------------------------------------------------
// The inspector shows a position twice: in the scene's world frame, and in the
// robot's base frame — the frame a task config's `workspace_bounds` is written
// in. The robot rows are derived views, never stored; saves send world
// coordinates.
const _robotQuat = new THREE.Quaternion();
const _robotScratch = new THREE.Vector3();

function robotBase() {
  return liveRecords().find((rec) => rec.entry.kind === 'robot') || null;
}

/** A world point in the robot's base frame, or null when the scene has no robot. */
function worldToRobot(position, out = new THREE.Vector3()) {
  const robot = robotBase();
  if (!robot) return null;
  return out.copy(position).sub(robot.group.position)
    .applyQuaternion(_robotQuat.copy(robot.group.quaternion).invert());
}

/** The inverse, for a number typed into the robot row. */
function robotToWorld(position, out = new THREE.Vector3()) {
  const robot = robotBase();
  if (!robot) return null;
  return out.copy(position).applyQuaternion(robot.group.quaternion)
    .add(robot.group.position);
}

/** Set one axis of a group's position, measured in the robot's base frame. */
function applyRobotField(group, index, value) {
  const local = worldToRobot(group.position, _robotScratch);
  if (!local) return;
  local.setComponent(index, value);
  robotToWorld(local, group.position);
}
// Degrees to 3 dp with trailing zeros stripped.
const degrees3 = (radians) => String(Number(THREE.MathUtils.radToDeg(radians).toFixed(3)));

function refreshReadout() {
  const recs = selectionRecords();
  if (!recs.length) {
    for (const id of TRANSFORM_FIELDS) {
      const el = document.getElementById(id);
      el.value = '';
      el.placeholder = '';
      el.classList.remove('mixed');
    }
    document.getElementById('v-ori').textContent = '-';
    refreshSizeReadout();
    return;
  }
  const groups = recs.map((rec) => rec.group);
  setField('p0', groups.map((g) => fixed4(g.position.x)));
  setField('p1', groups.map((g) => fixed4(g.position.y)));
  setField('p2', groups.map((g) => fixed4(g.position.z)));
  // Derived, never stored. Hidden when there is no robot, or the robot is
  // itself the selection.
  const robot = robotBase();
  const robotRow = document.getElementById('robot-pos-row');
  const showRobot = !!robot && !recs.includes(robot);
  robotRow.hidden = !showRobot;
  if (showRobot) {
    const local = groups.map((g) => worldToRobot(g.position));
    setField('bp0', local.map((v) => fixed4(v.x)));
    setField('bp1', local.map((v) => fixed4(v.y)));
    setField('bp2', local.map((v) => fixed4(v.z)));
  }
  setField('s0', groups.map((g) => fixed4(g.scale.x)));
  setField('s1', groups.map((g) => fixed4(g.scale.y)));
  setField('s2', groups.map((g) => fixed4(g.scale.z)));
  // group.quaternion is local; for a camera that local frame is the robot
  // link its config stores poses against.
  const eulers = groups.map((g) => {
    _readEuler.setFromQuaternion(g.quaternion, 'XYZ');
    return [_readEuler.x, _readEuler.y, _readEuler.z];
  });
  setField('r0', eulers.map((e) => degrees3(e[0])));
  setField('r1', eulers.map((e) => degrees3(e[1])));
  setField('r2', eulers.map((e) => degrees3(e[2])));
  const quats = groups.map((g) => f3(g.quaternion.toArray()));
  document.getElementById('v-ori').textContent =
    quats.every((q) => q === quats[0]) ? quats[0] : `mixed (${recs.length})`;
  // Last: the size row is a view of the scale row just written.
  refreshSizeReadout();
}

const TRANSFORM_FIELDS = ['p0', 'p1', 'p2', 'r0', 'r1', 'r2', 's0', 's1', 's2'];
// Labels used when a field has to refuse what was typed into it.
const TRANSFORM_FIELD_LABELS = {
  p0: 'Position X', p1: 'Position Y', p2: 'Position Z',
  bp0: 'Position X from the robot base', bp1: 'Position Y from the robot base',
  bp2: 'Position Z from the robot base',
  r0: 'Rotation X', r1: 'Rotation Y', r2: 'Rotation Z',
  s0: 'Scale X', s1: 'Scale Y', s2: 'Scale Z',
};
const _readEuler = new THREE.Euler();
const _writeEuler = new THREE.Euler();

// Degrees round-trip through the quaternion, so the value shown after an edit
// can differ from the value typed (190 comes back as -170).
function eulerDegrees(quaternion) {
  _readEuler.setFromQuaternion(quaternion, 'XYZ');
  return [_readEuler.x, _readEuler.y, _readEuler.z].map(THREE.MathUtils.radToDeg);
}

// Set one Euler axis (degrees) from typed entry.
function applyEulerField(group, index, degrees) {
  const current = eulerDegrees(group.quaternion);
  current[index] = degrees;
  _writeEuler.set(...current.map(THREE.MathUtils.degToRad), 'XYZ');
  group.quaternion.setFromEuler(_writeEuler);
}

// --- uniform scale ---------------------------------------------------------
// Optional lock that keeps scale edits uniform across all three axes. Off by
// default: deliberate stretching is still legitimate.

let scaleLocked = false;
// The scale a gizmo drag started from; a locked drag measures its ratio here.
const dragStartScale = new THREE.Vector3(1, 1, 1);

// Uniform when locked by the user, or forced by a multi-selection.
const uniformScale = () => scaleLocked || selection.size > 1;

function setScaleLocked(on) {
  scaleLocked = on;
  const button = document.getElementById('m-uniform');
  button.classList.toggle('on', scaleLocked);
  button.textContent = 'Uniform';
  button.title = scaleLocked
    ? 'Uniform scale on — typing or dragging one axis moves all three'
    : 'Uniform scale off — each axis is independent';
}

/**
 * Force a locked scale to stay uniform, measured from where the drag began.
 * `base` defaults to the object's pre-drag scale; a group drag passes (1,1,1)
 * because the pivot's scale is itself the ratio.
 */
function enforceUniformScale(group, base = dragStartScale) {
  const scale = group.scale;
  const start = base;
  if (![start.x, start.y, start.z].every((v) => Number.isFinite(v) && v > 0)) return;
  // The axis moved furthest, in ratio terms, drives the other two.
  const ratios = [scale.x / start.x, scale.y / start.y, scale.z / start.z];
  let ratio = 1;
  for (const candidate of ratios) {
    if (Number.isFinite(candidate) && Math.abs(candidate - 1) > Math.abs(ratio - 1)) {
      ratio = candidate;
    }
  }
  if (ratio <= 0) return;
  scale.set(start.x * ratio, start.y * ratio, start.z * ratio);
}

document.getElementById('m-uniform').onclick = () => {
  setScaleLocked(!scaleLocked);
  setStatus(scaleLocked
    ? 'Uniform scale on — one axis drives all three.'
    : 'Uniform scale off — axes are independent.');
};
setScaleLocked(false);

// --- size in metres --------------------------------------------------------
// The size row is the scale edit stated in metres: the object's bounding box
// is native size times scale, so a target size solves as scale = target /
// native, per axis. The shown size is recomputed from native x scale every
// time; nothing new is written to the scene JSON.

const SIZE_UNITS = [
  { name: 'm', per: 1, dp: 4, step: 0.005 },
  { name: 'cm', per: 100, dp: 2, step: 0.5 },
  { name: 'mm', per: 1000, dp: 1, step: 5 },
];
const SIZE_FIELDS = ['b0', 'b1', 'b2'];
// An axis thinner than this has no size to divide by (matches OmniGibson's
// `native_bbox > 1e-4` threshold).
const MIN_NATIVE_M = 1e-4;

let sizeUnit = 0;
// On by default: a resize about a reconstructed asset's arbitrary origin can
// lift a prop off its surface or sink it.
let sizeBasePlanted = true;

const _sizeBox = new THREE.Box3();
const _sizeChild = new THREE.Box3();
const _sizeVec = new THREE.Vector3();
const _sizeBefore = new THREE.Vector3();
const _sizeAfter = new THREE.Vector3();

/**
 * The asset's bounding box at scale 1, along its own axes, in metres.
 *
 * Prefers the extractor's declared measurement; falls back to measuring the
 * loaded proxy with the transform set aside. Returns null for a record with
 * no measurable geometry (a camera, or a proxy that failed to load).
 */
function nativeSize(rec) {
  if (rec.nativeSize) return rec.nativeSize;
  const declared = rec.entry && rec.entry.nativeSize;
  if (Array.isArray(declared) && declared.length === 3
      && declared.every((v) => Number.isFinite(v) && v >= 0)) {
    rec.nativeSize = new THREE.Vector3().fromArray(declared);
    return rec.nativeSize;
  }
  if (rec.isCamera) return null;
  // A null measurement is not cached: the proxy may still be loading.
  rec.nativeSize = measureNativeSize(rec);
  return rec.nativeSize;
}

/**
 * Measure a record's own-frame box off the loaded proxy. Counts only meshes
 * stamped with this record's name, so children owned by other records (e.g.
 * cameras parented to the robot) do not inflate the box.
 */
function measureNativeSize(rec) {
  const g = rec.group;
  _lcPos.copy(g.position); _lcQuat.copy(g.quaternion); _lcScale.copy(g.scale);
  try {
    g.position.set(0, 0, 0);
    g.quaternion.identity();
    g.scale.set(1, 1, 1);
    g.updateWorldMatrix(false, true);
    _sizeBox.makeEmpty();
    g.traverse((child) => {
      if (!child.isMesh || !child.geometry) return;
      if (child.userData.owner !== rec.name) return;
      if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
      _sizeChild.copy(child.geometry.boundingBox).applyMatrix4(child.matrixWorld);
      _sizeBox.union(_sizeChild);
    });
    if (_sizeBox.isEmpty()) return null;
    return _sizeBox.getSize(new THREE.Vector3());
  } finally {
    // Always restore the transform, even if the measurement throws.
    g.position.copy(_lcPos); g.quaternion.copy(_lcQuat); g.scale.copy(_lcScale);
    g.updateWorldMatrix(false, true);
  }
}

/** What the object measures right now, in metres: native size times scale. */
function currentSize(rec, out = new THREE.Vector3()) {
  const native = nativeSize(rec);
  if (!native) return null;
  return out.copy(native).multiply(rec.group.scale);
}

/** Whether a record can be sized: measurable geometry, not a camera, and
    flagged `scalable` (the robot is not; the room is). */
const sizeable = (rec) => !!rec && !rec.isCamera && rec.entry.scalable && !!nativeSize(rec);

/** A THREE.Vector3 as a plain array, or null — for the read-only check surface. */
const asArray = (v) => (v ? v.toArray() : null);

const unitOf = () => SIZE_UNITS[sizeUnit];
const toDisplay = (metres) => metres * unitOf().per;
const toMetres = (shown) => shown / unitOf().per;

function setSizeUnit(index) {
  sizeUnit = ((index % SIZE_UNITS.length) + SIZE_UNITS.length) % SIZE_UNITS.length;
  const unit = unitOf();
  const button = document.getElementById('m-size-unit');
  button.textContent = unit.name;
  const next = SIZE_UNITS[(sizeUnit + 1) % SIZE_UNITS.length].name;
  button.title = `Sizes are in ${{ m: 'metres', cm: 'centimetres', mm: 'millimetres' }[unit.name]}`
    + ` — click for ${next}`;
  for (const id of SIZE_FIELDS) {
    const el = document.getElementById(id);
    el.step = unit.step;
    el.setAttribute('aria-label',
      el.getAttribute('aria-label').replace(/(metres|centimetres|millimetres)$/,
        { m: 'metres', cm: 'centimetres', mm: 'millimetres' }[unit.name]));
  }
  refreshReadout();
}

function setSizeBasePlanted(on) {
  sizeBasePlanted = on;
  const button = document.getElementById('m-size-base');
  button.classList.toggle('on', sizeBasePlanted);
  button.title = sizeBasePlanted
    ? 'Base planted: the bottom of the box stays put, so a resized prop keeps '
      + 'standing on its surface. Click to resize about the asset\'s own origin instead.'
    : 'Resizing about the asset\'s own origin, like the scale row — a prop may '
      + 'sink or float. Click to keep its base planted.';
}

/** Why *rec* cannot be sized along local axis *index*, or null if it can.
    Checked before anything is written, so a refused edit pushes no undo. */
function sizeRefusal(rec, index, metres) {
  const native = nativeSize(rec);
  if (!native) return 'it has no geometry to measure';
  const along = native.getComponent(index);
  if (!(along > MIN_NATIVE_M)) {
    return `it is flat along ${'XYZ'[index]} — there is no size there to scale`;
  }
  if (!(Number.isFinite(metres / along) && metres / along > 0)) {
    return 'that size does not solve to a positive scale';
  }
  return null;
}

/**
 * Run *change*, which rewrites *rec*'s scale, and keep its base where it was.
 * Honours the base-planted toggle; with it off this is just `change()`.
 */
function withBasePlanted(rec, change) {
  // Where the base is now, before the scale changes underneath it.
  const planted = sizeBasePlanted && !contactBox(rec.group, _sizeBox).isEmpty();
  if (planted) {
    _sizeBox.getCenter(_sizeBefore);
    _sizeBefore.z = _sizeBox.min.z;
  }

  change();

  if (!planted) return;
  // Remeasured rather than derived: under a rotation the world-aligned box's
  // centre does not move by the scale ratio.
  rec.group.updateWorldMatrix(false, true);
  contactBox(rec.group, _sizeBox);
  if (_sizeBox.isEmpty()) return;
  _sizeBox.getCenter(_sizeAfter);
  _sizeAfter.z = _sizeBox.min.z;
  // Keep the footprint centred and resting on the same plane.
  rec.group.position.add(_sizeBefore.sub(_sizeAfter));
}

/** Resize *rec* so its box measures *metres* along local axis *index*. */
function applySizeToRecord(rec, index, metres) {
  const ratio = metres / nativeSize(rec).getComponent(index);
  withBasePlanted(rec, () => {
    if (uniformScale()) rec.group.scale.setScalar(ratio);
    else rec.group.scale.setComponent(index, ratio);
    rec.lastValidScale.copy(rec.group.scale);
  });
}

/** Note when the size columns are not the room's: size is along the object's
    own axes, so a yawed prop's X column can be the room's Y. */
function refreshSizeFrame() {
  const note = document.getElementById('size-frame');
  if (!note) return;
  const recs = selectionRecords().filter(sizeable);
  const turned = recs.filter((rec) => {
    _sizeVec.set(1, 0, 0).applyQuaternion(rec.group.quaternion);
    const x = Math.abs(_sizeVec.x);
    _sizeVec.set(0, 0, 1).applyQuaternion(rec.group.quaternion);
    // Aligned means each local axis lies along a world axis; checking X and Z
    // is enough, since Y is their cross product.
    return !(x > 0.999 || x < 0.001) || Math.abs(_sizeVec.z) < 0.999;
  });
  note.style.display = turned.length ? '' : 'none';
  if (!turned.length) return;
  note.textContent = turned.length === recs.length && recs.length === 1
    ? `${turned[0].name} is turned — size is along its own axes, not the room's`
    : `${turned.length} of these are turned — size is along each object's own axes`;
}

// --- when the sim will not honour the box ----------------------------------
// Two authored conditions break "the object's box is native x scale"; both are
// invisible in the viewer, so the panel warns (see extract.scale_fidelity).

const FIDELITY_NOTE = {
  'link-offset': (rec, detail) => {
    // Name at most three links.
    const links = detail.offsetLinks || [];
    const parts = links.slice(0, 3).join(', ')
      + (links.length > 3 ? ` and ${links.length - 3} more` : '');
    return `${rec.name} has parts (${parts}) that OmniGibson scales in place `
      + 'rather than with the body, so the size in the sim will differ from this '
      + 'one — a little near 1, a lot further away. Settle and measure before '
      + 'trusting it.';
  },
  'root-scale': (rec) =>
    `${rec.name}'s USD scales its own root, and OmniGibson overwrites the scale `
    + 'it is given for such an asset. This size is what the browser draws; the '
    + 'sim will load the object at its authored size.',
};

/** The single worst fidelity problem in the selection, phrased for the panel. */
function fidelityWarning(recs) {
  for (const kind of ['root-scale', 'link-offset']) {
    const hit = recs.find((rec) => {
      const detail = rec.entry && rec.entry.scaleFidelity;
      return detail && detail.kind === kind;
    });
    if (!hit) continue;
    const note = FIDELITY_NOTE[kind](hit, hit.entry.scaleFidelity);
    const others = recs.filter((rec) => rec.entry && rec.entry.scaleFidelity
      && rec.entry.scaleFidelity.kind !== 'linear').length - 1;
    return note + (others > 0 ? ` (${others} more like this in the selection.)` : '');
  }
  // A degraded proxy measures only the part of the asset that converted.
  const degraded = recs.find((rec) => rec.entry && rec.entry.status === 'degraded');
  return degraded
    ? `${degraded.name}'s proxy is missing prims, so this box is around only the `
      + 'part of it that converted.'
    : '';
}

function refreshSizeWarning() {
  const note = document.getElementById('size-warn');
  if (!note) return;
  const text = fidelityWarning(selectionRecords().filter(sizeable));
  note.style.display = text ? '' : 'none';
  note.textContent = text;
}

function refreshSizeReadout() {
  const recs = selectionRecords();
  const shown = recs.filter(sizeable).map((rec) => currentSize(rec, new THREE.Vector3()));
  for (let i = 0; i < SIZE_FIELDS.length; i += 1) {
    const el = document.getElementById(SIZE_FIELDS[i]);
    if (!shown.length) {
      if (document.activeElement !== el) {
        el.value = '';
        el.placeholder = '';
        el.classList.remove('mixed');
      }
      continue;
    }
    setField(SIZE_FIELDS[i],
      shown.map((v) => toDisplay(v.getComponent(i)).toFixed(unitOf().dp)));
  }
  refreshSizeFrame();
  refreshSizeWarning();
}

document.getElementById('m-size-unit').onclick = () => {
  setSizeUnit(sizeUnit + 1);
  setStatus(`Sizes shown in ${unitOf().name}.`);
};

document.getElementById('m-size-base').onclick = () => {
  setSizeBasePlanted(!sizeBasePlanted);
  setStatus(sizeBasePlanted
    ? 'Resizing keeps the base planted — a prop stays on its surface.'
    : 'Resizing about the asset\'s own origin — a prop may need dropping again.');
};

for (let index = 0; index < SIZE_FIELDS.length; index += 1) {
  const id = SIZE_FIELDS[index];
  const el = document.getElementById(id);
  // An empty "mixed" box steps from zero, so arrow keys are refused.
  el.addEventListener('keydown', (event) => {
    if (!el.classList.contains('mixed')) return;
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    setStatus('These differ — type a size to set it on all of them.', 'err');
  });
  el.addEventListener('blur', () => refreshReadout());
  el.addEventListener('change', () => {
    const recs = selectionRecords().filter(sizeable);
    if (!recs.length) {
      const chosen = selectionRecords();
      setStatus(!chosen.length
        ? 'Select an object to resize.'
        : (scaleLockedRecords().length
          ? scaleRefusal()
          : 'Nothing selected has a measurable box, so there is no size to set.'), 'err');
      refreshReadout();
      return;
    }
    const typed = parseFloat(el.value);
    if (!Number.isFinite(typed)) { refreshReadout(); return; }
    const metres = toMetres(typed);
    if (!(metres > 0)) {
      setStatus('A size must be greater than zero.', 'err');
      refreshReadout();
      return;
    }
    const refused = recs
      .map((rec) => ({ rec, why: sizeRefusal(rec, index, metres) }))
      .filter((r) => r.why);
    if (refused.length === recs.length) {
      setStatus(`Could not set that size — ${refused[0].rec.name}: ${refused[0].why}`,
                'err');
      refreshReadout();
      return;
    }
    const moving = recs.filter((rec) => !refused.some((r) => r.rec === rec));
    // One undo entry, for exactly the objects about to move.
    pushUndo(moving, 'size entry');
    for (const rec of moving) {
      applySizeToRecord(rec, index, metres);
      if (rec === flying) syncFlyFromRecord();
    }
    const axis = 'XYZ'[index];
    afterGroupEdit(moving, moving.length > 1
      ? `Set ${axis} to ${typed} ${unitOf().name} on ${moving.length} objects`
        + `${refused.length ? `; ${refused.length} refused, see the console` : ''}.`
      : `${moving[0].name} is now ${sizeSentence(moving[0])}`
        + `${sizeBasePlanted ? ', base planted.' : '.'}`);
    if (refused.length) {
      console.warn('size entry refused:',
                   refused.map((r) => `${r.rec.name}: ${r.why}`));
    }
  });
}

/** "73.0 x 73.0 x 65.6 mm" — the whole box, since one axis rarely moves alone. */
function sizeSentence(rec) {
  const size = currentSize(rec, _sizeVec);
  if (!size) return 'unmeasurable';
  const unit = unitOf();
  return `${[size.x, size.y, size.z]
    .map((v) => toDisplay(v).toFixed(unit.dp)).join(' x ')} ${unit.name}`;
}

setSizeUnit(0);
setSizeBasePlanted(true);

// --- copy and paste a transform --------------------------------------------
// Copies a transform via localStorage so it survives reloads and scene
// switches. localStorage rather than the system clipboard: the clipboard API
// needs a secure context, which a --host 0.0.0.0 bind is not.

const XFORM_CLIP_KEY = 'simfoundry.light-editor.transform';

function readTransformClip() {
  try {
    const raw = JSON.parse(localStorage.getItem(XFORM_CLIP_KEY) || 'null');
    if (!raw || !Array.isArray(raw.position) || !Array.isArray(raw.orientation)) return null;
    return raw;
  } catch {
    return null;
  }
}

function refreshClipboardNote() {
  const clip = readTransformClip();
  const note = document.getElementById('xform-clip');
  const paste = document.getElementById('btn-paste-xform');
  if (note) {
    note.textContent = clip
      ? `clipboard: ${clip.from}${clip.scene ? ` · ${clip.scene}` : ''}`
      : '';
    note.title = clip
      ? `${clip.from} from ${clip.scene || 'this scene'}\n`
        + `pos ${clip.position.map((v) => v.toFixed(3)).join(' ')}\n`
        + `quat ${clip.orientation.map((v) => v.toFixed(3)).join(' ')}`
      : '';
  }
  if (paste) {
    // Paste is single-object: one absolute pose on a set would stack them.
    const many = selection.size > 1;
    paste.disabled = !selected || !clip || many;
    paste.title = many
      ? 'Paste works on one object — one pose on a set would stack them at a point'
      : (clip ? `Paste the transform copied from ${clip.from}` : 'Nothing copied yet');
  }
}

function copyTransform() {
  if (!selected) return;
  const g = selected.group;
  const clip = {
    position: g.position.toArray(),
    orientation: g.quaternion.toArray(),
    scale: g.scale.toArray(),
    from: selected.name,
    // The scene name (not the filename), so a cross-scene paste names its source.
    scene: (activeManifest && activeManifest.scene_json || '')
      .split('/').pop().split('_scene_state_')[0],
    camera: !!selected.isCamera,
  };
  try {
    localStorage.setItem(XFORM_CLIP_KEY, JSON.stringify(clip));
  } catch {
    setStatus('Could not hold that transform — this browser refused storage.', 'err');
    return;
  }
  refreshClipboardNote();
  setStatus(`Copied ${selected.name}'s transform.`);
}

function pasteTransform() {
  const clip = readTransformClip();
  if (!selected || !clip) return;
  pushUndo(selected, 'paste transform');
  const g = selected.group;
  g.position.fromArray(clip.position);
  g.quaternion.fromArray(clip.orientation);
  // Scale is pasted only onto scalable non-cameras. `isCamera` is tested on
  // its own because a camera's entry is a stub that claims `editable`.
  const scaled = !selected.isCamera && !!selected.entry.scalable
    && Array.isArray(clip.scale)
    && clip.scale.every((v) => Number.isFinite(v) && v > 0);
  if (scaled) {
    g.scale.fromArray(clip.scale);
    selected.lastValidScale.copy(g.scale);
  }
  if (selected === flying) syncFlyFromRecord();
  markDirty();
  refreshReadout();
  setStatus(`Pasted ${clip.from}'s transform onto ${selected.name}`
    + (scaled ? '.' : ' (position and rotation only — scale is locked here).'));
}

document.getElementById('btn-copy-xform').onclick = copyTransform;
document.getElementById('btn-paste-xform').onclick = pasteTransform;
// Reflect a clipboard that survived a reload or scene switch.
refreshClipboardNote();

// --- size clipboard ---------------------------------------------------------
// A separate clipboard from the transform one above: pasting a size must not
// touch position or rotation.

const SIZE_CLIP_KEY = 'simfoundry.light-editor.size';

function readSizeClip() {
  try {
    const raw = JSON.parse(localStorage.getItem(SIZE_CLIP_KEY) || 'null');
    if (!raw || !Array.isArray(raw.size) || raw.size.length !== 3) return null;
    return raw;
  } catch {
    return null;
  }
}

function refreshSizeClipboardNote() {
  const clip = readSizeClip();
  const paste = document.getElementById('btn-paste-size');
  if (!paste) return;
  const many = selection.size > 1;
  const measurable = !!selected && !selected.isCamera && sizeable(selected) && nativeSize(selected);
  paste.disabled = !measurable || !clip || many;
  paste.title = many
    ? 'Paste works on one object — one size on a set would resize them all to it'
    : (clip ? `Paste the size copied from ${clip.from} (${clip.size.map((v) => v.toFixed(3)).join(' × ')} m)`
            : 'Nothing copied yet');
}

function copySize() {
  if (!selected || selected.isCamera || !sizeable(selected)) return;
  const size = currentSize(selected, new THREE.Vector3());
  if (!size) return;
  const clip = {
    size: size.toArray(),
    from: selected.name,
    scene: (activeManifest && activeManifest.scene_json || '')
      .split('/').pop().split('_scene_state_')[0],
  };
  try {
    localStorage.setItem(SIZE_CLIP_KEY, JSON.stringify(clip));
  } catch {
    setStatus('Could not hold that size — this browser refused storage.', 'err');
    return;
  }
  refreshSizeClipboardNote();
  setStatus(`Copied ${selected.name}'s size.`);
}

function pasteSize() {
  const clip = readSizeClip();
  const rec = selected;
  if (!rec || rec.isCamera || !clip || !sizeable(rec) || !nativeSize(rec)) return;
  // Same gate as the typed size row, before the undo entry: a flat axis or a
  // zero in the clipboard would otherwise solve to Infinity or collapse the
  // prop. Only the axes actually divided by are checked.
  const axes = uniformScale() ? [0] : [0, 1, 2];
  const why = axes.map((i) => sizeRefusal(rec, i, clip.size[i])).find(Boolean);
  if (why) {
    setStatus(`Could not paste that size — ${rec.name}: ${why}`, 'err');
    return;
  }
  pushUndo(rec, 'paste size');
  withBasePlanted(rec, () => {
    const native = nativeSize(rec);
    if (uniformScale()) {
      rec.group.scale.setScalar(clip.size[0] / native.getComponent(0));
    } else {
      for (let i = 0; i < 3; i += 1) {
        rec.group.scale.setComponent(i, clip.size[i] / native.getComponent(i));
      }
    }
    rec.lastValidScale.copy(rec.group.scale);
  });
  markDirty();
  refreshReadout();
  setStatus(`Pasted ${clip.from}'s size onto ${rec.name}.`);
}

document.getElementById('btn-copy-size').onclick = copySize;
document.getElementById('btn-paste-size').onclick = pasteSize;
refreshSizeClipboardNote();

// --- per-row reset ---------------------------------------------------------
// Each row resets on its own, to the same baseline Revert uses: the last
// transform this editor saved, or loaded if it has saved none.

const ROW_RESETS = {
  position: (g, initial) => g.position.fromArray(initial.position),
  rotation: (g, initial) => g.quaternion.fromArray(initial.orientation),
  scale: (g, initial) => g.scale.fromArray(initial.scale),
  // The size row is the scale row in metres, so its reset restores scale.
  size: (g, initial) => g.scale.fromArray(initial.scale),
};

function resetTransformRow(kind) {
  const recs = selectionRecords();
  if (!recs.length) return;
  pushUndo(recs, `reset ${kind}`);
  for (const rec of recs) {
    // Each record resets to its own baseline.
    ROW_RESETS[kind](rec.group, rec.initial);
    rec.lastValidScale.copy(rec.group.scale);
    if (rec === flying) syncFlyFromRecord();
    refreshDirty(rec);
  }
  rebuildPivot();
  renderList();
  refreshReadout();
  setStatus(recs.length > 1
    ? `Reset ${kind} on ${recs.length} objects to their last saved values.`
    : `Reset ${recs[0].name}'s ${kind} to the last saved value.`);
}

for (const kind of Object.keys(ROW_RESETS)) {
  document.getElementById(`rst-${kind}`).onclick = () => resetTransformRow(kind);
}
// The robot row is a view of position, so its reset is the position reset.
document.getElementById('rst-robot').onclick = () => resetTransformRow('position');

for (const [id, apply] of [
  ['p0', (g, v) => { g.position.x = v; }], ['p1', (g, v) => { g.position.y = v; }],
  ['p2', (g, v) => { g.position.z = v; }],
  // Position in the robot's base frame. The ids start with `b` because the
  // commit handler reads id[0] and `r` would file these under rotation.
  ['bp0', (g, v) => applyRobotField(g, 0, v)],
  ['bp1', (g, v) => applyRobotField(g, 1, v)],
  ['bp2', (g, v) => applyRobotField(g, 2, v)],
  // With uniform scale on (or a multi-selection), one typed number sets all
  // three axes.
  ['s0', (g, v) => (uniformScale() ? g.scale.setScalar(v) : (g.scale.x = v))],
  ['s1', (g, v) => (uniformScale() ? g.scale.setScalar(v) : (g.scale.y = v))],
  ['s2', (g, v) => (uniformScale() ? g.scale.setScalar(v) : (g.scale.z = v))],
  ['r0', (g, v) => applyEulerField(g, 0, v)],
  ['r1', (g, v) => applyEulerField(g, 1, v)],
  ['r2', (g, v) => applyEulerField(g, 2, v)],
]) {
  const el = document.getElementById(id);
  // A mixed field is empty, and an empty number box steps from zero — so the
  // arrow keys are refused rather than read as absolutes.
  el.addEventListener('keydown', (event) => {
    if (!el.classList.contains('mixed')) return;
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    setStatus('These differ — type a value to set it on all of them.', 'err');
  });
  // On leaving the field, show what the object actually holds.
  el.addEventListener('blur', () => refreshReadout());
  el.addEventListener('change', () => {
    // A typed number is absolute and lands on every selected member.
    const recs = selectionRecords();
    if (!recs.length) return;
    const v = parseFloat(el.value);
    if (!Number.isFinite(v)) {
      setStatus(`${TRANSFORM_FIELD_LABELS[id] || 'That'} needs a number — `
        + 'the field has been put back.', 'err');
      refreshReadout();
      return;
    }
    if (id.startsWith('s') && v <= 0) { setStatus('Scale must be positive.', 'err'); refreshReadout(); return; }
    const kind = id[0] === 's' ? 'scale' : (id[0] === 'r' ? 'rotation' : 'position');
    // Scale writes only to scalable non-cameras (the robot is posable but not
    // scalable); position and rotation write to every selected object.
    const writable = kind === 'scale'
      ? recs.filter((rec) => !rec.isCamera && rec.entry.scalable)
      : recs;
    if (!writable.length) { setStatus(scaleRefusal(), 'err'); refreshReadout(); return; }
    const skipped = recs.length - writable.length;
    pushUndo(writable, `${kind} entry`);
    for (const rec of writable) {
      apply(rec.group, v);
      rec.lastValidScale.copy(rec.group.scale);
      if (rec === flying) syncFlyFromRecord();
    }
    const left = skipped
      ? ` ${scaleLockedRecords().map((rec) => rec.name).join(', ')} left alone: `
        + `${SCALE_LOCK_WHY}.`
      : '';
    afterGroupEdit(writable, writable.length > 1 || skipped
      ? `Set ${kind} ${'XYZ'[+id[1]]} to ${v} on `
        + `${writable.length > 1 ? `${writable.length} objects` : writable[0].name}.${left}`
      : '');
  });
}

// --- nudging ---------------------------------------------------------------
// Arrows nudge along the world axes that best match their on-screen direction;
// Page Up/Down are vertical.

const NUDGE_KEYS = {
  ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'forward', ArrowDown: 'back',
  PageUp: 'up', PageDown: 'down',
};

const NUDGE_M = 0.005;          // one press
const NUDGE_COARSE = 0.05;      // with Shift
const NUDGE_FINE = 0.001;       // with Alt

const _nudgeDir = new THREE.Vector3();
const _nudgeRight = new THREE.Vector3();
const _nudgeFwd = new THREE.Vector3();

function nudgeSelection(direction, coarse, fine) {
  // Never move objects under a live gizmo drag.
  if (gizmo.dragging) return;
  const recs = selectionRecords().filter((rec) => !rec.isCamera);
  if (!recs.length) { setStatus('Select an object to nudge.', 'err'); return; }
  const step = fine ? NUDGE_FINE : (coarse ? NUDGE_COARSE : NUDGE_M);

  if (direction === 'up' || direction === 'down') {
    _nudgeDir.copy(WORLD_UP).multiplyScalar(direction === 'up' ? step : -step);
  } else {
    // The camera's right and forward, flattened into the ground plane and
    // snapped to the nearest world axis.
    camera.getWorldDirection(_nudgeFwd);
    _nudgeFwd.z = 0;
    if (_nudgeFwd.lengthSq() < 1e-9) _nudgeFwd.set(1, 0, 0);
    _nudgeFwd.normalize();
    _nudgeRight.crossVectors(_nudgeFwd, WORLD_UP).normalize().negate();
    const axis = (direction === 'left' || direction === 'right') ? _nudgeRight : _nudgeFwd;
    const sign = (direction === 'right' || direction === 'forward') ? 1 : -1;
    // Snap to the dominant world axis so repeated presses do not drift.
    _nudgeDir.set(Math.abs(axis.x) >= Math.abs(axis.y) ? Math.sign(axis.x) : 0,
                  Math.abs(axis.y) > Math.abs(axis.x) ? Math.sign(axis.y) : 0, 0)
             .multiplyScalar(step * sign);
  }

  // One undo entry per burst of presses.
  beginEditBurst(recs);
  for (const rec of recs) rec.group.position.add(_nudgeDir);
  afterGroupEdit(recs, recs.length > 1
    ? `${recs.length} objects nudged ${(step * 1000).toFixed(0)} mm.`
    : `${recs[0].name} nudged ${(step * 1000).toFixed(0)} mm.`);
}

// World-axis nudges, as distinct from the screen-aligned arrows above.
// Keyed [axis index, sign]: W/S are +/-X, A/D are +/-Y, Space is +Z.
const AXIS_NUDGE_KEYS = {
  w: [0, 1], s: [0, -1], a: [1, 1], d: [1, -1], ' ': [2, 1],
};
const AXIS_NAMES = ['X', 'Y', 'Z'];
const _axisNudge = new THREE.Vector3();

function nudgeAxis(axis, sign, coarse, fine) {
  // Never move objects under a live gizmo drag.
  if (gizmo.dragging) return;
  const recs = selectionRecords().filter((rec) => !rec.isCamera);
  if (!recs.length) {
    setStatus('Select an object to nudge.', 'err');
    return;
  }
  const step = fine ? NUDGE_FINE : (coarse ? NUDGE_COARSE : NUDGE_M);
  _axisNudge.set(0, 0, 0);
  _axisNudge.setComponent(axis, step * sign);

  beginEditBurst(recs);
  for (const rec of recs) rec.group.position.add(_axisNudge);
  const how = `${(step * 1000).toFixed(0)} mm ${sign > 0 ? '+' : '-'}${AXIS_NAMES[axis]}.`;
  afterGroupEdit(recs, recs.length > 1
    ? `${recs.length} objects nudged ${how}`
    : `${recs[0].name} nudged ${how}`);
}

// --- scaling from the keyboard ---------------------------------------------
// Quick keyboard scaling, whatever mode the gizmo is in. Multiplicative, so a
// step is the same gesture at any size and cannot walk a scale through zero.
// Keyed by character, so a numpad "+" and a shifted "=" are one press.
const SCALE_KEYS = { '+': 1, '=': 1, '-': -1, '_': -1 };
// Also keyed by physical key: on macOS Option+"-" arrives as an en dash, so a
// character-only binding would miss the fine step.
const SCALE_CODES = {
  Equal: 1, NumpadAdd: 1, Minus: -1, NumpadSubtract: -1,
};
const SCALE_STEP = 1.05;        // one press
const SCALE_FINE = 1.01;        // with Alt
// Soft limits for the keyboard step alone (OmniGibson asserts abs(scale) >
// 1e-4 on load); the gizmo, a typed size and a paste may all go further.
const SCALE_MIN = 0.001;
const SCALE_MAX = 1000;

const _scaleOffset = new THREE.Vector3();

/**
 * Scale the whole selection by one step. Always uniform (a non-uniform group
 * scale would shear; see applyPivotDelta). One object honours the base-planted
 * toggle; a set scales about its middle, like a group gizmo drag.
 */
function scaleSelection(up, fine) {
  // Never move objects under a live gizmo drag.
  if (gizmo.dragging) return;
  const recs = selectionRecords().filter((rec) => !rec.isCamera && rec.entry.scalable);
  if (!recs.length) { setStatus(scaleRefusal(), 'err'); return; }

  const step = fine ? SCALE_FINE : SCALE_STEP;
  // The step is clamped rather than the press refused, so an object approaching
  // the limit slows to a stop instead of the key going dead a step early.
  let allowed = up ? Infinity : 0;
  let stuck = null;
  for (const rec of recs) {
    const s = rec.group.scale;
    const edge = up ? Math.max(s.x, s.y, s.z) : Math.min(s.x, s.y, s.z);
    const room = up ? SCALE_MAX / edge : SCALE_MIN / edge;
    if (up ? room <= 1 : room >= 1) stuck = stuck || { rec, edge };
    allowed = up ? Math.min(allowed, room) : Math.max(allowed, room);
  }
  const k = up ? Math.max(1, Math.min(step, allowed))
               : Math.min(1, Math.max(1 / step, allowed));
  if (!Number.isFinite(k) || Math.abs(k - 1) < 1e-9) {
    const at = stuck ? `${stuck.rec.name} is at ${stuck.edge.toPrecision(3)}×` : 'It is';
    setStatus(`${at} — the keyboard step stops at `
      + `${up ? SCALE_MAX : SCALE_MIN}×. The gizmo and the size fields do not.`, 'err');
    return;
  }

  beginEditBurst(recs, 'scale');
  if (recs.length === 1) {
    // Same path as the size fields, so both honour base-planted.
    withBasePlanted(recs[0], () => { recs[0].group.scale.multiplyScalar(k); });
  } else {
    // About the middle of the set: members spread apart as they grow.
    rebuildPivot();
    for (const rec of recs) {
      const g = rec.group;
      _scaleOffset.copy(g.position).sub(pivot.position).multiplyScalar(k);
      g.position.copy(pivot.position).add(_scaleOffset);
      g.scale.multiplyScalar(k);
    }
  }
  // Keep lastValidScale current; a rejected gizmo drag restores from it.
  for (const rec of recs) rec.lastValidScale.copy(rec.group.scale);

  const how = `×${k.toFixed(3)}`;
  afterGroupEdit(recs, recs.length > 1
    ? `${recs.length} objects scaled ${how} about their middle.`
    : `${recs[0].name} scaled ${how} — now ${f3(recs[0].group.scale)}.`);
}

// Ctrl pressed and released alone nudges -Z. It fires on keyup so Ctrl+key
// combos are not also read as nudges.
let ctrlAlone = false;
let ctrlModifiers = null;

function armCtrlNudge(e) {
  ctrlAlone = true;
  ctrlModifiers = { coarse: e.shiftKey, fine: e.altKey };
}

let editBurstOpen = false;
let editBurstTimer = null;
let editBurstKind = null;
// Opens (or extends) a single undo entry for a burst of repeated edits.
// Changing `kind` starts a fresh entry.
function beginEditBurst(recs, kind = 'nudge') {
  if (!editBurstOpen || editBurstKind !== kind) {
    pushUndo(recs, kind);
    editBurstOpen = true;
    editBurstKind = kind;
  }
  clearTimeout(editBurstTimer);
  editBurstTimer = setTimeout(() => { editBurstOpen = false; }, 600);
}

// Close the burst when the selection changes.
function closeEditBurst() {
  clearTimeout(editBurstTimer);
  editBurstOpen = false;
  editBurstKind = null;
}

// F frames the selection; Shift+F frames the whole scene.
function frameSelection() {
  clearViewPreset();
  const recs = selectionRecords().filter((rec) => !rec.isCamera);
  if (!recs.length) { frameScene(); return; }
  const box = new THREE.Box3();
  for (const rec of recs) box.expandByObject(rec.group);
  if (box.isEmpty()) { frameScene(); return; }
  const centre = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() * 1.4, 0.25);
  orbit.target.copy(centre);
  camera.position.copy(centre).add(new THREE.Vector3(radius, -radius, radius * 0.8));
  orbit.update();
  setStatus(recs.length > 1
    ? `Framed ${recs.length} objects. Shift+F frames the whole scene.`
    : `Framed ${recs[0].name}. Shift+F frames the whole scene.`);
}

/**
 * Where to stand back from *centre*, at *radius*, to look at the whole scene.
 *
 * (+r, -r, 0.8r) normally. A Gaussian-splat room only renders well from the
 * directions its scan saw, so with a splat present the view comes in on the
 * exterior sensors' mean bearing instead, at the same distance and height.
 */
function overviewOffset(centre, radius) {
  const offset = new THREE.Vector3(radius, -radius, radius * 0.8);
  const splat = [...objects.values()].some((rec) => rec.splat && rec.present !== false);
  if (!splat) return offset;
  const sensors = [...objects.values()].filter((rec) => rec.isCamera);
  if (!sensors.length) return offset;
  const bearing = new THREE.Vector3();
  for (const rec of sensors) bearing.add(rec.group.getWorldPosition(new THREE.Vector3()));
  bearing.divideScalar(sensors.length).sub(centre).setZ(0);
  // Sensors sitting over the centre give no bearing at all; the fixed one stands.
  if (bearing.lengthSq() < 1e-6) return offset;
  bearing.normalize().multiplyScalar(radius * Math.SQRT2);
  return offset.set(bearing.x, bearing.y, radius * 0.8);
}

function frameScene() {
  clearViewPreset();
  const box = new THREE.Box3();
  for (const rec of liveRecords()) {
    if (rec.entry.kind !== 'background') box.expandByObject(rec.group);
  }
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() * 0.7, 0.5);
  orbit.target.copy(center);
  camera.position.copy(center).add(overviewOffset(center, radius));
  orbit.update();
}

// --- picking ---------------------------------------------------------------
const raycaster = new THREE.Raycaster();
// A Raycaster tests layer 0 by default. Camera bodies are overlay geometry, so
// without this they stop being clickable the moment they move off layer 0.
raycaster.layers.enable(LAYER_OVERLAY);
const pointer = new THREE.Vector2();
let downAt = null;
// Set the moment the gizmo takes a press, cleared by the release that follows.
let gizmoInteracted = false;

// Dragging while flying a camera aims it; picking gives way.
let lookingFrom = null;

renderer.domElement.addEventListener('pointerdown', (e) => {
  downAt = { x: e.clientX, y: e.clientY };
  // A click is not a bare Ctrl press: disarm the Ctrl nudge.
  ctrlAlone = false;
  // TransformControls has already set `dragging` if it took this press; a
  // press it did not take clears any latch left from a previous interaction.
  if (!gizmo.dragging) gizmoInteracted = false;
  if (!flying || e.button !== 0) return;
  lookingFrom = { x: e.clientX, y: e.clientY };
  // Capture keeps a swing going when the pointer leaves the letterboxed canvas.
  // Not every pointer id can be captured, and failing to is not worth aborting.
  try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* ignore */ }
});

renderer.domElement.addEventListener('pointermove', (e) => {
  if (placing) { updatePlacement(e); return; }
  if (!flying || !lookingFrom) return;
  // Drag right to swing right, drag up to tilt up.
  look(
    -(e.clientX - lookingFrom.x) * LOOK_RADIANS_PER_PIXEL,
    -(e.clientY - lookingFrom.y) * LOOK_RADIANS_PER_PIXEL,
  );
  lookingFrom = { x: e.clientX, y: e.clientY };
});

for (const type of ['pointerup', 'pointercancel']) {
  renderer.domElement.addEventListener(type, (e) => {
    if (!lookingFrom) return;
    lookingFrom = null;
    if (renderer.domElement.hasPointerCapture(e.pointerId)) {
      renderer.domElement.releasePointerCapture(e.pointerId);
    }
    endFlyEdit();
  });
}

// OrbitControls only swallows the context menu while it is enabled, which it is
// not in here.
renderer.domElement.addEventListener('contextmenu', (e) => { if (flying) e.preventDefault(); });

// In camera view the wheel sets fly speed instead of zooming.
renderer.domElement.addEventListener('wheel', (e) => {
  // Zooming out of a sensor's viewpoint is leaving it, same as orbiting away.
  if (!flying) { clearViewPreset(); return; }
  e.preventDefault();
  setFlySpeed(flySpeed * (e.deltaY < 0 ? 1.25 : 0.8));
  setStatus(`Speed ${flySpeed.toFixed(2)} m/s.`);
}, { passive: false });

renderer.domElement.addEventListener('pointerup', (e) => {
  // A click while placing commits, and must not also be read as a pick.
  if (placing) {
    if (downAt && Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) <= 4) {
      commitPlacement();
    }
    downAt = null;
    return;
  }
  // The release that ends any gizmo interaction, however short, is that
  // interaction's release — never a pick.
  if (gizmoInteracted) { gizmoInteracted = false; downAt = null; return; }
  if (flying || gizmo.dragging || !downAt) return;
  // Ignore camera drags; only a near-stationary click counts as a pick.
  if (Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return;

  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  // Remember a hit on the room so the deselect below can explain itself.
  let roomHit = null;

  for (const hit of raycaster.intersectObjects(scene.children, true)) {
    const owner = hit.object.userData.owner;
    if (!owner) continue;
    const rec = objects.get(owner);
    // Only pick what the active tab lists, so a stray click cannot attach the
    // gizmo to something the panel is not showing.
    if (!rec || !rec.entry.posable || rec.present === false) continue;
    if (!!rec.isCamera !== (activeTab === 'cameras')) continue;
    // The room is posable but not pickable here: every missed click lands on
    // it, and a stray drag would slide the whole scan. Select it from the list.
    if (rec.entry.kind === 'background') { roomHit = roomHit || rec; continue; }
    // Raycasting ignores `visible`; skip hidden cameras.
    if (rec.isCamera && !cameraOverlaysOn) continue;
    // Same for records hidden by the eye toggle.
    if (rec.hiddenForView) continue;
    // Shift extends the selection; the robot never joins a multi-select.
    if (e.shiftKey && !rec.isCamera && rec.entry.editable) toggleSelected(rec);
    else selectOrLookFrom(rec);
    return;
  }
  // A plain click on nothing deselects; a Shift+click on nothing does not.
  if (e.shiftKey) return;
  const had = selection.size;
  select(null);
  if (roomHit) {
    setStatus(`${had ? 'Deselected. ' : ''}That click landed on ${roomHit.name}, `
      + 'the room. It is not selectable from the viewport, because one stray drag '
      + 'there would slide the whole scan — its row in the object list selects it.');
  }
});

// --- gizmo controls --------------------------------------------------------
const modeButtons = { translate: 'm-translate', rotate: 'm-rotate', scale: 'm-scale' };
let currentSpace = 'world';

function setMode(mode) {
  gizmo.setMode(mode);
  for (const [m, id] of Object.entries(modeButtons)) {
    document.getElementById(id).classList.toggle('on', m === mode);
  }
  // Scaling is only meaningful in the object's own frame.
  gizmo.setSpace(mode === 'scale' ? 'local' : currentSpace);
}
for (const [mode, id] of Object.entries(modeButtons)) {
  document.getElementById(id).onclick = () => setMode(mode);
}

document.getElementById('m-world').onclick = (e) => {
  currentSpace = currentSpace === 'world' ? 'local' : 'world';
  e.target.textContent = currentSpace === 'world' ? 'World' : 'Local';
  e.target.classList.toggle('on', currentSpace === 'world');
  if (gizmo.getMode() !== 'scale') gizmo.setSpace(currentSpace);
};

// --- the gizmo toolbar goes where you put it -------------------------------
// The toolbar is draggable and remembers where it was left. The whole bar is
// the handle, buttons included: a press on a button stays a click until it
// has travelled `PIP_DRAG_SLOP` pixels.

const GIZMO_POS_KEY = 'simfoundry.light-editor.gizmo-toolbar';
const GIZMO_PAD = 6;                 // px of viewport it may not be pushed past

const gizmoBar = document.getElementById('gizmo-toolbar');
// Its original spot in the strip, so "put it back" is exact.
const gizmoHome = { parent: gizmoBar.parentNode, next: gizmoBar.nextSibling };
let gizmoPlaced = null;              // {x, y} once moved, else null

function clampGizmoBar(at) {
  const rect = gizmoBar.getBoundingClientRect();
  return {
    x: Math.round(THREE.MathUtils.clamp(
      at.x, GIZMO_PAD, Math.max(GIZMO_PAD, viewport.clientWidth - rect.width - GIZMO_PAD))),
    y: Math.round(THREE.MathUtils.clamp(
      at.y, GIZMO_PAD, Math.max(GIZMO_PAD, viewport.clientHeight - rect.height - GIZMO_PAD))),
  };
}

function applyGizmoPlacement() {
  if (!gizmoPlaced) return;
  gizmoPlaced = clampGizmoBar(gizmoPlaced);
  gizmoBar.style.setProperty('--gizmo-left', `${gizmoPlaced.x}px`);
  gizmoBar.style.setProperty('--gizmo-top', `${gizmoPlaced.y}px`);
}

/** Take the bar out of the top strip and place it at a viewport coordinate. */
function placeGizmoBar(at) {
  if (gizmoBar.parentNode !== viewport) viewport.appendChild(gizmoBar);
  gizmoBar.classList.add('placed');
  gizmoPlaced = at;
  applyGizmoPlacement();
}

/** Put it back in the strip, centred, and forget the saved position. */
function resetGizmoBar() {
  gizmoBar.classList.remove('placed');
  gizmoBar.style.removeProperty('--gizmo-left');
  gizmoBar.style.removeProperty('--gizmo-top');
  gizmoHome.parent.insertBefore(gizmoBar, gizmoHome.next);
  gizmoPlaced = null;
  try { localStorage.removeItem(GIZMO_POS_KEY); } catch { /* best effort */ }
  setStatus('Gizmo bar back at the top.');
}

function saveGizmoPlacement() {
  if (!gizmoPlaced) return;
  try {
    localStorage.setItem(GIZMO_POS_KEY, JSON.stringify(gizmoPlaced));
  } catch {
    // Placement still works without localStorage.
  }
}

function restoreGizmoPlacement() {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(GIZMO_POS_KEY) || 'null');
  } catch { saved = null; }
  if (!saved || !Number.isFinite(saved.x) || !Number.isFinite(saved.y)) return;
  // Clamped on the way in, so a position saved on a wider window stays reachable.
  placeGizmoBar(saved);
}

// When the last drag ended; a click just after it is the drag's own release.
let draggedAt = 0;
const DRAG_CLICK_MS = 250;

gizmoBar.addEventListener('click', (event) => {
  if (performance.now() - draggedAt > DRAG_CLICK_MS) return;
  draggedAt = 0;
  event.preventDefault();
  event.stopPropagation();
}, true);

gizmoBar.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  const rect = gizmoBar.getBoundingClientRect();
  const host = viewport.getBoundingClientRect();
  const grab = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const from = { x: event.clientX, y: event.clientY };
  let moved = false;
  // Listened for on `window`, not with setPointerCapture: reparenting the bar
  // mid-drag would drop its pointer capture.
  const move = (e) => {
    if (!moved && Math.hypot(e.clientX - from.x, e.clientY - from.y) <= PIP_DRAG_SLOP) return;
    if (!moved) {
      moved = true;
      gizmoBar.classList.add('dragging');
    }
    e.preventDefault();
    placeGizmoBar({ x: e.clientX - host.left - grab.x, y: e.clientY - host.top - grab.y });
  };

  const finish = () => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', finish);
    window.removeEventListener('pointercancel', finish);
    if (!moved) return;              // a press that never travelled is a click
    gizmoBar.classList.remove('dragging');
    saveGizmoPlacement();
    // Timestamp so the click that ends this drag is swallowed instead of
    // landing on whichever button the pointer stopped over.
    draggedAt = performance.now();
  };

  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', finish);
  window.addEventListener('pointercancel', finish);
});

// Double-click on the bar's own background puts it back at the top.
gizmoBar.addEventListener('dblclick', (event) => {
  if (event.target !== gizmoBar) return;
  if (gizmoPlaced) resetGizmoBar();
});

restoreGizmoPlacement();

let snapOn = false;
document.getElementById('m-snap').onclick = (e) => {
  snapOn = !snapOn;
  gizmo.setTranslationSnap(snapOn ? 0.01 : null);
  gizmo.setRotationSnap(snapOn ? THREE.MathUtils.degToRad(15) : null);
  gizmo.setScaleSnap(snapOn ? 0.05 : null);
  e.target.textContent = snapOn ? 'Snap: 1cm/15°' : 'Snap: off';
  e.target.classList.toggle('on', snapOn);
};

document.getElementById('m-bg').onclick = () => {
  // Drive every background to one shared state, not a per-record flip.
  const backgrounds = [...objects.values()].filter((r) => r.entry.kind === 'background');
  const hide = backgrounds.some((r) => !r.hiddenForView);
  for (const rec of backgrounds) {
    rec.hiddenForView = hide;
    updateVisibility(rec);
  }
  refreshBackgroundToggle();
  renderList();
};

function toggleCameraOverlays() {
  cameraOverlaysOn = !cameraOverlaysOn;
  applyCameraOverlayVisibility();
  setStatus(cameraOverlaysOn
    ? 'Camera frustums shown.'
    : 'Camera frustums hidden — press C or Cams to bring them back.');
}

document.getElementById('m-cams').onclick = toggleCameraOverlays;

// Grid and origin-axes toggle. A view setting only: both live on the overlay
// layer, so hiding them changes nothing about what is saved or what a sensor
// renders.
let gridOn = true;

function setGridVisible(on) {
  gridOn = on;
  grid.visible = gridOn;
  originAxes.visible = gridOn;
  // Lit means shown, same as every other button in the Visibility section.
  document.getElementById('m-grid').classList.toggle('on', gridOn);
}

function toggleGrid() {
  setGridVisible(!gridOn);
  setStatus(gridOn
    ? 'Ground grid and origin axes shown.'
    : 'Grid hidden — Shift+G or Grid brings it back.');
}

document.getElementById('m-grid').onclick = toggleGrid;

// A wireframe Box3Helper per live prop, drawn in world space. Off by default —
// a debug aid, not part of the shot.
let boxesOn = false;
// record name -> { helper, sig }: `sig` is the transform the box was measured
// at; a null helper marks a proxy with no geometry.
const boxHelpers = new Map();
const _boxMeasured = new THREE.Box3();

/**
 * Whether *rec* has moved since its box was measured, recording that it has.
 * The signature is the group's position, quaternion and scale — nothing in
 * this file moves a prop any other way. A signature of NaNs never compares
 * equal, so a newly seen record measures on its first frame.
 */
function boxTransformChanged(rec, sig) {
  const { position: p, quaternion: q, scale: s } = rec.group;
  if (sig[0] === p.x && sig[1] === p.y && sig[2] === p.z
      && sig[3] === q.x && sig[4] === q.y && sig[5] === q.z && sig[6] === q.w
      && sig[7] === s.x && sig[8] === s.y && sig[9] === s.z) {
    return false;
  }
  sig[0] = p.x; sig[1] = p.y; sig[2] = p.z;
  sig[3] = q.x; sig[4] = q.y; sig[5] = q.z; sig[6] = q.w;
  sig[7] = s.x; sig[8] = s.y; sig[9] = s.z;
  return true;
}

// Called from the frame loop; measures only records whose transform changed.
// contactBox walks a proxy's vertices, which is too slow to do unconditionally.
function updateBoundingBoxes() {
  if (!boxesOn) return;
  const seen = new Set();
  for (const rec of liveRecords()) {
    if (rec.isCamera || rec.entry.kind === 'robot' || rec.entry.kind === 'background') continue;
    // No box around a hidden record.
    if (rec.hiddenForView) continue;
    seen.add(rec.name);
    let entry = boxHelpers.get(rec.name);
    if (!entry) {
      entry = { helper: null, sig: new Float64Array(10).fill(NaN) };
      boxHelpers.set(rec.name, entry);
    }
    if (!boxTransformChanged(rec, entry.sig)) continue;
    // Precise mode: the conservative default over-boxes a rotated mesh.
    const box = contactBox(rec.group, _boxMeasured);
    if (box.isEmpty()) continue;
    if (!entry.helper) {
      // Cloned: the helper keeps the box it is handed, and this one is scratch.
      entry.helper = new THREE.Box3Helper(box.clone(), 0x76b900);
      // Overlay layer, so it never appears in the exterior camera previews.
      markAsOverlay(entry.helper);
      scene.add(entry.helper);
    } else {
      entry.helper.box.copy(box);
    }
  }
  for (const [name, entry] of boxHelpers) {
    if (!seen.has(name)) {
      if (entry.helper) scene.remove(entry.helper);
      boxHelpers.delete(name);
    }
  }
}

function setBoxesVisible(on) {
  boxesOn = on;
  document.getElementById('m-boxes').classList.toggle('on', boxesOn);
  if (boxesOn) {
    // Measure everything fresh: boxes are not maintained while off.
    updateBoundingBoxes();
  } else {
    for (const entry of boxHelpers.values()) {
      if (entry.helper) scene.remove(entry.helper);
    }
    boxHelpers.clear();
  }
}

function toggleBoxes() {
  setBoxesVisible(!boxesOn);
  setStatus(boxesOn
    ? "Bounding boxes shown for every object."
    : 'Bounding boxes hidden.');
}

document.getElementById('m-boxes').onclick = toggleBoxes;

/**
 * Stand the selection up without turning it around: roll and pitch go, yaw
 * survives. `ZYX` order reads the yaw about world up first, so exactly the
 * tilt is discarded. Shift clears the whole rotation instead.
 */
document.getElementById('btn-upright').onclick = (event) => {
  const recs = selectionRecords();
  if (!recs.length) return;
  const clearAll = event.shiftKey;
  pushUndo(recs, 'upright');
  const scratch = new THREE.Euler();
  for (const rec of recs) {
    if (clearAll) {
      rec.group.quaternion.identity();
    } else {
      scratch.setFromQuaternion(rec.group.quaternion, 'ZYX');
      rec.group.quaternion.setFromAxisAngle(WORLD_UP, scratch.z);
    }
    if (rec === flying) syncFlyFromRecord();
  }
  const what = clearAll ? 'rotation cleared' : 'set upright, heading kept';
  afterGroupEdit(recs, recs.length > 1
    ? `${recs.length} objects ${what}.`
    : `${recs[0].name} ${what}.`);
};

// --- drop to surface --------------------------------------------------------
// Rest the selection on the work surface. Without physics this is geometric
// contact, not a settled pose — verify in OmniGibson before relying on it.
// The ray starts above the table and stops at it, so the floor and desk legs
// are never candidates.
const TABLE_TOL_M = 0.05;      // how far under the table plane still counts as it
const DROP_REACH_M = 0.5;      // how far above it to start looking

/** The height of the work surface: the table-centre marker's z, else the
    scene's ground plane, else zero. */
function tableHeight() {
  if (tableHasPoint) return tableMarker.position.z;
  if (groundLive) return groundLive.height;
  return 0;
}

document.getElementById('btn-drop').onclick = () => {
  const recs = selectionRecords().filter((rec) => !rec.isCamera);
  if (!recs.length) return;

  // Every selection member is excluded from every ray (a member about to move
  // is not a surface), and so is the robot.
  const targets = [];
  for (const rec of objects.values()) {
    if (selection.has(rec) || rec.present === false) continue;
    if (rec.entry.kind === 'robot') continue;
    // Hidden props are not surfaces to land on; the background is exempt.
    if (rec.hiddenForView && rec.entry.kind !== 'background') continue;
    rec.group.traverse((c) => { if (c.isMesh) targets.push(c); });
  }

  const table = tableHeight();
  const lowest = table - TABLE_TOL_M;
  // Each object falls on its own, not by a shared distance.
  const drops = [];
  const noSurface = [];
  for (const rec of recs) {
    // Precise box: its underside is what lands on the table.
    const box = contactBox(rec.group);
    if (box.isEmpty()) continue;
    const centre = box.getCenter(new THREE.Vector3());
    // Start above both the object and the table, so a prop below the desk
    // still sees the desk; the reach past the table keeps stacking working.
    const from = Math.max(box.max.z, table) + DROP_REACH_M;
    const down = new THREE.Raycaster(
      new THREE.Vector3(centre.x, centre.y, from),
      new THREE.Vector3(0, 0, -1), 0, from - lowest,
    );
    // `far` stops the ray at the table; the floor is never a candidate.
    const hit = down.intersectObjects(targets, false)[0];
    if (hit) {
      drops.push([rec, hit.point.z - box.min.z, hit.object.userData.owner]);
    } else {
      // Nothing over the table at that x, y: put it at table height and say so.
      drops.push([rec, table - box.min.z, null]);
      noSurface.push(rec.name);
    }
  }
  if (!drops.length) return;

  const moved = drops.map(([rec]) => rec);
  pushUndo(moved, 'drop to surface');
  for (const [rec, dz] of drops) rec.group.position.z += dz;
  afterGroupEdit(moved);

  // The room is one mesh; a hit on it reads as "the table" only because the
  // ray stops at the table.
  const onto = (owner) => {
    const rec = owner && objects.get(owner);
    return rec && rec.entry.kind === 'background' ? 'the table' : owner;
  };
  const adrift = noSurface.length
    ? ` Nothing over the table under ${noSurface.join(', ')} — left at table `
      + 'height. Re-seat on table centre puts it on the desk.'
    : '';
  if (recs.length > 1) {
    setStatus(`Dropped ${moved.length} onto the table.${adrift}`,
      noSurface.length ? 'err' : '');
  } else if (noSurface.length) {
    setStatus(`${moved[0].name}: nothing over the table there — put it at table `
      + `height (${table.toFixed(3)} m). Re-seat on table centre puts it on the desk.`,
      'err');
  } else {
    setStatus(`Dropped ${moved[0].name} onto ${onto(drops[0][2])}.`);
  }
};

window.addEventListener('keydown', (e) => {
  // Any key struck while Ctrl is held makes this a combo: disarm the -Z nudge
  // here, above the combo branches, every one of which returns.
  if (e.ctrlKey && e.key !== 'Control') ctrlAlone = false;

  // Undo/redo work even while a numeric field has focus, matching every other
  // editor; the bare mode keys below deliberately do not.
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
    e.preventDefault();
    if (e.shiftKey) redo(); else undo();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
    e.preventDefault();
    redo();
    return;
  }

  // Select every prop. Above the bare-key branches so it cannot be read as an
  // "a" nudge, and it leaves a focused number field its own select-all.
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
    if (isTyping() || activeTab !== 'objects') return;
    e.preventDefault();
    selectAllProps();
    return;
  }

  // Esc dismisses the frontmost layer first. The unsaved-work dialog is not in
  // this chain: it handles Escape itself, in the capture phase.
  if (e.key === 'Escape' && confirmIsOpen()) return;
  // The tour and help sit above every other layer.
  if (e.key === 'Escape' && tourIsOpen()) { e.preventDefault(); closeTour(); return; }
  if (e.key === 'Escape' && helpIsOpen()) { e.preventDefault(); closeHelp(); return; }
  if (e.key === 'Escape' && placing) {
    e.preventDefault();
    cancelPlacement('Placement cancelled — nothing was imported.');
    return;
  }
  if (e.key === 'Escape' && exportIsOpen()) {
    e.preventDefault();
    closeExport();
    return;
  }
  if (e.key === 'Escape' && pickerIsOpen()) {
    e.preventDefault();
    closePicker();
    return;
  }
  if (e.key === 'Escape' && composeIsOpen()) {
    e.preventDefault();
    closeCompose();
    return;
  }
  if (e.key === 'Escape' && launcherIsOpen()) {
    e.preventDefault();
    closeLauncher();
    return;
  }
  if (e.key === 'Escape' && libraryIsOpen()) {
    e.preventDefault();
    closeLibrary();
    return;
  }

  // Asked of the DOM, not a latched flag: Chrome fires no blur for an input
  // disabled while focused.
  if (isTyping() || (e.target.tagName === 'INPUT' && !e.target.disabled)) return;
  const k = e.key.toLowerCase();

  if (e.key === '?') { e.preventDefault(); openHelp(); return; }

  if (flying) {
    flyFast = e.shiftKey;
    flySlow = e.altKey;
    if (e.key === 'Escape') { exitFly(); select(null); return; }
    // In fly mode W..D walk. Ctrl is the descend key, so it cannot disqualify
    // a press; Meta still does (OS shortcuts).
    const direction = FLY_CODES[e.code];
    if (direction && !e.metaKey) {
      // Space would otherwise scroll the page or re-trigger the last button.
      e.preventDefault();
      if (!flyKeys.has(direction)) beginFlyEdit('fly');
      flyKeys.add(direction);
    }
    return;
  }

  // Arrow keys nudge the selection along screen-aligned axes.
  if (NUDGE_KEYS[e.key] && activeTab === 'objects') {
    e.preventDefault();
    nudgeSelection(NUDGE_KEYS[e.key], e.shiftKey, e.altKey);
    return;
  }

  // + and − scale the selection, whatever the gizmo is set to. Both shifted
  // and unshifted glyphs count ("+" is Shift+"=" on most layouts); Ctrl/Meta
  // are left to the browser's own zoom.
  const scaleDir = SCALE_KEYS[e.key] ?? SCALE_CODES[e.code];
  if (typeof scaleDir === 'number' && !e.ctrlKey && !e.metaKey
      && activeTab === 'objects') {
    e.preventDefault();
    scaleSelection(scaleDir > 0, e.altKey);
    return;
  }

  // Single-letter shortcuts must not fire as part of a browser shortcut.
  const plain = !e.ctrlKey && !e.metaKey && !e.altKey;

  if (e.key === 'Control' && !e.repeat && activeTab === 'objects'
      && selected && !selected.isCamera) {
    armCtrlNudge(e);
  }

  // World-axis nudges. `plain` is deliberately not required: Alt is the fine
  // modifier, so Alt+W must nudge rather than fall through.
  const axisNudge = AXIS_NUDGE_KEYS[k === ' ' ? ' ' : k];
  if (axisNudge && !e.ctrlKey && !e.metaKey && activeTab === 'objects'
      && selected && !selected.isCamera) {
    // Space scrolls the page and re-triggers the last-clicked button otherwise.
    e.preventDefault();
    nudgeAxis(axisNudge[0], axisNudge[1], e.shiftKey, e.altKey);
    return;
  }

  // With nothing selected, the same keys walk the free view; the nudge block
  // above already returned if there was a selection. Ctrl/Meta disqualify,
  // Alt (the fine modifier) does not.
  const walk = WALK_CODES[e.code];
  if (walk && freeWalkArmed() && !e.ctrlKey && !e.metaKey) {
    // Keep the flight-control modifier flags current here too.
    flyFast = e.shiftKey;
    flySlow = e.altKey;
    // Space would otherwise scroll the page or re-fire the last button.
    e.preventDefault();
    walkKeys.add(walk);
    return;
  }

  // Gizmo modes: M and R sit outside the nudge set above. Scale has no key —
  // +/- already scale directly; the mode button covers a single non-uniform axis.
  if (plain && k === 'm') setMode('translate');
  else if (plain && k === 'r') setMode('rotate');
  else if (plain && k === 'f') { if (e.shiftKey) frameScene(); else frameSelection(); }
  // Only bound outside fly mode, where W..D are the walk instead.
  else if (plain && k === 'c') toggleCameraOverlays();
  else if (plain && k === 'v') setPipEnabled(!pipOn);
  // G toggles the guides; Shift+G the grid.
  else if (plain && k === 'g') {
    if (e.shiftKey) toggleGrid(); else setGuidesEnabled(!guidesOn);
  }
  // preventDefault: the keystroke would otherwise also be typed into the
  // launcher's filter box.
  else if (plain && k === 'o') { e.preventDefault(); openLauncher(); }
  else if (e.key === 'Escape') {
    // One press clears the whole set; say how many.
    const cleared = selection.size;
    if (select(null) && cleared > 1) setStatus(`Deselected ${cleared} objects.`);
  }
  // Delete is undoable and never touches disk, so it needs no confirmation.
  else if ((e.key === 'Delete' || e.key === 'Backspace') && activeTab === 'objects') {
    e.preventDefault();
    removeSelected();
  } else if (k === 'd' && (e.ctrlKey || e.metaKey) && activeTab === 'objects') {
    // Tab-guarded like Delete: neither means anything on the cameras tab.
    e.preventDefault();
    duplicateSelected();
  }
});

window.addEventListener('keyup', (e) => {
  flyFast = e.shiftKey;
  flySlow = e.altKey;
  flyKeys.delete(FLY_CODES[e.code]);
  walkKeys.delete(WALK_CODES[e.code]);
  if (flying && flyKeys.size === 0) endFlyEdit();

  // Ctrl released alone, having been armed on its keydown: -Z. Anything that
  // used Ctrl as a modifier cleared the flag on the way through.
  if (e.key === 'Control' && ctrlAlone && !flying) {
    ctrlAlone = false;
    if (activeTab === 'objects' && selected && !selected.isCamera) {
      nudgeAxis(2, -1, ctrlModifiers.coarse, ctrlModifiers.fine);
    }
  }
});

// Keys held when the window loses focus never send a keyup, and the camera
// would keep walking on its own when it came back.
window.addEventListener('blur', () => {
  flyKeys.clear();
  walkKeys.clear();
  flyFast = flySlow = false;
  if (flying) endFlyEdit();
});

document.getElementById('btn-reset').onclick = () => {
  const recs = selectionRecords();
  if (!recs.length) return;
  pushUndo(recs, 'revert');   // reverting is itself undoable
  for (const rec of recs) {
    // Each record reverts to its own baseline.
    const { position, orientation, scale } = rec.initial;
    rec.group.position.fromArray(position);
    rec.group.quaternion.fromArray(orientation);
    rec.group.scale.fromArray(scale);
    rec.lastValidScale.copy(rec.group.scale);
    if (rec === flying) syncFlyFromRecord();
    if (rec.isCamera) cameraDirty.delete(rec.name);
    else dirty.delete(rec.name);
  }
  document.getElementById('btn-save-cameras').disabled = cameraDirty.size === 0 || sceneMoved;
  refreshSaveButton();
  rebuildPivot();
  renderList(); refreshReadout();
  setStatus(recs.length > 1
    ? `Reverted ${recs.length} objects.`
    : `Reverted ${recs[0].name}.`);
};

// Ask the server whether removals break task bindings (a task config binds
// scene objects by name or category); the YAMLs are on disk, not in the browser.
async function taskWarnings() {
  const remove = removedRecords().filter((r) => !r.isCamera).map((r) => r.name);
  const keep = liveRecords().filter((r) => r.added).map((r) => r.name);
  try {
    const res = await fetch('/api/check_bindings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({ remove, keep }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    return body.warnings || [];
  } catch (err) {
    // Never fatal: this is advice, and the save it advises on still works.
    console.warn('task binding check unavailable', err);
    return [];
  }
}

/** One line for a list of task problems, worst first. The server sorts, so
    the first entry is the worst. */
function describeProblems(problems) {
  if (!problems || !problems.length) return '';
  const blocking = problems.filter((p) => p.severity === 'breaks');
  const lead = (blocking[0] || problems[0]).message;
  const head = blocking.length
    ? `${blocking.length} reason(s) it will not run: `
    : `${problems.length} thing(s) to check: `;
  return head + lead
    + (problems.length > 1 ? ` (+${problems.length - 1} more)` : '');
}

function describeTaskWarnings(warnings) {
  if (!warnings.length) return '';
  const breaks = warnings.filter((w) => w.severity === 'breaks');
  const lead = (breaks[0] || warnings[0]).message;
  return `${warnings.length} task binding warning(s): ${lead}`
    + (warnings.length > 1 ? ` (+${warnings.length - 1} more)` : '');
}

// --- review & export -------------------------------------------------------
// Saves the scene, the cameras it was authored against, the task that binds
// its objects, and the command that runs them, as one transaction.

let exportState = null;      // the last review or export the server returned
let exportInFlight = false;

const exportModal = () => document.getElementById('export-modal');
const exportIsOpen = () => !exportModal().hidden;

// --- physics: mass and friction, one object at a time ----------------------
// Both numbers come from the object's USD via the manifest (`entry.physics`)
// and are written back into `init_info.args`, the kwargs OmniGibson hands to
// the object's constructor.

/** The live mass and friction of a record, or null if it has no physics. */
function physicsOf(rec) {
  const facts = rec && rec.entry && rec.entry.physics;
  if (!facts) return null;
  if (!rec.physics) {
    // Materialised on first use: `rec.physics` is what the fields show,
    // `entry.physics` stays what was loaded.
    rec.physics = {
      link: facts.link || null,
      authoredMass: facts.authored_mass,
      // The scene's override wins over the asset's authored mass.
      mass: facts.mass != null ? facts.mass : facts.authored_mass,
      // Every rigid-body link the asset has; one friction value applies to all.
      links: facts.links || [],
      friction: facts.friction,
    };
    rec.physicsSaved = { mass: rec.physics.mass, friction: rec.physics.friction };
  }
  return rec.physics;
}

/**
 * Whether a record's physics can be edited at all. A function declaration so it
 * hoists: `refreshSelectionUI` can call it during module evaluation.
 */
function hasPhysics(rec) {
  return !!(rec && !rec.isCamera && rec.entry.editable && rec.entry.physics);
}

function markPhysicsDirty(rec) {
  const live = physicsOf(rec);
  const saved = rec.physicsSaved || {};
  const same = live.mass === saved.mass && live.friction === saved.friction;
  if (same) physicsDirty.delete(rec.name); else physicsDirty.add(rec.name);
  refreshSaveButton();
  renderList();
}

/** Fill the mass and friction fields for the selection (single object only). */
function refreshPhysics() {
  const block = document.getElementById('physics-block');
  const massField = document.getElementById('p-mass');
  const frictionField = document.getElementById('p-friction');
  const note = document.getElementById('physics-note');
  const rec = selection.size === 1 ? selected : null;

  if (!hasPhysics(rec)) {
    block.style.display = 'none';
    massField.value = '';
    frictionField.value = '';
    return;
  }
  block.style.display = '';
  const live = physicsOf(rec);
  const saved = rec.physicsSaved || {};

  // Never overwrite a field being typed in: a mid-keystroke reformat moves the caret.
  if (document.activeElement !== massField) {
    massField.value = live.mass != null ? live.mass : '';
  }
  if (document.activeElement !== frictionField) {
    frictionField.value = live.friction != null ? live.friction : '';
  }
  massField.disabled = false;
  massField.placeholder = live.authoredMass != null ? `${live.authoredMass}` : 'not authored';
  // Friction needs a rigid-body link from the USD to write to; with none the
  // field is disabled and the server refuses the edit too.
  frictionField.disabled = !live.links.length;
  frictionField.placeholder = live.links.length ? 'PhysX default' : 'asset not readable';
  frictionField.title = live.links.length
    ? `Sets static and dynamic friction together, on all ${live.links.length} `
      + `link(s) of this object: ${live.links.join(', ')}`
    : 'This asset has no readable rigid body, so there is nothing to apply friction to';
  // Enabled only when pressing it would change something: mass reverts to the
  // authored value, friction to what the scene last saved.
  document.getElementById('rst-mass').disabled = live.mass === live.authoredMass;
  document.getElementById('rst-friction').disabled = live.friction === saved.friction;

  const parts = [];
  if (live.authoredMass != null) {
    parts.push(live.mass != null && Math.abs(live.mass - live.authoredMass) > 1e-9
      ? `asset says ${live.authoredMass} kg`
      : `${live.authoredMass} kg as authored`);
  } else {
    parts.push('no mass authored in the asset');
  }
  // Friction is a real OmniGibson constructor argument, applied at load; mass
  // is only recorded on the scene and needs a consumer to apply it.
  parts.push(live.link
    ? 'friction is applied on load; mass is recorded for the scene but needs a consumer to apply it'
    : 'friction unavailable: no rigid-body link in this asset');
  note.textContent = parts.join(' · ');
}

function readPhysicsField(field, low, high) {
  const raw = field.value.trim();
  if (raw === '') return { blank: true };
  const value = Number(raw);
  if (!Number.isFinite(value) || value < low || value > high) return { bad: true };
  return { value };
}

function wirePhysicsField(id, low, high, apply) {
  const field = document.getElementById(id);
  // Warn while typing, commit when the field is left: committing per keystroke
  // would apply half-typed numbers.
  field.addEventListener('input', () => {
    field.classList.toggle('bad', !!readPhysicsField(field, low, high).bad);
  });
  field.addEventListener('change', () => {
    const rec = selection.size === 1 ? selected : null;
    if (!hasPhysics(rec)) return;
    const read = readPhysicsField(field, low, high);
    field.classList.toggle('bad', !!read.bad);
    if (read.bad) {
      setStatus(`${field.id === 'p-mass' ? 'Mass' : 'Friction'} has to be `
                + `between ${low} and ${high}.`, 'err');
      return;
    }
    apply(physicsOf(rec), read.blank ? null : read.value);
    markPhysicsDirty(rec);
    refreshPhysics();
  });
}

wirePhysicsField('p-mass', 0.01, 50, (live, value) => {
  // Blanking the box means "whatever the asset says", not "zero".
  live.mass = value == null ? live.authoredMass : value;
});
wirePhysicsField('p-friction', 0, 2, (live, value) => { live.friction = value; });

document.getElementById('rst-mass').onclick = () => {
  const rec = selection.size === 1 ? selected : null;
  if (!hasPhysics(rec)) return;
  const live = physicsOf(rec);
  live.mass = live.authoredMass;
  markPhysicsDirty(rec);
  refreshPhysics();
  setStatus(`${rec.name}: mass back to the asset's ${live.authoredMass ?? 'unset'} kg.`);
};
document.getElementById('rst-friction').onclick = () => {
  const rec = selection.size === 1 ? selected : null;
  if (!hasPhysics(rec)) return;
  physicsOf(rec).friction = (rec.physicsSaved || {}).friction ?? null;
  markPhysicsDirty(rec);
  refreshPhysics();
  setStatus(`${rec.name}: friction back to what the scene has saved.`);
};

// --- joints: an articulated prop's degrees of freedom ----------------------
// Offered for any object whose *asset* has joints, even when the saved values
// are all zero. The two halves differ:
//   value  -> state.registry.object_registry[name].joint_pos; OmniGibson
//             restores it on load.
//   limits -> init_info.args.joint_limits; recorded on the scene only, applied
//             by a consumer after load (the shared USD is never edited).

/** The live joint state of a record, or null if its asset has no joints. */
function jointsOf(rec) {
  const facts = rec && rec.entry && rec.entry.joints;
  if (!facts) return null;
  if (!rec.joints) {
    // Materialised on first use: `rec.joints` is what the fields show,
    // `entry.joints` stays what was loaded.
    rec.joints = {
      list: facts.joints || [],
      addressable: !!facts.addressable,
      values: (facts.values || []).slice(),
      limits: structuredClone(facts.limits || {}),
      scale: facts.scale || null,
    };
    rec.jointsSaved = {
      values: rec.joints.values.slice(),
      limits: structuredClone(rec.joints.limits),
    };
  }
  return rec.joints;
}

/**
 * Whether a record's joints can be edited at all. A function declaration so it
 * hoists, like `hasPhysics`.
 */
function hasJoints(rec) {
  return !!(rec && !rec.isCamera && rec.entry.editable && rec.entry.joints);
}

/** The authored range of one joint, or null when it is a continuous hinge. */
function authoredLimit(joint) {
  const { lower, upper } = joint;
  // lower >= upper means "unlimited" (PhysX reads it that way).
  if (lower == null || upper == null || lower >= upper) return null;
  return { lower, upper };
}

/** What the simulator would use for a joint: the scene's override, or the asset's. */
function effectiveLimit(live, joint) {
  return live.limits[joint.name] || authoredLimit(joint);
}

/**
 * The authored range as OmniGibson will really enforce it: a prismatic range is
 * multiplied by the object's scale at load. Null when nothing changes it, and
 * null for a non-uniform scale, where the factor depends on the joint's axis.
 */
function scaledLimit(live, joint) {
  const factor = live.scale && live.scale.factor;
  const authored = authoredLimit(joint);
  if (!factor || joint.type !== 'prismatic' || !authored) return null;
  return { lower: authored.lower * factor, upper: authored.upper * factor };
}

/**
 * The range a value is flagged outside of: the typed override if any, else the
 * union of the authored range and the scaled one.
 */
function flagRange(live, joint) {
  if (live.limits[joint.name]) return live.limits[joint.name];
  const authored = authoredLimit(joint);
  const scaled = scaledLimit(live, joint);
  if (!authored) return null;
  if (!scaled) return authored;
  return { lower: Math.min(authored.lower, scaled.lower),
           upper: Math.max(authored.upper, scaled.upper) };
}

function markJointsDirty(rec) {
  const live = jointsOf(rec);
  const saved = rec.jointsSaved || { values: [], limits: {} };
  const same = JSON.stringify(live.values) === JSON.stringify(saved.values)
    && JSON.stringify(live.limits) === JSON.stringify(saved.limits);
  if (same) jointsDirty.delete(rec.name); else jointsDirty.add(rec.name);
  refreshSaveButton();
  renderList();
}

/**
 * Joint names with their shared prefix stripped; the full name stays on the
 * row's `title`.
 */
function jointShortNames(list) {
  const names = list.map((joint) => joint.name);
  if (names.length < 2) return names;
  let prefix = 0;
  while (prefix < names[0].length
    && names.every((name) => name[prefix] === names[0][prefix])) prefix++;
  // Cut one `_`-separated segment before the divergence, so
  // `..._to_drawer_1_link` becomes `drawer_1_link` rather than `1_link`.
  const cut = names[0].lastIndexOf('_', prefix) + 1;
  const keep = cut > 1 ? names[0].lastIndexOf('_', cut - 2) + 1 : cut;
  return keep > 0 && names.every((name) => name.length > keep)
    ? names.map((name) => name.slice(keep))
    : names;
}

// Which record the rows on screen were built for. Rebuilding them on every
// selection refresh would destroy the field being typed into.
let jointsRenderedFor = null;

function jointField(value, step) {
  const field = document.createElement('input');
  field.type = 'number';
  field.step = String(step);
  field.value = value == null ? '' : Number(value.toFixed(4));
  return field;
}

/** Tooltip for each of a row's three field labels. */
const JOINT_FIELD_HELP = {
  value: 'The joint coordinate: metres for a sliding joint, radians for a hinge. '
    + 'Saved into the scene state and restored on load',
  lower: 'Lower travel limit. Recorded on the scene as an override — the asset\'s '
    + 'own limit is what the simulator reads today',
  upper: 'Upper travel limit. Recorded on the scene as an override — the asset\'s '
    + 'own limit is what the simulator reads today',
};

/** Rows currently built, in joint order, with the widgets each one owns. */
const jointRows = [];

/** The live preview: renderer, scene and the per-joint subtrees, or null. */
let jointView = null;

/** A running "Preview range" animation, or null. Picture only; edits nothing. */
let jointSweep = null;

/** What `closeJointEditor` has to undo, or null when the window is shut. */
let jointEditorTeardown = null;

/** Update the panel's button and rebuild the editor's rows for the selection. */
function refreshJoints() {
  const block = document.getElementById('joints-block');
  const grid = document.getElementById('joints-grid');
  const note = document.getElementById('joints-note');
  const rec = selection.size === 1 ? selected : null;

  if (!hasJoints(rec)) {
    block.style.display = 'none';
    // No single selection: close the editor and free its GL context.
    closeJointEditor();
    jointsRenderedFor = null;
    return;
  }
  block.style.display = '';
  const live = jointsOf(rec);
  const short = jointShortNames(live.list);

  if (jointsRenderedFor !== rec.name) {
    // A *different* object closes the open editor; null just means "rebuild
    // these rows" and must not close it.
    if (jointsRenderedFor !== null) closeJointEditor();
    // Rows are built even while the editor is hidden, so opening it never
    // rebuilds fields that may hold a half-typed number.
    grid.replaceChildren();
    jointRows.length = 0;
    live.list.forEach((joint, index) => {
      const row = document.createElement('div');
      row.className = 'xrow';
      row.dataset.joint = joint.name;

      const top = document.createElement('div');
      top.className = 'jtop';
      const label = document.createElement('span');
      label.className = 'jname';
      label.textContent = short[index];
      label.title = joint.name;
      const kind = document.createElement('span');
      kind.className = 'jkind';
      // The unit the row's fields are in: metres for a slide, radians for a hinge.
      kind.textContent = joint.type === 'prismatic'
        ? `slide · ${joint.axis} · metres`
        : `hinge · ${joint.axis} · radians`;
      top.append(label, kind);
      row.appendChild(top);

      // 5 mm steps for a slide, 0.05 rad (~3 degrees) for a hinge.
      const step = joint.type === 'prismatic' ? 0.005 : 0.05;
      const fields = document.createElement('div');
      fields.className = 'jfields';
      for (const text of ['value', 'lower', 'upper']) {
        const cell = document.createElement('span');
        cell.className = 'jlabel';
        cell.textContent = text;
        cell.title = JOINT_FIELD_HELP[text];
        fields.appendChild(cell);
      }

      const tools = document.createElement('span');
      tools.className = 'tools';
      const reset = document.createElement('button');
      reset.textContent = '↺';
      reset.title = `Back to the value this scene saved and the limits ${joint.name} is authored with`;
      reset.disabled = !live.addressable;
      reset.onclick = () => {
        const saved = rec.jointsSaved || { values: [], limits: {} };
        if (saved.values[index] != null) live.values[index] = saved.values[index];
        delete live.limits[joint.name];
        markJointsDirty(rec);
        jointsRenderedFor = null;              // values changed under the fields
        refreshJoints();
        setStatus(`${rec.name}: ${short[index]} back to the saved value and the asset's limits.`);
      };
      tools.appendChild(reset);
      fields.appendChild(tools);

      const value = jointField(live.values[index], step);
      value.setAttribute('aria-label', `${joint.name} value`);
      // Warn while typing, commit when the field is left, like the physics fields.
      value.oninput = () => {
        const typed = parseFloat(value.value);
        flagPastLimit(value, Number.isFinite(typed) ? typed : live.values[index],
                      flagRange(live, joint));
      };
      value.onchange = () => {
        const typed = parseFloat(value.value);
        if (!Number.isFinite(typed)) { jointsRenderedFor = null; refreshJoints(); return; }
        live.values[index] = typed;
        markJointsDirty(rec);
        flagPastLimit(value, typed, flagRange(live, joint));
        syncJointRow(index);
        showJointPose(index);
      };
      fields.appendChild(value);

      const authored = authoredLimit(joint);
      const current = effectiveLimit(live, joint);
      for (const side of ['lower', 'upper']) {
        const field = jointField(current ? current[side] : null, step);
        field.setAttribute('aria-label', `${joint.name} ${side} limit`);
        field.placeholder = authored ? String(Number(authored[side].toFixed(4))) : 'unlimited';
        // Committed on `change`: an override is written as a pair, so a
        // half-typed side must not define a range.
        field.onchange = () => {
          const bounds = effectiveLimit(live, joint) || { lower: 0, upper: 0 };
          const typed = parseFloat(field.value);
          if (!Number.isFinite(typed)) { jointsRenderedFor = null; refreshJoints(); return; }
          live.limits[joint.name] = { ...bounds, [side]: typed };
          markJointsDirty(rec);
          refreshJointValueFlags(rec);
          syncJointRow(index);
          // The ghosts stand at the limits, so re-pose them.
          showJointGhosts(index);
        };
        fields.appendChild(field);
      }
      row.appendChild(fields);

      // The slider: limits at the ends of the track, handle at the value. It is
      // the row's fourth <input>; the first three are value, lower, upper.
      const slide = document.createElement('div');
      slide.className = 'jslide';
      const low = document.createElement('span');
      low.className = 'jend';
      const high = document.createElement('span');
      high.className = 'jend';
      const range = document.createElement('input');
      range.type = 'range';
      range.step = String(joint.type === 'prismatic' ? 0.001 : 0.01);
      range.setAttribute('aria-label', `${joint.name} value slider`);
      // Dragging applies every value so the geometry follows the handle. The
      // dirty mark re-renders the object list, so it is only taken on leaving
      // clean and again on release.
      range.oninput = () => {
        const typed = parseFloat(range.value);
        if (!Number.isFinite(typed)) return;
        live.values[index] = typed;
        if (!jointsDirty.has(rec.name)) markJointsDirty(rec);
        syncJointRow(index);
        showJointPose(index);
      };
      range.onchange = () => markJointsDirty(rec);
      slide.append(low, range, high);
      row.appendChild(slide);

      const foot = document.createElement('div');
      foot.className = 'jfoot';
      const travel = document.createElement('span');
      travel.className = 'jtravel';
      const sweep = document.createElement('button');
      sweep.textContent = 'Preview range';
      sweep.title = 'Run this joint from its lower limit to its upper and back, '
        + 'in the picture only. Nothing is edited and nothing is saved.';
      sweep.onclick = () => startJointSweep(index);
      foot.append(travel, sweep);
      row.appendChild(foot);

      // Why this joint cannot be moved on screen; hidden when it can.
      const why = document.createElement('div');
      why.className = 'hint warn jwhy';
      why.hidden = true;
      row.appendChild(why);

      // Not addressable: the saved array and the asset disagree on the joint
      // count, so fields go blank and disabled; the note shows the raw array.
      if (!live.addressable) {
        row.querySelectorAll('input').forEach((field) => {
          field.value = '';
          field.disabled = true;
        });
        sweep.disabled = true;
      }

      grid.appendChild(row);
      jointRows.push({ row, joint, index, range, low, high, travel, sweep, why });
    });
    jointsRenderedFor = rec.name;
    jointRows.forEach((_, index) => syncJointRow(index));
  } else {
    // Same object: refresh only the numbers, and never under a caret.
    jointRows.forEach((entry, index) => {
      const fields = entry.row.querySelectorAll('input');
      const limit = effectiveLimit(live, entry.joint);
      if (!live.addressable) return;
      // `syncJointRow` owns the value field (the slider moves it too); the two
      // limit fields are updated here.
      ['lower', 'upper'].forEach((side, offset) => {
        const field = fields[offset + 1];
        if (document.activeElement !== field) {
          field.value = limit ? Number(limit[side].toFixed(4)) : '';
        }
      });
      syncJointRow(index);
    });
  }
  refreshJointValueFlags(rec);
  // Re-pose the preview to match the numbers.
  if (jointView) {
    jointRows.forEach((_, index) => { showJointPose(index); showJointGhosts(index); });
  }

  note.textContent = live.addressable
    ? `${live.list.length} joint(s) — open the editor to move them and see the range.`
    : `${live.list.length} joint(s) — not addressable: this scene stores `
      + `${live.values.length} value(s), so none of them can be edited.`;
  document.getElementById('joint-note').textContent = jointNoteText(live);
}

/** The long explanation, which lives beside the fields it is about. */
function jointNoteText(live) {
  if (!live.addressable) {
    return `This scene stores ${live.values.length} joint value(s) and the asset has `
      + `${live.list.length} — which value belongs to which joint cannot be told apart, `
      + `so none of them can be edited here. The scene holds: [${
        live.values.map((v) => Number(v.toFixed(4))).join(', ')}].`;
  }
  const parts = [];
  parts.push('Values are saved into the scene state and restored on load. '
    + 'The preview above is this object alone, moved by the numbers below — '
    + 'the scene behind does not move, because its proxy is baked at one '
    + 'configuration.');
  // A sliding joint's range is multiplied by the object's scale at load, so
  // the authored figure is not the travel the simulator allows.
  if (live.scale && live.list.some((joint) => joint.type === 'prismatic')) {
    parts.push(live.scale.factor
      ? `This object is scaled ${Number(live.scale.factor.toFixed(3))}×, and OmniGibson `
        + "scales a sliding joint's range with it — every slide's real travel is "
        + 'on its own row.'
      : 'This object is scaled unevenly, and OmniGibson scales a sliding '
        + "joint's range along that joint's own axis — so its real travel is "
        + 'not the figure shown.');
  }
  parts.push('Limits are recorded on the scene as an override; the simulator '
    + "reads the asset's own limits until a consumer applies them, like mass.");
  return parts.join(' ');
}

/**
 * The travel one joint is authored for, and the travel it will really get:
 * OmniGibson scales a *sliding* joint's range with the object at load; a hinge
 * is not scaled.
 */
function jointTravelText(live, joint) {
  const dp = (v, places = 4) => Number(v.toFixed(places));
  const authored = authoredLimit(joint);
  if (!authored) {
    return joint.type === 'revolute'
      ? 'no range authored — this hinge turns freely'
      : 'no range authored — this slide is unbounded';
  }
  const span = authored.upper - authored.lower;
  const parts = [`${dp(authored.lower)} → ${dp(authored.upper)} authored`];
  if (joint.type === 'prismatic') {
    const factor = live.scale && live.scale.factor;
    if (factor) parts.push(`×${dp(factor, 3)}`, `${dp(span * factor, 3)} m actual`);
    // Non-uniform scale: the factor depends on the joint's axis, so it is
    // reported as unknown rather than invented.
    else if (live.scale) parts.push('scaled, factor not computed');
    else parts.push(`${dp(span, 3)} m travel`);
  } else {
    parts.push(`${dp(span, 3)} rad travel`, `${dp(span * 180 / Math.PI, 1)}°`);
  }
  if (live.limits[joint.name]) {
    const over = live.limits[joint.name];
    parts.push(`this scene overrides it to ${dp(over.lower)} → ${dp(over.upper)}`);
  }
  return parts.join(' · ');
}

/** Put one row's slider, its track annotations and its travel readout in step. */
function syncJointRow(index) {
  const entry = jointRows[index];
  const rec = selection.size === 1 ? selected : null;
  if (!entry || !hasJoints(rec)) return;
  const live = jointsOf(rec);
  const value = live.values[index];
  const limit = effectiveLimit(live, entry.joint);
  const dp = (v, places = 4) => Number(v.toFixed(places));

  // The track spans the joint's range, widened to include a value outside it.
  const fallback = entry.joint.type === 'revolute' ? Math.PI : 0.5;
  let lower = limit ? limit.lower : -fallback;
  let upper = limit ? limit.upper : fallback;
  if (Number.isFinite(value)) { lower = Math.min(lower, value); upper = Math.max(upper, value); }
  entry.range.min = String(lower);
  entry.range.max = String(upper);
  if (live.addressable && Number.isFinite(value)) entry.range.value = String(value);
  // The number field follows the handle, but never under a caret.
  const field = entry.row.querySelector('input');
  if (live.addressable && Number.isFinite(value) && document.activeElement !== field) {
    field.value = dp(value);
    flagPastLimit(field, value, flagRange(live, entry.joint));
  }
  entry.low.textContent = limit ? String(dp(limit.lower)) : '−∞';
  entry.high.textContent = limit ? String(dp(limit.upper)) : '+∞';
  entry.low.title = entry.high.title = limit
    ? 'The range this joint travels in, as this scene has it'
    : 'The asset authors no range for this joint';
  entry.travel.textContent = jointTravelText(live, entry.joint);

  // A joint the picture cannot move disables its slider and sweep; its three
  // fields stay live and saved.
  const blocked = jointBlockedReason(entry.joint, index);
  entry.why.textContent = blocked || '';
  entry.why.hidden = !blocked;
  entry.range.disabled = !live.addressable || !!blocked;
  entry.sweep.disabled = !live.addressable || !limit || !!blocked;
}

/** Mark a value field that sits outside the range it is supposed to travel in. */
function flagPastLimit(field, value, limit) {
  const past = !!limit && (value < limit.lower - 1e-9 || value > limit.upper + 1e-9);
  field.classList.toggle('past-limit', past);
  field.title = past
    ? `Outside this joint's range (${Number(limit.lower.toFixed(4))} to `
      + `${Number(limit.upper.toFixed(4))}). Saved as typed — nothing here clamps it.`
    : '';
}

function refreshJointValueFlags(rec) {
  const live = jointsOf(rec);
  if (!live) return;
  [...document.querySelectorAll('#joints-grid .xrow')].forEach((row, index) => {
    const field = row.querySelector('input');
    if (field && live.addressable && live.values[index] != null) {
      flagPastLimit(field, live.values[index], flagRange(live, live.list[index]));
    }
  });
}

// --- the joint editor ------------------------------------------------------
// A floating window with a live 3D preview of the selected object: drag a
// slider and the drawer slides. Not a new write path — every control edits
// `live.values` / `live.limits` and the ordinary scene Save writes them.
// The manifest supplies, per joint, the node it moves (`child_node`), its axis
// line (`pivot`, `direction`) and the baked configuration (`rest`), all in
// stage-root space — the frame the proxy geometry is in.

const jointEditorIsOpen = () => !document.getElementById('joint-modal').hidden;

/** Why a joint cannot be moved on screen, or null when it can. */
function jointBlockedReason(joint, index) {
  if (!joint.child_node || !joint.pivot || !joint.direction) {
    return 'The extractor could not work out which link this joint moves, so the '
      + 'picture cannot show it. Its numbers are still edited and saved.';
  }
  const entry = jointView && jointView.entries[index];
  if (entry && !entry.group) {
    return `Nothing in this object's proxy is under ${joint.child}, so there is `
      + 'no geometry to move. Its numbers are still edited and saved.';
  }
  return null;
}

/**
 * Which of the clone's nodes each joint moves. Matched on the node name's
 * prefix (three.js sanitises glTF node names, so the manifest carries an
 * encoded name as `child_node`); longest match wins, so `.../link_1` cannot
 * claim `.../link_10`'s meshes. Only the topmost match is taken — moving a
 * parent and its child would move the child twice. Nested joint trees are not
 * supported.
 */
function assignJointSubtrees(root, entries) {
  const owners = new Map();
  root.traverse((node) => {
    if (!node.name) return;
    let best = null;
    for (const entry of entries) {
      if (!entry.childNode || !node.name.startsWith(entry.childNode)) continue;
      if (!best || entry.childNode.length > best.childNode.length) best = entry;
    }
    if (best) owners.set(node, best);
  });
  for (const [node, entry] of owners) {
    let nested = false;
    for (let up = node.parent; up; up = up.parent) if (owners.has(up)) { nested = true; break; }
    if (!nested) entry.nodes.push(node);
  }
}

/**
 * Put one joint's subtree at *value*. The group's transform is the motion away
 * from the baked pose: a slide along the axis, or a rotation about the line
 * (pivot, direction) with the translation that pins the pivot in place.
 */
function poseJointGroup(entry, group, value) {
  const delta = value - entry.rest;
  if (entry.type === 'prismatic') {
    group.quaternion.identity();
    group.position.copy(entry.direction).multiplyScalar(delta);
  } else {
    group.quaternion.setFromAxisAngle(entry.direction, delta);
    group.position.copy(entry.pivot)
      .sub(entry.pivot.clone().applyQuaternion(group.quaternion));
  }
}

/** Move one joint's geometry to a value, or to whatever the fields now say. */
function showJointPose(index, override) {
  const entry = jointView && jointView.entries[index];
  if (!entry || !entry.group) return;
  const rec = selection.size === 1 ? selected : null;
  const live = rec && jointsOf(rec);
  const value = override !== undefined ? override : (live && live.values[index]);
  if (!Number.isFinite(value)) return;
  poseJointGroup(entry, entry.group, value);
}

/**
 * Stand one joint's ghosts — two translucent copies of the moving link — at its
 * lower and upper limits. Re-posed, not rebuilt, when a limit moves.
 */
function showJointGhosts(index) {
  const entry = jointView && jointView.entries[index];
  if (!entry || !entry.ghosts) return;
  const rec = selection.size === 1 ? selected : null;
  const live = rec && jointsOf(rec);
  const limit = live && effectiveLimit(live, entry.joint);
  for (const side of ['lower', 'upper']) {
    const ghost = entry.ghosts[side];
    ghost.visible = !!limit;
    if (limit) poseJointGroup(entry, ghost, limit[side]);
  }
}

/**
 * Run one joint from its lower limit to its upper and back, once. Picture only:
 * `live.values` is never touched and nothing is marked dirty.
 */
function startJointSweep(index) {
  const entry = jointView && jointView.entries[index];
  const rec = selection.size === 1 ? selected : null;
  if (!entry || !entry.group || !hasJoints(rec)) return;
  const live = jointsOf(rec);
  const limit = effectiveLimit(live, entry.joint);
  const value = live.values[index];
  if (!limit || !Number.isFinite(value)) return;
  jointSweep = {
    index,
    started: performance.now(),
    // [target, seconds] per leg; the near end first.
    legs: [[limit.lower, 0.45], [limit.upper, 0.9], [value, 0.55]],
    from: value,
  };
  setStatus(`${rec.name}: previewing ${jointRows[index].row.dataset.joint} — nothing is being edited.`);
}

/** Advance a running sweep, or finish it. Called from the preview's own loop. */
function stepJointSweep(now) {
  const sweep = jointSweep;
  let elapsed = (now - sweep.started) / 1000;
  let from = sweep.from;
  for (const [to, seconds] of sweep.legs) {
    if (elapsed < seconds) {
      // Smoothstep easing.
      const t = elapsed / seconds;
      showJointPose(sweep.index, from + (to - from) * t * t * (3 - 2 * t));
      return;
    }
    elapsed -= seconds;
    from = to;
  }
  jointSweep = null;
  showJointPose(sweep.index);
}

/**
 * Build the isolated preview: this object alone, with its own renderer, scene,
 * camera and orbit control, all destroyed in `disposeJointPreview` (a leaked
 * GL context per open would exhaust the browser).
 */
function buildJointPreview(rec) {
  const container = document.getElementById('joint-preview');
  const live = jointsOf(rec);
  // Clone the inner group: it holds the proxy in stage-root space, the frame
  // `pivot` and `direction` are expressed in. The outer group carries the
  // authored pose and scale.
  const source = rec.group.children[0];
  // A proxy that failed to load has nothing to show; the joint numbers are
  // still editable.
  if (!source || new THREE.Box3().setFromObject(source).isEmpty()) {
    const said = document.createElement('div');
    said.className = 'hint';
    said.textContent = `${rec.name} has no geometry to show — `
      + `${rec.loadError || 'its proxy is empty'}. The joints below are still editable.`;
    container.replaceChildren(said);
    return;
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.replaceChildren(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x171b22);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 2.4));
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(1, -1.4, 2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5);
  fill.position.set(-1.4, 1, 0.6);
  scene.add(fill);

  // Deep clone: shares geometries and materials with the live scene object,
  // which is why the teardown must not dispose them.
  const clone = source.clone(true);
  clone.position.set(0, 0, 0);
  clone.quaternion.identity();
  clone.scale.set(1, 1, 1);
  scene.add(clone);

  // Solid and faint: a wireframe of these dense meshes reads as a second solid.
  const ghostMaterial = new THREE.MeshBasicMaterial({
    color: 0x76b900, transparent: true, opacity: 0.18, depthWrite: false,
  });
  const entries = live.list.map((joint) => ({
    joint,
    type: joint.type,
    childNode: joint.child_node || null,
    rest: Number.isFinite(joint.rest) ? joint.rest : 0,
    // `setFromAxisAngle` needs a unit axis.
    direction: joint.direction ? new THREE.Vector3().fromArray(joint.direction).normalize() : null,
    pivot: joint.pivot ? new THREE.Vector3().fromArray(joint.pivot) : null,
    nodes: [],
    group: null,
    ghosts: null,
  }));
  assignJointSubtrees(clone, entries);

  clone.updateMatrixWorld(true);
  for (const entry of entries) {
    if (!entry.nodes.length || !entry.direction || !entry.pivot) continue;
    const group = new THREE.Group();
    clone.add(group);
    // `attach` rather than `add`: it keeps each node's world transform, so
    // grouping the link changes nothing until a joint value does.
    for (const node of entry.nodes) group.attach(node);
    entry.group = group;
    entry.ghosts = { lower: new THREE.Group(), upper: new THREE.Group() };
    for (const side of ['lower', 'upper']) {
      for (const node of entry.nodes) {
        const ghost = node.clone(true);
        ghost.traverse((child) => { if (child.isMesh) child.material = ghostMaterial; });
        entry.ghosts[side].add(ghost);
      }
      // Outside the moving group, or the ghosts would travel with the link.
      clone.add(entry.ghosts[side]);
    }
  }

  jointView = { renderer, scene, clone, entries, ghostMaterial, frame: 0, width: 0, height: 0 };
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  camera.up.copy(WORLD_UP);
  jointView.camera = camera;
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  jointView.controls = controls;

  entries.forEach((_, index) => { showJointPose(index); showJointGhosts(index); });
  // Framed with the ghosts included, so the whole sweep is in the picture.
  const box = new THREE.Box3().setFromObject(clone);
  const centre = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
  const radius = box.isEmpty()
    ? 0.5 : Math.max(box.getSize(new THREE.Vector3()).length() * 0.5, 1e-3);
  // A three-quarter view from above, at the distance the frustum needs.
  const aspect = Math.max((container.clientWidth || 1) / (container.clientHeight || 1), 1e-3);
  const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
  const halfH = Math.atan(Math.tan(halfV) * aspect);
  // Fitted to the box's eight corners, not a bounding sphere: these props are
  // wide and flat, and a sphere fit leaves them small in the frame.
  const view = new THREE.Vector3(1.5, -1.75, 1.05).normalize();
  const side = new THREE.Vector3().crossVectors(WORLD_UP, view).normalize();
  const up = new THREE.Vector3().crossVectors(view, side).normalize();
  const tanV = Math.tan(halfV);
  const tanH = Math.tan(halfH);
  let distance = radius;
  if (!box.isEmpty()) {
    for (const x of [box.min.x, box.max.x]) {
      for (const y of [box.min.y, box.max.y]) {
        for (const z of [box.min.z, box.max.z]) {
          // A corner is inside the frustum when
          // `d >= |corner.up| / tanV + corner.view`, and likewise sideways.
          const v = new THREE.Vector3(x, y, z).sub(centre);
          const along = v.dot(view);
          distance = Math.max(distance,
                              Math.abs(v.dot(up)) / tanV + along,
                              Math.abs(v.dot(side)) / tanH + along);
        }
      }
    }
  }
  distance *= JOINT_FRAME_PAD;
  camera.position.copy(centre).add(view.clone().multiplyScalar(distance));
  camera.near = Math.max(radius * 0.02, 1e-4);
  camera.far = (distance + radius) * 8;
  camera.updateProjectionMatrix();
  controls.target.copy(centre);
  controls.update();

  const draw = () => {
    if (!jointView) return;
    jointView.frame = requestAnimationFrame(draw);
    // Poll the size each frame: the window is draggable and the page resizable.
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 190;
    if (width !== jointView.width || height !== jointView.height) {
      jointView.width = width;
      jointView.height = height;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
    if (jointSweep) stepJointSweep(performance.now());
    controls.update();
    renderer.render(scene, camera);
  };
  draw();
  // Blocked state is only knowable once the subtrees are grouped.
  jointRows.forEach((_, index) => syncJointRow(index));
}

/** Give back everything `buildJointPreview` took, including the GL context. */
function disposeJointPreview() {
  if (!jointView) return;
  const view = jointView;
  jointView = null;               // before anything else: `draw` reads it
  jointSweep = null;
  cancelAnimationFrame(view.frame);
  view.controls.dispose();
  // Only the material this window made: the clone's geometries and materials
  // belong to the live scene object.
  view.ghostMaterial.dispose();
  view.renderer.dispose();
  // dispose() leaves the GL context alive (browsers allow ~16), so force-lose
  // it; the canvas cannot supply another context afterwards.
  view.renderer.forceContextLoss();
  document.getElementById('joint-preview').replaceChildren();
}

// The joint window's last size, persisted per browser.
const JOINT_SIZE_KEY = 'simfoundry.light-editor.joint-window';
const JOINT_MIN_W = 660;    // must agree with #joint-card's min-width in index.html
const JOINT_MIN_H = 340;    // and its min-height
// Breathing room around the framed object; 1.0 puts the corners on the edge of
// the picture.
const JOINT_FRAME_PAD = 1.12;

function readJointSize() {
  try {
    const raw = JSON.parse(localStorage.getItem(JOINT_SIZE_KEY) || 'null');
    if (!raw || !Number.isFinite(raw.w) || !Number.isFinite(raw.h)) return null;
    // Clamped against *this* viewport, not the one it was saved on.
    return {
      w: Math.max(JOINT_MIN_W, Math.min(raw.w, Math.round(window.innerWidth * 0.96))),
      h: Math.max(JOINT_MIN_H, Math.min(raw.h, Math.round(window.innerHeight * 0.92))),
    };
  } catch {
    return null;             // private browsing, or a value from an older format
  }
}

function saveJointSize(rect) {
  if (!rect || rect.width < JOINT_MIN_W || rect.height < JOINT_MIN_H) return;
  try {
    localStorage.setItem(JOINT_SIZE_KEY,
                         JSON.stringify({ w: Math.round(rect.width), h: Math.round(rect.height) }));
  } catch { /* the window still resizes; it just forgets. Not worth a message. */ }
}

/**
 * Open the joint editor for the selection. The rows are already built, so
 * opening is a display change plus the GL context.
 */
function openJointEditor() {
  const rec = selection.size === 1 ? selected : null;
  if (!hasJoints(rec) || jointEditorIsOpen()) return;
  const modal = document.getElementById('joint-modal');
  const card = document.getElementById('joint-card');
  const backdrop = document.getElementById('joint-backdrop');
  const closeButton = document.getElementById('joint-close');
  const title = document.getElementById('joint-title');
  // The name can be elided at this width, so it is also the tooltip.
  title.textContent = title.title = `${rec.name} — joints`;
  // Recentre on every open: a dragged position can be off-screen after a resize.
  card.style.left = '';
  card.style.top = '';
  card.style.transform = '';
  // The size, unlike the position, is kept; CSS caps it at 96vw/92vh, so a
  // size saved on a larger display cannot open off-screen here.
  const saved = readJointSize();
  card.style.width = saved ? `${saved.w}px` : '';
  card.style.height = saved ? `${saved.h}px` : '';
  modal.hidden = false;

  const onKey = (event) => {
    // The unsaved-work dialog is modal over everything and handles Escape itself.
    if (confirmIsOpen()) return;
    if (event.key === 'Escape') {
      // stopPropagation too: Escape must not also deselect the object.
      event.preventDefault();
      event.stopPropagation();
      closeJointEditor();
      return;
    }
    // Keys struck inside this window must not reach the scene's shortcuts.
    if (card.contains(event.target)) event.stopPropagation();
  };
  window.addEventListener('keydown', onKey, true);
  backdrop.onclick = () => closeJointEditor();
  closeButton.onclick = () => closeJointEditor();
  card.querySelector('#joint-head').onpointerdown = dragJointCard;

  // One teardown, reached by every exit: button, backdrop, Escape, and a
  // selection change.
  jointEditorTeardown = () => {
    window.removeEventListener('keydown', onKey, true);
    backdrop.onclick = null;
    closeButton.onclick = null;
    card.querySelector('#joint-head').onpointerdown = null;
    // Measured before hiding: a hidden element has no box to save.
    saveJointSize(card.getBoundingClientRect());
    disposeJointPreview();
    modal.hidden = true;
    const button = document.getElementById('btn-joint-editor');
    if (button.offsetParent) button.focus();
  };
  buildJointPreview(rec);
  closeButton.focus();
}

/** Shut the joint editor, however it was asked for. Safe to call when shut. */
function closeJointEditor() {
  if (!jointEditorTeardown) return;
  const finish = jointEditorTeardown;
  jointEditorTeardown = null;    // before running it: every path lands here once
  finish();
}

/** Drag the window by its header. */
function dragJointCard(event) {
  // A drag started on the header's Close button would eat the click.
  if (event.target.closest('button')) return;
  const card = document.getElementById('joint-card');
  const rect = card.getBoundingClientRect();
  // Swap the centring transform for the actual position; the drag is then
  // plain arithmetic.
  card.style.transform = 'none';
  card.style.left = `${rect.left}px`;
  card.style.top = `${rect.top}px`;
  const grabX = event.clientX - rect.left;
  const grabY = event.clientY - rect.top;
  const clamp = (value, span, extent) => Math.max(0, Math.min(extent - span, value));
  const move = (e) => {
    card.style.left = `${clamp(e.clientX - grabX, rect.width, window.innerWidth)}px`;
    card.style.top = `${clamp(e.clientY - grabY, rect.height, window.innerHeight)}px`;
  };
  const drop = () => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', drop);
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', drop);
  event.preventDefault();
}

document.getElementById('btn-joint-editor').onclick = openJointEditor;

// --- what a save sends --------------------------------------------------
//
// Every save or export sends a *complete* snapshot, never a diff: the server
// compiles each write from the immutable startup document plus this snapshot,
// so an omitted field reverts to the startup value. `physicsEdit`/`jointsEdit`
// read the live panel state; `physicsSaved`/`jointsSaved` only drive the dirty
// marks and Revert buttons.

/**
 * A record's joints, as the complete state a save carries. An explicit null
 * means "no limit overrides" — omitting the field would keep the startup
 * document's value.
 */
function jointsEdit(rec) {
  if (!hasJoints(rec)) return {};
  // `jointsOf`, not `rec.joints`: the latter is materialised lazily by the panel.
  const live = jointsOf(rec);
  if (!live) return {};
  const edit = {};
  // Values only when the array maps onto the asset's joints; an unaddressable
  // object's array is passed over untouched.
  if (live.addressable && live.values.length) {
    edit.joint_values = live.values.slice();
  }
  // Limits only when the asset was read: the server validates each name
  // against the joints it found in the USD.
  if (live.list.length) {
    edit.joint_limits = Object.keys(live.limits).length
      ? structuredClone(live.limits) : null;
  }
  return edit;
}

/**
 * A record's physics, as the complete state a save carries. `mass: null`
 * means "no override, use the asset's own".
 */
function physicsEdit(rec) {
  if (!hasPhysics(rec)) return {};
  const live = physicsOf(rec);
  if (!live) return {};
  const edit = {
    mass: (live.mass != null && live.mass !== live.authoredMass) ? live.mass : null,
  };
  // Friction only for assets whose links the server read from the USD; sending
  // null otherwise would clear a coefficient stored under another link name.
  if (live.links && live.links.length) {
    edit.friction = live.friction != null ? live.friction : null;
  }
  return edit;
}

/**
 * The complete scene snapshot a save or an export sends. Also returns
 * `physics` and `joints` as maps of the panel-level state that went out, for
 * moving the baselines afterwards — the wire form is lossy (a null mass means
 * "no override", a different number per asset).
 */
function sceneEditPayload() {
  const edits = {};
  const remove = [];
  const physics = new Map();
  const joints = new Map();
  for (const rec of objects.values()) {
    if (rec.isCamera || !rec.entry.posable) continue;
    // Removed objects are named explicitly: the server rejects a snapshot that
    // neither keeps nor removes a posable object.
    if (rec.present === false) { remove.push(rec.name); continue; }
    const g = rec.group;
    edits[rec.name] = {
      position: g.position.toArray(),
      orientation: g.quaternion.toArray(),
      // Scale only for the scalable set (props and the room); the server
      // rejects a "scale" key from any other name, the robot included.
      ...(rec.entry.scalable ? { scale: g.scale.toArray() } : {}),
      ...physicsEdit(rec),
      ...jointsEdit(rec),
    };
    if (hasPhysics(rec)) {
      const live = physicsOf(rec);
      physics.set(rec.name, { mass: live.mass, friction: live.friction });
    }
    if (hasJoints(rec)) {
      const live = jointsOf(rec);
      joints.set(rec.name, {
        values: live.values.slice(), limits: structuredClone(live.limits),
      });
    }
  }
  return { edits, remove, physics, joints, ground: groundPayload() };
}

/**
 * Move every baseline to the snapshot that was *sent*, not to what is on
 * screen, so anything edited while the request was in flight stays dirty.
 *
 * @param {object} sent From `sceneEditPayload`, plus `cameraEdits` for an export.
 * @param {object} opts `groundInfo` as the server echoed it, and whether the
 *   camera config was actually written.
 */
function adoptSentBaselines(sent, { groundInfo, camerasWritten = false } = {}) {
  for (const name of sent.remove) {
    const rec = objects.get(name);
    if (rec) { rec.basePresent = false; refreshDirty(rec); }
  }
  for (const [name, snap] of Object.entries(sent.edits)) {
    const rec = objects.get(name);
    if (!rec) continue;
    rec.basePresent = true;
    rec.initial = {
      position: snap.position.slice(),
      orientation: snap.orientation.slice(),
      // The robot is sent without a scale (see the payload builder), so a
      // missing scale keeps the old baseline: never sent means never changed.
      scale: (snap.scale || rec.initial.scale).slice(),
    };
    // Recompute against the current pose so mid-save edits stay dirty.
    refreshDirty(rec);
    // `refreshDirty` covers transform and presence only; the physics and joint
    // baselines move here.
    const physics = sent.physics.get(name);
    if (physics && rec.physics) { rec.physicsSaved = physics; markPhysicsDirty(rec); }
    const joints = sent.joints.get(name);
    if (joints && rec.joints) { rec.jointsSaved = joints; markJointsDirty(rec); }
  }
  if (sent.cameraEdits && camerasWritten) {
    for (const [name, snap] of Object.entries(sent.cameraEdits)) {
      const rec = objects.get(name);
      if (!rec) continue;
      rec.initial = {
        position: snap.position.slice(),
        orientation: snap.orientation.slice(),
        scale: [1, 1, 1],
      };
      refreshDirty(rec);
    }
  }
  // Adopt what the server says it wrote; `undefined` means the response said
  // nothing about it, so the baseline stays put.
  if (groundInfo !== undefined) adoptSavedGround(groundInfo);
  renderList();
  // The Revert buttons compare live vs saved, so refresh after baselines move.
  refreshPhysics();
  refreshJoints();
  refreshSaveButton();
}

function cameraEditPayload() {
  const edits = {};
  for (const name of cameraDirty) {
    const rec = objects.get(name);
    if (!rec) continue;
    edits[name] = {
      position: rec.group.position.toArray(),
      orientation: rec.group.quaternion.toArray(),
    };
  }
  return edits;
}

async function callExport({ dryRun, allowInvalid = false }) {
  // The first heartbeat tells this page which revision it is on.
  await heartbeatReady;
  // Keep the exact snapshot sent, so the response handler adopts it rather
  // than whatever the viewport shows when the reply lands.
  const sent = { ...sceneEditPayload(), cameraEdits: cameraEditPayload() };
  const res = await fetch('/api/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
    body: JSON.stringify({
      dry_run: dryRun,
      // Only sent after the server has refused once and the user confirmed.
      ...(allowInvalid ? { allow_invalid: true } : {}),
      edits: sent.edits,
      remove: sent.remove,
      complete_snapshot: true,
      base_scene_sha256: activeManifest.base_scene_sha256,
      scene_revision: sceneRevision,
      promote_latest: document.getElementById('x-promote').checked,
      ground_plane: sent.ground,
      task: document.getElementById('x-task').value || null,
      camera_edits: sent.cameraEdits,
      camera_revision: cameraRevision,
      // The browser owns the geometric checks; the manifest records its view.
      layout_warnings: refreshLayoutWarnings().map(
        (w) => ({ name: w.name, kind: w.kind, text: w.text })),
    }),
  });
  const body = await res.json();
  if (!res.ok) {
    if (res.status === 409 && body.scene_revision !== undefined) {
      // Never adopt a revision without the content it names; see `setSceneStale`.
      setSceneStale(body.error);
    }
    // Status and body ride on the error: a 422 refusal is a question only the
    // caller can answer (export anyway or not).
    const err = new Error(body.error || res.statusText);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  if (body.scene_revision !== undefined) sceneRevision = body.scene_revision;
  if (body.camera_revision !== undefined) cameraRevision = body.camera_revision;
  // A review ignores it; an export adopts it as the new baseline.
  body.sent = sent;
  return body;
}

function renderExport(body) {
  exportState = body;
  const m = body.manifest;
  const runnable = body.command_runnable;

  const counts = [];
  if (m.scene.moved.length) counts.push(`${m.scene.moved.length} moved`);
  if (m.scene.added.length) counts.push(`${m.scene.added.length} added`);
  if (m.scene.removed.length) counts.push(`${m.scene.removed.length} removed`);
  document.getElementById('x-scene').innerHTML =
    `<div>${counts.join(', ') || 'no changes'}</div>`
    + `<div class="dim">${runnable ? m.scene.path : 'a new timestamped file beside '
        + m.scene.source}</div>`;

  const cams = m.cameras || {};
  const camEdits = Object.keys(cameraEditPayload()).length;
  // An export copies the mutable shared configs under unique names so the run
  // stays reproducible; the review is where that is worth saying.
  const snapshotting = ((m.export || {}).configs || 'snapshot') === 'snapshot';
  const pinned = snapshotting
    ? '<div class="dim">the export copies this config and the task config under '
      + '<code>exports/</code>, records a sha256 for each, and names the copies '
      + '— so editing the shared ones later cannot change what this run was</div>'
    : '<div class="dim">the command names the shared configs, which anyone can '
      + 'edit afterwards</div>';
  document.getElementById('x-cameras').innerHTML = cams.cfg_name
    ? `<div>${cams.cfg_name}${camEdits ? ` — ${camEdits} edited, will be written` : ''}</div>`
      + `<div class="dim">${cams.is_template
          ? 'these are the rig defaults; nobody has aimed them for this room yet'
          : 'authored placement'}</div>` + pinned
    : '<div class="dim">none loaded — start with --cameras to pin one, '
      + 'or the run config&rsquo;s own value stands</div>';

  const warnBox = document.getElementById('x-warnings');
  const all = [
    ...(m.warnings.layout || []).map((w) => `${w.name}: ${w.text}`),
    ...(m.warnings.task || []).map((w) => w.message),
  ];
  warnBox.innerHTML = all.length
    ? all.map((t) => `<div class="xwarn">⚠ ${t}</div>`).join('')
      + '<div class="hint">Recorded in the manifest either way — what the layout '
      + 'looked like when it was frozen is what a surprising result needs later.</div>'
    : '<div class="xwarn ok">nothing to flag</div>';

  document.getElementById('x-command').textContent = m.command.command;
  document.getElementById('x-copy').disabled = !runnable;
  document.getElementById('x-open').disabled = !runnable;
  document.getElementById('x-command-note').textContent = runnable
    ? `Run from ${m.command.cwd}. The path is pinned to this export, so promoting `
      + 'something else later cannot change what it evaluates.'
    : 'Export first — the command names the exact file it will write, '
      + 'and that file does not exist yet.';
  // A task YAML without a `language_instruction` has nothing to pin as the
  // eval prompt, so the run config's own prompt stands.
  if (m.task && !m.task.instruction) {
    document.getElementById('x-command-note').textContent +=
      ' This task config has no language_instruction, so the run config’s own '
      + 'prompt stands — set s15_eval.prompt by hand.';
  }
}

async function reviewExport() {
  if (exportInFlight || sceneStale) return;
  exportInFlight = true;
  document.getElementById('x-status').textContent = 'reviewing…';
  try {
    renderExport(await callExport({ dryRun: true }));
    document.getElementById('x-status').textContent = 'nothing written yet';
  } catch (err) {
    document.getElementById('x-status').textContent = `Review failed: ${err.message}`;
  } finally {
    exportInFlight = false;
  }
}

/**
 * The task this export defaults to: the Task panel's own choice when the
 * server can name a group for it, otherwise the best-confidence association.
 */
function preferredExportTask(select, tasks) {
  const panelGroup = (taskCfg && taskCfg.group) || null;
  if (panelGroup) {
    if (![...select.options].some((o) => o.value === panelGroup)) {
      const option = document.createElement('option');
      option.value = panelGroup;
      option.textContent = `${panelGroup}  ·  chosen in the Task panel`;
      option.title = activeTaskPath || '';
      select.appendChild(option);
    }
    return panelGroup;
  }
  // No panel selection: fall back to the best-confidence association.
  const first = (tasks || []).find((t) => t.group);
  return first ? first.group : '';
}

async function openExport() {
  exportModal().hidden = false;
  if (sceneStale) {
    document.getElementById('x-status').textContent = STALE_MESSAGE;
    return;
  }
  document.getElementById('x-promote').checked =
    document.getElementById('promote').checked;
  refreshPromoteNote('x-promote', 'x-promote-note');

  // Rebuilt on every open: the Task panel's choice can change between opens.
  const select = document.getElementById('x-task');
  try {
    const body = await callExport({ dryRun: true });
    select.innerHTML = '<option value="">(none — the run config default stands)</option>';
    for (const t of body.tasks || []) {
      if (!t.group) continue;
      const option = document.createElement('option');
      option.value = t.group;
      option.textContent = `${t.group}  ·  ${t.confidence}`;
      option.title = t.evidence || '';
      select.appendChild(option);
    }
    const wanted = preferredExportTask(select, body.tasks);
    select.value = wanted;
    document.getElementById('x-task-note').textContent = (taskCfg && taskCfg.group)
      ? 'Defaulted to the config the Task panel has open.'
      : (select.options.length > 1
        ? 'Association is a convention, not a declaration — the confidence says how it was inferred.'
        : 'No task config names this scene.');
    renderExport(body);
  } catch (err) {
    document.getElementById('x-status').textContent = `Review failed: ${err.message}`;
    return;
  }
  await reviewExport();
}

function closeExport() { exportModal().hidden = true; }

/**
 * The trailing clause on a promote checkbox's line: empty when unchecked,
 * a warning when ticked — promotion rewrites the file every downstream stage
 * opens by default.
 */
function refreshPromoteNote(checkboxId, noteId) {
  const note = document.getElementById(noteId);
  const box = document.getElementById(checkboxId);
  if (!note || !box) return;
  note.textContent = box.checked ? ' — every later stage reads it' : '';
}

document.getElementById('btn-review').onclick = openExport;
document.getElementById('export-close').onclick = closeExport;
document.getElementById('export-backdrop').onclick = closeExport;
document.getElementById('x-task').onchange = reviewExport;
document.getElementById('x-promote').onchange = () => {
  refreshPromoteNote('x-promote', 'x-promote-note');
  document.getElementById('promote').checked =
    document.getElementById('x-promote').checked;
  refreshSaveButton();
  reviewExport();
};

document.getElementById('x-export').onclick = async () => {
  if (exportInFlight) return;
  if (sceneStale) {
    document.getElementById('x-status').textContent = STALE_MESSAGE;
    return;
  }
  exportInFlight = true;
  document.getElementById('x-status').textContent = 'exporting…';
  try {
    let body;
    try {
      body = await callExport({ dryRun: false });
    } catch (err) {
      // Re-ask on an invalid task config: the exported command runs later, and
      // a task with an empty goal list reports every episode successful.
      if (err.status !== 422 || err.body?.reason !== 'invalid_task') throw err;
      const answer = await confirmDialog({
        title: 'That task config would not run',
        lines: [
          err.body.error,
          ...(err.body.problems || []).slice(0, 4).map((p) => `• ${p.message}`),
          'Export anyway, or pick another task first?',
        ],
        actions: [
          { label: 'Pick another task', value: 'cancel', primary: true },
          { label: 'Export anyway', value: 'export' },
        ],
      });
      if (answer !== 'export') {
        document.getElementById('x-status').textContent =
          `Not exported — ${describeProblems(err.body.problems)}`;
        return;
      }
      body = await callExport({ dryRun: false, allowInvalid: true });
    }
    renderExport(body);
    // The export is a save: adopt the sent payload and the ground plane the
    // manifest says was written.
    adoptSentBaselines(body.sent, {
      groundInfo: (body.manifest.scene || {}).ground_plane,
      camerasWritten: !!(body.manifest.cameras || {}).written,
    });
    document.getElementById('x-status').textContent =
      `Exported. Manifest: ${(body.manifest_path || '').split('/').pop()}`;
    setStatus(`Exported ${body.manifest.scene.path.split('/').pop()}`, 'ok');
    if (body.settle && body.settle.state === 'running') {
      pollSettle(body.settle.id, 'Exported');
    }
  } catch (err) {
    document.getElementById('x-status').textContent = `Export failed: ${err.message}`;
    setStatus(`Export failed: ${err.message}`, 'err');
  } finally {
    exportInFlight = false;
  }
};

document.getElementById('x-copy').onclick = async () => {
  const text = document.getElementById('x-command').textContent;
  try {
    await navigator.clipboard.writeText(text);
    document.getElementById('x-status').textContent = 'Command copied.';
  } catch (err) {
    // Clipboard access needs a secure context (a LAN bind over plain http is
    // not one); fall back to selecting the text.
    const pre = document.getElementById('x-command');
    const range = document.createRange();
    range.selectNodeContents(pre);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.getElementById('x-status').textContent =
      `Clipboard unavailable (${err.message}) — the command is selected, press Ctrl+C.`;
  }
};

document.getElementById('x-open').onclick = async () => {
  try {
    const res = await fetch('/api/open_folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({ path: exportState.output_dir }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    document.getElementById('x-status').textContent = `Opened ${body.opened}`;
  } catch (err) {
    document.getElementById('x-status').textContent =
      `${err.message} — the path is ${exportState.output_dir}`;
  }
};

// Cached from the last check so the "N warning(s)" note can open the popup
// without refetching.
let lastCheckLayout = [];
let lastCheckTasks = [];

document.getElementById('btn-check').onclick = async () => {
  const note = document.getElementById('check-note');
  note.textContent = 'checking…';
  note.className = 'hint';
  const layout = refreshLayoutWarnings();
  const tasks = await taskWarnings();
  lastCheckLayout = layout;
  lastCheckTasks = tasks;
  const total = layout.length + tasks.length;
  if (!total) {
    note.textContent = 'no problems found';
    setStatus('Checked: every object is in frame, within reach, clear of the others, '
      + 'and no task binding is broken.', 'ok');
    return;
  }
  note.textContent = `${total} warning(s)`;
  note.className = 'hint warn';
  const parts = [];
  if (layout.length) {
    parts.push(`${layout.length} layout warning(s) — click to see them`);
  }
  if (tasks.length) parts.push(describeTaskWarnings(tasks));
  setStatus(`${parts.join('. ')}. These never block a save; `
    + 'the tool cannot know what you meant.', 'err');
};

/**
 * Itemized warnings popup — one row per warning; clicking a layout row selects
 * and frames the object it's about. Reuses the generic confirm modal.
 */
function openWarningsModal() {
  const modal = document.getElementById('confirm-modal');
  const body = document.getElementById('confirm-body');
  const buttons = document.getElementById('confirm-actions');
  document.getElementById('confirm-title').textContent = 'Layout & task warnings';
  body.innerHTML = '';
  body.className = 'warn-modal-body';

  if (lastCheckLayout.length) {
    const h = document.createElement('div');
    h.className = 'warn-modal-heading';
    h.textContent = 'Layout';
    body.appendChild(h);
    for (const w of lastCheckLayout) {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'warn-modal-row';
      const name = document.createElement('strong');
      name.textContent = w.name;
      row.appendChild(name);
      row.appendChild(document.createTextNode(` ${w.text}`));
      row.title = `Select and frame ${w.name}`;
      row.onclick = () => {
        const rec = objects.get(w.name);
        // `finish`, not `modal.hidden`: every exit must remove the Escape listener.
        finish();
        if (rec) { select(rec); frameSelection(); }
      };
      body.appendChild(row);
    }
  }

  if (lastCheckTasks.length) {
    const h = document.createElement('div');
    h.className = 'warn-modal-heading';
    h.textContent = 'Task binding';
    body.appendChild(h);
    for (const w of lastCheckTasks) {
      const row = document.createElement('div');
      row.className = 'warn-modal-row static';
      row.textContent = w.message;
      body.appendChild(row);
    }
  }

  buttons.innerHTML = '';
  const close = document.createElement('button');
  close.textContent = 'Close';
  buttons.appendChild(close);
  modal.hidden = false;

  // The Escape listener is window-level and capture-phase, so every exit path
  // (close, backdrop, Escape, clicked row) must go through `finish` to remove it.
  const finish = () => {
    modal.hidden = true;
    window.removeEventListener('keydown', onKey, true);
  };
  const onKey = (e) => {
    if (e.key !== 'Escape') return;
    e.preventDefault();
    e.stopPropagation();
    finish();
  };
  window.addEventListener('keydown', onKey, true);
  close.onclick = finish;
  document.getElementById('confirm-backdrop').onclick = finish;
}

document.getElementById('check-note').onclick = () => {
  if (lastCheckLayout.length || lastCheckTasks.length) openWarningsModal();
};

document.getElementById('promote').addEventListener('change', refreshSaveButton);

// Named so the scene launcher can run a save on the user's behalf ("save and
// switch"). Resolves true only when the write landed.
async function saveScene() {
  if (saveInFlight) { setStatus('A save is already in progress…'); return false; }
  if (sceneStale) { setStatus(STALE_MESSAGE, 'err'); return false; }
  // Always a complete snapshot of every posable object; anything left out
  // reverts to the base scene. See `sceneEditPayload`.
  const sent = sceneEditPayload();
  // Checked up front so the save summary can include the layout warnings.
  const warnings = refreshLayoutWarnings();
  saveInFlight = true;
  setStatus('Saving…');
  await heartbeatReady;
  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({
        edits: sent.edits,
        remove: sent.remove,
        scene_revision: sceneRevision,
        complete_snapshot: true,
        base_scene_sha256: activeManifest.base_scene_sha256,
        promote_latest: document.getElementById('promote').checked,
        ground_plane: sent.ground,
      }),
    });
    const body = await res.json();
    if (!res.ok) {
      if (res.status === 409) {
        // Deliberately not adopting `body.scene_revision`: taking the number
        // without the content it names would let the next Save push this
        // page's stale snapshot through. Latch stale instead.
        setSceneStale(body.error);
        throw new Error(`${body.error} (your unsaved edits are still on this page; `
          + 'copy anything you need, then reload)');
      }
      throw new Error(body.error || res.statusText);
    }
    if (body.scene_revision !== undefined) sceneRevision = body.scene_revision;
    adoptSentBaselines(sent, { groundInfo: body.ground_plane_info });
    let saved = `Saved ${body.changed.length} object(s) → ${body.path.split('/').pop()}`;
    if (body.promoted) saved += '. _latest.json updated';
    else if (body.promotion_deferred) saved += '. _latest.json deferred to the settled file';
    if (warnings.length) {
      const worst = warnings.slice(0, 3).map((w) => `${w.name} ${w.text}`).join('; ');
      saved += `. ${warnings.length} layout warning(s): ${worst}`
        + (warnings.length > 3 ? ' …' : '');
    }
    const taskNotes = body.task_warnings || [];
    if (taskNotes.length) saved += `. ${describeTaskWarnings(taskNotes)}`;
    if (body.settle && body.settle.state === 'running') {
      setStatus(`${saved}. Settling under physics…`);
      pollSettle(body.settle.id, saved);
    } else {
      setStatus(saved, 'ok');
    }
    return true;
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, 'err');
    return false;
  } finally {
    saveInFlight = false;
  }
}

document.getElementById('btn-save').onclick = () => saveScene();

document.getElementById('btn-look').onclick = () => {
  if (flying) exitFly();
  else enterFly(selected);
};

async function saveCameras() {
  if (!cameraConfig) return false;
  if (cameraSaveInFlight) { setStatus('A camera save is already in progress…'); return false; }
  const edits = {};
  for (const rec of objects.values()) {
    if (!rec.isCamera) continue;
    // The robot's own cameras are drawable but not in this config; the server
    // refuses them by name.
    if (rec.readOnly) continue;
    // Local transform, relative to the parent link — what the external_sensors
    // config stores.
    edits[rec.name] = {
      position: rec.group.position.toArray(),
      orientation: rec.group.quaternion.toArray(),
    };
  }
  cameraSaveInFlight = true;
  setStatus('Saving cameras…');
  const outName = (document.getElementById('cam-out-name').value || '').trim() || undefined;
  const post = (extra = {}) => fetch('/api/save_cameras', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
    // The server refuses the write if the config moved past camera_revision.
    // A blank name means the room-keyed default; a typed one exports beside it.
    body: JSON.stringify({ edits, camera_revision: cameraRevision,
      out_name: outName, ...extra }),
  });
  try {
    let res = await post();
    let body = await res.json();
    // A name taken by a file this session did not write is a question, not a
    // failure. The digest is quoted back on confirm, so a file that changes
    // again while the dialog is open is stale in its turn.
    if (res.status === 409 && body.reason === 'exists') {
      const answer = await confirmDialog({
        title: `${(body.path || '').split('/').pop()} already exists`,
        lines: [
          body.error,
          'It was not written by this editing session, so overwriting it would '
          + 'replace a placement nobody here has seen.',
        ],
        actions: [
          { label: 'Choose another name', value: 'cancel', primary: true },
          { label: 'Overwrite it', value: 'overwrite' },
        ],
      });
      if (answer !== 'overwrite') {
        setStatus('Cameras not saved — pick another name.', 'err');
        return false;
      }
      res = await post({ overwrite_sha256: body.sha256 });
      body = await res.json();
    }
    if (!res.ok) throw new Error(body.error || res.statusText);
    if (typeof body.camera_revision === 'number') cameraRevision = body.camera_revision;
    // Baseline is the sent snapshot, not the live pose — see the scene save.
    for (const [name, snap] of Object.entries(edits)) {
      const rec = objects.get(name);
      if (!rec) continue;
      rec.initial = {
        position: snap.position.slice(),
        orientation: snap.orientation.slice(),
        scale: [1, 1, 1],
      };
      refreshDirty(rec);
    }
    renderList();
    // Resuming is keyed by the background: only a write to
    // `<background>_cameras.yaml` is picked up by other scenes in this room.
    const roomKeyed = !!body.background && body.cfg_name === `${body.background}_cameras`;
    const room = body.background ? `every scene shot in “${body.background}”` : 'this scene';
    document.getElementById('cam-memory').textContent = roomKeyed
      ? `saved — reopening ${room} loads this placement`
      : `saved as “${body.cfg_name}” — name it to use it; `
        + `${room} still opens on the room default`;
    setStatus(`Saved ${body.changed.length} camera(s) → ${body.path.split('/').pop()}  `
      + `(use external_sensors_cfg=${body.cfg_name}). `
      + (roomKeyed
        ? `Reopening ${room} starts from here.`
        : 'Saved beside the room default rather than as it, so opening another '
          + 'scene in this room will not pick it up.'), 'ok');
    return true;
  } catch (err) {
    setStatus(`Camera save failed: ${err.message}`, 'err');
    return false;
  } finally {
    cameraSaveInFlight = false;
  }
}

document.getElementById('btn-save-cameras').onclick = () => saveCameras();

// Settling boots Isaac Sim out of process and takes minutes, so the save
// returns immediately and the result is polled.
async function pollSettle(jobId, savedMessage) {
  const started = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 3000));
    let status;
    try {
      const res = await fetch(`/api/settle?id=${jobId}`);
      status = await res.json();
      if (!res.ok) throw new Error(status.error || res.statusText);
    } catch (err) {
      setStatus(`${savedMessage}. Settle status unavailable: ${err.message}`, 'err');
      return;
    }

    // Anything that is not a finished settle is handled explicitly.
    if (status.state === 'running' || status.state === 'queued') {
      const secs = Math.round((Date.now() - started) / 1000);
      const what = status.state === 'queued' ? 'Queued behind another settle' : 'Settling under physics';
      setStatus(`${savedMessage}. ${what}… ${secs}s`);
      if (Date.now() - started > SETTLE_GIVE_UP_MS) {
        setStatus(`${savedMessage}. Settle has not reported in `
          + `${Math.round(SETTLE_GIVE_UP_MS / 60000)} min — check the server log.`, 'err');
        return;
      }
      continue;
    }
    if (status.state === 'superseded') {
      setStatus(`${savedMessage}. Settle skipped: ${status.error || 'a newer save superseded it'}.`);
      return;
    }
    if (status.state === 'failed') {
      setStatus(`${savedMessage}. Settle FAILED: ${status.error || 'unknown'}`, 'err');
      return;
    }
    if (status.state !== 'done') {
      setStatus(`${savedMessage}. Settle returned an unknown state `
        + `"${status.state}" — check the server log.`, 'err');
      return;
    }

    const moved = status.moved || [];
    const jointsMoved = status.joints_moved || [];
    const unchecked = status.joints_unchecked || [];
    // Use the server's own verdict: it weighs root travel (metres), joint
    // drift (metres or degrees per joint) and comparability against separate
    // tolerances. `undefined` is a report from before the verdict existed;
    // fall back to the counts.
    const verified = status.ok === undefined
      ? !moved.length && !jointsMoved.length && !unchecked.length
      : status.ok === true;
    const file = (status.settled_path || '').split('/').pop() || '(no file)';
    // Always say whether _latest was updated; a refusal must be visible.
    let promotion = '';
    if (status.promoted) promotion = ' _latest.json updated.';
    else if (status.promotion_blocked) {
      promotion = ` _latest.json NOT updated: ${status.promotion_note || 'refused'}.`;
    } else if (status.promote_requested) promotion = ' _latest.json NOT updated.';

    if (verified && !moved.length && !jointsMoved.length && !unchecked.length) {
      setStatus(`${savedMessage}. Physics-verified: nothing moved → ${file}.${promotion}`, 'ok');
    } else {
      // One clause per kind of drift, each in its own unit.
      const bits = [];
      bits.push(moved.length
        ? `${moved.length} root(s) moved: ${moved.slice().sort((a, b) => b.delta - a.delta)
            .map((m) => `${m.name} ${m.delta.toFixed(3)} m`).join(', ')}`
        : '0 roots moved');
      if (jointsMoved.length) {
        bits.push(jointsMoved.flatMap((entry) => (entry.joints || []).map(
          (d) => `${entry.name} ${d.joint} drifted ${d.delta}${d.unit === 'deg' ? '°' : ' m'}`,
        )).join('; '));
      }
      if (unchecked.length) {
        bits.push(`${unchecked.length} object(s) could not be verified: `
          + unchecked.map((u) => `${u.name} (${u.reason})`).join('; '));
      }
      // Large movement means the object was floating or intersecting; a
      // drifted joint means gravity closed it.
      setStatus(`${savedMessage}. Settled with findings: ${bits.join('; ')} `
        + `→ ${file}.${promotion}`, 'err');
    }
    showSettleDrift(status);
    return;
  }
}

// --- the scene launcher ----------------------------------------------------
// A switch rebinds the server — a different extraction, editable set and
// camera rig — so the page ends by reloading. The server refuses to switch
// while holding unsaved imports, and this side asks about unsaved edits first.

let catalogue = null;              // last /api/scenes payload
let launcherFilter = '';
const expandedHistory = new Set(); // scene dirs whose variants are showing
// Set just before location.reload() so the beforeunload guard does not fire.
let switching = false;

const launcherIsOpen = () => !document.getElementById('launcher-modal').hidden;
const composeIsOpen = () => !document.getElementById('compose-modal').hidden;
const confirmIsOpen = () => !document.getElementById('confirm-modal').hidden;

/** Ask a question with several answers; resolves with the chosen action's value. */
function confirmDialog({ title, lines = [], actions }) {
  const modal = document.getElementById('confirm-modal');
  const body = document.getElementById('confirm-body');
  const buttons = document.getElementById('confirm-actions');
  document.getElementById('confirm-title').textContent = title;
  body.innerHTML = '';
  for (const line of lines) {
    if (Array.isArray(line)) {
      const list = document.createElement('ul');
      for (const item of line) {
        const li = document.createElement('li');
        li.textContent = item;
        list.appendChild(li);
      }
      body.appendChild(list);
    } else {
      const div = document.createElement('div');
      div.textContent = line;
      body.appendChild(div);
    }
  }
  buttons.innerHTML = '';
  modal.hidden = false;

  return new Promise((resolve) => {
    const finish = (value) => {
      modal.hidden = true;
      window.removeEventListener('keydown', onKey, true);
      resolve(value);
    };
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      // Capture phase and stopPropagation: Escape here must not also deselect
      // an object or close the launcher underneath.
      e.preventDefault();
      e.stopPropagation();
      finish('cancel');
    };
    window.addEventListener('keydown', onKey, true);
    document.getElementById('confirm-backdrop').onclick = () => finish('cancel');
    for (const action of actions) {
      const button = document.createElement('button');
      button.textContent = action.label;
      if (action.primary) button.className = 'primary';
      button.onclick = () => finish(action.value);
      buttons.appendChild(button);
    }
    const first = buttons.querySelector('button.primary') || buttons.firstChild;
    if (first) first.focus();
  });
}

/** Everything this page is holding that is not on disk. */
function unsavedSummary() {
  const lines = [];
  // Presence-filtered so a removal is not also counted as moved. The verbs
  // deliberately match the Save button's.
  const moved = [...dirty].filter((n) => {
    const rec = objects.get(n);
    return rec && !rec.added && rec.present !== false;
  });
  const added = [...objects.values()].filter((r) => r.added && r.present !== false);
  const removed = removedRecords().filter((r) => r.basePresent !== false);
  if (moved.length) lines.push(`${moved.length} moved object(s): ${moved.join(', ')}`);
  if (added.length) {
    lines.push(`${added.length} added object(s): ${added.map((r) => r.name).join(', ')}`);
  }
  if (removed.length) {
    lines.push(`${removed.length} removed object(s): ${removed.map((r) => r.name).join(', ')}`);
  }
  if (cameraDirty.size) lines.push(`${cameraDirty.size} moved camera(s): ${[...cameraDirty].join(', ')}`);
  // Physics, joint and range edits live in panel fields a reload rebuilds from
  // disk, so they are as lost as an unsaved transform.
  if (physicsDirty.size) lines.push(`${physicsDirty.size} retuned object(s): ${[...physicsDirty].join(', ')}`);
  // The ground plane has no per-object line to ride on, so it gets its own.
  if (groundDirty) lines.push('an edited ground plane');
  if (jointsDirty.size) lines.push(`${jointsDirty.size} object(s) with edited joints: ${[...jointsDirty].join(', ')}`);
  if (taskDirty) lines.push(`edited randomization ranges in ${taskFileName()}`);
  return lines;
}

/**
 * Deal with unsaved work before something that would throw it away.
 * Returns 'go' once it is safe to proceed, or 'cancel'; saving is the default.
 */
async function guardUnsaved(what) {
  // The predicate decides *whether* to ask; the summary only says what about.
  if (!hasUnsavedSceneEdits() && cameraDirty.size === 0 && !taskDirty) return 'go';
  const lines = unsavedSummary();
  const answer = await confirmDialog({
    title: 'Unsaved changes',
    lines: [
      `${what} reloads this page, and these changes are only here:`,
      lines,
      'Saving writes a new timestamped scene file; it does not touch _latest.json '
      + 'unless that box is ticked.'
      + (taskDirty ? ' Task ranges go back into the yaml itself.' : ''),
    ],
    actions: [
      { label: 'Save, then continue', value: 'save', primary: true },
      { label: 'Discard and continue', value: 'discard' },
      { label: 'Stay here', value: 'cancel' },
    ],
  });
  // "Stay here" leaves a status line so the refusal is visible.
  if (answer === 'cancel') { setStatus('Stayed on this scene.'); return 'cancel'; }
  if (answer === 'discard') return 'go';

  // Three files, three saves: the scene JSON, the camera YAML and the task YAML.
  let ok = true;
  if (hasUnsavedSceneEdits()) ok = await saveScene();
  if (ok && cameraDirty.size) ok = await saveCameras();
  if (ok && taskDirty) ok = await saveTaskRanges();
  if (!ok) {
    setStatus('Save failed, so nothing was switched — your edits are still here.', 'err');
    return 'cancel';
  }
  return 'go';
}

/** Open a scene, dealing with unsaved work on both sides first. */
async function openScene(path, { label } = {}) {
  if (await guardUnsaved(`Opening ${label || 'another scene'}`) === 'cancel') return;
  setStatus(`Opening ${label || path.split('/').pop()}…`);
  let discardPending = false;

  for (let attempt = 0; attempt < 2; attempt++) {
    let res, body;
    try {
      res = await fetch('/api/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
        body: JSON.stringify({ path, discard_pending: discardPending }),
      });
      body = await res.json();
    } catch (err) {
      setStatus(`Could not open that scene: ${err.message}`, 'err');
      return;
    }
    // The server holds imports this page may not know about (a second tab
    // could have made them), so the list comes from the server.
    if (res.status === 409 && body.pending_adds) {
      const answer = await confirmDialog({
        title: 'Unsaved imports on the server',
        lines: [
          'This session imported object(s) that no save has written yet:',
          body.pending_adds,
          'Opening another scene discards them. The copied asset files stay on '
          + 'disk under the scene’s objects/ directory.',
        ],
        actions: [
          { label: 'Stay here', value: 'cancel', primary: true },
          { label: 'Discard and open', value: 'discard' },
        ],
      });
      if (answer !== 'discard') { setStatus('Stayed on this scene.'); return; }
      discardPending = true;
      continue;
    }
    if (!res.ok) {
      // bind_scene leaves the previous scene bound when it fails, so this page
      // is still valid and must not reload.
      setStatus(`Could not open that scene: ${body.error || res.statusText}`, 'err');
      return;
    }
    if (body.unchanged) { closeLauncher(); setStatus('Already editing that scene.'); return; }
    switching = true;
    setStatus(`Opened ${body.name}. Reloading…`);
    location.reload();
    return;
  }
}

async function loadCatalogue() {
  const res = await fetch('/api/scenes', { cache: 'no-store' });
  if (!res.ok) throw new Error(`scenes -> HTTP ${res.status}`);
  catalogue = await res.json();
  return catalogue;
}

function sceneMeta(row) {
  if (row.error) return `unreadable: ${row.error}`;
  const parts = [];
  if (row.objects !== undefined) parts.push(`${row.props} prop(s) of ${row.objects}`);
  if (row.background) parts.push(row.background);
  if (row.missing) parts.push(`${row.missing} asset(s) missing`);
  if (row.composed_from) parts.push(`from ${row.composed_from.template_scene}`);
  // Only reconstructed scenes are tagged; shipping with SimFoundry is the
  // unremarkable case.
  if (row.source === 'generated') parts.push('from your video');
  return parts.join(' · ');
}

// --- task association for a scene -------------------------------------------
// Which task config to load alongside a scene: the export flow's auto-discovery
// lookup, offered at scene-open time with a manual override.

const TASK_FOR_SCENE_KEY = 'simfoundry.light-editor.task-for-scene';
// The task attached to the scene on screen; null when none. Read by the TASK
// sidebar section.
let activeTaskPath = null;

function readTaskForSceneMap() {
  try {
    const raw = JSON.parse(localStorage.getItem(TASK_FOR_SCENE_KEY) || '{}');
    return (raw && typeof raw === 'object' && !Array.isArray(raw)) ? raw : {};
  } catch {
    return {};
  }
}

function rememberTaskForScene(scenePath, taskPath) {
  const map = readTaskForSceneMap();
  // '' (not a missing key) records "explicitly no task" — distinct from never
  // having chosen, which still defaults to the best auto-discovered match.
  map[scenePath] = taskPath || '';
  try {
    localStorage.setItem(TASK_FOR_SCENE_KEY, JSON.stringify(map));
  } catch {
    // Best-effort: a disabled localStorage just means the pick does not
    // survive a reload.
  }
}

async function fetchTasksForScene(scenePath) {
  try {
    const res = await fetch(`/api/tasks?scene=${encodeURIComponent(scenePath)}`,
      { cache: 'no-store' });
    if (!res.ok) return [];
    const body = await res.json();
    return body.tasks || [];
  } catch {
    return [];
  }
}

function addCustomTaskOption(select, path, label) {
  if ([...select.options].some((o) => o.value === path)) return;
  const opt = document.createElement('option');
  opt.value = path;
  opt.textContent = label || path.split('/').pop();
  // The visible text is not unique; the title carries the path.
  opt.title = path;
  select.appendChild(opt);
}

/** Fill in a row's task dropdown: discovered matches, plus whatever's remembered. */
async function populateTaskPick(select, scenePath) {
  const map = readTaskForSceneMap();
  const chosen = Object.prototype.hasOwnProperty.call(map, scenePath);
  const remembered = map[scenePath] || '';
  const tasks = await fetchTasksForScene(scenePath);
  for (const t of tasks) {
    const opt = document.createElement('option');
    opt.value = t.path;
    opt.textContent = t.name + (t.confidence ? ` (${t.confidence})` : '');
    // Two files can carry the same task_name; the title tells them apart.
    opt.title = t.path;
    select.appendChild(opt);
  }
  if (remembered && ![...select.options].some((o) => o.value === remembered)) {
    addCustomTaskOption(select, remembered);
  }
  // Nothing chosen yet: default to the best auto-discovered match without
  // writing storage. A stored '' is an explicit "(no task)" and outranks it.
  select.value = chosen ? remembered : (tasks[0] ? tasks[0].path : '');
}

/**
 * Why *path* is not usable as this scene's task config, or null if it is.
 * Asked of the server — the check needs the file's contents — through the
 * same endpoint the TASK panel loads from.
 */
async function taskConfigProblem(path) {
  try {
    const res = await fetch(`/api/task_cfg?path=${encodeURIComponent(path)}`,
      { cache: 'no-store' });
    const body = await res.json();
    return res.ok ? null : (body.error || res.statusText);
  } catch (err) {
    return err.message;
  }
}

/**
 * Attach *taskPath* to *scenePath* and, when that is the scene on screen,
 * point the TASK panel at it. Returns false when the change did not happen.
 */
async function chooseTaskForScene(scenePath, taskPath) {
  // Validate before remembering: associations persist in localStorage and
  // re-apply on every reload.
  if (taskPath) {
    const problem = await taskConfigProblem(taskPath);
    if (problem) {
      setStatus(`Not attached — ${problem}`, 'err');
      return false;
    }
  }
  const open = scenePath === activeManifest.scene_json;
  // Guarded even when re-picking the file already attached:
  // `refreshTaskSection` rebuilds from disk and would destroy typed ranges.
  const same = (taskPath || null) === activeTaskPath;
  if (open && await guardTaskRanges(same
    ? `Re-reading ${taskFileName()}`
    : 'Attaching another task config') === 'cancel') {
    return false;
  }
  rememberTaskForScene(scenePath, taskPath);
  if (open) {
    activeTaskPath = taskPath || null;
    await refreshTaskSection();
  }
  return true;
}

function sceneRow(row, { current }) {
  const el = document.createElement('div');
  el.className = 'scene' + (current ? ' current' : '')
    + (row.missing || row.error ? ' broken' : '');
  el.dataset.scene = row.path;

  const name = document.createElement('span');
  name.className = 'name';
  name.textContent = row.name;
  const meta = document.createElement('span');
  meta.className = 'meta';
  // The open row keeps its facts, so a broken scene still shows why it is amber.
  const facts = sceneMeta(row);
  meta.textContent = current ? ['open now', facts].filter(Boolean).join(' · ') : facts;
  meta.title = row.path;
  el.append(name, meta);

  // History is opt-in per row so timestamped saves do not bury the list. The
  // count comes from `variants` — what the expanded list can actually show —
  // with `variant_count` as fallback and `variant_hidden` for held-back rows.
  const shown = Array.isArray(row.variants) ? row.variants.length : (row.variant_count || 1);
  const hidden = Number.isFinite(row.variant_hidden) ? row.variant_hidden : 0;
  if (shown > 1) {
    const history = document.createElement('button');
    history.className = 'history';
    history.textContent = `${shown} saves`;
    history.title = hidden
      ? `Earlier saves of this scene — the newest ${shown} of ${shown + hidden}`
      : 'Earlier saves of this scene';
    history.onclick = (e) => {
      e.stopPropagation();
      if (expandedHistory.has(row.dir)) expandedHistory.delete(row.dir);
      else expandedHistory.add(row.dir);
      renderLauncher();
    };
    el.appendChild(history);
  } else {
    el.appendChild(document.createElement('span'));
  }

  // Which task config opens alongside this scene. Populated lazily — a task
  // lookup per row on every page load is too costly for rows nobody clicks.
  const taskPick = document.createElement('select');
  taskPick.className = 'task-pick';
  taskPick.title = 'Task config to load with this scene';
  taskPick.onclick = (e) => e.stopPropagation();
  const noneOpt = document.createElement('option');
  noneOpt.value = '';
  noneOpt.textContent = '(no task)';
  taskPick.appendChild(noneOpt);
  taskPick.onchange = async () => {
    // The open scene's row is a live control over the TASK panel.
    if (!await chooseTaskForScene(row.path, taskPick.value || null)) {
      taskPick.value = activeTaskPath || '';
    }
  };
  el.appendChild(taskPick);
  populateTaskPick(taskPick, row.path);

  const taskBrowse = document.createElement('button');
  taskBrowse.className = 'task-browse';
  taskBrowse.textContent = '…';
  taskBrowse.title = 'Pick a task yaml by path';
  taskBrowse.onclick = (e) => {
    e.stopPropagation();
    openPicker({
      filter: 'yaml',
      mode: 'file',
      title: 'Choose a task config',
      // Added to the dropdown only once accepted; a rejected path must not be
      // re-offered.
      onChoose: async (path) => {
        if (!await chooseTaskForScene(row.path, path)) {
          taskPick.value = activeTaskPath || '';
          return;
        }
        addCustomTaskOption(taskPick, path);
        taskPick.value = path;
      },
    });
  };
  el.appendChild(taskBrowse);

  // Which scanned room the scene opens in — chosen with a scene rather than
  // stored in it, like the task pick.
  const roomPick = document.createElement('select');
  roomPick.className = 'task-pick room-pick';
  roomPick.title = 'Scanned room to open this scene in';
  roomPick.onclick = (e) => e.stopPropagation();
  const keepOpt = document.createElement('option');
  keepOpt.value = '';
  keepOpt.textContent = '(room as saved)';
  roomPick.appendChild(keepOpt);
  roomPick.onchange = () => {
    rememberRoomForScene(row.path, roomPick.value || null);
    // On the open scene's row the change applies immediately.
    if (current && roomPick.value) applyRoom(roomPick.value);
  };
  el.appendChild(roomPick);
  populateRoomPick(roomPick, row.path);

  // Switching is a two-step: the row arms, the button confirms. No room
  // argument — `openScene` ends in a page reload, so the choice travels
  // through storage and `applyChosenRoom` picks it up afterwards.
  if (!current) {
    const go = document.createElement('button');
    go.className = 'switch primary';
    go.textContent = 'Switch scene';
    go.title = `Open ${row.name}. This reloads the page.`;
    go.onclick = (event) => {
      event.stopPropagation();
      openScene(row.path, { label: row.name });
    };
    el.appendChild(go);
    el.onclick = () => armSceneRow(el);
  } else {
    // The open scene keeps the column empty so the rows still line up.
    el.appendChild(document.createElement('span'));
  }
  return el;
}

/**
 * Arm one scene row, disarming any other; clicking the armed row again
 * disarms it. The button takes focus, so Enter confirms.
 */
function armSceneRow(el) {
  const wasArmed = el.classList.contains('armed');
  // Both kinds of row: arming a timestamped save disarms its scene row too.
  for (const other of document.querySelectorAll('.scene.armed, .variant.armed')) {
    other.classList.remove('armed');
  }
  if (wasArmed) return;
  el.classList.add('armed');
  const go = el.querySelector('.switch');
  if (go) go.focus();
}

// --- which room a scene opens in -------------------------------------------
// Remembered per scene like the task pick and applied after the scene binds;
// nothing is written until Save.

const ROOM_FOR_SCENE_KEY = 'simfoundry.light-editor.room-for-scene';

function readRoomForSceneMap() {
  try {
    const raw = JSON.parse(localStorage.getItem(ROOM_FOR_SCENE_KEY) || '{}');
    return (raw && typeof raw === 'object' && !Array.isArray(raw)) ? raw : {};
  } catch {
    return {};
  }
}

function rememberRoomForScene(scenePath, room) {
  const map = readRoomForSceneMap();
  map[scenePath] = room || '';
  try {
    localStorage.setItem(ROOM_FOR_SCENE_KEY, JSON.stringify(map));
  } catch {
    // Best-effort, same as the task pick.
  }
}

/** The registered rooms, and which one a scene is already laid out in. */
async function fetchRooms(scenePath) {
  try {
    const url = scenePath
      ? `/api/backgrounds?scene=${encodeURIComponent(scenePath)}`
      : '/api/backgrounds';
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) return { current: null, rooms: [] };
    return await res.json();
  } catch {
    return { current: null, rooms: [] };
  }
}

async function populateRoomPick(select, scenePath) {
  const remembered = readRoomForSceneMap()[scenePath] || '';
  const { current, rooms } = await fetchRooms(scenePath);
  // Name the room the scene already has rather than an opaque "(room as saved)".
  select.options[0].textContent = current ? `${current} (as saved)` : '(no room)';
  for (const room of rooms) {
    const opt = document.createElement('option');
    opt.value = room.id;
    opt.textContent = room.label + (room.id === current ? ' — current' : '');
    select.appendChild(opt);
  }
  // A remembered pick matching the current room is not a change.
  select.value = remembered && remembered !== current ? remembered : '';
}

/**
 * Attach *room* to the open session. It arrives as a pending add (whatever it
 * replaces is marked removed), so it counts as unsaved work and reaches the
 * file only through Save — the same path an imported prop takes.
 */
async function applyRoom(room) {
  if (!room) return false;
  setStatus(`Attaching ${room}…`);
  let body;
  try {
    const res = await fetch('/api/background', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({ id: room }),
    });
    body = await res.json();
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  } catch (err) {
    setStatus(`Could not attach ${room}: ${err.message}`, 'err');
    return false;
  }

  // A room from the file is marked removed (listed under `remove`, undoable);
  // a room that was itself only a pending attach is withdrawn outright — the
  // scene never held it, so there is nothing to restore. A room can also
  // prescribe its robot(s) (`robot_entries` — two for a bimanual room like
  // the YAM workstation), swapped in the same way alongside it.
  const withdrawn = [...(body.dropped || []), ...(body.robot_dropped || [])];
  const rec = await instantiate(body.entry);   // publishes into `objects`
  rec.added = true;
  const snaps = [{ ...snapshot(rec), present: false }];
  const replaced = [...(body.replaced || [])];
  const robotRecs = [];
  for (const robotEntry of body.robot_entries || []) {
    const robotRec = await instantiate(robotEntry);
    robotRec.added = true;
    robotRecs.push(robotRec);
    snaps.push({ ...snapshot(robotRec), present: false });
  }
  if (robotRecs.length) replaced.push(...(body.robot_replaced || []));
  for (const name of replaced) {
    const old = objects.get(name);
    if (old) snaps.push({ ...snapshot(old), present: true });
  }
  // One undo entry for the whole swap: a single Ctrl+Z reverses every half.
  undoStack.push({ label: 'attach room', snaps });
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack.length = 0;
  updateHistoryButtons();
  for (const name of withdrawn) {
    const stale = objects.get(name);
    if (!stale) continue;
    if (selection.has(stale)) select(null);
    scene.remove(stale.group);
    objects.delete(name);
    dirty.delete(name);
    physicsDirty.delete(name);
  }
  // Prune history steps naming a withdrawn room/robot; drop steps left empty.
  if (withdrawn.length) {
    const gone = new Set(withdrawn);
    for (const stack of [undoStack, redoStack]) {
      for (let i = stack.length - 1; i >= 0; i--) {
        const step = stack[i];
        if (!Array.isArray(step.snaps)) continue;
        step.snaps = step.snaps.filter((s) => !gone.has(s.name));
        if (!step.snaps.length) stack.splice(i, 1);
      }
    }
    updateHistoryButtons();
  }
  for (const name of replaced) {
    const old = objects.get(name);
    if (old) setPresent(old, false);
  }
  setPresent(rec, true);
  for (const robotRec of robotRecs) setPresent(robotRec, true);
  refreshSaveButton();
  renderList();
  refreshBackgroundToggle();
  // Both are room-scoped server state fetched once at boot; without
  // re-fetching here they'd keep showing/targeting the room just left.
  loadTableCentre();
  loadGroundPlane();
  setStatus(`${body.label} attached`
    + (replaced.length ? `, replacing ${replaced.join(', ')}` : '')
    + '. Save to keep it.');
  return true;
}

/**
 * Attach the room this scene was opened with, if it is not already in it.
 * Runs after boot — the choice travels through storage across the reload —
 * and is a no-op once a save has written the room (`current` matches).
 */
async function applyChosenRoom() {
  const scenePath = activeManifest && activeManifest.scene_json;
  if (!scenePath) return;
  const wanted = readRoomForSceneMap()[scenePath] || '';
  if (!wanted) return;
  const { current } = await fetchRooms(scenePath);
  if (wanted === current) return;
  await applyRoom(wanted);
}

// The Scene panel's always-visible recent-scenes strip. Hits
// /api/scenes/recent, which skips the full discover_scenes() pass the
// launcher pays for.
async function loadRecentScenes() {
  const box = document.getElementById('scene-recent');
  try {
    const res = await fetch('/api/scenes/recent', { cache: 'no-store' });
    if (!res.ok) throw new Error(`scenes/recent -> HTTP ${res.status}`);
    const { recent } = await res.json();
    box.innerHTML = '';
    // Every scene the server has; the height cap lives in the CSS.
    const rows = recent || [];
    if (rows.length === 0) { box.hidden = true; return; }
    const heading = document.createElement('div');
    heading.className = 'scene-group';
    heading.textContent = 'Recent';
    // The heading stays put; only the rows scroll.
    const list = document.createElement('div');
    list.id = 'scene-recent-list';
    for (const row of rows) list.appendChild(sceneRow(row, { current: false }));
    box.append(heading, list);
    box.hidden = false;
  } catch (err) {
    box.hidden = true;
    console.warn('could not load recent scenes', err);
  }
}

// --- task range editor -------------------------------------------------------
// The per-group randomization ranges (`group_xyz_randomization` /
// `group_z_rot_randomization`) the active task yaml declares, for every group
// the config knows about. Which config that is follows activeTaskPath.

// Last /api/task_cfg payload: the groups the panel draws, the sha256 a save
// hands back, and the workspace box.
let taskCfg = null;
// Range edits live only in these inputs until Save, so they count as unsaved work.
let taskDirty = false;
// One POST at a time: a second in-flight save would carry a superseded sha256.
let taskSaving = false;

const taskGroups = () => (taskCfg && taskCfg.groups) || {};
const taskFileName = () => (activeTaskPath || '').split('/').pop();

// Radians in the yaml, degrees in the panel; rounded so 0.5236 rad shows as
// 30° rather than 30.000000000000004°.
const taskDegrees = (radians) => (Number.isFinite(radians)
  ? Math.round(radians * 180 / Math.PI * 1e4) / 1e4
  : null);

/**
 * Record, on a yaw field, the radians the file holds and the degrees shown
 * for them. The degree display is lossy, so `readTaskGroupSpec` uses this
 * pair to tell an untouched field from an edited one and post the file's own
 * radians back unchanged.
 */
function rememberFileZRot(input, radians) {
  input.dataset.fileRad = Number.isFinite(radians) ? String(radians) : '';
  input.dataset.fileShown = input.value;
}

async function initTaskSection() {
  await refreshTaskPick();
  await refreshTaskSection();
}

/** Fill the panel's own task picker, and adopt whatever it resolves to. */
async function refreshTaskPick() {
  const select = document.getElementById('task-pick');
  select.innerHTML = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = '(no task)';
  select.appendChild(none);
  await populateTaskPick(select, activeManifest.scene_json);
  activeTaskPath = select.value || null;
}

document.getElementById('task-pick').onchange = async (event) => {
  await chooseTaskForScene(activeManifest.scene_json, event.target.value || null);
  // A refused switch (unsaved ranges, stay here) must not leave the picker
  // showing a file the panel is not editing.
  event.target.value = activeTaskPath || '';
};

document.getElementById('btn-task-browse').onclick = () => {
  const select = document.getElementById('task-pick');
  openPicker({
    filter: 'yaml',
    mode: 'file',
    title: 'Choose a task config',
    onChoose: async (path) => {
      // Added only after acceptance: a refused path must not be re-offered.
      if (await chooseTaskForScene(activeManifest.scene_json, path)) {
        addCustomTaskOption(select, path);
      }
      select.value = activeTaskPath || '';
    },
  });
};

function updateTaskButtons() {
  document.getElementById('btn-task-save').disabled = !taskDirty || taskSaving;
  document.getElementById('btn-task-revert').disabled = !taskDirty || taskSaving;
}

function setTaskDirty(value) {
  taskDirty = value;
  updateTaskButtons();
}

const markTaskDirty = () => setTaskDirty(true);

/** Deal with unsaved range edits before something that would drop them. */
async function guardTaskRanges(what) {
  if (!taskDirty) return 'go';
  const answer = await confirmDialog({
    title: 'Unsaved ranges',
    lines: [
      `${what} reloads this panel, and these edits are only here:`,
      [`edited randomization ranges in ${taskFileName()}`],
      'Saving writes them back into that yaml itself.',
    ],
    actions: [
      { label: 'Save, then continue', value: 'save', primary: true },
      { label: 'Discard and continue', value: 'discard' },
      { label: 'Stay here', value: 'cancel' },
    ],
  });
  // Refusing leaves a status line, same as the scene guard.
  if (answer === 'cancel') {
    setStatus(`Still editing the ranges in ${taskFileName()}.`);
    return 'cancel';
  }
  if (answer === 'discard') {
    // Run the Revert path too, so a caller that guards twice is not asked again.
    renderTaskGroups();
    setTaskDirty(false);
    return 'go';
  }
  return (await saveTaskRanges()) ? 'go' : 'cancel';
}

/**
 * One numeric field. A non-number `value` leaves the input blank, which is
 * how "this group has no entry" is both shown and typed back away again.
 */
function taskNumField(labelText, cls, value, { disabled = false, title = '' } = {}) {
  const wrap = document.createDocumentFragment();
  const label = document.createElement('label');
  label.textContent = labelText;
  const input = document.createElement('input');
  input.type = 'number';
  input.step = cls === 'tg-zrot' ? '0.5' : '0.005';
  input.min = '0';
  input.className = cls;
  input.value = Number.isFinite(value) ? value : '';
  input.disabled = disabled;
  if (title) { label.title = title; input.title = title; }
  // The field's value just before this edit; `beforeinput` is the only moment
  // it can be read.
  let before = '';
  input.addEventListener('beforeinput', () => { before = input.value; });
  input.oninput = (event) => {
    // A paste the browser cannot read as a number empties the field,
    // indistinguishably from a deliberate clear — and blank is not zero here
    // (see `readTaskGroupSpec`). Put a non-numeric paste back; Delete and
    // Backspace still clear.
    if (input.value === '' && before !== '' && event.inputType === 'insertFromPaste') {
      input.value = before;
      setStatus(`That paste is not a number — ${labelText} is still ${before}.`, 'err');
    }
    // X, Y and Z are one 3-vector in the yaml: typing into any of them makes
    // the other blanks explicit zeros, shown on screen as what the save writes.
    if (cls !== 'tg-zrot' && input.value !== '') {
      for (const sel of ['.tg-x', '.tg-y', '.tg-z']) {
        const sibling = input.closest('.task-group').querySelector(sel);
        if (sibling && sibling !== input && sibling.value === '') sibling.value = '0';
      }
    }
    markTaskDirty();
    showWorkspaceNote(input.closest('.task-group'));
    refreshTaskRangeBoxes();
  };
  wrap.append(label, input);
  return wrap;
}

/** Put a group's fields back to what /api/task_cfg last reported for it. */
function setTaskGroupFields(block, values) {
  const xyz = Array.isArray(values.xyz) ? values.xyz : [null, null, null];
  ['.tg-x', '.tg-y', '.tg-z'].forEach((sel, i) => {
    block.querySelector(sel).value = Number.isFinite(xyz[i]) ? xyz[i] : '';
  });
  const zrotField = block.querySelector('.tg-zrot');
  const zrot = taskDegrees(values.z_rot);
  zrotField.value = zrot === null ? '' : zrot;
  rememberFileZRot(zrotField, values.z_rot);
  showWorkspaceNote(block);
  refreshTaskRangeBoxes();
}

/** What the group binds: its yaml keys, and what those found in this scene. */
function taskBindingNote(values) {
  const note = document.createElement('div');
  const keys = (values.keys || []).join(', ');
  const objects = (values.objects || []).join(', ');
  if (objects) {
    note.className = 'note';
    note.textContent = `${keys || '(no keys)'} → ${objects}`;
    return note;
  }
  // No bound object: the ranges below have nothing to apply to.
  note.className = 'note warn';
  note.textContent = keys
    ? `${keys} → nothing in this scene`
    : 'no semantic_group_mapping entry, and nothing in this scene';
  return note;
}

/**
 * Warn when a typed range cannot fit the workspace. `workspace_bounds` is in
 * the robot base frame and the reset clamps against its world-frame AABB,
 * which this panel cannot compute — so only widths are compared. Z is exact;
 * X and Y are measured against the box's narrower side and hedged to "may".
 */
function showWorkspaceNote(block) {
  const note = block && block.querySelector('.tg-bounds');
  if (!note) return;
  const bounds = (taskCfg && taskCfg.workspace_bounds) || null;
  const lower = bounds && bounds.lower;
  const upper = bounds && bounds.upper;
  note.hidden = true;
  if (!Array.isArray(lower) || !Array.isArray(upper)) return;
  const extent = [0, 1, 2].map((i) => upper[i] - lower[i]);
  if (!extent.every((e) => Number.isFinite(e) && e > 0)) return;

  // The reset passes no bounds for the robot group.
  if (block.dataset.group === 'robot') return;

  // A disabled axis is overwritten by predicate placement afterwards.
  const half = (sel) => {
    const input = block.querySelector(sel);
    const value = parseFloat(input.value);
    return !input.disabled && Number.isFinite(value) ? value : 0;
  };
  const narrow = Math.min(extent[0], extent[1]);
  const said = [];
  // The field holds a half-width; the message quotes the full span the test
  // actually compares.
  const spread = (sel) => (2 * half(sel)).toFixed(3);
  // X and Y are measured against the same side of the box, so name them together.
  const flat = [['X', '.tg-x'], ['Y', '.tg-y']]
    .filter(([, sel]) => 2 * half(sel) > narrow)
    .map(([axis, sel]) => `±${axis} ${half(sel).toFixed(3)} m is a ${spread(sel)} m spread`);
  if (flat.length) {
    said.push(`${flat.join(' and ')}, ${flat.length > 1 ? 'both ' : ''}wider than the `
      + `workspace's narrow side (${narrow.toFixed(2)} m), so reset may clamp `
      + (flat.length > 1 ? 'them' : 'it'));
  }
  if (2 * half('.tg-z') > extent[2]) {
    said.push(`±Z ${half('.tg-z').toFixed(3)} m is a ${spread('.tg-z')} m spread, taller `
      + `than the workspace (${extent[2].toFixed(2)} m), so reset clamps it`);
  }
  if (!said.length) return;
  note.textContent = `${said.join('; ')}.`;
  note.hidden = false;
}

function taskGroupRow(group, values) {
  const block = document.createElement('div');
  block.className = 'task-group';
  block.dataset.group = group;

  const head = document.createElement('div');
  head.className = 'task-group-head';
  const label = document.createElement('strong');
  label.textContent = group;
  const reset = document.createElement('button');
  reset.type = 'button';
  reset.textContent = '↺';
  reset.title = "Revert this group's fields to what's saved on disk";
  reset.onclick = () => setTaskGroupFields(block, values);
  head.append(label, reset);
  block.append(head, taskBindingNote(values));

  // Predicate placement recomputes X and Y at reset (after the randomization)
  // but preserves Z and yaw, so only those two fields go dead.
  const placed = values.predicate_placed === true;
  const why = 'X/Y set by predicate placement at reset';
  const xyz = Array.isArray(values.xyz) ? values.xyz : [null, null, null];
  const field = document.createElement('div');
  field.className = 'field';
  field.append(
    taskNumField('±X m', 'tg-x', xyz[0], { disabled: placed, title: placed ? why : '' }),
    taskNumField('±Y m', 'tg-y', xyz[1], { disabled: placed, title: placed ? why : '' }),
    taskNumField('±Z m', 'tg-z', xyz[2]),
  );
  const rotation = document.createElement('div');
  rotation.className = 'field rot';
  rotation.appendChild(taskNumField('±Z rot °', 'tg-zrot', taskDegrees(values.z_rot)));
  rememberFileZRot(rotation.querySelector('.tg-zrot'), values.z_rot);
  block.append(field, rotation);

  if (placed) {
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = `${why}. Z and yaw still apply.`;
    block.appendChild(note);
  }

  const bounds = document.createElement('div');
  bounds.className = 'note warn tg-bounds';
  bounds.hidden = true;
  block.appendChild(bounds);
  showWorkspaceNote(block);
  return block;
}

function renderTaskGroups() {
  const container = document.getElementById('task-groups');
  container.innerHTML = '';
  const names = Object.keys(taskGroups()).sort();
  if (!names.length) {
    const div = document.createElement('div');
    div.className = 'hint';
    // Three different empty states: no config attached, an unreadable file,
    // or a config that names no groups.
    if (!activeTaskPath) {
      div.textContent = 'No task config attached to this scene — pick one above.';
    } else if (!taskCfg) {
      div.textContent = `${taskFileName()} could not be read, so there are no `
        + 'ranges to show. The line above says why.';
    } else {
      div.textContent = 'This task config names no groups.';
    }
    container.appendChild(div);
    document.getElementById('m-task-ranges').disabled = true;
    refreshTaskRangeBoxes();
    return;
  }
  for (const group of names) container.appendChild(taskGroupRow(group, taskGroups()[group]));
  document.getElementById('m-task-ranges').disabled = false;
  refreshTaskRangeBoxes();
}

async function refreshTaskSection() {
  const note = document.getElementById('task-note');
  const spread = document.getElementById('task-spread');
  if (!activeTaskPath) {
    taskCfg = null;
    // renderTaskGroups says what is missing.
    note.textContent = '';
    spread.hidden = true;
    renderTaskGroups();
    setTaskDirty(false);
    return;
  }
  note.textContent = 'Loading…';
  try {
    const res = await fetch(`/api/task_cfg?path=${encodeURIComponent(activeTaskPath)}`,
      { cache: 'no-store' });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    taskCfg = body;
    // Warn up front when a save would drop the file's comments.
    note.textContent = taskFileName()
      + (body.round_trip === false ? ' — no ruamel.yaml here, so a save drops its comments' : '');
  } catch (err) {
    taskCfg = null;
    note.textContent = `Could not load ${taskFileName()}: ${err.message}`;
  }
  spread.hidden = !taskCfg;
  renderTaskGroups();
  setTaskDirty(false);
}

/**
 * One group's fields in the shape /api/task_cfg takes, or null when a field
 * is not a number. A blank posts null, which deletes the key ("not
 * randomized" vs "randomized by zero"); X/Y/Z are one 3-vector, so any typed
 * value makes the other blanks explicit zeros. An untouched yaw posts back
 * the file's radians rather than re-converting the displayed degrees — see
 * `rememberFileZRot`.
 */
function readTaskGroupSpec(block) {
  const typed = ['.tg-x', '.tg-y', '.tg-z'].map((sel) => block.querySelector(sel).value.trim());
  const xyz = typed.every((v) => v === '') ? null : typed.map((v) => (v === '' ? 0 : parseFloat(v)));
  const zrotField = block.querySelector('.tg-zrot');
  const deg = zrotField.value.trim();
  const shown = zrotField.dataset.fileShown ?? '';
  const fileRad = zrotField.dataset.fileRad ?? '';
  // Compared as a number so retyping the same reading still counts as
  // untouched; blank matches blank.
  const asFile = deg === ''
    ? shown === ''
    : (shown !== '' && parseFloat(deg) === parseFloat(shown));
  const zRot = asFile
    ? (fileRad === '' ? null : Number(fileRad))
    : (deg === '' ? null : parseFloat(deg) * Math.PI / 180);
  if ((xyz && !xyz.every(Number.isFinite)) || (zRot !== null && !Number.isFinite(zRot))) return null;
  return { xyz, z_rot: zRot };
}

// --- the reset-range overlay ---------------------------------------------------
// Draws, for every object a group binds, the ±X/±Y/±Z box a task reset may
// move its origin into, centred on the object. Offsets are sampled per world
// axis, so the box is axis-aligned. Ranges are read live off the panel's
// fields; a predicate-placed group's X and Y are dropped, and the ±Z rotation
// range is not drawn. Everything sits on the overlay layer, hidden from the
// sensor previews.
let taskRangesOn = false; // a view setting; not persisted, same as m-boxes
const taskRangeHelpers = new Map(); // "group name" -> { node, colour, sig }
let taskRangeSpecs = []; // rebuilt on panel changes, followed per frame

// One colour per group, off the camera palette.
function taskRangeColour(group) {
  const names = Object.keys(taskGroups()).sort();
  return CAMERA_COLOURS[Math.max(0, names.indexOf(group)) % CAMERA_COLOURS.length];
}

/** Re-read the panel's live ranges into specs. Cheap; runs on every edit. */
function refreshTaskRangeBoxes() {
  taskRangeSpecs = [];
  const groups = taskGroups();
  for (const block of document.querySelectorAll('#task-groups .task-group')) {
    const group = block.dataset.group;
    const values = groups[group];
    const spec = values && readTaskGroupSpec(block);
    if (!spec || !spec.xyz) continue;
    const xyz = values.predicate_placed === true ? [0, 0, spec.xyz[2]] : spec.xyz;
    const half = xyz.map((v) => Math.max(0, v));
    if (!half.some((v) => v > 0)) continue;
    for (const name of values.objects || []) {
      taskRangeSpecs.push({
        key: `${group} ${name}`, name, half, colour: taskRangeColour(group),
      });
    }
  }
  updateTaskRangeBoxes();
}

function removeTaskRangeHelper(key) {
  const entry = taskRangeHelpers.get(key);
  if (entry) scene.remove(entry.node);
  taskRangeHelpers.delete(key);
}

// Translucent fill + crisp edge in the group's colour. Built at unit size and
// scaled to the range, so an edit is a scale write, never a geometry rebuild.
function makeTaskRangeNode(colour) {
  const box = new THREE.BoxGeometry(1, 1, 1);
  const fill = new THREE.Mesh(box, new THREE.MeshBasicMaterial({
    color: colour, transparent: true, opacity: 0.09,
    side: THREE.DoubleSide, depthWrite: false,
  }));
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(box),
    new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.55 }));
  const node = new THREE.Group();
  node.add(markAsOverlay(fill), markAsOverlay(edge));
  return node;
}

/** Called every frame; a no-op while off. Follows the objects, reaps leavers. */
function updateTaskRangeBoxes() {
  if (!taskRangesOn) {
    for (const key of [...taskRangeHelpers.keys()]) removeTaskRangeHelper(key);
    return;
  }
  const seen = new Set();
  for (const spec of taskRangeSpecs) {
    const rec = objects.get(spec.name);
    if (!rec || rec.present === false || rec.hiddenForView) continue;
    seen.add(spec.key);
    let entry = taskRangeHelpers.get(spec.key);
    if (entry && entry.colour !== spec.colour) {
      removeTaskRangeHelper(spec.key);
      entry = null;
    }
    if (!entry) {
      entry = {
        node: makeTaskRangeNode(spec.colour), colour: spec.colour,
        sig: new Float64Array(6).fill(NaN), // NaN never equals, forces first write
      };
      scene.add(entry.node);
      taskRangeHelpers.set(spec.key, entry);
    }
    const p = rec.group.position;
    const sig = entry.sig;
    if (sig[0] === p.x && sig[1] === p.y && sig[2] === p.z
        && sig[3] === spec.half[0] && sig[4] === spec.half[1] && sig[5] === spec.half[2]) {
      continue;
    }
    sig[0] = p.x; sig[1] = p.y; sig[2] = p.z;
    [sig[3], sig[4], sig[5]] = spec.half;
    entry.node.position.copy(p);
    // A zero axis draws flat rather than vanishing the whole box; 1e-4 is the
    // repo's "no measurable extent" floor.
    entry.node.scale.set(Math.max(spec.half[0] * 2, 1e-4),
                         Math.max(spec.half[1] * 2, 1e-4),
                         Math.max(spec.half[2] * 2, 1e-4));
  }
  for (const key of [...taskRangeHelpers.keys()]) {
    if (!seen.has(key)) removeTaskRangeHelper(key);
  }
}

function setTaskRangesVisible(on) {
  taskRangesOn = on;
  document.getElementById('m-task-ranges').classList.toggle('on', taskRangesOn);
  if (taskRangesOn) refreshTaskRangeBoxes(); else updateTaskRangeBoxes();
}

document.getElementById('m-task-ranges').onclick = () => {
  setTaskRangesVisible(!taskRangesOn);
  setStatus(taskRangesOn
    ? 'Reset ranges shown: each box is where a task reset may put that object\'s origin.'
    : 'Reset ranges hidden.');
};

/**
 * Write the ranges on screen back into the task yaml; returns whether they
 * landed. Only the groups the panel shows are sent — the server leaves an
 * absent group alone.
 */
async function saveTaskRanges({ acceptCommentLoss = false } = {}) {
  if (!activeTaskPath || !taskCfg) return false;
  const groups = {};
  for (const block of document.querySelectorAll('#task-groups .task-group')) {
    const spec = readTaskGroupSpec(block);
    if (!spec) {
      setStatus(`${block.dataset.group}: every range must be a number, or blank.`, 'err');
      return false;
    }
    groups[block.dataset.group] = spec;
  }

  taskSaving = true;
  updateTaskButtons();
  try {
    const res = await fetch('/api/task_cfg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({
        path: activeTaskPath,
        // The hash the GET reported; the server refuses a stale one.
        sha256: taskCfg.sha256,
        groups,
        ...(acceptCommentLoss ? { accept_comment_loss: true } : {}),
      }),
    });
    const body = await res.json();

    if (res.status === 409 && body.reason === 'stale') {
      // Something else wrote the file since this panel read it; merging is
      // not possible from here.
      const answer = await confirmDialog({
        title: 'The file changed on disk',
        lines: [
          `${taskFileName()} has been written by something else since this panel read it.`,
          'Saving now would overwrite that. Reloading shows what is on disk and drops '
          + 'the ranges typed here.',
        ],
        actions: [
          { label: 'Reload the panel', value: 'reload', primary: true },
          { label: 'Keep editing', value: 'keep' },
        ],
      });
      if (answer === 'reload') await refreshTaskSection();
      setStatus(`Not saved — ${taskFileName()} changed on disk.`, 'err');
      return false;
    }
    if (res.status === 409 && body.reason === 'comment_loss') {
      const answer = await confirmDialog({
        title: 'Comments would be lost',
        lines: [
          `The server has no ruamel.yaml, so writing ${taskFileName()} means dumping the `
          + 'whole file back through PyYAML.',
          'The values survive that. Its comments, key order and formatting do not.',
          'Installing ruamel.yaml (requirements.txt) and restarting the server avoids it.',
        ],
        actions: [
          { label: 'Save anyway', value: 'save' },
          { label: 'Cancel', value: 'cancel', primary: true },
        ],
      });
      if (answer !== 'save') { setStatus('Ranges not saved.'); return false; }
      return await saveTaskRanges({ acceptCommentLoss: true });
    }
    if (!res.ok) throw new Error(body.error || res.statusText);

    setStatus(body.note
      ? `Saved ranges to ${taskFileName()} — ${body.note}.`
      : `Saved ranges to ${taskFileName()}.`,
    body.note ? 'err' : 'ok');
    // Re-read for the new sha256, and so a deleted key shows blank.
    await refreshTaskSection();
    return true;
  } catch (err) {
    setStatus(`Could not save ranges: ${err.message}`, 'err');
    return false;
  } finally {
    taskSaving = false;
    updateTaskButtons();
  }
}

document.getElementById('btn-task-save').onclick = () => {
  if (taskSaving) return;
  saveTaskRanges();
};

document.getElementById('btn-task-revert').onclick = () => {
  renderTaskGroups();
  setTaskDirty(false);
};

// --- generate a task from a prompt ------------------------------------------
// Sends the live object list and a viewport screenshot to the server
// (task_propose.py), which asks Gemini for one task yaml. Nothing reaches
// disk until Save.

let taskGenDir = null;

/**
 * Copy *source* onto a canvas whose long edge is at most *longEdge* pixels;
 * returns the source unchanged when it is already small enough.
 */
function downscaledCanvas(source, longEdge) {
  const scale = longEdge / Math.max(source.width, source.height);
  if (!(scale < 1)) return source;
  const shrunk = document.createElement('canvas');
  shrunk.width = Math.round(source.width * scale);
  shrunk.height = Math.round(source.height * scale);
  shrunk.getContext('2d').drawImage(source, 0, 0, shrunk.width, shrunk.height);
  return shrunk;
}

document.getElementById('btn-task-gen').onclick = async () => {
  const promptEl = document.getElementById('task-gen-prompt');
  const prompt = promptEl.value.trim();
  const note = document.getElementById('task-gen-note');
  if (!prompt) {
    note.textContent = 'Type a description of the task first.';
    return;
  }

  const objects = liveRecords()
    .filter((r) => !r.isCamera && r.entry.editable)
    .map((r) => ({ name: r.name, category: r.entry.category }));

  // Render synchronously right before the read-back — the WebGL draw buffer
  // may not survive past the frame that produced it — and downscale, because
  // the buffer is in device pixels and a full-size PNG can exceed the
  // server's request-size limit.
  renderer.render(scene, camera);
  const image = downscaledCanvas(renderer.domElement, 1600).toDataURL('image/png');

  const button = document.getElementById('btn-task-gen');
  button.disabled = true;
  note.textContent = 'Asking Gemini…';
  document.getElementById('task-gen-result').hidden = true;
  try {
    const res = await fetch('/api/task_propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({
        prompt, objects, image,
        robot_type: document.getElementById('task-gen-robot').value.trim() || 'franka',
      }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);

    // A refusal is an answer, not an error: report it and keep the result block hidden.
    if (body.refused) {
      document.getElementById('task-gen-result').hidden = true;
      note.textContent = `No task generated — ${body.reason}`;
      setStatus('That task cannot be built from this scene.', 'err');
      return;
    }

    document.getElementById('task-gen-yaml').textContent = body.yaml_text;
    document.getElementById('task-gen-filename').value = body.filename;
    taskGenDir = body.default_dir;
    document.getElementById('task-gen-dir').textContent = taskGenDir;
    document.getElementById('task-gen-result').hidden = false;
    // Surface the server's validation problems now, while the yaml is on
    // screen to be edited.
    const blocking = (body.problems || []).filter((p) => p.severity === 'breaks');
    note.textContent = (body.problems || []).length
      ? `Proposed "${body.task_name}" — ${describeProblems(body.problems)}`
      : `Proposed "${body.task_name}" — review before saving.`;
    setStatus(blocking.length
      ? `"${body.task_name}" needs fixing before it can run.`
      : `Proposed "${body.task_name}".`, blocking.length ? 'err' : 'ok');
  } catch (err) {
    note.textContent = `Could not generate a task: ${err.message}`;
  } finally {
    button.disabled = false;
  }
};

document.getElementById('btn-task-gen-folder').onclick = () => {
  openPicker({
    // The server writes `<dir>/<filename>`, so only a directory is valid here.
    mode: 'directory',
    title: 'Choose the folder to write the task config into',
    onChoose: (dir) => {
      taskGenDir = dir;
      document.getElementById('task-gen-dir').textContent = dir;
    },
  });
};

/**
 * Ask the server to write the generated yaml. `overwriteSha` is the digest of
 * the on-disk bytes the user confirmed overwriting; omitted on the first
 * attempt, which is how the server tells "create" from "replace".
 */
async function saveGeneratedTask(overwriteSha, { allowInvalid = false } = {}) {
  const filename = document.getElementById('task-gen-filename').value.trim();
  const yamlText = document.getElementById('task-gen-yaml').textContent;
  const res = await fetch('/api/task_create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
    body: JSON.stringify({
      dir: taskGenDir,
      filename,
      yaml_text: yamlText,
      ...(allowInvalid ? { allow_invalid: true } : {}),
      ...(overwriteSha ? { overwrite_sha256: overwriteSha } : {}),
    }),
  });
  const body = await res.json();
  return { ok: res.ok, status: res.status, body };
}

/**
 * Write the generated yaml, and optionally point this scene's TASK panel at
 * it. Attaching repoints the panel and drops its unsaved range edits, so it
 * goes through the guarded `chooseTaskForScene` path, and the ranges are
 * settled before anything is written.
 */
async function writeGeneratedTask({ attach }) {
  const note = document.getElementById('task-gen-note');
  if (attach && await guardTaskRanges('Attaching a generated task config') === 'cancel') {
    note.textContent = 'Not saved — the ranges you have open were left alone.';
    return;
  }
  let allowInvalid = false;
  let { ok, status, body } = await saveGeneratedTask(null);
  // 422 means the config would not run; offer to save it as a draft anyway.
  // Asked once — the answer carries through the overwrite loop below.
  if (!ok && status === 422) {
    const answer = await confirmDialog({
      title: 'This task would not run',
      lines: [
        body.error,
        ...(body.problems || []).slice(0, 4).map((p) => `• ${p.message}`),
        'Save it as a draft to fix by hand, or cancel and generate again?',
      ],
      actions: [
        { label: 'Cancel', value: 'cancel', primary: true },
        { label: 'Save as draft', value: 'draft' },
      ],
    });
    if (answer !== 'draft') {
      note.textContent = `Not saved — ${describeProblems(body.problems)}`;
      return;
    }
    allowInvalid = true;
    ({ ok, status, body } = await saveGeneratedTask(null, { allowInvalid }));
  }
  // A conflict can repeat: if the file changes while the dialog is open, the
  // server refuses again with the new digest. Bounded rather than raced.
  for (let attempt = 0; !ok && status === 409 && attempt < 3; attempt++) {
    const changed = body.reason === 'stale';
    const answer = await confirmDialog({
      title: changed ? 'That file changed while you were deciding' : 'File already exists',
      lines: [
        body.error,
        changed
          ? 'Nothing has been written. Overwrite what is on disk now?'
          : 'Overwrite it?',
      ],
      actions: [
        { label: 'Cancel', value: 'cancel', primary: true },
        { label: changed ? 'Overwrite anyway' : 'Overwrite', value: 'overwrite' },
      ],
    });
    if (answer !== 'overwrite') {
      note.textContent = 'Not saved — the file on disk was left alone.';
      return;
    }
    if (!body.sha256) {
      // No digest means the server could not read the file — nothing to confirm against.
      note.textContent = `Could not save: ${body.error || status}`;
      return;
    }
    ({ ok, status, body } = await saveGeneratedTask(body.sha256, { allowInvalid }));
  }
  if (!ok) {
    note.textContent = `Could not save: ${body.error || status}`;
    return;
  }
  // The server checks the config's object names against the open scene as it
  // writes; surface those warnings. They arrive as strings or as objects with
  // the text under `message` or `detail`.
  const warnings = (body.warnings || []).map(
    (w) => (typeof w === 'string' ? w : (w.message ?? w.detail ?? String(w.code ?? ''))));
  // Say outright whether this saved as a runnable task or a draft.
  const kind = body.runnable === false ? 'Saved as a draft to' : 'Saved';
  note.textContent = warnings.length
    ? `${kind} ${body.path}. ${warnings.length} thing(s) to fix: `
      + `${warnings.join('; ')}.`
    : `${kind} ${body.path}.`;
  setStatus(`${kind} ${body.path}.`, warnings.length ? 'err' : 'ok');
  if (!attach) {
    note.textContent += ' Not attached — the TASK panel is still on '
      + `${taskFileName() || 'no task'}.`;
    return;
  }
  // The guarded path; the ranges were settled above, so its guard is a no-op here.
  if (!await chooseTaskForScene(activeManifest.scene_json, body.path)) {
    note.textContent += ' Saved, but left unattached.';
    return;
  }
  // Refresh so the TASK dropdown shows the newly attached file.
  await refreshTaskPick();
}

document.getElementById('btn-task-gen-save').onclick = () => writeGeneratedTask({ attach: false });
document.getElementById('btn-task-gen-attach').onclick = () => writeGeneratedTask({ attach: true });

function renderLauncher() {
  const list = document.getElementById('launcher-list');
  list.innerHTML = '';
  if (!catalogue) { list.textContent = 'Loading…'; return; }

  const needle = launcherFilter.trim().toLowerCase();
  const matches = (row) => !needle
    || row.name.toLowerCase().includes(needle)
    || (row.background || '').toLowerCase().includes(needle);

  const seen = new Set();
  const group = (title, rows) => {
    const visible = rows.filter(matches);
    if (visible.length === 0) return;
    const heading = document.createElement('div');
    heading.className = 'scene-group';
    heading.textContent = `${title} (${visible.length})`;
    list.appendChild(heading);
    for (const row of visible) {
      seen.add(row.path);
      list.appendChild(sceneRow(row, { current: row.path === catalogue.current }));
      if (!expandedHistory.has(row.dir)) continue;
      for (const variant of row.variants || []) {
        if (variant.path === row.path) continue;
        const item = document.createElement('div');
        item.className = 'variant' + (variant.canonical ? ' canonical' : '');
        item.title = variant.path;
        const label = document.createElement('span');
        label.textContent = variant.label;
        // Armed and confirmed like the scene rows above.
        const go = document.createElement('button');
        go.className = 'switch primary';
        go.textContent = 'Switch scene';
        go.title = `Open ${row.name} (${variant.label}). This reloads the page.`;
        go.onclick = (event) => {
          event.stopPropagation();
          openScene(variant.path, { label: `${row.name} (${variant.label})` });
        };
        item.append(label, go);
        item.onclick = () => armSceneRow(item);
        list.appendChild(item);
      }
    }
  };

  const current = catalogue.scenes.find((s) => s.path === catalogue.current);
  if (current && matches(current)) group('Editing now', [current]);
  group('Recent', (catalogue.recent || []).filter((r) => !seen.has(r.path)));
  // Group by where the scene came from: generated, shipped, or elsewhere.
  const rest = (catalogue.scenes || []).filter((r) => !seen.has(r.path));
  group('Generated from your videos', rest.filter((r) => r.source === 'generated'));
  group('SimFoundry scenes', rest.filter((r) => r.source === 'preset'));
  // Scenes reached via --scene-root or beside --scene, claimed by neither known root.
  group('Elsewhere on this machine', rest.filter((r) => !r.source));

  if (list.children.length === 0) {
    list.textContent = needle ? `Nothing matches “${needle}”.` : 'No scenes found.';
  }
  document.getElementById('launcher-roots').textContent =
    (catalogue.roots || []).join('  ·  ');
}

async function openLauncher() {
  if (launcherIsOpen()) { closeLauncher(); return; }
  document.getElementById('launcher-modal').hidden = false;
  document.getElementById('launcher-search').focus();
  document.getElementById('launcher-note').textContent = 'Loading…';
  renderLauncher();
  try {
    await loadCatalogue();
    renderLauncher();
    const total = (catalogue.scenes || []).length;
    document.getElementById('launcher-note').textContent =
      `${total} scene(s). Opening one reloads this page.`;
  } catch (err) {
    document.getElementById('launcher-note').textContent = `Could not list scenes: ${err.message}`;
  }
}

function closeLauncher() {
  document.getElementById('launcher-modal').hidden = true;
}

document.getElementById('btn-launcher').onclick = openLauncher;
document.getElementById('btn-launcher-close').onclick = closeLauncher;
document.getElementById('launcher-backdrop').onclick = closeLauncher;
document.getElementById('launcher-search').addEventListener('input', (e) => {
  launcherFilter = e.target.value;
  renderLauncher();
});

// --- new scene from a template ---------------------------------------------
// A new scene inherits its structure (versions block, room, robot, ground
// plane) from an existing scene known to load; this dialog chooses the contents.

let composeTemplate = null;        // last /api/template payload
const composeKeep = new Set();

async function openCompose() {
  if (!catalogue) {
    try { await loadCatalogue(); } catch (err) {
      setStatus(`Could not list scenes: ${err.message}`, 'err');
      return;
    }
  }
  const select = document.getElementById('compose-template');
  select.innerHTML = '';
  for (const row of catalogue.scenes || []) {
    if (row.error) continue;
    const option = document.createElement('option');
    option.value = row.path;
    option.textContent = `${row.name} — ${sceneMeta(row)}`;
    select.appendChild(option);
  }
  if (select.options.length === 0) {
    setStatus('No readable scene to use as a template.', 'err');
    return;
  }
  select.value = catalogue.current && [...select.options].some((o) => o.value === catalogue.current)
    ? catalogue.current
    : select.options[0].value;
  document.getElementById('compose-modal').hidden = false;
  document.getElementById('compose-name').value = '';
  await loadComposeTemplate();
  refreshComposeSummary();
  document.getElementById('compose-name').focus();
}

function closeCompose() {
  document.getElementById('compose-modal').hidden = true;
}

async function loadComposeTemplate() {
  const path = document.getElementById('compose-template').value;
  const objects = document.getElementById('compose-objects');
  objects.textContent = 'Loading…';
  composeTemplate = null;
  composeKeep.clear();
  try {
    const res = await fetch('/api/template', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({ path }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    composeTemplate = body;
  } catch (err) {
    objects.textContent = `Could not read that template: ${err.message}`;
    document.getElementById('compose-inherits').textContent = '';
    return;
  }
  const inherited = [
    composeTemplate.background ? `room ${composeTemplate.background}` : 'no room',
    composeTemplate.robot ? `robot ${composeTemplate.robot}` : 'no robot',
    composeTemplate.has_versions ? 'version block' : 'no version block',
    composeTemplate.has_ground_plane ? 'ground plane' : 'no ground plane',
  ];
  document.getElementById('compose-inherits').textContent = `Inherits: ${inherited.join(' · ')}`;
  renderComposeObjects();
}

function renderComposeObjects() {
  const host = document.getElementById('compose-objects');
  host.innerHTML = '';
  if (!composeTemplate) return;
  for (const object of composeTemplate.objects) {
    const row = document.createElement('label');
    row.className = 'keep' + (object.editable ? '' : ' locked');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = object.editable ? composeKeep.has(object.name) : true;
    box.disabled = !object.editable || !object.resolvable;
    box.onchange = () => {
      if (box.checked) composeKeep.add(object.name);
      else composeKeep.delete(object.name);
      refreshComposeSummary();
    };
    const name = document.createElement('span');
    name.textContent = `${object.name}`;
    const tag = document.createElement('span');
    tag.className = 'tag';
    if (!object.resolvable) {
      tag.textContent = `${object.category} · asset missing`;
      row.title = 'The USD this object points at is not on this machine, so it '
        + 'cannot be carried over.';
    } else {
      tag.textContent = object.editable ? object.category : `${object.kind} · always included`;
    }
    row.append(box, name, tag);
    host.appendChild(row);
  }
}

/**
 * Repaint the name check, the destination line, the summary note and the button.
 *
 * @param {Object} [opts] Options.
 * @param {boolean} [opts.keepNote] Leave `#compose-note` as it is, so a
 *   failed create's error message survives this repaint.
 */
function refreshComposeSummary({ keepNote = false } = {}) {
  const name = document.getElementById('compose-name').value.trim();
  const field = document.getElementById('compose-name');
  const where = document.getElementById('compose-where');
  const note = document.getElementById('compose-note');
  const create = document.getElementById('btn-compose-create');

  // The same name rule the server enforces, checked while typing.
  const valid = /^[a-z][a-z0-9_]{1,62}[a-z0-9]$/.test(name);
  const taken = (catalogue?.scenes || []).some((s) => s.name === name);
  field.classList.toggle('bad', Boolean(name) && (!valid || taken));

  const root = catalogue?.compose_root || '';
  where.textContent = name && valid
    ? `Creates ${root}/${name}/${name}_scene_state_latest.json`
    : 'Lower-case letters, digits and underscores; must start with a letter.';

  const kept = composeKeep.size;
  if (!keepNote) {
    if (!name) note.textContent = 'Name it to continue.';
    else if (!valid) note.textContent = 'That name will not work as a directory.';
    else if (taken) note.textContent = `There is already a scene called ${name}.`;
    else note.textContent = kept ? `${kept} prop(s) come along.` : 'Room and robot only.';
  }
  create.disabled = !name || !valid || taken || !composeTemplate;
}

async function createComposedScene() {
  const name = document.getElementById('compose-name').value.trim();
  const template = document.getElementById('compose-template').value;
  if (await guardUnsaved(`Creating ${name}`) === 'cancel') return;

  const note = document.getElementById('compose-note');
  note.textContent = 'Creating…';
  document.getElementById('btn-compose-create').disabled = true;
  try {
    const res = await fetch('/api/compose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({
        template, name, keep: [...composeKeep], discard_pending: true,
      }),
    });
    const body = await res.json();
    if (!res.ok) {
      note.textContent = body.error || res.statusText;
      refreshComposeSummary({ keepNote: true });
      return;
    }
    switching = true;
    setStatus(`Created ${body.name}. Reloading…`);
    location.reload();
  } catch (err) {
    note.textContent = `Could not create it: ${err.message}`;
    refreshComposeSummary({ keepNote: true });
  }
}

document.getElementById('btn-compose-open').onclick = openCompose;
document.getElementById('btn-compose-close').onclick = closeCompose;
document.getElementById('compose-backdrop').onclick = closeCompose;
document.getElementById('compose-template').onchange = async () => {
  await loadComposeTemplate();
  refreshComposeSummary();
};
// Wrapped: the listener's InputEvent must not land in the options parameter.
document.getElementById('compose-name').addEventListener('input', () => refreshComposeSummary());
document.getElementById('compose-name').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !document.getElementById('btn-compose-create').disabled) {
    e.preventDefault();
    createComposedScene();
  }
});
document.getElementById('btn-compose-create').onclick = createComposedScene;
document.getElementById('btn-compose-none').onclick = () => {
  composeKeep.clear();
  renderComposeObjects();
  refreshComposeSummary();
};
document.getElementById('btn-compose-all').onclick = () => {
  for (const object of composeTemplate?.objects || []) {
    if (object.editable && object.resolvable) composeKeep.add(object.name);
  }
  renderComposeObjects();
  refreshComposeSummary();
};

// SETTLE_TIMEOUT_S bounds the settle subprocess; anything past it plus a
// margin means the server itself is wedged, so stop waiting.
const SETTLE_GIVE_UP_MS = 16 * 60 * 1000;

// Per-object drift from the last settle, shown against the object rows.
let settleDrift = new Map();
// Joint drift and unverifiable joints, keyed the same way as settleDrift.
let settleJointDrift = new Map();
let settleUnverified = new Map();

function showSettleDrift(status) {
  settleDrift = new Map((status.moved || []).map((m) => [m.name, m.delta]));
  // A joint-only failure moves no root, so it needs its own map.
  settleJointDrift = new Map((status.joints_moved || []).map(
    (entry) => [entry.name, (entry.joints || []).map(
      (d) => `${d.joint} drifted ${d.delta}${d.unit === 'deg' ? '°' : ' m'}`).join(', ')]));
  settleUnverified = new Map((status.joints_unchecked || []).map(
    (entry) => [entry.name, entry.reason]));
  renderList();
}

window.addEventListener('beforeunload', (event) => {
  // A scene switch reloads deliberately and has already asked about unsaved edits.
  if (switching) return;
  if (!hasUnsavedSceneEdits() && cameraDirty.size === 0 && !taskDirty) return;
  event.preventDefault();
  event.returnValue = '';
});

// Canvas size in CSS pixels (the drawing buffer itself is scaled by the device
// pixel ratio); camera view letterboxes it to the sensor's aspect.
let canvasWidth = 0;
let canvasHeight = 0;

function setCanvasSize(width, height) {
  if (Math.abs(width - canvasWidth) < 0.5 && Math.abs(height - canvasHeight) < 0.5) return;
  canvasWidth = width;
  canvasHeight = height;
  renderer.setSize(width, height);
}

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  // Before the flying early-return: a placed gizmo bar must be pulled back
  // inside a viewport that just shrank.
  applyGizmoPlacement();
  if (flying) return;      // the render loop re-letterboxes on the next frame
  renderer.domElement.style.margin = '';
  setCanvasSize(w, h);
}
window.addEventListener('resize', resize);
resize();

function renderFlyView() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  // Letterbox to the sensor's real aspect so the true framing is shown.
  const aspect = flying.proj.aspect;
  const fitWidth = Math.min(w, h * aspect);
  const fitHeight = fitWidth / aspect;
  setCanvasSize(fitWidth, fitHeight);
  renderer.domElement.style.margin = `${(h - fitHeight) / 2}px ${(w - fitWidth) / 2}px`;
  renderer.render(scene, flying.proj);
  if (overviewOn) drawOverviewInset(fitWidth, fitHeight);
}

// Third-person inset in the corner, showing where the flown camera sits in the room.
function drawOverviewInset(fitWidth, fitHeight) {
  const width = Math.max(120, Math.round(fitWidth * 0.28));
  const height = Math.round(width * 9 / 16);
  const pad = Math.max(6, Math.round(fitWidth * 0.015));
  const x = Math.round(fitWidth - width - pad);
  const y = pad;

  updateOverviewCam();
  // The camera being flown is hidden from the main pass; force it visible here.
  showCameraProxy(flying, true, { force: true });
  flying.helper.update();

  renderer.autoClear = false;
  renderer.setScissorTest(true);
  renderer.setScissor(x - 1, y - 1, width + 2, height + 2);
  renderer.setClearColor(0x76b900, 1);          // a one-pixel frame around it
  renderer.clear(true, false, false);
  renderer.setScissor(x, y, width, height);
  renderer.setViewport(x, y, width, height);
  renderer.setClearColor(0x14161a, 1);
  renderer.clear(true, true, false);
  renderer.render(scene, overviewCam);

  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, fitWidth, fitHeight);
  renderer.autoClear = true;
  showCameraProxy(flying, false);
}

// --- exterior camera previews ----------------------------------------------
// Live picture-in-picture previews of what each exterior sensor sees. The
// picture is drawn into the page's single canvas through a scissor rectangle;
// the DOM supplies the frame around it (caption, guides, drag/resize handles).
// Both halves derive from one geometry record so they stay in step.

const pipHost = document.getElementById('pips');
const pipPanes = new Map();          // camera name -> pane
let pipOrder = [];                   // back-to-front; the last one is on top
let pipOn = true;
let guidesOn = true;

// Chrome sizes in CSS pixels, duplicated from the stylesheet: the GL rectangle
// must land exactly on the hole in the DOM without measuring it every frame.
const PIP_BORDER = 1;
const PIP_HEAD = 18;
const PIP_FOOT = 15;
const PIP_PAD = 10;                  // margin from the viewport edge
const PIP_GAP = 8;
const PIP_MIN_W = 120;
const PIP_MAX_W = 640;
const PIP_DRAG_SLOP = 4;             // px of movement that stops a press being a click
const SVG_NS = 'http://www.w3.org/2000/svg';

const finite = (value, fallback) => (Number.isFinite(value) ? value : fallback);

// Height follows width: the pane keeps the sensor's real aspect.
const pipBodyHeight = (pane) => Math.max(1, Math.round(pane.w / pane.rec.proj.aspect));

function pipOuterSize(pane) {
  const chrome = PIP_BORDER * 2 + PIP_HEAD;
  return {
    w: pane.w + PIP_BORDER * 2,
    h: pane.minimized ? chrome : chrome + pipBodyHeight(pane) + PIP_FOOT,
  };
}

/**
 * The highest a pane may start: clear of the strip along the top of the
 * viewport, or the ordinary margin when there is no strip. A floor, not just
 * the default layout's `y`, because saved layouts are re-clamped against it.
 */
function pipTopFloor() {
  const strip = document.getElementById('viewport-top');
  const height = strip ? strip.offsetHeight : 0;
  return PIP_PAD + (height ? height + PIP_GAP : 0);
}

// Keep panes whole and inside the viewport: the GL scissor is clipped to the
// canvas, so an off-edge pane would show a cropped picture in a full-size frame.
function clampPane(pane) {
  const vw = viewport.clientWidth;
  const vh = viewport.clientHeight;
  // want* is where the pane was put; x/y/w are where it currently fits. Only
  // the wanted values are remembered, so a narrow window views the layout
  // rather than editing it.
  if (pane.wantX === undefined) { pane.wantX = pane.x; pane.wantY = pane.y; }
  if (pane.wantW === undefined) pane.wantW = pane.w;
  const maxW = Math.max(PIP_MIN_W, Math.min(PIP_MAX_W, vw - PIP_PAD * 2));
  pane.w = Math.round(THREE.MathUtils.clamp(pane.wantW, PIP_MIN_W, maxW));
  const { w, h } = pipOuterSize(pane);
  // The floor gives way on a viewport too short to honour it.
  const floor = Math.min(pipTopFloor(), Math.max(0, vh - h));
  pane.x = Math.round(THREE.MathUtils.clamp(pane.wantX, 0, Math.max(0, vw - w)));
  pane.y = Math.round(THREE.MathUtils.clamp(pane.wantY, floor, Math.max(floor, vh - h)));
}

const _latPos = new THREE.Vector3();
const _latQuat = new THREE.Quaternion();

/**
 * How far to the robot's left *rec* stands — bigger is further left (+Y is
 * the robot's left in OmniGibson), measured in the robot's own frame; with no
 * robot, world Y. The single place that decides which sensor is "left".
 */
function lateralOffset(rec, robot) {
  rec.group.getWorldPosition(_latPos);
  if (robot) {
    _latPos.sub(robot.group.position).applyQuaternion(
      _latQuat.copy(robot.group.quaternion).invert());
  }
  return _latPos.y;
}

// Cameras ordered left-to-right in the robot's frame, not config order, so the
// left-hand pane really is the left-hand camera.
function previewCameras() {
  const cams = [...objects.values()].filter((r) => r.isCamera);
  if (cams.length < 2) return cams;
  const robot = liveRecords().find((r) => r.entry.kind === 'robot');
  return cams
    .map((rec, index) => ({ rec, index, lateral: lateralOffset(rec, robot) }))
    .sort((a, b) => (b.lateral - a.lateral) || (a.index - b.index))
    .map((entry) => entry.rec);
}

function defaultPipLayout() {
  const order = previewCameras();
  const available = Math.max(PIP_MIN_W, viewport.clientWidth - PIP_PAD * 2);
  // Rig sensors take the row along the top; the robot's own (read-only)
  // cameras stack below the left-hand one.
  const inRow = order.filter((rec) => !rec.readOnly);
  const below = order.filter((rec) => rec.readOnly);
  // Prefer narrower previews over a second row.
  const perRow = Math.max(1, Math.min(
    Math.max(inRow.length, 1), Math.floor((available + PIP_GAP) / (PIP_MIN_W + PIP_GAP)),
  ));
  const width = Math.max(PIP_MIN_W, Math.min(
    Math.round(THREE.MathUtils.clamp(viewport.clientWidth * 0.23, PIP_MIN_W, 360)),
    Math.floor((available - PIP_GAP * (perRow - 1)) / perRow),
  ));

  const layout = new Map();
  // Start below the top strip so panes never cover its controls.
  let y = pipTopFloor();
  const paneHeight = (rec) => PIP_BORDER * 2 + PIP_HEAD + PIP_FOOT
    + Math.round(width / rec.proj.aspect);
  for (let start = 0; start < inRow.length; start += perRow) {
    const row = inRow.slice(start, start + perRow);
    let rowHeight = 0;
    row.forEach((rec, index) => {
      // Spread across the full width in outer (border-inclusive) widths, so a
      // row of two lands in the corners on the PIP_PAD margin.
      const outer = width + PIP_BORDER * 2;
      const x = row.length === 1
        ? PIP_PAD
        : Math.round(PIP_PAD + ((available - outer) * index) / (row.length - 1));
      layout.set(rec.name, { x, y, w: width });
      rowHeight = Math.max(rowHeight, paneHeight(rec));
    });
    y += rowHeight + PIP_GAP;
  }
  // Robot cameras run straight down the left column, below the last row.
  for (const rec of below) {
    layout.set(rec.name, { x: PIP_PAD, y, w: width });
    y += paneHeight(rec) + PIP_GAP;
  }
  return layout;
}

// Pane layout, remembered per camera config; everything read back is re-clamped.
const PIP_STORE_KEY = 'simfoundry.light-editor.pip-layout';

const pipStoreKey = () => (cameraConfig && cameraConfig.cfg_name) || 'default';

function readPipStore() {
  try {
    return JSON.parse(localStorage.getItem(PIP_STORE_KEY) || '{}') || {};
  } catch {
    return {};        // private browsing, or a value from an older format
  }
}

function savePipLayout() {
  const store = readPipStore();
  store[pipStoreKey()] = Object.fromEntries([...pipPanes].map(([name, pane]) => [
    // The wanted values, not the fitted ones a narrow window happened to allow.
    name, { x: pane.wantX ?? pane.x, y: pane.wantY ?? pane.y,
            w: pane.wantW ?? pane.w, minimized: pane.minimized },
  ]));
  try {
    localStorage.setItem(PIP_STORE_KEY, JSON.stringify(store));
  } catch { /* the panes still work; they just forget. Not worth a message. */ }
}

// Framing guides: action-safe (90%) and title-safe (80%) margins plus thirds.
function buildSafeFrame() {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'pip-safe');
  svg.setAttribute('viewBox', '0 0 100 100');
  // The pane carries the sensor's aspect, not a square, so the box is stretched.
  svg.setAttribute('preserveAspectRatio', 'none');
  const stroke = (d, opacity, dash) => {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#ffffff');
    path.setAttribute('stroke-opacity', opacity);
    // Keeps stroke width constant when the viewBox is stretched.
    path.setAttribute('vector-effect', 'non-scaling-stroke');
    if (dash) path.setAttribute('stroke-dasharray', dash);
    svg.appendChild(path);
  };
  stroke('M33.333 0V100M66.667 0V100M0 33.333H100M0 66.667H100', '0.18');
  stroke('M5 5H95V95H5Z', '0.42');
  stroke('M10 10H90V90H10Z', '0.3', '4 3');
  return svg;
}

function buildPipElement(pane) {
  const rec = pane.rec;
  const el = document.createElement('div');
  el.className = 'pip';
  el.dataset.camera = rec.name;
  el.style.setProperty('--pip-color', hexColour(rec.colour));

  const head = document.createElement('div');
  head.className = 'pip-head';
  head.title = 'Drag to move · double-click to put it back';
  const dot = document.createElement('span');
  dot.className = 'pip-dot';
  const name = document.createElement('span');
  name.className = 'pip-name';
  name.textContent = rec.name;
  const key = document.createElement('span');
  key.className = 'pip-key';
  const minimise = document.createElement('button');
  minimise.className = 'pip-min';
  head.append(dot, name, key, minimise);

  const body = document.createElement('div');
  body.className = 'pip-body';
  body.title = `Click to look through ${rec.name}`;
  const cross = document.createElement('div');
  cross.className = 'pip-cross';
  body.append(buildSafeFrame(), cross);

  const foot = document.createElement('div');
  foot.className = 'pip-foot';
  const res = document.createElement('span');
  res.className = 'pip-res';
  const grip = document.createElement('div');
  grip.className = 'pip-grip';
  grip.title = 'Drag to resize — the sensor’s aspect ratio is kept';
  foot.append(res, grip);

  el.append(head, body, foot);
  pipHost.appendChild(el);

  Object.assign(pane, {
    el, body, keyEl: key, resEl: res, minEl: minimise,
    guideEls: [...body.querySelectorAll('.pip-safe, .pip-cross')],
  });

  // The caption bar moves the pane; only the picture enters the camera, so
  // double-clicking the caption can still reset the pane.
  head.addEventListener('pointerdown', (e) => beginPipDrag(pane, e, 'move'));
  body.addEventListener('pointerdown', (e) => beginPipDrag(pane, e, 'enter'));
  grip.addEventListener('pointerdown', (e) => beginPipDrag(pane, e, 'resize'));
  // The button sits inside the drag handle, so it has to opt out of it.
  minimise.addEventListener('pointerdown', (e) => e.stopPropagation());
  minimise.onclick = (e) => { e.stopPropagation(); togglePipMinimized(pane); };
  head.addEventListener('dblclick', () => resetPip(pane));

  refreshPipChrome(pane);
}

function refreshPipChrome(pane) {
  const rec = pane.rec;
  const optics = rec.optics || {};
  const obs = rec.observation;

  // Which policy input this camera lands in (from the task configs, not the
  // rig config); an uncertain mapping is styled as a warning.
  pane.keyEl.textContent = (obs && obs.short) || '—';
  pane.keyEl.classList.toggle('uncertain', !!(obs && obs.certain === false));
  pane.keyEl.title = obs
    ? `${obs.key || 'not read by the evaluation stage'}\n${obs.detail}`
    : 'no task config was read — observation key unknown';

  const size = optics.width && optics.height
    ? `${optics.width}×${optics.height}`
    : `aspect ${rec.proj.aspect.toFixed(2)}`;
  const fov = optics.hFov && optics.vFov
    ? ` · ${optics.hFov.toFixed(0)}°×${optics.vFov.toFixed(0)}°`
    : '';
  pane.resEl.textContent = size + fov;
  pane.resEl.title = [
    `${rec.name}: ${size}${fov}`,
    optics.modalities && optics.modalities.length
      ? `modalities: ${optics.modalities.join(', ')}` : '',
    obs && obs.key ? `policy input: ${obs.key}` : '',
    obs && obs.detail ? obs.detail : '',
  ].filter(Boolean).join('\n');

  pane.minEl.textContent = pane.minimized ? '+' : '–';
  pane.minEl.title = pane.minimized
    ? `Show ${rec.name} again`
    : `Collapse ${rec.name} to its caption bar`;
  pane.el.classList.toggle('min', pane.minimized);
  for (const el of pane.guideEls) el.hidden = !guidesOn;
}

function applyPipGeometry(pane) {
  pane.el.style.left = `${pane.x}px`;
  pane.el.style.top = `${pane.y}px`;
  pane.el.style.width = `${pane.w + PIP_BORDER * 2}px`;
  pane.body.style.height = `${pipBodyHeight(pane)}px`;
}

function applyPipStacking() {
  pipOrder.forEach((name, index) => {
    const pane = pipPanes.get(name);
    // Below the drop overlay and the modals, which must stay on top of these.
    if (pane) pane.el.style.zIndex = String(20 + index);
  });
}

function bringPipToFront(pane) {
  const name = pane.rec.name;
  if (pipOrder[pipOrder.length - 1] === name) return;
  pipOrder = pipOrder.filter((other) => other !== name).concat(name);
  applyPipStacking();
}

// Clicking a preview selects its camera and steps inside it, switching tabs if needed.
function focusCamera(rec) {
  if (activeTab !== 'cameras') setTab('cameras');
  selectOrLookFrom(rec);
}

function beginPipDrag(pane, event, mode) {
  if (event.button !== 0) return;
  bringPipToFront(pane);
  const el = event.currentTarget;
  const start = {
    x: event.clientX, y: event.clientY, px: pane.x, py: pane.y, pw: pane.w,
  };
  let moved = false;
  // Pointer capture keeps a drag alive when the pointer leaves the pane.
  try { el.setPointerCapture(event.pointerId); } catch { /* not every pointer can be */ }

  const move = (e) => {
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    if (!moved && Math.hypot(dx, dy) <= PIP_DRAG_SLOP) return;
    moved = true;
    e.preventDefault();
    if (mode === 'resize') pane.wantW = start.pw + dx;
    else { pane.wantX = start.px + dx; pane.wantY = start.py + dy; }
    clampPane(pane);
    applyPipGeometry(pane);
  };

  const finish = (e) => {
    el.removeEventListener('pointermove', move);
    el.removeEventListener('pointerup', finish);
    el.removeEventListener('pointercancel', finish);
    if (el.hasPointerCapture && el.hasPointerCapture(e.pointerId)) {
      el.releasePointerCapture(e.pointerId);
    }
    if (moved) savePipLayout();
    // A press that never moved is a click; a click on the picture enters the camera.
    else if (mode === 'enter') focusCamera(pane.rec);
  };

  el.addEventListener('pointermove', move);
  el.addEventListener('pointerup', finish);
  el.addEventListener('pointercancel', finish);
}

function togglePipMinimized(pane) {
  pane.minimized = !pane.minimized;
  clampPane(pane);
  applyPipGeometry(pane);
  refreshPipChrome(pane);
  savePipLayout();
}

function resetPip(pane) {
  const base = defaultPipLayout().get(pane.rec.name);
  if (!base) return;
  Object.assign(pane, base, { minimized: false });
  pane.wantX = pane.x; pane.wantY = pane.y; pane.wantW = pane.w;
  clampPane(pane);
  applyPipGeometry(pane);
  refreshPipChrome(pane);
  savePipLayout();
  setStatus(`${pane.rec.name} preview put back where it started.`);
}

function buildPips() {
  const cams = previewCameras();
  if (cams.length === 0) return;
  const defaults = defaultPipLayout();
  const saved = readPipStore()[pipStoreKey()] || {};

  pipHost.innerHTML = '';
  pipPanes.clear();
  for (const rec of cams) {
    const base = defaults.get(rec.name) || { x: PIP_PAD, y: PIP_PAD, w: 220 };
    const stored = saved[rec.name] || {};
    const pane = {
      rec,
      x: finite(stored.x, base.x),
      y: finite(stored.y, base.y),
      w: finite(stored.w, base.w),
      minimized: stored.minimized === true,
    };
    buildPipElement(pane);
    clampPane(pane);
    applyPipGeometry(pane);
    pipPanes.set(rec.name, pane);
  }
  pipOrder = cams.map((rec) => rec.name);
  applyPipStacking();
  updatePipVisibility();
}

function updatePipVisibility() {
  // Hidden while flying (redundant inside a camera) and on viewports too
  // narrow to hold even one pane.
  const roomForAPane = viewport.clientWidth >= PIP_MIN_W + PIP_PAD * 2;
  pipHost.hidden = !(pipOn && !flying && pipPanes.size > 0 && roomForAPane);
}

function setPipEnabled(on) {
  pipOn = on;
  document.getElementById('m-pip').classList.toggle('on', pipOn);
  updatePipVisibility();
  setStatus(pipOn ? 'Camera views on.' : 'Camera views hidden.');
}

function setGuidesEnabled(on) {
  guidesOn = on;
  document.getElementById('m-guides').classList.toggle('on', guidesOn);
  for (const pane of pipPanes.values()) refreshPipChrome(pane);
  setStatus(guidesOn
    ? 'Preview guides on — 90% action-safe, 80% title-safe, thirds.'
    : 'Preview guides off.');
}

document.getElementById('m-pip').onclick = () => setPipEnabled(!pipOn);
document.getElementById('m-guides').onclick = () => setGuidesEnabled(!guidesOn);

// Re-clamps every pane; also driven by the panel seam, which changes the
// viewport without a window resize. Deliberately does not save: a squashed
// layout should not be written down.
function reflowPips() {
  for (const pane of pipPanes.values()) {
    clampPane(pane);
    applyPipGeometry(pane);
  }
  updatePipVisibility();
}
window.addEventListener('resize', reflowPips);

// --- dragging the seam -------------------------------------------------------
// Changing a flex sibling's width fires no window resize, so every path that
// changes the panel's width must call resize() and reflowPips() itself.
function setPanelWidth(px) {
  // The width asked for, kept apart from the width that fits — same as the preview panes.
  panelWantW = Number.isFinite(px) ? px : PANEL_DEFAULT_W;
  panelWidth = applyPanelWidth(px);
  resize();
  reflowPips();
  refreshGripValue();
  return panelWidth;
}

const panelGrip = document.getElementById('panel-grip');

function refreshGripValue() {
  panelGrip.setAttribute('aria-valuemin', String(PANEL_MIN_W));
  panelGrip.setAttribute('aria-valuemax', String(panelLimit()));
  panelGrip.setAttribute('aria-valuenow', String(panelWidth));
}
refreshGripValue();

panelGrip.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  const start = { x: event.clientX, w: panelWidth };
  let moved = false;
  let want = panelWidth;
  let queued = false;
  // Capture, because the pointer leaves a six-pixel handle on the first frame.
  try { panelGrip.setPointerCapture(event.pointerId); } catch { /* not every pointer can be */ }

  // renderer.setSize reallocates the drawing buffer, so the width is applied
  // once per animation frame rather than once per pointermove.
  const flush = () => {
    if (!queued) return;
    queued = false;
    setPanelWidth(want);
  };

  const move = (e) => {
    const dx = e.clientX - start.x;
    if (!moved && Math.abs(dx) <= PANEL_DRAG_SLOP) return;
    if (!moved) document.body.classList.add('resizing');
    moved = true;
    e.preventDefault();
    // The panel is on the right, so dragging the seam left makes it wider.
    want = start.w - dx;
    if (!queued) { queued = true; requestAnimationFrame(flush); }
  };

  const finish = (e) => {
    panelGrip.removeEventListener('pointermove', move);
    panelGrip.removeEventListener('pointerup', finish);
    panelGrip.removeEventListener('pointercancel', finish);
    if (panelGrip.hasPointerCapture && panelGrip.hasPointerCapture(e.pointerId)) {
      panelGrip.releasePointerCapture(e.pointerId);
    }
    document.body.classList.remove('resizing');
    if (!moved) return;
    flush();          // the last move may still be sitting in a queued frame
    savePanelWidth();
    setStatus(`Panel ${panelWidth} px.`);
  };

  panelGrip.addEventListener('pointermove', move);
  panelGrip.addEventListener('pointerup', finish);
  panelGrip.addEventListener('pointercancel', finish);
});

// No click listener: a click fires after a drag however far it went.
panelGrip.addEventListener('dblclick', () => {
  setPanelWidth(PANEL_DEFAULT_W);
  savePanelWidth();
  setStatus('Panel put back to its default width.');
});

// Keyboard resize for the seam; stopPropagation keeps the arrows from also
// nudging the selected object.
panelGrip.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 64 : 16;
  let next = null;
  if (e.key === 'ArrowLeft') next = panelWidth + step;
  else if (e.key === 'ArrowRight') next = panelWidth - step;
  else if (e.key === 'Home') next = PANEL_DEFAULT_W;
  if (next === null) return;
  e.preventDefault();
  e.stopPropagation();
  setPanelWidth(next);
  savePanelWidth();
  setStatus(`Panel ${panelWidth} px.`);
});

// Same drag mechanics as panelGrip above, on the vertical axis.
const objlistGrip = document.getElementById('objlist-grip');

objlistGrip.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  const start = { y: event.clientY, h: objlistHeight };
  let moved = false;
  let want = objlistHeight;
  let queued = false;
  try { objlistGrip.setPointerCapture(event.pointerId); } catch { /* not every pointer can be */ }

  const flush = () => {
    if (!queued) return;
    queued = false;
    objlistHeight = applyObjlistHeight(want);
  };

  const move = (e) => {
    const dy = e.clientY - start.y;
    if (!moved && Math.abs(dy) <= PANEL_DRAG_SLOP) return;
    if (!moved) document.body.classList.add('resizing-objlist');
    moved = true;
    e.preventDefault();
    want = start.h + dy;
    if (!queued) { queued = true; requestAnimationFrame(flush); }
  };

  const finish = (e) => {
    objlistGrip.removeEventListener('pointermove', move);
    objlistGrip.removeEventListener('pointerup', finish);
    objlistGrip.removeEventListener('pointercancel', finish);
    if (objlistGrip.hasPointerCapture && objlistGrip.hasPointerCapture(e.pointerId)) {
      objlistGrip.releasePointerCapture(e.pointerId);
    }
    document.body.classList.remove('resizing-objlist');
    if (!moved) return;
    flush();
    saveObjlistHeight();
    setStatus(`Object list ${objlistHeight} px.`);
  };

  objlistGrip.addEventListener('pointermove', move);
  objlistGrip.addEventListener('pointerup', finish);
  objlistGrip.addEventListener('pointercancel', finish);
});

// No click listener, matching panelGrip.
objlistGrip.addEventListener('dblclick', () => {
  objlistHeight = applyObjlistHeight(OBJLIST_DEFAULT_H);
  saveObjlistHeight();
  setStatus('Object list put back to its default height.');
});

objlistGrip.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 64 : 16;
  let next = null;
  if (e.key === 'ArrowUp') next = objlistHeight - step;
  else if (e.key === 'ArrowDown') next = objlistHeight + step;
  else if (e.key === 'Home') next = OBJLIST_DEFAULT_H;
  if (next === null) return;
  e.preventDefault();
  e.stopPropagation();
  objlistHeight = applyObjlistHeight(next);
  saveObjlistHeight();
  setStatus(`Object list ${objlistHeight} px.`);
});

// --- help drawer -------------------------------------------------------------
// The full key reference, one click or one `?` away.
const helpModal = document.getElementById('help-modal');

function helpIsOpen() { return !helpModal.hidden; }
function openHelp() { helpModal.hidden = false; }
function closeHelp() { helpModal.hidden = true; }

document.getElementById('btn-help').onclick = openHelp;
document.getElementById('help-close').onclick = closeHelp;
document.getElementById('help-backdrop').onclick = closeHelp;
document.getElementById('help-tour').onclick = () => { closeHelp(); openTour(); };

// --- first-run walkthrough ----------------------------------------------------
// Five-step intro, shown once per browser and reachable again from the Help drawer.
const TOUR_KEY = 'simfoundry.light-editor.tour-seen';
const TOUR_STEPS = [
  {
    title: 'Selecting things',
    text: 'Click an object in the viewport, or its row in the panel on the '
      + 'right, to select it. Shift+click adds another to the selection, and '
      + 'Ctrl+A selects every prop at once — the robot and the room stay out.',
  },
  {
    title: 'Moving things',
    text: 'A selected object gets a gizmo: drag an arrow to move it, an arc to '
      + 'turn it, or a handle to scale it. M and R switch the gizmo between '
      + 'move and rotate; + and − scale the selection without leaving either. '
      + 'The Transform panel below the list types exact numbers instead.',
  },
  {
    title: 'Placing on a surface',
    text: '"+ Add object" and Duplicate drop the new copy onto whatever '
      + 'surface is beneath the anchor object, not at the origin — so a plate '
      + 'lands on the table it is meant to sit on, not inside it.',
  },
  {
    title: 'Checking the layout',
    text: '"Check layout" looks for what is easy to miss by eye: objects '
      + 'overlapping, sitting outside a camera’s frame, or out of the '
      + 'robot’s reach. It warns; it never blocks a save.',
  },
  {
    title: 'Saving',
    text: '"Save scene JSON" writes your changes back to disk. "Review & '
      + 'Export…" bundles the scene, the cameras and a task into one '
      + 'package when you are ready to hand it off.',
  },
];
let tourStep = 0;

const tourModal = document.getElementById('tour-modal');

function tourIsOpen() { return !tourModal.hidden; }

function renderTourStep() {
  const step = TOUR_STEPS[tourStep];
  document.getElementById('tour-step-label').textContent =
    `${tourStep + 1} of ${TOUR_STEPS.length} — ${step.title}`;
  document.getElementById('tour-text').textContent = step.text;
  document.getElementById('tour-back').disabled = tourStep === 0;
  document.getElementById('tour-next').textContent =
    tourStep === TOUR_STEPS.length - 1 ? 'Done' : 'Next';
  const dots = document.getElementById('tour-dots');
  dots.innerHTML = '';
  TOUR_STEPS.forEach((_, i) => {
    const dot = document.createElement('span');
    if (i === tourStep) dot.className = 'on';
    dots.appendChild(dot);
  });
}

function openTour() {
  tourStep = 0;
  renderTourStep();
  tourModal.hidden = false;
}

function closeTour() {
  tourModal.hidden = true;
  try { localStorage.setItem(TOUR_KEY, '1'); } catch { /* still closes; just asks again next time */ }
}

function maybeShowTour() {
  let seen = false;
  try { seen = localStorage.getItem(TOUR_KEY) === '1'; } catch { /* private browsing: show it every time */ }
  if (!seen) openTour();
}

document.getElementById('tour-close').onclick = closeTour;
document.getElementById('tour-backdrop').onclick = closeTour;
document.getElementById('tour-back').onclick = () => {
  if (tourStep > 0) { tourStep--; renderTourStep(); }
};
document.getElementById('tour-next').onclick = () => {
  if (tourStep < TOUR_STEPS.length - 1) { tourStep++; renderTourStep(); }
  else closeTour();
};

function drawCameraPips(fullWidth, fullHeight) {
  // Back to front, matching the DOM stacking order.
  for (const name of pipOrder) {
    const pane = pipPanes.get(name);
    if (!pane || pane.minimized) continue;
    const height = pipBodyHeight(pane);
    const x = pane.x + PIP_BORDER;
    // WebGL's viewport origin is bottom-left; a pane's is top-left.
    const y = fullHeight - (pane.y + PIP_BORDER + PIP_HEAD + height);
    // clampPane keeps this true; skip rather than draw a partial rectangle.
    if (y < 0 || x < 0 || x + pane.w > fullWidth || y + height > fullHeight) continue;

    renderer.autoClear = false;
    renderer.setScissorTest(true);
    renderer.setScissor(x, y, pane.w, height);
    renderer.setViewport(x, y, pane.w, height);
    renderer.setClearColor(0x14161a, 1);
    renderer.clear(true, true, false);
    // proj sees only the content layer, so grids, gizmos and camera helpers
    // are excluded: the preview shows exactly what the sensor records.
    renderer.render(scene, pane.rec.proj);

    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, fullWidth, fullHeight);
    renderer.autoClear = true;
  }
}

const clock = new THREE.Clock();

(function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (flying) {
    updateFly(dt);
    updateCenterDistance(performance.now());
  } else {
    // Before `orbit.update()`, which reads the position and target this moves.
    updateFreeWalk(dt);
    orbit.update();
  }
  // Frustums are drawn in world space, so they have to follow their camera.
  for (const rec of objects.values()) {
    if (rec.isCamera && rec.helper.visible) rec.helper.update();
  }
  updateBoundingBoxes();
  updateTaskRangeBoxes();
  renderPreviewFrame(dt);
  if (flying) {
    renderFlyView();
  } else {
    renderer.render(scene, camera);
    if (pipOn) drawCameraPips(canvasWidth, canvasHeight);
  }
})();

// --- the table centre ------------------------------------------------------
// A hand-placed point on the table that Re-seat centres the arrangement on.
// Deliberately not a record in `objects`, or the prop filters would arrange
// and save it as a scene object. Stored in the room's sidecar, not the scene,
// so every scene in the room shares it.

let placingTable = false;
let tableState = null;              // last /api/table response
// Whether a point exists, kept apart from whether it is drawn: a hidden
// marker is still a placed one for Re-seat.
let tableHasPoint = false;
let tableMarkerHidden = false;
// Requested and already-applied yaw about the anchor, kept separately so
// pressing Re-seat twice does not rotate twice.
let tableYawDeg = 0;
let tableYawApplied = 0;
const tableMarker = new THREE.Group();
tableMarker.visible = false;
scene.add(tableMarker);

// Single place that decides whether the ring is drawn; placing always shows it.
function syncTableMarker() {
  tableMarker.visible = tableHasPoint && (placingTable || !tableMarkerHidden);
  const button = document.getElementById('btn-table-hide');
  if (button) {
    button.classList.toggle('on', tableMarkerHidden);
    button.textContent = tableMarkerHidden ? 'Show ring' : 'Hide ring';
    button.disabled = !tableHasPoint;
  }
}

{
  // Ring, stalk and cross marker, on the overlay layer so it never appears in
  // a sensor preview.
  const colour = 0x37d67a;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.045, 0.06, 32),
    new THREE.MeshBasicMaterial({ color: colour, side: THREE.DoubleSide,
      transparent: true, opacity: 0.9, depthTest: false }),
  );
  const stalk = new THREE.Mesh(
    new THREE.CylinderGeometry(0.004, 0.004, 0.18, 8),
    new THREE.MeshBasicMaterial({ color: colour, depthTest: false }),
  );
  stalk.rotation.x = Math.PI / 2;
  stalk.position.z = 0.09;
  const cross = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(
      [-0.09, 0, 0, 0.09, 0, 0, 0, -0.09, 0, 0, 0.09, 0], 3)),
    new THREE.LineBasicMaterial({ color: colour, depthTest: false }),
  );
  tableMarker.add(markAsOverlay(ring), markAsOverlay(stalk), markAsOverlay(cross));
}

function refreshTableReadout() {
  const p = tableMarker.position;
  tableMarker.rotation.z = THREE.MathUtils.degToRad(tableYawDeg);
  const yawField = document.getElementById('tc-yaw');
  if (yawField && document.activeElement !== yawField) yawField.value = tableYawDeg.toFixed(1);
  for (const [id, value] of [['tc0', p.x], ['tc1', p.y], ['tc2', p.z]]) {
    const field = document.getElementById(id);
    if (field && document.activeElement !== field) field.value = value.toFixed(4);
  }
  const save = document.getElementById('btn-table-save');
  if (save) save.disabled = !tableState || !tableState.room;
}

async function loadTableCentre() {
  try {
    const res = await fetch('./api/table');
    if (!res.ok) return;
    tableState = await res.json();
  } catch { return; }
  if (!tableState || !tableState.room) {
    const note = document.getElementById('table-note');
    if (note) note.textContent = 'no scanned room — nothing to put a table centre on';
    return;
  }

  const saved = tableState.centre;
  const estimate = tableState.estimate;
  const point = saved || (estimate && estimate.centre);
  const note = document.getElementById('table-note');
  if (point) {
    tableMarker.position.set(point[0], point[1], point[2]);
    tableHasPoint = true;
  }
  syncTableMarker();
  if (note) {
    if (saved) {
      note.textContent = `saved for ${tableState.room}`;
    } else if (estimate) {
      // Say plainly that this is a guess and how big a patch it came from.
      note.textContent = `estimated from the scan of ${tableState.room} — `
        + `${estimate.extent[0].toFixed(2)} x ${estimate.extent[1].toFixed(2)} m patch. `
        + 'Drag it onto the real centre, then Save.';
    } else {
      note.textContent = `no centre saved for ${tableState.room}`;
    }
  }
  refreshTableReadout();
}

function setPlacingTable(on) {
  if (on && (!tableState || !tableState.room)) {
    setStatus('This scene has no scanned room to place a table centre in.', 'err');
    return;
  }
  placingTable = on;
  if (on) {
    // Selecting an object while placing would leave the gizmo on two owners.
    select(null);
    tableHasPoint = true;
    gizmo.setMode('translate');
    gizmo.attach(tableMarker);
    setStatus('Drag the ring to the middle of the table, then Save table centre.');
  } else {
    gizmo.detach();
  }
  const button = document.getElementById('btn-table-place');
  if (button) {
    button.classList.toggle('on', on);
    button.textContent = on ? 'Done placing' : 'Place table centre';
  }
  syncTableMarker();
  refreshTableReadout();
}

async function saveTableCentre() {
  if (!tableState || !tableState.room) return;
  const p = tableMarker.position;
  try {
    const res = await fetch('./api/save_table', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      body: JSON.stringify({ centre: [p.x, p.y, p.z] }),
    });
    const body = await res.json();
    if (!res.ok) { setStatus(body.error || `HTTP ${res.status}`, 'err'); return; }
    tableState = body;
    const note = document.getElementById('table-note');
    if (note) note.textContent = `saved for ${tableState.room}`;
    setStatus(`Table centre saved for ${tableState.room}; every scene in this room `
      + 'will pick it up.');
  } catch (err) {
    setStatus(`Could not save the table centre: ${err.message}`, 'err');
  }
}

// Rigidly shift the layout so the anchor sits on the table centre. XY only:
// relative offsets survive and props stay on their support surface.
function reseatOnTableCentre() {
  // `tableHasPoint`, not visibility: a hidden ring is still a placed one.
  if (!tableState || !tableState.room || !tableHasPoint) {
    setStatus('Place a table centre first.', 'err');
    return;
  }
  const anchor = objects.get(anchorName);
  if (!anchor) { setStatus('Set an anchor first.', 'err'); return; }

  // The anchor's bounding-box centre, not its origin (a pivot may sit anywhere).
  const centre = new THREE.Box3().setFromObject(anchor.group).getCenter(new THREE.Vector3());

  // Yaw first, about the anchor, then translate. Applied as a delta against
  // what was applied last, so pressing Re-seat twice does not rotate twice.
  const movers = [anchor, ...propRecords()];
  pushUndo(movers, 'reseat');

  const turn = THREE.MathUtils.degToRad(tableYawDeg - tableYawApplied);
  if (Math.abs(turn) > 1e-9) {
    const spin = new THREE.Quaternion().setFromAxisAngle(WORLD_UP, turn);
    movers.forEach((rec) => {
      // Orbit the position about the anchor's centre...
      const offset = rec.group.position.clone().sub(centre);
      offset.applyQuaternion(spin);
      rec.group.position.copy(centre).add(offset);
      // ...and turn the object itself by the same amount.
      rec.group.quaternion.premultiply(spin);
    });
    tableYawApplied = tableYawDeg;
    // The spin moved the anchor; re-measure its centre before translating.
    new THREE.Box3().setFromObject(anchor.group).getCenter(centre);
  }

  const dx = tableMarker.position.x - centre.x;
  const dy = tableMarker.position.y - centre.y;
  movers.forEach((rec) => {
    rec.group.position.x += dx;
    rec.group.position.y += dy;
    refreshDirty(rec);
  });

  rebuildPivot();
  renderList();
  refreshReadout();
  const robot = liveRecords().find((r) => r.entry.kind === 'robot');
  let reach = '';
  if (robot) {
    const d = Math.hypot(tableMarker.position.x - robot.group.position.x,
      tableMarker.position.y - robot.group.position.y);
    // Warn when the re-seated arrangement is beyond the robot's reach.
    reach = ` ${anchorName} now sits ${d.toFixed(2)} m from ${robot.name}`
      + (d > ROBOT_REACH_M ? ' — beyond its reach.' : '.');
  }
  setStatus(`Re-seated ${movers.length} object(s) on the table centre, moved `
    + `${Math.hypot(dx, dy).toFixed(3)} m. Relative spacing unchanged.${reach}`);
}

document.getElementById('btn-table-place').onclick = () => setPlacingTable(!placingTable);
document.getElementById('btn-table-save').onclick = saveTableCentre;
document.getElementById('tc-yaw').addEventListener('change', (e) => {
  const value = parseFloat(e.target.value);
  if (!Number.isFinite(value)) { refreshTableReadout(); return; }
  tableYawDeg = value;
  refreshTableReadout();
});
document.getElementById('btn-table-hide').onclick = () => {
  tableMarkerHidden = !tableMarkerHidden;
  syncTableMarker();
};
document.getElementById('btn-reseat').onclick = reseatOnTableCentre;
for (const [id, axis] of [['tc0', 'x'], ['tc1', 'y'], ['tc2', 'z']]) {
  document.getElementById(id).addEventListener('change', (e) => {
    const value = parseFloat(e.target.value);
    if (!Number.isFinite(value)) { refreshTableReadout(); return; }
    tableMarker.position[axis] = value;
    tableHasPoint = true;
    syncTableMarker();
    refreshTableReadout();
  });
}

// bootDone lets the heartbeat wait for boot before rehydrating another tab's save.
boot().then(loadTableCentre).finally(() => bootDone());

// --- importing and exporting camera rigs ------------------------------------
// Panel half of the server's config list/load/export endpoints. Kept out of
// loadCameras() so a failure here cannot stop the cameras loading.

async function refreshCameraConfigs() {
  const list = document.getElementById('cam-config-list');
  if (!list) return;
  let payload;
  try {
    const res = await fetch('/api/camera_configs');
    if (!res.ok) return;
    payload = await res.json();
  } catch { return; }

  list.textContent = '';
  for (const row of payload.configs || []) {
    const option = document.createElement('option');
    option.value = row.name;
    // Label with the sensors; an unloadable config is disabled and says so.
    option.textContent = row.error
      ? `${row.name}  — unusable`
      : `${row.name}  (${(row.sensors || []).join(', ') || 'no sensors'})`;
    option.disabled = !!row.error;
    option.title = row.error || row.path;
    if (row.name === payload.current) option.selected = true;
    list.append(option);
  }
  const target = document.getElementById('cam-out-name');
  if (target) target.placeholder = payload.out_name || '(room default)';
}

async function importCameras(discard = false) {
  const list = document.getElementById('cam-config-list');
  const name = list && list.value;
  if (!name) return;
  try {
    const res = await fetch('/api/load_cameras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Editor-Token': EDITOR_TOKEN },
      // The server decides whether pending edits block this; the browser only reports them.
      body: JSON.stringify({ name, dirty: cameraDirty.size > 0, discard }),
    });
    const body = await res.json();
    if (res.status === 409 && body.needs_confirm) {
      if (window.confirm('The cameras have unsaved changes. Discard them and import '
        + `${name}?`)) await importCameras(true);
      return;
    }
    if (!res.ok) { setStatus(body.error || `HTTP ${res.status}`, 'err'); return; }

    // Rebuild from the server's new state: the rig can differ in sensor count,
    // optics and observation keys.
    for (const rec of [...objects.values()].filter((r) => r.isCamera)) {
      scene.remove(rec.helper);
      rec.group.parent?.remove(rec.group);
      objects.delete(rec.name);
    }
    cameraDirty.clear();
    await loadCameras();
    await refreshCameraConfigs();
    renderList();
    setStatus(`Imported ${body.name}: ${(body.sensors || []).join(', ')}.`, 'ok');
  } catch (err) {
    setStatus(`Could not import: ${err.message}`, 'err');
  }
}

// Fly to whichever sensor lands in a given policy input, keyed off the
// observation map rather than config order.
function flyToExterior(slot) {
  const wanted = `exterior_image_${slot}_left`;
  const rec = [...objects.values()].find(
    (r) => r.isCamera && r.observation && r.observation.key === wanted);
  const fallback = [...objects.values()].filter((r) => r.isCamera)[slot - 1];
  const target = rec || fallback;
  if (!target) { setStatus(`No camera for ${wanted}.`, 'err'); return; }
  setTab('cameras');
  select(target, { enterCamera: true });
  setStatus(`Flying ${target.name}${rec ? '' : ' (by config order — no observation key)'}.`);
}

document.getElementById('btn-import-cameras').onclick = () => importCameras();
document.getElementById('btn-fly-ext1').onclick = () => flyToExterior(1);
document.getElementById('btn-fly-ext2').onclick = () => flyToExterior(2);
refreshCameraConfigs();

// --- the ground plane ------------------------------------------------------
// Editor half of the saved scene's `ground_plane_info`: a height, and whether
// the simulator draws the plane. A Gaussian-splat room loads visual_only with
// no colliders, so without this plane every prop falls through it. Like the
// table marker, deliberately not a record in `objects` and not part of the
// undo history.

// The ground* state lives with the other per-scene state near the top of this file.

// A wide, faint disc with a rim stands in for OmniGibson's infinite plane;
// the edge makes the height readable at a glance.
const groundProxy = new THREE.Group();
groundProxy.visible = false;
{
  const colour = 0xf0a92b;
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(3.0, 96),
    new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.09,
      side: THREE.DoubleSide, depthWrite: false }),
  );
  const rim = new THREE.LineLoop(
    new THREE.CircleGeometry(3.0, 96),
    new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.55 }),
  );
  // CircleGeometry's first vertex is its centre; a LineLoop would draw a spoke to it.
  rim.geometry = new THREE.BufferGeometry().setAttribute(
    'position',
    new THREE.Float32BufferAttribute(
      Array.from({ length: 96 }, (_, i) => {
        const a = (i / 96) * Math.PI * 2;
        return [Math.cos(a) * 3.0, Math.sin(a) * 3.0, 0];
      }).flat(), 3),
  );
  groundProxy.add(markAsOverlay(disc), markAsOverlay(rim));
}
// Overlay layer, so the proxy never appears in a sensor preview; the checkbox
// states what the simulator will do.
scene.add(groundProxy);

const groundEl = (id) => document.getElementById(id);

function groundPayload() {
  if (groundLive === null) return null;
  return {
    position: [0, 0, groundLive.height],
    orientation: groundLive.orientation.slice(),
    visible: groundLive.visible,
  };
}

const groundSame = (a, b) => (a === null || b === null)
  ? a === b
  : Math.abs(a.height - b.height) < 1e-9 && a.visible === b.visible
    && a.orientation.every((v, i) => Math.abs(v - b.orientation[i]) < 1e-9);

/** Take a server-reported plane as the saved baseline. */
function adoptSavedGround(info) {
  groundSaved = info === null || info === undefined ? null : {
    height: info.position[2],
    orientation: info.orientation.slice(),
    visible: info.visible === undefined ? null : info.visible,
  };
  refreshGround();
}

function setGround(next, { adopt = false } = {}) {
  groundLive = next;
  if (adopt) groundSaved = next === null ? null : { ...next, orientation: next.orientation.slice() };
  refreshGround();
}

function refreshGround() {
  groundDirty = !groundSame(groundLive, groundSaved);
  groundProxy.position.z = groundLive ? groundLive.height : 0;
  groundProxy.visible = groundLive !== null && !groundHiddenForView;
  // Everything below reads the panel; guarded so a missing section cannot throw.
  if (!groundEl('ground-state')) { refreshSaveButton(); return; }

  const floorButton = groundEl('m-floor');
  if (floorButton) {
    floorButton.hidden = groundLive === null;
    // Lit means shown, same as every other button in the Visibility section.
    floorButton.classList.toggle('on', !groundHiddenForView);
  }

  groundEl('btn-ground-add').hidden = groundLive !== null;
  groundEl('btn-ground-remove').hidden = groundLive === null;
  groundEl('ground-controls').hidden = groundLive === null;
  const revert = groundEl('btn-ground-revert');
  if (revert) revert.disabled = !groundDirty;
  const table = groundEl('btn-ground-table');
  if (table) {
    const height = groundState && groundState.table_height;
    table.disabled = height === null || height === undefined;
    table.title = table.disabled
      ? 'No work-surface marker saved for this room yet'
      : `Put it at the marker's height (${height.toFixed(3)} m)`;
  }

  const heightField = groundEl('ground-height');
  if (heightField && groundLive && document.activeElement !== heightField) {
    heightField.value = groundLive.height.toFixed(4);
  }
  const visibleField = groundEl('ground-visible');
  if (visibleField && groundLive) visibleField.checked = groundLive.visible === true;

  const state = groundEl('ground-state');
  const note = groundEl('ground-note');
  const splat = groundState && groundState.background === 'splat';
  if (state) {
    if (groundLive === null && splat) {
      state.dataset.state = 'empty';
      state.textContent = 'This scene’s room is a Gaussian splat, which has no '
        + 'collision geometry. Without a ground plane every prop falls through it.';
    } else if (groundLive === null) {
      state.dataset.state = 'empty';
      state.textContent = 'No ground plane in this scene. The run config’s own '
        + 'floor plane stands, wherever it puts it.';
    } else {
      state.dataset.state = 'ok';
      state.textContent = `Props rest at z = ${groundLive.height.toFixed(3)} m.`
        + (groundLive.visible === false ? ' The simulator will not draw it.'
          : groundLive.visible === true ? ' The simulator will draw it.'
            : ' Visibility is left to the run config.');
    }
  }
  if (note) {
    note.textContent = groundLive === null ? '' : (splat && groundLive.visible === false
      ? 'The splat casts its contact shadows onto this plane, so hiding it also '
        + 'removes them. Whether a plane exists at all is the run config’s '
        + 'use_floor_plane, not the scene.'
      : 'Whether a plane exists at all is the run config’s use_floor_plane, '
        + 'not the scene; this says where it goes.');
  }

  refreshSaveButton();
}

/** A sensible height for a plane this scene does not have yet. */
function suggestedGroundHeight() {
  if (groundState && typeof groundState.table_height === 'number') {
    return groundState.table_height;
  }
  // Fall back to the lowest editable prop's origin, then zero.
  const props = liveRecords().filter((r) => !r.isCamera && r.entry.editable);
  if (props.length) return Math.min(...props.map((r) => r.group.position.z));
  return 0;
}

async function loadGroundPlane() {
  try {
    const res = await fetch('./api/ground_plane');
    if (!res.ok) return;
    groundState = await res.json();
  } catch { return; }
  adoptSavedGround(groundState.plane);
  setGround(groundSaved === null ? null : { ...groundSaved,
    orientation: groundSaved.orientation.slice() });
  if (groundSaved === null && groundState.background === 'splat') {
    setStatus('This scene has a Gaussian-splat room and no ground plane — props '
      + 'have nothing to rest on. Add one under Ground plane.', 'err');
    // Open the section by default for splat scenes; a stored user preference still wins.
    if (groundSection) groundSection.setDefault(false);
  }
}

groundEl('btn-ground-add').onclick = () => {
  const height = suggestedGroundHeight();
  setGround({
    height,
    orientation: [0, 0, 0, 1],
    // Hidden by default in a splat room; left to the run config elsewhere.
    visible: groundState && groundState.background === 'splat' ? false : null,
  });
  groundHiddenForView = false;
  refreshGround();
  setStatus(`Ground plane at z = ${height.toFixed(3)} m. `
    + 'Nothing is written until you save.');
};

groundEl('btn-ground-remove').onclick = () => {
  setGround(null);
  setStatus('Ground plane removed. The run config’s own floor plane will stand.');
};

groundEl('btn-ground-revert').onclick = () => {
  setGround(groundSaved === null ? null : { ...groundSaved,
    orientation: groundSaved.orientation.slice() });
  setStatus('Ground plane back to the saved value.');
};

groundEl('btn-ground-table').onclick = () => {
  if (!groundLive || !groundState || typeof groundState.table_height !== 'number') return;
  groundLive.height = groundState.table_height;
  refreshGround();
  setStatus(`Ground plane at the work-surface marker, z = ${groundLive.height.toFixed(3)} m.`);
};

// Both events: typing fires `input`, leaving the field fires `change`.
groundEl('ground-height').oninput = groundEl('ground-height').onchange = (e) => {
  if (!groundLive) return;
  const value = parseFloat(e.target.value);
  if (!Number.isFinite(value)) return;
  groundLive.height = value;
  refreshGround();
};

groundEl('ground-visible').onchange = (e) => {
  if (!groundLive) return;
  groundLive.visible = e.target.checked;
  refreshGround();
};

groundEl('m-floor').onclick = () => {
  groundHiddenForView = !groundHiddenForView;
  refreshGround();
};

// --- collapsible sections ---------------------------------------------------
// Collapsed state is remembered per browser; `defaultCollapsed` applies only
// when nothing is stored. The returned `setDefault` revises that default
// later (never a stored preference) for sections whose right default depends
// on data that arrives after they are built.
function initCollapsibleSection(sectionId, toggleId, storageKey, defaultCollapsed) {
  const section = groundEl(sectionId);
  const toggle = groundEl(toggleId);

  function setCollapsed(collapsed) {
    section.classList.toggle('collapsed', collapsed);
    toggle.setAttribute('aria-expanded', String(!collapsed));
  }

  function toggleCollapsed() {
    const collapsed = !section.classList.contains('collapsed');
    setCollapsed(collapsed);
    try { localStorage.setItem(storageKey, collapsed ? '1' : '0'); } catch { /* still toggles; just forgets */ }
  }

  toggle.onclick = toggleCollapsed;
  // role="button" on a <h2>: none of a real button's key handling comes free.
  toggle.onkeydown = (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    toggleCollapsed();
  };

  // Re-read rather than cached: the user may store an answer before setDefault runs.
  const stored = () => {
    try { return localStorage.getItem(storageKey); } catch { return null; }
  };

  const remembered = stored();
  setCollapsed(remembered === null ? defaultCollapsed : remembered === '1');
  return { setDefault: (collapsed) => { if (stored() === null) setCollapsed(collapsed); } };
}

groundSection = initCollapsibleSection('ground-section', 'ground-toggle',
  'simfoundry.light-editor.ground-collapsed', true);
initCollapsibleSection('task-gen-section', 'task-gen-toggle',
  'simfoundry.light-editor.task-gen-collapsed', true);
// Open by default: the ranges are what this section is for.
initCollapsibleSection('task-section', 'task-toggle',
  'simfoundry.light-editor.task-collapsed', false);
// Closed by default: table centre, recenter and arrange are one-shot tools.
initCollapsibleSection('arrange-section', 'arrange-toggle',
  'simfoundry.light-editor.arrange-collapsed', true);
// Open by default: the list and the inspector are the panel's working core.
initCollapsibleSection('objects', 'objects-heading',
  'simfoundry.light-editor.objects-collapsed', false);
initCollapsibleSection('selected-section', 'selected-toggle',
  'simfoundry.light-editor.selected-collapsed', false);
