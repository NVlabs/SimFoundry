// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Draws a NuRec Gaussian-splat room in the editor's three.js scene.
//
// `splat_io.py` extracts the gaussians from the `.nurec` payload into a
// `.splat` file; this module renders them as one instanced quad per gaussian,
// sized and oriented in the vertex shader from the projection of the
// gaussian's 3D covariance (standard EWA splatting).
//
//   * The gaussians live in textures, not vertex attributes: the shader looks
//     up the draw order by `gl_InstanceID` and the gaussian by the index it
//     finds there, so a re-sort uploads 4 bytes per gaussian. An instanced
//     attribute cannot work here -- three.js binds the vertex layout before
//     `onBeforeRender`, the only hook that knows which camera is about to
//     draw; uniforms are uploaded after it and can be swapped there.
//
//   * Every camera keeps its own draw order, re-sorted only when that camera
//     moves; the editor renders the same scene from several viewpoints per
//     frame, and they disagree about which gaussian is in front.
//
//   * Depth is tested but never written: props in front of the room occlude
//     it, props behind show through it, and the room never occludes itself.
//
// Colour is view-independent: `splat_io` keeps the spherical-harmonic DC term
// and drops degrees 1-3, so highlights are baked to their average.

// Through the import map, like `app.js`: two specifiers resolving to the same
// file would be two module instances with two sets of class identities, and
// the renderer would skip this object.
import * as THREE from 'three';

const SPLAT_MAGIC = 'SFSPLAT';

// Texture width for the gaussian data. The device's MAX_TEXTURE_SIZE caps the
// height, and a texture the driver refuses fails silently -- so the ceiling is
// measured before anything is allocated; see `deviceTextureLimit`.
const DATA_WIDTH = 2048;

// How far a camera may move before its order is stale. Metres for the
// position; the direction threshold is on a unit vector, so radians-ish.
const RESORT_DISTANCE = 0.01;
const RESORT_DIRECTION = 0.002;

const VERTEX_SHADER = /* glsl */`
precision highp float;
precision highp int;
precision highp sampler2D;

in vec2 quad;              // corner of the billboard, in units of sigma

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform sampler2D splatData;     // 3 texels per gaussian: centre, covariance
uniform sampler2D splatColour;   // 1 texel per gaussian: rgb + opacity
uniform sampler2D splatOrder;    // 1 texel per slot: which gaussian draws there
uniform ivec2 dataSize;
uniform ivec2 colourSize;
uniform ivec2 orderSize;
uniform vec2 viewport;           // drawing-buffer pixels of the current pass
uniform float opacityScale;
uniform float nearClip;          // NuRec's own, out of the room's config

out vec4 vColour;
out vec2 vQuad;

vec4 fetch(sampler2D tex, ivec2 size, int i) {
  return texelFetch(tex, ivec2(i % size.x, i / size.x), 0);
}

void main() {
  // Back to front for this camera. The instance number is a position in the
  // draw order, not a gaussian; the order texture says which gaussian it is.
  int id = int(fetch(splatOrder, orderSize, gl_InstanceID).r + 0.5);
  vec4 a = fetch(splatData, dataSize, id * 3);
  vec4 b = fetch(splatData, dataSize, id * 3 + 1);
  vec4 c = fetch(splatData, dataSize, id * 3 + 2);

  vec4 view = modelViewMatrix * vec4(a.xyz, 1.0);
  // Behind the eye there is no projection to take, and just in front of it a
  // scan's large faint gaussians project across the whole frame as a milky haze
  // over everything behind them. NuRec clips them at a distance recorded in the
  // room's own config, which splat_io carries into the .splat; using that
  // rather than a number of this file's choosing is what keeps the browser's
  // picture the same picture Isaac Sim renders. Parked outside the clip volume
  // rather than discarded in the fragment stage, which would still rasterise it.
  if (view.z > -nearClip) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  // Symmetric, so the column-major reading below is also the row-major one.
  mat3 sigma = mat3(a.w, b.x, b.y,
                    b.x, b.z, b.w,
                    b.y, b.w, c.x);

  float z = -view.z;
  // Focal length in pixels. P[0][0] is 1/(aspect*tan(fov/2)), which takes view
  // x to NDC; half the viewport takes NDC to pixels.
  float fx = projectionMatrix[0][0] * 0.5 * viewport.x;
  float fy = projectionMatrix[1][1] * 0.5 * viewport.y;

  // Jacobian of the perspective divide at this point, as a 3x3 with an empty
  // third row: sx = fx * view.x / z, sy = fy * view.y / z.
  mat3 J = mat3(
    fx / z,             0.0,                0.0,
    0.0,                fy / z,             0.0,
    fx * view.x / (z * z), fy * view.y / (z * z), 0.0);

  mat3 T = J * mat3(modelViewMatrix);
  mat3 screen = T * sigma * transpose(T);

  // Dilate by half a pixel in each direction. Without it a gaussian smaller
  // than a pixel projects to a sub-pixel ellipse that the rasteriser misses
  // entirely, and a scanned room is mostly those: it renders as holes.
  float sa = screen[0][0] + 0.3;
  float sd = screen[1][1] + 0.3;
  float sb = screen[0][1];

  float mid = 0.5 * (sa + sd);
  float radius = sqrt(max(0.0, mid * mid - (sa * sd - sb * sb)));
  float major = mid + radius;
  float minor = max(mid - radius, 0.1);
  if (major < 0.02) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  // Eigenvector of the larger eigenvalue. (major - sd, sb) is degenerate only
  // when the ellipse is already axis-aligned, which the branch covers.
  vec2 axis = abs(sb) < 1e-9
    ? (sa >= sd ? vec2(1.0, 0.0) : vec2(0.0, 1.0))
    : normalize(vec2(major - sd, sb));

  // The quad reaches two sigma; the fragment shader's cut-off agrees with it,
  // so the billboard is exactly the visible extent of the gaussian.
  vec2 majorAxis = min(sqrt(2.0 * major), 1024.0) * axis;
  vec2 minorAxis = min(sqrt(2.0 * minor), 1024.0) * vec2(axis.y, -axis.x);

  vec4 clip = projectionMatrix * view;
  vec2 offset = (quad.x * majorAxis + quad.y * minorAxis) / viewport * 2.0;

  vec4 col = fetch(splatColour, colourSize, id);
  vColour = vec4(col.rgb, col.a * opacityScale);
  vQuad = quad;
  gl_Position = vec4(clip.xy / clip.w + offset, clip.z / clip.w, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */`
precision highp float;

in vec4 vColour;
in vec2 vQuad;
out vec4 fragColour;

void main() {
  float power = -dot(vQuad, vQuad);
  if (power < -4.0) discard;                 // outside the billboard's ellipse
  float alpha = exp(power) * vColour.a;
  if (alpha < 0.004) discard;                // below one step of an 8-bit blend
  // Premultiplied, which is what the ONE / ONE_MINUS_SRC_ALPHA blend below
  // expects and what makes back-to-front compositing associative.
  fragColour = vec4(vColour.rgb * alpha, alpha);
}
`;

/**
 * Parse a `.splat` written by `splat_io.write_splat_file`.
 *
 * @param {ArrayBuffer} buffer Raw file bytes.
 * @returns {{header: object, centres: Float32Array, cov: Float32Array,
 *            colour: Uint8Array}}
 */
export function parseSplatFile(buffer) {
  const magic = new TextDecoder('latin1').decode(new Uint8Array(buffer, 0, 8));
  if (magic !== SPLAT_MAGIC) throw new Error('not a .splat file');
  const headerLength = new DataView(buffer).getUint32(8, true);
  const headerBytes = new Uint8Array(buffer, 12, headerLength);
  // The writer zero-pads the header to a 16-byte boundary so every array below
  // starts aligned; the padding is not part of the JSON.
  const text = new TextDecoder().decode(headerBytes).replace(/\0+$/, '');
  const header = JSON.parse(text);
  const count = header.count;

  let at = 12 + headerLength;
  const centres = new Float32Array(buffer, at, count * 3);
  at += count * 12;
  const cov = new Float32Array(buffer, at, count * 6);
  at += count * 24;
  const colour = new Uint8Array(buffer, at, count * 4);
  return { header, centres, cov, colour };
}

/**
 * Largest texture edge this device allows, in texels.
 *
 * Prefers the renderer that will draw the cloud; without one a throwaway
 * WebGL2 context is asked instead, and the first-frame check in
 * {@link SplatCloud#_checkUpload} catches any disagreement.
 *
 * @param {object} [renderer] A `THREE.WebGLRenderer`, if the caller has one.
 * @returns {number} Texels, or 0 when WebGL2 could not be reached and the
 *   limit is unknown; a caller must not refuse on a 0.
 */
function deviceTextureLimit(renderer) {
  const known = renderer?.capabilities?.maxTextureSize;
  if (known > 0) return known;
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (gl === null) return 0;
    const limit = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 0;
    // Release the context: a browser keeps only about sixteen live WebGL
    // contexts and drops the oldest, which here is the editor's own viewport.
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return limit;
  } catch (err) {
    return 0;
  }
}

/**
 * Drain the GL error queue, returning the first thing in it.
 *
 * Bounded rather than a `while`, so a driver that keeps reporting cannot take
 * the render loop with it.
 *
 * @param {WebGL2RenderingContext} gl
 * @returns {number} The first error code, or 0 for a clean queue.
 */
function drainGlErrors(gl) {
  let first = 0;
  for (let i = 0; i < 8; i++) {
    const code = gl.getError();
    if (code === 0) return first;
    if (first === 0) first = code;
  }
  return first;
}

/**
 * The room has more gaussians than this device can hold in a texture.
 *
 * Thrown from the constructor, inside `SplatCloud.load`'s promise, so a
 * caller that reports a failed load reports this too. The message stands on
 * its own; the fields are for a caller that wants to say it better.
 *
 * @param {number} count Gaussians in the file.
 * @param {number} rows Texture rows they need.
 * @param {number} limit Texels this device allows on a texture edge.
 * @returns {Error} With `name` `'SplatTooLargeError'` and `count`, `rows`,
 *   `limit` and `maxCount` fields.
 */
function splatTooLargeError(count, rows, limit) {
  const maxCount = Math.floor((limit * DATA_WIDTH) / 3);
  const error = new Error(
    `this Gaussian-splat room has ${count.toLocaleString()} gaussians, which need a `
    + `${DATA_WIDTH} x ${rows.toLocaleString()} texture -- taller than this device's `
    + `limit of ${limit.toLocaleString()} texels an edge, which holds `
    + `${maxCount.toLocaleString()} gaussians. Restart the editor with `
    + `--splat-budget ${maxCount} or lower.`);
  error.name = 'SplatTooLargeError';
  error.count = count;
  error.rows = rows;
  error.limit = limit;
  error.maxCount = maxCount;
  return error;
}

/**
 * A Gaussian-splat cloud, drawn as one three.js object.
 *
 * Construct it with `SplatCloud.load(url)`; the constructor takes already
 * parsed arrays so the tests can build a cloud without a fetch.
 */
export class SplatCloud {
  /**
   * @param {object} data From {@link parseSplatFile}.
   * @param {object} [options]
   * @param {string} [options.workerUrl] Where the sort worker lives.
   * @param {object} [options.renderer] The `THREE.WebGLRenderer` that will draw
   *   the cloud, whose texture limit decides whether this many gaussians can be
   *   drawn at all; without it the limit comes from a throwaway context.
   * @throws {Error} `SplatTooLargeError` when they cannot; see
   *   {@link splatTooLargeError}.
   */
  constructor(data, { workerUrl = './splat_sort_worker.js', renderer = null } = {}) {
    this.count = data.header.count;
    this.header = data.header;
    this.centres = data.centres;

    const dataHeight = Math.ceil((this.count * 3) / DATA_WIDTH);
    // Checked before packing, which for a large room is hundreds of megabytes.
    const limit = deviceTextureLimit(renderer);
    if (limit > 0 && dataHeight > limit) {
      throw splatTooLargeError(this.count, dataHeight, limit);
    }

    const packed = new Float32Array(DATA_WIDTH * dataHeight * 4);
    for (let i = 0; i < this.count; i++) {
      const t = i * 12, c = i * 3, v = i * 6;
      packed[t + 0] = data.centres[c];
      packed[t + 1] = data.centres[c + 1];
      packed[t + 2] = data.centres[c + 2];
      packed[t + 3] = data.cov[v];          // xx
      packed[t + 4] = data.cov[v + 1];      // xy
      packed[t + 5] = data.cov[v + 2];      // xz
      packed[t + 6] = data.cov[v + 3];      // yy
      packed[t + 7] = data.cov[v + 4];      // yz
      packed[t + 8] = data.cov[v + 5];      // zz
    }
    this.dataTexture = new THREE.DataTexture(
      packed, DATA_WIDTH, dataHeight, THREE.RGBAFormat, THREE.FloatType);
    this.dataTexture.needsUpdate = true;

    const colourHeight = Math.ceil(this.count / DATA_WIDTH);
    const colours = new Uint8Array(DATA_WIDTH * colourHeight * 4);
    colours.set(data.colour);
    this.colourTexture = new THREE.DataTexture(
      colours, DATA_WIDTH, colourHeight, THREE.RGBAFormat, THREE.UnsignedByteType);
    // Deliberately *not* flagged sRGB: 3DGS stores display-space bytes and the
    // shader writes them straight to the framebuffer. An sRGB flag would make
    // the driver decode to linear on sample and the room would render about a
    // stop too dark; blending in sRGB is also what NuRec and Isaac Sim do.
    this.colourTexture.needsUpdate = true;

    this.orderHeight = Math.ceil(this.count / DATA_WIDTH);

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        splatData: { value: this.dataTexture },
        splatColour: { value: this.colourTexture },
        splatOrder: { value: null },        // per camera; set in onBeforeRender
        dataSize: { value: new THREE.Vector2(DATA_WIDTH, dataHeight) },
        colourSize: { value: new THREE.Vector2(DATA_WIDTH, colourHeight) },
        orderSize: { value: new THREE.Vector2(DATA_WIDTH, this.orderHeight) },
        viewport: { value: new THREE.Vector2(1, 1) },
        opacityScale: { value: 1 },
        // 0.2 m is NuRec's default, for a room written before the header
        // carried the field. Zero would let a gaussian sitting on the eye
        // point project across the entire frame.
        nearClip: { value: this.header.nearClip || 0.2 },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      side: THREE.DoubleSide,
    });

    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('quad', new THREE.BufferAttribute(
      new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), 2));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    geometry.instanceCount = this.count;
    // Set rather than computed: `computeBoundingSphere` reads a three-component
    // `position` attribute, and this geometry's is a two-component `quad`. The
    // bounds come from the file.
    const low = new THREE.Vector3().fromArray(this.header.bounds.min);
    const high = new THREE.Vector3().fromArray(this.header.bounds.max);
    geometry.boundingBox = new THREE.Box3(low, high);
    geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new THREE.Sphere());
    this.geometry = geometry;

    this.object = new THREE.Mesh(geometry, this.material);
    this.object.name = 'splatCloud';
    this.object.frustumCulled = true;
    // A gaussian is not a surface: clicks fall through to whatever is behind
    // rather than reporting a contact point the room does not have.
    this.object.raycast = () => {};
    this.object.onBeforeRender = (renderer, scene, camera) => {
      this._beforeRender(renderer, camera);
    };

    /** @type {?Error} Set if the first frame's upload was refused. */
    this.uploadError = null;
    this._uploadChecked = false;

    /** @type {Map<string, object>} camera uuid -> its order and staleness */
    this._orders = new Map();
    this._generation = 0;
    this._viewport = new THREE.Vector4();
    this._modelView = new THREE.Matrix4();
    this._startWorker(workerUrl);
  }

  /**
   * Fetch and build a cloud.
   *
   * @param {string} url Path to a `.splat`.
   * @param {object} [options] Passed to the constructor.
   * @returns {Promise<SplatCloud>}
   */
  static async load(url, options) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
    return new SplatCloud(parseSplatFile(await response.arrayBuffer()), options);
  }

  _startWorker(workerUrl) {
    this.worker = null;
    try {
      this.worker = new Worker(new URL(workerUrl, import.meta.url), { type: 'module' });
    } catch (err) {
      // Degrade to the unsorted draw order rather than losing the room.
      console.warn('splat sort worker unavailable; drawing unsorted', err);
      return;
    }
    this.worker.onmessage = (event) => this._onSorted(event.data);
    // Copied out and transferred; the parsed arrays are views onto the
    // download buffer, which can then be collected.
    const centres = this.centres.slice().buffer;
    this.centres = null;
    this.worker.postMessage({ type: 'init', centres }, [centres]);
  }

  _orderFor(camera) {
    let record = this._orders.get(camera.uuid);
    if (record === undefined) {
      // Padded to the texture's full extent and seeded with the stored order,
      // so the room draws (badly blended) before the worker's first answer.
      const data = new Float32Array(DATA_WIDTH * this.orderHeight);
      for (let i = 0; i < this.count; i++) data[i] = i;
      const texture = new THREE.DataTexture(
        data, DATA_WIDTH, this.orderHeight, THREE.RedFormat, THREE.FloatType);
      texture.needsUpdate = true;
      record = {
        texture,
        data,
        position: new THREE.Vector3(Infinity, Infinity, Infinity),
        direction: new THREE.Vector3(),
        pending: 0,
      };
      this._orders.set(camera.uuid, record);
    }
    return record;
  }

  _beforeRender(renderer, camera) {
    // The first frame is the first time the driver is asked for these
    // textures: three.js uploads a DataTexture lazily, at the draw that
    // samples it. Checked once -- reading the error queue costs a stall.
    if (!this._uploadChecked) {
      this._uploadChecked = true;
      this.uploadError = this._checkUpload(renderer);
    }

    renderer.getCurrentViewport(this._viewport);
    this.material.uniforms.viewport.value.set(this._viewport.z, this._viewport.w);

    const record = this._orderFor(camera);
    this.material.uniforms.splatOrder.value = record.texture;
    // three.js re-uploads a material's uniforms only when the material changes
    // between draws; one cloud drawn from several cameras is the same material
    // each time, so without this the later passes would keep the first
    // camera's order and viewport.
    this.material.uniformsNeedUpdate = true;

    // Compared on the camera rather than the composed model-view, so an
    // unmoved camera costs one vector subtraction a frame.
    const position = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const direction = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const moved = position.distanceTo(record.position) > RESORT_DISTANCE
      || direction.distanceTo(record.direction) > RESORT_DIRECTION;
    if (!moved || record.pending || this.worker === null) return;

    record.position.copy(position);
    record.direction.copy(direction);
    record.pending = ++this._generation;
    this._modelView.multiplyMatrices(camera.matrixWorldInverse, this.object.matrixWorld);
    this.worker.postMessage({
      type: 'sort',
      camera: camera.uuid,
      generation: record.pending,
      modelView: this._modelView.elements,
    });
  }

  _onSorted(message) {
    if (message.type !== 'sorted') return;
    const record = this._orders.get(message.camera);
    if (record === undefined) return;
    record.pending = 0;
    // Copied into the texture's own array: it is padded to whole rows, and
    // three.js uploads the array the DataTexture was built around.
    record.data.set(new Float32Array(message.order));
    record.texture.needsUpdate = true;
    // Hand the buffer back for reuse instead of reallocating per sort. The
    // transfer list takes the buffer, never the view over it.
    this.worker.postMessage(
      { type: 'recycle', order: message.order }, [message.order.buffer]);
  }

  /**
   * Put the gaussian textures on the GPU now, and report a driver refusal.
   *
   * Safe to call from `onBeforeRender` even though it binds texture unit 0:
   * `setProgram` resets the unit allocation and rebinds every sampler before
   * the draw that follows.
   *
   * @param {object} renderer The `THREE.WebGLRenderer` about to draw the cloud.
   * @returns {?Error} Null when the upload was clean, otherwise an `Error` with
   *   a `glError` field -- `SplatTooLargeError` when the count is the reason.
   */
  _checkUpload(renderer) {
    const gl = renderer.getContext();
    // Clear whatever the rest of the frame left in the queue, so it is not
    // blamed on this room.
    drainGlErrors(gl);
    renderer.initTexture(this.dataTexture);
    renderer.initTexture(this.colourTexture);
    const code = drainGlErrors(gl);
    if (code === 0) return null;

    const limit = renderer.capabilities.maxTextureSize;
    const rows = this.dataTexture.image.height;
    const error = rows > limit
      ? splatTooLargeError(this.count, rows, limit)
      : Object.assign(
        new Error(
          `this device refused the textures for this Gaussian-splat room's `
          + `${this.count.toLocaleString()} gaussians (GL error ${code}); it will draw `
          + `as an empty room. Restart the editor with a lower --splat-budget.`),
        { name: 'SplatUploadError' });
    error.glError = code;
    // Logged as well as returned: this runs mid-frame, long after the load the
    // caller was awaiting.
    console.error(error.message);
    return error;
  }

  /**
   * Opacity multiplier applied to every gaussian.
   *
   * Currently unused: the editor's only opacity control over a splat room is
   * the all-or-nothing Background toggle.
   *
   * @param {number} scale Multiplied into every gaussian's stored alpha.
   */
  setOpacity(scale) {
    this.material.uniforms.opacityScale.value = scale;
  }

  /** Release the GPU resources and stop the worker. */
  dispose() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.geometry.dispose();
    this.material.dispose();
    this.dataTexture.dispose();
    this.colourTexture.dispose();
    for (const record of this._orders.values()) record.texture.dispose();
    this._orders.clear();
  }
}
