// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Depth-sorts a Gaussian-splat cloud, off the main thread.
//
// Splats are alpha-blended and never write depth, so they must be drawn back
// to front; a full reorder on every view change is too slow to run between
// frames. It runs here instead, and the viewport draws with the newest order
// that has arrived. One sort in flight per camera; a request arriving
// mid-sort replaces the pending one, because only the latest view matters.
//
// The sort is a 16-bit counting sort, O(n). Its output is a Float32Array
// because it is uploaded as a float texture; indices stay exact in float32 up
// to 2^24, far past any splat budget.

const BUCKETS = 65536;

let centres = null;          // Float32Array(count * 3), the gaussian centres
let count = 0;
let bucket = null;           // scratch: holds each splat's depth, then its bucket
let counts = new Uint32Array(BUCKETS);
const spare = [];            // order buffers handed back for reuse

function takeBuffer() {
  const buffer = spare.pop();
  return buffer && buffer.length === count ? buffer : new Float32Array(count);
}

function sort(modelView) {
  // View-space z of each centre. three.js matrices are column-major, so the
  // third row of the matrix is elements 2, 6, 10 and 14.
  const m2 = modelView[2], m6 = modelView[6], m10 = modelView[10], m14 = modelView[14];

  let low = Infinity, high = -Infinity;
  for (let i = 0, p = 0; i < count; i++, p += 3) {
    const z = m2 * centres[p] + m6 * centres[p + 1] + m10 * centres[p + 2] + m14;
    bucket[i] = z;                       // stored as a float here, rebucketed below
    if (z < low) low = z;
    if (z > high) high = z;
  }
  // Everything at one depth, or a degenerate matrix: any order is as correct as
  // any other, and rebucketing would divide by zero.
  if (!(high > low)) {
    const identity = takeBuffer();
    for (let i = 0; i < count; i++) identity[i] = i;
    return identity;
  }

  const scale = (BUCKETS - 1) / (high - low);
  counts.fill(0);
  for (let i = 0; i < count; i++) {
    // Ascending in view-space z. The camera looks down -z, so the most negative
    // is the farthest away and lands first: back to front, which is what the
    // blend needs.
    const b = (bucket[i] - low) * scale | 0;
    bucket[i] = b;
    counts[b]++;
  }
  let running = 0;
  for (let b = 0; b < BUCKETS; b++) {
    const n = counts[b];
    counts[b] = running;
    running += n;
  }
  const order = takeBuffer();
  for (let i = 0; i < count; i++) order[counts[bucket[i]]++] = i;
  return order;
}

self.onmessage = (event) => {
  const message = event.data;

  if (message.type === 'init') {
    centres = new Float32Array(message.centres);
    count = centres.length / 3;
    bucket = new Float32Array(count);
    spare.length = 0;
    self.postMessage({ type: 'ready', count });
    return;
  }

  if (message.type === 'recycle') {
    // Already a Float32Array over the transferred buffer; rewrapping would
    // copy the allocation this message exists to avoid.
    if (message.order) spare.push(message.order);
    return;
  }

  if (message.type === 'sort') {
    if (centres === null) return;
    const order = sort(message.modelView);
    self.postMessage(
      { type: 'sorted', camera: message.camera, generation: message.generation, order },
      [order.buffer],
    );
  }
};
