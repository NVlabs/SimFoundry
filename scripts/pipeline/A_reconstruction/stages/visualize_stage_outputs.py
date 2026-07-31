# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive reviewer for stage artifacts with rerun feedback output.

Supported modes:
- depth   : step-2 outputs (DA3 or FoundationStereo)
- upsample: step-6 per-object artifacts
- mesh    : step-7 per-object manifest + outputs
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
from matplotlib.widgets import Button, TextBox, CheckButtons
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir, list_object_iteration_indices


from simfoundry import CFG_DIR
logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)


@dataclass
class ReviewItem:
    key: str
    title: str
    images: dict[str, str]
    metadata: dict[str, Any]


def _read_img(path: str | None):
    if path is None or not os.path.exists(path):
        return None
    return np.array(Image.open(path))


def _build_depth_items(cfg) -> list[ReviewItem]:
    mode = cfg.s2_depth.backend
    if mode == "fs":
        out_dir = cfg.s2_fs.out_dir
        keys = sorted([f.replace("_vis.png", "") for f in os.listdir(out_dir) if f.endswith("_vis.png")])
        return [
            ReviewItem(
                key=k,
                title=f"FS: {k}",
                images={
                    "vis": f"{out_dir}/{k}_vis.png",
                    "disp": f"{out_dir}/{k}_disp.png",
                    "rgb": f"{out_dir}/{k}_rgb.png",
                },
                metadata={"backend": "fs"},
            )
            for k in keys
        ]

    da_dir = f"{cfg.s2_da.out_dir}/da"
    # DA3 output naming can vary. Prefer png previews if available.
    image_files = sorted([f for f in os.listdir(da_dir) if f.endswith(".png")]) if os.path.isdir(da_dir) else []
    return [
        ReviewItem(
            key=f,
            title=f"DA3: {f}",
            images={"preview": f"{da_dir}/{f}"},
            metadata={"backend": "da3"},
        )
        for f in image_files
    ]


def _build_upsample_items(cfg) -> list[ReviewItem]:
    out_dir = cfg.s6_upsample.out_dir
    upsampled_dir = f"{out_dir}/upsampled"
    ids = list_object_iteration_indices(os.listdir(upsampled_dir) if os.path.isdir(upsampled_dir) else [], suffix=".png")
    items = []
    for idx in ids:
        key = f"iter_{idx}"
        items.append(
            ReviewItem(
                key=key,
                title=f"Upsample: {key}",
                images={
                    "padded": f"{out_dir}/padded/{key}.png",
                    "infilled": f"{out_dir}/infilled/{key}.png",
                    "upsampled": f"{out_dir}/upsampled/{key}.png",
                    "transparent": f"{out_dir}/upsampled/{key}_transparent.png",
                },
                metadata={"iter_idx": idx},
            )
        )
    return items


def _build_mesh_items(cfg) -> list[ReviewItem]:
    out_dir = cfg.s7_mesh.out_dir
    manifest_dir = f"{out_dir}/manifest"
    if not os.path.isdir(manifest_dir):
        return []

    items: list[ReviewItem] = []
    for fname in sorted(os.listdir(manifest_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = f"{manifest_dir}/{fname}"
        with open(fpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
        key = fname.replace(".json", "")
        items.append(
            ReviewItem(
                key=key,
                title=f"Mesh: {key}",
                images={
                    "input": meta.get("input_img_path"),
                },
                metadata=meta,
            )
        )
    return items


class Reviewer:
    def __init__(self, items: list[ReviewItem], feedback_path: str):
        self.items = items
        self.feedback_path = feedback_path
        self.idx = 0
        self.notes: dict[str, str] = {}
        self.rerun: dict[str, bool] = {}

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.25)

        self.prev_btn = Button(plt.axes([0.10, 0.05, 0.1, 0.06]), "Prev")
        self.next_btn = Button(plt.axes([0.21, 0.05, 0.1, 0.06]), "Next")
        self.save_btn = Button(plt.axes([0.32, 0.05, 0.1, 0.06]), "Save")
        self.flags = CheckButtons(plt.axes([0.45, 0.03, 0.2, 0.1]), ["rerun"], [False])
        self.note_box = TextBox(plt.axes([0.67, 0.05, 0.3, 0.06]), "Note")

        self.prev_btn.on_clicked(self._prev)
        self.next_btn.on_clicked(self._next)
        self.save_btn.on_clicked(self._save)
        self.flags.on_clicked(self._toggle_rerun)
        self.note_box.on_submit(self._set_note)

        self._render()

    def _cur(self) -> ReviewItem:
        return self.items[self.idx]

    def _render(self):
        item = self._cur()
        self.fig.suptitle(item.title)

        image_names = list(item.images.keys())[:3]
        for ax_i in range(3):
            ax = self.axes[ax_i]
            ax.clear()
            if ax_i < len(image_names):
                name = image_names[ax_i]
                img = _read_img(item.images[name])
                ax.set_title(name)
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.axis("off")
            else:
                ax.axis("off")

        self.note_box.set_val(self.notes.get(item.key, ""))
        self.fig.canvas.draw_idle()

    def _set_note(self, text: str):
        self.notes[self._cur().key] = text

    def _toggle_rerun(self, _label: str):
        key = self._cur().key
        self.rerun[key] = not self.rerun.get(key, False)

    def _prev(self, _event):
        self.idx = max(0, self.idx - 1)
        self._render()

    def _next(self, _event):
        self.idx = min(len(self.items) - 1, self.idx + 1)
        self._render()

    def _save(self, _event):
        payload = {
            "notes": self.notes,
            "rerun": {k: v for k, v in self.rerun.items() if v},
            "items": [i.key for i in self.items],
        }
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved feedback to %s", self.feedback_path)

    def show(self):
        plt.show()


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    mode = cfg.stage_review.mode
    if mode == "depth":
        items = _build_depth_items(cfg)
        feedback_path = f"{cfg.s2_da.out_dir if cfg.s2_depth.backend == 'da3' else cfg.s2_fs.out_dir}/stage_review_feedback.json"
    elif mode == "upsample":
        items = _build_upsample_items(cfg)
        feedback_path = f"{cfg.s6_upsample.out_dir}/stage_review_feedback.json"
    elif mode == "mesh":
        items = _build_mesh_items(cfg)
        feedback_path = f"{cfg.s7_mesh.out_dir}/stage_review_feedback.json"
    else:
        raise ValueError(f"Unsupported stage_review.mode: {mode}")

    if not items:
        raise ValueError(f"No items found for mode={mode}")

    Reviewer(items=items, feedback_path=feedback_path).show()


if __name__ == "__main__":
    main()
