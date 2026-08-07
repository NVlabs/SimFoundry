# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import trimesh
import shutil
from simfoundry.utils.python_utils import atomic_copyfile, atomic_output_path
import imageio
import inspect
import tempfile
import numpy as np
from PIL import Image
import os
import torch


def validate_mesh_file(fpath):
    """
    Raises if @fpath is not a usable mesh.

    Parsing is not validity. trimesh.load() only proves the container decoded: a GLB whose
    vertices are all NaN, or which carries no geometry at all, loads without complaint. Publishing
    one marks the object "done", and the failure then surfaces several stages later inside pose
    matching or sim-ready conversion, far from the backend that produced it.

    Args:
        fpath (str): Path to the mesh file to check

    Raises:
        ValueError: If the file has no geometry, no faces, non-finite vertices, or is fully
            degenerate (every vertex coincident)
    """
    scene = trimesh.load(fpath, force="scene")
    geometries = [g for g in scene.geometry.values() if getattr(g, "vertices", None) is not None]
    vertex_blocks = [
        block for block in (np.asarray(g.vertices, dtype=np.float64) for g in geometries)
        if block.size
    ]
    if not vertex_blocks:
        raise ValueError(f"exported mesh contains no vertices: {fpath}")
    if not any(len(getattr(g, "faces", ())) for g in geometries):
        raise ValueError(f"exported mesh contains no faces: {fpath}")
    vertices = np.concatenate(vertex_blocks, axis=0)
    if not np.isfinite(vertices).all():
        n_bad = int((~np.isfinite(vertices)).any(axis=1).sum())
        raise ValueError(
            f"exported mesh has {n_bad} non-finite (NaN/inf) vertex/vertices: {fpath}"
        )
    # Reject only the fully-degenerate case (every vertex at one point). Requiring all three
    # axes to be positive would also reject legitimately planar geometry, and a false rejection
    # here fails the whole object.
    if float((vertices.max(axis=0) - vertices.min(axis=0)).max()) <= 0.0:
        raise ValueError(f"exported mesh is degenerate — zero extent on every axis: {fpath}")


def publish_mesh_atomically(out_fpath, write_fn, validate=True):
    """
    Writes a mesh artifact so that no reader ever observes a partially-written file.

    Consumers treat the presence of the final path as "this object is done": stage 58 streaming
    polls for it to advance, and 3_generate_cousin_meshes.py skips any target that already
    exists. Exporting straight to that path means a crash, an OOM, or a reader arriving mid-write
    leaves a truncated file that is then accepted as a finished mesh.

    Writes to a sibling temporary path (same directory, so the rename stays within one
    filesystem), optionally re-loads it to prove the export completed, then publishes with
    os.replace(), which is atomic. On any failure the temporary file is removed and the
    destination is left untouched.

    Args:
        out_fpath (str): Final path to publish to
        write_fn (callable): Called with the temporary path; must write the artifact there
        validate (bool): Whether to re-load the written file before publishing
    """
    out_fpath = os.path.abspath(out_fpath)
    out_dir = os.path.dirname(out_fpath)
    os.makedirs(out_dir, exist_ok=True)
    # Keep the real extension so exporters that infer format from it still work, and use a
    # prefix that cannot match the `*_mesh.glb` / `*_shape.obj` patterns consumers glob for.
    extension = os.path.splitext(out_fpath)[1] or ".glb"
    fd, tmp_fpath = tempfile.mkstemp(dir=out_dir, prefix=".tmp-publish-", suffix=extension)
    os.close(fd)
    try:
        write_fn(tmp_fpath)
        if validate:
            # A truncated, malformed, empty or NaN-bearing export fails here, while out_fpath
            # still holds whatever it held before (usually nothing).
            validate_mesh_file(tmp_fpath)
        # mkstemp deliberately creates 0600. os.replace preserves the temp file's mode, so
        # without this every published mesh would be owner-only — unreadable to other users on a
        # shared workstation, and to any downstream service or archive step. Restore the 0644
        # that a plain export() would have produced under a standard umask.
        os.chmod(tmp_fpath, 0o644)
        os.replace(tmp_fpath, out_fpath)
    except BaseException:
        try:
            os.remove(tmp_fpath)
        except OSError:
            pass
        raise


class ShapeGenerator:
    """
    Class for shape generation
    """

    def generate_shape(
        self,
        out_fpath,
        image_path=None,
        prompt=None,
        visualize=False,
        **kwargs,
    ):
        """
        Generates untextured mesh shape based on inputs

        Args:
            out_fpath: Absolute output file path to write generated shape .glb file to
            image_path (None or str): Absolute path to image to condition generation, if any
            prompt (None or str): Text prompt to condition generation, if any
            visualize (bool): Whether to visualize generated shape
            kwargs (Any): Any additional arguments to pass to generation call
        """
        raise NotImplementedError


class TextureGenerator:
    """
    Class for texture generation
    """

    def generate_texture(
        self,
        shape_fpath,
        out_fpath,
        image_path=None,
        prompt=None,
        visualize=False,
        **kwargs,
    ):
        """
        Generates textured mesh shape based on input shape and other inputs

        Args:
            shape_fpath: Absolute path to shape file to condition texture generation
            out_fpath: Absolute output file path to write generated textured shape .glb file to
            image_path (None or str): Absolute path to image to condition generation, if any
            prompt (None or str): Text prompt to condition generation, if any
            visualize (bool): Whether to visualize generated shape
            kwargs (Any): Any additional arguments to pass to generation call
        """
        raise NotImplementedError


class MeshGenerator(ShapeGenerator, TextureGenerator):
    """
    Class for shape + texture generation
    """
    def __init__(
            self,
            create_shape_pipeline=True,
            create_texture_pipeline=True,
            low_vram=False,
    ):
        """
        Args:
            create_shape_pipeline (bool): Whether to create shape pipeline model
            create_texture_pipeline (bool): Whether to create texture pipeline model
            low_vram (bool): If True, enable model CPU offloading to reduce GPU VRAM usage.
                Subclasses may use this flag to move sub-models to CPU when not in use.
        """
        self.low_vram = low_vram
        self.create_pipelines(
            create_shape_pipeline=create_shape_pipeline,
            create_texture_pipeline=create_texture_pipeline,
        )

        super().__init__()

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):
        raise NotImplementedError

    def generate_mesh(
        self,
        out_fpath,
        shape_image_path=None,
        shape_prompt=None,
        shape_kwargs=None,
        texture_image_path=None,
        texture_prompt=None,
        texture_kwargs=None,
        visualize=False,
        save_intermediates=False,
        **kwargs,
    ):
        """
        Generates a fully textured mesh

        Args:
            out_fpath: Absolute output file path to write generated shape .glb file to
            shape_image_path (None or str): Absolute path to image to condition generation, if any
            shape_prompt (None or str): Text prompt to condition shape generation, if any
            shape_kwargs (None or dict): Any additional arguments to pass to shape generation call
            texture_image_path (None or str): Absolute path to image to condition texture generation, if any
            texture_prompt (None or str): Text prompt to condition texture generation, if any
            texture_kwargs (None or dict): Any additional arguments to pass to texture generation call
            visualize (bool): Whether to visualize generated shape
            kwargs (Any): Any additional arguments to pass to generation call
        """
        # By default, simply run shape and texture generation sequentially
        assert out_fpath.endswith(".glb")
        shape_fpath = out_fpath.replace(".glb", "_untextured.glb")
        shape_kwargs = dict() if shape_kwargs is None else shape_kwargs
        texture_kwargs = dict() if texture_kwargs is None else texture_kwargs

        # Route loose **kwargs (i.e. config-supplied generation options) to whichever half accepts
        # them. Without this they are silently swallowed here: this signature's **kwargs makes
        # every name look accepted to filter_generation_kwargs, so nothing is reported as dropped,
        # yet only shape_kwargs/texture_kwargs were ever forwarded. That is the default path —
        # Hunyuan does not override generate_mesh.
        if kwargs:
            shape_extra = filter_generation_kwargs(
                self.generate_shape, kwargs, reserved=RESERVED_GENERATION_KWARGS["generate_shape"]
            )
            texture_extra = filter_generation_kwargs(
                self.generate_texture, kwargs, reserved=RESERVED_GENERATION_KWARGS["generate_texture"]
            )
            shape_kwargs = {**shape_extra, **shape_kwargs}
            texture_kwargs = {**texture_extra, **texture_kwargs}
            unused = sorted(set(kwargs) - set(shape_kwargs) - set(texture_kwargs))
            if unused:
                print(
                    f"WARNING: {type(self).__name__} accepts neither shape nor texture argument(s) "
                    f"{unused} - dropped from generation_kwargs"
                )
        self.generate_shape(
            out_fpath=shape_fpath,
            image_path=shape_image_path,
            prompt=shape_prompt,
            visualize=visualize,
            **shape_kwargs,
        )

        torch.cuda.empty_cache()

        self.generate_texture(
            shape_fpath=shape_fpath,
            out_fpath=out_fpath,
            image_path=texture_image_path,
            prompt=texture_prompt,
            visualize=visualize,
            **texture_kwargs,
        )


class Hunyuan(MeshGenerator):

    REPO_PATH = None

    def __init__(
        self,
        create_shape_pipeline=True,
        create_texture_pipeline=True,
        max_num_views=6,
        resolution=512,
        low_vram=False,
    ):
        """
        Args:
            create_shape_pipeline (bool): Whether to create shape pipeline model
            create_texture_pipeline (bool): Whether to create texture pipeline model
            max_num_views (int): Maximum number of views to use during texture generation
            resolution (int): Resolution of generated images during texture generation
            low_vram (bool): If True, enable model CPU offloading to reduce GPU VRAM usage.
                Moves each sub-model (conditioner, DiT, VAE) to GPU only when needed,
                significantly reducing peak memory at a small speed cost.
        """
        self.shape_pipeline = None
        self.texture_pipeline = None
        self.max_num_views = max_num_views
        self.resolution = resolution
        super().__init__(
            create_shape_pipeline=create_shape_pipeline,
            create_texture_pipeline=create_texture_pipeline,
            low_vram=low_vram,
        )

    @classmethod
    def set_repo_path(cls, repo_path):
        # Record only. sys.path is mutated in create_pipelines, i.e. when this backend is
        # actually instantiated: stage scripts call set_repo_path for EVERY backend at import
        # time, and hy3dpaint contains a top-level regular package literally named `src`. A
        # regular package shadows a namespace package anywhere on sys.path, so eagerly adding it
        # broke Pixal3D's NAF loader (`from src.model.naf import NAF`) with
        # "No module named 'src.model'" even when hunyuan was not the selected backend.
        cls.REPO_PATH = repo_path

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):
        # Local import now to avoid dependency crashing depending on environment being run currently
        assert self.REPO_PATH is not None, f"Must set absolute path {self.__class__.__name__}'s repo path via set_repo_path()!"
        import sys
        for sub in ("hy3dshape", "hy3dpaint"):
            path = f"{self.REPO_PATH}/{sub}"
            if path not in sys.path:
                sys.path.insert(0, path)
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        if create_shape_pipeline:
            self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
            self.shape_pipeline.enable_flashvdm(mc_algo='mc')
            if self.low_vram:
                self.shape_pipeline.enable_model_cpu_offload()

        if create_texture_pipeline:
            cfg = Hunyuan3DPaintConfig(
                max_num_view=self.max_num_views,
                resolution=self.resolution,
            )
            cfg.realesrgan_ckpt_path = f"{self.REPO_PATH}/{cfg.realesrgan_ckpt_path}"
            cfg.multiview_cfg_path = f"{self.REPO_PATH}/{cfg.multiview_cfg_path}"
            cfg.custom_pipeline = f"{self.REPO_PATH}/{cfg.custom_pipeline}"
            self.texture_pipeline = Hunyuan3DPaintPipeline(config=cfg)

    def generate_shape(
        self,
        out_fpath,
        image_path=None,
        prompt=None,
        visualize=False,
        num_inference_steps=50,
        timesteps=None,
        sigmas=None,
        eta=0.0,
        guidance_scale=5.0, #5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384, #384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type="trimesh",
        enable_pbar=True,
        mask=None,
    ):
        assert self.shape_pipeline is not None, f"shape pipeline was not created for {self.__class__.__name__}!"
        assert prompt is None, f"Cannot use prompt when generating shape for {self.__class__.__name__}!"
        # Forward the parameters rather than re-stating their defaults as literals: the previous
        # form accepted every argument above and then ignored all of them, so callers (and now
        # s7_mesh.generation_kwargs) could not change anything. The values below are identical to
        # the signature defaults, so default behavior is unchanged.
        mesh_untextured = self.shape_pipeline(
            image=image_path,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            eta=eta,
            guidance_scale=guidance_scale,
            generator=generator,
            box_v=box_v,
            octree_resolution=octree_resolution,
            mc_level=mc_level,
            mc_algo=mc_algo,
            num_chunks=num_chunks,
            output_type=output_type,
            enable_pbar=enable_pbar,
            mask=mask,
        )[0]
        if visualize:
            mesh_untextured.show()
        with atomic_output_path(out_fpath) as _tmp_out:
            mesh_untextured.export(_tmp_out)

    def generate_texture(
        self,
        shape_fpath,
        out_fpath,
        image_path=None,
        prompt=None,
        visualize=False,
        use_remesh=True,
    ):
        assert self.texture_pipeline is not None, f"texture pipeline was not created for {self.__class__.__name__}!"
        assert prompt is None, f"Cannot use prompt when generating texture for {self.__class__.__name__}!"
        assert out_fpath.endswith(".glb"), f"out_fpath must end with .glb, got: {out_fpath}"
        textured_mesh_path = self.texture_pipeline(shape_fpath, image_path=image_path, use_remesh=use_remesh)
        textured_mesh_path_glb = textured_mesh_path.replace(".obj", ".glb")
        # Published atomically: the streaming watcher keys on this filename, and a
        # multi-tens-of-MB copy is long enough for a consumer to open it half-written.
        atomic_copyfile(textured_mesh_path_glb, out_fpath)
        if visualize:
            tm = trimesh.load(textured_mesh_path_glb)
            tm.show()


class Direct3D(ShapeGenerator):
    def __init__(self):
        # Local import now to avoid dependency crashing depending on environment being run currently
        from direct3d_s2.pipeline import Direct3DS2Pipeline

        # Create pipeline
        self.pipeline = Direct3DS2Pipeline.from_pretrained(
          'wushuang98/Direct3D-S2',
          subfolder="direct3d-s2-v-1-1"
        )
        self.pipeline.to("cuda")

        super().__init__()

    def generate_shape(
        self,
        out_fpath,
        image_path=None,
        prompt=None,
        visualize=False,
        sdf_resolution=1024,
        remove_interior=True,
        remesh=True,
    ):
        mesh = self.pipeline(
            image_path,
            sdf_resolution=sdf_resolution,  # 512 or 1024
            remove_interior=remove_interior,
            remesh=remesh,  # Switch to True if you need to reduce the number of triangles.
        )["mesh"]
        if visualize:
            mesh.show()
        with atomic_output_path(out_fpath) as _tmp_out:
            mesh.export(_tmp_out)



class Trellis(MeshGenerator):

    REPO_PATH = None

    def __init__(
        self,
        create_shape_pipeline=True,
        create_texture_pipeline=True,
        low_vram=False,
    ):
        """
        Args:
            create_shape_pipeline (bool): Whether to create shape pipeline model
            create_texture_pipeline (bool): Whether to create texture pipeline model
            low_vram (bool): If True, enable model CPU offloading to reduce GPU VRAM usage.
        """
        self.pipeline = None

        super().__init__(
            create_shape_pipeline=create_shape_pipeline,
            create_texture_pipeline=create_texture_pipeline,
            low_vram=low_vram,
        )

    @classmethod
    def set_repo_path(cls, repo_path):
        # Record only; sys.path is mutated in create_pipelines. See Hunyuan.set_repo_path.
        cls.REPO_PATH = repo_path

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):
        # Local import now to avoid dependency crashing depending on environment being run currently
        import sys
        if self.REPO_PATH is not None and self.REPO_PATH not in sys.path:
            sys.path.insert(0, self.REPO_PATH)
        from trellis.pipelines import TrellisImageTo3DPipeline
        os.environ['SPCONV_ALGO'] = 'native'

        assert create_shape_pipeline and create_texture_pipeline, f"create_shape_pipeline and create_texture_pipeline must both be True!"

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        self.pipeline.cuda()

    def generate_mesh(
        self,
        out_fpath,
        shape_image_path=None,
        shape_prompt=None,
        shape_kwargs=None,
        texture_image_path=None,
        texture_prompt=None,
        texture_kwargs=None,
        visualize=False,
        simplify=0.5,
        texture_size=1024,
        save_intermediates=False,
        seed=1,
        **kwargs,
    ):
        """
        Generates a fully textured mesh

        Args:
            out_fpath: Absolute output file path to write generated shape .glb file to
            shape_image_path (None or str): Absolute path to image to condition generation, if any
            shape_prompt (None or str): Text prompt to condition shape generation, if any
            shape_kwargs (None or dict): Any additional arguments to pass to shape generation call
            texture_image_path (None or str): Absolute path to image to condition texture generation, if any
            texture_prompt (None or str): Text prompt to condition texture generation, if any
            texture_kwargs (None or dict): Any additional arguments to pass to texture generation call
            visualize (bool): Whether to visualize generated shape
            simplify (float): Ratio of triangles to remove in the simplification process
            texture_size (int): Size of the texture used for the output GLB
            save_intermediates (bool): Whether to save intermediate files, i.e.: GS / radiance field / mesh videos
            seed (int): Seed to use for mesh generation
            kwargs (Any): Any additional arguments to pass to generation call
        """
        assert shape_image_path == texture_image_path, f"shape_image_path and texture_image_path must both be identical!"
        assert shape_prompt == texture_prompt, f"shape_prompt and texture_prompt must both be identical!"
        assert out_fpath.endswith(".glb"), f"out_fpath must end with .glb!"

        from trellis.utils import render_utils, postprocessing_utils

        # Run pipeline
        image = Image.open(shape_image_path)
        outputs = self.pipeline.run(
            image,
            seed=seed,
        )
        if save_intermediates:
            # Render the outputs
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(out_fpath.replace(".glb", "_vis_gs.mp4"), video, fps=30)
            video = render_utils.render_video(outputs['radiance_field'][0])['color']
            imageio.mimsave(out_fpath.replace(".glb", "_vis_rf.mp4"), video, fps=30)
            video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(out_fpath.replace(".glb", "_vis_mesh.mp4"), video, fps=30)
            outputs['gaussian'][0].save_ply(out_fpath.replace(".glb", "_pc.ply"))

        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=simplify,  # 0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=texture_size,  # Size of the texture used for the GLB
        )
        if visualize:
            glb.show()
        with atomic_output_path(out_fpath) as _tmp_out:
            glb.export(_tmp_out)


class Trellis2(MeshGenerator):
    REPO_PATH = None

    # TODO: keep for backward compatibility
    # TODO: TRELLIS2 can generate both shape and texture in one pipeline, or just textures, implement this later
    def __init__(self, create_shape_pipeline=True, create_texture_pipeline=True, low_vram=False):
        super().__init__(create_shape_pipeline, create_texture_pipeline, low_vram=low_vram)

    @classmethod
    def set_repo_path(cls, repo_path):
        # Record only; sys.path is mutated in create_pipelines. See Hunyuan.set_repo_path.
        cls.REPO_PATH = repo_path
        import os

        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):
        import sys
        if self.REPO_PATH is not None and self.REPO_PATH not in sys.path:
            sys.path.insert(0, self.REPO_PATH)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.renderers import EnvMap
        import torch
        import cv2

        # TODO: add option for texture generation pipeline
        self.envmap = EnvMap(torch.tensor(
                        cv2.cvtColor(cv2.imread(f'{self.REPO_PATH}/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                        dtype=torch.float32, device='cuda'
                    ))
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        self.pipeline.cuda()


    def generate_mesh(
    self,
    out_fpath,
    shape_image_path=None,
    shape_prompt=None,
    shape_kwargs=None,
    texture_image_path=None,
    texture_prompt=None,
    texture_kwargs=None,
    visualize=False,
    simplify=0.5,
    texture_size=4096,
    save_intermediates=False,
    seed=42,
    **kwargs,
    ):
        assert self.pipeline is not None, f"pipeline was not created for {self.__class__.__name__}!"
        """
        Generates a fully textured mesh

        Args:
            out_fpath: Absolute output file path to write generated shape .glb file to
            shape_image_path (None or str): Absolute path to image to condition generation, if any
            shape_prompt (None or str): Text prompt to condition shape generation, if any
            shape_kwargs (None or dict): Any additional arguments to pass to shape generation call
            texture_image_path (None or str): Absolute path to image to condition texture generation, if any
            texture_prompt (None or str): Text prompt to condition texture generation, if any
            texture_kwargs (None or dict): Any additional arguments to pass to texture generation call
            visualize (bool): Whether to visualize generated shape
            simplify (float): Ratio of triangles to remove in the simplification process
            texture_size (int): Size of the texture used for the output GLB
            save_intermediates (bool): Whether to save intermediate files, i.e.: GS / radiance field / mesh videos
            seed (int): Seed to use for mesh generation
            kwargs (Any): Any additional arguments to pass to generation call
        """

        assert shape_image_path == texture_image_path, f"shape_image_path and texture_image_path must both be identical!"
        assert shape_prompt == texture_prompt, f"shape_prompt and texture_prompt must both be identical!"
        assert out_fpath.endswith(".glb"), f"out_fpath must end with .glb!"

        image = Image.open(shape_image_path)

        # Build sampler params from kwargs.
        _PARAM_RENAME = {
            "sampling_steps": "steps",
        }
        ss_sampler_params = {}
        shape_slat_sampler_params = {}
        tex_slat_sampler_params = {}
        _SAMPLER_PARAM_MAP = {
            "ss_": ss_sampler_params,
            "shape_slat_": shape_slat_sampler_params,
            "tex_slat_": tex_slat_sampler_params,
        }
        for key, value in kwargs.items():
            for prefix, target_dict in _SAMPLER_PARAM_MAP.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    param_name = _PARAM_RENAME.get(param_name, param_name)
                    target_dict[param_name] = value
                    break

        pipeline_type = kwargs.get("pipeline_type", None)
        resolution = kwargs.get("resolution", None)
        # Map resolution string to pipeline_type if provided
        if resolution is not None and pipeline_type is None:
            res = int(resolution)
            pipeline_type = {512: "512", 1024: "1024", 1536: "1536_cascade"}.get(res, f"{res}_cascade")

        run_kwargs = dict(seed=seed)
        if ss_sampler_params:
            run_kwargs["sparse_structure_sampler_params"] = ss_sampler_params
        if shape_slat_sampler_params:
            run_kwargs["shape_slat_sampler_params"] = shape_slat_sampler_params
        if tex_slat_sampler_params:
            run_kwargs["tex_slat_sampler_params"] = tex_slat_sampler_params
        if pipeline_type is not None:
            run_kwargs["pipeline_type"] = pipeline_type
        # max_num_tokens controls detail level in cascade pipeline (higher = more detail, including internal structures)
        if "max_num_tokens" in kwargs:
            run_kwargs["max_num_tokens"] = kwargs["max_num_tokens"]

        mesh = self.pipeline.run(image, **run_kwargs)[0]
        mesh.simplify(16777216) # nvdiffrast limit

        # Render Video
        if save_intermediates:
            from trellis2.utils import render_utils
            video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=self.envmap))
            object_name = os.path.basename(out_fpath).replace(".glb", "")
            renders_dir = os.path.join(os.path.dirname(out_fpath), "renders")
            os.makedirs(renders_dir, exist_ok=True)
            imageio.mimsave(f"{renders_dir}/{object_name}.mp4", video, fps=15)

        num_faces = kwargs.get('num_faces', 300000)
        # Export to GLB
        import o_voxel
        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   num_faces, #default is 1000000
            texture_size        =   texture_size,
            remesh              =   kwargs.get('remesh', True),
            remesh_band         =   kwargs.get('remesh_band', 1.0),
            remesh_project      =   kwargs.get('remesh_project', 0.9),  # Project vertices back to original surface (0.9 = 90% snap back)
            verbose             =   kwargs.get('verbose', True)
        )

        if visualize:
            glb.show()
        with atomic_output_path(out_fpath) as _tmp_out:
            glb.export(_tmp_out)


# Upstream Pixal3D defines exactly these cascade variants
# (deps/Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py run()).
_PIXAL3D_PIPELINE_TYPES = ("1024_cascade", "1536_cascade")


class _RejectingBackgroundRemover:
    """
    Stand-in for Pixal3D's BiRefNet background remover that downloads and runs nothing.

    Pixal3D's published `pipeline.json` selects `briaai/RMBG-2.0`, and
    `Pixal3DImageTo3DPipeline.from_pretrained` constructs it eagerly (pixal3d_image_to_3d.py:123)
    — downloading gated, CC-BY-NC-4.0 weights and executing remote code — before anything checks
    whether the input even needs background removal.

    Every SimFoundry caller feeds RGBA cutouts (stage 6 `_transparent.png`, cousin PNGs), which
    take upstream's `has_alpha` branch and never touch this model. Substituting this class keeps
    that path working while making the non-commercial dependency unreachable rather than merely
    unused. If a caller ever does supply an image without alpha, they get this explanation
    instead of a silent gated download.

    Supports the attribute surface the pipeline touches: `.to()` / `.cpu()` during device moves.
    """

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name") or (args[0] if args else "<unknown>")

    def to(self, *args, **kwargs):
        return self

    def cpu(self, *args, **kwargs):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def eval(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Pixal3D was asked to remove an image background using '{self.model_name}', which "
            f"SimFoundry deliberately does not load: it is gated and carries non-commercial "
            f"(CC-BY-NC-4.0) terms. Supply an RGBA image whose alpha channel already isolates the "
            f"object — stage 6 writes exactly that as `*_transparent.png`. "
            f"See THIRD_PARTY_LICENSES.md item 13e."
        )


class Pixal3D(MeshGenerator):
    """
    Pixal3D (TencentARC) single-image -> textured-mesh backend.

    https://github.com/TencentARC/Pixal3D (SIGGRAPH 2026, arXiv 2605.10922, MIT licensed)

    Geometry and PBR texture come out of one 3-stage cascade (sparse structure -> shape SLat ->
    texture SLat), so this overrides generate_mesh and leaves generate_shape / generate_texture
    unimplemented. It must be selected as BOTH shape_model and texture_model.

    ORIENTATION: Pixal3D is *pixel-aligned* — it generates in the input view's frame rather than
    a canonical frame, unlike hunyuan/trellis2. Validate against stage 8 pose matching before
    trusting a full scene run.

    GATED WEIGHTS: DINOv3 is the only gated model this backend downloads, and it is repointed
    below from upstream's ungated third-party mirror to Meta's official repo (accepted DINOv3
    terms + `hf auth login` required; those terms carry their own conditions relevant to
    deployment and distribution). Pixal3D's published pipeline.json also selects
    `briaai/RMBG-2.0` (trust_remote_code=True, CC-BY-NC-4.0) for background removal, but
    create_pipelines() substitutes _RejectingBackgroundRemover before the pipeline is built, so
    this integration does not download or execute BRIA code or weights. RGBA inputs whose alpha
    already isolates the object are required instead. See THIRD_PARTY_LICENSES.md items 13b/13e.
    """

    REPO_PATH = None
    MODEL_PATH = "TencentARC/Pixal3D"           # inference.py:MODEL_PATH
    MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"    # inference.py:MOGE_MODEL_NAME
    # Upstream inference.py hardcodes `camenduru/dinov3-...`, an ungated third-party mirror that
    # routes around Meta's manual-approval gate for DINOv3. Point at the official repo instead;
    # requires `hf auth login` plus accepted DINOv3 terms, as SAM 3 already does.
    DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    # Directory holding revision-pinned local snapshots written by install_pixal3d.sh, one
    # subdirectory per model. Loading from these instead of a bare repo id matters for more than
    # reproducibility: neither Pixal3D's from_pretrained (deps/Pixal3D/pixal3d/pipelines/base.py)
    # nor the DINOv3/MoGe loaders accept a `revision`, so a bare repo id silently resolves to
    # whatever that repo's main branch holds at first run. In particular the pipeline.json that
    # selects the background remover is remote state — with a local snapshot it cannot change
    # under us between the substitution below and the assert that checks it held.
    WEIGHTS_DIR_ENV_VAR = "SIMFOUNDRY_PIXAL3D_WEIGHTS_DIR"
    # Loading an unpinned model is a provenance failure, not a style preference: the weights, and
    # the pipeline.json that selects the background remover, are mutable remote state. Refuse by
    # default and require an explicit opt-in, so a clean node or container cannot silently
    # resolve HEAD-of-main instead of the audited revision.
    ALLOW_UNPINNED_ENV_VAR = "SIMFOUNDRY_PIXAL3D_ALLOW_UNPINNED"
    SNAPSHOT_SUBDIRS = {
        "pixal3d": "Pixal3D",
        "dinov3": "dinov3-vitl16-pretrain-lvd1689m",
        "moge": "moge-2-vitl",
    }
    # Loaders disagree about what a "local model" is. Pixal3D's from_pretrained looks for
    # pipeline.json in a directory and transformers accepts a directory, but MoGe's
    # from_pretrained takes a checkpoint FILE (or a repo id it downloads model.pt from) and
    # torch.load()s the path directly — handing it a directory raises IsADirectoryError.
    SNAPSHOT_FILES = {
        "moge": "model.pt",
    }
    # Revisions install_pixal3d.sh pins and records in each snapshot's `.simfoundry-revision`.
    # KEEP IN SYNC with *_MODEL_REVISION in that script. Existence alone is not integrity: an
    # interrupted `hf download`, a hand-edited snapshot, or a bumped pin downloaded over an older
    # tree all leave a directory that looks installed. The marker is written last, so matching it
    # is what distinguishes a complete, expected snapshot from a plausible-looking one.
    SNAPSHOT_REVISIONS = {
        "pixal3d": "0b31f9160aa400719af409098bff7936a932f726",
        "moge": "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
        "dinov3": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    }
    REVISION_MARKER = ".simfoundry-revision"
    # Upstream loads NAF with an UNREVISIONED torch.hub.load("valeoai/NAF", trust_repo=True)
    # (deps/Pixal3D/.../image_conditioned_proj.py:_load_naf). Left alone, that clones and executes
    # whatever GitHub serves at call time. install_pixal3d.sh pre-populates a pinned, checksummed
    # copy, but only in the invoking user's torch hub dir - a different user, a changed TORCH_HOME,
    # or a cleared cache silently restores the live-fetch behavior. create_pipelines redirects the
    # call to a local checkout and fails closed when none is present.
    NAF_DIR_ENV_VAR = "SIMFOUNDRY_NAF_DIR"
    NAF_HUB_DIRNAME = "valeoai_NAF_main"

    @classmethod
    def resolve_model_source(cls, key, repo_id):
        """
        Returns a local pinned snapshot path for @key if one was installed, else @repo_id.

        Args:
            key (str): Key into SNAPSHOT_SUBDIRS
            repo_id (str): Hugging Face repo id to fall back to

        Returns:
            str: A local directory path, or @repo_id when no snapshot is present
        """
        weights_dir = os.environ.get(cls.WEIGHTS_DIR_ENV_VAR)
        if not weights_dir and cls.REPO_PATH is not None:
            # Default layout written by install_pixal3d.sh, alongside the checkout in deps/.
            weights_dir = os.path.join(os.path.dirname(os.path.abspath(cls.REPO_PATH)), "pixal3d-weights")
        if not weights_dir:
            return repo_id
        candidate = os.path.join(weights_dir, cls.SNAPSHOT_SUBDIRS[key])
        if not os.path.isdir(candidate):
            return repo_id

        # A snapshot only counts as pinned if it carries the revision we expect. Otherwise treat
        # it as absent, so the caller's unpinned-weights guard decides rather than this silently
        # loading whatever happens to be on disk.
        expected = cls.SNAPSHOT_REVISIONS.get(key)
        marker_path = os.path.join(candidate, cls.REVISION_MARKER)
        if expected is not None:
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    found = f.read().strip()
            except OSError:
                print(
                    f"WARNING: {candidate} has no {cls.REVISION_MARKER}; treating it as not "
                    f"installed. An interrupted download leaves exactly this state — re-run "
                    f"install_pixal3d.sh, or delete the directory first to force a clean fetch."
                )
                return repo_id
            if found != expected:
                print(
                    f"WARNING: {candidate} is at revision {found or '<empty>'} but this build "
                    f"expects {expected}; treating it as not installed. Delete the directory and "
                    f"re-run install_pixal3d.sh (it will not re-download over a stale tree)."
                )
                return repo_id

        inner_file = cls.SNAPSHOT_FILES.get(key)
        if inner_file is not None:
            inner_path = os.path.join(candidate, inner_file)
            # Fall back to the repo id rather than handing over a path the loader will choke on.
            return inner_path if os.path.isfile(inner_path) else repo_id
        return candidate

    def __init__(self, create_shape_pipeline=True, create_texture_pipeline=True, low_vram=False):
        """
        Args:
            create_shape_pipeline (bool): Must be True; shape and texture share one pipeline
            create_texture_pipeline (bool): Must be True; shape and texture share one pipeline
            low_vram (bool): If True, keep sub-models on CPU and page them to GPU per stage,
                reducing peak VRAM from ~18GB to ~10-12GB at the cost of speed
        """
        if not (create_shape_pipeline and create_texture_pipeline):
            # Fail here rather than part-way through the stage loop with a bare NotImplementedError
            # from TextureGenerator.generate_texture, after a multi-minute model load.
            raise ValueError(
                f"{self.__class__.__name__} generates shape and texture in a single pipeline and "
                f"cannot be split across models. Set shape_model and texture_model both to 'pixal3d'."
            )
        # Must precede super().__init__(), which calls create_pipelines() before returning.
        self.pipeline = None
        self.moge_model = None
        self._inference = None
        super().__init__(
            create_shape_pipeline=create_shape_pipeline,
            create_texture_pipeline=create_texture_pipeline,
            low_vram=low_vram,
        )

    @classmethod
    def set_repo_path(cls, repo_path):
        import sys
        # Stage modules call this at import time in every mamba env, so an unconditional
        # sys.path.insert would make `pixal3d` / `inference` importable in envs that cannot
        # actually run them. deps/ is shared across envs; only wire it up if really present.
        if not os.path.isdir(repo_path):
            return
        # Record only; sys.path is mutated in create_pipelines. See Hunyuan.set_repo_path.
        cls.REPO_PATH = repo_path

    @classmethod
    def resolve_naf_dir(cls):
        """
        Returns a local pinned NAF checkout containing hubconf.py, or None if none is installed.

        Search order: $SIMFOUNDRY_NAF_DIR, the weights dir install_pixal3d.sh writes, then torch's
        hub cache (where the installer also places a pinned, checksummed copy).
        """
        import torch.hub

        candidates = []
        env_dir = os.environ.get(cls.NAF_DIR_ENV_VAR)
        if env_dir:
            candidates.append(env_dir)
        if cls.REPO_PATH is not None:
            weights_dir = os.environ.get(cls.WEIGHTS_DIR_ENV_VAR) or os.path.join(
                os.path.dirname(os.path.abspath(cls.REPO_PATH)), "pixal3d-weights"
            )
            candidates.append(os.path.join(weights_dir, "NAF"))
        candidates.append(os.path.join(torch.hub.get_dir(), cls.NAF_HUB_DIRNAME))

        for candidate in candidates:
            if os.path.isfile(os.path.join(candidate, "hubconf.py")):
                return candidate
        return None

    def _guarded_hub_load(self, original_load):
        """
        Builds a torch.hub.load replacement that never reaches the network for NAF.

        Calls for any other repo are delegated to @original_load untouched.
        """
        import torch.hub

        naf_dir = self.resolve_naf_dir()
        env_var = self.NAF_DIR_ENV_VAR

        def guarded_load(repo_or_dir, model, *args, **kwargs):
            if "NAF" not in str(repo_or_dir):
                return original_load(repo_or_dir, model, *args, **kwargs)
            if naf_dir is None:
                raise RuntimeError(
                    "Pixal3D needs the NAF feature upsampler, and SimFoundry will not let it "
                    "clone and execute unpinned code from GitHub at generation time. No pinned "
                    "checkout was found - run scripts/installation/install_pixal3d.sh without "
                    f"--skip-weights, or point {env_var} at a checkout containing hubconf.py. "
                    "NOTE: that checkout vendors src/layers/rope.py under Meta's DINOv3 License; "
                    "see THIRD_PARTY_LICENSES.md item 13c."
                )
            # Meaningless (and rejected) for a local-directory load.
            kwargs.pop("trust_repo", None)
            kwargs.pop("source", None)
            print(f"[Pixal3D] NAF loaded from pinned local checkout: {naf_dir}")
            return torch.hub._load_local(naf_dir, model, *args, **kwargs)

        return guarded_load

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):
        # Local import now to avoid dependency crashing depending on environment being run currently
        assert self.REPO_PATH is not None, (
            f"Must set absolute path to {self.__class__.__name__}'s repo via set_repo_path()! "
            f"Install it with scripts/installation/install_pixal3d.sh"
        )
        import importlib.util
        import sys

        if self.REPO_PATH not in sys.path:
            sys.path.insert(0, self.REPO_PATH)

        # setdefault rather than assignment: ATTN_BACKEND is read by TRELLIS/TRELLIS.2 too, and
        # PYTORCH_CUDA_ALLOC_CONF may have been tuned by the caller.
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("ATTN_BACKEND", "flash_attn")  # set ATTN_BACKEND=sdpa if no flash-attn

        # Pixal3D ships no setup.py; its entry point is a repo-root inference.py. Load it by path
        # under a distinct name, since `inference` is too generic to claim as a top-level module.
        spec = importlib.util.spec_from_file_location(
            "pixal3d_inference", os.path.join(self.REPO_PATH, "inference.py")
        )
        self._inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._inference)

        # Prefer revision-pinned local snapshots when install_pixal3d.sh wrote them.
        model_source = self.resolve_model_source("pixal3d", self.MODEL_PATH)
        dinov3_source = self.resolve_model_source("dinov3", self.DINOV3_REPO)
        moge_source = self.resolve_model_source("moge", self.MOGE_MODEL_NAME)
        unpinned = [
            (label, repo_id)
            for label, source, repo_id in (
                ("Pixal3D", model_source, self.MODEL_PATH),
                ("DINOv3", dinov3_source, self.DINOV3_REPO),
                ("MoGe-2", moge_source, self.MOGE_MODEL_NAME),
            )
            if source == repo_id
        ]
        if unpinned:
            detail = ", ".join(f"{label} ('{repo_id}')" for label, repo_id in unpinned)
            if os.environ.get(self.ALLOW_UNPINNED_ENV_VAR) != "1":
                # Fail closed. Previously this only warned and carried on, so a host without
                # snapshots quietly resolved whatever each repo's main branch held at first run —
                # including the pipeline.json that decides the background remover.
                raise RuntimeError(
                    f"No pinned local snapshot found for: {detail}. Loading these would resolve "
                    f"mutable Hugging Face repos at runtime, so the run would not be "
                    f"reproducible and the audited revision is not guaranteed. Install the "
                    f"snapshots with scripts/installation/install_pixal3d.sh (or point "
                    f"{self.WEIGHTS_DIR_ENV_VAR} at an existing weights directory). To accept "
                    f"unpinned weights anyway — development only — set "
                    f"{self.ALLOW_UNPINNED_ENV_VAR}=1."
                )
            print(
                f"WARNING: {self.ALLOW_UNPINNED_ENV_VAR}=1 — loading {detail} from mutable "
                f"Hugging Face repos. Results are not reproducible across runs."
            )

        for cond_cfg in self._inference.IMAGE_COND_CONFIGS.values():
            cond_cfg["model_name"] = dinov3_source

        # Swap the background remover out *before* the pipeline is built. from_pretrained does
        # `getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])`
        # (pixal3d_image_to_3d.py:123), so by the time we could null the attribute the gated,
        # non-commercial briaai/RMBG-2.0 weights have already been downloaded and its remote code
        # executed. Patching the factory is the only point at which that is avoidable.
        import pixal3d.pipelines.rembg as pixal3d_rembg

        # NAF is fetched during init_pipeline (low_vram pre-loads the upsampler) and again lazily
        # on first use, so the guard has to cover both. Patch torch.hub.load for the duration.
        import torch.hub

        original_birefnet = pixal3d_rembg.BiRefNet
        original_hub_load = torch.hub.load
        pixal3d_rembg.BiRefNet = _RejectingBackgroundRemover
        torch.hub.load = self._guarded_hub_load(original_hub_load)
        try:
            # init_pipeline, not Pixal3DImageTo3DPipeline.from_pretrained: the latter leaves the
            # four image_cond_model_* attributes as None and run() then asserts on them. Also not
            # pixal3d.pipelines.from_pretrained, whose pipeline.json dispatches to the wrong class.
            self.pipeline = self._inference.init_pipeline(
                model_source, device="cuda", low_vram=self.low_vram
            )
        finally:
            pixal3d_rembg.BiRefNet = original_birefnet
            torch.hub.load = original_hub_load

        # RuntimeError, not assert: `python -O` strips asserts, and this is the control that
        # keeps gated, non-commercial (CC-BY-NC-4.0) BRIA weights from being downloaded and their
        # remote code executed. It must hold under every interpreter flag.
        if not isinstance(self.pipeline.rembg_model, _RejectingBackgroundRemover):
            raise RuntimeError(
                "Pixal3D constructed a real background-removal model despite the patch above. "
                "Upstream's rembg wiring has changed; re-check "
                "pixal3d/pipelines/pixal3d_image_to_3d.py before running, or gated "
                "non-commercial weights may be downloaded. See THIRD_PARTY_LICENSES.md item 13e."
            )

        # MoGe-2 estimates the input FOV that run() requires. Held on the instance so the stage
        # loop does not reload it per object, but parked on CPU: upstream frees it before
        # generation, and the ~18GB / ~10-12GB peaks assume it is not resident.
        self.moge_model = self._inference.load_moge_model("cpu", moge_source)

    def generate_mesh(
        self,
        out_fpath,
        shape_image_path=None,
        shape_prompt=None,
        shape_kwargs=None,
        texture_image_path=None,
        texture_prompt=None,
        texture_kwargs=None,
        visualize=False,
        texture_size=4096,
        save_intermediates=False,
        seed=42,
        **kwargs,
    ):
        """
        Generates a fully textured mesh

        Args:
            out_fpath: Absolute output file path to write generated shape .glb file to
            shape_image_path (None or str): Absolute path to image to condition generation
            shape_prompt (None or str): Unused; Pixal3D is image-conditioned only, must be None
            shape_kwargs (None or dict): Unused; pass generation options via kwargs
            texture_image_path (None or str): Must be identical to shape_image_path
            texture_prompt (None or str): Unused; must be None
            texture_kwargs (None or dict): Unused; pass generation options via kwargs
            visualize (bool): Whether to visualize generated mesh
            texture_size (int): Size of the texture used for the output GLB
            save_intermediates (bool): Whether to save the preprocessed conditioning image and
                the estimated camera parameters alongside the mesh
            seed (int): Seed to use for mesh generation
            kwargs (Any): resolution (1024 or 1536), fov (radians), num_faces, remesh*,
                max_num_tokens, and ss_ / shape_slat_ / tex_slat_ prefixed sampler params
        """
        # ValueError rather than assert throughout: `python -O` strips asserts, and these guard
        # against silently generating a wrong mesh or reaching the disabled background remover.
        import json
        import math
        import tempfile
        import numpy as np

        if self.pipeline is None:
            raise RuntimeError(f"pipeline was not created for {self.__class__.__name__}!")
        if shape_image_path != texture_image_path:
            raise ValueError("shape_image_path and texture_image_path must be identical!")
        if shape_prompt is not None or texture_prompt is not None:
            raise ValueError(f"{self.__class__.__name__} is image-conditioned only!")
        if not out_fpath.endswith(".glb"):
            raise ValueError(f"out_fpath must end with .glb, got: {out_fpath}")

        # Upstream only defines the cascade for two values (pixal3d_image_to_3d.py run():
        # '1024_cascade', '1536_cascade'); anything else fails deep inside run().
        #
        # Resolve ONE effective pipeline_type and validate that, rather than validating
        # `resolution` and separately defaulting `pipeline_type`. Two holes made the previous
        # form unsafe, and they compounded:
        #   * `if "pipeline_type" not in kwargs` skipped the resolution check whenever the key was
        #     merely present — including `pipeline_type: null`, trivial to write in YAML.
        #   * kwargs.get("pipeline_type", <default>) returns a stored None rather than the
        #     default, so that same null was passed to run(), where
        #     `pipeline_type = pipeline_type or self.default_pipeline_type` selects the
        #     checkpoint's default. TencentARC/Pixal3D's pipeline.json sets that to
        #     "1536_cascade", so asking for resolution=1024 could silently run at 1536 and OOM a
        #     24 GiB card. `or` collapses null to the resolution-derived value.
        resolution = int(kwargs.get("resolution", 1024 if self.low_vram else 1536))
        pipeline_type = kwargs.get("pipeline_type") or f"{resolution}_cascade"
        if pipeline_type not in _PIXAL3D_PIPELINE_TYPES:
            raise ValueError(
                f"pipeline_type resolved to {pipeline_type!r}, but upstream defines only "
                f"{sorted(_PIXAL3D_PIPELINE_TYPES)}. Set resolution to 1024 or 1536, or pass an "
                f"explicit pipeline_type."
            )

        mesh_scale = float(kwargs.get("mesh_scale", 1.0))
        if not math.isfinite(mesh_scale) or mesh_scale <= 0.0:
            raise ValueError(f"mesh_scale must be a positive finite number, got: {mesh_scale}")
        extend_pixel = int(kwargs.get("extend_pixel", 0))
        image_resolution = int(kwargs.get("image_resolution", 512))
        if image_resolution <= 0:
            raise ValueError(f"image_resolution must be positive, got: {image_resolution}")

        fov = kwargs.get("fov", None)
        if fov is not None:
            fov = float(fov)
            # Radians. A degrees value (or 0) silently produces a wildly wrong camera distance
            # rather than an error, so bound it.
            if not math.isfinite(fov) or not (0.0 < fov < math.pi):
                raise ValueError(
                    f"fov is the horizontal field of view in RADIANS and must be in (0, pi), "
                    f"got: {fov}"
                )

        source_image = Image.open(shape_image_path)
        # Upstream decides whether to run background removal with exactly this test
        # (pixal3d_image_to_3d.py:153-157). Reproduce it here so an input that would fall through
        # to the (deliberately unavailable) background remover is rejected up front, naming the
        # file, rather than failing later inside preprocess_image.
        has_alpha = False
        alpha = None
        if source_image.mode == "RGBA":
            alpha = np.array(source_image)[:, :, 3]
            has_alpha = not np.all(alpha == 255)
        if not has_alpha:
            raise ValueError(
                f"{self.__class__.__name__} requires an RGBA image whose alpha channel already "
                f"isolates the object, because SimFoundry does not load Pixal3D's gated, "
                f"non-commercial background-removal model. Got mode={source_image.mode!r} with "
                f"{'no alpha channel' if source_image.mode != 'RGBA' else 'a fully-opaque alpha channel'}: "
                f"{shape_image_path}"
            )
        # A fully- or near-fully-transparent cutout passes the has_alpha test but leaves upstream's
        # foreground crop (`np.argwhere(alpha > 0.8 * 255)`) empty, which fails later as an opaque
        # numpy error about an empty axis. Catch it here against the same threshold.
        if not np.any(alpha > 0.8 * 255):
            raise ValueError(
                f"{self.__class__.__name__} found no foreground: no pixel has alpha > 204, so "
                f"there is nothing to reconstruct. Check the upstream segmentation for: "
                f"{shape_image_path}"
            )

        # Preprocess once and reuse: this composites RGBA over black, so letting run() redo it
        # would re-segment an already-composited image.
        image = self.pipeline.preprocess_image(source_image)

        if fov is not None:
            # Adapted from Pixal3D inference.py:200-209 (MIT, Copyright (c) 2026 Tencent; see
            # THIRD_PARTY_LICENSES.md item 0f) — upstream has no reusable helper for this path,
            # only the manual-FOV branch of run_inference().
            camera_angle_x = float(fov)
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            distance = self._inference.distance_from_fov(
                camera_angle_x,
                grid_point,
                torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
                mesh_scale,
                image_resolution,
            )["distance_from_x"]
            camera_params = {
                "camera_angle_x": camera_angle_x,
                "distance": distance,
                "mesh_scale": mesh_scale,
            }
        else:
            # MoGe must see the preprocessed image: its FOV estimate is width-normalized, so the
            # raw arbitrary-aspect stage-6 crop would describe a different image than the one the
            # pipeline consumes.
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "cond.png")
                image.save(tmp_path)
                self.moge_model.to("cuda")
                try:
                    camera_params = self._inference.get_camera_params_wild_moge(
                        tmp_path, self.moge_model, "cuda",
                        mesh_scale, extend_pixel, image_resolution,
                    )
                finally:
                    self.moge_model.cpu()
                    torch.cuda.empty_cache()

        # Build sampler params from kwargs, same convention as Trellis2.generate_mesh.
        _PARAM_RENAME = {
            "sampling_steps": "steps",
        }
        ss_sampler_params = {}
        shape_slat_sampler_params = {}
        tex_slat_sampler_params = {}
        _SAMPLER_PARAM_MAP = {
            "ss_": ss_sampler_params,
            "shape_slat_": shape_slat_sampler_params,
            "tex_slat_": tex_slat_sampler_params,
        }
        for key, value in kwargs.items():
            for prefix, target_dict in _SAMPLER_PARAM_MAP.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    param_name = _PARAM_RENAME.get(param_name, param_name)
                    target_dict[param_name] = value
                    break

        run_kwargs = dict(
            seed=seed,
            camera_params=camera_params,
            pipeline_type=pipeline_type,
            preprocess_image=False,   # already done above; run() would otherwise redo it
            return_latent=True,       # required: to_glb's grid_size comes from the returned res
        )
        if ss_sampler_params:
            run_kwargs["sparse_structure_sampler_params"] = ss_sampler_params
        if shape_slat_sampler_params:
            run_kwargs["shape_slat_sampler_params"] = shape_slat_sampler_params
        if tex_slat_sampler_params:
            run_kwargs["tex_slat_sampler_params"] = tex_slat_sampler_params
        if "max_num_tokens" in kwargs:
            run_kwargs["max_num_tokens"] = kwargs["max_num_tokens"]

        torch.manual_seed(seed)
        mesh_list, (_shape_slat, _tex_slat, res) = self.pipeline.run(image, **run_kwargs)
        mesh = mesh_list[0]

        if save_intermediates:
            # Upstream has no render-video path, so save the conditioning inputs instead — which
            # is what is actually needed to debug a bad or mis-posed mesh.
            object_name = os.path.basename(out_fpath).replace(".glb", "")
            renders_dir = os.path.join(os.path.dirname(out_fpath), "renders")
            os.makedirs(renders_dir, exist_ok=True)
            image.save(f"{renders_dir}/{object_name}_cond.png")
            with open(f"{renders_dir}/{object_name}_camera.json", "w", encoding="utf-8") as f:
                json.dump({k: float(v) for k, v in dict(camera_params).items()}, f, indent=2)

        num_faces = kwargs.get('num_faces', 300000)
        # Export to GLB. o_voxel.postprocess.to_glb accepts grid_size or voxel_size; Pixal3D
        # returns a grid resolution, whereas Trellis2 above passes mesh.voxel_size.
        import o_voxel
        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   self.pipeline.pbr_attr_layout,
            grid_size           =   res,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   num_faces,  # upstream default is 1000000
            texture_size        =   texture_size,
            remesh              =   kwargs.get('remesh', True),
            remesh_band         =   kwargs.get('remesh_band', 1),
            # Upstream uses 0 here; Trellis2 uses 0.9. Do not carry the TRELLIS.2-tuned value
            # over to a pixel-aligned model without measuring it.
            remesh_project      =   kwargs.get('remesh_project', 0),
            use_tqdm            =   kwargs.get('use_tqdm', True),
        )

        # Axis conversion, adapted from Pixal3D inference.py:272-278 (MIT, Copyright (c) 2026
        # Tencent; see THIRD_PARTY_LICENSES.md item 0f). to_glb does not apply it.
        axis_transform = np.array([
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0, -1.0, 0.0],
            [ 0.0, -1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float64)
        glb.apply_transform(axis_transform)

        if visualize:
            glb.show()
        publish_mesh_atomically(out_fpath, glb.export)


# Single source of truth for backend name -> class. Stage scripts must not define their own
# copies: the two that did drifted apart, and `trellis2` silently went missing from the cousin
# stage for its entire lifetime.
MESH_GENERATORS = {
    "direct3d": Direct3D,
    "hunyuan": Hunyuan,
    "pixal3d": Pixal3D,
    "trellis": Trellis,
    "trellis2": Trellis2,
}


def get_mesh_generator_cls(name):
    """
    Resolves a mesh generator name to its class.

    Args:
        name (str): Backend name, one of the keys of MESH_GENERATORS

    Returns:
        type: The generator class registered under @name

    Raises:
        ValueError: If @name is not a registered backend. Raised (rather than asserted) so the
            check survives `python -O`, matching simfoundry.pipeline.depth_backends.create_backend.
    """
    if name not in MESH_GENERATORS:
        raise ValueError(
            f"Unknown mesh generator '{name}'. Valid generators: {sorted(MESH_GENERATORS)}"
        )
    return MESH_GENERATORS[name]


# Argument names the stage scripts bind themselves when calling a generator. Config-supplied
# options must never include these, or the call raises
# "got multiple values for keyword argument". Note `visualize` is a named parameter of every
# backend, so signature filtering alone does not catch it.
RESERVED_GENERATION_KWARGS = {
    "generate_mesh": (
        "out_fpath", "shape_image_path", "texture_image_path", "shape_prompt", "texture_prompt",
        "shape_kwargs", "texture_kwargs", "visualize", "save_intermediates",
    ),
    "generate_shape": ("out_fpath", "image_path", "prompt", "visualize"),
    "generate_texture": ("out_fpath", "shape_fpath", "image_path", "prompt", "visualize"),
}


def filter_generation_kwargs(generate_fn, kwargs, reserved=()):
    """
    Drops the entries that @generate_fn cannot accept, or that the caller already binds.

    Backend signatures are heterogeneous: Trellis2.generate_mesh and Pixal3D.generate_mesh take
    **kwargs and absorb anything, while Hunyuan.generate_shape and Direct3D.generate_shape
    enumerate fixed parameters and raise TypeError on an unexpected name. Config-supplied options
    are therefore filtered per call rather than assumed to be universally accepted — otherwise
    setting any generation option (or even inheriting the global seed) breaks those backends.

    @reserved additionally removes names the call site passes explicitly. Those would otherwise
    survive both branches — a var-keyword backend accepts anything, and a name like `visualize`
    is in every fixed signature — and collide at call time.

    Args:
        generate_fn (callable): The bound generation method the kwargs will be passed to
        kwargs (dict): Candidate generation options
        reserved (Iterable[str]): Names the caller binds itself, see RESERVED_GENERATION_KWARGS

    Returns:
        dict: The subset of @kwargs safe to forward to @generate_fn
    """
    reserved = set(reserved)
    parameters = inspect.signature(generate_fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {key: value for key, value in kwargs.items() if key not in reserved}
    accepted = {
        name for name, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    } - reserved
    return {key: value for key, value in kwargs.items() if key in accepted}


def make_generator(generator_cls, *, low_vram=False, **kwargs):
    """
    Instantiates a mesh generator while passing low_vram only when supported.

    Not every backend accepts low_vram (e.g. Direct3D takes no constructor arguments at all),
    so it is filtered against the actual signature rather than passed unconditionally.

    Args:
        generator_cls (type): Generator class to instantiate
        low_vram (bool): Whether to request CPU offloading, if @generator_cls supports it
        kwargs (Any): Any additional constructor arguments

    Returns:
        ShapeGenerator or TextureGenerator: The instantiated generator
    """
    if "low_vram" in inspect.signature(generator_cls).parameters:
        kwargs["low_vram"] = low_vram
    return generator_cls(**kwargs)

