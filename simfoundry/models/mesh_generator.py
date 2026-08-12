# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import trimesh
import shutil
from simfoundry.utils.python_utils import atomic_copyfile, atomic_output_path
import imageio
from PIL import Image
import os
import torch


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
        cls.REPO_PATH = repo_path
        import sys
        sys.path.insert(0, f"{cls.REPO_PATH}/hy3dshape")
        sys.path.insert(0, f"{cls.REPO_PATH}/hy3dpaint")

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):
        # Local import now to avoid dependency crashing depending on environment being run currently
        assert self.REPO_PATH is not None, f"Must set absolute path {self.__class__.__name__}'s repo path via set_repo_path()!"
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
        mesh_untextured = self.shape_pipeline(
            image=image_path,
            num_inference_steps=50,
            timesteps=None,
            sigmas=None,
            eta=0.0,
            guidance_scale=5.0,  # 5.0,
            generator=None,
            box_v=1.01,
            octree_resolution=384,  # 384,
            mc_level=0.0,
            mc_algo=None,
            num_chunks=8000,
            output_type="trimesh",
            enable_pbar=True,
            mask=None,
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
        cls.REPO_PATH = repo_path
        import sys
        sys.path.insert(0, f"{cls.REPO_PATH}/hy3dshape")
        sys.path.insert(0, f"{cls.REPO_PATH}/hy3dpaint")

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):
        # Local import now to avoid dependency crashing depending on environment being run currently
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
        cls.REPO_PATH = repo_path
        import sys
        import os
        sys.path.insert(0, f"{cls.REPO_PATH}")

        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory

    def create_pipelines(self, create_shape_pipeline, create_texture_pipeline):

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


    
      
