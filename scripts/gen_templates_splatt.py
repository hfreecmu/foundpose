#!/usr/bin/env python3

"""Synthesizes object templates."""

from vine_prune.utils.paths import (
    OBJECT_DIR,
    # YCB_PROCESSED_DIR,
    FP_BOP_PATH,
    FP_DINO_PATH,
    GAUSSIAN_MESH_SPLATTING_DIR,
    FOUNDPOSE_PATH
    )
from vine_prune.utils.general_utils import splat_to_image_color, create_pose

import sys
sys.path.append(FOUNDPOSE_PATH)
sys.path.append(FP_BOP_PATH)
sys.path.append(FP_DINO_PATH)
sys.path.append(GAUSSIAN_MESH_SPLATTING_DIR)

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import os

import cv2

import numpy as np

from foundpose_utils import (
    misc as foundpose_misc,
    json_util,
    config_util,
    logging,
    misc,
    structs
)

from foundpose_utils.structs import AlignedBox2f, PinholePlaneCameraModel

from foundpose_utils.misc import warp_depth_image, warp_image
from foundpose_utils import geometry, renderer_builder
from foundpose_utils.renderer_base import RenderType

from bop_toolkit_lib import inout

import pycolmap
import trimesh

###
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import PipelineParams
from renderer.gaussian_renderer import my_render
###

class GenTemplatesOpts(NamedTuple):
    """Options that can be specified via the command line."""

    # Viewpoint options.
    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    images_per_view: int = 1

    # Rendering options.
    ssaa_factor: float = 1.0

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Other options.
    save_templates: bool = True
    overwrite: bool = True
    debug: bool = True

    # depth_range: Tuple[int] = None

def synthesize_templates(opts: GenTemplatesOpts, args) -> None: 
    object_name = args.object_name
    baseline = args.baseline
    is_obj = args.is_obj
    is_ycb = args.is_ycb
    force_dc = args.force_dc

    assert not (is_obj and is_ycb)
    assert (is_obj or is_ycb)

    if baseline is None:
        raise RuntimeError('Baseline required')

    if is_obj:
        data_dir = os.path.join(OBJECT_DIR, object_name)

        splat_dir = os.path.join(data_dir, 'splat')
        sfm_dir = os.path.join(data_dir, 'colmap', 'sparse', '0')

        splat_path = os.path.join(splat_dir, 'scale_center.ply')
    else:
        data_dir = os.path.join(YCB_PROCESSED_DIR, object_name)
        sfm_dir = os.path.join(data_dir, f'{object_name}_colmap', 'sparse', '0')
        splat_path = os.path.join(data_dir, 'obj_splat.ply')

    # Fix the random seed for reproducibility.
    np.random.seed(0)

    # Prepare a logger and a timer.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    reconstruction = pycolmap.Reconstruction(sfm_dir)
    fx = reconstruction.cameras[1].focal_length_x
    fy = reconstruction.cameras[1].focal_length_y
    cx = reconstruction.cameras[1].principal_point_x
    cy = reconstruction.cameras[1].principal_point_y
    intrinsics = [fx, fy, cx, cy]

    height = reconstruction.cameras[1].height
    width = reconstruction.cameras[1].width
    dims = [height, width]

    logger.info(f"Camera details are read ")

    render_camera_model = PinholePlaneCameraModel(
        width=int(dims[1] * opts.ssaa_factor),
        height=int(dims[0] * opts.ssaa_factor),
        f=(
            intrinsics[0] * opts.ssaa_factor,
            intrinsics[1] * opts.ssaa_factor,
        ),
        c=(
            intrinsics[2] * opts.ssaa_factor,
            intrinsics[3] * opts.ssaa_factor,
        )
    )
    print("camera model created")


    # Define radii of the view spheres on which we will sample viewpoints.
    # The specified number of radii is sampled uniformly in the range of
    # camera-object distances from the test split of the specified dataset.
    # depth_range = opts.depth_range

    splat_mesh = trimesh.load(splat_path)
    bounding_sphere = splat_mesh.bounding_sphere
    object_radius = bounding_sphere.to_dict()['radius']

    theta_x = 2*np.arctan(width / (2*fx))
    theta_y = 2*np.arctan(height / (2*fy))
    theta = min(theta_x, theta_y)
    depth_to_use = object_radius / np.sin(theta / 2)
    depth_to_use *= 1000
    # I used 5 for the spong and it worked
    # before was 1.5
    depth_to_use *= 5#1.5 
    depth_range = [depth_to_use]

    min_depth = np.min(depth_range)
    max_depth = np.max(depth_range)
    depth_range_size = max_depth - min_depth
    depth_cell_size = depth_range_size / float(opts.num_viewspheres)
    viewsphere_radii = []
    for depth_cell_id in range(opts.num_viewspheres):
        viewsphere_radii.append(min_depth + (depth_cell_id + 0.5) * depth_cell_size)

    # Generate viewpoints from which the object model will be rendered.
    views_sphere = []
    for radius in viewsphere_radii:
        views_sphere += foundpose_misc.sample_views(
            min_n_views=opts.min_num_viewpoints,
            radius=radius,
            mode="fibonacci",
        )[0]
    logger.info(f"Sampled points on the sphere: {len(views_sphere)}")

    # Add in-plane rotations.
    if opts.num_inplane_rotations == 1:
        views = views_sphere
    else:
        inplane_angle = 2 * np.pi / opts.num_inplane_rotations
        views = []
        for view_sphere in views_sphere:
            for inplane_id in range(opts.num_inplane_rotations):
                R_inplane = geometry.rotation_matrix_numpy(
                    inplane_angle * inplane_id, np.array([0, 0, 1])
                )[:3, :3]
                views.append(
                    {
                        "R": R_inplane.dot(view_sphere["R"]),
                        "t": R_inplane.dot(view_sphere["t"]),
                    }
                )
    logger.info(f"Number of views: {len(views)}")

    timer.elapsed("Time for setting up the stage")

    if force_dc:
        sh_degree = 0
    else:
        sh_degree = 3

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(splat_path) 
    gauss_means = gaussians.get_xyz.detach().cpu().numpy().mean(axis=0)   

    parser = ArgumentParser()
    pipeline_par = PipelineParams(parser)
    args, _ = parser.parse_known_args()
    pipeline = pipeline_par.extract(args)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Generate template
    # Prepare output folder.
    object_lid = object_name

    output_dir = os.path.join(data_dir, 'foundpose')

    print("output_dir: ", output_dir)
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")

    # Save parameters to a file.
    config_path = os.path.join(output_dir, "config.json")
    json_util.save_json(config_path, opts)

    # Prepare folder for saving templates.
    templates_rgb_dir = os.path.join(output_dir, "rgb")
    if opts.save_templates:
        os.makedirs(templates_rgb_dir, exist_ok=True)

    # templates_depth_dir = os.path.join(output_dir, "depth")
    # if opts.save_templates:
    #     os.makedirs(templates_depth_dir, exist_ok=True)

    templates_mask_dir = os.path.join(output_dir, "mask")
    if opts.save_templates:
        os.makedirs(templates_mask_dir, exist_ok=True)


    # Prepare a metadata list.
    metadata_list = []

    timer.elapsed("Time for preparing object data")

    template_list = []
    template_counter = 0
    for view_id, view in enumerate(views):
        logger.info(
            f"Rendering object {object_lid}, view {view_id}/{len(views)}..."
        )

        # add for mean of gaussian
        view['t'] += gauss_means[:, None] * 1000

        for _ in range(opts.images_per_view):

            timer.start()

            R = view['R']
            t = view['t'][:, 0] / 1000

            intrinsics = [render_camera_model.f[0], render_camera_model.f[1],
                          render_camera_model.c[0], render_camera_model.c[1]]
            dims = [render_camera_model.height, render_camera_model.width]

            # tmp = reconstruction.images[1].cam_from_world.matrix()
            # R = tmp[0:3, 0:3]
            # t = tmp[0:3, 3]

            res_pkg = my_render(gaussians, pipeline, background,
                                intrinsics, dims, R.T, t, is_tensor=False)
        
            image = splat_to_image_color(res_pkg['render'])
            # depth = res_pkg['depth'].cpu().numpy().squeeze(0) * 1000
            # mask = np.any(image > 0, axis=-1).astype(np.uint8) * 255
            mask = (res_pkg['alpha'].cpu().numpy().squeeze(0) > 0.5).astype(np.uint8)*255

            output = {}
            output[RenderType.COLOR] = image
            # output[RenderType.DEPTH] = depth
            output[RenderType.MASK] = mask

            # cv2.imshow('test', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # #cv2.imshow('test', mask)
            # cv2.waitKey(0)
            
            # Calculate 2D bounding box of the object and make sure
            # it is within the image.
            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(foundpose_misc.calc_2d_box(xs, ys))
            object_box = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            if (
                object_box.left == 0
                or object_box.top == 0
                or object_box.right == dims[1] - 1
                or object_box.bottom == dims[0] - 1
            ):
                raise ValueError("The model does not fit the viewport.")
            
            # to apply cropping logic
            trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])
            R_c2m = trans_m2c.R.T
            trans_c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))
            trans_c2m_matrix = misc.get_rigid_matrix(trans_c2m)
            render_camera_model_c2w = PinholePlaneCameraModel(
                    width=render_camera_model.width,
                    height=render_camera_model.height,
                    f=render_camera_model.f,
                    c=render_camera_model.c,
                    T_world_from_eye=trans_c2m_matrix,
                )
            
            # Optionally crop the object region.
            if opts.crop:
                # Get box for cropping.
                crop_box = foundpose_misc.calc_crop_box(
                    box=object_box,
                    make_square=True,
                )

                # Construct a virtual camera focused on the box.
                crop_camera_model_c2w = foundpose_misc.construct_crop_camera(
                    box=crop_box,
                    camera_model_c2w=render_camera_model_c2w,
                    viewport_size=(
                        int(opts.crop_size[0] * opts.ssaa_factor),
                        int(opts.crop_size[1] * opts.ssaa_factor),
                    ),
                    viewport_rel_pad=opts.crop_rel_pad,
                )
                
                # assert opts.ssaa_factor == 1.0
                # size_diff = opts.crop_size

                new_trans_c2m_matrix = crop_camera_model_c2w.T_world_from_eye
                new_R_c2m = new_trans_c2m_matrix[0:3, 0:3]
                new_t_c2m = new_trans_c2m_matrix[0:3, 3]
                new_R = new_R_c2m.T
                new_t = -new_R @ new_t_c2m

                R = new_R
                t = new_t / 1000

                crop_intrinsics = [crop_camera_model_c2w.f[0], crop_camera_model_c2w.f[1],
                                   crop_camera_model_c2w.c[0], crop_camera_model_c2w.c[1]]
                crop_dims = [crop_camera_model_c2w.height, crop_camera_model_c2w.width]
                
                del res_pkg
                torch.cuda.empty_cache()

                res_pkg = my_render(gaussians, pipeline, background,
                                    np.array(crop_intrinsics), crop_dims, R.T, t, is_tensor=False)
            
                image = splat_to_image_color(res_pkg['render'])
                # depth = res_pkg['depth'].cpu().numpy().squeeze(0) * 1000
                # mask = np.any(image > 0, axis=-1).astype(np.uint8) * 255
                mask = (res_pkg['alpha'].cpu().numpy().squeeze(0) > 0.5).astype(np.uint8)*255

                output = {}
                output[RenderType.COLOR] = image
                # output[RenderType.DEPTH] = depth
                output[RenderType.MASK] = mask

                del res_pkg
                t_baseline = np.copy(t)
                t_baseline[0] -= 0.02
                res_pkg = my_render(gaussians, pipeline, background,
                                    np.array(crop_intrinsics), crop_dims, R.T, t_baseline, is_tensor=False)
                right_image = splat_to_image_color(res_pkg['render'])
                output['right_image'] = right_image

                # cv2.imshow('test', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)

                # The virtual camera is becoming the main camera.
                camera_model_c2w = crop_camera_model_c2w.copy()
                scale_factor = opts.crop_size[0] / float(
                    crop_camera_model_c2w.width
                )
                camera_model_c2w.width = opts.crop_size[0]
                camera_model_c2w.height = opts.crop_size[1]
                camera_model_c2w.c = (
                    camera_model_c2w.c[0] * scale_factor,
                    camera_model_c2w.c[1] * scale_factor,
                )
                camera_model_c2w.f = (
                    camera_model_c2w.f[0] * scale_factor,
                    camera_model_c2w.f[1] * scale_factor,
                )

            # In case we are not cropping.
            else:
                raise RuntimeError('only crop supported')

            # Downsample the renderings to the target size in case of SSAA.
            if opts.ssaa_factor != 1.0:
                raise RuntimeError('not supported right now')
                target_size = (camera_model_c2w.width, camera_model_c2w.height)
                for output_key in output.keys():
                    if output_key in [RenderType.COLOR]:
                        interpolation = cv2.INTER_AREA
                    else:
                        interpolation = cv2.INTER_NEAREST

                    output[output_key] = misc.resize_image(
                        image=output[output_key],
                        size=target_size,
                        interpolation=interpolation,
                    )

                # cv2.imshow('test', cv2.cvtColor(output[RenderType.COLOR], cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)
            else:
                pass

            # Record the template in the template list.
            template_list.append(
                {
                    "seq_id": template_counter,
                }
            )

            # Model and world coordinate frames are aligned.
            trans_m2w = structs.RigidTransform(R=np.eye(3), t=np.zeros((3, 1)))

            # The object is fully visible.
            visibility = 1.0

            # Recalculate the object bounding box (it changed if we constructed the virtual camera).
            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(foundpose_misc.calc_2d_box(xs, ys))
            object_box = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            rgb_image = output[RenderType.COLOR]
            # depth_image = output[RenderType.DEPTH]
            right_image = output['right_image']

            timer.elapsed("Time for template generation")

            # Save template rgb, depth and mask.
            timer.start()
            rgb_path = os.path.join(
                templates_rgb_dir, f"{template_counter:06d}.png"
            )
            logger.info(f"Saving template RGB {template_counter} to: {rgb_path}")
            inout.save_im(rgb_path, rgb_image)
            inout.save_im(rgb_path.replace('.png', '_right.png'), right_image)

            # depth_path = os.path.join(
            #     templates_depth_dir, f"{template_counter:06d}.png"
            # )
            # logger.info(f"Saving template depth map {template_counter} to: {depth_path}")
            # inout.save_depth(depth_path, depth_image)

            # np.save(depth_path.replace('.png', '.npy'), depth_image)
            # depth_path = os.path.join(
            #     templates_depth_dir, f"{template_counter:06d}.npy"
            # )
            # logger.info(f"Saving template depth map {template_counter} to: {depth_path}")
            # np.save(depth_path, depth_image)

            # Save template mask.
            mask_path = os.path.join(
                templates_mask_dir, f"{template_counter:06d}.png"
            )
            logger.info(f"Saving template binary mask {template_counter} to: {mask_path}")
            inout.save_im(mask_path, output[RenderType.MASK])

            data = {
                "lid": object_lid,
                "template_id": template_counter,
                "pose": trans_m2w,
                "boxes_amodal": np.array([object_box.array_ltrb()]).tolist(),
                "visibilities": np.array([visibility]).tolist(),
                "cameras": camera_model_c2w.to_json(),
                "rgb_image_path": rgb_path,
                # "depth_map_path": depth_path,
                "binary_mask_path": mask_path,
            }
            timer.elapsed("Time for template saving")

            metadata_list.append(data)

            template_counter += 1

    # Save the metadata to be read from object repre.
    metadata_path = os.path.join(output_dir, "metadata.json")
    json_util.save_json(metadata_path, metadata_list)


def main() -> None:
    opts, args, _ = config_util.load_opts_from_json_or_command_line(
        GenTemplatesOpts,
    )

    synthesize_templates(opts, args)


if __name__ == "__main__":
    with torch.no_grad():
        main()
