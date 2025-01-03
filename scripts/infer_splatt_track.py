#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import os
import gc
import time

from typing import List, NamedTuple, Optional, Tuple

import cv2

import numpy as np

import torch

from foundpose_utils.misc import array_to_tensor, tensor_to_array, tensors_to_arrays

from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc


from foundpose_utils import (
    corresp_util,
    config_util,
    eval_errors,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    data_util,
    renderer_builder,
    json_util, 
    logging,
    misc,
    structs,
    cluster_util,
    template_util
)

from foundpose_utils.structs import AlignedBox2f, PinholePlaneCameraModel
from foundpose_utils.misc import warp_depth_image, warp_image
from foundpose_utils.renderer_base import RenderType

import imageio

###
from scene import GaussianModel
from utils.graphics_utils import focal2fov
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams
###

def my_render(gaussians, pipeline, background, intrinsics, dims, R, T):
    fx, fy, cx, cy = intrinsics
    image_height, image_width = dims

    FoVx = focal2fov(fx, image_width)
    FoVy = focal2fov(fy, image_height)

    cx = (cx - image_width / 2) / image_width * 2
    cy = (cy - image_height / 2) / image_height * 2

    dummy_image = torch.ones(3, image_height, image_width).float().to('cuda')

    cam = Camera(colmap_id=None, R=R, T=T, 
                  FoVx=FoVx, FoVy=FoVy, 
                  cx=cx, cy=cy,
                  image=dummy_image, 
                  gt_alpha_mask=None,
                  image_name=None, uid=None,
                  semantic_feature=None,
                  )
    
    res_pkg = render(cam, gaussians, pipeline, background)
    return res_pkg


logger: logging.Logger = logging.get_logger()


class InferOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    repre_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    max_sym_disc_step: float = 0.01

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Object instance options.
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # Feature extraction options.
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # Feature matching options.
    match_template_type: str = "tfidf"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # PnP options.
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True

    final_pose_type: str = "best_coarse"

    # Other options.
    save_estimates: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True
    debug: bool = True

    splat_path: str = None

    # I added and am hardcoding for now
    features_patch_size: int = 14
    ssaa_factor: float = 2.0
    cluster_num: int = 2048

    # except for this one
    template_desc_opts: Optional[repre_util.TemplateDescOpts] = None
    debug_desc_opts: Optional[repre_util.TemplateDescOpts] = None

DATA_DIR = '/home/hfreeman/Downloads/feat_test/bundlesdf'
def infer(opts: InferOpts) -> None:

    # Prepare a logger and a timer.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = misc_util.Timer(enabled=opts.debug)
    timer.start()

    # Prepare feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    timer.elapsed("Time for setting up the stage")

    # Create a renderer.
    MODEL_PATH = '/home/hfreeman/Downloads/mesh_test/textured_mesh_mm.obj'
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(renderer_type=renderer_type, model_path=MODEL_PATH)
    gaussians = GaussianModel(3)
    gaussians.load_ply(opts.splat_path) 

    parser = ArgumentParser()
    pipeline_par = PipelineParams(parser)
    args, _ = parser.parse_known_args()
    pipeline = pipeline_par.extract(args)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    object_lid = 'pruners'

    # Run inference for each specified object.
    timer.start()

    # The output folder is named with slugified dataset path.
    version = opts.version
    if version == "":
        version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
    output_dir = os.path.join(
        bop_config.output_path, "inference", signature, str(object_lid)
    )
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(
        bop_config.output_path, "inference", signature, str(object_lid) + '_vis'
    )
    os.makedirs(vis_dir, exist_ok=True)

    # Save parameters to a file.
    config_path = os.path.join(output_dir, "config.json")
    json_util.save_json(config_path, opts)

    # pose_evaluator = eval_util.EvaluatorPose([object_lid])

    # Load the object representation.
    logger.info(
        f"Loading representation for object {object_lid} from dataset {opts.object_dataset}..."
    )
    base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
    repre_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, opts.object_dataset, object_lid
    )
    repre = repre_util.load_object_repre(
        repre_dir=repre_dir,
        tensor_device=device,
    )

    logger.info("Object representation loaded.")
    repre_np = repre_util.convert_object_repre_to_numpy(repre)

    # Build a kNN index from object feature vectors.
    visual_words_knn_index = None
    if opts.match_template_type == "tfidf":
        visual_words_knn_index = knn_util.KNN(
            k=repre.template_desc_opts.tfidf_knn_k,
            metric=repre.template_desc_opts.tfidf_knn_metric
        )
        visual_words_knn_index.fit(repre.feat_cluster_centroids)

    # Build per-template KNN index with features from that template.
    template_knn_indices = []
    if opts.match_feat_matching_type == "cyclic_buddies":
        logger.info("Building per-template KNN indices...")
        for template_id in range(len(repre.template_cameras_cam_from_model)):
            logger.info(f"Building KNN index for template {template_id}...")
            tpl_feat_mask = repre.feat_to_template_ids == template_id
            tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

            template_feats = repre.feat_vectors[tpl_feat_ids]

            # Build knn index for object features.
            template_knn_index = knn_util.KNN(k=1, metric="l2")
            template_knn_index.fit(template_feats.cpu())
            template_knn_indices.append(template_knn_index)
        logger.info("Per-template KNN indices built.")

    logging.log_heading(
        logger,
        f"Object: {object_lid}, vertices: {len(repre.vertices)}",
        style=logging.WHITE_BOLD,
    )

    timer.elapsed("Time for preparing object data")

    color_dir = os.path.join(DATA_DIR, 'color')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(DATA_DIR, 'rgb')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(DATA_DIR, 'undistorted')
    mask_dir = os.path.join(DATA_DIR, 'mask')
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(DATA_DIR, 'masks')
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(DATA_DIR, 'mask_obj')
    K_path = os.path.join(DATA_DIR, 'cam_K.txt')

    K = np.loadtxt(K_path)
    orig_camera_c2w = PinholePlaneCameraModel(
        width=640,
        height=480,
        f=(K[0,0], K[1,1]),
        c=(K[0,2], K[1,2])
    )

    # Generate grid points at which to sample the feature vectors.
    if opts.crop:
        grid_size = opts.crop_size
    else:
        grid_size = orig_image_size
    grid_points = feature_util.generate_grid_points(
        grid_size=grid_size,
        cell_size=opts.grid_cell_size,
    )
    grid_points = grid_points.to(device)

    # this is for repre
    datasets_path = bop_config.datasets_path
    bop_camera = dataset_params.get_camera_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
    bop_camera_width = bop_camera['im_size'][0]
    bop_camera_height = bop_camera['im_size'][1]
    max_image_side = max(bop_camera_width, bop_camera_height)
    image_side = opts.features_patch_size * int(
        max_image_side / opts.features_patch_size
    )
    camera_model = PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(bop_camera['K'][0,0], bop_camera['K'][1,1]),
        c=(
            bop_camera['K'][0,2] - 0.5 * (bop_camera_width - image_side),
            bop_camera['K'][1,2] - 0.5 * (bop_camera_height - image_side),
        )
    )
    render_camera_model = PinholePlaneCameraModel(
        width=int(camera_model.width * opts.ssaa_factor),
        height=int(camera_model.height * opts.ssaa_factor),
        f=(
            camera_model.f[0] * opts.ssaa_factor,
            camera_model.f[1] * opts.ssaa_factor,
        ),
        c=(
            camera_model.c[0] * opts.ssaa_factor,
            camera_model.c[1] * opts.ssaa_factor,
        )
    )

    match_top_n_templates = opts.match_top_n_templates
    #

    filenames = []
    for filename in os.listdir(color_dir):
        if not (filename.endswith('.png') or filename.endswith('.jpg')):
            continue

        filenames.append(filename)

    filenames = sorted(filenames)

    for filename in filenames:
        
        timer.start()

        basename = filename.replace('.png', '').replace('.jpg', '')

        image_path = os.path.join(color_dir, filename)
        mask_path = os.path.join(mask_dir, basename + '.png')

        orig_image_np_hwc = imageio.imread(image_path) / 255.0
        orig_mask_modal = imageio.imread(mask_path)

        orig_image_np_hwc[orig_mask_modal == 0] = 0.0

        ys, xs = orig_mask_modal.nonzero()
        box = np.array(misc.calc_2d_box(xs, ys))

        orig_box_amodal = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )
        
        # Optional cropping.
        if not opts.crop:
            camera_c2w = orig_camera_c2w
            image_np_hwc = orig_image_np_hwc
            mask_modal = orig_mask_modal
            box_amodal = orig_box_amodal
        else:
            # Get box for cropping.
            crop_box = misc_util.calc_crop_box(
                box=orig_box_amodal,
                make_square=True,
            )

            # Construct a virtual camera focused on the crop.
            crop_camera_model_c2w = misc_util.construct_crop_camera(
                box=crop_box,
                camera_model_c2w=orig_camera_c2w,
                viewport_size=opts.crop_size,
                viewport_rel_pad=opts.crop_rel_pad,
            )

            # Map images to the virtual camera.
            interpolation = (
                cv2.INTER_AREA
                if crop_box.width >= crop_camera_model_c2w.width
                else cv2.INTER_LINEAR
            )
            image_np_hwc = warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_image_np_hwc,
                interpolation=interpolation,
            )
            mask_modal = warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_mask_modal,
                interpolation=cv2.INTER_NEAREST,
            )

            # Recalculate the object bounding box (it changed if we constructed the virtual camera).
            ys, xs = mask_modal.nonzero()
            box = np.array(misc_util.calc_2d_box(xs, ys))
            box_amodal = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            # The virtual camera is becoming the main camera.
            camera_c2w = crop_camera_model_c2w

        timer.elapsed("Time for preparation")
        timer.start()

        # Extract feature map from the crop.
        image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
        image_tensor_bchw = image_tensor_chw.unsqueeze(0)
        extractor_output = extractor(image_tensor_bchw)
        feature_map_chw = extractor_output["feature_maps"][0]

        timer.elapsed("Time for feature extraction")

        # Keep only points inside the object mask.
        mask_modal_tensor = array_to_tensor(mask_modal).to(device)
        query_points = feature_util.filter_points_by_mask(
            grid_points, mask_modal_tensor
        )

        # Subsample query points if we have too many.
        if query_points.shape[0] > opts.max_num_queries:
            perm = torch.randperm(query_points.shape[0])
            query_points = query_points[perm[: opts.max_num_queries]]
            msg = (
                "Randomly sumbsampled queries "
                f"({perm.shape[0]} -> {query_points.shape[0]}))"
            )
            logging.log_heading(logger, msg, style=logging.RED_BOLD)

        # Extract features at the selected points, of shape (num_points, feat_dims).
        timer.start()
        query_features = feature_util.sample_feature_map_at_points(
            feature_map_chw=feature_map_chw,
            points=query_points,
            image_size=(image_np_hwc.shape[1], image_np_hwc.shape[0]),
        ).contiguous()

        timer.elapsed("Time for grid sample")
        timer.start()
        # Potentially project features to a PCA space.
        if (
            query_features.shape[1] != repre.feat_vectors.shape[1]
            and len(repre.feat_raw_projectors) != 0
        ):
            query_features_proj = projector_util.project_features(
                feat_vectors=query_features,
                projectors=repre.feat_raw_projectors,
            ).contiguous()

            _c, _h, _w = feature_map_chw.shape
            feature_map_chw_proj = (
                projector_util.project_features(
                    feat_vectors=feature_map_chw.permute(1, 2, 0).view(-1, _c),
                    projectors=repre.feat_raw_projectors,
                )
                .view(_h, _w, -1)
                .permute(2, 0, 1)
            )
        else:
            query_features_proj = query_features
            feature_map_chw_proj = feature_map_chw

        timer.elapsed("Time for projection")
        timer.start()

        # Establish 2D-3D correspondences.
        if not len(query_points) > 0:
            raise RuntimeError('need more than 0 query points')
        
        corresp = corresp_util.establish_correspondences(
            query_points=query_points,
            query_features=query_features_proj,
            object_repre=repre,
            template_matching_type=opts.match_template_type,
            template_knn_indices=template_knn_indices,
            feat_matching_type=opts.match_feat_matching_type,
            top_n_templates=match_top_n_templates,
            top_k_buddies=opts.match_top_k_buddies,
            visual_words_knn_index=visual_words_knn_index,
            debug=opts.debug,
        )


        timer.elapsed("Time for corresp")
        timer.start()

        logger.info(
            f"Number of corresp: {[len(c['coord_2d']) for c in corresp]}"
        )

        # Estimate coarse poses from corespondences.
        coarse_poses = []
        for corresp_id, corresp_curr in enumerate(corresp):

            # We need at least 3 correspondences for P3P.
            num_corresp = len(corresp_curr["coord_2d"])
            if num_corresp < 6:
                logger.info(f"Only {num_corresp} correspondences, skipping.")
                continue

            (
                coarse_pose_success,
                R_m2c_coarse,
                t_m2c_coarse,
                inliers_coarse,
                quality_coarse,
            ) = pnp_util.estimate_pose(
                corresp=corresp_curr,
                camera_c2w=camera_c2w,
                pnp_type=opts.pnp_type,
                pnp_ransac_iter=opts.pnp_ransac_iter,
                pnp_inlier_thresh=opts.pnp_inlier_thresh,
                pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
                pnp_refine_lm=opts.pnp_refine_lm,
            )

            logger.info(
                f"Quality of coarse pose {corresp_id}: {quality_coarse}"
            )

            if coarse_pose_success:
                coarse_poses.append(
                    {
                        "type": "coarse",
                        "R_m2c": R_m2c_coarse,
                        "t_m2c": t_m2c_coarse,
                        "corresp_id": corresp_id,
                        "quality": quality_coarse,
                        "inliers": inliers_coarse,
                    }
                )

        # Find the best coarse pose.
        best_coarse_quality = None
        best_coarse_pose_id = 0
        for coarse_pose_id, pose in enumerate(coarse_poses):
            if (
                best_coarse_quality is None
                or pose["quality"] > best_coarse_quality
            ):
                best_coarse_pose_id = coarse_pose_id
                best_coarse_quality = pose["quality"]

        timer.elapsed("Time for coarse pose")

        timer.start()
        
        if opts.final_pose_type in [
            "best_coarse",
        ]:

            # If no successful coarse pose, continue.
            if len(coarse_poses) == 0:
                continue

            # Select the refined pose corresponding to the best coarse pose as the final pose.
            final_pose = coarse_poses[best_coarse_pose_id]

        else:
            raise ValueError(f"Unknown final pose type {opts.final_pose_type}")

        timer.elapsed("Time for selecting final pose")

        # Visualizations and saving of results.
        vis_tiles = []

        # Increment hypothesis id by one for each found pose hypothesis.
        pose_m2w = None
        pose_m2w_coarse = None

        # Express the estimated pose as an m2w transformation.
        pose_est_m2c = structs.ObjectPose(
            R=final_pose["R_m2c"], t=final_pose["t_m2c"]
        )
        trans_c2w = camera_c2w.T_world_from_eye

        trans_m2w = trans_c2w.dot(misc.get_rigid_matrix(pose_est_m2c))
        pose_m2w = structs.ObjectPose(
            R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:]
        )

        # Get image for visualization.
        vis_base_image = (255 * image_np_hwc).astype(np.uint8)

        # Convert correspondences from tensors to numpy arrays.
        best_corresp_np = tensors_to_arrays(
            corresp[final_pose["corresp_id"]]
        )

        # pose_eval_dict = pose_evaluator.update_without_anno(
        #             scene_id=None,
        #             im_id=basename,
        #             inst_id=None,
        #             hypothesis_id=hypothesis_id,
        #             object_repre_vertices=tensor_to_array(repre.vertices),
        #             obj_lid=object_lid,
        #             object_pose_m2w=pose_m2w,
        #             orig_camera_c2w=orig_camera_c2w,
        #             camera_c2w=orig_camera_c2w,
        #             time_per_inst=None,
        #             corresp=best_corresp_np,
        #             inlier_radius=(opts.pnp_inlier_thresh),
        #         )

        # Optionally visualize the results.
        if opts.vis_results:

            # IDs and scores of the matched templates.
            matched_template_ids = [c["template_id"] for c in corresp]
            matched_template_scores = [c["template_score"] for c in corresp]

            timer.start()

            vis_tiles += vis_util.vis_inference_results(
                    base_image=vis_base_image,
                    object_repre=repre_np,
                    object_lid=object_lid,
                    object_pose_m2w=pose_m2w, # pose_m2w,
                    object_pose_m2w_gt=None,
                    feature_map_chw=feature_map_chw,
                    feature_map_chw_proj=feature_map_chw_proj,
                    vis_feat_map=opts.vis_feat_map,
                    object_box=box_amodal.array_ltrb(),
                    object_mask=mask_modal,
                    camera_c2w=camera_c2w,
                    corresp=best_corresp_np,
                    matched_template_ids=matched_template_ids,
                    matched_template_scores=matched_template_scores,
                    best_template_ind=final_pose["corresp_id"],
                    renderer=renderer,
                    pose_eval_dict=None,
                    corresp_top_n=opts.vis_corresp_top_n,
                    inlier_thresh=(opts.pnp_inlier_thresh),
                    object_pose_m2w_coarse=pose_m2w_coarse,
                    pose_eval_dict_coarse=None,
                    # For paper visualizations:
                    vis_for_paper=opts.vis_for_paper,
                    extractor=extractor,
                )
            
            timer.elapsed("Time for visualization")

        # Assemble visualization tiles to a grid and save it.
        if len(vis_tiles):
            if repre.feat_vis_projectors[0].pca.n_components == 12:
                pca_tiles = np.vstack(vis_tiles[1:5])
                vis_tiles = np.vstack([vis_tiles[0]] + vis_tiles[5:])
                vis_grid = np.hstack([vis_tiles, pca_tiles])
            else:
                vis_grid = np.vstack(vis_tiles)
            ext = ".png" if opts.vis_for_paper else ".jpg"
            vis_path = os.path.join(
                output_dir,
                f"{basename}{ext}",
            )
            inout.save_im(vis_path, vis_grid)
            logger.info(f"Visualization saved to {vis_path}")

        R = trans_m2w[:3, :3]
        t = trans_m2w[:3, 3] / 1000

        # intrinsics = [camera_c2w.f[0], camera_c2w.f[1],
        #               camera_c2w.c[0], camera_c2w.c[1]]
        # dims = [camera_c2w.height, camera_c2w.width]
        intrinsics = [orig_camera_c2w.f[0], orig_camera_c2w.f[1],
                        orig_camera_c2w.c[0], orig_camera_c2w.c[1]]
        dims = [orig_camera_c2w.height, orig_camera_c2w.width]
        res_pkg = my_render(gaussians, pipeline, background,
                    intrinsics, dims, R.T, t)

        image = (res_pkg['render'].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        mask = np.any(image > 0, axis=-1).astype(np.uint8) * 255
        depth = res_pkg['depth'].squeeze(0).cpu().numpy() * 1000
        # features = res_pkg['feature_map'].cpu().numpy()
        # TODO maybe one more pnp iteration?

        orig_image = (orig_image_np_hwc*255).astype(np.uint8)

        overlay_im = (orig_image.astype(float) / 255.0)*0.5 + (image.astype(float) / 255.0)*0.5
        overlay_im = (overlay_im.clip(0.0, 1.0) * 255).round().astype(np.uint8)

        vis_im = np.hstack((orig_image, image, overlay_im))
        vis_im = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
        cv2.imshow('test', vis_im)
        cv2.waitKey(1)
        vis_path = os.path.join(
                vis_dir,
                f"{basename}{ext}",
            )
        cv2.imwrite(vis_path, vis_im)
        
        pose_path = os.path.join(
                output_dir,
                f"{basename}.npy",
            )
        
        M = np.eye(4)
        M[0:3, 0:3] = R
        M[0:3, 3] = t
        np.savetxt(pose_path, M)

        # ##########################
        # now make repre
        intrinsics = [render_camera_model.f[0], render_camera_model.f[1],
                          render_camera_model.c[0], render_camera_model.c[1]]
        dims = [render_camera_model.height, render_camera_model.width]

        res_pkg = my_render(gaussians, pipeline, background,
                            intrinsics, dims, R.T, t)

        image = (res_pkg['render'].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        depth = res_pkg['depth'].cpu().numpy().squeeze(0) * 1000
        mask = np.any(image > 0, axis=-1).astype(np.uint8) * 255
        # cv2.imshow('test', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        output = {}
        output[RenderType.COLOR] = image
        output[RenderType.DEPTH] = depth
        output[RenderType.MASK] = mask

        ys, xs = output[RenderType.MASK].nonzero()
        box = np.array(misc.calc_2d_box(xs, ys))
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
        
        trans_m2c = structs.RigidTransform(R=R, t=t[:, None]*1000)
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
        
        if opts.crop:
            # Get box for cropping.
            crop_box = misc.calc_crop_box(
                box=object_box,
                make_square=True,
            )

            # Construct a virtual camera focused on the box.
            crop_camera_model_c2w = misc.construct_crop_camera(
                box=crop_box,
                camera_model_c2w=render_camera_model_c2w,
                viewport_size=(
                    int(opts.crop_size[0] * opts.ssaa_factor),
                    int(opts.crop_size[1] * opts.ssaa_factor),
                ),
                viewport_rel_pad=opts.crop_rel_pad,
            )

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
                                crop_intrinsics, crop_dims, R.T, t)
        
            image = (res_pkg['render'].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
            depth = res_pkg['depth'].cpu().numpy().squeeze(0) * 1000
            mask = np.any(image > 0, axis=-1).astype(np.uint8) * 255

            output = {}
            output[RenderType.COLOR] = image
            output[RenderType.DEPTH] = depth
            output[RenderType.MASK] = mask

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
        
        ys, xs = output[RenderType.MASK].nonzero()
        box = np.array(misc.calc_2d_box(xs, ys))
        object_box = AlignedBox2f(
            left=box[0],
            top=box[1],
            right=box[2],
            bottom=box[3],
        )

        ### above was template, below is gen_repre
        camera_sample = camera_model_c2w.to_json()
        camera_world_from_cam = PinholePlaneCameraModel(
                width=camera_sample["ImageSizeX"],
                height=camera_sample["ImageSizeY"],
                f=(camera_sample["fx"],camera_sample["fy"]),
                c=(camera_sample["cx"],camera_sample["cy"]),
                T_world_from_eye=np.array(camera_sample["T_WorldFromCamera"])
            )

        image_arr = output[RenderType.COLOR]
        depth_image_arr = output[RenderType.DEPTH]
        mask_image_arr = output[RenderType.MASK]

        image_chw = array_to_tensor(image_arr).to(torch.float32).permute(2,0,1).to(device) / 255.0
        depth_image_hw = array_to_tensor(depth_image_arr).to(torch.float32).to(device)
        object_mask_modal = array_to_tensor(mask_image_arr).to(torch.float32).to(device)
        
        object_pose_rigid_matrix = np.eye(4)
        T_world_from_model = (
            array_to_tensor(object_pose_rigid_matrix)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_world = torch.linalg.inv(T_world_from_model)
        T_world_from_camera = (
            array_to_tensor(camera_world_from_cam.T_world_from_eye)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_camera = torch.matmul(T_model_from_world, T_world_from_camera)

        (
            feat_vectors,
            feat_to_vertex_ids,
            vertices_in_model,
        ) = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_image_hw,
            object_mask=object_mask_modal,
            camera=camera_world_from_cam,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=opts.grid_cell_size,
            debug=False,
        )

        mock_template_id = 0
        feat_to_template_ids = mock_template_id * torch.ones(
            feat_vectors.shape[0], dtype=torch.int32, device=device
        )

        image_chw_uint8 = (image_chw * 255).to(torch.uint8)

        new_camera_model = camera_world_from_cam.copy()
        new_camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)

        if not len(repre.feat_raw_projectors) == 1:
            raise RuntimeError('exprected one raw proj')
        
        pca_projector = repre.feat_raw_projectors[0]
        feat_raw_projectors = repre.feat_raw_projectors
        feat_vis_projectors = repre.feat_vis_projectors

        repre = repre_util.FeatureBasedObjectRepre(
            vertices=torch.cat([vertices_in_model]),
            feat_vectors=torch.cat([feat_vectors]),
            feat_opts=repre_util.FeatureOpts(extractor_name=opts.extractor_name),
            feat_to_vertex_ids=torch.cat([feat_to_vertex_ids]),
            feat_to_template_ids=torch.cat([feat_to_template_ids]),
            templates=torch.stack([image_chw_uint8]),
            template_cameras_cam_from_model=[new_camera_model],
        )

        feat_vectors = repre.feat_vectors
        feat_vectors = pca_projector.transform(feat_vectors)

        cluster_num = min(opts.cluster_num, feat_vectors.shape[0])

        centroids, cluster_ids, centroid_distances = cluster_util.kmeans(
            samples=feat_vectors,
            num_centroids=cluster_num,
            verbose=False,
        )

        repre.feat_cluster_centroids = centroids
        repre.feat_to_cluster_ids = cluster_ids

        if opts.debug_desc_opts is not None:
            repre.template_desc_opts = opts.debug_desc_opts

            # Calculate tf-idf descriptors.
            if opts.template_desc_opts.desc_type == "tfidf":

                assert feat_vectors is not None
                assert repre.feat_cluster_centroids is not None
                assert repre.feat_to_cluster_ids is not None
                assert repre.feat_to_template_ids is not None
                assert repre.templates is not None

                repre.template_descs, repre.feat_cluster_idfs = (
                    template_util.calc_tfidf_descriptors(
                        feat_vectors=feat_vectors,
                        feat_words=repre.feat_cluster_centroids,
                        feat_to_word_ids=repre.feat_to_cluster_ids,
                        feat_to_template_ids=repre.feat_to_template_ids,
                        num_templates=len(repre.templates),
                        tfidf_knn_k=opts.template_desc_opts.tfidf_knn_k,
                        tfidf_soft_assign=opts.template_desc_opts.tfidf_soft_assign,
                        tfidf_soft_sigma_squared=opts.template_desc_opts.tfidf_soft_sigma_squared,
                    )
                )

            else:
                raise ValueError(
                    f"Unknown template descriptor type: {opts.template_desc_opts.desc_type}"
                )

        repre.feat_raw_projectors = feat_raw_projectors
        repre.feat_vis_projectors = feat_vis_projectors
        repre.feat_vectors = feat_vectors

        repre_np = repre_util.convert_object_repre_to_numpy(repre)

        # Build a kNN index from object feature vectors.
        visual_words_knn_index = None
        if opts.match_template_type == "tfidf":
            visual_words_knn_index = knn_util.KNN(
                k=repre.template_desc_opts.tfidf_knn_k,
                metric=repre.template_desc_opts.tfidf_knn_metric
            )
            visual_words_knn_index.fit(repre.feat_cluster_centroids)

        # Build per-template KNN index with features from that template.
        template_knn_indices = []
        if opts.match_feat_matching_type == "cyclic_buddies":
            for template_id in range(len(repre.template_cameras_cam_from_model)):
                tpl_feat_mask = repre.feat_to_template_ids == template_id
                tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

                template_feats = repre.feat_vectors[tpl_feat_ids]

                # Build knn index for object features.
                template_knn_index = knn_util.KNN(k=1, metric="l2")
                template_knn_index.fit(template_feats.cpu())
                template_knn_indices.append(template_knn_index)

        match_top_n_templates = 1
        # ##################################


        # Empty unused GPU cache variables.
        if device == "cuda":
            time_start = time.time()
            torch.cuda.empty_cache()
            gc.collect()
            time_end = time.time()
            logger.info(f"Garbage collection took {time_end - time_start} seconds.")


def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    infer(opts)


if __name__ == "__main__":
    with torch.no_grad():
        main()
