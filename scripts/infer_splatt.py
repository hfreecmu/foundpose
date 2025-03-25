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
    template_util,
)

from foundpose_utils.structs import AlignedBox2f, PinholePlaneCameraModel
from foundpose_utils.misc import warp_depth_image, warp_image

import imageio
from scipy.spatial.transform import Rotation
from scipy.interpolate import RectBivariateSpline
import copy

###
from scene import GaussianModel
from utils.graphics_utils import focal2fov
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams
###

# TODO matching when patch less than 14
# Pose graph opt?

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
                  #semantic_feature=None,
                  )
    
    res_pkg = render(cam, gaussians, pipeline, background)
    return res_pkg

def rotation_distance(rot1, rot2) -> float:
    rot1 = Rotation.from_matrix(rot1)
    rot2 = Rotation.from_matrix(rot2)

    # Compute the relative rotation
    relative_rotation = rot1.inv() * rot2
    
    # Convert to angle-axis to get the geodesic distance
    angle = relative_rotation.magnitude()
    return angle

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

    # hard coding for template render
    features_patch_size: int = 14
    ssaa_factor: float = 1.0
    template_crop_size = (420, 420)

    depth_range: Tuple[int] = None
    data_dir: str = None
    splat_path: str = None
    model_path: str = None

    # rot_thresh: float = 60.0
    rot_thresh: float = -1.0

    # gauss opt
    num_opt_iters: int = 200
    opt_lr: float = 1e-3

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
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(renderer_type=renderer_type, model_path=opts.model_path)
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
    repre_orig = copy.deepcopy(repre)

    logger.info("Object representation loaded.")
    # repre_np = repre_util.convert_object_repre_to_numpy(repre)

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

        template_knn_indices_orig = copy.deepcopy(template_knn_indices)
        logger.info("Per-template KNN indices built.")

    logging.log_heading(
        logger,
        f"Object: {object_lid}, vertices: {len(repre.vertices)}",
        style=logging.WHITE_BOLD,
    )

    timer.elapsed("Time for preparing object data")

    color_dir = os.path.join(opts.data_dir, 'color')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(opts.data_dir, 'rgb')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(opts.data_dir, 'undistorted')
    mask_dir = os.path.join(opts.data_dir, 'mask')
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(opts.data_dir, 'masks')
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(opts.data_dir, 'mask_obj')
    K_path = os.path.join(opts.data_dir, 'cam_K.txt')

    K = np.loadtxt(K_path)
    orig_camera_c2w = None

    # Generate grid points at which to sample the feature vectors.
    if opts.crop:
        grid_size = opts.crop_size
    else:
        raise RuntimeError('only crop supported')
    grid_points = feature_util.generate_grid_points(
        grid_size=grid_size,
        cell_size=opts.grid_cell_size,
    )
    grid_points = grid_points.to(device)

    ### for render template
    bop_camera = dataset_params.get_camera_params(datasets_path=bop_config.datasets_path, dataset_name=opts.object_dataset)
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

    ###

    filenames = []
    for filename in os.listdir(color_dir):
        if not (filename.endswith('.png') or filename.endswith('.jpg')):
            continue

        filenames.append(filename)

    filenames = sorted(filenames)

    prev_trans = None
    for filename in filenames:

        # identifier = int(filename.split('.')[0])
        # if identifier < 22 or identifier > 26:
        #     continue

        repre_np = repre_util.convert_object_repre_to_numpy(repre)

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
        
        if orig_camera_c2w is None:
            orig_camera_c2w = PinholePlaneCameraModel(
                width=orig_image_np_hwc.shape[1],
                height=orig_image_np_hwc.shape[0],
                f=(K[0,0], K[1,1]),
                c=(K[0,2], K[1,2])
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
            top_n_templates=opts.match_top_n_templates,
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
                        "template_score": corresp_curr['template_score'].item()
                    }
                )

        # pose_candidates = []
        # pose_qualities = []
        # for cp in coarse_poses:
        #     pose_est_m2c = structs.ObjectPose(
        #         R=cp["R_m2c"], t=cp["t_m2c"]
        #     )
        #     pc_trans_c2w = camera_c2w.T_world_from_eye
        #     pc_trans_m2w = pc_trans_c2w.dot(misc.get_rigid_matrix(pose_est_m2c))

        #     pose_candidates.append(pc_trans_m2w)
        #     pose_qualities.append(cp['quality'])

        # pose_candidates = np.array(pose_candidates)
        # output_path = os.path.join(output_dir, f"{basename}_pc.npy")
        # np.save(output_path, pose_candidates)

        # if prev_trans is not None:
        #     unfilt_coarse_poses = coarse_poses
        #     coarse_poses = []
        #     debug_rot_dists = []
        #     for cp in unfilt_coarse_poses:
        #         cp_pose_est_m2c = structs.ObjectPose(
        #             R=cp["R_m2c"], t=cp["t_m2c"]
        #         )
                
        #         cp_trans_c2w = camera_c2w.T_world_from_eye
        #         cp_trans_m2w = cp_trans_c2w.dot(misc.get_rigid_matrix(cp_pose_est_m2c))

        #         curr_rot = cp_trans_m2w[0:3, 0:3]
        #         prev_rot = prev_trans[0:3, 0:3]

        #         rot_dist = rotation_distance(curr_rot, prev_rot) * 180 / np.pi

        #         debug_rot_dists.append(rot_dist)

        #         if opts.rot_thresh <= 0 or rot_dist < opts.rot_thresh:
        #             coarse_poses.append(cp)

        if len(coarse_poses) == 0:
            breakpoint()
            raise RuntimeError('need a coarse pose', filename)

        # Find the best coarse pose.
        best_coarse_quality = None
        best_coarse_pose_id = 0

        qualities = np.array([pose['quality'] for pose in coarse_poses])
        template_scores = np.array([pose['template_score'] for pose in coarse_poses])

        qual_inds = np.argsort(-qualities)
        ts_inds = np.argsort(-template_scores)

        if qual_inds[0] == template_scores[0]:
            best_coarse_pose_id = qual_inds[0]
        else:
            top_qual_ind = qual_inds[0]
            top_ts_ind = ts_inds[0]

            quals_rat = qualities[top_qual_ind] / qualities[top_ts_ind]
            if True or quals_rat >= 1.25:
                best_coarse_pose_id = top_qual_ind
            else:
                best_coarse_pose_id = top_ts_ind

        # best_coarse_quality = coarse_poses[best_coarse_pose_id]['quality']

        # for coarse_pose_id, pose in enumerate(coarse_poses):
        #     if (
        #         best_coarse_quality is None
        #         or pose["quality"] > best_coarse_quality
        #     ):
        #         best_coarse_pose_id = coarse_pose_id
        #         best_coarse_quality = pose["quality"]

        timer.elapsed("Time for coarse pose")

        timer.start()
        
        if opts.final_pose_type in [
            "best_coarse",
        ]:

            # If no successful coarse pose, continue.
            # if len(coarse_poses) == 0:
            #     raise RuntimeError('need a coarse pose', filename)

            # Select the refined pose corresponding to the best coarse pose as the final pose.
            final_pose = coarse_poses[best_coarse_pose_id]

        else:
            raise ValueError(f"Unknown final pose type {opts.final_pose_type}")

        timer.elapsed("Time for selecting final pose")

        # Visualizations and saving of results.
        vis_tiles = []

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
                    object_pose_m2w_coarse=None,
                    pose_eval_dict_coarse=None,
                    # For paper visualizations:
                    vis_for_paper=opts.vis_for_paper,
                    extractor=extractor,
                )
            
            timer.elapsed("Time for visualization")

        ext = ".png" if opts.vis_for_paper else ".jpg"
        # Assemble visualization tiles to a grid and save it.
        if len(vis_tiles):
            if repre.feat_vis_projectors[0].pca.n_components == 12:
                pca_tiles = np.vstack(vis_tiles[1:5])
                vis_tiles = np.vstack([vis_tiles[0]] + vis_tiles[5:])
                vis_grid = np.hstack([vis_tiles, pca_tiles])
            else:
                vis_grid = np.vstack(vis_tiles)
        
            vis_path = os.path.join(
                output_dir,
                f"{basename}{ext}",
            )
            inout.save_im(vis_path, vis_grid)
            logger.info(f"Visualization saved to {vis_path}")

        timer.start()

        R = trans_m2w[:3, :3]
        t = trans_m2w[:3, 3] / 1000

        intrinsics = [orig_camera_c2w.f[0], orig_camera_c2w.f[1],
                        orig_camera_c2w.c[0], orig_camera_c2w.c[1]]
        dims = [orig_camera_c2w.height, orig_camera_c2w.width]
        res_pkg = my_render(gaussians, pipeline, background,
                    intrinsics, dims, R.T, t)

        image = (res_pkg['render'].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)


        orig_image = (orig_image_np_hwc*255).astype(np.uint8)

        overlay_im = (orig_image.astype(float) / 255.0)*0.5 + (image.astype(float) / 255.0)*0.5
        overlay_im = (overlay_im.clip(0.0, 1.0) * 255).round().astype(np.uint8)

        vis_im = np.hstack((orig_image, image, overlay_im))
        vis_im = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
        # cv2.imshow('test', vis_im)
        # cv2.waitKey(1)
        vis_path = os.path.join(
                vis_dir,
                f"{basename}{ext}",
            )
        cv2.imwrite(vis_path, vis_im)
        
        pose_path = os.path.join(
                output_dir,
                f"{basename}.txt",
            )
        
        M = np.eye(4)
        M[0:3, 0:3] = R
        M[0:3, 3] = t
        np.savetxt(pose_path, M)

        prev_trans = trans_m2w

        #breakpoint()

        timer.elapsed("Time for my logic")

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
    logger: logging.Logger = logging.get_logger()
    with torch.no_grad():
        main()
