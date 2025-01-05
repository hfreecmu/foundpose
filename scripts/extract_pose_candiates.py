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

import bop_toolkit_lib.config as bop_config


from foundpose_utils import (
    corresp_util,
    config_util,
    feature_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    json_util, 
    logging,
    misc,
    structs,
)

from foundpose_utils.structs import AlignedBox2f, PinholePlaneCameraModel
from foundpose_utils.misc import warp_depth_image, warp_image

import imageio

###
from scene import GaussianModel
from utils.graphics_utils import focal2fov
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams
###

from infer_splatt import InferOpts

def extract_pose_candidates(opts: InferOpts) -> None:

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

    object_lid = 'pruners'

    # Run inference for each specified object.
    timer.start()

    # The output folder is named with slugified dataset path.
    version = opts.version
    if version == "":
        version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
    output_dir = os.path.join(
        bop_config.output_path, "pose_candidates", signature, str(object_lid)
    )
    os.makedirs(output_dir, exist_ok=True)

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

        # HERE
        timer.elapsed("Done extracting pose candidates")

        pose_candidates = []
        for cp in coarse_poses:
            pose_est_m2c = structs.ObjectPose(
                R=cp["R_m2c"], t=cp["t_m2c"]
            )
            trans_c2w = camera_c2w.T_world_from_eye
            trans_m2w = trans_c2w.dot(misc.get_rigid_matrix(pose_est_m2c))

            pose_candidates.append(trans_m2w)

        pose_candidates = np.array(pose_candidates)
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, pose_candidates)

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
    extract_pose_candidates(opts)


if __name__ == "__main__":
    logger: logging.Logger = logging.get_logger()
    with torch.no_grad():
        main()
