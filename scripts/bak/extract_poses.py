import os
import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation
import torch
import cv2

import bop_toolkit_lib.config as bop_config
from foundpose_utils import (
    config_util,
    misc
)

from infer_splatt import InferOpts

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

def rotation_distance(rot1, rot2) -> float:
    rot1 = Rotation.from_matrix(rot1)
    rot2 = Rotation.from_matrix(rot2)

    # Compute the relative rotation
    relative_rotation = rot1.inv() * rot2
    
    # Convert to angle-axis to get the geodesic distance
    angle = relative_rotation.magnitude()
    return angle

def extract_poses(opts: InferOpts) -> None:
    object_lid = 'pruners'
    version = opts.version
    if version == "":
        raise RuntimeError('opts needs version')
    signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
    pose_dir = os.path.join(
        bop_config.output_path, "inference", signature, str(object_lid)
    )

    output_dir = os.path.join(
        bop_config.output_path, "graph_search", signature, str(object_lid)
    )
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(
        bop_config.output_path, "graph_search", signature, str(object_lid) + '_vis'
    )
    os.makedirs(vis_dir, exist_ok=True)

    color_dir = os.path.join(opts.data_dir, 'color')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(opts.data_dir, 'rgb')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(opts.data_dir, 'undistorted')
    
    filenames = []
    for filename in os.listdir(pose_dir):
        if not filename.endswith('pc.npy'):
            continue

        filenames.append(filename)
    
    filenames = sorted(filenames)

    num_images = len(filenames)
    pose_dict = {}
    G = nx.DiGraph()

    for f_idx, filename in enumerate(filenames):
        pose_candidates = np.load(os.path.join(pose_dir, filename))

        pose_dict[f_idx] = pose_candidates

        for pc_idx, pc in enumerate(pose_candidates):
            G.add_node((f_idx, pc_idx))

    
    for f_idx in range(num_images - 1):
        for pc_idx, pc in enumerate(pose_dict[f_idx]):
            for next_pc_idx, next_pc in enumerate(pose_dict[f_idx + 1]):
                distance = rotation_distance(pc[0:3, 0:3], next_pc[0:3, 0:3]) * 180 / np.pi

                # if distance > rot_thresh:
                #     continue

                G.add_edge((f_idx, pc_idx), (f_idx + 1, next_pc_idx), weight=distance)


    root = "root"
    for pc_idx in range(pose_dict[0].shape[0]):
        G.add_edge(root, (0, pc_idx), weight=0)

    tail = "tail"
    for pc_idx in range(pose_dict[num_images - 1].shape[0]):
        G.add_edge((num_images - 1, pc_idx), tail, weight=0)

    shortest_path = nx.shortest_path(G, source=root, target=tail, weight='weight')

    gaussians = GaussianModel(3)
    gaussians.load_ply(opts.splat_path) 

    parser = ArgumentParser()
    pipeline_par = PipelineParams(parser)
    args, _ = parser.parse_known_args()
    pipeline = pipeline_par.extract(args)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    K_path = os.path.join(opts.data_dir, 'cam_K.txt')
    K = np.loadtxt(K_path)
    intrinsics = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    dims_path = os.path.join(opts.data_dir, 'cam_dims.txt')
    dims = np.loadtxt(dims_path).astype(int).tolist()

    for node in shortest_path:
        if node in ['root', 'tail']:
            continue

        f_idx, pc_idx = node
        M = pose_dict[f_idx][pc_idx]

        M[0:3, 3] /= 1000
        R = M[0:3, 0:3]
        t = M[0:3, 3]

        res_pkg = my_render(gaussians, pipeline, background,
                            intrinsics, dims, R.T, t)
        
        image = (res_pkg['render'].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        filename = filenames[f_idx]
        filename = filename.replace('_pc.npy', '.npy')

        output_path = os.path.join(output_dir, filename)
        np.savetxt(output_path, M)

        orig_image_name = filename.replace('.npy', '.png')
        orig_image_path = os.path.join(color_dir, orig_image_name)
        if not os.path.exists(orig_image_path):
            orig_image_name = filename.replace('.npy', '.jpg')
            orig_image_path = os.path.join(color_dir, orig_image_name)

        orig_image = cv2.imread(orig_image_path)
        vis_im = np.hstack((orig_image, image))

        vis_path = os.path.join(vis_dir, filename.replace('.npy', '.png'))
        cv2.imwrite(vis_path, vis_im)

def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    extract_poses(opts)

if __name__ == "__main__":
    with torch.no_grad():
        main()