import os
import numpy as np
import torch.optim as optim
import pytorch3d.transforms
import cv2

import bop_toolkit_lib.config as bop_config
from foundpose_utils import (
    config_util,
    misc
)

from infer_splatt import InferOpts

###
import torch
from scene import GaussianModel
from utils.graphics_utils import focal2fov
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams

from gauss_pose_renderer import render as render_cam_pose
###

def my_render(gaussians, pipeline, background, intrinsics, dims, R, T,
              render_pose=False):
    fx, fy, cx, cy = intrinsics
    image_height, image_width = dims

    FoVx = focal2fov(fx, image_width)
    FoVy = focal2fov(fy, image_height)

    cx = (cx - image_width / 2) / image_width * 2
    cy = (cy - image_height / 2) / image_height * 2

    dummy_image = torch.ones(3, image_height, image_width).float().to('cuda')
    
    if not render_pose:
        cam = Camera(colmap_id=None, R=R, T=T, 
                    FoVx=FoVx, FoVy=FoVy, 
                    cx=cx, cy=cy,
                    image=dummy_image, 
                    gt_alpha_mask=None,
                    image_name=None, uid=None,
                    semantic_feature=None,
                    )
        res_pkg = render(cam, gaussians, pipeline, background)
    else:
        cam = Camera(colmap_id=None, R=R.detach().cpu().numpy(), T=T.detach().cpu().numpy(), 
                    FoVx=FoVx, FoVy=FoVy, 
                    cx=cx, cy=cy,
                    image=dummy_image, 
                    gt_alpha_mask=None,
                    image_name=None, uid=None,
                    semantic_feature=None,
                    )
        
        view_matrix = torch.concatenate((R.T, T.unsqueeze(1)), axis=1)
        view_matrix = torch.concatenate((view_matrix, 
                                         torch.as_tensor([0, 0, 0, 1]).unsqueeze(0).float().cuda()), axis=0)
        view_matrix = view_matrix.T

        res_pkg = render_cam_pose(view_matrix, cam, gaussians, pipeline, background)
    return res_pkg

rotation_activation = torch.nn.functional.normalize
l1_loss = torch.nn.L1Loss(reduction="none")

def optim_poses(opts: InferOpts) -> None:
    object_lid = 'pruners'
    version = opts.version
    if version == "":
        raise RuntimeError('opts needs version')
    signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
    pose_dir = os.path.join(
        bop_config.output_path, "inference", signature, str(object_lid)
    )

    output_dir = os.path.join(
        bop_config.output_path, "poses_opt", signature, str(object_lid)
    )
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(
        bop_config.output_path, "poses_opt", signature, str(object_lid) + '_vis'
    )
    os.makedirs(vis_dir, exist_ok=True)

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

    gt_mask_dir = os.path.join(opts.data_dir, 'mask_obj')
    color_dir = os.path.join(opts.data_dir, 'undistorted')

    for filename in os.listdir(pose_dir):
        if not filename.endswith('.txt'):
            continue

        pose_path = os.path.join(pose_dir, filename)
        gt_mask_path = os.path.join(gt_mask_dir, filename.replace('.txt', '.png'))
        orig_image_path = os.path.join(color_dir, filename.replace('.txt', '.jpg'))

        M = np.loadtxt(pose_path)
        R = M[0:3, 0:3]
        t = M[0:3, 3]
        R_torch = torch.from_numpy(R).float().cuda()
        t_torch = torch.from_numpy(t).float().cuda()
        quat_torch = pytorch3d.transforms.matrix_to_quaternion(R_torch.unsqueeze(0)).squeeze(0)

        gt_mask = cv2.imread(gt_mask_path, -1).astype(float) / 255.0
        gt_mask = torch.from_numpy(gt_mask).float().cuda()

        quat_torch.requires_grad = True
        t_torch.requires_grad = True
        optimizer = optim.Adam([quat_torch, t_torch], opts.opt_lr)

        for iter in range(opts.num_opt_iters):
            optimizer.zero_grad()

            R_torch = pytorch3d.transforms.quaternion_to_matrix(rotation_activation(quat_torch.unsqueeze(0))).squeeze(0)
            res_pkg = my_render(gaussians, pipeline, background,
                        intrinsics, dims, R_torch.T, t_torch, render_pose=True)
            
            alpha = res_pkg['alpha'][0]

            loss = l1_loss(alpha, gt_mask)
            loss = torch.sum(loss) / torch.sum(gt_mask > 0.0)

            loss.backward()
            optimizer.step()

            if (iter + 1) % 20 == 0:
                print('Done ', filename, iter, loss.item())

        R_torch = pytorch3d.transforms.quaternion_to_matrix(rotation_activation(quat_torch.unsqueeze(0))).squeeze(0)
        R = R_torch.detach().cpu().numpy()
        t = t_torch.detach().cpu().numpy()

        M = np.eye(4)
        M[0:3, 0:3] = R
        M[0:3, 3] = t

        output_path = os.path.join(output_dir, filename)
        np.savetxt(output_path, M)

        with torch.no_grad():
            res_pkg = my_render(gaussians, pipeline, background,
                                intrinsics, dims, R.T, t)
        
        image = (res_pkg['render'].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        orig_image = cv2.imread(orig_image_path)
        vis_im = np.hstack((orig_image, image))

        vis_path = os.path.join(vis_dir, filename.replace('.txt', '.png'))
        cv2.imwrite(vis_path, vis_im)

        exit(0)

def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    optim_poses(opts)

if __name__ == "__main__":
    main()