import os
import numpy as np
import cv2
import pytorch3d.transforms
import torch.optim as optim
from PIL import Image
import copy

from gauss_pose_renderer import render as render_cam_pose

###
import torch
from scene import GaussianModel
from utils.graphics_utils import focal2fov
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams
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
                    #semantic_feature=None,
                    )
        res_pkg = render(cam, gaussians, pipeline, background)
    else:
        cam = Camera(colmap_id=None, R=R.detach().cpu().numpy(), T=T.detach().cpu().numpy(), 
                    FoVx=FoVx, FoVy=FoVy, 
                    cx=cx, cy=cy,
                    image=dummy_image, 
                    gt_alpha_mask=None,
                    image_name=None, uid=None,
                    #semantic_feature=None,
                    )
        
        view_matrix = torch.concatenate((R.T, T.unsqueeze(1)), axis=1)
        view_matrix = torch.concatenate((view_matrix, 
                                         torch.as_tensor([0, 0, 0, 1]).unsqueeze(0).float().cuda()), axis=0)
        view_matrix = view_matrix.T

        res_pkg = render_cam_pose(view_matrix, cam, gaussians, pipeline, background)
    return res_pkg

def test(data_dir, splat_path, test_pose_path):
    gaussians = GaussianModel(3)
    gaussians.load_ply(splat_path) 

    parser = ArgumentParser()
    pipeline_par = PipelineParams(parser)
    args, _ = parser.parse_known_args()
    pipeline = pipeline_par.extract(args)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    K_path = os.path.join(data_dir, 'cam_K.txt')
    K = np.loadtxt(K_path)
    intrinsics = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    dims_path = os.path.join(data_dir, 'cam_dims.txt')
    dims = np.loadtxt(dims_path).astype(int).tolist()

    test_pose = np.loadtxt(test_pose_path)
    R = test_pose[0:3, 0:3]
    t = test_pose[0:3, 3]

    res_pkg_0 = my_render(gaussians, pipeline, background,
                        intrinsics, dims, R.T, t)
    
    R_torch = torch.from_numpy(R).float().cuda()
    t_torch = torch.from_numpy(t).float().cuda()
    
    res_pkg_1 = my_render(gaussians, pipeline, background,
                          intrinsics, dims, R_torch.T, t_torch, render_pose=True)
    
    img_0 = res_pkg_0['render']
    img_1 = res_pkg_1['render']

    depth_0 = res_pkg_0['depth']
    depth_1 = res_pkg_1['depth']
    
    print(torch.max(torch.abs(img_0 - img_1)).item())
    print(torch.max(torch.abs(depth_0 - depth_1)).item())


DATA_DIR = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/0_pruner_rotate'
SPLAT_PATH = '/home/hfreeman/harry_ws/repos/SuGaR/output/from_scratch/vanilla_gs/pruners_colmap/mask.ply'
TEST_POSE_PATH = 'output/sugar/inference/lmo_v1/pruners/000161.txt'
if __name__ == "__main__":
    with torch.no_grad():
        test(DATA_DIR, SPLAT_PATH, TEST_POSE_PATH)
