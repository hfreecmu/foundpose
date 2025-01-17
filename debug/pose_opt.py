import os
import numpy as np
import cv2
import pytorch3d.transforms
import torch.optim as optim
from PIL import Image
import copy

from foundpose_utils import (
    feature_util,
)


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

def interpolate_to_patch_size(img_bchw, patch_size=14):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W),
                                               mode='bicubic')
    return img_bchw, target_H, target_W

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

def test(data_dir, splat_path, test_pose_path, test_image_path, test_mask_path):
    gaussians = GaussianModel(3)
    gaussians.load_ply(splat_path) 

    parser = ArgumentParser()
    pipeline_par = PipelineParams(parser)
    args, _ = parser.parse_known_args()
    pipeline = pipeline_par.extract(args)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    resize_factor=2

    K_path = os.path.join(data_dir, 'cam_K.txt')
    K = np.loadtxt(K_path)
    intrinsics = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    dims_path = os.path.join(data_dir, 'cam_dims.txt')
    dims = np.loadtxt(dims_path).astype(int).tolist()

    intrinsics = [k / resize_factor for k in intrinsics]
    dims = [k // resize_factor for k in dims]

    test_pose = np.loadtxt(test_pose_path)
    R = test_pose[0:3, 0:3]
    t = test_pose[0:3, 3]

    res_pkg = my_render(gaussians, pipeline, background,
                        intrinsics, dims, R.T, t)
    
    image = (res_pkg['render'].detach().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    orig_image = cv2.imread(test_image_path)
    orig_image = cv2.resize(orig_image, (dims[1], dims[0]))
    vis_im = np.hstack((orig_image, image))
    output_path = '/home/hfreeman/Downloads/test_im.png'
    cv2.imwrite(output_path, vis_im)

    R_torch = torch.from_numpy(R).float().cuda()
    t_torch = torch.from_numpy(t).float().cuda()
    quat_torch = pytorch3d.transforms.matrix_to_quaternion(R_torch.unsqueeze(0)).squeeze(0)

    quat_torch.requires_grad = True
    t_torch.requires_grad = True
    R_torch = pytorch3d.transforms.quaternion_to_matrix(quat_torch.unsqueeze(0)).squeeze(0)
    
    res_pkg = my_render(gaussians, pipeline, background,
                        intrinsics, dims, R_torch.T, t_torch, render_pose=True)
    
    image = (res_pkg['render'].detach().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    vis_im = np.hstack((orig_image, image))
    output_path = '/home/hfreeman/Downloads/test_im_2.png'
    cv2.imwrite(output_path, vis_im)

    extractor_name = 'dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1'
    extractor = feature_util.make_feature_extractor(extractor_name)
    extractor.to('cuda')

    test_mask = cv2.imread(test_mask_path, -1).astype(float) / 255.0
    pil_mask = Image.fromarray((test_mask * 255).astype('uint8'))
    resized_pil_mask = pil_mask.resize((dims[1], dims[0]), resample=Image.NEAREST)
    test_mask_torch = torch.from_numpy(np.array(resized_pil_mask) / 255.0).float().cuda()

    test_image = copy.deepcopy(Image.open(test_image_path))
    test_image = np.array(test_image).astype(np.float) / 255.0
    test_image = test_image * test_mask[:, :, None]
    test_image = cv2.resize(test_image, (dims[1], dims[0]))
    test_image = torch.from_numpy(test_image).float().cuda().permute(2, 0, 1)
    test_image_bchw, _, _ = interpolate_to_patch_size(test_image.unsqueeze(0))

    with torch.no_grad():
       extractor_output = extractor(test_image_bchw)
    test_feature = extractor_output["feature_maps"][0]
    
    # l1_loss = torch.nn.L1Loss(reduction="none")
    # def l1_loss_fn(network_output, gt):
    #     l1_loss = torch.abs((network_output - gt)).mean()
    #     return l1_loss

    rotation_activation = torch.nn.functional.normalize

    optimizer = optim.Adam([quat_torch, t_torch], lr=1e-3)
    for iter in range(200):
        optimizer.zero_grad()

        R_torch = pytorch3d.transforms.quaternion_to_matrix(rotation_activation(quat_torch.unsqueeze(0))).squeeze(0)
        res_pkg = my_render(gaussians, pipeline, background,
                        intrinsics, dims, R_torch.T, t_torch, render_pose=True)
        pred_image = res_pkg['render'] * test_mask_torch[None]

        image_tensor_bchw, _, _ = interpolate_to_patch_size(pred_image.unsqueeze(0))
        extractor_output = extractor(image_tensor_bchw)
        feature_map_chw = extractor_output["feature_maps"][0]

        norm_loss = torch.linalg.norm(feature_map_chw - test_feature, dim=0)
        loss = torch.mean(norm_loss)
        
        # alpha = res_pkg['alpha'][0]

        # loss = l1_loss(alpha, test_mask)
        # loss = torch.sum(loss) / torch.sum(test_mask > 0.0)

        # loss = l1_loss_fn(alpha, test_mask)
        # loss = l1_loss_fn(res_pkg['render'] * test_mask[None], test_image * test_mask[None])

        loss.backward()
        optimizer.step()

        if (iter + 1) % 10 == 0:
            print('Done ', iter, loss.item())

    R_torch = pytorch3d.transforms.quaternion_to_matrix(rotation_activation(quat_torch.unsqueeze(0))).squeeze(0)
    res_pkg = my_render(gaussians, pipeline, background,
                        intrinsics, dims, R_torch.T, t_torch, render_pose=True)
    
    image = (res_pkg['render'].detach().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    vis_im = np.hstack((orig_image, image))
    output_path = '/home/hfreeman/Downloads/test_im_3.png'
    cv2.imwrite(output_path, vis_im)
    breakpoint()

DATA_DIR = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/0_pruner_rotate'
SPLAT_PATH = '/home/hfreeman/harry_ws/repos/feature-3dgs/output/pruners/mask.ply'
TEST_POSE_PATH = 'output/inference/lmo_v1/pruners/000161.txt'
TEST_IMAGE_PATH = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/0_pruner_rotate/undistorted/000161.jpg'
TEST_MASK_PATH = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/0_pruner_rotate/mask_obj/000161.png'
if __name__ == "__main__":
    test(DATA_DIR, SPLAT_PATH, TEST_POSE_PATH, TEST_IMAGE_PATH, TEST_MASK_PATH)
breakpoint()