import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
import cv2
import bop_toolkit_lib.config as bop_config

from foundpose_utils import (
    config_util,
    logging,
    misc
)

from infer_splatt import InferOpts, my_render

###
from scene import GaussianModel
from utils.graphics_utils import focal2fov
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams
###

def ensure_quaternion_continuity(quaternions):
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i], quaternions[i - 1]) < 0:
            quaternions[i] *= -1
    return quaternions

def filter_poses(opts: InferOpts) -> None:
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    logger.info('Setting up')

    gaussians = GaussianModel(3)
    gaussians.load_ply(opts.splat_path) 

    parser = ArgumentParser()
    pipeline_par = PipelineParams(parser)
    args, _ = parser.parse_known_args()
    pipeline = pipeline_par.extract(args)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    object_lid = 'pruners'

    version = opts.version
    if version == "":
        raise RuntimeError('Expected version')
    signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
    infer_dir = os.path.join(
        bop_config.output_path, "inference", signature, str(object_lid)
    )

    output_dir = os.path.join(
        bop_config.output_path, "filtered_poses", signature, str(object_lid)
    )
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(
        bop_config.output_path, "filtered_poses", signature, str(object_lid) + '_vis'
    )
    os.makedirs(vis_dir, exist_ok=True)

    color_dir = os.path.join(opts.data_dir, 'color')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(opts.data_dir, 'rgb')
    if not os.path.exists(color_dir):
        color_dir = os.path.join(opts.data_dir, 'undistorted')

    K_path = os.path.join(opts.data_dir, 'cam_K.txt')
    K = np.loadtxt(K_path)
    intrinsics = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    dims_path = os.path.join(opts.data_dir, 'cam_dims.txt')
    dims = np.loadtxt(dims_path).astype(int).tolist()

    logger.info('Loading data')

    filenames = []
    for filename in os.listdir(infer_dir):
        if not filename.endswith('.txt'):
            continue

        filenames.append(filename)

    filenames = sorted(filenames)

    ts = []
    quats = []
    for filename in filenames:
        M = np.loadtxt(os.path.join(infer_dir, filename))
        R = M[0:3, 0:3]
        quat = Rotation.from_matrix(R).as_quat()

        ts.append(M[0:3, 3])
        quats.append(quat)

    quats = ensure_quaternion_continuity(quats)

    logger.info('Smoothing poses')

    # smoothed_translations = savgol_filter(ts, 13, 2, axis=0)
    # smoothed_quaternions = savgol_filter(quats, 13, 2, axis=0)
    smoothed_translations = np.copy(ts)
    smoothed_quaternions = np.copy(quats)
    
    smoothed_quaternions /= np.linalg.norm(smoothed_quaternions, axis=1, keepdims=True)

    logger.info('Saving results')

    for st, sq, filename in zip(smoothed_translations, smoothed_quaternions, filenames):
        sR = Rotation.from_quat(sq).as_matrix()

        smoothed_pose = np.eye(4)
        smoothed_pose[0:3, 0:3] = sR
        smoothed_pose[0:3, 3] = st

        output_path = os.path.join(output_dir, filename)
        np.savetxt(output_path, smoothed_pose)

        res_pkg = my_render(gaussians, pipeline, background,
                            intrinsics, dims, sR.T, st)

        image = (res_pkg['render'].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        orig_image_name = filename.replace('.txt', '.png')
        orig_image_path = os.path.join(color_dir, orig_image_name)
        if not os.path.exists(orig_image_path):
            orig_image_name = filename.replace('.txt', '.jpg')
            orig_image_path = os.path.join(color_dir, orig_image_name)

        orig_image = cv2.imread(orig_image_path)
        vis_im = np.hstack((orig_image, image))

        vis_path = os.path.join(vis_dir, filename.replace('.txt', '.png'))
        cv2.imwrite(vis_path, vis_im)

def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    filter_poses(opts)

if __name__ == "__main__":
    with torch.no_grad():
        main()
        


    