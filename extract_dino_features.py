import os
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import trange
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import imageio

from foundpose_utils.misc import array_to_tensor
from foundpose_utils import (
    feature_util,
)

def interpolate_to_patch_size(img_bchw, patch_size=14):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W),
                                               mode='bicubic')
    return img_bchw, target_H, target_W

def is_valid_image(filename):
    ext_test_flag = any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])
    is_file_flag = os.path.isfile(filename)
    return ext_test_flag and is_file_flag

def main(args):
    base_dir = args.source_path
    image_dir = os.path.join(base_dir, 'images')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(base_dir, 'rgb')
    assert os.path.isdir(image_dir), f"Image directory {image_dir} does not exist."
    dinov2_feat_dir = os.path.join(base_dir, 'dinov2_vits14')
    os.makedirs(dinov2_feat_dir, exist_ok=True)
    mask_dir = os.path.join(base_dir, 'masks')

    image_paths = [os.path.join(image_dir, fn) for fn in os.listdir(image_dir)]
    image_paths = [fn for fn in image_paths if is_valid_image(fn)]
    image_paths.sort()

    assert len(image_paths) > 0, f"No valid images found in {image_dir}."
    print(f"Found {len(image_paths)} images.")

    dinov2_feat_path_list = []
    mask_path_list = []
    for image_path in image_paths:
        feat_fn = os.path.splitext(os.path.basename(image_path))[0] + '.npy'
        dinov2_feat_path = os.path.join(dinov2_feat_dir, feat_fn)
        dinov2_feat_path_list.append(dinov2_feat_path)

        mask_fn = os.path.splitext(os.path.basename(image_path))[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_fn)
        mask_path_list.append(mask_path)
    
    print("Loading DINOv2 model...")
    extractor = feature_util.make_feature_extractor(args.extractor_name)
    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    for i in trange(len(image_paths)):
        orig_image_np_hwc = imageio.imread(image_path) / 255.0
        orig_mask_modal = imageio.imread(mask_path_list[i])
        orig_image_np_hwc[orig_mask_modal == 0] = 0.0
        image_tensor_chw = array_to_tensor(orig_image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
        image_tensor_bchw = image_tensor_chw.unsqueeze(0)
        image_tensor_bchw, _, _ = interpolate_to_patch_size(image_tensor_bchw)
        extractor_output = extractor(image_tensor_bchw)
        feature_map_chw = extractor_output["feature_maps"][0]
        features_chw = feature_map_chw.cpu().numpy()

        np.save(dinov2_feat_path_list[i], features_chw)

if __name__ == "__main__":
    parser = ArgumentParser("Compute reference features for feature splatting")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--extractor_name", type=str, default="dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1")
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
