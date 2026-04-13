import yaml
with open("config/inference_config.yml") as f:
    infer_cfg = yaml.safe_load(f)

import sys
sys.path.append(infer_cfg["raft_path"])
import torch
from core.utils.utils import InputPadder

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict

def compute_optical_flow(model, img1, img2):
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    _, flow = model(img1, img2, iters=20, test_mode=True)
    return flow  # shape: [1, 2, H, W]

def pre_compute_optical_flow(model, img1, img2):
    img1 = (img1.to("cuda", dtype=torch.float16)*2 - 1).clamp(-1,1)
    img2 = (img2.to("cuda", dtype=torch.float16)*2 - 1).clamp(-1,1)
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    _, flow = model(img1, img2, iters=20, test_mode=True)
    flow = padder.unpad(flow)
    return flow  # shape: [1, 2, H, W]

def compute_ref_patch_coords_from_LR(flow, tile_size=1024, tile_overlap=64):
    stride = tile_size - tile_overlap
    _, _, H, W = flow.shape

    ref_patch_coords = []

    dx = flow[0, 0]
    dy = flow[0, 1]

    for i in range(0, H - tile_size + 1, stride):
        for j in range(0, W - tile_size + 1, stride):
            patch_dx = dx[i:i+tile_size, j:j+tile_size]
            patch_dy = dy[i:i+tile_size, j:j+tile_size]
            
            mean_dx = patch_dx.mean().item()
            mean_dy = patch_dy.mean().item()

            ref_center_x = j + tile_size // 2 + mean_dx
            ref_center_y = i + tile_size // 2 + mean_dy

            ref_top_left_x = int(ref_center_x - tile_size // 2)
            ref_top_left_y = int(ref_center_y - tile_size // 2)

            ref_top_left_x = max(0, min(ref_top_left_x, W - tile_size))
            ref_top_left_y = max(0, min(ref_top_left_y, H - tile_size))

            ref_patch_coords.append((ref_top_left_y, ref_top_left_x))

    return ref_patch_coords