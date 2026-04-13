import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import yaml
with open("config/inference_config.yml") as f:
    infer_cfg = yaml.safe_load(f)

sys.path.append('.')
sys.path.append(os.path.join(infer_cfg["raft_path"], "core"))
from accelerate.utils import set_seed
set_seed(42, deterministic=True)
import copy
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from models.transformer_sd3_fusem import SD3TransformerFuseMotion2DModel
from diffusers.image_processor import  VaeImageProcessor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_timesteps
import torch
from torchvision import transforms
import torch.nn.functional as F
import argparse
from argparse import Namespace
from tqdm import tqdm
from raft import RAFT
from data.data_ref_workers import Real_ESRGANRef_VAL_Dataset_Steps
from live_utils.util_hooks import add_hook
from live_utils.optical_utils import remove_module_prefix, compute_optical_flow, pre_compute_optical_flow, compute_ref_patch_coords_from_LR
from live_utils.motion_network import MotionEncoder
from live_utils.wavelet_color_fix import wavelet_color_fix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/sample")
    parser.add_argument("--output_dir_name", type=str, default="output_dir_name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None)
    return parser.parse_args()

def generate_c2_weight(tile_size):
    """generate C² Blending mask"""
    H = W = tile_size
    y = torch.linspace(-1, 1, H).view(-1, 1).expand(H, W)
    x = torch.linspace(-1, 1, W).view(1, -1).expand(H, W)
    mask = (1 - x ** 2) * (1 - y ** 2)
    return mask.clamp(min=0)[None, None]  # shape: [1, 1, H, W]

def pad_image_to_fit(image, tile_size, tile_overlap, mode="reflect"):
    _, _, H, W = image.shape
    stride = tile_size - tile_overlap

    pad_h = (stride - (H - tile_overlap) % stride) % stride
    pad_w = (stride - (W - tile_overlap) % stride) % stride

    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    return padded, (H, W)

def split_into_tiles(image, tile_size, tile_overlap):
    _, _, h, w = image.shape
    tiles = []
    indices = []
    
    for i in range(0, h - tile_overlap, tile_size - tile_overlap):
        for j in range(0, w - tile_overlap, tile_size - tile_overlap):
            tile = image[:, :, i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
            indices.append((i, j))

    return tiles, indices

def stitch_tiles_back(tiles, indices, tile_size, tile_overlap, orig_size=None, weight_mask=None):
    h_max = max([i[0] for i in indices]) + tile_size
    w_max = max([i[1] for i in indices]) + tile_size
    
    stitched_image = torch.zeros(tiles[0].shape[0], 3, h_max, w_max).to(tiles[0].device)
    count_map = torch.zeros_like(stitched_image)

    for tile, (i, j) in zip(tiles, indices):
        _, _, h, w = tile.shape
        mask = weight_mask[:, :, :h, :w] if weight_mask is not None else 1.0
        stitched_image[:, :, i:i+h, j:j+w] += tile * mask
        count_map[:, :, i:i+h, j:j+w] += mask

    count_map[count_map == 0] = 1
    result = stitched_image / (count_map + 1e-8)

    if orig_size is not None:
        H, W = orig_size
        result = result[:, :, :H, :W]

    return result

        
def collate_fn(examples, weight_dtype=torch.float16):
    lr_img = [example["lr_img"] for example in examples]
    ref_img = [example["ref_img"] for example in examples]
    ref_lr_img = [example["ref_lr_img"] for example in examples]
    prompt_embeds = torch.stack([example["prompt_embeds_input"] for example in examples])
    pooled_prompt_embeds = torch.stack([example["pooled_prompt_embeds_input"] for example in examples])
    img_name = [example["img_name"] for example in examples]
    
    lr_img = torch.stack(lr_img)
    ref_img = torch.stack(ref_img)
    ref_lr_img = torch.stack(ref_lr_img)

    batch = {
        "lr_img": lr_img.to(dtype=weight_dtype),
        "ref_img": ref_img.to(dtype=weight_dtype),
        "ref_lr_img": ref_lr_img.to(dtype=weight_dtype),
        "prompt_embeds": prompt_embeds.to(dtype=weight_dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds.to(dtype=weight_dtype),
        "img_name": img_name[0],
             }
    return batch

def tile_sample(lq, ref, ref_lq, flow, transformer, transformer_ref, raft_model, motion_encoder, noise_scheduler, timesteps, null_prompt_embeds_input, null_pooled_prompt_embeds_input):
    with torch.no_grad():
        layers = []

        for i in range(24):
            base_name = f'transformer_blocks.{i}.attn'
            layers.append(f'{base_name}.to_k')
            layers.append(f'{base_name}.to_v')
            layers.append(f'{base_name}.add_k_proj')
            layers.append(f'{base_name}.add_v_proj')
        
        tile_size = 1024
        tile_overlap = 64
        

        lq_padded, orig_size = pad_image_to_fit(lq, tile_size, tile_overlap)
        image_tiles, tile_indices = split_into_tiles(lq_padded, tile_size, tile_overlap)
        
        ref_padded, _ = pad_image_to_fit(ref, tile_size, tile_overlap)
        ref_lq_padded, _ = pad_image_to_fit(ref_lq, tile_size, tile_overlap)
        flow, _ = pad_image_to_fit(flow, tile_size, tile_overlap, mode="constant")
        
        ref_patch_coords = compute_ref_patch_coords_from_LR(flow, tile_size, tile_overlap)

        processed_tiles = []

        for i, (image_tile, (y, x)) in enumerate(zip(image_tiles, ref_patch_coords)):
            print(f"Processing tile {i+1}/{len(image_tiles)}...")
            
            pixel_values = image_tile.to("cuda", dtype=weight_type)
            pixel_values = (pixel_values*2 -1).clamp(-1,1)
            
            ref_tile = ref_padded[:, :, y:y+tile_size, x:x+tile_size]
            ref_lq_tile = ref_lq_padded[:, :, y:y+tile_size, x:x+tile_size]
            pixel_values_ref = ref_tile.to("cuda", dtype=weight_type)
            pixel_values_ref = (pixel_values_ref*2 -1).clamp(-1,1)
            pixel_values_ref_lr = ref_lq_tile.to("cuda", dtype=weight_type)
            pixel_values_ref_lr = (pixel_values_ref_lr*2 -1).clamp(-1,1)
            
            flow = compute_optical_flow(raft_model, pixel_values_ref_lr, pixel_values)
                
            flow = flow.to("cuda", dtype=weight_type)
            
            optical_ref = motion_encoder(flow)
            optical_ref = [p.to(dtype=weight_type) for p in optical_ref]
            
            model_input = (vae.encode(pixel_values).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
            model_input = model_input.to("cuda", dtype=weight_type)
            model_input_ref = (vae.encode(pixel_values_ref).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
            model_input_ref = model_input_ref.to("cuda", dtype=weight_type)
            
            noise_scheduler_copy = copy.deepcopy(noise_scheduler)

            noise = torch.randn_like(model_input).to("cuda", dtype=weight_type)
            latents = 0.8 * model_input + 0.2 * noise
            
            features = {}
            add_hook(transformer_ref, features, layers)

            for step, t in enumerate(timesteps):
                features.clear()
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                
                model_pred_ref = transformer_ref(
                    hidden_states=model_input_ref.detach(),
                    timestep=timestep,
                    encoder_hidden_states=null_prompt_embeds_input,
                    pooled_projections=null_pooled_prompt_embeds_input,
                    return_dict=False,
                )[0].detach()

                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=null_prompt_embeds_input,
                    pooled_projections=null_pooled_prompt_embeds_input,
                    kvs = features,
                    optical_ref = optical_ref,
                    return_dict=False,
                )[0]
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = noise_scheduler_copy.step(model_pred, t, latents, return_dict=False)[0]

            latents = latents.to(dtype=vae.dtype)
            output_tile = vae.decode((latents / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
            processed_tiles.append(output_tile)

        weight_mask = generate_c2_weight(tile_size).to("cuda")
        image = stitch_tiles_back(processed_tiles, tile_indices, tile_size, tile_overlap, orig_size, weight_mask)
            
    return image

def main(args, index):
    global num_inference_steps
    with torch.no_grad():
        batch = collate_fn([valid_data[index]],weight_dtype=weight_type)
        pixel_values = batch["lr_img"]
        pixel_values_ref = batch["ref_img"]
        pixel_values_ref_lr = batch["ref_lr_img"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        img_name = batch["img_name"]
        # cal on device
        pixel_values = pixel_values.to(args.device)
        pixel_values_ref = pixel_values_ref.to(args.device)
        pixel_values_ref_lr = pixel_values_ref_lr.to(args.device)
        prompt_embeds = prompt_embeds.to(args.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(args.device)
        
        flow = pre_compute_optical_flow(raft_model, pixel_values, pixel_values_ref_lr)
        
        scale = infer_cfg["ref_res"] / infer_cfg["lr_res"]
        
        _, _, height, width = pixel_values.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        pixel_values =  F.interpolate(pixel_values, size=(new_height, new_width), mode='bicubic')
        pixel_values_ref_lr = F.interpolate(pixel_values_ref_lr, size=(new_height, new_width), mode='bicubic')
        flow = F.interpolate(flow, size=(new_height, new_width), mode='bicubic')
        flow[:, 0] *= new_width / width
        flow[:, 1] *= new_height / height
        
        
        mu = None

        scheduler_kwargs = {}
        if noise_scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = pixel_values.shape
            image_seq_len = (int(height) // image_processor.vae_scale_factor // transformer_ref.config.patch_size) * (
                int(width) // image_processor.vae_scale_factor // transformer_ref.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                noise_scheduler.config.get("base_image_seq_len", 256),
                noise_scheduler.config.get("max_image_seq_len", 4096),
                noise_scheduler.config.get("base_shift", 0.5),
                noise_scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            noise_scheduler,
            num_inference_steps,
            args.device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        
        timesteps = timesteps.to(device=pixel_values.device)
        
        
        image = tile_sample(pixel_values, pixel_values_ref, pixel_values_ref_lr, flow, transformer, transformer_ref, raft_model, motion_encoder, noise_scheduler, timesteps, prompt_embeds, pooled_prompt_embeds)
        image = image_processor.postprocess(image.clamp(-1,1).cpu())[0]     
        lr = transforms.ToPILImage()(pixel_values.squeeze(0).detach().cpu())
        image_pil_image = wavelet_color_fix(target=image, source=lr)
        return lr, image_pil_image , img_name

if __name__ == "__main__":
    args = parse_args()
    weight_type = torch.float16
    trained_model_path = infer_cfg["pretrained_weight_path"]
    pretrained_raft_model_path = infer_cfg["raft_weight_path"]
    motion_encoder_model_path = infer_cfg["motion_encoder_weight_path"]

    # load model
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(infer_cfg["SD3_weight_path"],subfolder="scheduler", torch_dtype=weight_type)
    transformer = SD3TransformerFuseMotion2DModel.from_pretrained(trained_model_path,subfolder="transformer", torch_dtype=weight_type)
    transformer_ref = SD3Transformer2DModel.from_pretrained(trained_model_path, subfolder="transformer_ref", torch_dtype=weight_type)
    vae = AutoencoderKL.from_pretrained(infer_cfg["SD3_weight_path"], subfolder="vae", torch_dtype=weight_type)
    image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    
    raft_args = Namespace(
        model=None,
        path=None,
        small=False,
        mixed_precision=True,
        alternate_corr=False
    )
    raft_model = RAFT(raft_args)
    raft_state_dict = remove_module_prefix(torch.load(pretrained_raft_model_path, map_location="cpu"))
    raft_model.load_state_dict(raft_state_dict)
    motion_encoder = MotionEncoder()
    motion_encoder_state_dict = torch.load(motion_encoder_model_path, map_location="cpu")
    motion_encoder.load_state_dict(motion_encoder_state_dict)
        
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    transformer_ref.requires_grad_(False)
    raft_model.requires_grad_(False)
    motion_encoder.requires_grad_(False)

    vae = vae.to(args.device,dtype=weight_type)
    transformer = transformer.to(args.device,dtype=weight_type)
    transformer_ref = transformer_ref.to(args.device,dtype=weight_type)
    raft_model = raft_model.to(args.device,dtype=weight_type)
    motion_encoder = motion_encoder.to(args.device,dtype=weight_type)
    
    lr_path = infer_cfg["lr_path"]
    ref_path = infer_cfg["ref_path"]
    ref_lr_path = infer_cfg["ref_lr_path"]
    valid_data = Real_ESRGANRef_VAL_Dataset_Steps(root_dir_path=lr_path, ref_dir_path=ref_path, ref_lr_dir_path=ref_lr_path, device=args.device)
    
    num_inference_steps = 6
    
    indices = torch.linspace(923, noise_scheduler.config.num_train_timesteps-1, num_inference_steps)
    indices = indices.long()
    sigmas = noise_scheduler.sigmas[indices].to(device="cpu")
    
    noise_scheduler.set_shift(1.0)

    datalen = valid_data.datalen
    for dataset_index in datalen:
        if dataset_index == datalen[0]:
            name = "demo"
            pre_dataset_index = 0
        for index in tqdm(range(pre_dataset_index, dataset_index), desc=f"Dataset {name} : {pre_dataset_index} to {dataset_index}"):
            batch = collate_fn([valid_data[index]], weight_dtype=weight_type)
            img_name = batch["img_name"]
            base_name = img_name.split('.')[0]

            save_path = os.path.join(args.output_dir, f"{name}") 
            save_path_sr = os.path.join(save_path, args.output_dir_name)
            if not os.path.exists(save_path_sr):
                os.makedirs(save_path_sr)

            lr , image_pil_image , img_name = main(args, index)
            out_path = os.path.join(save_path_sr, f"{base_name}.png")
            image_pil_image.save(out_path)