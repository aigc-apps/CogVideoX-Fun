

import json
import os

import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf
from PIL import Image

from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from cogvideox.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid

# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# Config and model path
model_name          = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
sampler_name        = "DDIM_Origin"

# Load pretrained model if need
transformer_path    = None 
vae_path            = None
lora_path           = None

# Other params
sample_size         = [384, 672]
video_length        = 49
fps                 = 8

# If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
partial_video_length = None
overlap_video_length = 4

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_image_start  = "asset/1.png"
validation_image_end    = None

# prompts
prompt                  = "The dog is shaking head. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
negative_prompt         = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
lora_weight             = 0.55
save_path               = "samples/cogvideox-fun-videos_i2v"

transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
    model_name, 
    subfolder="transformer",
).to(weight_dtype)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name, 
    subfolder="vae"
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)
# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

if transformer.config.in_channels != vae.config.latent_channels:
    pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )
else:
    pipeline = CogVideoX_Fun_Pipeline.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )
if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight)

if partial_video_length is not None:
    init_frames = 0
    last_frames = init_frames + partial_video_length
    while init_frames < video_length:
        if last_frames >= video_length:
            if pipeline.vae.quant_conv.weight.ndim==5:
                mini_batch_encoder = 4
                _partial_video_length = video_length - init_frames
                _partial_video_length = int((_partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            else:
                _partial_video_length = video_length - init_frames
            
            if _partial_video_length <= 0:
                break
        else:
            _partial_video_length = partial_video_length

        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image, None, video_length=_partial_video_length, sample_size=sample_size)
        
        with torch.no_grad():
            sample = pipeline(
                prompt, 
                num_frames = _partial_video_length,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,

                video        = input_video,
                mask_video   = input_video_mask
            ).videos
        
        if init_frames != 0:
            mix_ratio = torch.from_numpy(
                np.array([float(_index) / float(overlap_video_length) for _index in range(overlap_video_length)], np.float32)
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            new_sample[:, :, -overlap_video_length:] = new_sample[:, :, -overlap_video_length:] * (1 - mix_ratio) + \
                sample[:, :, :overlap_video_length] * mix_ratio
            new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim = 2)

            sample = new_sample
        else:
            new_sample = sample

        if last_frames >= video_length:
            break

        validation_image = [
            Image.fromarray(
                (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
            ) for _index in range(-overlap_video_length, 0)
        ]

        init_frames = init_frames + _partial_video_length - overlap_video_length
        last_frames = init_frames + _partial_video_length
else:
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

    with torch.no_grad():
        sample = pipeline(
            prompt, 
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,

            video        = input_video,
            mask_video   = input_video_mask
        ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight)

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

index = len([path for path in os.listdir(save_path)]) + 1
prefix = str(index).zfill(8)

if video_length == 1:
    video_path = os.path.join(save_path, prefix + ".png")

    image = sample[0, :, 0]
    image = image.transpose(0, 1).transpose(1, 2)
    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(video_path)
else:
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_grid(sample, video_path, fps=fps)
