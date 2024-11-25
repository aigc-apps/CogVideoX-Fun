import json
import os

import cv2
import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf
from PIL import Image
from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                          T5EncoderModel, T5Tokenizer)

from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from cogvideox.pipeline.pipeline_cogvideox_inpaint import \
    CogVideoX_Fun_Pipeline_Inpaint
from cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from cogvideox.utils.utils import get_video_to_video_latent, save_videos_grid

# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# model path
model_name          = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
sampler_name        = "DDIM_Origin"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None
# Other params
sample_size         = [384, 672]
# V1.0 and V1.1 support up to 49 frames of video generation,
# while V1.5 supports up to 85 frames.  
video_length        = 49
fps                 = 8

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you are preparing to redraw the reference video, set validation_video and validation_video_mask. 
# If you do not use validation_video_mask, the entire video will be redrawn; 
# if you use validation_video_mask, only a portion of the video will be redrawn.
# Please set a larger denoise_strength when using validation_video_mask, such as 1.00 instead of 0.70
validation_video        = "asset/1.mp4"
validation_video_mask   = None 
denoise_strength        = 0.70

# prompts
prompt                  = "A cute cat is playing the guitar. "
negative_prompt         = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
lora_weight             = 0.55
save_path               = "samples/cogvideox-fun-videos_v2v"

transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
    model_name, 
    subfolder="transformer",
    low_cpu_mem_usage=True,
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

video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
if video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
    additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
    video_length += additional_frames * vae.config.temporal_compression_ratio
input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=sample_size, validation_video_mask=validation_video_mask, fps=fps)

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

        video       = input_video,
        mask_video  = input_video_mask,
        strength    = denoise_strength,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight)
    
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

index = len([path for path in os.listdir(save_path)]) + 1
prefix = str(index).zfill(8)
    
if video_length == 1:
    save_sample_path = os.path.join(save_path, prefix + f".png")

    image = sample[0, :, 0]
    image = image.transpose(0, 1).transpose(1, 2)
    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_sample_path)
else:
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_grid(sample, video_path, fps=fps)