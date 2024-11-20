"""Modified from CogVideoX-Fun/scripts/train_lora.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import logging
import math
import os
import shutil
import sys
import json
import random
from contextlib import contextmanager
from typing import List, Optional, Union

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from decord import VideoReader
from einops import rearrange
from packaging import version
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

import cogvideox.reward.reward_fn as reward_fn
from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.pipeline.pipeline_cogvideox_inpaint import \
    CogVideoX_Fun_Pipeline_Inpaint, get_resize_crop_region_for_grid
from cogvideox.utils.lora_utils import create_network, merge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@contextmanager
def video_reader(*args, **kwargs):
    """A context manager to solve the memory leak of decord.
    """
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def log_validation(vae, text_encoder, tokenizer, transformer3d, network, loss_fn, args, accelerator, weight_dtype, global_step, validation_prompts_idx):
    logger.info("Running validation... ")

    transformer3d_val = CogVideoXTransformer3DModel.from_pretrained_2d(
        args.pretrained_model_name_or_path, subfolder="transformer",
    ).to(weight_dtype)
    transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    if args.vae_gradient_checkpointing:
        # Initialize a new vae if gradient checkpointing is enabled.
        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        ).to(weight_dtype)
    pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae if args.vae_gradient_checkpointing else accelerator.unwrap_model(vae).to(weight_dtype),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=transformer3d_val,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline = merge_lora(
        pipeline, None, 1, accelerator.device, state_dict=accelerator.unwrap_model(network).state_dict(), transformer_only=True
    )

    to_tensor = torchvision.transforms.ToTensor()
    validation_loss, validation_reward = 0, 0
    for i in range(len(validation_prompts_idx)):
        validation_idx, validation_prompt = validation_prompts_idx[i]
        logger.info(f"Process index: {accelerator.process_index}, validation_idx: {validation_idx}, validation_prompt: {validation_prompt}")
        with torch.no_grad():
            with torch.autocast("cuda", dtype=weight_dtype):
                video_length = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_length != 1 else 1
                input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[args.validation_sample_height, args.validation_sample_width])
                
                if args.seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                
                sample = pipeline(
                    validation_prompt, 
                    num_frames = video_length,
                    negative_prompt = "bad detailed",
                    height      = args.validation_sample_height,
                    width       = args.validation_sample_width,
                    guidance_scale = 7,
                    generator   = generator, 

                    video        = input_video,
                    mask_video   = input_video_mask,
                ).videos
                sample_saved_path = os.path.join(args.output_dir, f"validation_sample/sample-{global_step}-{validation_idx}.mp4")
                save_videos_grid(sample, sample_saved_path, fps=8)

                num_sampled_frames = 4
                sampled_frames_list = []
                with video_reader(sample_saved_path) as vr:
                    sampled_frame_idx_list = np.linspace(0, len(vr), num_sampled_frames, endpoint=False, dtype=int)
                    sampled_frame_list = vr.get_batch(sampled_frame_idx_list).asnumpy()
                    sampled_frames = torch.stack([to_tensor(frame) for frame in sampled_frame_list], dim=0)
                    sampled_frames_list.append(sampled_frames)
                
                sampled_frames = torch.stack(sampled_frames_list)
                sampled_frames = rearrange(sampled_frames, "b t c h w -> b c t h w")
                loss, reward = loss_fn(sampled_frames, [validation_prompt])
                validation_loss, validation_reward = validation_loss + loss, validation_reward + reward
    
    validation_loss = validation_loss / len(validation_prompts_idx)
    validation_reward = validation_reward / len(validation_prompts_idx)

    del pipeline
    del transformer3d_val
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return validation_loss, validation_reward


def load_prompts(prompt_path, prompt_column="prompt", start_idx=None, end_idx=None):
    prompt_list = []
    if prompt_path.endswith(".txt"):
        with open(prompt_path, "r") as f:
            for line in f:
                prompt_list.append(line.strip())
    elif prompt_path.endswith(".jsonl"):
        with open(prompt_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                prompt_list.append(item[prompt_column])
    else:
        raise ValueError("The prompt_path must end with .txt or .jsonl.")
    prompt_list = prompt_list[start_idx:end_idx]

    return prompt_list


# Modified from cogvideox.pipeline.pipeline_cogvideox_inpaint.CogVideoX_Fun_Pipeline_Inpaint._get_t5_prompt_embeds
def get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


# Modified from cogvideox.pipeline.pipeline_cogvideox_inpaint.CogVideoX_Fun_Pipeline_Inpaint.encode_prompt
def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, negative_prompt_embeds


# Modified from cogvideox.pipeline.pipeline_cogvideox_inpaint.CogVideoX_Fun_Pipeline_Inpaint.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    import inspect

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


# Modified from cogvideox.pipeline.pipeline_cogvideox_inpaint.CogVideoX_Fun_Pipeline_Inpaint._prepare_rotary_positional_embeddings
def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: torch.device = "cpu"
):
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = 720 // (vae_scale_factor_spatial * patch_size)
    base_size_height = 480 // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid(
        (grid_height, grid_width), base_size_width, base_size_height
    )
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        use_real=True,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--validation_prompt_path",
        type=str,
        default=None,
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=1,
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_sample_height",
        type=int,
        default=512,
        help="The height of sampling videos in validation.",
    )
    parser.add_argument(
        "--validation_sample_width",
        type=int,
        default=512,
        help="The width of sampling videos in validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing (for DiT) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--vae_gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing (for VAE) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default="normal",
        help="The path to the training prompt file.",
    )
    parser.add_argument(
        '--train_sample_height', 
        type=int,
        default=384,
        help='The height of sampling videos in training'
    )
    parser.add_argument(
        '--train_sample_width', 
        type=int,
        default=672,
        help='The width of sampling videos in training'
    )
    parser.add_argument(
        "--video_length", 
        type=int,
        default=49,
        help="The number of frames to generate in training and validation."
    )
    parser.add_argument(
        '--eta', 
        type=float,
        default=0.0,
        help='eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, '
        'with 0.0 being fully deterministic and 1.0 being equivalent to the DDPM sampler.'
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float,
        default=6.0,
        help="The classifier-free diffusion guidance."
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int,
        default=50,
        help="The number of denoising steps in training and validation."
    )
    parser.add_argument(
        "--num_decoded_latents",
        type=int,
        default=3,
        help="The number of latents to be decoded."
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=None,
        help="The number of sampled frames for the reward function."
    )
    parser.add_argument(
        "--reward_fn", 
        type=str,
        default="HPSReward",
        help='The reward function.'
    )
    parser.add_argument(
        "--reward_fn_kwargs",
        type=str,
        default=None,
        help='The keyword arguments of the reward function.'
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        default=False,
        help="Whether to use the backprop training mode.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Sanity check for validation
    do_validation = (args.validation_prompt_path is not None or args.validation_prompts is not None)
    if do_validation:
        if not (os.path.exists(args.validation_prompt_path) or args.validation_prompt_path.endswith(".txt")):
            raise ValueError("The `--validation_prompt_path` must be a txt file containing prompts.")
        if args.validation_batch_size < accelerator.num_processes or args.validation_batch_size % accelerator.num_processes != 0:
            raise ValueError("The `--validation_batch_size` must be divisible by the number of processes.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    # Use DDIM instead of DDPM to sample training videos.
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)

    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,
            torch_dtype=weight_dtype
        )

        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    transformer3d = CogVideoXTransformer3DModel.from_pretrained_2d(
        args.pretrained_model_name_or_path, subfolder="transformer"
    )

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    # Lora will work with this...
    network = create_network(
        1.0,
        args.rank,
        args.network_alpha,
        text_encoder,
        transformer3d,
        neuron_dropout=None,
        add_lora_in_attn_temporal=True,
    )
    network.apply_to(text_encoder, transformer3d, args.train_text_encoder, True)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_scale_factor_temporal = vae.config.temporal_compression_ratio
    num_channels_latent = vae.config.latent_channels

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                save_model(safetensor_save_path, accelerator.unwrap_model(models[-1]))
                if not args.use_deepspeed:
                    for _ in range(len(weights)):
                        weights.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
    
    if args.vae_gradient_checkpointing:
        # Since 3D casual VAE need a cache to decode all latents autoregressively, .Thus, gradient checkpointing can only be 
        # enabled when decoding the first batch (i.e. the first three) of latents, in which case the cache is not being used.
        if args.num_decoded_latents > 3:
            raise ValueError("The vae_gradient_checkpointing is not supported for num_decoded_latents > 3.")
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    logging.info("Add network parameters")
    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    # loss function
    reward_fn_kwargs = {}
    if args.reward_fn_kwargs is not None:
        reward_fn_kwargs = json.loads(args.reward_fn_kwargs)
    if accelerator.is_main_process:
        # Check if the model is downloaded in the main process.
        loss_fn = getattr(reward_fn, args.reward_fn)(device="cpu", dtype=weight_dtype, **reward_fn_kwargs)
    accelerator.wait_for_everyone()
    loss_fn = getattr(reward_fn, args.reward_fn)(device=accelerator.device, dtype=weight_dtype, **reward_fn_kwargs)
    
    # Get RL training prompts
    prompt_list = load_prompts(args.prompt_path)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(prompt_list) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    network, optimizer, lr_scheduler = accelerator.prepare(
        network, optimizer, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(prompt_list) / args.train_batch_size / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(prompt_list)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(os.path.join(os.path.join(args.output_dir, path), "lora_diffusion_pytorch_model.safetensors"))
            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataloader_iterations = 100
        train_loss = 0.0
        train_reward = 0.0
        # In the following training loop, randomly select training prompts and use the 
        # `CogVideoX_Fun_Pipeline_Inpaint` to sample videos, calculate rewards, and update the network.
        for _ in range(train_dataloader_iterations):
            # train_prompt = random.sample(prompt_list, args.train_batch_size)
            train_prompt = random.choices(prompt_list, k=args.train_batch_size)
            logger.info(f"train_prompt: {train_prompt}")
            
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = args.guidance_scale > 1.0
            
            # Encode input prompt
            prompt_embeds, negative_prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                train_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt="",
                dtype=weight_dtype,
                device=accelerator.device,
            )
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

            # Prepare timesteps
            timesteps = noise_scheduler.timesteps

            # Prepare latents
            latent_shape = [
                len(train_prompt),
                (args.video_length - 1) // vae_scale_factor_temporal + 1,
                num_channels_latent,
                args.train_sample_height // vae_scale_factor_spatial,
                args.train_sample_width // vae_scale_factor_spatial,
            ]

            with accelerator.accumulate(transformer3d):
                with accelerator.autocast():
                    latents = torch.randn(*latent_shape, device=accelerator.device, dtype=weight_dtype)
                    latents = latents * noise_scheduler.init_noise_sigma

                    mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
                    masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )
                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)

                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    # Prepare extra step kwargs.
                    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.eta)

                    # Create rotary embeds if required
                    image_rotary_emb = (
                        prepare_rotary_positional_embeddings(
                            args.train_sample_height,
                            args.train_sample_width,
                            latents.size(1),
                            vae_scale_factor_spatial,
                            unwrap_model(transformer3d).config.patch_size,
                            device=accelerator.device
                        )
                        if unwrap_model(transformer3d).config.use_rotary_positional_embeddings
                        else None
                    )

                    # Denoising loop
                    for i, t in enumerate(tqdm(timesteps)):
                        # DRaFT-K: the reward gradient is back propagated only for the last K steps.
                        if args.backprop:
                            # backprop_cutoff_idx = random.randint(0, args.num_sampling_steps - 1)  # random
                            # Simply setting K=1 results in the best reward vs. compute tradeoff.
                            backprop_cutoff_idx = args.num_inference_steps - 1  # last
                            if i >= backprop_cutoff_idx:
                                for param in network.parameters():
                                    param.requires_grad = True
                            else:
                                for param in network.parameters():
                                    param.requires_grad = False
                        
                         # for DPM-solver++
                        old_pred_original_sample = None
                        
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        timestep = t.expand(latent_model_input.shape[0])

                        # predict noise model_output
                        noise_pred = transformer3d(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            return_dict=False,
                            inpaint_latents=inpaint_latents
                        )[0]
                        noise_pred = noise_pred.float()

                        # perform guidance
                        guidance_scale = args.guidance_scale
                        # if args.use_dynamic_cfg:
                        #     guidance_scale = 1 + guidance_scale * (
                        #         (1 - math.cos(math.pi * ((args.num_inference_steps - t.item()) / args.num_inference_steps) ** 5.0)) / 2
                        #     )
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        if not isinstance(noise_scheduler, CogVideoXDPMScheduler):
                            latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                        else:
                            latents, old_pred_original_sample = noise_scheduler.step(
                                noise_pred,
                                old_pred_original_sample,
                                t,
                                timesteps[i - 1] if i > 0 else None,
                                latents,
                                **extra_step_kwargs,
                                return_dict=False,
                            )
                        latents = latents.to(prompt_embeds.dtype)
                    
                    # decode latents (tensor)
                    latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                    # Since the casual VAE decoding consumes a large amount of VRAM, and we need to keep the decoding 
                    # operation within the computational graph. Thus, we only decode the first args.num_decoded_latents 
                    # to calculate the reward.
                    sampled_frame_indices = list(range(args.num_decoded_latents))
                    sampled_latents = latents[:, :, sampled_frame_indices, :, :]
                    sampled_latents = 1 / vae.config.scaling_factor * sampled_latents
                    sampled_frames = vae.decode(sampled_latents).sample
                    sampled_frames = (sampled_frames / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]

                    if global_step % args.checkpointing_steps == 0:
                        saved_file = f"sample-{global_step}-{accelerator.process_index}.mp4"
                        save_videos_grid(
                            sampled_frames.to(torch.float32).detach().cpu(),
                            os.path.join(args.output_dir, "train_sample", saved_file),
                            fps=8
                        )
                    
                    if args.num_sampled_frames is not None:
                        num_frames = sampled_frames.size(2) - 1
                        sampled_frames_indices = torch.linspace(0, num_frames, steps=args.num_sampled_frames).long()
                        sampled_frames = sampled_frames[:, :, sampled_frames_indices, :, :]
                    # compute loss and reward
                    loss, reward = loss_fn(sampled_frames, train_prompt)

                    # Gather the losses and rewards across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    avg_reward = accelerator.gather(reward.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    train_reward += avg_reward.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        total_norm = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                        # If `args.use_deepspeed` is enabled, `total_norm` cannot be logged by accelerator.
                        if not args.use_deepspeed:
                            accelerator.log({"total_norm": total_norm}, step=global_step)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        accelerator.log({"train_loss": train_loss, "train_reward": train_reward}, step=global_step)
                        train_loss = 0.0
                        train_reward = 0.0

                        if global_step % args.checkpointing_steps == 0:
                            # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                            if args.use_deepspeed or accelerator.is_main_process:
                                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                                if args.checkpoints_total_limit is not None:
                                    checkpoints = os.listdir(args.output_dir)
                                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                    if len(checkpoints) >= args.checkpoints_total_limit:
                                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                        removing_checkpoints = checkpoints[0:num_to_remove]

                                        logger.info(
                                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                        )
                                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                        for removing_checkpoint in removing_checkpoints:
                                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                            shutil.rmtree(removing_checkpoint)
                                
                                if not args.save_state:
                                    safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                                    save_model(safetensor_save_path, accelerator.unwrap_model(network))
                                    logger.info(f"Saved safetensor to {safetensor_save_path}")
                                else:
                                    accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                                    accelerator.save_state(accelerator_save_path)
                                    logger.info(f"Saved state to {accelerator_save_path}")

                        # Validation (distributed)
                        if do_validation and (global_step % args.validation_steps) == 0:
                            if args.validation_prompts is None and args.validation_prompt_path.endswith(".txt"):
                                validation_prompts = []
                                with open(args.validation_prompt_path, "r") as f:
                                    for line in f:
                                        validation_prompts.append(line.strip())
                                # Do not select randomly to ensure that `args.validation_prompts` is the same for each process.
                                args.validation_prompts = validation_prompts[:args.validation_batch_size]
                            
                            validation_prompts_idx = [(i, p) for i, p in enumerate(args.validation_prompts)]
                            
                            accelerator.wait_for_everyone()
                            with accelerator.split_between_processes(validation_prompts_idx) as splitted_prompts_idx:
                                validation_loss, validation_reward = log_validation(
                                    vae,
                                    text_encoder,
                                    tokenizer,
                                    transformer3d,
                                    network,
                                    loss_fn,
                                    args,
                                    accelerator,
                                    weight_dtype,
                                    global_step,
                                    splitted_prompts_idx
                                )
                                avg_validation_loss = accelerator.gather(validation_loss).mean()
                                avg_validation_reward = accelerator.gather(validation_reward).mean()
                                if accelerator.is_main_process:
                                    accelerator.log({"validation_loss": avg_validation_loss, "validation_reward": avg_validation_reward}, step=global_step)
                            accelerator.wait_for_everyone()
                    
                    logs = {"step_loss": loss.detach().item(), "step_reward": reward.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    
                    if global_step >= args.max_train_steps:
                        break


if __name__ == "__main__":
    main()
