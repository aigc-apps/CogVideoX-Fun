"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
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
import copy
import gc
import logging
import math
import os
import pickle
import shutil
import sys

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, AutoencoderKLCogVideoX, DDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (BertModel, BertTokenizer, CLIPImageProcessor,
                          CLIPVisionModelWithProjection, MT5Tokenizer,
                          T5EncoderModel, T5Tokenizer)
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from cogvideox.data.bucket_sampler import (ASPECT_RATIO_512,
                                           ASPECT_RATIO_RANDOM_CROP_512,
                                           ASPECT_RATIO_RANDOM_CROP_PROB,
                                           AspectRatioBatchImageSampler,
                                           AspectRatioBatchImageVideoSampler,
                                           AspectRatioBatchSampler,
                                           RandomSampler, get_closest_ratio)
from cogvideox.pipeline.pipeline_cogvideox import CogVideoX_FUN_Pipeline
from cogvideox.pipeline.pipeline_cogvideox_inpaint import CogVideoX_FUN_Pipeline_Inpaint
from cogvideox.data.dataset_image import CC15M
from cogvideox.data.dataset_image_video import (ImageVideoDataset,
                                                ImageVideoSampler,
                                                get_random_mask)
from cogvideox.data.dataset_video import VideoDataset, WebVid10M
from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.utils.lora_utils import create_network, merge_lora, unmerge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid

if is_wandb_available():
    import wandb


def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list
            
    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p = number_list_prob)
    else:
        return rng.choice(number_list, p = number_list_prob)

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, tokenizer, transformer3d, network, args, accelerator, weight_dtype, global_step):
    try:
        logger.info("Running validation... ")

        transformer3d_val = CogVideoXTransformer3DModel.from_pretrained_2d(
            args.pretrained_model_name_or_path, subfolder="transformer",
        ).to(weight_dtype)
        transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())
        
        if args.train_mode != "normal":
            pipeline = CogVideoX_FUN_Pipeline_Inpaint.from_pretrained(
                args.pretrained_model_name_or_path, 
                vae=accelerator.unwrap_model(vae).to(weight_dtype), 
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_val,
                torch_dtype=weight_dtype,
            )
        else:
            pipeline = CogVideoX_FUN_Pipeline.from_pretrained(
                args.pretrained_model_name_or_path, 
                vae=accelerator.unwrap_model(vae).to(weight_dtype), 
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_val,
                torch_dtype=weight_dtype
            )

        pipeline = pipeline.to(accelerator.device)
        pipeline = merge_lora(
            pipeline, None, 1, accelerator.device, state_dict=accelerator.unwrap_model(network).state_dict(), transformer_only=True
        )

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        for i in range(len(args.validation_prompts)):
            with torch.no_grad():
                if args.train_mode != "normal":
                    with torch.autocast("cuda", dtype=weight_dtype):
                        video_length = int(args.video_sample_n_frames // vae.mini_batch_encoder * vae.mini_batch_encoder) if args.video_sample_n_frames != 1 else 1
                        input_video, input_video_mask, clip_image = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[args.video_sample_size, args.video_sample_size])
                        sample = pipeline(
                            args.validation_prompts[i], 
                            video_length = args.video_sample_n_frames,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            guidance_scale = 7,
                            generator   = generator, 

                            video        = input_video,
                            mask_video   = input_video_mask,
                            clip_image   = clip_image, 
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.gif"))

                        video_length = 1
                        input_video, input_video_mask, clip_image = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[args.video_sample_size, args.video_sample_size])
                        sample = pipeline(
                            args.validation_prompts[i], 
                            video_length = 1,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            generator   = generator, 

                            video        = input_video,
                            mask_video   = input_video_mask,
                            clip_image   = clip_image, 
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-image-{i}.gif"))
                else:
                    with torch.autocast("cuda", dtype=weight_dtype):
                        sample = pipeline(
                            args.validation_prompts[i], 
                            video_length = args.video_sample_n_frames,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            generator   = generator
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.gif"))

                        sample = pipeline(
                            args.validation_prompts[i], 
                            video_length = 1,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            generator   = generator
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-image-{i}.gif"))

        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        return None

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
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
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
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
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
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
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--noise_share_in_frames", action="store_true", help="Whether enable noise share in frames."
    )
    parser.add_argument(
        "--noise_share_in_frames_ratio", type=float, default=0.5, help="Noise share ratio.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--image_repeat_in_forward",
        type=int,
        default=0,
        help="Num of repeat image in forward.",
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
        '--tokenizer_max_length', 
        type=int,
        default=226,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The format of training data. Support `"normal"`'
            ' (default), `"inpaint"`.'
        ),
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
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

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
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

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
    network.apply_to(text_encoder, transformer3d, args.train_text_encoder and not args.training_with_video_token_length, True)

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

                with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                    pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

        def load_model_hook(models, input_dir):
            pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    loaded_number, _ = pickle.load(file)
                    batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

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
            # weight_decay=args.adam_weight_decay,
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

    # Get the training dataset
    sample_n_frames_bucket_interval = 4

    train_dataset = ImageVideoDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, enable_inpaint=True if args.train_mode != "normal" else False,
    )
    
    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )
        if args.keep_all_node_same_token_length:
            if args.token_sample_size > 256:
                numbers_list = list(range(256, args.token_sample_size + 1, 128))

                if numbers_list[-1] != args.token_sample_size:
                    numbers_list.append(args.token_sample_size)
            else:
                numbers_list = [256]
            numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]
        else:
            numbers_list = None

        def get_length_to_frame_num(token_length):
            if args.image_sample_size > args.video_sample_size:
                sample_sizes = list(range(256, args.image_sample_size + 1, 128))

                if sample_sizes[-1] != args.image_sample_size:
                    sample_sizes.append(args.image_sample_size)
            else:
                sample_sizes = [256]
            
            length_to_frame_num = {
                sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
            }

            return length_to_frame_num

        def collate_fn(examples):
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(
                target_token_length, 
            )

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []

            # Get ratio
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size], rng=rng)

                aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))
                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        if rng is None:
                            local_video_sample_size = np.random.choice(choice_list)
                        else:
                            local_video_sample_size = rng.choice(choice_list)
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(
                                args.video_sample_size, rng=rng)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
            closest_size = [int(x / 16) * 16 for x in closest_size]
            if args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / 16) * 16 for x in random_sample_size]

            for example in examples:
                if args.random_ratio_crop:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                else:
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                new_examples["pixel_values"].append(transform(pixel_values))
                new_examples["text"].append(example["text"])
                batch_video_length = int(
                    min(
                        batch_video_length,
                        (len(pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1, 
                    )
                )

                if batch_video_length == 0:
                    batch_video_length = 1

                if args.train_mode != "normal":
                    mask = get_random_mask(new_examples["pixel_values"][-1].size())
                    mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                    new_examples["mask_pixel_values"].append(mask_pixel_values)
                    new_examples["mask"].append(mask)

            new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example[:batch_video_length] for example in new_examples["mask"]])

            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids,
                    return_dict=False
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
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
    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
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
    logger.info(f"  Num examples = {len(train_dataset)}")
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
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            pkl_path = os.path.join(os.path.join(args.output_dir, path), "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            from safetensors.torch import load_file, safe_open
            state_dict = load_file(os.path.join(os.path.join(args.output_dir, path), "lora_diffusion_pytorch_model.safetensors"))
            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
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

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                if args.train_mode != "normal":
                    mask_pixel_values, texts = batch['mask_pixel_values'].cpu(), batch['text']
                    mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(mask_pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.gif", rescale=True)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                if args.training_with_video_token_length:
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                        else:
                            batch['text'] = batch['text'] * 4
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                        else:
                            batch['text'] = batch['text'] * 2
                
                if args.train_mode != "normal":
                    mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                    mask = batch["mask"].to(weight_dtype)
                    if args.training_with_video_token_length:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
                            mask = torch.tile(mask, (4, 1, 1, 1, 1))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
                            mask = torch.tile(mask, (2, 1, 1, 1, 1))

                def create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        last_element = 0.90
                        remaining_sum = 1.0 - last_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [other_elements_value] * (length - 1) + [last_element]
                        return special_list
                    
                if args.keep_all_node_same_token_length:
                    actual_token_length = index_rng.choice(numbers_list)

                    actual_video_length = (min(
                            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                    actual_video_length = int(max(actual_video_length, 1))
                else:
                    actual_video_length = None

                if args.random_frame_crop:
                    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                    select_frames_prob = np.array(create_special_list(len(select_frames)))
                    
                    if rng is None:
                        temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                    else:
                        temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                    if args.keep_all_node_same_token_length:
                        temp_n_frames = min(actual_video_length, temp_n_frames)

                    pixel_values = pixel_values[:, :temp_n_frames, :, :]

                    if args.train_mode != "normal":
                        mask_pixel_values = mask_pixel_values[:, :temp_n_frames, :, :]
                        mask = mask[:, :temp_n_frames, :, :]

                if args.train_mode != "normal":
                    t2v_flag = [(_mask == 1).all() for _mask in mask]
                    new_t2v_flag = []
                    for _mask in t2v_flag:
                        if _mask and np.random.rand() < 0.90:
                            new_t2v_flag.append(0)
                        else:
                            new_t2v_flag.append(1)
                    t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _slice_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                            vae._clear_fake_context_parallel_cache()
                        return torch.cat(new_pixel_values, dim = 0)
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _slice_vae(pixel_values)
                    else:
                        latents = _slice_vae(pixel_values)
                    latents = latents * vae.config.scaling_factor

                    if args.train_mode != "normal":
                        mask = rearrange(mask, "b f c h w -> b c f h w")
                        mask = 1 - mask
                        mask = resize_mask(mask, latents)

                        mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_mask_pixel_values = []
                        for i in range(0, mask_pixel_values.shape[0], bs):
                            mask_pixel_values_bs = mask_pixel_values[i : i + bs]
                            mask_pixel_values_bs = vae.encode(mask_pixel_values_bs)[0]
                            mask_pixel_values_bs = mask_pixel_values_bs.sample()
                            new_mask_pixel_values.append(mask_pixel_values_bs)
                            vae._clear_fake_context_parallel_cache()
                        mask_latents = torch.cat(new_mask_pixel_values, dim = 0)

                        if vae_stream_2 is not None:
                            torch.cuda.current_stream().wait_stream(vae_stream_2) 

                        inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                        inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents
                        inpaint_latents = inpaint_latents * vae.config.scaling_factor
                        inpaint_latents = rearrange(inpaint_latents, "b c f h w -> b f c h w")

                    latents = rearrange(latents, "b c f h w -> b f c h w")
                        
                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'], 
                            max_length=args.tokenizer_max_length, 
                            padding="max_length", 
                            add_special_tokens=True, 
                            truncation=True, 
                            return_tensors="pt"
                        )
                        prompt_embeds = text_encoder(
                            prompt_ids.input_ids.to(latents.device),
                            return_dict=False
                        )[0]

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz = latents.shape[0]
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                # Sample a random timestep for each image
                # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                timesteps = timesteps.long()

                # Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
                def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
                    tw = tgt_width
                    th = tgt_height
                    h, w = src
                    r = h / w
                    if r > (th / tw):
                        resize_height = th
                        resize_width = int(round(th / h * w))
                    else:
                        resize_width = tw
                        resize_height = int(round(tw / w * h))

                    crop_top = int(round((th - resize_height) / 2.0))
                    crop_left = int(round((tw - resize_width) / 2.0))

                    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

                def _prepare_rotary_positional_embeddings(
                    height: int,
                    width: int,
                    num_frames: int,
                    device: torch.device
                ):
                    vae_scale_factor_spatial = (
                        2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
                    )
                    grid_height = height // (vae_scale_factor_spatial * unwrap_model(transformer3d).config.patch_size)
                    grid_width = width // (vae_scale_factor_spatial * unwrap_model(transformer3d).config.patch_size)
                    base_size_width = 720 // (vae_scale_factor_spatial * unwrap_model(transformer3d).config.patch_size)
                    base_size_height = 480 // (vae_scale_factor_spatial * unwrap_model(transformer3d).config.patch_size)

                    grid_crops_coords = get_resize_crop_region_for_grid(
                        (grid_height, grid_width), base_size_width, base_size_height
                    )
                    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                        embed_dim=transformer3d.config.attention_head_dim,
                        crops_coords=grid_crops_coords,
                        grid_size=(grid_height, grid_width),
                        temporal_size=num_frames,
                        use_real=True,
                    )
                    freqs_cos = freqs_cos.to(device=device)
                    freqs_sin = freqs_sin.to(device=device)
                    return freqs_cos, freqs_sin

                height, width = batch["pixel_values"].size()[-2], batch["pixel_values"].size()[-1]
                # 7. Create rotary embeds if required
                image_rotary_emb = (
                    _prepare_rotary_positional_embeddings(height, width, latents.size(1), latents.device)
                    if unwrap_model(transformer3d).config.use_rotary_positional_embeddings
                    else None
                )
                prompt_embeds = prompt_embeds.to(device=latents.device)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # predict the noise residual
                noise_pred = transformer3d(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents if args.train_mode != "normal" else None,
                )[0]
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                if args.motion_sub_loss and noise_pred.size()[1] > 2:
                    gt_sub_noise = noise_pred[:, 1:, :].float() - noise_pred[:, :-1, :].float()
                    pre_sub_noise = target[:, 1:, :].float() - target[:, :-1, :].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
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

                if accelerator.is_main_process:
                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            transformer3d,
                            network,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    transformer3d,
                    network,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_model(safetensor_save_path, accelerator.unwrap_model(network))
        if args.save_state:
            accelerator.save_state(accelerator_save_path)
        logger.info(f"Saved state to {accelerator_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()