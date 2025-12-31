"""
Modified from compute_dire.py -- https://github.com/ZhendongWang6/DIRE
"""

import argparse
import math
import os
import cv2
import time
import numpy as np
from PIL import Image
from mpi4py import MPI
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from DIRE.guided_diffusion.guided_diffusion import dist_util, logger
from DIRE.guided_diffusion.guided_diffusion.image_datasets import load_data_for_reverse
from DIRE.guided_diffusion.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from dataset import DATASET_ROUTES


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def save_image(imgs: torch.Tensor, root_dir: str, idx: int, step: int, split_idx: int, class_split: str) -> None:
    # denormalize
    imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).contiguous()
    rank = MPI.COMM_WORLD.Get_rank()  # current process rank

    for i in range(imgs.shape[0]):
        img_idx = (idx + i) * MPI.COMM_WORLD.Get_size() + rank + 1

        if img_idx <= split_idx:
            save_dir = f"{root_dir}/{step}/train-small/{class_split}"
        else:
            save_dir = f"{root_dir}/{step}/val-small/{class_split}"

        save_path = os.path.join(save_dir, f"output_{img_idx}_step_{step}.png")
        cv2.imwrite(save_path, cv2.cvtColor(imgs[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))


def gen_noised_images(args, model, diffusion, save_dir, save_steps: int, imgs, group_cnt):
    for step in range(0, int(args.timestep_respacing[4:]) + 1):  # (0, 25)
        target_train_dir = f"{save_dir}/{step}/train-small/{args.class_split}"
        target_val_dir = f"{save_dir}/{step}/val-small/{args.class_split}"
        os.makedirs(target_train_dir, exist_ok=True)
        os.makedirs(target_val_dir, exist_ok=True)

    imgs = reshape_image(imgs, args.image_size)
    save_image(imgs, save_dir, group_cnt, 0, args.split_idx, args.class_split)  # original images

    # Stepwise noise inversion
    for t, step in enumerate(diffusion.ddim_reverse_sample_loop_progressive(
            model,
            shape=(args.batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            real_step=args.real_step,
    )):
        if (t + 1) % save_steps == 0:  # Save every k steps
            noised_images = step["sample"]
            save_image(noised_images, save_dir, group_cnt, t + 1, args.split_idx, args.class_split)


def main(dataset_root, class_split):
    args = create_argparser(dataset_root, class_split).parse_args()

    dist_util.setup_dist(os.environ["CUDA_VISIBLE_DEVICES"])
    logger.configure(dir='./logs')
    logger.log(str(args))

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("Created model and diffusion")

    data = load_data_for_reverse(data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size)
    logger.log("Created data loader")

    logger.log("Generating noised images...")
    os.makedirs(args.save_dir, exist_ok=True)

    num_samples = len(os.listdir(args.images_dir))
    batch_size = args.batch_size

    for group_cnt in tqdm(range(0, math.ceil(num_samples / MPI.COMM_WORLD.size), batch_size)):
        if (MPI.COMM_WORLD.size * group_cnt + MPI.COMM_WORLD.size * batch_size) > num_samples:
            batch_size = (num_samples - MPI.COMM_WORLD.size * group_cnt) // MPI.COMM_WORLD.size
            args.batch_size = batch_size
        else:
            batch_size = args.batch_size

        imgs, _, _ = next(data)
        imgs = imgs[:batch_size]
        imgs = imgs.to(dist_util.dev())

        gen_noised_images(args, model, diffusion, args.save_dir, args.save_steps, imgs, group_cnt)

        logger.log(f" -- {group_cnt + batch_size} groups completed")

    # dist.barrier()
    logger.log("Noised images generation complete!")


def create_argparser(dataset_root, class_split):
    defaults = model_and_diffusion_defaults()
    custom = dict(
        # model_path="DIRE/guided_diffusion/models/256x256_diffusion_uncond.pt",
        # image_size=256,
        # batch_size=32,

        model_path="DIRE/guided_diffusion/models/512x512_diffusion.pt",
        image_size=512,
        batch_size=16,

        split_idx=5400,
        class_split=class_split,
        images_dir=f"{dataset_root}/{class_split}",
        save_dir=f"{dataset_root}/stepwise-noised",
        # save_dir = f"results/noised/{time.strftime('%Y%m%d_%H%M', time.localtime())}"

        timestep_respacing="ddim24",
        save_steps=1,
        real_step=0,
        clip_denoised=True,
        continue_reverse=False,
        has_subfolder=False,

        attention_resolutions="32,16,8",
        diffusion_steps=1000,
        dropout=0.1,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True
    )
    defaults.update(custom)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    start = time.time()
    print("Start Time: " + str(time.localtime()))

    # Original samples DDIM inversion
    main(f"{DATASET_ROUTES['SDV1.4']}/val", 'ai')
    main(f"{DATASET_ROUTES['SDV1.4']}/val", 'nature')
    # Hard samples DDIM inversion
    # main(f"{DATASET_ROUTES['SDV1.4']}/train/hard-samples", 'ai')
    # main(f"{DATASET_ROUTES['SDV1.4']}/train/hard-samples", 'nature')

    end = time.time()
    print("End Time: " + str(time.localtime()))
    print("Elapsed Time: " + str(end - start))
