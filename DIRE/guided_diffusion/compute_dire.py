"""
Modified from guided-diffusion/scripts/image_sample.py
"""

import argparse
import os
import torch
import time

import cv2
from mpi4py import MPI

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_for_reverse
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main(dataset_name, split, num_samples, dataset_type):
    args = create_argparser(dataset_name, split, num_samples, dataset_type).parse_args()

    dist_util.setup_dist(os.environ["CUDA_VISIBLE_DEVICES"])
    logger.configure(dir=args.recons_dir)

    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)
    logger.log(str(args))

    logger.log("Creating model and diffusion")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False)
    model.to(dist_util.dev())
    logger.log("have created model and diffusion")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    )
    logger.log("have created data loader")

    logger.log("computing recons & DIRE ...")
    have_finished_images = 0
    while have_finished_images < args.num_samples:
        if (have_finished_images + MPI.COMM_WORLD.size * args.batch_size) > args.num_samples and (
            args.num_samples - have_finished_images
        ) % MPI.COMM_WORLD.size == 0:
            batch_size = (args.num_samples - have_finished_images) // MPI.COMM_WORLD.size
        else:
            batch_size = args.batch_size
        all_images = []
        all_labels = []
        imgs, out_dicts, paths = next(data)
        imgs = imgs[:batch_size]
        paths = paths[:batch_size]

        imgs = imgs.to(dist_util.dev())
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes
        reverse_fn = diffusion.ddim_reverse_sample_loop
        imgs = reshape_image(imgs, args.image_size)

        latent = reverse_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        recons = sample_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=latent,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        dire = th.abs(imgs - recons)
        recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons = recons.permute(0, 2, 3, 1)
        recons = recons.contiguous()

        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.contiguous()

        dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
        dire = dire.permute(0, 2, 3, 1)
        dire = dire.contiguous()

        gathered_samples = [th.zeros_like(recons) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, recons)  # gather not supported with NCCL

        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        have_finished_images += len(all_images) * batch_size
        # print(th.mean(res.float()))
        recons = recons.cpu().numpy()
        for i in range(len(recons)):
            if args.has_subfolder:
                recons_save_dir = os.path.join(args.recons_dir, paths[i].split("/")[-2])
                dire_save_dir = os.path.join(args.dire_dir, paths[i].split("/")[-2])
            else:
                recons_save_dir = args.recons_dir
                dire_save_dir = args.dire_dir
            fn_save = os.path.basename(paths[i])
            os.makedirs(recons_save_dir, exist_ok=True)
            os.makedirs(dire_save_dir, exist_ok=True)
            cv2.imwrite(
                f"{dire_save_dir}/{fn_save}", cv2.cvtColor(dire[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(f"{recons_save_dir}/{fn_save}", cv2.cvtColor(recons[i].astype(np.uint8), cv2.COLOR_RGB2BGR))
        logger.log(f"have finished {have_finished_images} samples")

    dist.barrier()
    logger.log("finish computing recons & DIRE!")


dataset_root = "/data/datasets/GenImage"

DATASET_ROUTES = {
    "GLIDE": f"{dataset_root}/glide/imagenet_glide",
    "VQDM": f"{dataset_root}/VQDM/imagenet_ai_0419_vqdm",
    "ADM": f"{dataset_root}/ADM/imagenet_ai_0508_adm",
    "BigGAN": f"{dataset_root}/BigGAN/imagenet_ai_0419_biggan",
    "Wukong": f"{dataset_root}/wukong/imagenet_ai_0424_wukong",
    "Midjourney": f"{dataset_root}/Midjourney/imagenet_midjourney",
    "SDV14": f"{dataset_root}/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4",
    "SDV15": f"{dataset_root}/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5"
}


def create_argparser(dataset_name, split, num_samples, dataset_type):
    defaults = model_and_diffusion_defaults()
    if dataset_type == "base_train":
        custom = dict(
            model_path="models/256x256_diffusion_uncond.pt",
            image_size=256,
            batch_size=32,
            # model_path="models/512x512_diffusion.pt",
            # image_size=512,
            # batch_size=16,

            timestep_respacing="ddim24",
            # num_samples=6000,
            num_samples=num_samples,
            images_dir=f"{DATASET_ROUTES[dataset_name]}/val/stepwise-noised/0/train-small/{split}",
            recons_dir=f"/data/datasets/DiffusionForensics/recons/train/GenImage-{dataset_name}/{split}",
            dire_dir=f"/data/datasets/DiffusionForensics/dire/train/GenImage-{dataset_name}/{split}",
            real_step=0,
            clip_denoised=True,
            continue_reverse=False,
            has_subfolder=False,
            use_ddim=True,
            class_cond=False,

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
    elif dataset_type == "base_val":
        custom = dict(
            model_path="models/256x256_diffusion_uncond.pt",
            image_size=256,
            batch_size=32,
            # model_path="models/512x512_diffusion.pt",
            # image_size=512,
            # batch_size=16,

            timestep_respacing="ddim24",
            # num_samples=6000,
            num_samples=num_samples,
            images_dir=f"{DATASET_ROUTES[dataset_name]}/val/stepwise-noised/0/val-small/{split}",
            recons_dir=f"/data/datasets/DiffusionForensics/recons/val/GenImage-{dataset_name}/{split}",
            dire_dir=f"/data/datasets/DiffusionForensics/dire/val/GenImage-{dataset_name}/{split}",
            real_step=0,
            clip_denoised=True,
            continue_reverse=False,
            has_subfolder=False,
            use_ddim=True,
            class_cond=False,

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
    elif dataset_type == "hard_val":
        custom = dict(
            model_path="models/256x256_diffusion_uncond.pt",
            image_size=256,
            batch_size=32,
            # model_path="models/512x512_diffusion.pt",
            # image_size=512,
            # batch_size=16,

            timestep_respacing="ddim24",
            # num_samples=6000,
            num_samples=num_samples,
            images_dir=f"{DATASET_ROUTES[dataset_name]}/train/hard-samples/stepwise-noised/0/val-small/{split}",
            recons_dir=f"/data/datasets/DiffusionForensics/recons/val/GenImage-{dataset_name}-hard/{split}",
            dire_dir=f"/data/datasets/DiffusionForensics/dire/val/GenImage-{dataset_name}-hard/{split}",
            real_step=0,
            clip_denoised=True,
            continue_reverse=False,
            has_subfolder=False,
            use_ddim=True,
            class_cond=False,

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
    else:
        print("Error dataset type!")
    defaults.update(custom)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    start = time.time()
    print("Start Time: " + str(time.localtime()))

    main("GLIDE", "ai", 5400, "base_train")
    main("GLIDE", "nature", 5400, "base_train")
    main("GLIDE", "ai", 600, "base_val")
    main("GLIDE", "nature", 600, "base_val")

    print("End Time: " + str(time.localtime()))
    print("Elapsed Secs: " + str(time.time() - start))

