"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os, sys
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from torch import nn
from torch.autograd import Variable
import torch.utils.data

import sys
# sys.path.append('/mnt/beegfs/mixed-tier/work/zcai/WorkSpace/NeRF/nerfacc/examples')
import tqdm
from utils.nerfacc_radiance_fields.mlp import VanillaNeRFRadianceField
from utils.nerfacc_radiance_fields.utils import render_image_ori, set_random_seed

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from einops import rearrange

from nerfacc import ContractionType, OccupancyGrid

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


if __name__ == "__main__":

    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="Barn",
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-4.0,-4.0,-4.0,4.0,4.0,4.0",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    # CL params
    parser.add_argument('--task_number', type=int, default=10,
                        help='task_number')
    parser.add_argument('--task_curr', type=int, default=9,
                        help='task_number [0, N-1]')
    parser.add_argument('--task_split_method', type=str, default='seq',
                        help='seq or random')
    parser.add_argument('--rep_size', type=int, default=0,
                        help='0 to number of images')      
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed, wrong random seed can lead to nan loss')
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--EWC_weight",
        type=float,
        default=1e5,
    )
    parser.add_argument(
        "--smallAABB",
        type=int,
        default=0,
        help="whether to use a small bounding box",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default='dataset/WAT',
        help="total number of tasks",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)

    if os.path.isfile("/home/zcai/.cache/torch_extensions/py39_cu117/nerfacc_cuda/lock"):
        print("lock file exists in cache")
        os.remove("/home/zcai/.cache/torch_extensions/py39_cu117/nerfacc_cuda/lock")
    else:
        print("lock file not exists")

    render_n_samples = 1024
    psnr_func = PeakSignalNoiseRatio(data_range=1)
    ssim_func = StructuralSimilarityIndexMeasure(data_range=1)
    lpip_func = LearnedPerceptualImagePatchSimilarity('vgg')
    for p in lpip_func.net.parameters():
        p.requires_grad = False
    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e4
        render_step_size = 1e-2
    else:
        contraction_type = ContractionType.AABB
        if args.smallAABB:
            args.aabb = [-1.5,-1.5,-1.5,1.5,1.5,1.5]

        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()
        # render_step_size = 1.5 * math.sqrt(3) / render_n_samples

    print("rendering step size = {}".format(render_step_size))

    # setup the radiance field we want to train.
    max_steps = args.max_steps
    grad_scaler = torch.cuda.amp.GradScaler(1)
    radiance_field = VanillaNeRFRadianceField(net_width = args.dim).to(device)

    id_rep = None
    for task_curr in range(args.task_number):
        args.task_curr = task_curr

        print("training on task {}".format(args.task_curr))
        # only for test
        if task_curr > 0:
            print("compute fisher diagonal for EWC")
            # copy old radiance field
            # compute fisher matrix
            radiance_field_old = deepcopy(radiance_field)
            # params_old = {n: p for n, p in radiance_field.named_parameters() if p.requires_grad}
            fisher_diag, param_old = {}, {}
            
            for n, p in deepcopy(radiance_field_old).named_parameters():
                if p.requires_grad:
                    p.data.zero_()
                    fisher_diag[n] = variable(p.data)

            for n, p in deepcopy(radiance_field_old).named_parameters():
                if p.requires_grad:
                    param_old[n] = variable(p.data)

            radiance_field_old.eval()
            steps_fisher = min(1000, 1920*1080*10//train_dataset.num_rays)
            print("steps_fisher = {}".format(steps_fisher))
            for i in range(steps_fisher):
                data=train_dataset[i%len(train_dataset)]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # render
                rgb, acc, depth, n_rendering_samples = render_image_ori(
                    radiance_field_old,
                    occupancy_grid,
                    rays,
                    scene_aabb,
                    # rendering options
                    near_plane=near_plane,
                    far_plane=far_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=args.cone_angle,
                )
                alive_ray_mask = acc.squeeze(-1) > 0

                radiance_field_old.zero_grad()
                # compute the diagonal of fisher information matrix
                # in the case of regression, the gradient of l2 loss and the negative log-likelihood only differs by a constant factor, we need to rescale the EWC loss anyway, so use l2 directly here
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                loss.backward()

                for n, p in radiance_field_old.named_parameters():
                    if p.requires_grad:
                        fisher_diag[n].data += (p.grad.data ** 2 / float(steps_fisher))
                        # print("[is gradient zero?]: p.grad.data = {}".format(p.grad.data))

                # if i > 100:
                #     break

            fisher_diag = {n: p for n, p in fisher_diag.items()}
            # print("finished fisher_diag computation, fisher_diag = {}".format(fisher_diag))
            # exit()

        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 5 // 6,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        )

        # setup the dataset
        train_dataset_kwargs = {}
        test_dataset_kwargs = {}

        from utils.nerfacc_radiance_fields.datasets.lb.nerfpp import SubjectLoader_lb as SubjectLoader
        data_root_fp = args.data_root
        target_sample_batch_size = 1 << 16
        grid_resolution = 128

        train_dataset = SubjectLoader(
            subject_id=args.scene,
            root_fp=data_root_fp,
            split=args.train_split,
            task_number = args.task_number,
            task_curr= args.task_curr,
            task_split_method = args.task_split_method,
            rep_size = args.rep_size,
            num_rays=target_sample_batch_size // render_n_samples,
            id_rep = id_rep,
            **train_dataset_kwargs,
        )

        train_dataset.images = train_dataset.images.to(device)
        train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
        train_dataset.K = train_dataset.K.to(device)
        id_rep = train_dataset.rep_buf.copy()

        test_dataset = SubjectLoader(
            subject_id=args.scene,
            root_fp=data_root_fp,
            split="test",
            num_rays=None,
            **test_dataset_kwargs,
        )
        test_dataset.images = test_dataset.images.to(device)
        test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
        test_dataset.K = test_dataset.K.to(device)

        occupancy_grid = OccupancyGrid(
            roi_aabb=args.aabb,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(device)

        # training
        step = 0
        tic = time.time()
        for epoch in range(10000000):
            if step == max_steps:
                print("training stops, step = {}".format(step))
                sys.stdout.flush()
                break
            for i in range(len(train_dataset)):
                radiance_field.train()
                data = train_dataset[i]

                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # update occupancy grid
                occupancy_grid.every_n_step(
                    step=step,
                    occ_eval_fn=lambda x: radiance_field.query_opacity(
                        x, render_step_size
                    ),
                )

                # render
                rgb, acc, depth, n_rendering_samples = render_image_ori(
                    radiance_field,
                    occupancy_grid,
                    rays,
                    scene_aabb,
                    # rendering options
                    near_plane=near_plane,
                    far_plane=far_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=args.cone_angle,
                )
                if n_rendering_samples == 0:
                    continue

                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays
                    * (target_sample_batch_size / float(n_rendering_samples))
                )
                train_dataset.update_num_rays(num_rays)
                alive_ray_mask = acc.squeeze(-1) > 0

                # compute loss
                EWC_loss = 0
                if task_curr > 0:
                    for n, p in radiance_field.named_parameters():
                        if p.requires_grad:
                            EWC_loss += (fisher_diag[n] * (p-param_old[n]) ** 2).sum()

                loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask]) + args.EWC_weight * EWC_loss
                # print("EWC_loss = {}".format(EWC_loss))
                
                optimizer.zero_grad()
                # do not unscale it because we are using Adam.
                grad_scaler.scale(loss).backward()
                optimizer.step()
                scheduler.step()

                if step % 5000 == 0:
                    elapsed_time = time.time() - tic
                    loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                    print(
                        f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                        f"loss={loss:.5f} | "
                        f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                        f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
                        f"EWC_loss={EWC_loss:.15f} |"
                    )
                
                if step == max_steps:
                    # print("training stops, step = {}".format(step))
                    break

                step += 1


# print("step == max_steps = {}/{}/{},   step > 0 = {}, args.task_curr == (args.task_number - 1) = {}".format(step == max_steps, step, max_steps, step > 0, args.task_curr == (args.task_number - 1)))
# if step == max_steps and step > 0 and args.task_curr == (args.task_number - 1):
# evaluation
# result_dir = f'examples/results/nerfpp/EWC/{args.EWC_weight}/{args.scene}_{args.rep_size}'
result_dir = f'results/nerfpp/EWC/{args.scene}_{args.rep_size}'
os.makedirs(result_dir, exist_ok=True)
radiance_field.eval()

psnrs, ssims, lpips = [], [], []
# psnrs_ngp = []
with torch.no_grad():
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        # rendering
        rgb, acc, depth, _ = render_image_ori(
            radiance_field,
            occupancy_grid,
            rays,
            scene_aabb,
            # rendering options
            near_plane=None,
            far_plane=None,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=args.cone_angle,
            # test options
            test_chunk_size=args.test_chunk_size,
        )

        # compute ngp psnr
        psnrs.append(psnr_func(rgb.cpu(), pixels.cpu()))


        rgb_save = (rgb.cpu().numpy()*255).astype(np.uint8)
        imageio.imsave(os.path.join(result_dir, '{}_{}.png'.format(i, psnrs[-1])), rgb_save)

        rgb_pred = rearrange(rgb, 'h w c -> 1 c h w').cpu()
        rgb_gt = rearrange(pixels, 'h w c -> 1 c h w').cpu()
        ssims.append(ssim_func(rgb_pred, rgb_gt))
        # lpips
        lpips.append(lpip_func(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1)))

psnr_avg = sum(psnrs) / len(psnrs)
ssim_avg = sum(ssims)/len(ssims)
lpip_avg = sum(lpips)/len(lpips)
print(f"evaluation: psnr_avg={psnr_avg}, ssim = {ssim_avg}, lpip = {lpip_avg}")

