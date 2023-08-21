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
import tqdm

import sys
from utils.nerfacc_radiance_fields.mlp import VanillaNeRFRadianceFieldG
from utils.nerfacc_radiance_fields.utils import render_image, set_random_seed

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from einops import rearrange

from nerfacc import ContractionType, OccupancyGrid

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
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
        "--smallAABB",
        type=int,
        default=0,
        help="whether to use a small bounding box",
    )

    parser.add_argument(
        "--dim_a",
        type=int,
        default=48,
        help="dimension of appearance code",
    )
    parser.add_argument(
        "--dim_g",
        type=int,
        default=16,
        help="dimension of geometry code",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10,
        help="total number of tasks",
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

    # setup the radiance field we want to train.
    max_steps = args.max_steps

    grad_scaler = torch.cuda.amp.GradScaler(1)
    # radiance_field = VanillaNeRFRadianceField(net_width = args.dim).to(device)
    radiance_field = VanillaNeRFRadianceFieldG(net_width = args.dim, vocab_size = args.vocab_size, dim_a = args.dim_a, dim_g = args.dim_g).to(device)

    id_rep = None
    for task_curr in range(args.task_number):
        args.task_curr = task_curr

        print("training on task {}".format(args.task_curr))

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

        from utils.nerfacc_radiance_fields.datasets.lb.colmap import SubjectLoader_lb as SubjectLoader

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
        train_dataset.task_ids = train_dataset.task_ids.to(device)

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
        test_dataset.task_ids = test_dataset.task_ids.to(device)

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
            # if args.smallAABB:
            #     args.aabb = [-1.5,-1.5,-1.5,1.5,1.5,1.5]

            scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
            near_plane = None
            far_plane = None
            # # # auto aabb
            # camera_locs = torch.cat(
            #     [train_dataset.camtoworlds, test_dataset.camtoworlds]
            # )[:, :3, -1]
            # aabb_diff = (torch.cat(
            #     [camera_locs.max(dim=0).values - camera_locs.min(dim=0).values]
            # )).tolist()
            # print("Using auto aabb_diff", aabb_diff, (torch.cat(
            #     [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
            # )).tolist())

            # render_step_size = (
            #     max(aabb_diff)
            #     * math.sqrt(3)
            #     / render_n_samples
            # )
            # render_step_size = 1.5 * math.sqrt(3) / render_n_samples
            render_step_size = (
                (scene_aabb[3:] - scene_aabb[:3]).max()
                * math.sqrt(3)
                / render_n_samples
            ).item()
        
        print("rendering step size = {}".format(render_step_size))
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
                task_id = data['task_id']

                occupancy_grid.every_n_step(
                    step=step,
                    occ_eval_fn=lambda x: radiance_field.query_opacity(
                        x, torch.randint(0, args.vocab_size, (x.shape[0], ), device = device), render_step_size
                    ),
                )

                # render
                rgb, acc, depth, n_rendering_samples = render_image(
                    radiance_field,
                    occupancy_grid,
                    rays,
                    task_id,
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
                loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

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
                    )
                
                if step == max_steps:
                    # print("training stops, step = {}".format(step))
                    break

                step += 1




# print("step == max_steps = {}/{}/{},   step > 0 = {}, args.task_curr == (args.task_number - 1) = {}".format(step == max_steps, step, max_steps, step > 0, args.task_curr == (args.task_number - 1)))
# if step == max_steps and step > 0 and args.task_curr == (args.task_number - 1):
# evaluation
result_dir = f'results/WAT/NT_ER/{args.scene}_{args.rep_size}'
os.makedirs(result_dir, exist_ok=True)
radiance_field.eval()

# save the trained model
out_dict = {'model': radiance_field, 'occupancy_grid': occupancy_grid}
torch.save(out_dict, result_dir+'/model.torchSave')

psnrs, ssims, lpips = [], [], []
# psnrs_ngp = []
with torch.no_grad():
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        task_id = data['task_id'].flatten()

        # print("data = {}, render_bkgd = {}, rays = {}, pixels = {}, task_id = {}".format(data, render_bkgd, rays, pixels, task_id))

        # rendering
        rgb, acc, depth, _ = render_image(
            radiance_field,
            occupancy_grid,
            rays,
            task_id,
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
        # mse = F.mse_loss(rgb, pixels)
        # psnr = -10.0 * torch.log(mse) / np.log(10.0)
        # psnrs.append(psnr.item())
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

