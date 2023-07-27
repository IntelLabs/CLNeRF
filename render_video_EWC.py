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
        default='dataset/WOT',
        help="total number of tasks",
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="starting frame to render",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=10000,
        help="end frame to render",
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
    # max_steps = args.max_steps
    max_steps = 100 # test

    grad_scaler = torch.cuda.amp.GradScaler(1)
    
    # just read out the model 
    model_dir = f'results/WOT/EWC/{args.scene}_{args.rep_size}'
    out_dict_read = torch.load(model_dir+'/model.torchSave')
    radiance_field = out_dict_read['model'].to(device).eval()
    occupancy_grid = out_dict_read['occupancy_grid'].to(device)
    from utils.nerfacc_radiance_fields.datasets.lb.colmap_render import SubjectLoader_lb as SubjectLoader_render
    data_root_fp = args.data_root
    target_sample_batch_size = 1 << 16
    grid_resolution = 128
    
    contraction_type = ContractionType.AABB

    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None

    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()

    test_dataset_kwargs = {}
    test_dataset = SubjectLoader_render(
            subject_id=args.scene,
            root_fp=data_root_fp,
            split="render",
            num_rays=None,
            **test_dataset_kwargs,
        )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)
    test_dataset.task_ids = test_dataset.task_ids.to(device)



# print("step == max_steps = {}/{}/{},   step > 0 = {}, args.task_curr == (args.task_number - 1) = {}".format(step == max_steps, step, max_steps, step > 0, args.task_curr == (args.task_number - 1)))
# if step == max_steps and step > 0 and args.task_curr == (args.task_number - 1):
# evaluation
result_dir = f'results/WOT/EWC/{args.scene}_{args.rep_size}/video'
os.makedirs(result_dir, exist_ok=True)


if args.frame_start >= len(test_dataset):
    print("rendering already finished")
    exit()

args.frame_end = min(len(test_dataset)-1, args.frame_end)

with torch.no_grad():
    for i in tqdm.tqdm(range(args.frame_start, args.frame_end+1)):
        data = test_dataset[i]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        task_id = data['task_id'].flatten()

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

        rgb_save = (rgb.cpu().numpy()*255).astype(np.uint8)
        # rgb_video_writer.append_data(rgb_save)

        imageio.imsave(os.path.join(result_dir, '{}.png'.format(i)), rgb_save)

        
# rgb_video_writer.close()
# depth_video_writer.close()