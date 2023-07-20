"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os
from tqdm import tqdm

from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays
import glob
from .ray_utils import center_poses
# from .color_utils import read_image


def _load_renderings(data_dir: str, split: str):

    img_paths = sorted(glob.glob(os.path.join(data_dir, split, 'rgb/*')))
    poses = sorted(glob.glob(os.path.join(data_dir, split, 'pose/*.txt')))

    images = []
    camtoworlds = []
    print(f'Loading {len(img_paths)} {split} images ...')
    for img_path, pose in tqdm(zip(img_paths, poses)):
        camtoworlds += [np.loadtxt(pose).reshape(4, 4)[:3]]

        rgba = imageio.imread(img_path)
        # print("rgba.shape = {}".format(rgba.shape))
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    return images, camtoworlds


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]

    # WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 0.01, 16.0
    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "random",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        # assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (split
                                                    in ["train", "trainval"])
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        """Load images from disk."""
        if not root_fp.startswith("/"):
            # allow relative path. e.g., "./data/nerf_synthetic/"
            root_fp = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                root_fp,
            )

        self.root_dir = os.path.join(root_fp, subject_id)

        self.images, self.camtoworlds = _load_renderings(self.root_dir, split)
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.read_intrinsics()
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def read_intrinsics(self):
        K = np.loadtxt(glob.glob(
            os.path.join(self.root_dir, 'train/intrinsics/*.txt'))[0],
                       dtype=np.float32).reshape(4, 4)[:3, :3]
        w, h = Image.open(
            glob.glob(os.path.join(self.root_dir, 'train/rgb/*'))[0]).size
        # w, h = int(w), int(h)
        self.K = torch.FloatTensor(K)
        self.HEIGHT = int(h)
        self.WIDTH = int(w)

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        if rgba.shape[-1] == 4:
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
        else:
            pixels = rgba

        device = "cuda:0"

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=device)

        if rgba.shape[-1] == 4:
            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v
               for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        device = "cuda:0"

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays, ),
                )
            else:
                image_id = [index]
            x = torch.randint(0, self.WIDTH, size=(num_rays, ))
            y = torch.randint(0, self.HEIGHT, size=(num_rays, ))
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH),
                torch.arange(self.HEIGHT),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5) / self.K[1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True)

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, rgba.shape[-1]))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba,
                                 (self.HEIGHT, self.WIDTH, rgba.shape[-1]))

        rays = Rays(origins=origins.to(device), viewdirs=viewdirs.to(device))

        return {
            "rgba": rgba.to(device),  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3 or 4] or [num_rays, 3]
        }
