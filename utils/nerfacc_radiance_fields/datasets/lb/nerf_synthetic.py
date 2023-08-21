"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from ..utils import Rays
import math, random

def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


class SubjectLoader_lb(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
        "Chair",
        "Drums",
        "Ficus",
        "Hotdog",
        "Lego",
        "Materials",
        "Mic",
        "Ship",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        task_number: int = 10,
        task_curr: int = 9,
        task_split_method: str = "seq",
        rep_size: int = 0,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        id_rep = None
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )

        self.task_number = task_number
        self.task_curr = task_curr
        self.task_split_method = task_split_method
        self.rep_size = rep_size
        

        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        
        # split into 10 tasks
        num_img = self.camtoworlds.shape[0]
        img_per_task = math.ceil(num_img / self.task_number)
        print("img_per_task = {}".format(img_per_task))
        self.task_ids = []
        for i in range(num_img):
            self.task_ids.append(i // img_per_task)
        
        if split == 'train':
            # split the training data into 5 tasks
            # random.seed(0)
            # self.task_ids = self.split_tasks(num_img, self.task_number, self.task_split_method)
            # prepare training data
            self.id_task_curr = []
            self.id_rep = []
            
            for i in range(len(self.task_ids)):
                if self.task_ids[i] == self.task_curr:
                    self.id_task_curr.append(i)
                elif self.task_ids[i] < self.task_curr:
                    self.id_rep.append(i)
            if self.rep_size == 0 or self.task_curr == 0:
                self.id_train_final = self.id_task_curr
            else:
                # set random seed
                # we need to implement reservoir buffer
                if id_rep is not None:
                    self.id_rep = id_rep
                    print("self.id_rep = {}".format(self.id_rep))
                self.id_train_final = self.id_task_curr + self.id_rep
            # perform reservoir sampling
            if len(self.id_train_final) <= self.rep_size:
                self.rep_buf = self.id_train_final
            else:
                self.rep_buf = self.id_rep.copy()
                offset = self.task_curr * img_per_task
                for i, id_curr in enumerate(self.id_task_curr):
                    rand_int = random.randint(0, offset+i)
                    if rand_int < len(self.rep_buf):
                        self.rep_buf[rand_int] = id_curr
            print("rep_buf = {}".format(self.rep_buf))


            # print("[test] use {} ({}) from {} images (task_number = {}, rep_size = {})".format(len(self.id_train_final), self.id_train_final, num_img, self.task_number, self.rep_size))
        else:
            self.id_train_final = list(range(num_img))
        self.id_train_final.sort()

        if split == 'train':
            self.images = torch.from_numpy(self.images[self.id_train_final]).to(torch.uint8)
            self.camtoworlds = torch.from_numpy(self.camtoworlds[self.id_train_final]).to(torch.float32)        
        else:
            self.images = torch.from_numpy(self.images).to(torch.uint8)
            self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
            
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        self.height, self.width = self.images.shape[1:3]
        self.task_ids = (torch.tensor(self.task_ids).to(torch.uint8))[self.id_train_final]

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
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        task_id = self.task_ids[image_id]
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
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
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))
            task_id = task_id * torch.ones((self.height, self.width, 1), device = task_id.device)

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            'task_id': task_id.int()
        }
