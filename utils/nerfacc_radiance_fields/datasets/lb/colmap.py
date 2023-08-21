"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm, random

from ..utils import Rays
from ..colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

import math, random

_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "../../..", "pycolmap",
                    "pycolmap"))
from scene_manager import SceneManager


def name_to_task(img_paths):
    tasks, task_list, test_ids, train_ids = [], [], [], []
    for img_path in img_paths:
        task_folder_name = os.path.basename(os.path.dirname(img_path))
        if task_folder_name not in tasks:
            tasks.append(task_folder_name)
        task_list.append(tasks.index(task_folder_name))

    img_count = 0
    task_curr = -1
    for i in range(len(task_list)):
        if task_list[i] > task_curr:
            task_curr, img_count = task_list[i], 0
            test_ids.append(i)
        elif img_count % 8 == 0:
            test_ids.append(i)
        else:
            train_ids.append(i)
        img_count += 1

    return task_list, test_ids, train_ids


def _load_colmap(root_fp: str, subject_id: str, split: str, factor: int = 1):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0/")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    camdata = read_cameras_binary(os.path.join(colmap_dir, 'cameras.bin'))

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    # fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    if camdata[1].model == 'SIMPLE_RADIAL':
        fx = fy = camdata[1].params[0]
        cx = camdata[1].params[1]
        cy = camdata[1].params[2]
    elif camdata[1].model in ['PINHOLE', 'OPENCV']:
        fx = camdata[1].params[0]
        fy = camdata[1].params[1]
        cx = camdata[1].params[2]
        cy = camdata[1].params[3]
    else:
        raise ValueError(
            f"Please parse the intrinsics for camera model {camdata[1].model}!"
        )

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor

    # Extract extrinsic matrices in world-to-camera format.
    imdata = read_images_binary(os.path.join(colmap_dir, 'images.bin'))

    w2c_mats = []
    bottom = np.array([[0, 0, 0, 1.]])
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    image_names = [imdata[k].name for k in imdata]
    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    if factor > 1:
        image_dir_suffix = f"_{factor}"
    else:
        image_dir_suffix = ""
    colmap_image_dir = os.path.join(data_dir, "images")
    image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(os.listdir(colmap_image_dir))
    image_files = sorted(os.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [os.path.join(image_dir, f) for f in image_names]
    # get the task id
    task_ids, test_img_ids, train_img_ids = name_to_task(image_paths)

    print("loading images")

    images = [imageio.imread(x) for x in tqdm.tqdm(image_paths)]
    images = np.stack(images, axis=0)

    # Select the split.
    all_indices = np.arange(images.shape[0])

    split_indices = {
        "test": all_indices[test_img_ids],
        "train": all_indices[train_img_ids],
    }
    indices = split_indices[split]

    # center and rescale camera poses
    # center all cameras to AABB center
    camera_locs = camtoworlds[:, :3, -1]
    shift = (camera_locs.max(axis=0) - camera_locs.min(axis=0)) / 2.0

    pts3d = read_points3d_binary(
        os.path.join(data_dir, 'sparse/0/points3D.bin'))
    pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

    # compute near far
    xyz_world_h = np.concatenate([pts3d, np.ones((len(pts3d), 1))], -1)
    # Compute near and far bounds for each image individually
    fars = {}  # {id_: distance}
    for i in range(w2c_mats.shape[0]):
        xyz_cam_i = (xyz_world_h
                     @ w2c_mats[i].T)[:, :3]  # xyz in the ith cam coordinate
        xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] >
                              0]  # filter out points that lie behind the cam
        fars[i] = np.percentile(xyz_cam_i[:, 2], 99.9)
    max_far = np.fromiter(fars.values(), np.float32).max()

    scale = max_far / 1.25

    camtoworlds[:, :3, 3] -= shift
    camtoworlds[:, :3, 3] /= scale

    # All per-image quantities must be re-indexed using the split indices.
    images = images[indices]
    camtoworlds = camtoworlds[indices]

    return images, camtoworlds, K, np.array(task_ids)[indices]


class SubjectLoader_lb(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test", "render"]

    OPENGL_CAMERA = False

    def __init__(self,
                 subject_id: str,
                 root_fp: str,
                 split: str,
                 task_number: int = 10,
                 task_curr: int = 9,
                 task_split_method: str = "seq",
                 rep_size: int = 0,
                 color_bkgd_aug: str = "random",
                 num_rays: int = None,
                 near: float = None,
                 far: float = None,
                 batch_over_images: bool = True,
                 factor: int = 1,
                 id_rep=None):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        if split == 'render':
            self.split = 'test'
        else:
            self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (split
                                                    in ["train", "trainval"])

        self.task_number = task_number
        self.task_curr = task_curr
        self.task_split_method = task_split_method
        self.rep_size = rep_size

        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.images, self.camtoworlds, self.K, self.task_ids = _load_colmap(
            root_fp, subject_id, self.split, factor)

        num_img = self.camtoworlds.shape[0]
        img_per_task = math.ceil(num_img / self.task_number)
        print("img_per_task = {}".format(img_per_task))
        if split == 'train':
            # split the training data into 5 tasks
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
                if id_rep is not None:
                    self.id_rep = id_rep
                    print("self.id_rep = {}".format(self.id_rep))
                self.id_train_final = self.id_task_curr + self.id_rep
            print("id_train_final = {}, rep_size = {}".format(self.id_train_final, self.rep_size))
            if len(self.id_train_final) <= self.rep_size:
                self.rep_buf = self.id_train_final
            else:
                self.rep_buf = self.id_rep.copy()
                offset = self.task_curr * img_per_task
                for i, id_curr in enumerate(self.id_task_curr):
                    rand_int = random.randint(0, offset + i)
                    # print("[test reservoir] i = {}, offset = {}, randint = {}".format(i, offset, rand_int))
                    if len(self.rep_buf) < self.rep_size:
                        self.rep_buf.append(id_curr)
                    elif rand_int < len(self.rep_buf):
                        self.rep_buf[rand_int] = id_curr
            print("rep_buf = {}".format(self.rep_buf))
        else:
            self.id_train_final = list(range(num_img))
        self.id_train_final.sort()

        self.images = torch.from_numpy(self.images[self.id_train_final]).to(
            torch.uint8)
        self.camtoworlds = torch.from_numpy(
            self.camtoworlds[self.id_train_final]).to(torch.float32)

        self.K = torch.tensor(self.K).to(torch.float32)
        self.height, self.width = self.images.shape[1:3]
        self.task_ids = (torch.from_numpy(self.task_ids).to(
            torch.uint8))[self.id_train_final]

    def split_tasks(self, num_img, task_number, task_split_method):
        # return task id for each element in poses
        task_id = []
        if task_split_method == 'random':
            for i in range(num_img):
                task_id.append(random.randint(0, task_number - 1))
        else:
            # equally split task according to the id
            imgs_per_task = num_img // task_number
            for j in range(task_number):
                task_id += ([j] * imgs_per_task)
                # task_id.append()
            task_id += ([task_number - 1] *
                        (num_img - imgs_per_task * task_number))
        return task_id

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]

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

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v
               for k, v in data.items() if k not in ["rgb", "rays"]},
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
                    size=(num_rays, ),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(0,
                              self.width,
                              size=(num_rays, ),
                              device=self.images.device)
            y = torch.randint(0,
                              self.height,
                              size=(num_rays, ),
                              device=self.images.device)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        task_id = self.task_ids[image_id]
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
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height, self.width, 3))
            task_id = task_id * torch.ones(
                (self.height, self.width, 1), device=task_id.device)

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            'task_id': task_id.int()
        }
