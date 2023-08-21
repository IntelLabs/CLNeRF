import torch
import glob
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

from ..ray_utils import get_ray_directions
from ..color_utils import read_image

from ..base import BaseDataset
import random

class NeRFPPDataset_lb(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        self.task_number = kwargs.get('task_number', 5)
        self.task_curr = kwargs.get('task_curr', 4)
        self.task_split_method = kwargs.get('task_split_method', 'seq')
        self.rep_size = kwargs.get('rep_size', 0)

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        K = np.loadtxt(glob.glob(os.path.join(self.root_dir, 'train/intrinsics/*.txt'))[0],
                       dtype=np.float32).reshape(4, 4)[:3, :3]
        K[:2] *= self.downsample
        w, h = Image.open(glob.glob(os.path.join(self.root_dir, 'train/rgb/*'))[0]).size
        w, h = int(w*self.downsample), int(h*self.downsample)
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def split_tasks(self, poses, task_number, task_split_method):
        # return task id for each element in poses
        task_id = []
        if task_split_method == 'random':
            for i in range(len(poses)):
                task_id.append(random.randint(0, task_number - 1))
        else:
            # equally split task according to the id
            imgs_per_task = len(poses) // task_number
            for j in range(task_number):
                task_id += ([j] * imgs_per_task) 
                # task_id.append()
            task_id += ([task_number-1] * (len(poses)- imgs_per_task * task_number)) 
        return task_id

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'test_traj':
            poses_path = \
                sorted(glob.glob(os.path.join(self.root_dir, 'camera_path/pose/*.txt')))
            self.poses = [np.loadtxt(p).reshape(4, 4)[:3] for p in poses_path]
        else:
            if split=='trainval':
                img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'train/rgb/*')))+\
                            sorted(glob.glob(os.path.join(self.root_dir, 'val/rgb/*')))
                poses = sorted(glob.glob(os.path.join(self.root_dir, 'train/pose/*.txt')))+\
                       sorted(glob.glob(os.path.join(self.root_dir, 'val/pose/*.txt')))
            else:
                img_paths = sorted(glob.glob(os.path.join(self.root_dir, split, 'rgb/*')))
                poses = sorted(glob.glob(os.path.join(self.root_dir, split, 'pose/*.txt')))

            if split == 'train':
                # split the training data into 5 tasks
                random.seed(0)
                self.task_ids = self.split_tasks(poses, self.task_number, self.task_split_method)
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
                    self.id_train_final = self.id_task_curr + random.choices(self.id_rep, k = self.rep_size)
            else:
                self.id_train_final = list(range(len(poses)))

            self.id_train_final.sort()

            print(f'Loading {len(img_paths)} {split} images ...')
            for id_train in tqdm(self.id_train_final):
            # for img_path, pose in tqdm(zip(img_paths, poses)):
                img_path, pose = img_paths[id_train], poses[id_train]
                self.poses += [np.loadtxt(pose).reshape(4, 4)[:3]]

                img = read_image(img_path, self.img_wh)
                self.rays += [img]

            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
