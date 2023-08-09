import torch
import glob
import numpy as np
import os
from tqdm import tqdm

from ..ray_utils import get_ray_directions
from ..color_utils import read_image

from ..base import BaseDataset
import random

class NSVFDataset_lb(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        self.task_number = kwargs.get('task_number', 5)
        self.task_curr = kwargs.get('task_curr', 4)
        self.task_split_method = kwargs.get('task_split_method', 'seq')
        self.rep_size = kwargs.get('rep_size', 0)
        

        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max+xyz_min)/2
            self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

            # hard-code fix the bound error for some scenes...
            if 'Mic' in self.root_dir: self.scale *= 1.2
            elif 'Lego' in self.root_dir: self.scale *= 1.1

            self.read_meta(split)

    def read_intrinsics(self):
        if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
            with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * self.downsample
            if 'Synthetic' in self.root_dir:
                w = h = int(800*self.downsample)
            else:
                w, h = int(1920*self.downsample), int(1080*self.downsample)

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'BlendedMVS' in self.root_dir:
                w, h = int(768*self.downsample), int(576*self.downsample)
            elif 'Tanks' in self.root_dir:
                w, h = int(1920*self.downsample), int(1080*self.downsample)
            K[:2] *= self.downsample

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

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
        else:
            if split == 'train': prefix = '0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif split == 'trainvaltest': prefix = '[0-2]_'
            elif split == 'val': prefix = '1_'
            elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
            elif split == 'test': prefix = '1_' # test set for real scenes
            else: raise ValueError(f'{split} split not recognized!')
            img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
            poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

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
           
            print(f'Loading {len(self.id_train_final)} {split} images ...')
            print('id_train_final = {}'.format(self.id_train_final))
            for id_train in tqdm(self.id_train_final):
                img_path, pose = img_paths[id_train], poses[id_train]
                c2w = np.loadtxt(pose)[:3]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]

                img = read_image(img_path, self.img_wh)
                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0

                self.rays += [img]

            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)




