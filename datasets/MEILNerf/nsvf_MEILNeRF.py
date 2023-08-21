import torch
import glob
import numpy as np
import os
from tqdm import tqdm

from ..ray_utils import get_ray_directions
from ..color_utils import read_image

from ..base import BaseDataset
import random

class NSVFDataset_MEILNeRF(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        self.task_number = kwargs.get('task_number', 5)
        self.task_curr = kwargs.get('task_curr', 4)
        self.task_split_method = kwargs.get('task_split_method', 'seq')
        self.rep_size = kwargs.get('rep_size', 0)
        self.rep_dir = kwargs.get('rep_dir', '')
        self.nerf_rep = kwargs.get('nerf_rep', True)

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

        if split == 'train' or split == 'rep': prefix = '0_'
        elif split == 'trainval': prefix = '[0-1]_'
        elif split == 'trainvaltest': prefix = '[0-2]_'
        elif split == 'val': prefix = '1_'
        elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
        elif split == 'test': prefix = '1_' # test set for real scenes
        else: raise ValueError(f'{split} split not recognized!')
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
        poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

        if split == 'train' or split == 'rep':
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


            if self.rep_size == 0:
                self.id_train_final = self.id_task_curr
            elif split == 'rep':
                self.id_train_final = self.id_task_curr + self.id_rep
                print("task_curr = {}/{}".format(self.task_curr, (self.task_number-1)))
            else:
                # set random seed
                # choose randomly if we are in the first task
                dir_name = self.rep_dir
                if self.task_curr == 0:
                    rep_data = []
                else:
                    # read replay ID
                    rep_data = torch.load(os.path.join(dir_name, 'rep_buf.torchSave'))
                self.id_train_final = self.id_task_curr + rep_data
                # create replay data
                for i in range(len(self.task_ids)):
                    if self.task_ids[i] == self.task_curr:
                        # reservoir sampling
                        if len(rep_data) < self.rep_size:
                            rep_data.append(i)
                            print("[test reservoir]: putting rep data {} into replay buffer of size {}".format(i, len(rep_data)-1))
                        else:
                            id_sample = random.randint(0, i)
                            if id_sample < len(rep_data):
                                rep_data[id_sample] = i
                                print("[test reservoir]: putting rep data {} into slot {}".format(i, id_sample))
                os.makedirs(dir_name, exist_ok = True)
                torch.save(rep_data, os.path.join(dir_name, 'rep_buf.torchSave'))

            # replace replay images with nerf rendered version if they are not in the replay buffer
            if self.nerf_rep:
                dir_name = self.rep_dir
                for t, id_rep in enumerate(self.id_rep):
                    if id_rep not in self.id_train_final:
                        rep_name = os.path.join(dir_name, os.path.basename(img_paths[id_rep]))
                        if t % 10 == 0:
                            print("changing {} to {}".format(img_paths[id_rep], rep_name))
                        img_paths[id_rep] = rep_name
                self.id_train_final = self.id_task_curr + self.id_rep
            print("self.id_task_curr = {}, self.rep_size = {}, id_train_final = {}".format(self.id_task_curr, self.rep_size, self.id_train_final))
        else:
            self.id_train_final = list(range(len(poses)))

        self.id_train_final.sort()
        self.img_paths = img_paths

        print(f'Loading {len(self.id_train_final)} {split} images ...')
        print('id_train_final = {}'.format(self.id_train_final))
        
        self.rays = []
        self.poses = []

        if self.split == 'train':
            self.id_rep_MEIL = []
            self.id_curr_MEIL = []
        for j, id_train in enumerate(tqdm(self.id_train_final)):
            if self.split == 'train':
                if id_train in self.id_rep:
                    self.id_rep_MEIL.append(j)
                else:
                    self.id_curr_MEIL.append(j)
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

        if self.split == 'train':
            self.id_rep_MEIL = torch.tensor(self.id_rep_MEIL).long()
            self.id_curr_MEIL = torch.tensor(self.id_curr_MEIL).long()
            print("[test] id_rep_MEIL = {}, id_curr_MEIL = {}".format(self.id_rep_MEIL, self.id_curr_MEIL))

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            if self.task_curr == 0:
                # training pose is retrieved in train.py
                if self.ray_sampling_strategy == 'all_images': # randomly select images
                    img_idxs = np.random.choice(len(self.poses), self.batch_size)
                elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                    img_idxs = np.random.choice(len(self.poses), 1)[0]
                # randomly select pixels
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
                rays = self.rays[img_idxs, pix_idxs]
                sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                        'rgb': rays[:, :3], 'is_rep': torch.zeros(self.batch_size).int()}
                if self.rays.shape[-1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[:, 3:]
            else:
                # sample 2/3 from the current task and 1/3 from from old tasks
                # training pose is retrieved in train.py
                img_idxs = torch.cat((self.id_rep_MEIL[np.random.choice(self.id_rep_MEIL.shape[0], self.batch_size//3)], 
                self.id_curr_MEIL[np.random.choice(self.id_curr_MEIL.shape[0], self.batch_size-self.batch_size//3)]), dim=0)
            
                # randomly select pixels
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
                rays = self.rays[img_idxs, pix_idxs]
                sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                        'rgb': rays[:, :3], 'is_rep': torch.cat((torch.ones(self.batch_size//3).int(), torch.zeros(self.batch_size-self.batch_size//3).int()))}
                if self.rays.shape[-1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[:, 3:]

        elif self.split.startswith('rep'):
            sample = {'pose': self.poses[idx], 'img_idxs': idx, 'fname': os.path.basename(self.img_paths[self.id_train_final[idx]]),'id_ori': self.id_train_final[idx],'task_id': self.task_ids[self.id_train_final[idx]]}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample



