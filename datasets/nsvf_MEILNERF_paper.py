import torch
import glob
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class NSVFDataset_MEILNERF_paper(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

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

    def read_meta(self, split):
        self.rays = []
        self.poses = []
        
        # if split == 'train': prefix = '0_'
        # elif split == 'test': prefix = '1_' # test set for real scenes
        # else: raise ValueError(f'{split} split not recognized!')
        # read order files
        meil_order_file = os.path.join(self.root_dir, 'MEIL_{}.txt'.format(split))
        # Open the file for reading
        with open(meil_order_file, "r") as file:
            # Read all the lines in the file
            lines = file.readlines()

        # Strip newline characters from each line and store in a list
        lines = [line.strip() for line in lines]
        img_paths, poses = [], []
        for line in lines:
            img_paths.append(os.path.join(self.root_dir, 'rgb', line[:-4]+'.png'))
            poses.append(os.path.join(self.root_dir, 'pose', line))
            
        # print("img_paths = {}, poses = {}".format(img_paths, poses))
        # exit()
        print("img_paths[-1] = {}, poses[-1] = {}, img_paths.shape = {}".format(img_paths[-1], poses[-1], len(img_paths)))

        # img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')), key=lambda x: x[-8:-4])
        # poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')), key=lambda x: x[-8:-4])

        # # MEIL-NERF
        # if split == 'train':
        #     img_paths = img_paths[:100]
        #     poses = poses[:100]
        #     print("[train test] img_paths[-1] = {}".format(img_paths[-1]))
        # else:
        #     # get the max id of train data
        #     prefix_train = '0_'
        #     id_final = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix_train+'*.png')), key=lambda x: x[-8:-4])[99][-8:-4]
        #     # get the first 100 training images
        #     id_selected = 0
        #     for i in range(len(img_paths)):
        #         if img_paths[i][-8:-4] <= id_final:
        #             id_selected += 1
        #         else:
        #             break
        #     img_paths = img_paths[:id_selected]
        #     poses = poses[:id_selected]
        #     print("[test test] img_paths[-1] = {}".format(img_paths[-1]))


        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path, pose in tqdm(zip(img_paths, poses)):
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



