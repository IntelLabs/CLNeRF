import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from ..ray_utils import *
from ..color_utils import read_image_phototour
from ..colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from ..base import BaseDataset
import pandas as pd
import random

class PhotoTourDatasetNerfw(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        # self.read_intrinsics()

        # in each mask, 1 is filtered pixel, 0 is a static pixel
        self.use_mask = kwargs.get('use_mask', 1) 
        self.f_train_val = kwargs.get('f_train_val', '') 
        self.measure_psnr = kwargs.get('psnr', 1)

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self, img_paths, img_path_to_id, crop_region = 'full'):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/cameras.bin'))

        self.img_wh, self.K, self.directions = [], [], []
        for i in range(len(img_paths)):
            # print("img_path_to_id = {}".format(img_path_to_id.keys()))
            cam_id = img_path_to_id[os.path.basename(img_paths[i])]
            # exit()
            h = int(camdata[cam_id].height*self.downsample)
            w = int(camdata[cam_id].width*self.downsample)
            self.img_wh.append(torch.tensor([w, h]))

            if camdata[cam_id].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[cam_id].params[0]*self.downsample
                cx = camdata[cam_id].params[1]*self.downsample
                cy = camdata[cam_id].params[2]*self.downsample
            elif camdata[cam_id].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[cam_id].params[0]*self.downsample
                fy = camdata[cam_id].params[1]*self.downsample
                cx = camdata[cam_id].params[2]*self.downsample
                cy = camdata[cam_id].params[3]*self.downsample
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            self.K.append(torch.FloatTensor([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0,  0,  1]]))
            self.directions.append(get_ray_directions(h, w, self.K[i], crop_region = crop_region))
            if i % 100 == 0:
                print("cam_id[{}] = {}, img_wh = {}, self.directions.shape = {}".format(i, cam_id, self.img_wh[i], self.directions[i].shape))
        # stack K for later use
        self.K = torch.stack(self.K)
        self.img_wh = torch.stack(self.img_wh).int()

        print("self.K.shape = {}, self.img_wh.shape = {}".format(self.K.shape, self.img_wh.shape))
        # exit()
        #     print("self.directions[{}] = {}/{}".format(i, self.directions.shape, self.directions.device))
        # exit()

    def get_split(self, train_val, img_names_all, perm, split):
        # train_val: training and validation split dictionary
        # img_names_all: all images' names in the original order stored in imdata
        # per: permutation index when image names are sorted
        # output: 1. img_names: image names in the split
        #         2. selected_idx: selected idx in perm where images are in the split, used to get the corresponding poses later 

        # img_names = [imdata[k].name for k in imdata if ]
        # map to train test
        img_to_split = {}
        for i in range(len(train_val['filename'])):
            img_to_split[train_val['filename'][i]] = train_val['split'][i]
        # img_names = [train_val['filename'][i] for i in range(len(train_val['filename'])) if train_val['split'][i] == split]
        # imdata_new = {}
        img_names, selected_idx = [], []
        for i, id_perm in enumerate(perm):
            if img_names_all[id_perm] in img_to_split:
                if img_to_split[img_names_all[id_perm]] == split:
                    img_names.append(img_names_all[id_perm])
                    selected_idx.append(i)

        return img_names, selected_idx
        # for i in range(len(train_val['filename'])):


    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/images.bin'))
        # print("imdata = {}".format(imdata.keys()))
        # exit()
        img_path_to_id = {}
        for v in imdata.values():
            img_path_to_id[v.name] = v.id
        
        # read train_val separation info
        train_val = pd.read_csv(self.f_train_val, sep='\t')
        # check whether this name exist in img_path_to_id
        # print("train_val = {}, id = {}, id_in_file = {}, len = {}".format(train_val['filename'][0], img_path_to_id[train_val['filename'][0]], train_val['id'][0], len(train_val['filename'])))
        # exit()

        img_names_all = [imdata[k].name for k in imdata]
        # get map from img path to camera id in the intrinsic database, used to construct the right intrinsics        
        perm = np.argsort(img_names_all)
        
        # get all training data
        if split == 'train':
            img_names, selected_idx = self.get_split(train_val, img_names_all, perm, 'train')
        else:
            img_names, selected_idx = self.get_split(train_val, img_names_all, perm, 'test')
        # print("img_names = {}, len = {}".format(img_names, len(img_names)))
        # exit()
        # img_names = [imdata[k].name for k in imdata]
        if '360_v2' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        
        mask_folder = 'seg'

        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        mask_paths = [os.path.join(self.root_dir, mask_folder, os.path.splitext(name)[0]+'.torchSave')
                     for name in sorted(img_names)]
        
        if split =='eval':
            self.read_intrinsics(img_paths, img_path_to_id, crop_region = 'left')
        elif split == 'test' and self.measure_psnr:
            self.read_intrinsics(img_paths, img_path_to_id, crop_region = 'right')
        else:
            self.read_intrinsics(img_paths, img_path_to_id)
        

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])

        # this part is wrong, should center poses for all data, including train and test, and then get the subset
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        # scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        scale = np.linalg.norm(self.poses[..., 3], axis=-1).max()

        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.poses = self.poses[selected_idx]
        

        self.rays, self.ts, self.img_ids, self.pix_ids, self.rays_dir = [], [], [], [], []

        print("[after train-test split] self.K.shape = {}, self.img_wh.shape = {}, self.directions.len = {}".format(self.K.shape, self.img_wh.shape, len(self.directions)))
        print(f'Loading {len(img_paths)} {split} images ...')
        if split == 'train':
            for t, img_path in enumerate(tqdm(img_paths)):
                buf = [] # buffer for ray attributes: rgb, etc

                # image has already been flattened into (hw, c)
                img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)

                mask = torch.load(mask_paths[t])
                # reshape mask
                mask = rearrange(mask, 'c h w -> (h w c)')

                # get the rays that is not masked
                buf += [img[mask == 0]]
                # and the pixel id of these locations (self.directions and imgs are already flattened)
                pIDs = (mask == 0).nonzero().flatten().tolist()

                if t % 100 == 0:
                    print("mask = {}/{}/{}/{}".format(mask, mask.min(), mask.max(), (mask == 0).sum()))
                    print("[test {}]: mask_name = {}, mask = {}, buf = {}, self.directions[t].shape = {}, img_wh = {}".format(t, mask_paths[t], mask.shape, buf[0].shape, self.directions[t].shape, self.img_wh[t]))

                # skip if the meta data is different from what we read from the image
                if self.img_wh[t][0] * self.img_wh[t][1] != mask.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                else:
                    self.rays += [torch.cat(buf, 1)]
                    self.rays_dir += [self.directions[t][mask == 0]]
                    self.img_ids += ([t] * buf[0].shape[0])
                    self.pix_ids += pIDs

                self.ts += [t]
            self.rays = torch.cat(self.rays)
            self.rays_dir = torch.cat(self.rays_dir)
            self.img_ids = torch.tensor(self.img_ids).int()
            self.pix_ids = torch.tensor(self.pix_ids).int()
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.ts = torch.tensor(self.ts).int()
            print("self.rays.size = {}, self.img_ids = {}, self.pix_ids = {}, self.rays_dir = {}".format(self.rays.shape, self.img_ids.shape, self.pix_ids.shape, self.rays_dir.shape))
            # exit()
        elif split == 'eval':
            # keep only the first half of the pixels per image
            for t, img_path in enumerate(tqdm(img_paths)):
                buf = [] # buffer for ray attributes: rgb, etc

                # image has already been flattened into (hw, c)
                img = read_image_phototour(img_path, blend_a=False, crop_region = 'left')
                img = torch.FloatTensor(img)

                mask = torch.load(mask_paths[t])
                # print("[before] mask.shape = {}".format(mask.shape))
                mask = mask[:, :, :mask.shape[2]//2]
                # print("[after] mask.shape = {}".format(mask.shape))
                # reshape mask
                mask = rearrange(mask, 'c h w -> (h w c)')

                # only get the first half of the data
                # mask = mask[:mask.shape[0]]
                
                # get the rays that is not masked
                buf += [img[mask == 0]]
                # and the pixel id of these locations (self.directions and imgs are already flattened)
                pIDs = (mask == 0).nonzero().flatten().tolist()

                if t % 100 == 0:
                    print("mask = {}/{}/{}/{}".format(mask, mask.min(), mask.max(), (mask == 0).sum()))
                    print("[test {}]: mask_name = {}, mask = {}, buf = {}, self.directions[t].shape = {}, img_wh = {}".format(t, mask_paths[t], mask.shape, buf[0].shape, self.directions[t].shape, self.img_wh[t]))

                # skip if the meta data is different from what we read from the image
                if ((self.img_wh[t][0]//2) * self.img_wh[t][1]) != mask.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                else:
                    rays_dir = self.directions[t]
                    self.rays += [torch.cat(buf, 1)]
                    self.rays_dir += [rays_dir[mask == 0]]
                    self.img_ids += ([t] * buf[0].shape[0])
                    self.pix_ids += pIDs

                self.ts += [-1-t]
                # if t >= 100:
                #     break
            self.rays = torch.cat(self.rays)
            self.rays_dir = torch.cat(self.rays_dir)
            self.img_ids = torch.tensor(self.img_ids).int()
            self.pix_ids = torch.tensor(self.pix_ids).int()
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.ts = torch.tensor(self.ts).int()
            print("self.rays.size = {}, self.img_ids = {}, self.pix_ids = {}, self.rays_dir = {}".format(self.rays.shape, self.img_ids.shape, self.pix_ids.shape, self.rays_dir.shape))
        else:
            self.is_valid = []
            for t, img_path in enumerate(tqdm(img_paths)):
                buf = [] # buffer for ray attributes: rgb, etc

                if self.measure_psnr:
                    img = read_image_phototour(img_path, blend_a=False, crop_region = 'right')
                    self.img_wh[t][0] = self.img_wh[t][0]-self.img_wh[t][0]//2
                    # print("self.directions[t] = {}, self.img_wh = {}".format(self.directions[t].shape, self.img_wh[t]))
                else:
                    img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)
                if self.img_wh[t][0] * self.img_wh[t][1] != img.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                    self.is_valid.append(0)
                else:
                    self.is_valid.append(1)
                # print("[test]: loading img {}, size = {}, img_wh = {}".format(img_path, img.shape, self.img_wh[t]))
                buf += [img]

                self.rays += [torch.cat(buf, 1)]
            # self.rays = torch.stack(self.rays) # (N_images, hw, ?)
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.is_valid = torch.tensor(self.is_valid).int().nonzero().flatten()
            print("is_valid = {}".format(self.is_valid.shape))
        # exit()
    
    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        elif self.split.startswith('eval'):
            return 1000
        return len(self.is_valid)

    def __getitem__(self, idx):
        if self.split.startswith('train') or self.split.startswith('eval'):
            # training pose is retrieved in train.py
            # if self.ray_sampling_strategy == 'all_images': # randomly select images
            #     img_idxs = np.random.choice(len(self.poses), self.batch_size)
            # elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
            #     img_idxs = np.random.choice(len(self.poses), 1)[0]
            # # randomly select pixels
            # pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            # rays = torch.FloatTensor([self.rays[img_idxs[i], pix_idxs[i]] for i in range(self.batch_size)])
            # rays = torch.FloatTensor(self.rays)
            # randomly sample a batch of pixels
            ray_idxs = np.random.choice(self.rays.shape[0], self.batch_size)
            rays = torch.FloatTensor(self.rays[ray_idxs])
            img_idxs = self.img_ids[ray_idxs].long()
            pix_idxs = self.pix_ids[ray_idxs].long()
            # print("img_idxs = {}/{}/{}, pix_idxs = {}/{}/{}, self.directions[0].shape = {}".format(img_idxs.shape, img_idxs.min(), img_idxs.max(), pix_idxs.shape, pix_idxs.min(), pix_idxs.max(), self.directions[0].shape))
            # ray_dirs = []
            # for i in range(self.batch_size):
            #     ray_dirs.append()
            # ray_dirs = torch.FloatTensor([self.directions[img_idxs[i].item()][pix_idxs[i].item()] for i in range(self.batch_size)])
            # print("rays.shape = {}, img_idxs.shape = {}, pix_idxs.shape = {}, ray_dirs.shape = {}".format(rays.shape, img_idxs.shape, pix_idxs.shape, ray_dirs.shape))
            # exit()
            ray_dirs = torch.FloatTensor(self.rays_dir[ray_idxs])
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3],
                      'ts': self.ts[img_idxs],
                      'ray_dirs': ray_dirs}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            idx = self.is_valid[idx]
            sample = {'pose': self.poses[idx], 'img_idxs': idx, 'ts': torch.tensor([0]).int(), 'ray_dirs': self.directions[idx], 'img_wh': self.img_wh[idx]}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample






class PhotoTourDatasetNerfw_lb(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        # self.read_intrinsics()

        # in each mask, 1 is filtered pixel, 0 is a static pixel
        self.use_mask = kwargs.get('use_mask', 1) 
        self.f_train_val = kwargs.get('f_train_val', '') 
        self.measure_psnr = kwargs.get('psnr', 1)

        self.task_number = kwargs.get('task_number', 10)
        self.task_curr = kwargs.get('task_curr', 9)
        self.task_split_method = kwargs.get('task_split_method', 'seq')
        self.rep_size = kwargs.get('rep_size', 0)
        # self.img_per_appearance = kwargs.get('img_per_appearance', 10)
        
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self, img_paths, img_path_to_id, crop_region = 'full'):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/cameras.bin'))

        self.img_wh, self.K, self.directions = [], [], []
        for i in range(len(img_paths)):
            # print("img_path_to_id = {}".format(img_path_to_id.keys()))
            cam_id = img_path_to_id[os.path.basename(img_paths[i])]
            # exit()
            h = int(camdata[cam_id].height*self.downsample)
            w = int(camdata[cam_id].width*self.downsample)
            self.img_wh.append(torch.tensor([w, h]))

            if camdata[cam_id].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[cam_id].params[0]*self.downsample
                cx = camdata[cam_id].params[1]*self.downsample
                cy = camdata[cam_id].params[2]*self.downsample
            elif camdata[cam_id].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[cam_id].params[0]*self.downsample
                fy = camdata[cam_id].params[1]*self.downsample
                cx = camdata[cam_id].params[2]*self.downsample
                cy = camdata[cam_id].params[3]*self.downsample
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            self.K.append(torch.FloatTensor([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0,  0,  1]]))
            self.directions.append(get_ray_directions(h, w, self.K[i], crop_region = crop_region))
            if i % 100 == 0:
                print("cam_id[{}] = {}, img_wh = {}, self.directions.shape = {}".format(i, cam_id, self.img_wh[i], self.directions[i].shape))
        # stack K for later use
        self.K = torch.stack(self.K)
        self.img_wh = torch.stack(self.img_wh).int()

        print("self.K.shape = {}, self.img_wh.shape = {}".format(self.K.shape, self.img_wh.shape))
        # exit()
        #     print("self.directions[{}] = {}/{}".format(i, self.directions.shape, self.directions.device))
        # exit()

    def get_split(self, train_val, img_names_all, perm, split):
        # train_val: training and validation split dictionary
        # img_names_all: all images' names in the original order stored in imdata
        # per: permutation index when image names are sorted
        # output: 1. img_names: image names in the split
        #         2. selected_idx: selected idx in perm where images are in the split, used to get the corresponding poses later 

        # img_names = [imdata[k].name for k in imdata if ]
        # map to train test
        img_to_split = {}
        for i in range(len(train_val['filename'])):
            img_to_split[train_val['filename'][i]] = train_val['split'][i]
        # img_names = [train_val['filename'][i] for i in range(len(train_val['filename'])) if train_val['split'][i] == split]
        # imdata_new = {}
        img_names, selected_idx = [], []
        for i, id_perm in enumerate(perm):
            if img_names_all[id_perm] in img_to_split:
                if img_to_split[img_names_all[id_perm]] == split:
                    img_names.append(img_names_all[id_perm])
                    selected_idx.append(i)

        return img_names, selected_idx
        # for i in range(len(train_val['filename'])):



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

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/images.bin'))
        # print("imdata = {}".format(imdata.keys()))
        # exit()
        img_path_to_id = {}
        for v in imdata.values():
            img_path_to_id[v.name] = v.id
        
        # read train_val separation info
        train_val = pd.read_csv(self.f_train_val, sep='\t')
        # check whether this name exist in img_path_to_id
        # print("train_val = {}, id = {}, id_in_file = {}, len = {}".format(train_val['filename'][0], img_path_to_id[train_val['filename'][0]], train_val['id'][0], len(train_val['filename'])))
        # exit()

        img_names_all = [imdata[k].name for k in imdata]
        # get map from img path to camera id in the intrinsic database, used to construct the right intrinsics        
        perm = np.argsort(img_names_all)
        
        # get all training data
        if split == 'train':
            img_names, selected_idx = self.get_split(train_val, img_names_all, perm, 'train')
            # get (replay) data for the current task
            random.seed(0)
            self.task_ids = self.split_tasks(img_names, self.task_number, self.task_split_method)
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

            self.id_train_final.sort()
            print("self.id_train_final = {}/{}".format(self.id_train_final, len(self.id_train_final)))
        else:
            img_names, selected_idx = self.get_split(train_val, img_names_all, perm, 'test')


        # print("img_names = {}, len = {}".format(img_names, len(img_names)))
        # exit()
        # img_names = [imdata[k].name for k in imdata]
        if '360_v2' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        
        mask_folder = 'seg'

        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        mask_paths = [os.path.join(self.root_dir, mask_folder, os.path.splitext(name)[0]+'.torchSave')
                     for name in sorted(img_names)]
        
        if split =='eval':
            self.read_intrinsics(img_paths, img_path_to_id, crop_region = 'left')
        elif split == 'test' and self.measure_psnr:
            self.read_intrinsics(img_paths, img_path_to_id, crop_region = 'right')
        else:
            self.read_intrinsics(img_paths, img_path_to_id)
        

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])

        # this part is wrong, should center poses for all data, including train and test, and then get the subset
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        # scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        scale = np.linalg.norm(self.poses[..., 3], axis=-1).max()

        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.poses = self.poses[selected_idx]
        

        self.rays, self.ts, self.img_ids, self.t_ids, self.pix_ids, self.rays_dir = [], [], [], [], [], []

        print("[after train-test split] self.K.shape = {}, self.img_wh.shape = {}, self.directions.len = {}".format(self.K.shape, self.img_wh.shape, len(self.directions)))
        print(f'Loading {len(img_paths)} {split} images ...')
        if split == 'train':
            # for t, img_path in enumerate(tqdm(img_paths)):
            for i, t in enumerate(tqdm(self.id_train_final)):
                img_path = img_paths[t]
                buf = [] # buffer for ray attributes: rgb, etc

                # image has already been flattened into (hw, c)
                img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)

                mask = torch.load(mask_paths[t])
                # reshape mask
                mask = rearrange(mask, 'c h w -> (h w c)')

                # get the rays that is not masked
                buf += [img[mask == 0]]
                # and the pixel id of these locations (self.directions and imgs are already flattened)
                pIDs = (mask == 0).nonzero().flatten().tolist()

                if i % 100 == 0:
                    print("mask = {}/{}/{}/{}".format(mask, mask.min(), mask.max(), (mask == 0).sum()))
                    print("[test {}]: mask_name = {}, mask = {}, buf = {}, self.directions[t].shape = {}, img_wh = {}".format(t, mask_paths[t], mask.shape, buf[0].shape, self.directions[t].shape, self.img_wh[t]))

                # skip if the meta data is different from what we read from the image
                if self.img_wh[t][0] * self.img_wh[t][1] != mask.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                else:
                    self.rays += [torch.cat(buf, 1)]
                    self.rays_dir += [self.directions[t][mask == 0]]
                    self.img_ids += ([t] * buf[0].shape[0])
                    self.t_ids += ([i] * buf[0].shape[0]) 
                    self.pix_ids += pIDs

                self.ts += [t]
            self.rays = torch.cat(self.rays)
            self.rays_dir = torch.cat(self.rays_dir)
            self.img_ids = torch.tensor(self.img_ids).int()
            self.pix_ids = torch.tensor(self.pix_ids).int()
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.ts = torch.tensor(self.ts).int()
            self.t_ids = torch.tensor(self.t_ids).int()
            print("self.rays.size = {}, self.img_ids = {}, self.pix_ids = {}, self.rays_dir = {}".format(self.rays.shape, self.img_ids.shape, self.pix_ids.shape, self.rays_dir.shape))
            # exit()
        elif split == 'eval':
            # keep only the first half of the pixels per image
            for t, img_path in enumerate(tqdm(img_paths)):
                buf = [] # buffer for ray attributes: rgb, etc

                # image has already been flattened into (hw, c)
                img = read_image_phototour(img_path, blend_a=False, crop_region = 'left')
                img = torch.FloatTensor(img)

                mask = torch.load(mask_paths[t])
                # print("[before] mask.shape = {}".format(mask.shape))
                mask = mask[:, :, :mask.shape[2]//2]
                # print("[after] mask.shape = {}".format(mask.shape))
                # reshape mask
                mask = rearrange(mask, 'c h w -> (h w c)')

                # only get the first half of the data
                # mask = mask[:mask.shape[0]]
                
                # get the rays that is not masked
                buf += [img[mask == 0]]
                # and the pixel id of these locations (self.directions and imgs are already flattened)
                pIDs = (mask == 0).nonzero().flatten().tolist()

                if t % 100 == 0:
                    print("mask = {}/{}/{}/{}".format(mask, mask.min(), mask.max(), (mask == 0).sum()))
                    print("[test {}]: mask_name = {}, mask = {}, buf = {}, self.directions[t].shape = {}, img_wh = {}".format(t, mask_paths[t], mask.shape, buf[0].shape, self.directions[t].shape, self.img_wh[t]))

                # skip if the meta data is different from what we read from the image
                if ((self.img_wh[t][0]//2) * self.img_wh[t][1]) != mask.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                else:
                    rays_dir = self.directions[t]
                    self.rays += [torch.cat(buf, 1)]
                    self.rays_dir += [rays_dir[mask == 0]]
                    self.img_ids += ([t] * buf[0].shape[0])
                    self.pix_ids += pIDs

                self.ts += [-1-t]
                # if t >= 100:
                #     break
            self.rays = torch.cat(self.rays)
            self.rays_dir = torch.cat(self.rays_dir)
            self.img_ids = torch.tensor(self.img_ids).int()
            self.pix_ids = torch.tensor(self.pix_ids).int()
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.ts = torch.tensor(self.ts).int()
            print("self.rays.size = {}, self.img_ids = {}, self.pix_ids = {}, self.rays_dir = {}".format(self.rays.shape, self.img_ids.shape, self.pix_ids.shape, self.rays_dir.shape))
        else:
            self.is_valid = []
            for t, img_path in enumerate(tqdm(img_paths)):
                buf = [] # buffer for ray attributes: rgb, etc

                if self.measure_psnr:
                    img = read_image_phototour(img_path, blend_a=False, crop_region = 'right')
                    self.img_wh[t][0] = self.img_wh[t][0]-self.img_wh[t][0]//2
                    # print("self.directions[t] = {}, self.img_wh = {}".format(self.directions[t].shape, self.img_wh[t]))
                else:
                    img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)
                if self.img_wh[t][0] * self.img_wh[t][1] != img.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                    self.is_valid.append(0)
                else:
                    self.is_valid.append(1)
                # print("[test]: loading img {}, size = {}, img_wh = {}".format(img_path, img.shape, self.img_wh[t]))
                buf += [img]

                self.rays += [torch.cat(buf, 1)]
            # self.rays = torch.stack(self.rays) # (N_images, hw, ?)
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.is_valid = torch.tensor(self.is_valid).int().nonzero().flatten()
            print("is_valid = {}".format(self.is_valid.shape))
        # exit()
    
    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        elif self.split.startswith('eval'):
            return 1000
        return len(self.is_valid)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            # if self.ray_sampling_strategy == 'all_images': # randomly select images
            #     img_idxs = np.random.choice(len(self.poses), self.batch_size)
            # elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
            #     img_idxs = np.random.choice(len(self.poses), 1)[0]
            # # randomly select pixels
            # pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            # rays = torch.FloatTensor([self.rays[img_idxs[i], pix_idxs[i]] for i in range(self.batch_size)])
            # rays = torch.FloatTensor(self.rays)
            # randomly sample a batch of pixels
            ray_idxs = np.random.choice(self.rays.shape[0], self.batch_size)
            rays = torch.FloatTensor(self.rays[ray_idxs])
            img_idxs = self.img_ids[ray_idxs].long()
            pix_idxs = self.pix_ids[ray_idxs].long()
            # print("img_idxs = {}/{}/{}, pix_idxs = {}/{}/{}, self.directions[0].shape = {}".format(img_idxs.shape, img_idxs.min(), img_idxs.max(), pix_idxs.shape, pix_idxs.min(), pix_idxs.max(), self.directions[0].shape))
            # ray_dirs = []
            # for i in range(self.batch_size):
            #     ray_dirs.append()
            # ray_dirs = torch.FloatTensor([self.directions[img_idxs[i].item()][pix_idxs[i].item()] for i in range(self.batch_size)])
            # print("rays.shape = {}, img_idxs.shape = {}, pix_idxs.shape = {}, ray_dirs.shape = {}".format(rays.shape, img_idxs.shape, pix_idxs.shape, ray_dirs.shape))
            # exit()
            ray_dirs = torch.FloatTensor(self.rays_dir[ray_idxs])
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3],
                      'ts': self.ts[self.t_ids[ray_idxs].long()],
                      'ray_dirs': ray_dirs}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        elif self.split.startswith('eval'):
            ray_idxs = np.random.choice(self.rays.shape[0], self.batch_size)
            rays = torch.FloatTensor(self.rays[ray_idxs])
            img_idxs = self.img_ids[ray_idxs].long()
            pix_idxs = self.pix_ids[ray_idxs].long()

            ray_dirs = torch.FloatTensor(self.rays_dir[ray_idxs])
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3],
                      'ts': self.ts[img_idxs],
                      'ray_dirs': ray_dirs}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            idx = self.is_valid[idx]
            sample = {'pose': self.poses[idx], 'img_idxs': idx, 'ts': torch.tensor([0]).int(), 'ray_dirs': self.directions[idx], 'img_wh': self.img_wh[idx]}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample



class PhotoTourDatasetNerfw_CLNerf(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        # self.read_intrinsics()

        # in each mask, 1 is filtered pixel, 0 is a static pixel
        self.use_mask = kwargs.get('use_mask', 1) 
        self.f_train_val = kwargs.get('f_train_val', '') 
        self.measure_psnr = kwargs.get('psnr', 1)

        self.task_number = kwargs.get('task_number', 10)
        self.task_curr = kwargs.get('task_curr', 9)
        self.task_split_method = kwargs.get('task_split_method', 'seq')
        self.rep_size = kwargs.get('rep_size', 0)
        self.rep_dir = kwargs.get('rep_dir', '')
        self.nerf_rep = kwargs.get('nerf_rep', True)
        # self.img_per_appearance = kwargs.get('img_per_appearance', 10)
        
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self, img_paths, img_path_to_id, crop_region = 'full'):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/cameras.bin'))

        self.img_wh, self.K, self.directions = [], [], []
        for i in range(len(img_paths)):
            # print("img_path_to_id = {}".format(img_path_to_id.keys()))
            cam_id = img_path_to_id[os.path.basename(img_paths[i])]
            # exit()
            h = int(camdata[cam_id].height*self.downsample)
            w = int(camdata[cam_id].width*self.downsample)
            self.img_wh.append(torch.tensor([w, h]))

            if camdata[cam_id].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[cam_id].params[0]*self.downsample
                cx = camdata[cam_id].params[1]*self.downsample
                cy = camdata[cam_id].params[2]*self.downsample
            elif camdata[cam_id].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[cam_id].params[0]*self.downsample
                fy = camdata[cam_id].params[1]*self.downsample
                cx = camdata[cam_id].params[2]*self.downsample
                cy = camdata[cam_id].params[3]*self.downsample
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            self.K.append(torch.FloatTensor([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0,  0,  1]]))
            self.directions.append(get_ray_directions(h, w, self.K[i], crop_region = crop_region))
            if i % 100 == 0:
                print("cam_id[{}] = {}, img_wh = {}, self.directions.shape = {}".format(i, cam_id, self.img_wh[i], self.directions[i].shape))
        # stack K for later use
        self.K = torch.stack(self.K)
        self.img_wh = torch.stack(self.img_wh).int()

        print("self.K.shape = {}, self.img_wh.shape = {}".format(self.K.shape, self.img_wh.shape))
        # exit()
        #     print("self.directions[{}] = {}/{}".format(i, self.directions.shape, self.directions.device))
        # exit()

    def get_split(self, train_val, img_names_all, perm, split):
        # train_val: training and validation split dictionary
        # img_names_all: all images' names in the original order stored in imdata
        # per: permutation index when image names are sorted
        # output: 1. img_names: image names in the split
        #         2. selected_idx: selected idx in perm where images are in the split, used to get the corresponding poses later 

        # img_names = [imdata[k].name for k in imdata if ]
        # map to train test
        img_to_split = {}
        for i in range(len(train_val['filename'])):
            img_to_split[train_val['filename'][i]] = train_val['split'][i]
        # img_names = [train_val['filename'][i] for i in range(len(train_val['filename'])) if train_val['split'][i] == split]
        # imdata_new = {}
        img_names, selected_idx = [], []
        for i, id_perm in enumerate(perm):
            if img_names_all[id_perm] in img_to_split:
                if img_to_split[img_names_all[id_perm]] == split:
                    img_names.append(img_names_all[id_perm])
                    selected_idx.append(i)

        return img_names, selected_idx
        # for i in range(len(train_val['filename'])):



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

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/images.bin'))
        # print("imdata = {}".format(imdata.keys()))
        # exit()
        img_path_to_id = {}
        for v in imdata.values():
            img_path_to_id[v.name] = v.id
        
        # read train_val separation info
        train_val = pd.read_csv(self.f_train_val, sep='\t')
        # check whether this name exist in img_path_to_id
        # print("train_val = {}, id = {}, id_in_file = {}, len = {}".format(train_val['filename'][0], img_path_to_id[train_val['filename'][0]], train_val['id'][0], len(train_val['filename'])))
        # exit()

        img_names_all = [imdata[k].name for k in imdata]
        # get map from img path to camera id in the intrinsic database, used to construct the right intrinsics        
        perm = np.argsort(img_names_all)
        
        # get all training data
        if split == 'train' or split == 'rep':
            img_names, selected_idx = self.get_split(train_val, img_names_all, perm, 'train')
            img_paths = [os.path.join(self.root_dir, 'images', name)
                for name in sorted(img_names)]
            # get (replay) data for the current task
            random.seed(0)
            self.task_ids = self.split_tasks(img_names, self.task_number, self.task_split_method)
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

            self.id_train_final.sort()
            print("self.id_train_final = {}/{}".format(self.id_train_final, len(self.id_train_final)))
        else:
            img_names, selected_idx = self.get_split(train_val, img_names_all, perm, 'test')


        # print("img_names = {}, len = {}".format(img_names, len(img_names)))
        # exit()
        # img_names = [imdata[k].name for k in imdata]
        if '360_v2' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        
        mask_folder = 'seg'

        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        mask_paths = [os.path.join(self.root_dir, mask_folder, os.path.splitext(name)[0]+'.torchSave')
                     for name in sorted(img_names)]
        
        if split =='eval':
            self.read_intrinsics(img_paths, img_path_to_id, crop_region = 'left')
        elif split == 'test' and self.measure_psnr:
            self.read_intrinsics(img_paths, img_path_to_id, crop_region = 'right')
        else:
            self.read_intrinsics(img_paths, img_path_to_id)
        

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])

        # this part is wrong, should center poses for all data, including train and test, and then get the subset
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        # scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        scale = np.linalg.norm(self.poses[..., 3], axis=-1).max()

        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.poses = self.poses[selected_idx]
        

        self.rays, self.ts, self.img_ids, self.t_ids, self.pix_ids, self.rays_dir = [], [], [], [], [], []

        print("[after train-test split] self.K.shape = {}, self.img_wh.shape = {}, self.directions.len = {}".format(self.K.shape, self.img_wh.shape, len(self.directions)))
        print(f'Loading {len(img_paths)} {split} images ...')
        if split == 'train' :
            # for t, img_path in enumerate(tqdm(img_paths)):
            for i, t in enumerate(tqdm(self.id_train_final)):
                img_path = img_paths[t]
                buf = [] # buffer for ray attributes: rgb, etc

                # image has already been flattened into (hw, c)
                img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)

                mask = torch.load(mask_paths[t])
                # reshape mask
                mask = rearrange(mask, 'c h w -> (h w c)')

                # get the rays that is not masked
                buf += [img[mask == 0]]
                # and the pixel id of these locations (self.directions and imgs are already flattened)
                pIDs = (mask == 0).nonzero().flatten().tolist()

                if i % 100 == 0:
                    print("mask = {}/{}/{}/{}".format(mask, mask.min(), mask.max(), (mask == 0).sum()))
                    print("[test {}]: mask_name = {}, mask = {}, buf = {}, self.directions[t].shape = {}, img_wh = {}".format(t, mask_paths[t], mask.shape, buf[0].shape, self.directions[t].shape, self.img_wh[t]))

                # skip if the meta data is different from what we read from the image
                if self.img_wh[t][0] * self.img_wh[t][1] != mask.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                else:
                    self.rays += [torch.cat(buf, 1)]
                    self.rays_dir += [self.directions[t][mask == 0]]
                    self.img_ids += ([t] * buf[0].shape[0])
                    self.t_ids += ([i] * buf[0].shape[0]) 
                    self.pix_ids += pIDs

                self.ts += [t]
            self.rays = torch.cat(self.rays)
            self.rays_dir = torch.cat(self.rays_dir)
            self.img_ids = torch.tensor(self.img_ids).int()
            self.pix_ids = torch.tensor(self.pix_ids).int()
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.ts = torch.tensor(self.ts).int()
            self.t_ids = torch.tensor(self.t_ids).int()
            print("self.rays.size = {}, self.img_ids = {}, self.pix_ids = {}, self.rays_dir = {}".format(self.rays.shape, self.img_ids.shape, self.pix_ids.shape, self.rays_dir.shape))
            # exit()
        elif split == 'rep':
            self.is_valid, self.img_wh_selected = [], []
            # for t, img_path in enumerate(tqdm(img_paths)):
            self.img_paths = img_paths
            for i, t in enumerate(tqdm(self.id_train_final)):
                img_path = img_paths[t]
                buf = [] # buffer for ray attributes: rgb, etc

                img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)
                if self.img_wh[t][0] * self.img_wh[t][1] != img.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                    self.is_valid.append(0)
                else:
                    self.is_valid.append(1)
                # print("[test]: loading img {}, size = {}, img_wh = {}".format(img_path, img.shape, self.img_wh[t]))
                buf += [img]
                self.ts += [t]

                self.rays += [torch.cat(buf, 1)]
            # self.rays = torch.stack(self.rays) # (N_images, hw, ?)
            self.ts = torch.tensor(self.ts).int()
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.is_valid = torch.tensor(self.is_valid).int().nonzero().flatten()
            print("is_valid = {}".format(self.is_valid.shape))
        elif split == 'eval':
            # keep only the first half of the pixels per image
            for t, img_path in enumerate(tqdm(img_paths)):
                buf = [] # buffer for ray attributes: rgb, etc

                # image has already been flattened into (hw, c)
                img = read_image_phototour(img_path, blend_a=False, crop_region = 'left')
                img = torch.FloatTensor(img)

                mask = torch.load(mask_paths[t])
                # print("[before] mask.shape = {}".format(mask.shape))
                mask = mask[:, :, :mask.shape[2]//2]
                # print("[after] mask.shape = {}".format(mask.shape))
                # reshape mask
                mask = rearrange(mask, 'c h w -> (h w c)')

                # only get the first half of the data
                # mask = mask[:mask.shape[0]]
                
                # get the rays that is not masked
                buf += [img[mask == 0]]
                # and the pixel id of these locations (self.directions and imgs are already flattened)
                pIDs = (mask == 0).nonzero().flatten().tolist()

                if t % 100 == 0:
                    print("mask = {}/{}/{}/{}".format(mask, mask.min(), mask.max(), (mask == 0).sum()))
                    print("[test {}]: mask_name = {}, mask = {}, buf = {}, self.directions[t].shape = {}, img_wh = {}".format(t, mask_paths[t], mask.shape, buf[0].shape, self.directions[t].shape, self.img_wh[t]))

                # skip if the meta data is different from what we read from the image
                if ((self.img_wh[t][0]//2) * self.img_wh[t][1]) != mask.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                else:
                    rays_dir = self.directions[t]
                    self.rays += [torch.cat(buf, 1)]
                    self.rays_dir += [rays_dir[mask == 0]]
                    self.img_ids += ([t] * buf[0].shape[0])
                    self.pix_ids += pIDs

                self.ts += [-1-t]
                # if t >= 100:
                #     break
            self.rays = torch.cat(self.rays)
            self.rays_dir = torch.cat(self.rays_dir)
            self.img_ids = torch.tensor(self.img_ids).int()
            self.pix_ids = torch.tensor(self.pix_ids).int()
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.ts = torch.tensor(self.ts).int()
            print("self.rays.size = {}, self.img_ids = {}, self.pix_ids = {}, self.rays_dir = {}".format(self.rays.shape, self.img_ids.shape, self.pix_ids.shape, self.rays_dir.shape))
        else:
            self.is_valid = []
            for t, img_path in enumerate(tqdm(img_paths)):
                buf = [] # buffer for ray attributes: rgb, etc

                if self.measure_psnr:
                    img = read_image_phototour(img_path, blend_a=False, crop_region = 'right')
                    self.img_wh[t][0] = self.img_wh[t][0]-self.img_wh[t][0]//2
                    # print("self.directions[t] = {}, self.img_wh = {}".format(self.directions[t].shape, self.img_wh[t]))
                else:
                    img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)
                if self.img_wh[t][0] * self.img_wh[t][1] != img.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                    self.is_valid.append(0)
                else:
                    self.is_valid.append(1)
                # print("[test]: loading img {}, size = {}, img_wh = {}".format(img_path, img.shape, self.img_wh[t]))
                buf += [img]

                self.rays += [torch.cat(buf, 1)]
            # self.rays = torch.stack(self.rays) # (N_images, hw, ?)
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.is_valid = torch.tensor(self.is_valid).int().nonzero().flatten()
            print("is_valid = {}".format(self.is_valid.shape))
        # exit()
    
    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        elif self.split.startswith('eval'):
            return 1000
        return len(self.is_valid)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            # if self.ray_sampling_strategy == 'all_images': # randomly select images
            #     img_idxs = np.random.choice(len(self.poses), self.batch_size)
            # elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
            #     img_idxs = np.random.choice(len(self.poses), 1)[0]
            # # randomly select pixels
            # pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            # rays = torch.FloatTensor([self.rays[img_idxs[i], pix_idxs[i]] for i in range(self.batch_size)])
            # rays = torch.FloatTensor(self.rays)
            # randomly sample a batch of pixels
            ray_idxs = np.random.choice(self.rays.shape[0], self.batch_size)
            rays = torch.FloatTensor(self.rays[ray_idxs])
            img_idxs = self.img_ids[ray_idxs].long()
            pix_idxs = self.pix_ids[ray_idxs].long()
            # print("img_idxs = {}/{}/{}, pix_idxs = {}/{}/{}, self.directions[0].shape = {}".format(img_idxs.shape, img_idxs.min(), img_idxs.max(), pix_idxs.shape, pix_idxs.min(), pix_idxs.max(), self.directions[0].shape))
            # ray_dirs = []
            # for i in range(self.batch_size):
            #     ray_dirs.append()
            # ray_dirs = torch.FloatTensor([self.directions[img_idxs[i].item()][pix_idxs[i].item()] for i in range(self.batch_size)])
            # print("rays.shape = {}, img_idxs.shape = {}, pix_idxs.shape = {}, ray_dirs.shape = {}".format(rays.shape, img_idxs.shape, pix_idxs.shape, ray_dirs.shape))
            # exit()
            ray_dirs = torch.FloatTensor(self.rays_dir[ray_idxs])
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3],
                      'ts': self.ts[self.t_ids[ray_idxs].long()],
                      'ray_dirs': ray_dirs}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        elif self.split.startswith('eval'):
            ray_idxs = np.random.choice(self.rays.shape[0], self.batch_size)
            rays = torch.FloatTensor(self.rays[ray_idxs])
            img_idxs = self.img_ids[ray_idxs].long()
            pix_idxs = self.pix_ids[ray_idxs].long()

            ray_dirs = torch.FloatTensor(self.rays_dir[ray_idxs])
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3],
                      'ts': self.ts[img_idxs],
                      'ray_dirs': ray_dirs}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        elif self.split.startswith('rep'):
            sample = {'img_wh': self.img_wh[self.id_train_final[idx]], 'ray_dirs': self.directions[self.id_train_final[idx]], 'pose': self.poses[idx], 'ts': torch.tensor([self.ts[idx].item()]).int(), 'img_idxs': idx, 'fname': os.path.basename(self.img_paths[self.id_train_final[idx]]),'id_ori': self.id_train_final[idx],'task_id': self.task_ids[self.id_train_final[idx]]}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays
        else:
            idx = self.is_valid[idx]
            sample = {'pose': self.poses[idx], 'img_idxs': idx, 'ts': torch.tensor([0]).int(), 'ray_dirs': self.directions[idx], 'img_wh': self.img_wh[idx]}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample

