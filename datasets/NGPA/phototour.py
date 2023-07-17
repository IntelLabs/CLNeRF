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


class PhotoTourDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        # self.read_intrinsics()

        # in each mask, 1 is filtered pixel, 0 is a static pixel
        self.use_mask = kwargs.get('use_mask', 1) 

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self, img_paths, img_path_to_id):
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
            self.directions.append(get_ray_directions(h, w, self.K[i]))
            if i % 100 == 0:
                print("cam_id[{}] = {}, img_wh = {}, self.directions.shape = {}".format(i, cam_id, self.img_wh[i], self.directions[i].shape))
        # stack K for later use
        self.K = torch.stack(self.K)
        self.img_wh = torch.stack(self.img_wh).int()

        print("self.K.shape = {}, self.img_wh.shape = {}".format(self.K.shape, self.img_wh.shape))
        # exit()
        #     print("self.directions[{}] = {}/{}".format(i, self.directions.shape, self.directions.device))
        # exit()

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/images.bin'))
        img_path_to_id = {}
        for v in imdata.values():
            img_path_to_id[v.name] = v.id
        # self.img_ids = []
        # self.image_paths = {}  # {id: filename}
        # for filename in list(self.files['filename']):
        #     id_ = img_path_to_id[filename]
        #     self.image_paths[id_] = filename
        #     self.img_ids += [id_]

        img_names = [imdata[k].name for k in imdata]
        
        # get map from img path to camera id in the intrinsic database, used to construct the right intrinsics        


        perm = np.argsort(img_names)
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
        
        self.read_intrinsics(img_paths, img_path_to_id)


        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.rays, self.ts, self.img_ids, self.pix_ids, self.rays_dir = [], [], [], [], []
        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        if 'HDR-NeRF' in self.root_dir: # HDR-NeRF data
            if 'syndata' in self.root_dir: # synthetic
                # first 17 are test, last 18 are train
                self.unit_exposure_rgb = 0.73
                if split=='train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                            f'train/*[024].png')))
                    self.poses = np.repeat(self.poses[-18:], 3, 0)
                elif split=='test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                            f'test/*[13].png')))
                    self.poses = np.repeat(self.poses[:17], 2, 0)
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
            else: # real
                self.unit_exposure_rgb = 0.5
                # even numbers are train, odd numbers are test
                if split=='train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*0.jpg')))[::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*2.jpg')))[::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*4.jpg')))[::2]
                    self.poses = np.tile(self.poses[::2], (3, 1, 1))    
                elif split=='test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*1.jpg')))[1::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*3.jpg')))[1::2]
                    self.poses = np.tile(self.poses[1::2], (2, 1, 1))
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
        else:
            # use every 8th image as test set
            if split=='train':
                img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
                mask_paths = [x for i, x in enumerate(mask_paths) if i%8!=0]
                self.K = self.K[[i for i in range(self.K.shape[0]) if i%8!=0]]
                self.img_wh = self.img_wh[[i for i in range(self.img_wh.shape[0]) if i%8!=0]]
                self.directions = [x for i, x in enumerate(self.directions) if i%8!=0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
            elif split == 'eval':
                img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
                mask_paths = [x for i, x in enumerate(mask_paths) if i%8==0]
                self.K = self.K[[i for i in range(self.K.shape[0]) if i%8==0]]
                self.img_wh = self.img_wh[[i for i in range(self.img_wh.shape[0]) if i%8==0]]
                self.directions = [x for i, x in enumerate(self.directions) if i%8==0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])
            elif split=='test':
                img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
                mask_paths = [x for i, x in enumerate(mask_paths) if i%8==0]
                self.K = self.K[[i for i in range(self.K.shape[0]) if i%8==0]]
                self.img_wh = self.img_wh[[i for i in range(self.img_wh.shape[0]) if i%8==0]]
                self.directions = [x for i, x in enumerate(self.directions) if i%8==0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])

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
                img = read_image_phototour(img_path, blend_a=False)
                img = torch.FloatTensor(img)

                mask = torch.load(mask_paths[t])
                # reshape mask
                mask = rearrange(mask, 'c h w -> (h w c)')

                # only get the first half of the data
                img, mask = img[:img.shape[0]//2], mask[:mask.shape[0]//2]
                
                # get the rays that is not masked
                buf += [img[mask == 0]]
                # and the pixel id of these locations (self.directions and imgs are already flattened)
                pIDs = (mask == 0).nonzero().flatten().tolist()

                if t % 100 == 0:
                    print("mask = {}/{}/{}/{}".format(mask, mask.min(), mask.max(), (mask == 0).sum()))
                    print("[test {}]: mask_name = {}, mask = {}, buf = {}, self.directions[t].shape = {}, img_wh = {}".format(t, mask_paths[t], mask.shape, buf[0].shape, self.directions[t].shape, self.img_wh[t]))

                # skip if the meta data is different from what we read from the image
                if (self.img_wh[t][0] * self.img_wh[t][1])//2 != mask.shape[0]:
                    print("skip data generation for image {} (meta size = {}, actual size = {})".format(mask_paths[t], self.img_wh[t], mask.shape[0])) 
                else:
                    rays_dir = self.directions[t][:self.directions[t].shape[0]//2]
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
        if self.split.startswith('train') or self.split.startswith('eval'):
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