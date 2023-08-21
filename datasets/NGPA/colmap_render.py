import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from ..ray_utils import *
from ..color_utils import read_image
from ..colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from ..base import BaseDataset
import random


def name_to_task(img_paths):
    tasks, task_list, test_ids = [], [], []
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
        img_count += 1
    return task_list, test_ids



class ColmapDataset_NGPA_CLNerf_render(BaseDataset):

    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        self.task_number = kwargs.get('task_number', 5)
        # print("[test]: task_number = {}".format(self.task_number))
        self.task_curr = kwargs.get('task_curr', 4)
        self.task_split_method = kwargs.get('task_split_method', 'seq')
        self.rep_size = kwargs.get('rep_size', 0)
        self.rep_dir = kwargs.get('rep_dir', '')
        self.nerf_rep = kwargs.get('nerf_rep', True)
        self.frames_per2images = kwargs.get('frames_per2images', 100)

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(
            os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height * self.downsample)
        w = int(camdata[1].width * self.downsample)
        self.img_wh = (w, h)
        print("self.img_wh = {}".format(self.img_wh))

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.downsample
            cx = camdata[1].params[1] * self.downsample
            cy = camdata[1].params[2] * self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.downsample
            fy = camdata[1].params[1] * self.downsample
            cx = camdata[1].params[2] * self.downsample
            cy = camdata[1].params[3] * self.downsample
        else:
            raise ValueError(
                f"Please parse the intrinsics for camera model {camdata[1].model}!"
            )
        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(
            os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.root_dir and self.downsample < 1:  # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [
            os.path.join(self.root_dir, folder, name)
            for name in sorted(img_names)
        ]
        # get the task id
        task_ids, test_img_ids = name_to_task(img_paths)

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [
                np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            ]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[
            perm, :3]  # (N_images, 3, 4) cam2world matrices
        
        pts3d = read_points3d_binary(
            os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        # compute near far
        self.xyz_world = pts3d
        xyz_world_h = np.concatenate(
            [self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
        # Compute near and far bounds for each image individually
        self.nears, self.fars = {}, {}  # {id_: distance}
        for i, id_ in enumerate(imdata):
            xyz_cam_i = (xyz_world_h @ w2c_mats[i].T
                         )[:, :3]  # xyz in the ith cam coordinate
            xyz_cam_i = xyz_cam_i[
                xyz_cam_i[:,
                          2] > 0]  # filter out points that lie behind the cam
            self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
            self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

        max_far = np.fromiter(self.fars.values(), np.float32).max()
        min_near = np.fromiter(self.nears.values(), np.float32).min()

        scale = max_far / 8.0
        print("[test] near_far = {}/{}, scale = {}".format(
            min_near, max_far, scale))
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.rays = []
        self.ts = []

        # train test split
        if split == 'train' or split == 'rep':
            img_paths = [
                x for i, x in enumerate(img_paths) if i not in test_img_ids
            ]
            self.poses = np.array(
                [x for i, x in enumerate(self.poses) if i not in test_img_ids])
            self.task_ids = [
                x for i, x in enumerate(task_ids) if i not in test_img_ids
            ]
        elif split == 'test':
            img_paths = [
                x for i, x in enumerate(img_paths) if i in test_img_ids
            ]
            self.poses = np.array(
                [x for i, x in enumerate(self.poses) if i in test_img_ids])
            self.task_ids = [
                x for i, x in enumerate(task_ids) if i in test_img_ids
            ]

        # print("indices = {}".format([x for i, x in enumerate(perm) if i in test_img_ids]))
        
        # exit()
        self.img_paths = img_paths
        img_paths_with_id = list(enumerate(self.img_paths))
        sorted_filenames = sorted(img_paths_with_id, key=custom_sort_key)
        sorted_order = [index for index, _ in sorted_filenames]

        # print("sorted_order = {}".format(sorted_order))
        # exit()

        self.img_paths = [filename for _, filename in sorted_filenames]
        self.poses = self.poses[sorted_order]
        self.task_ids = [self.task_ids[i_sort] for i_sort in sorted_order]

        if split == 'train' or split == 'rep':
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
                print("task_curr = {}/{}".format(self.task_curr,
                                                 (self.task_number - 1)))
            else:
                # set random seed
                # choose randomly if we are in the first task
                dir_name = self.rep_dir
                if self.task_curr == 0:
                    rep_data = []
                else:
                    # read replay ID
                    rep_data = torch.load(
                        os.path.join(dir_name, 'rep_buf.torchSave'))
                self.id_train_final = self.id_task_curr + rep_data
                # create replay data
                for i in range(len(self.task_ids)):
                    if self.task_ids[i] == self.task_curr:
                        # reservoir sampling
                        if len(rep_data) < self.rep_size:
                            rep_data.append(i)
                            print(
                                "[test reservoir]: putting rep data {} into replay buffer of size {}"
                                .format(i,
                                        len(rep_data) - 1))
                        else:
                            id_sample = random.randint(0, i)
                            if id_sample < len(rep_data):
                                rep_data[id_sample] = i
                                print(
                                    "[test reservoir]: putting rep data {} into slot {}"
                                    .format(i, id_sample))
                os.makedirs(dir_name, exist_ok=True)
                torch.save(rep_data, os.path.join(dir_name,
                                                  'rep_buf.torchSave'))

            # replace replay images with nerf rendered version if they are not in the replay buffer
            if self.nerf_rep:
                dir_name = self.rep_dir
                for t, id_rep in enumerate(self.id_rep):
                    if id_rep not in self.id_train_final:
                        rep_name = os.path.join(
                            dir_name, os.path.basename(img_paths[id_rep]))
                        if t % 10 == 0:
                            print("changing {} to {}".format(
                                img_paths[id_rep], rep_name))
                        img_paths[id_rep] = rep_name
                self.id_train_final = self.id_task_curr + self.id_rep
            print(
                "self.id_task_curr = {}, self.rep_size = {}, id_train_final = {}"
                .format(self.id_task_curr, self.rep_size, self.id_train_final))
        else:
            self.id_train_final = list(range(len(self.poses)))
        self.id_train_final.sort()

        print("[split-{}] len(img_paths) = {}, id_train = {}".format(
            self.split, len(img_paths), self.id_train_final))
        print(f'Loading {len(img_paths)} {split} images ...')
        # for i, img_path in enumerate(tqdm(img_paths)):
        for img_id, id_train in enumerate(tqdm(self.id_train_final)):

            img_path = self.img_paths[id_train]
            buf = []  # buffer for ray attributes: rgb, etc

            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            buf += [img]

            self.rays += [torch.cat(buf, 1)]
            self.ts += [self.task_ids[id_train]]

        self.rays = torch.stack(self.rays)  # (N_images, hw, ?)

        print("id_train_final = {}".format(self.id_train_final))
        self.poses = torch.FloatTensor(
            self.poses[self.id_train_final])  # (N_images, 3, 4)
        self.ts = torch.tensor(self.ts).int()
        print("self.poses.shape = {}/{}".format(self.poses.shape, self.poses[:2]))
        
        if self.split == 'test':
            self.poses_interpolate = self.poses[0].reshape((1, 3, 4))
            self.ts_interpolate = torch.tensor([self.ts[0]])
            self.task_ids_interpolate = [self.task_ids[0]]
            for i in range(1, len(self.poses)):
                if self.task_ids[i] == self.task_ids_interpolate[-1]:
                    # produce N interpolated poses between the two poses
                    interpolated_poses = interpolate_poses_shortest(self.poses_interpolate[-1].reshape((3,4)), self.poses[i].reshape(3,4), self.frames_per2images)
                    for pose_curr in interpolated_poses:
                        self.poses_interpolate = torch.cat((self.poses_interpolate, pose_curr.reshape(1,3,4)), dim = 0)
                        self.ts_interpolate = torch.cat((self.ts_interpolate, torch.tensor([self.ts[i]])))
                        self.task_ids_interpolate.append(self.task_ids[i])
                self.poses_interpolate = torch.cat((self.poses_interpolate, self.poses[i].reshape((1,3,4))), dim=0)
                self.ts_interpolate = torch.cat((self.ts_interpolate, torch.tensor([self.ts[i]])))
                self.task_ids_interpolate.append(self.task_ids[i])
            print("self.poses_interpolate = {}, self.ts_interpolate = {}, self.task_ids_interpolate = {}".format(self.poses_interpolate.shape, self.ts_interpolate.shape, len(self.task_ids_interpolate) ))

        
    def __len__(self):
        return len(self.poses_interpolate)

    def __getitem__(self, idx):
        sample = {
            'pose': self.poses_interpolate[idx],
            'img_idxs': idx,
            'ts': torch.tensor([self.ts_interpolate[idx].item()]).int()
        }

        return sample

