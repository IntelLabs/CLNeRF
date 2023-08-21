import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGPGv2
from models.rendering_NGPA import render, MAX_SAMPLES
# from models.rendering import render_ori

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils.utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGPGv2(scale=self.hparams.scale, vocab_size=self.hparams.task_curr+1, rgb_act=rgb_act, dim_a = self.hparams.dim_a, dim_g = self.hparams.dim_g)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
            embed_id = batch['ts']
        else:
            poses = batch['pose']
            directions = self.directions
            embed_id = batch['ts'][0].to(self.device) * torch.ones(self.directions.flatten().size(0), dtype=batch['ts'].dtype, device = self.device)
            # print("embed_id = {}".format(embed_id.device))

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, embed_id, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'task_number': self.hparams.task_number,
                  'task_curr': self.hparams.task_curr,
                  'task_split_method': self.hparams.task_split_method,
                  'rep_size': self.hparams.rep_size,
                  'rep_dir': f'results/NGPGv2/CLNerf/{self.hparams.dataset_name}/{self.hparams.exp_name}/rep',
                  'nerf_rep': self.hparams.nerf_rep}

        self.test_dataset = dataset(split='test', **kwargs)
        # self.rep_dataset = dataset(split='rep', **kwargs)
        self.video_folder = f'results/video_demo/{self.hparams.render_fname}/{self.hparams.dataset_name}/{self.hparams.exp_name}_{self.hparams.render_fname}'
        os.makedirs(f'results/video_demo/{self.hparams.render_fname}/{self.hparams.dataset_name}/{self.hparams.exp_name}_{self.hparams.render_fname}', exist_ok=True)
        self.rgb_video_writer = imageio.get_writer(self.video_folder+'/rgb.mp4', fps=60)
        self.depth_video_writer = imageio.get_writer(self.video_folder+'/depth.mp4', fps=60)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.test_dataset.directions.to(self.device))
        self.register_buffer('poses', self.test_dataset.poses.to(self.device))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.rep_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        pass
        
    def training_step(self, batch, batch_nb, *args):
        pass


    def on_validation_start(self):
        torch.cuda.empty_cache()
        print("start validation")
        if not self.hparams.no_save_test:
            self.val_dir = self.video_folder
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        results = self(batch, split='test')
        w, h = self.test_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)


        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            self.rgb_video_writer.append_data(rgb_pred)
            self.depth_video_writer.append_data(depth)


    def validation_epoch_end(self, outputs):
        self.rgb_video_writer.close()
        self.depth_video_writer.close()

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_nb):
        pass

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/NGPGv2_CL/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/NGPGv2_CL/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    if hparams.task_curr != hparams.task_number - 1:
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs+1,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=DDPPlugin(find_unused_parameters=False)
                                if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=-1 if hparams.val_only else 0,
                        precision=16)
    else:  
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=DDPPlugin(find_unused_parameters=False)
                                if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=-1 if hparams.val_only else 0,
                        precision=16)

    trainer.validate(system, ckpt_path=hparams.ckpt_path)
