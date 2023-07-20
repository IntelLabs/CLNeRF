import torch
from torch import nn
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """

    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan, ws,
                              deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts,
         rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan, ws, deltas, ts,
                                         rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):

    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results['rgb'] - target['rgb'])**2

        o = results['opacity'] + 1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity * (-o * torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d


class MEILNeRFLoss(nn.Module):

    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def forward(self, results, target, lambda_p, **kwargs):
        d = {}
        is_rep = target['is_rep']
        id_new = torch.where(is_rep == 0)[0]
        id_old = torch.where(is_rep == 1)[0]
        loss = ((results['rgb'][id_new] - target['rgb'][id_new])**
                2).sum() / float(id_new.shape[0])
        if id_old.shape[0] > 0:
            loss += torch.sum(
                torch.sqrt(
                    (results['rgb'][id_old] - target['rgb'][id_old]).pow(2) +
                    (1e-3)**2)) * lambda_p / float(id_old.shape[0])
        d['rgb'] = loss

        o = results['opacity'] + 1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity * (-o * torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d
