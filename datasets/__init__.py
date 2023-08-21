from .nsvf import NSVFDataset
from .lb.nsvf import NSVFDataset_lb
from .CLNerf.nsvf import NSVFDataset_CLNerf
from .MEILNerf.nsvf_MEILNeRF import NSVFDataset_MEILNeRF
from .CLNerf.nsvf_TaTSeq import NSVFDataset_TaTSeq_CLNerf
from .MEILNerf.nsvf_TaTSeq import NSVFDataset_TaTSeq_MEIL
from .MEILNerf.nerfpp import NeRFPPDataset_MEIL

from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .lb.nerfpp import NeRFPPDataset_lb
from .CLNerf.nerfpp import NeRFPPDataset_CLNerf
from .NGPA.colmap import ColmapDataset_NGPA, ColmapDataset_NGPA_lb, ColmapDataset_NGPA_CLNerf, ColmapDataset_NGPA_MEIL
from .NGPA.colmap_render import ColmapDataset_NGPA_CLNerf_render

dataset_dict = {
    'nsvf': NSVFDataset,  # check
    'nsvf_lb': NSVFDataset_lb,
    'nsvf_CLNerf': NSVFDataset_CLNerf,  # check
    'nsvf_MEILNERF': NSVFDataset_MEILNeRF,  # check
    'nsvf_TaTSeq_CLNerf': NSVFDataset_TaTSeq_CLNerf,
    'nsvf_TaTSeq_MEILNERF': NSVFDataset_TaTSeq_MEIL,
    'colmap': ColmapDataset,
    'colmap_ngpa': ColmapDataset_NGPA,  # check
    'colmap_ngpa_lb': ColmapDataset_NGPA_lb,
    'colmap_ngpa_CLNerf': ColmapDataset_NGPA_CLNerf,  # check
    'colmap_ngpa_CLNerf_render': ColmapDataset_NGPA_CLNerf_render,  # check
    'colmap_ngpa_MEIL': ColmapDataset_NGPA_MEIL,  # check
    'nerfpp': NeRFPPDataset,  # check
    'nerfpp_lb': NeRFPPDataset_lb,  # check
    'nerfpp_CLNerf': NeRFPPDataset_CLNerf,  # check
    'nerfpp_MEIL': NeRFPPDataset_MEIL  # check
}
