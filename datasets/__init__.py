from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .nsvf_MEILNERF import NSVFDataset_MEILNERF
from .nsvf_MEILNERF_paper import NSVFDataset_MEILNERF_paper
from .lb.nsvf import NSVFDataset_lb
from .lb.nsvf_MEILNERF import NSVFDataset_lb_MEILNERF
from .lb.nsvf_MEILNERF_paper import NSVFDataset_lb_MEILNERF_paper
from .CLNerf.nsvf import NSVFDataset_CLNerf
from .CLNerf.nsvf_MEILNERF import NSVFDataset_CLNerf_MEILNERF
from .CLNerf.nsvf_MEILNERF_paper import NSVFDataset_CLNerf_MEILNERF_paper
from .MEILNerf.nsvf_MEILNeRF import NSVFDataset_MEILNeRF
from .CLNerf.nsvf_TaTSeq import NSVFDataset_TaTSeq_CLNerf
from .MEILNerf.nsvf_TaTSeq import NSVFDataset_TaTSeq_MEIL
from .MEILNerf.nsvf_TaTSeq100 import NSVFDataset_TaTSeq100_MEIL
from .MEILNerf.nerfpp import NeRFPPDataset_MEIL
from .MEILNerf.nsvf_MEILNERF_paper import NSVFDataset_MEIL_MEILNERF_paper

from .NGPA.nsvf import NSVFDataset_NGPA, NSVFDataset_NGPA_lb, NSVFDataset_NGPA_CLNerf
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .lb.nerfpp import NeRFPPDataset_lb
from .CLNerf.nerfpp import NeRFPPDataset_CLNerf
from .rtmv import RTMVDataset
from .NGPA.phototour import PhotoTourDataset
from .NGPA.phototour_nerfw import PhotoTourDatasetNerfw, PhotoTourDatasetNerfw_lb, PhotoTourDatasetNerfw_CLNerf
from .NGPA.phototour_nerfw_minScale import PhotoTourDatasetNerfwMS, PhotoTourDatasetNerfwMS_lb, PhotoTourDatasetNerfwMS_CLNerf
from .MEILNerf.phototour_nerfw_minScale import PhotoTourDatasetNerfwMS_MEIL
from .NGPA.colmap import ColmapDataset_NGPA, ColmapDataset_NGPA_lb, ColmapDataset_NGPA_CLNerf, ColmapDataset_NGPA_MEIL 



dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'nsvf_MEILNERF': NSVFDataset_MEILNERF,
                'nsvf_MEILNERF_paper': NSVFDataset_MEILNERF_paper,
                'nsvf_lb': NSVFDataset_lb,
                'nsvf_lb_MEILNERF': NSVFDataset_lb_MEILNERF,
                'nsvf_lb_MEILNERF_paper': NSVFDataset_lb_MEILNERF_paper,
                'nsvf_CLNerf': NSVFDataset_CLNerf,
                'nsvf_CLNerf_MEILNERF': NSVFDataset_CLNerf_MEILNERF,
                'nsvf_CLNerf_MEILNERF_paper': NSVFDataset_CLNerf_MEILNERF_paper,
                'nsvf_MEILNERF': NSVFDataset_MEILNeRF,
                'nsvf_TaTSeq_CLNerf': NSVFDataset_TaTSeq_CLNerf,
                'nsvf_TaTSeq_MEILNERF': NSVFDataset_TaTSeq_MEIL,
                'nsvf_TaTSeq100_MEILNERF': NSVFDataset_TaTSeq100_MEIL,
                'nsvf_MEIL_MEILNERF_paper': NSVFDataset_MEIL_MEILNERF_paper,
                'nsvf_ngpa': NSVFDataset_NGPA,
                'nsvf_ngpa_lb': NSVFDataset_NGPA_lb,
                'nsvf_ngpa_CLNerf': NSVFDataset_NGPA_CLNerf,
                'phototour': PhotoTourDataset,
                'phototour_nerfw': PhotoTourDatasetNerfw,
                'phototour_nerfw_lb': PhotoTourDatasetNerfw_lb,
                'phototour_nerfw_CLNerf': PhotoTourDatasetNerfw_CLNerf,
                'phototour_nerfwMS': PhotoTourDatasetNerfwMS,
                'phototour_nerfwMS_lb': PhotoTourDatasetNerfwMS_lb,
                'phototour_nerfwMS_CLNerf': PhotoTourDatasetNerfwMS_CLNerf,
                'phototour_nerfwMS_MEIL': PhotoTourDatasetNerfwMS_MEIL,
                'colmap': ColmapDataset,
                'colmap_ngpa': ColmapDataset_NGPA,
                'colmap_ngpa_lb': ColmapDataset_NGPA_lb,
                'colmap_ngpa_CLNerf': ColmapDataset_NGPA_CLNerf,
                'colmap_ngpa_MEIL': ColmapDataset_NGPA_MEIL,
                'nerfpp': NeRFPPDataset,
                'nerfpp_lb': NeRFPPDataset_lb,
                'nerfpp_CLNerf': NeRFPPDataset_CLNerf,
                'nerfpp_MEIL': NeRFPPDataset_MEIL,
                'rtmv': RTMVDataset}