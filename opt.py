import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir',
                        type=str,
                        required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='nsvf',
                        choices=[
                            'nsvf', 'nsvf_lb', 'nsvf_CLNerf', 'nsvf_MEILNERF',
                            'nsvf_TaTSeq_CLNerf', 'nsvf_TaTSeq_MEILNERF',
                            'colmap', 'colmap_ngpa', 'colmap_ngpa_CLNerf',
                            'colmap_ngpa_MEIL', 'colmap_ngpa_lb', 'nerfpp',
                            'nerfpp_lb', 'nerfpp_CLNerf', 'nerfpp_MEIL', 'colmap_ngpa_CLNerf_render'
                        ],
                        help='which dataset to train/test')
    parser.add_argument('--split',
                        type=str,
                        default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample',
                        type=float,
                        default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure',
                        action='store_true',
                        default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w',
                        type=float,
                        default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size',
                        type=int,
                        default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy',
                        type=str,
                        default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # experimental training options
    parser.add_argument('--optimize_ext',
                        action='store_true',
                        default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument(
        '--random_bg',
        action='store_true',
        default=False,
        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips',
                        action='store_true',
                        default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only',
                        action='store_true',
                        default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test',
                        action='store_true',
                        default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name',
                        type=str,
                        default='exp',
                        help='experiment name')
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument(
        '--weight_path',
        type=str,
        default=None,
        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # CL params
    parser.add_argument('--task_number',
                        type=int,
                        default=5,
                        help='task_number')
    parser.add_argument('--task_curr',
                        type=int,
                        default=4,
                        help='task_number [0, N-1]')
    parser.add_argument('--task_split_method',
                        type=str,
                        default='seq',
                        help='seq or random')
    parser.add_argument('--rep_size',
                        type=int,
                        default=0,
                        help='0 to number of images')
    parser.add_argument('--nerf_rep',
                        type=bool,
                        default=True,
                        help='whether to use nerf replay')

    # NGPA param
    parser.add_argument('--vocab_size',
                        type=int,
                        default=10,
                        help='number of embeddings')
    parser.add_argument('--dim_a',
                        type=int,
                        default=48,
                        help='dimension of embeddings')
    parser.add_argument('--dim_g',
                        type=int,
                        default=16,
                        help='dimension of geometry embeddings')

    # phototour param
    parser.add_argument('--num_epochs_eval',
                        type=int,
                        default=5,
                        help='number of training epochs')
    parser.add_argument(
        '--weight_path_eval',
        type=str,
        default=None,
        help='pretrained checkpoint to load (excluding optimizers, etc)')
    # nerfw train test separation
    parser.add_argument('--f_train_val',
                        type=str,
                        default=None,
                        help='training and validation separation file')
    parser.add_argument('--psnr',
                        type=int,
                        default=1,
                        help='whether to measure psnr in ngpa')
    parser.add_argument('--prune_threshold',
                        type=float,
                        default=0.01,
                        help='pruning threshold')
    parser.add_argument('--log2T',
                        type=int,
                        default=19,
                        help='hash table size')
    parser.add_argument('--N_neuron_d',
                        type=int,
                        default=64,
                        help='number of neurons for the depth mlp')
    parser.add_argument('--N_neuron_c',
                        type=int,
                        default=64,
                        help='number of neurons for the color mlp')
    parser.add_argument('--N_layers_d',
                        type=int,
                        default=1,
                        help='number of layers for the depth mlp')
    parser.add_argument('--N_layers_c',
                        type=int,
                        default=2,
                        help='number of layers for the color mlp')

    # warmup
    parser.add_argument('--warmup',
                        type=int,
                        default=0,
                        help='whether to use warmup')
    parser.add_argument('--lr_min',
                        type=float,
                        default=0.033,
                        help='minimum learning rate')
    parser.add_argument('--render_fname',
                        type=str,
                        default='CLNeRF',
                        help='the extra name used when render video demos')

    return parser.parse_args()
