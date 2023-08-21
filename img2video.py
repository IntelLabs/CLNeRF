import cv2
import numpy as np
import imageio
from PIL import ImageFont, ImageDraw, Image
import sys
from datasets import dataset_dict
import math, os

def get_task_ids(root_dir, task_number):
    dataset = dataset_dict['colmap_ngpa_CLNerf_render']
    kwargs = {'root_dir': root_dir,
            'downsample': 1.0,
            'task_number': task_number,
            'task_curr': task_number-1,
            }
    test_dataset = dataset(split='test', **kwargs)
    return test_dataset.task_ids_interpolate.copy()

# The first command line argument is the folder path
folder = sys.argv[1]
root_dir = sys.argv[2]
task_number = int(sys.argv[3])

# get the number of frames from the corresponding dataset and check whether we have the same number of frames
task_ids = get_task_ids(root_dir, task_number)

# filenames = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')))
filenames = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')), key=lambda f: int(os.path.splitext(f)[0]))

if len(filenames) != len(task_ids):
    print("[error] len(fnames) = {}, task_ids = {}".format(len(filenames), len(task_ids)))
    exit()

print("filenames = {}".format(filenames[:20]))
# define properties of the output video
fps = 60  # frames per second, you can change it to your desired value
output_file = folder+'/rgb.mp4'  # output file name, you can change it

# create video from images
with imageio.get_writer(output_file, mode='I', fps=fps) as writer:
    for filename in filenames:
        print("processing {}".format(filename))
        image = imageio.imread(os.path.join(folder, filename))
        writer.append_data(image)
