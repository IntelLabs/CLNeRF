import os
import sys
import imageio

# The first command line argument is the folder path
folder = sys.argv[1]

# filenames = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')))
filenames = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')), key=lambda f: int(os.path.splitext(f)[0]))

print("filenames = {}".format(filenames[:20]))
# define properties of the output video
fps = 30  # frames per second, you can change it to your desired value
output_file = folder+'/rgb.mp4'  # output file name, you can change it

# create video from images
with imageio.get_writer(output_file, mode='I', fps=fps) as writer:
    for filename in filenames:
        image = imageio.imread(os.path.join(folder, filename))
        writer.append_data(image)
