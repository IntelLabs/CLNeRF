import cv2
import numpy as np
import imageio
from PIL import ImageFont, ImageDraw, Image
import sys
from datasets import dataset_dict
import math

def get_task_ids(root_dir, task_number):
    dataset = dataset_dict['colmap_ngpa_CLNerf_render']
    kwargs = {'root_dir': root_dir,
            'downsample': 1.0,
            'task_number': task_number,
            'task_curr': task_number-1,
            }
    test_dataset = dataset(split='test', **kwargs)
    return test_dataset.task_ids_interpolate.copy()

# write code to merge multiple methods into the same video
def merge_videos_multi(videoA_paths, videoB_path, reference_video_path, output_path, textAs, textB, textC, task_ids, use_UB = 0):
    # Read input videos
    videoAs = []
    for videoA_path in videoA_paths:
        videoAs.append(cv2.VideoCapture(videoA_path))
    videoB = cv2.VideoCapture(videoB_path)
    reference_video = cv2.VideoCapture(reference_video_path)

    # Get video dimensions and calculate the output video size
    width = int(videoB.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoB.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width = width
    output_height = height

    # Create an imageio writer object to output the merged video
    out = imageio.get_writer(output_path, fps=30)

    # compute the method ID and slide location for each frame
    # saparate frames into tasks
    frames_per_task = []
    task_id_prev = -1
    for task_id in task_ids:
        if task_id != task_id_prev:
            frames_per_task.append(0)
        else:
            frames_per_task[-1] += 1
        task_id_prev = task_id
    print("frames_per_task = {}".format(frames_per_task))

    # frames per task per method
    frames_per_method = []
    for i in range(len(frames_per_task)):
        frames_per_method.append((frames_per_task[i]-100)//4)

    print("frames_per_method = {}".format(frames_per_method))

    # produce the frame_number and method id for slidings
    frame_to_slide = []
    frame_to_switch = []
    frame_curr = 0
    methods = textAs
    # methods = ['NT', 'EWC', 'ER', 'MEIL-NeRF']
    for i in range(len(frames_per_method)):
        for method in range(len(methods)):
            frame_curr += frames_per_method[i]
            frame_to_slide.append(frame_curr)
            if method < len(methods)-1:
                frame_to_switch.append(frame_curr+1)
        frame_curr += (frames_per_task[i] - frames_per_method[i]*4)
        frame_to_switch.append(frame_curr)

    print("frame_to_slide = {}, frame_to_switch = {}".format(frame_to_slide, frame_to_switch))
    # exit()
    
    # flag of the method used
    frame_curr = 0
    method_curr = 0
    id_slide = 0
    id_switch = 0
    print("frame_curr = {}, method_curr = {}".format(frame_curr, method_curr))
    # exit()

    while videoAs[0].isOpened() and videoB.isOpened() and reference_video.isOpened():
        for i in range(len(textAs)):
            if i == method_curr:
                retA, frameA = videoAs[i].read()            
            else:
                videoAs[i].read()
                
        retB, frameB = videoB.read()
        retRef, frameRef = reference_video.read()

        if not retA or not retB or not retRef:
            break
        
        task_id = task_ids.pop(0)

        if id_slide >= len(frame_to_slide) or frame_curr != frame_to_slide[id_slide]:
            slide = False
        else:
            print("frame_curr = {}, we do slide now".format(frame_curr))
            slide = True
            id_slide += 1

        # Crop and merge video frames
        left_half_A = frameA[:, :width // 2]
        right_half_B = frameB[:, width // 2:]
        merged_frame = np.hstack((left_half_A, right_half_B))

        # Add reference video to the top left corner
        if use_UB:
            ref_height, ref_width, _ = frameRef.shape
            small_ref = cv2.resize(frameRef, (ref_width // 4, ref_height // 4))
            merged_frame[:small_ref.shape[0], :small_ref.shape[1]] = small_ref

        # Calculate font size based on video width
        font_scale_A = width / 800
        font_scale_B = width / 800
        if use_UB:
            font_scale_ref = (ref_width // 4) / 1000

        # Add bounding boxes and descriptive texts
        cv2.rectangle(merged_frame, (0, 0), (width // 2, height), (0, 0, 0), 2)
        cv2.rectangle(merged_frame, (width // 2, 0), (width, height), (0, 0, 0), 2)
        if use_UB:
            cv2.rectangle(merged_frame, (0, 0), (ref_width // 4, ref_height // 4), (0, 0, 0), 2)
        
#         # Convert to PIL Image
        merged_frame_pil = Image.fromarray(cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(merged_frame_pil)

#         # Specify font styles
        fontA = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(width / 30))
        if use_UB:
            fontB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int((ref_width // 4) / 15))

        # Add bounding boxes and descriptive texts
        draw.text((10, int(height * 0.92)), textAs[method_curr], font=fontA, fill=(255,255,255))
        draw.text((int(width * 0.7), int(height * 0.92)), textB+" (t = {})".format(task_id + 1), font=fontA, fill=(255,255,255))
        if use_UB:
            draw.text((10, ref_height // 4 * 0.85), textC, font=fontB, fill=(255,255,255))
# cv2.putText(merged_frame, 'UB (reference)', (10, ref_height // 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale_ref, (255,255,255), 1, cv2.LINE_AA)
        
        # Convert back to OpenCV image (numpy array) and to BGR color scheme
        merged_frame = cv2.cvtColor(np.array(merged_frame_pil), cv2.COLOR_RGB2BGR)

        # Convert color format BGR to RGB, as imageio uses RGB
        merged_frame_rgb = cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB)

        # Write the merged frame to the output video
        out.append_data(merged_frame_rgb)

        if slide:
            # 100 frames (from middle) to the right
            # 200 frames to the left
            # 100 frames back to the middle
            step_size = math.ceil(width // 200)

            locs = []            
            loc_curr = width // 2
            for step in range(100):
                loc_curr = min(width, loc_curr + step_size)
                locs.append(loc_curr)
            for steps in range(200):
                loc_curr = max(0, loc_curr - step_size)
                locs.append(loc_curr)
            for steps in range(105):
                loc_curr = min(width // 2, loc_curr + step_size)
                locs.append(loc_curr)

            for loc_curr in locs:
                 # Crop and merge video frames
                left_half_A = frameA[:, :loc_curr]
                right_half_B = frameB[:, loc_curr:]
                merged_frame = np.hstack((left_half_A, right_half_B))

                # Add reference video to the top left corner
                if use_UB:
                    ref_height, ref_width, _ = frameRef.shape
                    small_ref = cv2.resize(frameRef, (ref_width // 4, ref_height // 4))
                    merged_frame[:small_ref.shape[0], :small_ref.shape[1]] = small_ref

                # Calculate font size based on video width
                font_scale_A = width / 800
                font_scale_B = width / 800
                if use_UB:
                    font_scale_ref = (ref_width // 4) / 1000

                # Add bounding boxes and descriptive texts
                cv2.rectangle(merged_frame, (0, 0), (loc_curr, height), (0, 0, 0), 2)
                cv2.rectangle(merged_frame, (loc_curr, 0), (width, height), (0, 0, 0), 2)
                if use_UB:
                    cv2.rectangle(merged_frame, (0, 0), (ref_width // 4, ref_height // 4), (0, 0, 0), 2)
                
        #         # Convert to PIL Image
                merged_frame_pil = Image.fromarray(cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(merged_frame_pil)

        #         # Specify font styles
                fontA = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(width / 30))
                if use_UB:
                    fontB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int((ref_width // 4) / 15))

                # Add bounding boxes and descriptive texts
                # if loc_curr > widthA * 0.3:
                draw.text((10, int(height * 0.92)), textAs[method_curr], font=fontA, fill=(255,255,255))
                draw.text((int(width * 0.7), int(height * 0.92)), textB+" (t = {})".format(task_id+1), font=fontA, fill=(255,255,255))
                if use_UB:
                    draw.text((10, ref_height // 4 * 0.85), textC, font=fontB, fill=(255,255,255))
        # cv2.putText(merged_frame, 'UB (reference)', (10, ref_height // 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale_ref, (255,255,255), 1, cv2.LINE_AA)
                
                # Convert back to OpenCV image (numpy array) and to BGR color scheme
                merged_frame = cv2.cvtColor(np.array(merged_frame_pil), cv2.COLOR_RGB2BGR)

                # Convert color format BGR to RGB, as imageio uses RGB
                merged_frame_rgb = cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB)

                # Write the merged frame to the output video
                out.append_data(merged_frame_rgb)
            
            # id_FaM += 1
            # if id_FaM == len(frame_and_method):
            #     method_curr = 0
            # else:
            #     method_curr = 

        frame_curr += 1
        if id_switch < len(frame_to_switch) and frame_curr == frame_to_switch[id_switch]:
            id_switch += 1
            method_curr = (method_curr + 1) % 4
            print("frame_curr = {}, switch to method {}".format(frame_curr, method_curr))

        # # Show the merged frame (optional)
        # cv2.imshow("Merged Video", merged_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # Release resources
    for i in range(len(videoAs)):
        videoAs[i].release()
    videoB.release()
    reference_video.release()
    out.close()

def merge_videos(videoA_path, videoB_path, reference_video_path, output_path, textA, textB, textC, task_ids, use_UB = 0):
    # Read input videos
    videoA = cv2.VideoCapture(videoA_path)
    videoB = cv2.VideoCapture(videoB_path)
    reference_video = cv2.VideoCapture(reference_video_path)

    # Get video dimensions and calculate the output video size
    widthA = int(videoA.get(cv2.CAP_PROP_FRAME_WIDTH))
    heightA = int(videoA.get(cv2.CAP_PROP_FRAME_HEIGHT))
    widthB = int(videoB.get(cv2.CAP_PROP_FRAME_WIDTH))
    heightB = int(videoB.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width = max(widthA, widthB)
    output_height = max(heightA, heightB)

    # Create an imageio writer object to output the merged video
    out = imageio.get_writer(output_path, fps=30)

    # print("num_frames = {}, task_ids_number = {}".format(int(videoA.get(cv2.CAP_PROP_FRAME_COUNT)), len(task_ids)))

    task_prev = -1
    while videoA.isOpened() and videoB.isOpened() and reference_video.isOpened():
        retA, frameA = videoA.read()
        retB, frameB = videoB.read()
        retRef, frameRef = reference_video.read()

        if not retA or not retB or not retRef:
            break
        
        task_id = task_ids.pop(0)

        # do some left right slidings once we go into a new task
        if task_id != task_prev:
            task_prev = task_id
            slide = True
        else:
            slide = False

        # Crop and merge video frames
        left_half_A = frameA[:, :widthA // 2]
        right_half_B = frameB[:, widthB // 2:]
        merged_frame = np.hstack((left_half_A, right_half_B))

        # Add reference video to the top left corner
        if use_UB:
            ref_height, ref_width, _ = frameRef.shape
            small_ref = cv2.resize(frameRef, (ref_width // 4, ref_height // 4))
            merged_frame[:small_ref.shape[0], :small_ref.shape[1]] = small_ref

        # Calculate font size based on video width
        font_scale_A = widthA / 800
        font_scale_B = widthB / 800
        if use_UB:
            font_scale_ref = (ref_width // 4) / 1000

        # Add bounding boxes and descriptive texts
        cv2.rectangle(merged_frame, (0, 0), (widthA // 2, heightA), (0, 0, 0), 2)
        cv2.rectangle(merged_frame, (widthA // 2, 0), (widthA, heightA), (0, 0, 0), 2)
        if use_UB:
            cv2.rectangle(merged_frame, (0, 0), (ref_width // 4, ref_height // 4), (0, 0, 0), 2)
        
#         # Convert to PIL Image
        merged_frame_pil = Image.fromarray(cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(merged_frame_pil)

#         # Specify font styles
        fontA = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(widthA / 30))
        if use_UB:
            fontB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int((ref_width // 4) / 15))

        # Add bounding boxes and descriptive texts
        draw.text((10, int(heightA * 0.92)), textA, font=fontA, fill=(255,255,255))
        draw.text((int(widthA * 0.7), int(heightA * 0.92)), textB+" (t = {})".format(task_id + 1), font=fontA, fill=(255,255,255))
        if use_UB:
            draw.text((10, ref_height // 4 * 0.85), textC, font=fontB, fill=(255,255,255))
# cv2.putText(merged_frame, 'UB (reference)', (10, ref_height // 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale_ref, (255,255,255), 1, cv2.LINE_AA)
        
        # Convert back to OpenCV image (numpy array) and to BGR color scheme
        merged_frame = cv2.cvtColor(np.array(merged_frame_pil), cv2.COLOR_RGB2BGR)

        # Convert color format BGR to RGB, as imageio uses RGB
        merged_frame_rgb = cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB)

        # Write the merged frame to the output video
        out.append_data(merged_frame_rgb)

        if slide:
            # 100 frames (from middle) to the right
            # 200 frames to the left
            # 100 frames back to the middle
            step_size = math.ceil(widthA // 120)

            locs = []            
            loc_curr = widthA // 2
            for step in range(60):
                loc_curr = min(widthA, loc_curr + step_size)
                locs.append(loc_curr)
            for steps in range(120):
                loc_curr = max(0, loc_curr - step_size)
                locs.append(loc_curr)
            for steps in range(65):
                loc_curr = min(widthA // 2, loc_curr + step_size)
                locs.append(loc_curr)

            for loc_curr in locs:
                 # Crop and merge video frames
                left_half_A = frameA[:, :loc_curr]
                right_half_B = frameB[:, loc_curr:]
                merged_frame = np.hstack((left_half_A, right_half_B))

                # Add reference video to the top left corner
                if use_UB:
                    ref_height, ref_width, _ = frameRef.shape
                    small_ref = cv2.resize(frameRef, (ref_width // 4, ref_height // 4))
                    merged_frame[:small_ref.shape[0], :small_ref.shape[1]] = small_ref

                # Calculate font size based on video width
                font_scale_A = widthA / 800
                font_scale_B = widthB / 800
                if use_UB:
                    font_scale_ref = (ref_width // 4) / 1000

                # Add bounding boxes and descriptive texts
                cv2.rectangle(merged_frame, (0, 0), (loc_curr, heightA), (0, 0, 0), 2)
                cv2.rectangle(merged_frame, (loc_curr, 0), (widthA, heightA), (0, 0, 0), 2)
                if use_UB:
                    cv2.rectangle(merged_frame, (0, 0), (ref_width // 4, ref_height // 4), (0, 0, 0), 2)
                
        #         # Convert to PIL Image
                merged_frame_pil = Image.fromarray(cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(merged_frame_pil)

        #         # Specify font styles
                fontA = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(widthA / 30))
                if use_UB:
                    fontB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int((ref_width // 4) / 15))

                # Add bounding boxes and descriptive texts
                # if loc_curr > widthA * 0.3:
                draw.text((10, int(heightA * 0.92)), textA, font=fontA, fill=(255,255,255))
                draw.text((int(widthA * 0.7), int(heightA * 0.92)), textB+" (t = {})".format(task_id+1), font=fontA, fill=(255,255,255))
                if use_UB:
                    draw.text((10, ref_height // 4 * 0.85), textC, font=fontB, fill=(255,255,255))
        # cv2.putText(merged_frame, 'UB (reference)', (10, ref_height // 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale_ref, (255,255,255), 1, cv2.LINE_AA)
                
                # Convert back to OpenCV image (numpy array) and to BGR color scheme
                merged_frame = cv2.cvtColor(np.array(merged_frame_pil), cv2.COLOR_RGB2BGR)

                # Convert color format BGR to RGB, as imageio uses RGB
                merged_frame_rgb = cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB)

                # Write the merged frame to the output video
                out.append_data(merged_frame_rgb)
                        

        # # Show the merged frame (optional)
        # cv2.imshow("Merged Video", merged_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # Release resources
    videoA.release()
    videoB.release()
    reference_video.release()
    out.close()
    # cv2.destroyAllWindows()

# # # Replace these paths with your own video paths
# videoA_path = sys.argv[1]
# videoB_path = sys.argv[2]
# reference_video_path = sys.argv[3]
# output_path = sys.argv[4]
# textA = sys.argv[5]
# root_dir = sys.argv[6]
# task_number = int(sys.argv[7])
# use_UB = int(sys.argv[8])

# # videoA_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/MEIL/colmap_ngpa_CLNerf_render/ninja_10_MEIL/rgb.mp4"
# # videoB_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4"
# # reference_video_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4"
# # output_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/test_videos/output_ref_bb.mp4"
# # textA = 'MEIL'
# textB = 'CLNeRF'
# textC = 'UB (reference)'

# task_ids = get_task_ids(root_dir, task_number)

# merge_videos(videoA_path, videoB_path, reference_video_path, output_path, textA, textB, textC, task_ids, use_UB)

# # Replace these paths with your own video paths
videoA_path = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
videoB_path = sys.argv[5]
reference_video_path = sys.argv[6]
output_path = sys.argv[7]
textA = [sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11]]
root_dir = sys.argv[12]
task_number = int(sys.argv[13])
use_UB = int(sys.argv[14])

# videoA_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/MEIL/colmap_ngpa_CLNerf_render/ninja_10_MEIL/rgb.mp4"
# videoB_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4"
# reference_video_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4"
# output_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/test_videos/output_ref_bb.mp4"
# textA = 'MEIL'
textB = 'CLNeRF'
textC = 'UB (reference)'

print("task_number = {}".format(task_number))
task_ids = get_task_ids(root_dir, task_number)

# print("task_ids = {}".format(task_ids))

merge_videos_multi(videoA_path, videoB_path, reference_video_path, output_path, textA, textB, textC, task_ids, use_UB)
