import cv2
import numpy as np
import imageio
from PIL import ImageFont, ImageDraw, Image
import sys


def merge_videos(videoA_path, videoB_path, reference_video_path, output_path, textA, textB, textC):
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

    while videoA.isOpened() and videoB.isOpened() and reference_video.isOpened():
        retA, frameA = videoA.read()
        retB, frameB = videoB.read()
        retRef, frameRef = reference_video.read()

        if not retA or not retB or not retRef:
            break

        # Crop and merge video frames
        left_half_A = frameA[:, :widthA // 2]
        right_half_B = frameB[:, widthB // 2:]
        merged_frame = np.hstack((left_half_A, right_half_B))

        # Add reference video to the top left corner
        ref_height, ref_width, _ = frameRef.shape
        small_ref = cv2.resize(frameRef, (ref_width // 4, ref_height // 4))
        merged_frame[:small_ref.shape[0], :small_ref.shape[1]] = small_ref

        # Calculate font size based on video width
        font_scale_A = widthA / 800
        font_scale_B = widthB / 800
        font_scale_ref = (ref_width // 4) / 1000

        # Add bounding boxes and descriptive texts
        cv2.rectangle(merged_frame, (0, 0), (widthA // 2, heightA), (0, 0, 0), 2)
        cv2.rectangle(merged_frame, (widthA // 2, 0), (widthA, heightA), (0, 0, 0), 2)
        cv2.rectangle(merged_frame, (0, 0), (ref_width // 4, ref_height // 4), (0, 0, 0), 2)
        
#         # Convert to PIL Image
        merged_frame_pil = Image.fromarray(cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(merged_frame_pil)

#         # Specify font styles
        fontA = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(widthA / 30))
        fontB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int((ref_width // 4) / 15))

        # Add bounding boxes and descriptive texts
        draw.text((10, int(heightA * 0.92)), textA, font=fontA, fill=(255,255,255))
        draw.text((widthA * 0.85, int(heightA * 0.92)), textB, font=fontA, fill=(255,255,255))
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

# # Replace these paths with your own video paths
videoA_path = sys.argv[1]
videoB_path = sys.argv[2]
reference_video_path = sys.argv[3]
output_path = sys.argv[4]
textA = sys.argv[5]

# videoA_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/MEIL/colmap_ngpa_CLNerf_render/ninja_10_MEIL/rgb.mp4"
# videoB_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4"
# reference_video_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4"
# output_path = "/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/test_videos/output_ref_bb.mp4"
# textA = 'MEIL'
textB = 'CLNeRF'
textC = 'UB (reference)'

merge_videos(videoA_path, videoB_path, reference_video_path, output_path, textA, textB, textC)

