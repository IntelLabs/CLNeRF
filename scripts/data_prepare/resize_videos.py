import cv2
import os
from pathlib import Path

def resize_video(input_file_path, output_file_path):
    # Open the video file
    video = cv2.VideoCapture(input_file_path)

    # Check if the video file is opened successfully
    if not video.isOpened():
        print("Error opening video file")

    # Get the original video frame width and height
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the new size: half the original width and height
    new_width = int(frame_width / 2)
    new_height = int(frame_height / 2)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_file_path, fourcc, 30.0, (new_width, new_height))

    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (new_width, new_height))

            # Write the resized frame
            out.write(frame)
        else:
            break

    # Release everything when the job is finished
    video.release()
    out.release()
    cv2.destroyAllWindows()


def process_videos(input_directory_path, output_directory_path):
    video_extensions = ['.mp4', '.avi', '.MOV', '.flv', '.mkv']  # add more if needed

    # Convert to Path objects
    input_directory_path = Path(input_directory_path)
    output_directory_path = Path(output_directory_path)

    # Iterate over all files in the directory and subdirectories
    for input_file_path in input_directory_path.rglob('*'):
        if input_file_path.suffix in video_extensions:
            relative_path = input_file_path.relative_to(input_directory_path)
            output_file_path = output_directory_path / relative_path.with_suffix('.mov')

            # Create output directories if they don't exist
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            resize_video(str(input_file_path), str(output_file_path))


# Usage:
process_videos('/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/dataset/WAT/car', '/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/dataset/WAT/car_resized')
process_videos('/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/dataset/WAT/grill', '/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/dataset/WAT/grill_resized')
