import cv2
import os
import subprocess

def remove_audio(input_path, output_path_no_audio):
    # Use ffmpeg to remove audio from the video
    command = f"ffmpeg -i {input_path} -an -vcodec copy {output_path_no_audio}"
    subprocess.call(command, shell=True)

def convert_video(input_path, crop_x=1274, crop_y=720, target_fps=30):
    # Check if the file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return
    
    # Get the filename and directory from the path
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create a silent version of the input file
    input_path_no_audio = os.path.join(input_dir, f"{input_filename}_no_audio.mp4")
    remove_audio(input_path, input_path_no_audio)
    
    # Create the output file path
    output_path = os.path.join(input_dir, f"{input_filename}_converted.mp4")
    
    # Open the silent input video
    cap = cv2.VideoCapture(input_path_no_audio)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path_no_audio}")
        return
    
    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original Video Properties: FPS={original_fps}, Width={original_width}, Height={original_height}")

    # Calculate frame skipping ratio as an integer
    frame_skip_ratio = int(round(original_fps / target_fps))

    # Check if the original resolution is sufficient for cropping
    if original_width < crop_x or original_height < crop_y:
        print("Error: Original video resolution is smaller than crop dimensions.")
        return

    # Set codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (crop_x, crop_y))
    
    frame_count = 0
    processed_frames = 0
    
    # Calculate cropping start points to center the crop
    start_x = max((original_width - crop_x) // 2, 0)
    start_y = max((original_height - crop_y) // 2, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process only every n-th frame according to the frame skip ratio
        if frame_count % frame_skip_ratio == 0:
            # Crop and center the frame
            frame = frame[start_y:start_y + crop_y, start_x:start_x + crop_x]
        
            # Resize the frame to ensure it matches crop_x and crop_y
            resized_frame = cv2.resize(frame, (crop_x, crop_y))
        
            # Write the frame to the output video
            out.write(resized_frame)
            processed_frames += 1  # Count processed frames for tracking
        
        frame_count += 1
    
    # Finish and release all resources
    cap.release()
    out.release()
    
    print(f"Converted video saved to: {output_path}")
    print(f"Processed frames: {processed_frames} at target FPS: {target_fps}")

# Example usage:
input_video = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/untrained_videos/GH010346.MP4'
convert_video(input_video)
