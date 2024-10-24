import cv2
import os

def convert_video(input_path, crop_x=1274, crop_y=720, target_fps=30):
    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return
    
    # Get the directory and file name from the input path
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Construct the output path in the same directory
    output_path = os.path.join(input_dir, f"{input_filename}_converted.mp4")
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Check if the video is opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        return
    
    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original Video Properties: FPS={original_fps}, Width={original_width}, Height={original_height}")

    # Calculate frame skipping ratio
    frame_skip_ratio = original_fps / target_fps
    
    # Set the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (crop_x, crop_y))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only process every nth frame based on the frame skip ratio
        if frame_count % frame_skip_ratio < 1:
            # Crop the frame if necessary
            frame = frame[0:crop_y, 0:crop_x]
        
            # Resize the frame (optional, in case input resolution is not exactly 1274x720)
            resized_frame = cv2.resize(frame, (crop_x, crop_y))
        
            # Write the frame to the output video
            out.write(resized_frame)
        
        frame_count += 1
    
    # Release everything when job is finished
    cap.release()
    out.release()
    print(f"Converted video saved to: {output_path}")

# Example usage:
input_video = '/local/data2/LIA_LIU/7_vid_test-First-2024-10-23/videos/1-1.mp4'
convert_video(input_video)

