import cv2
import os
import subprocess

def remove_audio(input_path, output_path_no_audio):
    # Använd ffmpeg för att ta bort ljud från videon
    command = f"ffmpeg -i {input_path} -an -vcodec copy {output_path_no_audio}"
    subprocess.call(command, shell=True)

def convert_video(input_path, crop_x=1274, crop_y=720, target_fps=30):
    # Kontrollera om filen existerar
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return
    
    # Hämta filnamn och katalog från sökvägen
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Skapa en ljudlös version av input-filen
    input_path_no_audio = os.path.join(input_dir, f"{input_filename}_no_audio.mp4")
    remove_audio(input_path, input_path_no_audio)
    
    # Skapa output-filens sökväg
    output_path = os.path.join(input_dir, f"{input_filename}_converted.mp4")
    
    # Öppna den ljudlösa inputvideon
    cap = cv2.VideoCapture(input_path_no_audio)
    
    # Kontrollera om videon öppnades framgångsrikt
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path_no_audio}")
        return
    
    # Hämta ursprungliga videoegenskaper
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original Video Properties: FPS={original_fps}, Width={original_width}, Height={original_height}")

    # Räkna om frame skipping ratio till ett heltal
    frame_skip_ratio = int(round(original_fps / target_fps))

    # Kontrollera om originalupplösningen är tillräcklig för cropping
    if original_width < crop_x or original_height < crop_y:
        print("Error: Original video resolution is smaller than crop dimensions.")
        return

    # Sätt codec och skapa VideoWriter-objekt
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec för .mp4
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (crop_x, crop_y))
    
    frame_count = 0
    processed_frames = 0
    
    # Beräkna beskärningsstartpunkten för att centrera beskärningen
    start_x = max((original_width - crop_x) // 2, 0)
    start_y = max((original_height - crop_y) // 2, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Bearbeta endast var n:te frame enligt frame skip ratio
        if frame_count % frame_skip_ratio == 0:
            # Beskär och centrera ramen
            frame = frame[start_y:start_y + crop_y, start_x:start_x + crop_x]
        
            # Skala om ramen för att säkerställa att den matchar crop_x och crop_y
            resized_frame = cv2.resize(frame, (crop_x, crop_y))
        
            # Skriv ramen till outputvideon
            out.write(resized_frame)
            processed_frames += 1  # Räkna bearbetade ramar för uppföljning
        
        frame_count += 1
    
    # Avsluta och släpp alla resurser
    cap.release()
    out.release()
    
    print(f"Converted video saved to: {output_path}")
    print(f"Processed frames: {processed_frames} at target FPS: {target_fps}")

# Exempel på användning:
input_video = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/basil_latest_video/GH010342.MP4'
convert_video(input_video)
