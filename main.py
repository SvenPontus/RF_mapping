import deeplabcut
from dlchelperclass import DlcHelperClass as dhc 

# For running many projects
"""
paths = [
    "one_vid_hrnet48-liu-2024-11-25",
    "one_vid_res50-liu-2024-11-25",
    "one_vid_stride32-liu-2024-11-25"

]


base_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/"


for path in paths:
    project_path = base_path + path.strip()  
    
    config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)
    
    deeplabcut.train_network(config_path)
    
    deeplabcut.evaluate_network(config_path, plotting=True)
    
    deeplabcut.analyze_videos(config_path, video_path)
    
    deeplabcut.create_labeled_video(config_path, videos=video_path)

    dhc.save_mean_likelihood_to_file(project_path=project_path)
    dhc.plot_loss_to_png(project_path)


    



    
project_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/hrnet_w48_2-conv_vid-2024-11-18"

config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)
    
deeplabcut.train_network(config_path)

deeplabcut.evaluate_network(config_path, plotting=True)

deeplabcut.analyze_videos(config_path, video_path)

deeplabcut.create_labeled_video(config_path, videos=video_path)

dhc.save_mean_likelihood_to_file(project_path=project_path)
dhc.plot_loss_to_png(project_path)
"""


import os
import glob

# Predict several videos and models

paths = [
    "hrnet_w48_2-conv_vid-2024-11-18",
]
base_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/"

for path in paths:
    project_path = base_path + path.strip()  
    
    config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)

    video_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/untrained_videos/Predict_unseen_6_vid_hrnet_w48_2_2024_11_18"

    # Assuming your videos have extensions like .mp4, .avi, etc.
    video_extensions = ('*.mp4', '*.avi', '*.mkv')  # Add other extensions if needed
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_path, ext)))

    for video_file in video_files:
        
        deeplabcut.analyze_videos(config_path, video_file)
        
        deeplabcut.create_labeled_video(config_path, videos=video_file)

