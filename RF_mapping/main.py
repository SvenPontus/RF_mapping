import os
import glob

import deeplabcut

from dlchelperclass import DlcHelperClass as dhc 

# For running many projects
"""
paths = [
    '',
]


base_path = ''


for path in paths:
    project_path = base_path + path.strip()  
    
    config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)
    
    deeplabcut.train_network(config_path)
    
    deeplabcut.evaluate_network(config_path, plotting=True)
    
    deeplabcut.analyze_videos(config_path, video_path)
    
    deeplabcut.create_labeled_video(config_path, videos=video_path)

    dhc.save_mean_likelihood_to_file(project_path=project_path)
    dhc.plot_loss_to_png(project_path)

"""

# Predict several videos and models
"""
paths = [
    '',
]
base_path = ''

for path in paths:
    project_path = base_path + path.strip()  
    
    config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)

    video_path = ''

    # Assuming your videos have extensions like .mp4, .avi, etc.
    video_extensions = ('*.mp4', '*.avi', '*.mkv')  # Add other extensions if needed
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_path, ext)))

    for video_file in video_files:
        
        deeplabcut.analyze_videos(config_path, video_file)
        
        deeplabcut.create_labeled_video(config_path, videos=video_file)
        
        dhc.save_mean_likelihood_to_file(project_path=project_path)
        
        dhc.plot_loss_to_png(project_path)
"""
