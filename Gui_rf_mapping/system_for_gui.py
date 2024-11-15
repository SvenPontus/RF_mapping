import os

import deeplabcut

class SystemForGui:

    def __init__(self, video_path):
        self.video_path = video_path
        self.config_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/Gui_rf_mapping/hrnet_w32-conv_vid_2024_11_12_test/config.yaml"

    def yaml_path(self, yaml_path):    
        # Find config file path
        config_path = os.path.join(yaml_path, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError("Config file not found in project path.")        
        return config_path
    
    def analyze_video(self):
        deeplabcut.analyze_videos(self.config_path, save_as_csv=True, videos=self.video_path)

    def label_video(self):
        deeplabcut.create_labeled_video(self.config_path, videos=self.video_path)

""" Validation?
    def video_path_method(self, video_path):
        # Gather paths of all videos in the project path
        video_dir = os.path.join(video_path, 'videos')
        video_path = [os.path.join(video_dir, f) for f in\
                        os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        return video_path
"""    
    
