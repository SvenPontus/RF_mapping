import os

import deeplabcut

class SystemForGui:

    def yaml_path(self, yaml_path):
        
        # Find config file path
        config_path = os.path.join(yaml_path, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError("Config file not found in project path.")        
        return config_path
            
    def video_path(self, video_path):
        # Gather paths of all videos in the project path
        video_dir = os.path.join(video_path, 'videos')
        video_path = [os.path.join(video_dir, f) for f in\
                        os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        return video_path
    
    def analyze_video(self):
        deeplabcut.analyze_videos(config_path, videos=video_path)

    def label_video(self):
        deeplabcut.create_labeled_video(config_path, videos=video_path)