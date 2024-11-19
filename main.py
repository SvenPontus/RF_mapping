import deeplabcut
from dlchelperclass import DlcHelperClass as dhc 

# For running many projects

paths = [
    "dlcrnet_stride32_ms5_2-conv_vid-2024-11-18",
    "hrnet_w32_2-conv_vid-2024-11-18",
    "hrnet_w48_2-conv_vid-2024-11-18"
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


"""
project_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/hrnet_w48_2-conv_vid-2024-11-18"

config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)
    
deeplabcut.train_network(config_path)

deeplabcut.evaluate_network(config_path, plotting=True)

deeplabcut.analyze_videos(config_path, video_path)

deeplabcut.create_labeled_video(config_path, videos=video_path)

dhc.save_mean_likelihood_to_file(project_path=project_path)
dhc.plot_loss_to_png(project_path)
"""