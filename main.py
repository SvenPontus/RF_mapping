import deeplabcut
from dlchelperclass import DlcHelperClass as dhc 
"""
video_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/res_101_epoch_100-conv_vid-2024-11-06"

# Specify the path to your config.yaml file
config_path = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/test_10_000_epochs-conv_vid-2024-10-28/config.yaml'

# Create the training dataset
deeplabcut.create_training_dataset(config_path)

# Start training with log output every 100 iterations and save the model every 5000 iterations
# deeplabcut.train_network(config_path, maxiters=1_000_000, displayiters=100, saveiters=5000)

# Analyze a video with the trained model

deeplabcut.analyze_videos(config_path, [ 
                                        video_path + '240frames_converted.mp4',
                                        video_path + 'RF_mapping_untracked_converted.mp4',
                                        video_path + '1-1_converted.mp4',
                                        video_path + '1-2_converted.mp4',
                                        video_path + 'smalltest_converted.mp4',
                                        video_path + '2-2k7-120-hand_converted.mp4',
                                        video_path + 'squaretest_converted.mp4',
                                        '/local/data2/LIA_LIU_PONTUS/LIA_LIU/untrained_videos/GH010342_converted.mp4'
                                        ])

deeplabcut.evaluate_network(config_path, plotting=True)

# deeplabcut.analyze_videos(config_path, '/local/data2/LIA_LIU_PONTUS/LIA_LIU/untrained_videos/GH010342_converted.mp4')
                          
deeplabcut.create_labeled_video(config_path, [ 
                                        video_path + '240frames_converted.mp4',
                                        video_path + 'RF_mapping_untracked_converted.mp4',
                                        video_path + '1-1_converted.mp4',
                                        video_path + '1-2_converted.mp4',
                                        video_path + 'smalltest_converted.mp4',
                                        video_path + '2-2k7-120-hand_converted.mp4',
                                        video_path + 'squaretest_converted.mp4',
                                        '/local/data2/LIA_LIU_PONTUS/LIA_LIU/untrained_videos/GH010342_converted.mp4'
                                        ])
"""

"""

project_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/dekr_w18-conv_vid-2024-11-12"

config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)

deeplabcut.train_network(config_path)

deeplabcut.evaluate_network(config_path, plotting=True)

deeplabcut.analyze_videos(project_path, video_path, pcutoff=0.9)

deeplabcut.create_labeled_video(config_path, videos=video_path, pcutoff=0.9)

dhc.save_mean_likelihood_to_file(project_path=project_path)
dhc.plot_loss_to_png(project_path)

"""

paths = [
    "dekr_w18-conv_vid-2024-11-12",
    "dekr_w32-conv_vid-2024-11-12",
    "dekr_w48-conv_vid-2024-11-12",
    "dlcrnet_stride32_ms5-conv_vid-2024-11-12",
    "hrnet_w18-conv_vid-2024-11-12",
    "hrnet_w32-conv_vid-2024-11-12",
    "HRNet_w48-conv_vid-2024-11-12",
    "res_50-conv_vid-2024-11-12",
    "res_101-conv_vid-2024-11-12"
]


base_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/"


for path in paths:
    project_path = base_path + path.strip()  
    
    config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)
    
    deeplabcut.train_network(config_path)
    
    deeplabcut.evaluate_network(config_path, plotting=True)
    
    deeplabcut.analyze_videos(config_path, video_path)
    
    deeplabcut.create_labeled_video(config_path, videos=video_path, pcutoff=0.8)

    dhc.save_mean_likelihood_to_file(project_path=project_path)
    dhc.plot_loss_to_png(project_path)




