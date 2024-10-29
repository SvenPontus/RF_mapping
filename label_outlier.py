import deeplabcut

video_path = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/test_10_000_epochs-conv_vid-2024-10-28/videos/' 

config_path = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/test_10_000_epochs-conv_vid-2024-10-28/config.yaml'

deeplabcut.extract_outlier_frames(config=config_path,
                                   shuffle=1,
                                  automatic=False,
                                  videos=[ 
                                        video_path + '240frames_converted.mp4',
                                        video_path + 'RF_mapping_untracked_converted.mp4',
                                        video_path + '1-1_converted.mp4',
                                        video_path + '1-2_converted.mp4',
                                        video_path + 'smalltest_converted.mp4',
                                        video_path + '2-2k7-120-hand_converted.mp4',
                                        video_path + 'squaretest_converted.mp4',
                                        '/local/data2/LIA_LIU_PONTUS/LIA_LIU/untrained_videos/GH010342_converted.mp4'
                                        ]
                                  )


#deeplabcut.label_frames(config_path=config_path)
