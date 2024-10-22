import deeplabcut

config_path = "/local/data2/LIA_LIU/testing_pontus-testing-2024-10-18/config.yaml"

video_path = "/local/data2/LIA_LIU/raw_test_240/240frames.mp4"
deeplabcut.analyze_videos(config_path, [video_path])
deeplabcut.create_labeled_video(config_path, video_path, draw_skeleton = True, save_frames = True, dotsize = 1)