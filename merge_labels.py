import deeplabcut

# video_path = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/test_10_000_epochs-conv_vid-2024-10-28/videos/' 

# Specify the path to your config.yaml file
config_path = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/test_10_000_epochs-conv_vid-2024-10-28/config.yaml'

collected_data_h5 = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/test_10_000_epochs-conv_vid-2024-10-28/labeled-data/squaretest_converted/CollectedData_conv_vid.h5'

deeplabcut.merge_datasets(config=config_path)