import deeplabcut

# Specify the path to your config.yaml file
config_path = '/local/data2/LIA_LIU_PONTUS/LIA_LIU/res_50_test-conv_vid-2024-11-04/config.yaml'

deeplabcut.merge_datasets(config=config_path)