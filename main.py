import deeplabcut

# Specify the path to your config.yaml file
config_path = '/local/data2/LIA_LIU/testing_pontus-testing-2024-10-18/config.yaml'

# Create the training dataset
deeplabcut.create_training_dataset(config_path)

# Start training with log output every 100 iterations and save the model every 5000 iterations
deeplabcut.train_network(config_path, maxiters=50_000, displayiters=100, saveiters=5000)

# Analyze a video with the trained model
deeplabcut.analyze_videos(config_path, ['/local/data2/LIA_LIU/testing_pontus-testing-2024-10-18/videos/squaretest.mp4'])

# deeplabcut.evaluate_network(config_path, plotting=True)

