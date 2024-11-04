import os
import pandas as pd

class DLCProjectHelper:
    """
    A class for analyzing DeepLabCut (DLC) project files and extracting
    relevant information,
    such as mean likelihood from labeled data and model loss metrics.
    
    Methods
    -------
    show_mean_likelihood(project_path)
        Prints the mean likelihood of pose estimation from .h5 files
        within the specified project path.
        
    show_model_losses(project_path)
        Displays the model loss metrics from the training log file
        located in the specified project path.
    
    get_config_and_video_paths(project_path)
        Returns the path to the config.yaml file and a list of video
        file paths in the project.
    """
    
    @staticmethod
    def show_mean_likelihood(project_path):
        """
        Prints the mean likelihood for all likelihood values in each
        .h5 file within the specified project path.

        Parameters
        ----------
        project_path : str
            The path to the DLC project directory.
            
        Returns
        -------
        None
            This method prints the mean likelihood of each file and
            does not return any value.
        """

        # Search for .h5 files in the project path
        h5_files = [f for f in os.listdir(project_path) if f.endswith('.h5')]
        
        if not h5_files:
            print("No .h5 files found in the directory.")
            return
        
        for i, file_name in enumerate(h5_files, 1):
            file_path = os.path.join(project_path, file_name)
    
            # Get video name
            display_name = file_name.split('_converted')[0]

            # Read .h5 file and print mean likelihood
            try:
                df = pd.read_hdf(file_path)
                likelihood_df = df.xs('likelihood', level=2, axis=1)
                mean_likelihood = likelihood_df.values.mean()
                print(f"Mean likelihood for all values in {display_name}: "
                        f"{mean_likelihood:.4f}")
            except Exception as e:
                print(f"Couldn't read file path: {file_path}: {e}")
    
    @staticmethod
    def show_model_losses(project_path):
        """
        Prints the training and validation losses of the model from
        the log file in the specified project path.

        Parameters
        ----------
        project_path : str
            The directory path to the DLC project,
            where the log file is located.
            
        Returns
        -------
        None
            This method prints the loss information
            and does not return any value.
        """

        log_file_path = os.path.join(project_path, 'training', 'log.txt')
        
        if not os.path.exists(log_file_path):
            print("Log file not found in the specified path.")
            return
        
        try:
            with open(log_file_path, 'r') as log_file:
                # Print the last 10 lines to show final training results
                lines = log_file.readlines()
                print("Model Training Loss Summary (Last 10 lines):")
                for line in lines[-10:]:
                    print(line.strip())
        except Exception as e:
            print(f"Couldn't read log file: {log_file_path}: {e}")
    
    @staticmethod
    def get_config_and_video_paths(project_path):
        """
        Returns the path to the config.yaml file and
        a list of video file paths in the specified project directory.

        Parameters
        ----------
        project_path : str
            The directory path to the DLC project folder.
        
        Returns
        -------
        tuple
            A tuple containing the path to the config.yaml file (str)
            and a list of paths to video files (list of str).
        """

        # Find config file path
        config_path = os.path.join(project_path, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError("Config file not found in project path.")
        
        # Gather paths of all videos in the project path
        video_dir = os.path.join(project_path, 'videos')
        video_path = [os.path.join(video_dir, f) for f in\
                        os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        
        return config_path, video_path
