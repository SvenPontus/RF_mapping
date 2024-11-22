import os
import pandas as pd
import matplotlib.pyplot as plt

"""
A class for analyzing DeepLabCut (DLC) project files and extracting
relevant information, such as mean likelihood from labeled data and model loss metrics.

Methods
-------
save_mean_likelihood_to_file(project_path)
    Saves the mean likelihood for all likelihood values in each
    .h5 file within the specified project path to a text file.
    
plot_loss_to_png(project_path)
    Plots the training and validation losses from the 'learning_stats.csv' file and saves
    the plot as a PNG image.

get_config_and_video_paths(project_path)
    Returns the path to the config.yaml file and a list of video
    file paths in the project.
"""

class DlcHelperClass:
    @staticmethod
    def save_mean_likelihood_to_file(project_path):
        """
        Saves the mean likelihood for all likelihood values in each
        .h5 file within the specified project path to a text file.
        
        The method scans the project directory for .h5 files, computes the mean
        likelihood for each file, and writes the results into a text file 
        called 'avg_likelihood.txt' in the project folder.
        
        Parameters
        ----------
        project_path : str
            The path to the DLC project directory.
            
        Returns
        -------
        None
            This method creates a text file with the mean likelihood for each .h5 file.
        """
        video_path = project_path + "/videos/"

        project_name = project_path.split('/')[-1].split('-')[0]
        # Search for .h5 files in the project path
        h5_files = [f for f in os.listdir(video_path) if f.endswith('.h5')]
        
        if not h5_files:
            print("No .h5 files found in the directory.")
            return
        
        # Path to save the mean likelihood results
        output_file = os.path.join(project_path, 'avg_likelihood_sum.txt')

        # Open the file for writing
        with open(output_file, 'w') as f:
            f.write(f"{project_name}\n\nMean Likelihood for Each File:\n\n")  # Header for the file

            all_likelihood = []
            for i, file_name in enumerate(h5_files, 1):
                file_path = os.path.join(video_path, file_name)
        
                # Get video name (without "_converted")
                display_name = file_name.split('_converted')[0]

                # Read .h5 file and calculate mean likelihood
                try:
                    df = pd.read_hdf(file_path)
                    likelihood_df = df.xs('likelihood', level=2, axis=1)
                    mean_likelihood = likelihood_df.values.mean()
                    all_likelihood.append(mean_likelihood)
                    
                    # Write the mean likelihood to the file
                    f.write(f"Mean likelihood for {display_name}: {mean_likelihood:.4f}\n")
                except Exception as e:
                    f.write(f"Couldn't read file {file_name}: {e}\n")
            
            # Calculate the overall average likelihood
            if all_likelihood:         
                overall_avg = sum(all_likelihood) / len(all_likelihood)         
                f.write(f"\nOverall Average Likelihood Across All Files: {overall_avg:.4f}\n")     
            
            else: 
                f.write("\nNo likelihood data found in the files.\n")

        print(f"Mean likelihoods saved to {output_file}")

    @staticmethod
    def plot_loss_to_png(project_path):
        """
        Plots the training and validation losses from the 'learning_stats.csv' file and saves the plot as a PNG image.
        The function searches for 'learning_stats.csv' recursively within the given project directory.
        
        It extracts the training and validation loss values, then creates a plot and saves it to the project directory.
        
        Parameters
        ----------
        project_path : str
            The path to the DLC project directory.
            
        Returns
        -------
        None
            This method saves a PNG plot of the training and validation losses to the project directory.
        """
        
        # Search for 'learning_stats.csv' in the subdirectories of the project path
        stats_file = None
        for root, dirs, files in os.walk(project_path):
            if 'learning_stats.csv' in files:
                stats_file = os.path.join(root, 'learning_stats.csv')
                break
        
        # If no file is found, print a message and return
        if stats_file is None:
            print(f"learning_stats.csv not found in the directory or subdirectories of {project_path}.")
            return
        
        # Load the CSV file
        try:
            df = pd.read_csv(stats_file)
            
            # Check if the necessary columns for loss values exist
            if 'losses/train.total_loss' not in df.columns or 'losses/eval.total_loss' not in df.columns:
                print("No 'losses/train.total_loss' or 'losses/eval.total_loss' columns found in the learning_stats.csv.")             
                return

            # Extract the loss values
            epochs = df.index
            train_losses = df['losses/train.total_loss']  # Adjust column name if needed
            val_losses = df['losses/eval.total_loss']  # Adjust column name if needed

            if df['losses/eval.total_loss'].isna().any():
                # Fill NaNs with interpolation for smooth plotting (optional)
                df['losses/eval.total_loss'].interpolate(method='linear', inplace=True)
            
            # Plot the loss values
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, label='Train Loss', color='tab:red')
            plt.plot(epochs, val_losses, label='Validation Loss', color='tab:blue')
            plt.title('Training and Validation Loss Over Time')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Save the plot as a PNG image
            output_image = os.path.join(project_path, 'training_validation_loss.png')
            plt.savefig(output_image)

            # Optionally, show the plot
            plt.close()

            print(f"Training and validation loss plot saved to {output_image}")
        
        except Exception as e:
            print(f"Failed to process {stats_file}: {e}")

        # Append the last 5 loss values to 'avg_likelihood_sum.txt'
        avg_likelihood_file = os.path.join(project_path, 'avg_likelihood_sum.txt')
        if os.path.exists(avg_likelihood_file):
            with open(avg_likelihood_file, 'a') as f:
                f.write("\nLast 5 Training Losses:\n")
                f.write(", ".join([f"{x:.4f}" for x in train_losses[-5:]]) + "\n")
                f.write("Last 5 Validation Losses:\n")
                f.write(", ".join([f"{x:.4f}" for x in val_losses[-5:]]) + "\n")
            print(f"Appended loss values to {avg_likelihood_file}")
        else:
            print(f"{avg_likelihood_file} not found. Loss values not appended.")


    @staticmethod
    def get_config_and_video_paths(project_path):
        """
        Returns the path to the config.yaml file and a list of video file paths
        in the specified project directory.
        
        The method searches the project directory for the 'config.yaml' file and
        for video files ('.mp4' or '.avi' extensions) located in the 'videos' subfolder.
        
        Parameters
        ----------
        project_path : str
            The directory path to the DLC project folder.
        
        Returns
        -------
        tuple
            A tuple containing the path to the config.yaml file (str)
            and a list of paths to video files (list of str).
        
        Raises
        ------
        FileNotFoundError
            If the config.yaml file is not found in the project directory.
        """
        
        # Find config file path
        config_path = os.path.join(project_path, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError("Config file not found in project path.")
        
        # Gather paths of all videos in the project path
        video_dir = os.path.join(project_path, 'videos')
        video_path = [os.path.join(video_dir, f) for f in\
                        os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        return config_path, video_path
    
    @staticmethod
    def get_video_paths(path):
        """
        Retrieves a list of full paths for all video files in the specified directory.
        
        Args:
            videos_dir_path (str): Path to the directory containing video files.
            
        Returns:
            list: A list of full paths to video files with specified extensions, 
                  or an empty list if an error occurs or the directory is empty.
        """
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_paths = []
        
        try:
            video_paths = [
                os.path.join(path, file) 
                for file in os.listdir(path) 
                if file.lower().endswith(video_extensions)
            ]
        except FileNotFoundError:
            print(f"Error: Directory '{path}' does not exist.")
        except PermissionError:
            print(f"Error: Insufficient permissions to access '{path}'.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        return video_paths