import deeplabcut
from dlchelperclass import DlcHelperClass as dhc

# List of all paths
paths = [
    "dekr_w18-conv_vid-2024-11-12/",
    "dekr_w32-conv_vid-2024-11-12/",
    "dekr_w48-conv_vid-2024-11-12/",
    "dlcrnet_stride32_ms5-conv_vid-2024-11-12/",
    "hrnet_w18-conv_vid-2024-11-12/",
    "hrnet_w32-conv_vid-2024-11-12/",
    "HRNet_w48-conv_vid-2024-11-12/",
    "res_50-conv_vid-2024-11-12",
    "res_101-conv_vid-2024-11-12"
]

# Base path for the projects
base_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/"

# Loop through each project
for path in paths:
    project_path = base_path + path.strip()  # Remove any extra whitespace
    
    # Retrieve config and video paths
    config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)
    
    # Parameters to vary during network training
    training_params = {
        "learning_rate": [0.001, 0.0001],  # Different learning rates
        "batch_size": [8, 16, 32],         # Different batch sizes
        "num_iterations": [50000, 100000]  # Different numbers of iterations
    }
    
    # Test various training configurations
    for lr in training_params["learning_rate"]:
        for batch in training_params["batch_size"]:
            for iters in training_params["num_iterations"]:
                # Configure network parameters if supported by deeplabcut
                deeplabcut.train_network(config_path, learning_rate=lr, batch_size=batch, num_iterations=iters)
    
    # Parameters for network evaluation
    eval_plot = True
    eval_shuffle = 1  # Test different shuffle values if relevant
    
    # Evaluate the network with varying parameters
    deeplabcut.evaluate_network(config_path, plotting=eval_plot, shuffle=eval_shuffle)
    
    # Parameters for video analysis
    analyze_params = {
        "pcutoff": [0.8, 0.9],  # Change probability cutoff
        "save_as_csv": [True, False]  # Save results as CSV or not
    }
    
    # Analyze video with different probability cutoffs and save formats
    for cutoff in analyze_params["pcutoff"]:
        for save_csv in analyze_params["save_as_csv"]:
            deeplabcut.analyze_videos(project_path, video_path, pcutoff=cutoff, save_as_csv=save_csv)
    
    # Parameters for creating labeled video
    create_video_params = {
        "pcutoff": [0.8, 0.9],
        "displaybodyparts": [True, False]  # Show or hide body parts in the labeled video
    }
    
    # Create labeled video with different settings
    for cutoff in create_video_params["pcutoff"]:
        for display_parts in create_video_params["displaybodyparts"]:
            deeplabcut.create_labeled_video(config_path, videos=video_path, pcutoff=cutoff, displaybodyparts=display_parts)
    
    # Additional parameters for dhc functions
    likelihood_output_path = project_path + "/likelihood_data.csv"  # Specify unique file path if desired
    
    # Save mean likelihood to file
    dhc.save_mean_likelihood_to_file(project_path=project_path, output_path=likelihood_output_path)
    
    # Parameters for plot_loss_to_png
    plot_params = {
        "figsize": (10, 6),  # Figure size
        "dpi": 300           # Plot resolution
    }
    
    # Generate loss plot with specific size and resolution
    dhc.plot_loss_to_png(project_path, figsize=plot_params["figsize"], dpi=plot_params["dpi"])


# minimize version

"""import deeplabcut
from dlchelperclass import DlcHelperClass as dhc

# List of all paths
paths = [
    "dekr_w18-conv_vid-2024-11-12/",
    "dekr_w32-conv_vid-2024-11-12/",
    "dekr_w48-conv_vid-2024-11-12/",
    "dlcrnet_stride32_ms5-conv_vid-2024-11-12/",
    "hrnet_w18-conv_vid-2024-11-12/",
    "hrnet_w32-conv_vid-2024-11-12/",
    "HRNet_w48-conv_vid-2024-11-12/",
    "res_50-conv_vid-2024-11-12",
    "res_101-conv_vid-2024-11-12"
]

# Base path for the projects
base_path = "/local/data2/LIA_LIU_PONTUS/LIA_LIU/"

# Loop through each project
for path in paths:
    project_path = base_path + path.strip()  # Remove any extra whitespace
    
    # Retrieve config and video paths
    config_path, video_path = dhc.get_config_and_video_paths(project_path=project_path)
    
    # Important training parameters (reduced)
    training_params = {
        "learning_rate": [0.001, 0.0001],  # Key learning rates
        "num_iterations": [50000, 100000]  # Key iteration counts
    }
    
    # Test reduced training configurations
    for lr in training_params["learning_rate"]:
        for iters in training_params["num_iterations"]:
            # Train network with minimized parameter combinations
            deeplabcut.train_network(config_path, learning_rate=lr, num_iterations=iters)
    
    # Key video analysis parameters
    analyze_pcutoff = [0.8, 0.9]  # Critical probability cutoffs

    # Analyze video with different probability cutoffs
    for cutoff in analyze_pcutoff:
        deeplabcut.analyze_videos(project_path, video_path, pcutoff=cutoff, save_as_csv=True)
    
    # Key labeled video creation parameters
    display_bodyparts_options = [True, False]  # Show or hide body parts

    # Create labeled video with display option variations
    for display_parts in display_bodyparts_options:
        deeplabcut.create_labeled_video(config_path, videos=video_path, pcutoff=cutoff, displaybodyparts=display_parts)
    
    # Additional processing for dhc functions with fixed plotting parameters
    likelihood_output_path = project_path + "/likelihood_data.csv"
    dhc.save_mean_likelihood_to_file(project_path=project_path, output_path=likelihood_output_path)
    
    # Generate loss plot with fixed size and resolution
    dhc.plot_loss_to_png(project_path, figsize=(10, 6), dpi=300)
"""