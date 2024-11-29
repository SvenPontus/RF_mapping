import numpy as np
import pandas as pd
import cv2
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PostProcessDLC:
    def __init__(self, video_path, h5_path, pickle_path):
        self.video_path = video_path
        self.h5_path = h5_path
        self.pickle_path = pickle_path
        self.segment_length = 20

        # Automatically load and process data upon initialization
        self._load_video_metadata()
        self._load_h5_data()
        self._load_pickle_data()
        self._calculate_bending_angle()
        self._synchronize_signals()

    def _load_video_metadata(self):
        # Extract the metadata from the video
        video = cv2.VideoCapture(self.video_path)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.release()
    
    def _load_pickle_data(self):
        # Load nerve signal data from the pickle file
        with open(self.pickle_path, 'rb') as f:
            self.nerve_data = pd.read_pickle(f)
        self.spikes = self.nerve_data['spikes']
        self.iff = self.nerve_data['iff']
        self.accelerometer = self.nerve_data['accelerometer']

    def _load_h5_data(self):
        # Load coordinates data from the H5 file
        # Renaming columns and remove 'scorer' level
        self.df = pd.read_hdf(self.h5_path)
        self.df.columns = [f"{bodypart}_{coord}" for bodypart,
                           coord in zip(self.df.columns.get_level_values(1),
                                        self.df.columns.get_level_values(2))]
        # Add 'realtime' column to represent timestamp in seconds
        self.df['realtime'] = np.arange(len(self.df)) / self.fps
        # Sub-DataFrame 1: df_monofil for FR, FG, and FB columns
        self.df_monofil = \
            self.df.loc[:, self.df.columns.str.startswith(('FR',
                                                            'FG',
                                                            'FB'))\
                                & ~self.df.columns.str.endswith('likelihood')]
        # Sub-DataFrame 2: df_square 
        self.df_square = \
            self.df.loc[:, self.df.columns.str.startswith(('Top_left',
                                                            'Top_right',
                                                            'Bottom_left',
                                                            'Bottom_right'))\
                                & ~self.df.columns.str.endswith('likelihood')]
        # Sub-DataFrame 3: df_likelihoods for all likelihood columns
        self.df_likelihoods = \
            self.df.loc[:, self.df.columns.str.endswith('likelihood')]

    def perform_homography(self):
        # Number of frames and points
        nframes = self.df_monofil.shape[0]
        nfilpoints = int(self.df_monofil.shape[1] / 2)  # Monofilament points (x, y per point)
        nsquarepoints = int(self.df_square.shape[1] / 2)  # Square points (x, y per point)

        # Prepare raw arrays for monofilament and square points
        monofil_raw = np.zeros((nframes, nfilpoints, 2))
        square_raw = np.zeros((nframes, nsquarepoints, 2))

        # Populate raw arrays
        for i in range(nfilpoints):
           monofil_raw[:, i, :] = self.df_monofil.iloc[:, [i*2, i*2 + 1]].to_numpy()
        for i in range(nsquarepoints):
            square_raw[:, i, :] = self.df_square.iloc[:, [i*2, i*2 + 1]].to_numpy()

        # Initialize arrays for corrected points
        monofil_corrected = np.zeros((nframes, nfilpoints, 2))
        square_corrected = np.zeros((nframes, nsquarepoints, 2))

        # Define target corners in real-world coordinates (in mm)
        target_corners = np.array([[0, 0], [0, self.segment_length_mm],
                                   [self.segment_length_mm, 0],
                                   [self.segment_length_mm,
                                    self.segment_length_mm]])

        # Apply homography on each frame
        for t in range(nframes):
            # Current frame's points
            corners = square_raw[t, :, :]
            filament = monofil_raw[t, :, :]

            # Fit homography model using RANSAC
            model_robust, _ = ransac((corners, target_corners),
                                     ProjectiveTransform,
                                     min_samples=4,
                                     residual_threshold=2,
                                     max_trials=1000)

            # Transform points
            square_corrected[t, :, :] = model_robust(corners)
            monofil_corrected[t, :, :] = model_robust(filament)

        # Store corrected points back into DataFrames
        self.df_square_corrected = pd.DataFrame(
            square_corrected.reshape(-1, nsquarepoints * 2),
            columns=[f"{col}_corr" for col in self.df_square.columns]
        )
        self.df_monofil_corrected = pd.DataFrame(
            monofil_corrected.reshape(-1, nfilpoints * 2),
            columns=[f"{col}_corr" for col in self.df_monofil.columns]
        )
        print("Homography transformation completed and stored in corrected DataFrames.")
          
    def print_likelihood_averages(self.df_likelihoods):
        # Calculate the mean for each column (body part)
        averages = self.df_likelihoods.mean()
        for body_part, avg in averages.items():
            print(f"Average likelihood for {body_part}: {avg:.4f}")


    def calculate_bending_angle(self):
        # Process bending angle for points FR2, FG2, FB1
        A = self.df_monofil[['FR2_x', 'FR2_y']].values
        B = self.df_monofil[['FG2_x', 'FG2_y']].values
        C = self.df_monofil[['FB1_x', 'FB1_y']].values

        AB = B - A
        BC = C - B
        dot_product = np.einsum('ij,ij->i', AB, BC)
        magnitude_AB = np.linalg.norm(AB, axis=1)
        magnitude_BC = np.linalg.norm(BC, axis=1)
        angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_BC))
        self.angle_degrees = np.degrees(angle_radians)

    def synchronize_signals(self):
        # Normalize and synchronize nerve signal data to match frames
        self.force_mean = self.accelerometer / 50
        self.freq_mean = self.iff / 2
        self.nerve_mean = self.spikes

        # Adjust with shift parameters as needed
        self.z1 = np.roll(self.angle_degrees, -7)
        self.z2 = np.roll(self.force_mean, 7)
        self.z3 = np.roll(self.freq_mean, 7)
        self.z4 = np.roll(self.nerve_mean, 7)

    def plot_animated_signals(self):
        # Plot animated signals and export as a GIF
        fig, ax = plt.subplots()
        
        ax.set_xlim(0, len(self.z2))
        ax.set_ylim(-10, 100)
        
        line1, = ax.plot([], [], 'r', label='Force')
        line2, = ax.plot([], [], 'b', label='Frequency')
        line3, = ax.plot([], [], 'k', label='Angle')
        line4, = ax.plot([], [], 'g', label='Spikes')
        ax.legend()