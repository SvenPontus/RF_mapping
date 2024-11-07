import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

# Homography
def execute_homography(square): # time x 4 x 2 (time, corner ID, xy)
    nbframes = 0
    ncorners = 0
 
    # Define target corners and segment length
    segment_length_mm = 20
    target_corners = {
        'x': [0, 0, segment_length_mm, segment_length_mm],
        'y': [0, segment_length_mm, 0, segment_length_mm]
    }
    target_corners_xy = np.column_stack((target_corners['x'], target_corners['y']))
 
    # Initialize corrected arrays
    square_corrected = np.zeros((nbframes, ncorners, 2))
    # Perform homography transformation on each frame
    for t in range(nbframes):
        print(f"frame = {t + 1}/{nbframes}")
        corners = square['raw'][t, :, :]
 
        # Fit projective transform (homography)
        model_robust, _ = ransac((corners, target_corners_xy), ProjectiveTransform, min_samples=4, residual_threshold=2, max_trials=1000)
 
        # Transform points
        square_corrected[t, :, :] = model_robust(corners)
 
    print("done.")
 
    return square_corrected
 
 
def plot_square(square, square_corrected): # time x 4 x 2 (time, corner ID, xy)
    # Animation loop for tracking points
    for t in range(nbframes):
        plt.figure(figsize=(12, 6))
       
        # Raw data subplot
        plt.subplot(1, 2, 1)
        plt.title(f"Frame {t + 1}/{nbframes}")
        for p in range(trackingpoints['nfilpoints']):
            plt.plot(monofilament['raw'][t, p, 0], monofilament['raw'][t, p, 1], '.', markersize=10)
        for p in range(trackingpoints['nsquarepoints']):
            plt.plot(square['raw'][t, p, 0], square['raw'][t, p, 1], 'x', markersize=10)
        plt.xlim([xmin * 0.9, xmax * 1.1])
        plt.ylim([ymin * 0.9, ymax * 1.1])
 
        # Corrected data subplot
        plt.subplot(1, 2, 2)
        plt.plot(square_corrected[t, :, 0], square_corrected[t, :, 1], 'rx', markersize=10)
        plt.plot(square_corrected[t, [0, 1], 0], square_corrected[t, [0, 1], 1], '-', color="#EDB120", linewidth=1)
        plt.plot(square_corrected[t, [0, 2], 0], square_corrected[t, [0, 2], 1], '-', color="#EDB120", linewidth=1)
        plt.plot(square_corrected[t, [2, 3], 0], square_corrected[t, [2, 3], 1], '-', color="#EDB120", linewidth=1)
        plt.plot(square_corrected[t, [3, 1], 0], square_corrected[t, [3, 1], 1], '-', color="#EDB120", linewidth=1)