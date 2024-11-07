import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
import os


# Initializing relative to libraries and PC

# Jump to the current directory
os.chdir(os.path.dirname(__file__))

# Add paths (equivalent in Python could be handled by setting up modules or importing directly if Python files are in these directories)

# I.0 Load data
# Load the raw signal
winZoom = pd.read_pickle('Unit_27_Zoom_101_ST14-01.pkl')  # Assuming data is saved as a pickle; MATLAB files can also be read with scipy.io
winZoom_data = winZoom['FullPeriod_D']['ContD']['D']
winZoom_metadata = {k: v for k, v in winZoom.items() if k != 'FullPeriod_D'}

# Mapping relevant data
mng = {
    'spikes': winZoom_data['Nervespike1'],
    'iff': winZoom_data['Freq'],
    'accelerometer': winZoom_data['Force'],
    'time_sec': winZoom_data['Sec_FromStart'][:, 0]
}

# Load and prepare tracking points from CSV
trackingpoints = {}
trackingpoints['locs_raw'] = pd.read_csv('38_38.csv').iloc[:, 1:]  # Skip the first column (assumes Var1 is the first column)
trackingpoints['frames'] = trackingpoints['locs_raw'].iloc[:, 0]  # Frames are the first column in MATLAB

# Redefine variable names
new_variable_names = []
point_id = 0

for p in range(0, trackingpoints['locs_raw'].shape[1], 3):
    point_id += 1
    new_variable_names += [f'P{point_id}_x', f'P{point_id}_y', f'ND{point_id}']

point_id = 0
for p in range(18, trackingpoints['locs_raw'].shape[1], 3):  # Starting at 19th position in MATLAB (18 in zero-indexed Python)
    point_id += 1
    new_variable_names += [f'square_c{point_id}_x', f'square_c{point_id}_y', f'ND{point_id + 10}']

trackingpoints['locs_raw'].columns = new_variable_names

print("done.")

# I.1 Split track data into filament and square and correct when needed

# Delete columns with 'ND' in their names
trackingpoints['locs'] = trackingpoints['locs_raw'].loc[:, ~trackingpoints['locs_raw'].columns.str.contains('ND')]

# Invert the Y-axis of the monofilament tracking system
headers_Y = trackingpoints['locs'].filter(regex='_y').columns
data_Y = trackingpoints['locs'][headers_Y]
max_Y = data_Y.to_numpy().max()
data_Y_inverted = abs(data_Y - max_Y)
trackingpoints['locs'][headers_Y] = data_Y_inverted

# Splitting tracked data into filament and square tracking points
trackingpoints.update({
    'nfilpoints': 6,
    'nsquarepoints': 4,
    'nframes': trackingpoints['frames'].shape[0]
})

# Monofilament
monofilament = {
    'raw': np.zeros((trackingpoints['nframes'], trackingpoints['nfilpoints'], 2))
}
point_id = 0
for p in range(0, trackingpoints['nfilpoints'] * 2, 2):
    monofilament['raw'][:, point_id, :] = trackingpoints['locs'].iloc[:, [p, p+1]].to_numpy()
    point_id += 1

# Square around the RF
square = {
    'raw': np.zeros((trackingpoints['nframes'], trackingpoints['nsquarepoints'], 2))
}
point_id = 0
for p in range(trackingpoints['nfilpoints'] * 2, trackingpoints['locs'].shape[1], 2):
    square['raw'][:, point_id, :] = trackingpoints['locs'].iloc[:, [p, p+1]].to_numpy()
    point_id += 1

print("done.")


# Homography
# Initialize corrected arrays
monofilament['corrected'] = np.zeros((trackingpoints['nframes'], trackingpoints['nfilpoints'], 2))
square['corrected'] = np.zeros((trackingpoints['nframes'], trackingpoints['nsquarepoints'], 2))

# Define target corners and segment length
segment_length_mm = 20
target_corners = {
    'x': [0, 0, segment_length_mm, segment_length_mm],
    'y': [0, segment_length_mm, 0, segment_length_mm]
}
target_corners_xy = np.column_stack((target_corners['x'], target_corners['y']))

# Perform homography transformation on each frame
for t in range(trackingpoints['nframes']):
    print(f"frame = {t + 1}/{trackingpoints['nframes']}")
    corners = square['raw'][t, :, :]
    filament = monofilament['raw'][t, :, :]

    # Fit projective transform (homography)
    model_robust, _ = ransac((corners, target_corners_xy), ProjectiveTransform, min_samples=4, residual_threshold=2, max_trials=1000)

    # Transform points
    square['corrected'][t, :, :] = model_robust(corners)
    monofilament['corrected'][t, :, :] = model_robust(filament)

print("done.")

# Display data (I.2)
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("Raw data from winZoomSC")

# Plot spikes
axs[0].plot(mng['time_sec'], mng['spikes'], 'r')
axs[0].set_title("Action potentials")

# Plot instantaneous firing frequency (iff)
axs[1].plot(mng['time_sec'], mng['iff'], 'g')
axs[1].set_title("Instantaneous Firing Frequency")

# Plot accelerometer data
axs[2].plot(mng['time_sec'], mng['accelerometer'], 'b')
axs[2].set_title("Accelerometer (V)")
plt.tight_layout()
plt.show()

# Determine min and max for x and y axes
x_values = trackingpoints['locs'].iloc[:, ::2].values  # Odd columns for x
y_values = trackingpoints['locs'].iloc[:, 1::2].values  # Even columns for y
xmin, xmax = x_values.min(), x_values.max()
ymin, ymax = y_values.min(), y_values.max()

# Animation loop for tracking points
for t in range(trackingpoints['nframes']):
    plt.figure(figsize=(12, 6))
    
    # Raw data subplot
    plt.subplot(1, 2, 1)
    plt.title(f"Frame {t + 1}/{trackingpoints['nframes']}")
    for p in range(trackingpoints['nfilpoints']):
        plt.plot(monofilament['raw'][t, p, 0], monofilament['raw'][t, p, 1], '.', markersize=10)
    for p in range(trackingpoints['nsquarepoints']):
        plt.plot(square['raw'][t, p, 0], square['raw'][t, p, 1], 'x', markersize=10)
    plt.xlim([xmin * 0.9, xmax * 1.1])
    plt.ylim([ymin * 0.9, ymax * 1.1])

    # Corrected data subplot
    plt.subplot(1, 2, 2)
    plt.plot(square['corrected'][t, :, 0], square['corrected'][t, :, 1], 'rx', markersize=10)
    plt.plot(square['corrected'][t, [0, 1], 0], square['corrected'][t, [0, 1], 1], '-', color="#EDB120", linewidth=1)
    plt.plot(square['corrected'][t, [0, 2], 0], square['corrected'][t, [0, 2], 1], '-', color="#EDB120", linewidth=1)
    plt.plot(square['corrected'][t, [2, 3], 0], square['corrected'][t, [2, 3], 1], '-', color="#EDB120", linewidth=1)
    plt.plot(square['corrected'][t, [3, 1], 0], square['corrected'][t, [3, 1], 1], '-', color="#EDB120", linewidth=1)
    for p in range(trackingpoints['nfilpoints']):
        plt.plot(monofilament['corrected'][t, p, 0], monofilament['corrected'][t, p, 1], '.', markersize=10)
    plt.xlim([-0.1, 20.1])
    plt.ylim([-0.1, 20.1])

    plt.pause(0.01)
    plt.clf()  # Clear the figure for the next frame

print("done.")

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Constants
FRAME_INTERVAL = 40
DISTANCE = 3  # Minimum distance between peaks
SHIFT_ANGLE = -7
SHIFT_FORCE = 7
SHIFT_FREQ = 7
SHIFT_NERVE = 7

# Example trackingpoints and mng data for demonstration (replace these with actual data)
trackingpoints = {
    'locs': {
        'P2_x': np.array([]),  # Replace with actual x coordinates for P2
        'P2_y': np.array([]),  # Replace with actual y coordinates for P2
        'P4_x': np.array([]),  # Replace with actual x coordinates for P4
        'P4_y': np.array([]),  # Replace with actual y coordinates for P4
        'P5_x': np.array([]),  # Replace with actual x coordinates for P5
        'P5_y': np.array([]),  # Replace with actual y coordinates for P5
    }
}

mng = {
    'time_sec': np.array([]),  # Replace with actual time data
    'spikes': np.array([]),    # Replace with actual spike data
    'iff': np.array([]),       # Replace with actual instantaneous firing frequency data
    'accelerometer': np.array([]),  # Replace with actual accelerometer data
}

# I. Calculate the bending angle of the monofilament
A = np.column_stack((trackingpoints['locs']['P2_x'], trackingpoints['locs']['P2_y']))
B = np.column_stack((trackingpoints['locs']['P4_x'], trackingpoints['locs']['P4_y']))
C = np.column_stack((trackingpoints['locs']['P5_x'], trackingpoints['locs']['P5_y']))

# Vectors and angles
AB = B - A
BC = C - B
dot_product = np.einsum('ij,ij->i', AB, BC)
magnitude_AB = np.linalg.norm(AB, axis=1)
magnitude_BC = np.linalg.norm(BC, axis=1)
angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_BC))
angle_degrees = np.degrees(angle_radians)

# II. Normalize signals
n = 0
num_frames = len(mng['accelerometer']) // FRAME_INTERVAL

force_mean = []
freq_mean = []
nerve_mean = []

for i in range(num_frames):
    force_mean.append(np.mean(mng['accelerometer'][n:n + FRAME_INTERVAL]))
    freq_mean.append(np.max(mng['iff'][n:n + FRAME_INTERVAL]))
    nerve_mean.append(np.sum(mng['spikes'][n:n + FRAME_INTERVAL]))
    n += FRAME_INTERVAL

force_mean = np.array(force_mean) / 50
freq_mean = np.array(freq_mean) / 2
nerve_mean = np.array(nerve_mean)

# III. Shift signals
z = angle_degrees  # Assuming z is the bending angle
z1 = np.roll(z, SHIFT_ANGLE)
z2 = np.roll(force_mean, SHIFT_FORCE)
z3 = np.roll(freq_mean, SHIFT_FREQ)
z4 = np.roll(nerve_mean, SHIFT_NERVE)

# IV. Find peaks
peaks_angle, _ = find_peaks(z, distance=DISTANCE, height=0.2)
peaks_freq, _ = find_peaks(z3, distance=DISTANCE, height=0.2)

# V. Plot synchronized signals
plt.figure(figsize=(10, 6))
plt.plot(z2, 'r', label='Force')
plt.plot(z3, 'b', label='Frequency')
plt.plot(peaks_freq, z3[peaks_freq], '^', label='Freq Peaks')
plt.plot(z, 'k', label='Angle')
plt.plot(peaks_angle, z[peaks_angle], '^', label='Angle Peaks')
plt.plot(z4, 'g', label='Spikes')
plt.xlabel('Frames')
plt.ylabel('Signal Intensity')
plt.legend()
plt.title('Synchronized Signals: Force, Frequency, Angle, and Spikes')
plt.show()


import numpy as np
from scipy.signal import find_peaks, correlate
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import ProjectiveTransform

# Constants and data placeholders
FRAME_INTERVAL = 40
DISTANCE = 3  # Minimum distance between peaks
SHIFT_ANGLE = -7
SHIFT_FORCE = 7
SHIFT_FREQ = 7
SHIFT_NERVE = 7

# Example data (replace with actual data)
force_mean = np.array([])  # Placeholder for force data
freq_mean = np.array([])   # Placeholder for frequency data
nerve_mean = np.array([])  # Placeholder for nerve data
z = np.array([])           # Placeholder for angle data
trackingpoints = {'locs': {'P2_x': np.array([]), 'P2_y': np.array([]),
                           'P4_x': np.array([]), 'P4_y': np.array([]),
                           'P5_x': np.array([]), 'P5_y': np.array([])}}
monofilament_locs = np.array([])  # Placeholder for monofilament location data

# I. Signal without shift and plotting
plt.figure()
plt.plot(force_mean, 'r', label='Force')
plt.plot(freq_mean, 'b', label='Frequency')
pks, locs = find_peaks(freq_mean, distance=DISTANCE, height=0.2)
plt.plot(locs, pks, '^', label='Freq Peaks')
plt.plot(z, 'k', label='Angle')
pks, locs = find_peaks(z, distance=DISTANCE, height=0.2)
plt.plot(locs, pks, '^', label='Angle Peaks')
plt.plot(nerve_mean, 'g', label='Nerve')
plt.legend()
plt.show()

# II. Cross-correlation
x1, y1 = correlate(z, force_mean, mode='full', method='auto'), np.arange(-len(force_mean)+1, len(force_mean))
plt.figure()
plt.title('Cross-correlation')
plt.plot(y1, x1)
plt.show()

# III. Plotting points from CSV or image
frame_no = 2257
# Replace cs array with your data containing coordinates at each frame
# cs = np.array(...)  # Example data
# plt.plot(cs[frame_no, [20, 23, 26, 29]], cs[frame_no, [21, 24, 27, 30]], 'o') 

# IV. Polynomial fit to approximate angle
# Assuming cs array contains coordinates data per frame, replace cs with actual data
# filament_X = [2, 5, 8, 11, 14, 17] # Example filament X coordinates columns
# filament_Y = [3, 6, 9, 12, 15, 18] # Example filament Y coordinates columns

# Polyplot
# x = [cs[frame_no, col] for col in filament_X]
# y = [cs[frame_no, col] for col in filament_Y]
# p = np.polyfit(x, y, 4)
# y_fit = np.polyval(p, x)
# plt.plot(x, y, 'o', x, y_fit, '-')

# V. Homography estimation for coordinates
def calcul_taille_image(x_coords, y_coords):
    """Calculate length and width based on corner coordinates."""
    length_img = int(np.max(y_coords) - np.min(y_coords))
    width_img = int(np.max(x_coords) - np.min(x_coords))
    return length_img, width_img

def estimation_homographie(length_img, width_img, x1, y1):
    """Estimate homography transformation based on provided coordinates."""
    src = np.column_stack((x1, y1))
    dst = np.array([[0, 0], [width_img - 1, 0], [width_img - 1, length_img - 1], [0, length_img - 1]])
    transform = ProjectiveTransform()
    transform.estimate(src, dst)
    return transform

# Calculate homography based on four corner points
square_x = [20, 23, 26, 29]  # Example square corner X coordinates in the data
square_y = [21, 24, 27, 30]  # Example square corner Y coordinates in the data
filament_X = 17
filament_Y = 18

i = 2980  # Image frame index (example)
x1 = monofilament_locs[i, square_x]
y1 = monofilament_locs[i, square_y]
f_xy = [monofilament_locs[i, filament_X], monofilament_locs[i, filament_Y]]

length_img, width_img = calcul_taille_image(x1, y1)
H = estimation_homographie(length_img, width_img, x1, y1)

# Create final blank image array
im_f = np.zeros((length_img, width_img))

# VI. Creating a table for coordinates and exporting as CSV
coordinates_df = pd.DataFrame({
    'Top_left_x': monofilament_locs[:, 20],
    'Top_left_y': monofilament_locs[:, 21],
    'Top_Right_x': monofilament_locs[:, 23],
    'Top_right_y': monofilament_locs[:, 24],
    'Bottom_left_x': monofilament_locs[:, 26],
    'Bottom_left_y': monofilament_locs[:, 27],
    'Bottom_right_x': monofilament_locs[:, 29],
    'Bottom_right_y': monofilament_locs[:, 30],
    'Tip_x': monofilament_locs[:, 17],
    'Tip_y': monofilament_locs[:, 18],
    'Angle': z,
    'Force': force_mean,
    'Frequency': freq_mean,
    'Nerve': nerve_mean
})

coordinates_df.to_csv('coordinates.csv', index=False)

import numpy as np
import matplotlib.pyplot as plt

def calcul_taille_image(x1, y1):
    # Placeholder function to calculate image dimensions based on square coordinates
    length_img = int(max(y1) - min(y1))
    width_img = int(max(x1) - min(x1))
    return length_img, width_img

def estimation_homographie(length_img, width_img, x1, y1):
    # Placeholder function for homography estimation
    # You should replace this with your actual implementation
    return np.eye(3)  # Identity matrix as a placeholder

def compute_homography(input_data, num_images, square_coords, filament_indices):
    # Initialize storage for results
    homography_results = np.zeros((num_images, 8))

    for i in range(num_images):
        # Extract coordinates for the current frame
        x1 = input_data[i, square_coords[:, 0]]
        y1 = input_data[i, square_coords[:, 1]]
        
        # Calculate image dimensions
        length_img, width_img = calcul_taille_image(x1, y1)
        
        # Estimate homography
        H = estimation_homographie(length_img, width_img, x1, y1)
        
        # Prepare matrix for homography transformation
        m = np.ones((4, 4))
        m[0:2, :] = np.array([x1, y1])
        
        # Apply inverse homography to find new coordinates
        m_inv = np.linalg.solve(H, m[0:3, :])
        n = m_inv / m_inv[2, :]
        
        # Store results (flatten the first two coordinates)
        homography_results[i, :] = n[0:2, :].flatten()
        
    return homography_results

# Example usage
n_img = 10  # Number of images
input_data = np.random.rand(n_img, 10)  # Replace with your actual input data
square_coords = np.array([[0, 1], [2, 3]])  # Replace with actual indices for square coordinates
filament_indices = [4, 5]  # Replace with actual filament X and Y indices

# Compute homographies
new_xy_fil = compute_homography(input_data, n_img, square_coords, filament_indices)

# Visualize results
for i in range(n_img):
    plt.plot(new_xy_fil[i, 0:4:2], new_xy_fil[i, 1:4:2], '.', markersize=20)
plt.title('Homography Results')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def calcul_taille_image(x1, y1):
    """Calculate the dimensions of the image based on square coordinates."""
    length_img = int(max(y1) - min(y1))
    width_img = int(max(x1) - min(x1))
    return length_img, width_img

def estimation_homographie(length_img, width_img, x1, y1):
    """Estimate homography based on image dimensions and square coordinates."""
    # Placeholder: return an identity matrix for now
    return np.eye(3)

def conversion(im_f):
    """Convert image dimensions to a matrix format (placeholder)."""
    return np.zeros((4, 4))  # Replace with actual conversion logic

def compute_homographies(input_data, n_img, square_x, square_y, filament_X, filament_Y):
    """Compute homographies for all images and extract new coordinates."""
    new_xy_fil = np.zeros((n_img, 2))
    
    for i in range(n_img):
        x1 = input_data[i, square_x]
        y1 = input_data[i, square_y]
        f_xy = input_data[i, [filament_X, filament_Y]]
        
        length_img, width_img = calcul_taille_image(x1, y1)
        H = estimation_homographie(length_img, width_img, x1, y1)

        im_f = np.zeros((length_img, width_img))
        m = conversion(im_f)
        m = m[:3, :]
        n_pixel = m.shape[1]

        m = np.ones((4, 4))
        m[0:2, :] = np.array([x1, y1])
        n = np.linalg.solve(np.eye(4), m[:3, :])  # Solve for n
        n[0, :] /= n[-1, :]
        n[1, :] /= n[-1, :]
        
        m_inv = np.linalg.solve(H, m[:3, :])
        n[0, :] = m_inv[0, :] / m_inv[2, :]
        n[1, :] = m_inv[1, :] / m_inv[2, :]
        n[2, :] = 1
        n = np.floor(n).astype(int)
        
        n_xy = n[:2, :]
        
        # Use a KDTree for fast nearest neighbor search
        tree = cKDTree(n_xy.T)
        id_fil = tree.query(f_xy)[1]
        
        new_xy_fil[i, :] = [m[0, id_fil], m[1, id_fil]]
        
        # Additional results can be stored similarly
        # new_xy_fil_1[i, :] = ...
        # new_xy_fil_2[i, :] = ...
        # new_xy_fil_3[i, :] = ...
        # new_xy_fil_4[i, :] = ...

    return new_xy_fil

def plot_3D_mapping(new_xy_fil, freq_mean):
    """Plot the 3D receptive field mapping."""
    x1 = new_xy_fil[:, 0]
    y1 = new_xy_fil[:, 1]
    z5 = freq_mean
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x1, y1, z5, cmap='viridis')
    ax.set_title('3D image for Receptive field mapping')
    ax.set_xlabel('monofilament tip x')
    ax.set_ylabel('monofilament tip y')
    ax.set_zlabel('frequency')
    plt.show()

# Example usage
n_img = 10  # Number of images
input_data = np.random.rand(n_img, 20)  # Replace with actual input data
square_x = [1, 3, 5, 7]  # Indices for square X coordinates
square_y = [2, 4, 6, 8]  # Indices for square Y coordinates
filament_X = 9  # Index for filament X coordinate
filament_Y = 10  # Index for filament Y coordinate
freq_mean = np.random.rand(n_img)  # Replace with actual frequency data

# Compute homographies
new_xy_fil = compute_homographies(input_data, n_img, square_x, square_y, filament_X, filament_Y)

# Plot the results
plot_3D_mapping(new_xy_fil, freq_mean)

# Additional visualization for a specific frame
frame_no = 1  # Example frame number
plt.figure()
plt.plot(new_xy_fil[frame_no, square_x], new_xy_fil[frame_no, square_y], 'r-', label='Square')
plt.title('Frame Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
