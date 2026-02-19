import pickle
import matplotlib.pyplot as plt

# Load the heatmap data from the pickle file
pickle_file_path = '/home/annika/Documents/batvik_baselines/test39_test42_superglue.pickle'
with open(pickle_file_path, 'rb') as pickle_file:
    heatmap_data = pickle.load(pickle_file)

# keep heatmap_data values above 10 and set to 1
heatmap_data[heatmap_data < 5] = 0
heatmap_data[heatmap_data > 5] = 1

# Create the heatmap using imshow
plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Match Count')

# Set labels and title
plt.xlabel('Traj 1 (m)')
plt.ylabel('Traj 2 (m)')

# Show the plot
plt.show()
