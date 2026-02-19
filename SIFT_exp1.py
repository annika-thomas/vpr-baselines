import csv
import os
import cv2
from helper_functions import *

def import_csv(file_path):
    data = []
    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(row)
    return data

def create_all_to_all_associations(array1, array2):
    associations = []
    for index1, item1 in enumerate(array1):
        for index2, item2 in enumerate(array2):
            associations.append((item1, item2, index1, index2))
    return associations

def main():
    # Example file paths
    csv_file_path1 = "/home/annika/Documents/keyframes/keyframes_test11_190_670.csv"    
    csv_file_path2 = "/home/annika/Documents/keyframes/keyframes_test11_190_670.csv"

    # Import data from the CSV files
    array1 = import_csv(csv_file_path1)
    array2 = import_csv(csv_file_path2)

    keyframes1 = array1[1:]
    keyframes2 = array2[1:]

    # Create all-to-all associations
    associations = create_all_to_all_associations(keyframes1, keyframes2)
    #print(associations[1])
    scores = [[item[2], item[3], 0] for item in associations]
    #print(scores)
    print(len(associations))

    for idx in range(0, len(associations)):
        filePath1 = os.path.join('/home/annika/HighBayData/test11/pngs/undistorted_images/t265_fisheye1', str(associations[idx][0])[2:-2])
        #print(filePath1)
        print((idx)/len(associations))
        #print(associations[idx][3])
        img1 = cv2.imread(filePath1, 0)

        filePath2 = os.path.join('/home/annika/HighBayData/test11/pngs/undistorted_images/t265_fisheye1', str(associations[idx][1])[2:-2])
        #print(filePath2)
        img2 = cv2.imread(filePath2, 0)

        score = edge_match_sift(img1, img2)
        #print(type(score))
        if (score > 75):
            match = 1
        else: 
            match = 0
        scores[idx][2] = match
        #print(scores)
        

    print(scores)

    # Extract x, y, and color values every entry
    x = [item[0] for item in scores]
    y = [item[1] for item in scores]
    colors = [item[2] for item in scores]

    # Calculate the grid size based on the extracted x and y values
    x_grid_size = int(max(x)) + 1
    y_grid_size = int(max(y)) + 1

    # Create a 2D array to hold the color values for each grid cell
    heatmap_data = np.zeros((y_grid_size, x_grid_size))

    # Populate the heatmap data with the color values
    for i, (x_val, y_val, color_val) in enumerate(zip(x, y, colors)):
        heatmap_data[int(y_val), int(x_val)] = color_val

    # Create the heatmap using imshow
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Match Count')

    # Set labels and title
    plt.xlabel('Keyframe Traj 1 (0.2 m)')
    plt.ylabel('Keyframe Traj 2 (0.2 m)')
    plt.title('Highbay C-Shape Same Traverse')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()