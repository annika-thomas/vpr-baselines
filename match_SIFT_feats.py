import csv
import os
import cv2
from helper_functions import *
import argparse
from utils import readConfig
import string
import pickle
import time

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
            associations.append((item1, item2))
    return associations

def main(args):

    # These should be command line arguments.
    pathToPathConfig = args.config
    dataset1Name = args.dataset_1
    dataset2Name = args.dataset_2

    output_directory = '/home/annika/Documents/batvik_gt/'
    os.makedirs(output_directory, exist_ok=True)

    pathConfig = readConfig(pathToPathConfig)

        #input_file_path = pathConfig.DataBasePath + "/" + dataset1Name + "_orb.pickle"
    input_file_path1 = f"/home/annika/Documents/batvik_gt/test{dataset1Name}_middle_sift.pickle"

    # Load pickle file of image descriptors
    with open(input_file_path1, 'rb') as input_file1:
        loaded_data1 = pickle.load(input_file1)

    start_time = time.time()

    # Create an array counting from 1 to n
    keyframes_array1 = [i for i in range(1, len(loaded_data1) + 1)]

    input_file_path2 = f"/home/annika/Documents/batvik_gt/test{dataset2Name}_middle_sift.pickle"

    # Load pickle file of image descriptors
    with open(input_file_path2, 'rb') as input_file2:
        loaded_data2 = pickle.load(input_file2)

    # Create an array counting from 1 to n
    keyframes_array2 = [i for i in range(1, len(loaded_data2) + 1)]


    # Create all-to-all associations
    associations = create_all_to_all_associations(keyframes_array1, keyframes_array2)

    scores = [[item[0], item[1], 0] for item in associations]
    print(scores)

    #print("associations[0] ", associations)

    #print("loaded data: ", loaded_data[1])

    for idx in range(0, len(associations)):
         assoc1 = associations[idx][0]
         assoc2 = associations[idx][1]

         #print(assoc1)
         #print(assoc2)

         desc1 = loaded_data1[assoc1-1]
         desc2 = loaded_data2[assoc2-1]

         print(idx/len(associations))
        
         matches = match_sift(desc1, desc2)

    #     #plot_matches(img1, img2, edges1, edges2, matches)

    #     #print(len(img1))
    #     #score = len(matches)/len(edges1)
         score = matches

        #print(type(score))
         #if (score > 30):
         #   match = 1
         #else: 
         #   match = 0
         scores[idx][2] = score
    #     #print(scores)
        

    #print(scores)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    # Extract x, y, and color values every entry
    x = [item[0] for item in scores[::]]
    y = [item[1] for item in scores[::]]
    colors = [item[2] for item in scores[::]]

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
    plt.xlabel('Traj 1 (m)')
    plt.ylabel('Traj 2 (m)')
    plt.title('Batvik Different Seasons Trajectories')

    # Save all descriptors in a single pickle file
    output_file_path = output_directory + f'test{dataset1Name}_test{dataset2Name}_middle_sift.pickle'
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(heatmap_data, output_file)

    print(f"Saved all descriptors to {output_file_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='sift_exp1.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument('--dataset_1', required=True, help="Name of dataset to use")
    parser.add_argument('--dataset_2', required=True, help="Name of dataset to use")

    args = parser.parse_args()
    main(args)