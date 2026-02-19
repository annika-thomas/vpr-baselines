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

def filter_image_names_and_descriptors(image_names, descriptors, start_idx, end_idx):
    filtered_image_names = []
    filtered_descriptors = []
    filtered_indices = []
    
    for i, (image_name, descriptor) in enumerate(zip(image_names, descriptors)):
        # Extract the index from the image name
        image_index = int(image_name.split('_')[-1].split('.')[0])
        
        # Check if the index is within the specified range
        if start_idx <= image_index <= end_idx:
            filtered_image_names.append(image_name)
            filtered_descriptors.append(descriptor)
            filtered_indices.append(i)
    
    return filtered_image_names, filtered_descriptors, filtered_indices

def main(args):

    # These should be command line arguments.
    pathToPathConfig = args.config

    matchingConfig = readConfig("config/matching_config.yml")
    pathConfig = readConfig(pathToPathConfig)

    output_directory = pathConfig.pathtoOutputData
    os.makedirs(output_directory, exist_ok=True)

    for experimentName in matchingConfig.experiments:

        experimentConfig = matchingConfig.experiments[experimentName]
        experimentDescription = experimentConfig.description

        dataset1Name = experimentConfig.dataset1Name
        dataset2Name = experimentConfig.dataset2Name
        start_idx_dataset_1 = experimentConfig.startIdx1
        end_idx_dataset_1 = experimentConfig.endIdx1
        start_idx_dataset_2 = experimentConfig.startIdx2
        end_idx_dataset_2 = experimentConfig.endIdx2

        print("Experiment description: ", experimentDescription)

        #input_file_path = pathConfig.DataBasePath + "/" + dataset1Name + "_orb.pickle"
        input_file_path1 = os.path.join(output_directory, f'test{dataset2Name}_orb.pickle')
        #f"/home/annika/Documents/batvik_baselines/test{dataset1Name}_orb.pickle"

        #print("Dataset1: ", dataset1Name)
        #print("Dataset2: ", dataset2Name)

        # Load pickle file of image descriptors
        with open(input_file_path1, 'rb') as input_file1:
            loaded_data1 = pickle.load(input_file1)

        # Create an array counting from 1 to n
        keyframes_array1 = [i for i in range(1, len(loaded_data1) + 1)]
        frame_ds1 = []
        desc_ds1 = []

        for i in range(len(loaded_data1)):
            frame_ds1.append(loaded_data1[i][0])
            #print(frame_ds1[i])

        for j in range(len(loaded_data1)):
            desc_ds1.append(loaded_data1[j][1])
            # print(desc_ds1[j])

        image_names_subset1, descriptors_subset1, filtered_indices1 = filter_image_names_and_descriptors(frame_ds1, desc_ds1, start_idx_dataset_1, end_idx_dataset_1)

        #print("FirstIndex1: ", filtered_indices1[0])
        #print("LastIndex1: ", filtered_indices1[-1])
        #print(image_names_subset1)

        input_file_path2 = os.path.join(output_directory, f'test{dataset2Name}_orb.pickle')
        #f"/home/annika/Documents/batvik_baselines/test{dataset2Name}_orb.pickle"

        # Load pickle file of image descriptors
        with open(input_file_path2, 'rb') as input_file2:
            loaded_data2 = pickle.load(input_file2)

        frame_ds2 = []
        desc_ds2 = []

        for k in range(len(loaded_data2)):
            frame_ds2.append(loaded_data2[k][0])
            # print(frame_ds2[k])

        for l in range(len(loaded_data2)):
            desc_ds2.append(loaded_data2[l][1])
            # print(desc_ds2[l])

        image_names_subset2, descriptors_subset2, filtered_indices2 = filter_image_names_and_descriptors(frame_ds2, desc_ds2, start_idx_dataset_2, end_idx_dataset_2)

        #print("FirstIndex2: ", filtered_indices2[0])
        #print("LastIndex2: ", filtered_indices2[-1])
        #print(image_names_subset2)

        # Create an array counting from 1 to n
        keyframes_array2 = [i for i in range(1, len(loaded_data2) + 1)]

        #print(len(loaded_data))
        #print(keyframes_array)


        # # Create all-to-all associations
        #associations = create_all_to_all_associations(image_names_subset1, image_names_subset2)
        associations = create_all_to_all_associations(list(range(1,len(image_names_subset1)+1)), list(range(1,len(image_names_subset2)+1)))  

        #print("len(associations):", len(associations))

        scores = [[item[0], item[1], 0] for item in associations]
        start_time = time.time()

        #print(descriptors_subset1[0])

        # #print("associations[0] ", associations[0][0])

        # #print("loaded data: ", loaded_data[1])

        for idx in range(0, len(associations)):
            assoc1 = associations[idx][0]
            assoc2 = associations[idx][1]
            #print("assoc1: ", assoc1)
            #print("assoc2: ", assoc2)

            desc1 = descriptors_subset1[assoc1-1]
            desc2 = descriptors_subset2[assoc2-1]

            #print(desc1)

            
            # UNCOMMENT TO GET PROGRESS OF MATCHING
            # if (idx%1000 ==0):
            #    print("progress: ", idx/len(associations))
            
            matches = match_edge_descriptors_orb(desc1, desc2)

            # #plot_matches(img1, img2, edges1, edges2, matches)

            # #print(len(img1))
            # #score = len(matches)/len(edges1)
            score = len(matches)

            # #print(type(score))
            # #if (score > 2):
            # #   match = 1
            # #else: 
            # #   match = 0
            scores[idx][2] = score
            # #print(scores)
            
        # #print(scores)

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
        #plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')

        # Add colorbar
        # cbar = plt.colorbar()
        # cbar.set_label('Match Count')

        # # Set labels and title
        # plt.xlabel('Traj 1 (m)')
        # plt.ylabel('Traj 2 (m)')
        # plt.title(experimentDescription)

        # Save all descriptors in a single pickle file
        output_file_path = output_directory + f'test{dataset1Name}_test{dataset2Name}_orb.pickle'
        with open(output_file_path, 'wb') as output_file:
            pickle.dump(heatmap_data, output_file)

        # print(f"Saved all descriptors to {output_file_path}")

        # Show the plot
        #plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='match_ORB_exps.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")

    args = parser.parse_args()
    main(args)