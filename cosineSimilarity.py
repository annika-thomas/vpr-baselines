from matplotlib import pyplot as plt
import numpy as np
import os
from utils import readConfig
import pickle
import argparse

def cosine_similarity(vec1, vec2):
    # Flatten the vectors if they are 2D arrays of shape (1, n)
    if vec1.ndim == 2 and vec1.shape[0] == 1:
        vec1 = vec1.flatten()
    if vec2.ndim == 2 and vec2.shape[0] == 1:
        vec2 = vec2.flatten()

    # Check if the vectors are indeed 1D arrays now
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Input vectors should be 1D arrays or 2D arrays with one dimension being 1.")

    # Calculate the dot product between the two vectors
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitude (norm) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Calculate the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)

    # Check for numerical issues
    if np.isnan(similarity).any():
        raise ValueError("Numerical issues encountered, make sure the input vectors are not zero vectors.")

    return similarity

def create_all_to_all_associations(array1, array2):
    associations = []
    for index1, item1 in enumerate(array1):
        for index2, item2 in enumerate(array2):
            associations.append((item1, item2))
    return associations

import os
import numpy as np
import natsort

def load_npy_files(directory):
    npy_contents = []
    counter = 0
    # List all .npy files in the given directory and sort them numerically
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    sorted_npy_files = natsort.natsorted(npy_files)
    
    # Iterate over the sorted list of .npy files
    for file in sorted_npy_files:
        # Construct full file path
        file_path = os.path.join(directory, file)
        # Load the .npy file
        data = np.load(file_path)
        # Append the filename (without extension) and data to the list
        npy_contents.append((os.path.splitext(file)[0], data))
        counter += 1

    return npy_contents, counter

def main(args):

    
    # These should be command line arguments.
    pathToPathConfig = args.config
    dataset1Name = args.dataset_1
    dataset2Name = args.dataset_2

    output_directory = '/home/annika/Documents/batvik_baselines/Anyloc/anyloc_results/'
    os.makedirs(output_directory, exist_ok=True)

    pathConfig = readConfig(pathToPathConfig)
    # Example usage:
    # Replace 'your_directory_path' with the actual path to the directory containing .npy files
    # directory_path = '/home/annika/Downloads/27output-20231109T183037Z-001/27output'
    # directory_path = '/home/annika/Downloads/drive-download-20231112T222409Z-001'
    # directory_path = '/home/annika/Downloads/27output-20231113T191648Z-001/27output'
    #directory_path = '/home/annika/Downloads/27output-20231114T175504Z-001/27output'
    #directory_path = '/home/annika/Downloads/27output-20231114T232659Z-001/27output'
    directory_path = f'/home/annika/Documents/batvik_baselines/Anyloc/{dataset1Name}output'
    npy_files_contents, counter = load_npy_files(directory_path)

    associations = create_all_to_all_associations(npy_files_contents, npy_files_contents)
    associationsCount = create_all_to_all_associations(list(range(1,len(npy_files_contents)+1)), list(range(1,len(npy_files_contents)+1))) 

    # associations(associations number/index, first or second object, object number/vector)
    #print(associations[1])
    #print(associations[1][0])
    #print(associations[1][1])
    #print(len(associations))

    scores = [[item[0], item[1], 0] for item in associationsCount]
    #print(scores)

    for idx in range(0, len(associations)):
         assoc1 = associations[idx][0]
         assoc2 = associations[idx][1]

         #print(assoc1)
         #print(assoc2)

         desc1 = associations[idx][0][1]
         desc2 = associations[idx][1][1]

         print(idx/len(associations))
        
         matches = cosine_similarity(desc1, desc2)

    #     #plot_matches(img1, img2, edges1, edges2, matches)

    #     #print(len(img1))
    #     #score = len(matches)/len(edges1)
         score = matches

        #  if (score > 0.75):
        #    match = 1
        #  else: 
        #    match = 0
         scores[idx][2] = score
    #     #print(scores)
        
    # print([item[0] for item in scores[::]])
    # print([item[1] for item in scores[::]])
    # #print([item[2] for item in scores[::]])

    # #print(scores)

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print("Elapsed time: ", elapsed_time)

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
    plt.xlabel(f'Traj {dataset1Name} (m)')
    plt.ylabel(f'Traj {dataset2Name} (m)')
    plt.title('Batvik Traverses AnyLoc')

    # Save all descriptors in a single pickle file
    output_file_path = output_directory + f'test{dataset1Name}_test{dataset2Name}_anyloc.pickle'
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(heatmap_data, output_file)

    print(f"Saved all descriptors to {output_file_path}")

    # Show the plot
    plt.show()

    npy_files_contents1 = npy_files_contents

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='sift_exp1.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument('--dataset_1', required=True, help="Name of dataset to use")
    parser.add_argument('--dataset_2', required=True, help="Name of dataset to use")

    args = parser.parse_args()
    main(args)



