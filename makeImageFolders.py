import csv
import os
import cv2
from helper_functions import *
import argparse
from utils import readConfig
import string
import pickle

def import_csv(file_path):
    data = []
    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(row)
    return data

def main(args):

    # These should be command line arguments.
    pathToPathConfig = args.config
    dataset1Name = args.dataset_1

    output_directory = f'/home/annika/Documents/batvik_baselines/{dataset1Name}'
    os.makedirs(output_directory, exist_ok=True)

    pathConfig = readConfig(pathToPathConfig)

    print(pathConfig.KeyframesBasePath)

    # csv_file_path1 = "/home/annika/Documents/keyframes/keyframes_test10_160_465.csv"  
    csv_file_path1 = pathConfig.KeyframesBasePath + '/frame_csvs/frame_csvs/batvik_' + dataset1Name + "_sam_track_cache.pickle_frames.csv"
    
    #csv_file_path1 = pathConfig.KeyframesBasePath + "/blob_tracking_experiment_2_cache_41_165_4400.pickle_frames.csv"

    print(csv_file_path1)
    array1 = import_csv(csv_file_path1)

    keyframes1 = array1[1:]
    # print(keyframes1)

    # Create all-to-all associations
    # associations = create_all_to_all_associations(keyframes1, keyframes2)

    # Create a directory to save the images
    images_output_directory = os.path.join(output_directory, 'images')
    os.makedirs(images_output_directory, exist_ok=True)

    # Process and save each image.
    for idx, keyframe in enumerate(keyframes1):
        path1 = os.path.join(pathConfig.DataBasePath, dataset1Name, 'camera')
        filePath1 = os.path.join(path1, str(keyframe)[2:-2])

        # Load the image.
        img1 = cv2.imread(filePath1, 0)

        # Process the image (this function needs to be defined).
        #edges1, desc1 = edge_processing_orb(img1, threshold=200)

        # Define the image save path.
        img_save_path = os.path.join(output_directory, str(keyframe)[2:-2])

        # Save the image.
        cv2.imwrite(img_save_path, img1)

        # You can also save descriptors or other data per image here as needed.
        # For example, saving descriptors:

        print(f"Saved image {idx+1}/{len(keyframes1)}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='get_ORB_feats.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument('--dataset_1', required=True, help="Name of dataset to use")

    args = parser.parse_args()
    main(args)