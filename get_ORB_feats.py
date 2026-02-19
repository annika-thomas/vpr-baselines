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

    output_directory = '/home/annika/Documents/batvik_baselines/'
    os.makedirs(output_directory, exist_ok=True)

    # These should be command line arguments.
    pathToPathConfig = args.config
    dataset1Name = args.dataset_1

    pathConfig = readConfig(pathToPathConfig)

    print(pathConfig.KeyframesBasePath)

    # csv_file_path1 = "/home/annika/Documents/keyframes/keyframes_test10_160_465.csv"  
    csv_file_path1 = pathConfig.KeyframesBasePath + '/frame_csvs/frame_csvs/batvik_' + dataset1Name + "_sam_track_cache.pickle_frames.csv"
    
    #csv_file_path1 = pathConfig.KeyframesBasePath + "/blob_tracking_experiment_2_cache_41_165_4400.pickle_frames.csv"

    print(csv_file_path1)
    array1 = import_csv(csv_file_path1)

    keyframes1 = array1[1:]
    print(keyframes1)

    # Create all-to-all associations
    # associations = create_all_to_all_associations(keyframes1, keyframes2)

    all_descriptors = []

    for idx in range(0, len(keyframes1)):
        #path1 = pathConfig.DataBasePath +"/" + dataset1Name + '/pngs/undistorted_images/t265_fisheye1'
        path1 = pathConfig.DataBasePath +"/" + dataset1Name + '/camera'
        filePath1 = os.path.join(path1,str(keyframes1[idx])[2:-2])
        #print(filePath1)
        print((idx)/len(keyframes1))
        #print(associations[idx][3])
        img1 = cv2.imread(filePath1, 0)

        edges1, desc1 = edge_processing_orb(img1, threshold=200)

        #all_descriptors.append((desc1))
        all_descriptors.append([str(keyframes1[idx])[2:-2], desc1])
        

    #print(all_descriptors)

    # Save all descriptors in a single pickle file
    output_file_path = output_directory + f'test{dataset1Name}_orb.pickle'
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(all_descriptors, output_file)

    print(f"Saved all descriptors to {output_file_path}")

    print(f"all descriptors: {all_descriptors}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='get_ORB_feats.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument('--dataset_1', required=True, help="Name of dataset to use")

    args = parser.parse_args()
    main(args)