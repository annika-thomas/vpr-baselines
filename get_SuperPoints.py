import csv
import os
import cv2
from helper_functions import *
import argparse
from utils import readConfig
import string
import pickle
import time
from pathlib import Path
import random
import numpy as np
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt

from SuperGluePretrainedNetwork.models.superpoint import SuperPoint

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def import_csv(file_path):
    data = []
    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(row)
    return data

def main(args):

    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': 0.005,
            'max_keypoints': args.max_keypoints
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    matching = Matching(config).eval().to(device)

    output_directory = '/home/annika/Documents/batvik_baselines/'
    os.makedirs(output_directory, exist_ok=True)

    # These should be command line arguments.
    pathToPathConfig = args.config
    dataset1Name = args.dataset_1

    pathConfig = readConfig(pathToPathConfig)

    print(pathConfig.KeyframesBasePath)

    # csv_file_path1 = "/home/annika/Documents/keyframes/keyframes_test10_160_465.csv"  
    #csv_file_path1 = pathConfig.KeyframesBasePath + "/blob_tracking_experiment_2_cache_47_1695_3640.pickle_frames.csv"
    #csv_file_path1 = pathConfig.KeyframesBasePath + '/keyframes_batvik' + dataset1Name + "_middle_traverse.csv"
    csv_file_path1 = pathConfig.KeyframesBasePath + '/frame_csvs/frame_csvs/batvik_' + dataset1Name + "_sam_track_cache.pickle_frames.csv"

    print(csv_file_path1)
    array1 = import_csv(csv_file_path1)

    keyframes1 = array1[1:]
    print("len(keyframes1): ", len(keyframes1))

    #Create all-to-all associations
    #associations = create_all_to_all_associations(keyframes1, keyframes2)
    rot0, rot1 = 0, 0

    all_descriptors = []

    for idx in range(0, len(keyframes1)):
        #path1 = pathConfig.DataBasePath +"/" + dataset1Name + '/camera'
        path1 = pathConfig.DataBasePath +"/" + dataset1Name + '/camera'
        filePath1 = os.path.join(path1,str(keyframes1[idx])[2:-2])

        #print((idx)/len(keyframes1))
        print(idx/len(keyframes1))

        img1 = cv2.imread(filePath1, 0)
        #print("img1: ", img1)

        image0, inp0, scales0 = read_image(
                filePath1, device, args.resize, rot0, args.resize_float)

        # get superpoint descriptors

        #img1 = cv2.imread(filePath1, 0)
        # img1 = cv2.resize(img1, (640, 480))
        # img1 = img1.astype('float32') / 255.0
        # data = {'image0': image0}
        # #data = {'image1': img1}
        # data = {k: torch.from_numpy(v)[None].to(device) for k, v in data.items()}
        # pred = matching(data)
        
        #imgSuperpoint = SuperPoint(config.get('superpoint', {}))
        imgSuperpoint = SuperPoint(config.get('superpoint', {})).to(device)

        superpoints = imgSuperpoint({'image': inp0.to(device)})

        #print(superpoints['descriptors'])

        # print superpoints 'descriptors'
        #print(superpoints['descriptors'].shape)

        #print(pred)

        #kp, desc1 = edge_processing_sift(img1)

        all_descriptors.append([str(keyframes1[idx])[2:-2], superpoints])
        #print(all_descriptors[idx][1])
        

    #print(all_descriptors)

    # Save all descriptors in a single pickle file
    output_file_path = output_directory + f'test{dataset1Name}_superpoint.pickle'
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(all_descriptors, output_file)

    print(f"Saved all descriptors to {output_file_path}")

    #print(f"all descriptors: {all_descriptors}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='get_sift_feats.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument('--dataset_1', required=True, help="Name of dataset to use")
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    args = parser.parse_args()
    main(args)