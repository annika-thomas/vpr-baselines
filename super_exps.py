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

def create_all_to_all_associations(array1, array2):
    associations = []
    for index1, item1 in enumerate(array1):
        for index2, item2 in enumerate(array2):
            associations.append((item1, item2))
    return associations

def filter_image_names_and_descriptors(image_names, descriptors, start_idx, end_idx):
    filtered_image_names = []
    filtered_descriptors = []
    
    for image_name, descriptor in zip(image_names, descriptors):
        # Extract the index from the image name
        image_index = int(image_name.split('_')[-1].split('.')[0])
        
        # Check if the index is within the specified range
        if start_idx <= image_index <= end_idx:
            filtered_image_names.append(image_name)
            filtered_descriptors.append(descriptor)
    
    return filtered_image_names, filtered_descriptors

def main(args):

    # This is a command line argument
    pathToPathConfig = args.config

    print("Length of resize: ", len(args.resize))

    if len(args.resize) == 2 and args.resize[1] == -1:
        args.resize = args.resize[0:1]
    if len(args.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            args.resize[0], args.resize[1]))
    elif len(args.resize) == 1 and args.resize[0] > 0:
        print('Will resize max dimension to {}'.format(args.resize[0]))
    elif len(args.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    matchingConfig = readConfig("config/matching_config.yml")
    pathConfig = readConfig(pathToPathConfig)

    # Load the SuperPoint and SuperGlue models.
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

    output_directory = pathConfig.pathToOutputData
    os.makedirs(output_directory, exist_ok=True)

    for experimentName in matchingConfig.experiments:

        #timer = AverageTimer(newline=True)

        experimentConfig = matchingConfig.experiments[experimentName]
        experimentDescription = experimentConfig.description

        dataset1Name = experimentConfig.dataset1Name
        dataset2Name = experimentConfig.dataset2Name
        start_idx_dataset_1 = experimentConfig.startIdx1
        end_idx_dataset_1 = experimentConfig.endIdx1
        start_idx_dataset_2 = experimentConfig.startIdx2
        end_idx_dataset_2 = experimentConfig.endIdx2

        print("Experiment description: ", experimentDescription)

        input_file_path1 = os.path.join(output_directory, f'test{dataset1Name}_superpoint.pickle')
        # f"/home/annika/Documents/batvik_baselines/test{dataset1Name}_sift.pickle"

        # Load pickle file of image descriptors
        with open(input_file_path1, 'rb') as input_file1:
            loaded_data1 = pickle.load(input_file1)

        # Create an array counting from 1 to n
        #keyframes_array1 = [i for i in range(1, len(loaded_data1) + 1)]
        frame_ds1 = []
        desc_ds1 = []

        for i in range(len(loaded_data1)):
            frame_ds1.append(loaded_data1[i][0])
            #print(frame_ds1[i])

        for j in range(len(loaded_data1)):
            desc_ds1.append(loaded_data1[j][1])
            # print(desc_ds1[j])

        image_names_subset1, descriptors_subset1 = filter_image_names_and_descriptors(frame_ds1, desc_ds1, start_idx_dataset_1, end_idx_dataset_1)

        #print(image_names_subset1)

        input_file_path2 = os.path.join(output_directory, f'test{dataset2Name}_superpoint.pickle')
        
        #= f"/home/annika/Documents/batvik_baselines/test{dataset2Name}_sift.pickle"

        # Load pickle file of image descriptors
        # with open(input_file_path2, 'rb') as input_file2:
        #     loaded_data2 = pickle.load(input_file2)

        loaded_data2 = loaded_data1

        frame_ds2 = []
        desc_ds2 = []

        for k in range(len(loaded_data2)):
            frame_ds2.append(loaded_data2[k][0])
            # print(frame_ds2[k])

        for l in range(len(loaded_data2)):
            desc_ds2.append(loaded_data2[l][1])
            # print(desc_ds2[l])

        image_names_subset2, descriptors_subset2 = filter_image_names_and_descriptors(frame_ds2, desc_ds2, start_idx_dataset_2, end_idx_dataset_2)

        # # Create all-to-all associations
        associations = create_all_to_all_associations(image_names_subset1, image_names_subset2)
        associationsNums = create_all_to_all_associations(list(range(1,len(image_names_subset1)+1)), list(range(1,len(image_names_subset2)+1)))  

        #print("len(associations):", len(associations))

        scores = [[item[0], item[1], 0] for item in associationsNums]

        input_dir1 = os.path.join(output_directory, f'{dataset1Name}')
        input_dir2 = os.path.join(output_directory, f'{dataset2Name}')
        rot0, rot1 = 0, 0

        match_times_all = []
        search_times = []
        assoc_prev = 'nada'
        match_times = 0

        kp1 = [d['keypoints'] for d in descriptors_subset1]
        kp2 = [d['keypoints'] for d in descriptors_subset2]

        d1 = [d['descriptors'] for d in descriptors_subset1]
        d2 = [d['descriptors'] for d in descriptors_subset2]

        s1 = [d['scores'] for d in descriptors_subset1]
        s2 = [d['scores'] for d in descriptors_subset2]

        for idx in range(0, len(associations)):
            assoc1 = associations[idx][0]
            assoc2 = associations[idx][1]
            #print("assoc1: ", assoc1)
            #print("assoc2: ", assoc2)



            stem0, stem1 = Path(assoc1).stem, Path(assoc2).stem
            image1dir = os.path.join(input_dir1, assoc1)
            # #print("image1dir: ", image1dir)
            image2dir = os.path.join(input_dir2, assoc2)

            #print(image1dir)
            #print("image2dir: ", image2dir)

            # descriptors_subset1 has tensors for 'scores' then 'descriptors' then 'keypoints' - how do I make a subset of just the 'keypoints' tensors

            # desc1 = descriptors_subset1['keypoints']
            # desc2 = descriptors_subset2['keypoints']

            image0, inp0, scales0 = read_image(
                image1dir, device, args.resize, rot0, args.resize_float)
            image1, inp1, scales1 = read_image(
                image2dir, device, args.resize, rot1, args.resize_float)
            

            #print('inp1', inp1)

            # if image0 is None or image1 is None:
            #     print('Problem reading image pair: {} {}'.format(
            #         input_dir1/assoc1, input_dir2/assoc2))
            #     exit(1)
            #timer.update('load_image')

            # print(desc1)

            #all_keypoints = [d['keypoints'] for d in data]

            # desc1 = [d['keypoints'] for d in descriptors_subset1]
            # desc2 = [d['keypoints'] for d in descriptors_subset2]

            #print(desc1)

            desc1k = kp1[associationsNums[idx][0]-1]
            desc2k = kp2[associationsNums[idx][1]-1]

            desc1d = d1[associationsNums[idx][0]-1]
            desc2d = d2[associationsNums[idx][1]-1]

            desc1s = s1[associationsNums[idx][0]-1]
            desc2s = s2[associationsNums[idx][1]-1]


            pred = matching({'keypoints0': desc1k, 'keypoints1': desc2k, 'image0': inp0, 'image1': inp1, 'descriptors0': desc1d, 'descriptors1': desc2d, 'scores0': desc1s, 'scores1': desc2s})


            #if do_match:
            # Perform the matching.
            #pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            #kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            #timer.update('matcher')

            # # Write the matches to disk.
            # out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
            #                'matches': matches, 'match_confidence': conf}
            # np.savez(str(matches_path), **out_matches)

            # Keep the matching keypoints.
            # valid = matches > -1
            # mkpts0 = kpts0[valid]
            # mkpts1 = kpts1[matches[valid]]
            # mconf = conf[valid]


            # image0, inp0, scales0 = read_image(
            #     image1dir, device, args.resize, rot0, args.resize_float)
            # image1, inp1, scales1 = read_image(
            #     image2dir, device, args.resize, rot1, args.resize_float)
            # if image0 is None or image1 is None:
            #     print('Problem reading image pair: {} {}'.format(
            #         input_dir1/assoc1, input_dir2/assoc2))
            #     exit(1)
            # #timer.update('load_image')
                
            # time_start = time.time()

            # pred = matching({'image0': inp0, 'image1': inp1})
            # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            # kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            # matches, conf = pred['matches0'], pred['matching_scores0']
            # #timer.update('matcher')

            # time_end = time.time()

            # match_time = time_end - time_start
            # match_times_all.append(match_time)

            #print(desc1k)

            kpts0 = desc1k[0].cpu().numpy()
            kpts1 = desc2k[0].cpu().numpy()

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            
            # Keep the matching keypoints.
            valid = matches > -1

            # keep matches with confidence above 0.7
            valid = valid & (conf > 0.7)

            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            #print('mconf: ', mconf)

            # print the values of each match
            totalImageConf = 0
            for i in range(len(mkpts0)):
                #print("Match ", i, ":", mkpts0[i], ":", mkpts1[i], ":", mconf[i])
                totalImageConf = totalImageConf + mconf[i]

            #print("Total image confidence: ", totalImageConf)
            # if totalImageConf > 0:
            #     print("total image confidence: ", totalImageConf)
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            #viz_path = output_directory / '{}_{}_matches.{}'.format(stem0, stem1, args.viz_extension)

            # viz_path = os.path.join(output_directory, '{}_{}_matches.{}'.format(stem0, stem1, args.viz_extension))

            # make_matching_plot(
            #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            #     text, viz_path, args.show_keypoints,
            #     args.fast_viz, args.opencv_display, 'Matches', small_text)

            # timer.update('viz_match')

            # desc1 = descriptors_subset1[assoc1-1]
            # desc2 = descriptors_subset2[assoc2-1]

            # #print(desc1)

            # UNCOMMENT TO GET PROGRESS OF MATCHING
            if (idx%10 ==0):
               print("progress: ", idx/len(associations))
            
            # matches = match_sift(desc1, desc2)

            # # #plot_matches(img1, img2, edges1, edges2, matches)

            # # #print(len(img1))
            # # #score = len(matches)/len(edges1)
            score = totalImageConf

            # # #print(type(score))
            # # #if (score > 2):
            # # #   match = 1
            # # #else: 
            # # #   match = 0
            scores[idx][2] = score
            # #print(scores)

            # if assoc_prev == assoc1 or assoc_prev == 'nada':
            #     match_times = match_times + match_time
            # else:
            #     search_times.append(match_times)
            #     print("added seach time: ", match_times)
            #     match_times = 0

            # assoc_prev = assoc1

            
        # mean and standard deviation of search times
        search_times = np.array(search_times)
        print("mean search time: ", np.mean(search_times))
        print("std search time: ", np.std(search_times))

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

        # # make Matplotlib not use agg because it's a non-GUI backend
        # plt.switch_backend('TkAgg')

        # # Create the heatmap using imshow
        # plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')

        # # Add colorbar
        # # cbar = plt.colorbar()
        # # cbar.set_label('Match Count')

        # # # Set labels and title
        # plt.xlabel('Traj 1 (m)')
        # plt.ylabel('Traj 2 (m)')
        # plt.title(experimentDescription)

        # Save all descriptors in a single pickle file
        output_file_path = output_directory + f'test{dataset1Name}_test{dataset2Name}_superglue.pickle'
        with open(output_file_path, 'wb') as output_file:
            pickle.dump(heatmap_data, output_file)

        print(f"Saved all descriptors to {output_file_path}")

        # Show the plot
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='match_SIFT_exps.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')

    args = parser.parse_args()
    main(args)