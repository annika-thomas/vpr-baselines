import argparse
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from copy import deepcopy
from LoFTR.src.utils.plotting import make_matching_figure
from LoFTR.src.loftr import default_cfg
from LoFTR.src.loftr import LoFTR
#from kornia.feature import LoFTR
import os
from utils import readConfig
import csv
import time
import matplotlib.pyplot as plt
import pickle

def resize_image(image, max_length):
    h, w = image.shape
    if max(h, w) > max_length:
        scale_factor = max_length / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        new_h, new_w = (new_h // 8) * 8, (new_w // 8) * 8
        return cv2.resize(image, (new_w, new_h))
    else:
        return image
    
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

def resize_image(image, max_length):
    h, w = image.shape
    if max(h, w) > max_length:
        # Calculate the scale factor while maintaining the aspect ratio
        scale_factor = max_length / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        # Ensure the new dimensions are divisible by 8
        new_h, new_w = (new_h // 8) * 8, (new_w // 8) * 8
        return cv2.resize(image, (new_w, new_h))
    else:
        return image

def main(args):

    # This is a command line argument
    pathToPathConfig = args.config

    matchingConfig = readConfig("config/matching_config.yml")
    pathConfig = readConfig(pathToPathConfig)

    output_directory = pathConfig.pathToOutputData
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

        torch.cuda.empty_cache()

        pathToPathConfig = args.config
        pathConfig = readConfig(pathToPathConfig)

        cfg = default_cfg

        _default_cfg = deepcopy(default_cfg)
        _default_cfg['coarse']['temp_bug_fix'] = True
        matcher = LoFTR(config=_default_cfg)
        checkpoint_path = os.path.join(os.path.dirname(__file__), "outdoor_ds.ckpt")
        matcher.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        matcher = matcher.eval().cuda()

        imgs_dir1 = '/home/annika/Documents/batvik_baselines/41'
        imgs_dir2 = '/home/annika/Documents/batvik_baselines/42'

        img_names1 = sorted(os.listdir(imgs_dir1))
        img_names2 = sorted(os.listdir(imgs_dir2))

        print("len(img_names1): ", len(img_names1))
        print("len(img_names2): ", len(img_names2))

        #Create all-to-all associations
        associations = create_all_to_all_associations(img_names1, img_names2)
        associationsNum = create_all_to_all_associations(list(range(1,len(img_names1)+1)), list(range(1,len(img_names2)+1)))

        scores = [[item[0], item[1], 0] for item in associations]
        scores = [[item[0], item[1], 0] for item in associationsNum]

        rot0, rot1 = 0, 0

        # load and resize all the images
        imgNo = 0

        images1 = []
        for img_name in img_names1:
            print(img_name)
            img = cv2.imread(os.path.join(imgs_dir1, img_name), cv2.IMREAD_GRAYSCALE)
            img = resize_image(img, 896)
            images1.append(img)
            imgNo = imgNo + 1
            #print(imgNo)

        images2 = []
        for img_name in img_names2:
            print(img_name)
            img = cv2.imread(os.path.join(imgs_dir2, img_name), cv2.IMREAD_GRAYSCALE)
            img = resize_image(img, 896)
            images2.append(img)
            imgNo = imgNo + 1
            #print(imgNo)


        match_time = []
        search_times = []
        assoc_prev = -1
        match_times = 0

        for idx in range(0, 30):
            assoc1idx = associationsNum[idx][0]
            assoc2idx = associationsNum[idx][1]

            img1 = images1[assoc1idx-1]
            img2 = images2[assoc2idx-1]

            img0 = torch.from_numpy(img1)[None][None].cuda() / 255.
            img1 = torch.from_numpy(img2)[None][None].cuda() / 255.

            # display img0 and img1
            # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # axes[0].imshow(img0[0, 0].cpu().numpy(), cmap='gray')
            # axes[0].set_title('Image 0')
            # axes[1].imshow(img1[0, 0].cpu().numpy(), cmap='gray')
            # axes[1].set_title('Image 1')
            # plt.show()
            

            batch = {'image0': img0, 'image1': img1}


            start_time = time.time()
            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()

            end_time = time.time()

            match_time = end_time - start_time

            #print("number of matches with mconf > 0.5: ", np.sum(mconf > 0.5))
            #print("number of matches with mconf > 0.7: ", np.sum(mconf > 0.7))

            scores[idx][2] = np.sum(mconf > 0.7)

            if (idx%1 ==0):
                print("progress: ", idx/len(associations))


            if assoc_prev == assoc1idx or assoc_prev == -1:
                match_times = match_times + match_time
            else:
                search_times.append(match_times)
                #print("added seach time: ", match_times)
                match_times = 0

            assoc_prev = assoc1idx  

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

        # Create the heatmap using imshow
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Match Count')

        # Set labels and title
        plt.xlabel('Traj 1 (m)')
        plt.ylabel('Traj 2 (m)')
        plt.title('Batvik Different Seasons Trajectories')

        output_file_path = '/home/annika/Documents/batvik_baselines/test41_test42_loftr.pickle'

        # Save all descriptors in a single pickle file

        with open(output_file_path, 'wb') as output_file:
            pickle.dump(heatmap_data, output_file)

        print(f"Saved all descriptors to {output_file_path}")

        # Show the plot
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    main(args)
