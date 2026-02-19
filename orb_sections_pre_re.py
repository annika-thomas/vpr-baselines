import argparse
from utils import readConfig
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import pandas as pd

def main(args):

    matchingConfig = readConfig("config/pre_re_config.yml")

    # Initializing csv for exporting info
    csv_data = []
    header = ['dataset1Name', 'dataset2Name', 'experimentDescription', 'Threshold', 'Recall', 'Precision']
    csv_data.append(header)

    for experimentName in matchingConfig.experiments:

        experimentConfig = matchingConfig.experiments[experimentName]
        experimentDescription = experimentConfig.Experimentdescription

        dataset1Name = experimentConfig.Dataset1
        dataset2Name = experimentConfig.Dataset2
        start_idx_dataset_1 = experimentConfig.FirstIndex1
        end_idx_dataset_1 = experimentConfig.LastIndex1
        start_idx_dataset_2 = experimentConfig.FirstIndex2
        end_idx_dataset_2 = experimentConfig.LastIndex2

        print("Experiment description: ", experimentDescription)
        print("Dataset1: ", dataset1Name)
        print("start_idx_dataset_1: ", start_idx_dataset_1)
        print("end_idx_dataset_1: ", end_idx_dataset_1)
        print("Dataset2: ", dataset2Name)
        print("start_idx_dataset_2: ", start_idx_dataset_2)
        print("end_idx_dataset_2: ", end_idx_dataset_2)

        gtFilePath = f'/home/annika/Documents/batvik_baselines/groundTruthOverlap/batvik_{dataset1Name}_vs_{dataset2Name}.pickle'
        intersectionAreas_gt, unionAreas_gt, iou_gt = loadGtResultsFromFile(gtFilePath)

        matchingOutputFilepath = f'/home/annika/Documents/batvik_baselines/test{dataset1Name}_test{dataset2Name}_orb.pickle'
        correspondencesFound = loadDataFromMatchingPickle(matchingOutputFilepath)
        # print(correspondencesFound)

        precisions = []
        recalls = []

        original_correspondencesFound = correspondencesFound.copy()

        max_correspondences = np.max(original_correspondencesFound).astype(int)

        print("Max correspondences: ", max_correspondences)

        # Create a mask for elements greater than 10
        # mask = correspondencesFound > 0

        # for (i = 0; i < max_correspondences; i=i+5) 

        correspondencesFoundMasked = correspondencesFound

        iou_gt = np.transpose(iou_gt)

        for i in range(0, max_correspondences+1, 2):

            # print("i: ", i)

            mask = original_correspondencesFound > i
            # Replace elements greater than 10 with 1
            correspondencesFoundMasked[mask] = 1
            correspondencesFoundMasked[~mask] = 0

            num_rows, num_cols = correspondencesFoundMasked.shape
            correspondencesFoundSection = correspondencesFoundMasked[:num_rows-1, :num_cols-1]

            # Pick out traverse section
            submatrix_iou_gt = iou_gt[start_idx_dataset_1:end_idx_dataset_1, start_idx_dataset_2:end_idx_dataset_2]
            submatrix_correspondencesFound = correspondencesFoundSection[start_idx_dataset_1:end_idx_dataset_1, start_idx_dataset_2:end_idx_dataset_2]

            # fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, sharex=True, sharey=True)
            # fig.suptitle('Test11')
            # ax1.imshow(submatrix_iou_gt)
            # ax1.set_title("IoU")
            # ax2.imshow(submatrix_correspondencesFound)
            # ax2.set_title("Correspondences")

            recallLimit = 0.333333
            precisionLimit = 0.01

            recall_mask = submatrix_iou_gt > recallLimit
            precision_mask = submatrix_iou_gt < precisionLimit

            # ax3.imshow(recall_mask)
            # ax3.set_title("recall mask")
            # ax4.imshow(precision_mask)
            # ax4.set_title("precision mask")

            # RECALL:
            # Convert 1s to True and 0s to False in both matrices
            recall_matrix = recall_mask.astype(bool)
            correspondences_matrix = submatrix_correspondencesFound.astype(bool)

            # Find the positions where recall_matrix is True and correspondences_matrix is False
            positions = (recall_matrix == True) & (correspondences_matrix == True)

            # Count the number of such positions
            count = np.count_nonzero(positions)

            # Count the total number of True values in the recall matrix
            total_recall_trues = np.count_nonzero(recall_matrix)
            #print("Total recall trues: ", total_recall_trues)
            # print(124*124)

            # Calculate the percentage
            if (total_recall_trues > 0):
                percentage = (count / total_recall_trues) * 100
            else:
                percentage = 0

            #print("Recall: ", percentage)

            # PRECISION: 
            # Convert 1s to True and 0s to False in both matrices
            precision_matrix = precision_mask.astype(bool)

            # print("Precision matrix: ", precision_matrix)

            positions_precision = (precision_matrix == False) & (correspondences_matrix == True)

            # Count the number of such positions
            count_precision = np.count_nonzero(positions_precision)

            # Count the total number of True values in the correspondences matrix
            total_precision_trues = np.count_nonzero(correspondences_matrix)
            #print("Total precision trues: ", total_precision_trues)

            # Calculate the percentage
            if (total_precision_trues > 0):
                percentage_precision = (count_precision / total_precision_trues) * 100
            else:
                percentage_precision = 0

            recalls.append(percentage)  # Convert to percentage

            precisions.append(percentage_precision)  # Convert to percentage

            csv_data.append([dataset1Name, dataset2Name, experimentDescription, i, percentage, percentage_precision])


            #plt.show()

        # Plotting the precision vs recall
        plt.plot(recalls, precisions, marker='o')
        plt.title(f'Precision vs Recall of {experimentDescription}')
        plt.xlabel('Recall (%)')
        plt.ylabel('Precision (%)')
        plt.grid(True)
        plt.show()

    # After your loop and plotting code:
    csv_filename = f'precision_recall_results.csv'

    # Define your output directory
    output_directory = '/home/annika/Documents/batvik_baselines/results_csvs'

    # Create the directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Combine the directory with the filename to create a path
    full_path = os.path.join(output_directory, csv_filename)

    # Write the CSV data to a file using the full path
    with open(full_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

def loadDataFromMatchingPickle(pathToFile):
    file = open(pathToFile, "rb")
    correspondencesFound = pickle.load(file)
    file.close()
    return correspondencesFound

def loadGtResultsFromFile(inputFilename):
    file = open(inputFilename, "rb")
    intersectionAreas, unionAreas, iou, onWater = pickle.load(file)
    file.close()
    return intersectionAreas, unionAreas, iou

if __name__=="__main__":
    parser = argparse.ArgumentParser(
    prog='python3 compareMatchingToGt.py',
    description='Compare matching results to ground truth',
    epilog='Ask Jouko for instructions whenever needed.')

    parser.add_argument('--config', required=True, help="Path to path configuration file")

    args = parser.parse_args()

    main(args)