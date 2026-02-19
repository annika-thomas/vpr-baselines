import argparse
from utils import readConfig
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


def threshold_matrix(matrix, threshold):
    # Create a new matrix with the same dimensions as the input matrix
    result_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    # Iterate through the input matrix and apply the threshold condition
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > threshold:
                result_matrix[i][j] = True

    return result_matrix

def main(args):
    

    gtFilePath = '/home/annika/Documents/highbay_gt/sam1_highbay_gt_11_vs_11.pickle'
    intersectionAreas_gt, unionAreas_gt, iou_gt = loadGtResultsFromFile(gtFilePath)

    matchingOutputFilepath = '/home/annika/Documents/batvik_gt/test39_test41_orb.pickle'
    correspondencesFound = loadDataFromMatchingPickle(matchingOutputFilepath)
    print(correspondencesFound)

    thresholdedCorrespondences = threshold_matrix(correspondencesFound, 10)

    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, sharex=True, sharey=True)
    fig.suptitle('Test11')
    ax1.imshow(correspondencesFound)
    ax1.set_title("Correspondences found")
    ax2.imshow(thresholdedCorrespondences)
    ax2.set_title("Correspondences found (thresholded)")

    recallLimit = 0.25
    precisionLimit = 0.05

    # recall_mask = iou_gt > recallLimit
    # precision_mask = iou_gt < precisionLimit

    # ax3.imshow(recall_mask)
    # ax3.set_title("recall mask")
    # ax4.imshow(precision_mask)
    # ax4.set_title("precision mask")

    # # RECALL:
    # # Convert 1s to True and 0s to False in both matrices
    # recall_matrix = recall_mask.astype(bool)
    # correspondences_matrix = correspondencesFound.astype(bool)

    # # Find the positions where recall_matrix is True and correspondences_matrix is False
    # positions = (recall_matrix == True) & (correspondences_matrix == True)

    # # Count the number of such positions
    # count = np.count_nonzero(positions)

    # # Count the total number of True values in the recall matrix
    # total_recall_trues = np.count_nonzero(recall_matrix)
    # print("Total recall trues: ", total_recall_trues)
    # print(124*124)

    # # Calculate the percentage
    # percentage = (count / total_recall_trues) * 100

    # print("Recall: ", percentage)

    # # PRECISION: 
    # # Convert 1s to True and 0s to False in both matrices
    # precision_matrix = precision_mask.astype(bool)

    # print("Precision matrix: ", precision_matrix)

    # positions_precision = (precision_matrix == False) & (correspondences_matrix == True)

    # # Count the number of such positions
    # count_precision = np.count_nonzero(positions_precision)

    # # Count the total number of True values in the correspondences matrix
    # total_precision_trues = np.count_nonzero(correspondences_matrix)
    # print("Total precision trues: ", total_precision_trues)

    # # Calculate the percentage
    # percentage_precision = (count_precision / total_precision_trues) * 100

    # print("Precision: ", percentage_precision)


    plt.show()


def loadDataFromMatchingPickle(pathToFile):
    file = open(pathToFile, "rb")
    correspondencesFound = pickle.load(file)
    file.close()
    return correspondencesFound

def loadGtResultsFromFile(inputFilename):
    file = open(inputFilename, "rb")
    intersectionAreas, unionAreas, iou = pickle.load(file)
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