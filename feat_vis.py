import csv
import os
import cv2
from helper_functions import *

def main():

    filePath1 = '/home/annika/Documents/multico/41/camera/image_00000810.jpg'
    filePath2 = '/home/annika/Documents/multico/42/camera/image_00000822.jpg'
    #filePath2 = '/home/annika/Documents/multico/27/camera/image_00000621.jpg'

    img1 = cv2.imread(filePath1, 0)
    img2 = cv2.imread(filePath2, 0)

    score = edge_match_orb(img1, img2)
    score = edge_match_sift(img1, img2)


if __name__ == "__main__":
    main()