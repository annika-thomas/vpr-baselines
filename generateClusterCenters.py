from utilities import *
import os
import requests
import shutil
import numpy as np
import torch
from torch import nn
import transformers as hft
from torch.nn import functional as F
import einops as ein
import fast_pytorch_kmeans as fpk
import faiss
import faiss.contrib.torch_utils
import random
import os
from PIL import Image
from sklearn.decomposition import PCA
from typing import Union, List, Tuple, Literal
from helper_functions import *
import argparse
from utils import readConfig
import csv
from torchvision import transforms as tvf

import matplotlib.pyplot as plt

def import_csv(file_path):
    data = []
    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(row)
    return data

def main(args):
    print("testing")

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

    # ---------------------------
    # BUILDING GLOBAL DESCRIPTORS
    # ---------------------------

    # Program parameters
    save_dir = os.path.join(output_directory, "cluster_centers")
    device = torch.device("cuda")
    # Dino_v2 properties (parameters)
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    num_c: int = 32
    # Domain for use case (deployment environment)
    domain: Literal["aerial", "indoor", "urban"] = "aerial"
    # Maximum image dimension
    max_img_size: int = 1024

    # ---------------------------
    # DINO V2 Extractor
    # ---------------------------

    # DINO extractor
    # if "extractor" in globals():
    #     print(f"Extractor already defined, skipping")
    # else:
    #     extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
    #         desc_facet, device=device)
    # # Base image transformations
    # base_tf = tvf.Compose([
    #     tvf.ToTensor(),
    #     tvf.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='get_ORB_feats.py')

    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument('--dataset_1', required=True, help="Name of dataset to use")

    args = parser.parse_args()
    main(args)


