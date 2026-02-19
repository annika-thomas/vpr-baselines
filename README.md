# Perception Baselines



## Getting started

This repo takes in filenames for images from the Batvik dataset, saves SIFT and ORB features, and matches those features using RANSAC. 

## Clone this repo

- [ ] [Clone](https://gitlab.com/mit-acl/lab/perception-baselines.git)

# To get SIFT features

Ensure that the config file is set up for your data location. 

For Batvik data, run `get_SIFT_feats.py` with the following command line argument with your dataset of choice:

```
python3 get_SIFT_feats.py --config=config/batvik.yml --dataset_1=27
```

# To get ORB features

Ensure that the config file is set up for your data location. 

For Batvik data, run `get_ORB_feats.py` with the following command line argument with your dataset of choice:

```
python3 get_ORB_feats.py --config=config/batvik.yml --dataset_1=27
```

# Matching SIFT features

To run all the experiments set up in the `matching_config.yml` file, run `match_SIFT_exps.py`

```
python3 match_SIFT_exps.py --config=config/batvik.yml
```

# Matching ORB features

To run all the experiments set up in the `matching_config.yml` file, run `match_ORB_exps.py`

```
python3 match_ORB_exps.py --config=config/batvik.yml
```
