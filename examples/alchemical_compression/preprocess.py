import pickle
import numpy as np
from uf3.alchemy import alchemy


bspline_config_filename = "./bspline_config.pkl"
features_filename = "./df_features.h5"
sparse_hdf5 = True  # same value as in featurize.py

# to create train_keys
data_prefixes = ["db_HEA_v2"]
data_lengths = [2859]  # use all structures
#training_idx_files = ["../../../../training_idx.txt"]

learning_weight = 0.3  # energy vs force weight
USE_MPI = False

#==============================================================================
with open(bspline_config_filename, "rb") as file:
    bspline_config = pickle.load(file)
model = alchemy.AlchemicalModel(bspline_config,
                                n_pseudo=2,  # dummy value; not necessary for preprocessing
                                )
train_keys = [data_prefix + "_" + str(i)
for data_prefix, data_len in zip(data_prefixes, data_lengths)
            for i in range(data_len)]
#training_idx_list = []
#for training_idx_file in training_idx_files:
#    with open(training_idx_file, 'r') as f:
#        lines = f.readlines()
#        lines = [line.strip() for line in lines]
#    training_idx_list.append(lines)
#train_keys = [data_prefix + "_" + str(i)
#              for data_prefix, training_idx in zip(data_prefixes, training_idx_list)
#              for i in training_idx]
model.preprocess_for_training(features_filename,
                              train_keys,
                              weight=learning_weight,
                              energy_key="energy",
                              progress="bar",
                              sparse_hdf5=sparse_hdf5,
                              USE_MPI=USE_MPI,
                              )
