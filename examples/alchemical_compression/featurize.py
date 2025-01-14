from concurrent.futures import ProcessPoolExecutor
import pickle
import itertools
import numpy as np
from uf3.data import io
from uf3.data import composition
from uf3.representation import bspline
from uf3.representation import process


element_list = ['Ta','W', 'Mo', 'Nb', 'V']
degree = 3
chemical_system = composition.ChemicalSystem(element_list=element_list,
                                             degree=degree)
print(chemical_system)
print()


# same knot construction for all interactions of a given order
r_min_map = {pair: 0.0 for pair in chemical_system.interactions_map[2]}
r_min_map.update( {trio: [0.0, 0.0, 0.0] for trio in chemical_system.interactions_map[3]} )
r_max_map = {pair: 6.0 for pair in chemical_system.interactions_map[2]}
r_max_map.update( {trio: [5.0, 5.0, 10.0] for trio in chemical_system.interactions_map[3]} )
resolution_map = {pair: 32 for pair in chemical_system.interactions_map[2]}
resolution_map.update( {trio: [10, 10, 20] for trio in chemical_system.interactions_map[3]} )
trailing_trim = 3
leading_trim = 0

n_cores = 32
data_filenames = ["../db_HEA_v2.xyz"]
data_prefixes = ["db_HEA_v2"]
features_filename = "df_features.h5"
bspline_config_filename = "bspline_config.pkl"
table_template = "features_{}"
sparse_hdf5 = True  # store in sparse format

#==============================================================================

bspline_config = bspline.BSplineBasis(chemical_system,
                                      r_min_map=r_min_map,
                                      r_max_map=r_max_map,
                                      resolution_map=resolution_map,
                                      leading_trim=leading_trim,
                                      trailing_trim=trailing_trim)

# EXPLICIT NO SYMMETRY
for trio in bspline_config.interactions_map.get(3, []):
    bspline_config.symmetry[trio] = 1
bspline_config.update_basis_functions()

with open(bspline_config_filename, "wb") as f:
    pickle.dump(bspline_config, f)
print(bspline_config)
print()

# Load the data
print("Loading data...")
data_coordinator = io.DataCoordinator()
for data_filename, data_prefix in zip(data_filenames, data_prefixes):
    data_coordinator.dataframe_from_trajectory(data_filename,
                                               prefix=data_prefix)
df_data = data_coordinator.consolidate()
print("Number of energies:", len(df_data))
print("Number of forces:", int(np.sum(df_data["size"]) * 3))
print(df_data.head())
print(df_data.tail())
print(f"fx of 1st structure in {data_prefixes[0]}:")
print(df_data.loc[data_prefixes[0] + '_0']['fx'])
print()

representation = process.BasisFeaturizer(bspline_config)
client = ProcessPoolExecutor(max_workers=n_cores)
print("Featurizing data...")
representation.batched_to_hdf(features_filename,
                              df_data,
                              client,
                              n_jobs = n_cores,
                              batch_size=50,
                              progress="bar",
                              table_template=table_template,
                              sparse_hdf5=sparse_hdf5,
                              )
print("Done featurizing data.\n")
