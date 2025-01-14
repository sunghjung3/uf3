import pickle
import numpy as np
from uf3.alchemy import alchemy


bspline_config_filename = "./bspline_config.pkl"

model_filename = "model.json"
alchemical_params_filename = "checkpoint/xxx/alchemical_model_params.npz"  # change xxx to desired iteration number

coverage_file = "./coverage.npz"

#==============================================================================

with open(bspline_config_filename, "rb") as file:
    bspline_config = pickle.load(file)

alchemical_params = np.load(alchemical_params_filename)
n_pseudo = dict()
for i in range(2, bspline_config.degree+1):
    coeff_ib = alchemical_params[f"coeff_{i}b"]
    n_pseudo[i] = coeff_ib.shape[1]

coverages = np.load(coverage_file)

model = alchemy.AlchemicalModel(bspline_config, n_pseudo)
model.initialize_parameters(alchemical_params)
model.decompress_alchemical_parameters()
model.data_coverage = coverages['data_coverage']
model.to_json(model_filename)
