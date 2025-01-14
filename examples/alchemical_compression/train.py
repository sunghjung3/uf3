import pickle
import numpy as np
from uf3.alchemy import alchemy


bspline_config_filename = "./bspline_config.pkl"
n_pseudo = {2: 3, 3: 5}

C_regularization = {'ridge_1b': 0.0, 'ridge_2b': 1e-10, 'ridge_3b': 1e-7,
                    'curvature_2b': 1e-8, 'curvature_3b': 1e-10,}
max_iter = 20
model_filename = "model.json"
#init_params = np.load("../pseudo_3-5_L2/checkpoint/0/alchemical_model_params.npz")
init_params = None
checkpoint = 1  # frequency of caching intermediate models to file
resume = False

# defaults from preprocessing
preprocessed_file = "./preprocessed.h5"
coverage_file = "./coverage.npz"
metadata_file = "./metadata.npz"

#==============================================================================

with open(bspline_config_filename, "rb") as file:
    bspline_config = pickle.load(file)
C_regularizers = alchemy.get_C_regularizers(bspline_config,
                                            **C_regularization,
                                            )

model = alchemy.AlchemicalModel(bspline_config,
                                n_pseudo,
                                )
print("Training...")
model.fit_from_file(preprocessed_file=preprocessed_file,
                    coverage_file=coverage_file,
                    metadata_file=metadata_file,
                    init_params=init_params,
                    C_regularizers=C_regularizers,
                    C_reg_free=True,
                    max_iter=max_iter,
                    checkpoint=checkpoint,
                    resume=resume,
                    )
print("Finished training!")
model.to_json(model_filename)
