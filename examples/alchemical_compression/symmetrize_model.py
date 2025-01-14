from uf3.regression import least_squares
from uf3.representation import bspline
from uf3.util import json_io
import copy
import numpy as np

in_model_filename = "./model.json"
out_model_filename = "./model_sym.json"

# level 1: symmetrize parameters but leave in expanded form
# level 2: symmetrize parameters and compress
level = 2

#===============================================================================

#model = least_squares.WeightedLinearModel.from_json(model_filename)
# EXPLICIT LOADING
dump = json_io.load_interaction_map(in_model_filename)
bspline_config = bspline.BSplineBasis.from_dict(dump)
bspline_config_default = copy.deepcopy(bspline_config)
for trio in bspline_config.interactions_map.get(3, []):
    bspline_config.symmetry[trio] = 1
bspline_config.update_basis_functions()
regularizer = dump.get("regularizer", 0)  # we don't want it to create the huge reg matrix
data_coverage = dump.get("data_coverage", None)
model = least_squares.WeightedLinearModel(bspline_config,
                                          regularizer=regularizer,
                                          data_coverage=data_coverage)
model.load(solution=dump)

# Symmetrize the model
trio_list = bspline_config.interactions_map.get(3, [])
component_sizes, component_offsets = bspline_config.get_interaction_partitions()
for trio in trio_list:
    if bspline_config_default.symmetry[trio] == 1:
        continue
    idx_lo = component_offsets[trio]
    idx_hi = idx_lo + component_sizes[trio]
    coefficients_slice = model.coefficients[idx_lo:idx_hi]
    data_coverage_slice = model.data_coverage[idx_lo:idx_hi]
    grid_coeff = bspline_config.decompress_3B(coefficients_slice, trio)
    grid_dc = bspline_config.decompress_3B(data_coverage_slice, trio)
    if bspline_config_default.symmetry[trio] == 2:
        grid_coeff = (grid_coeff + grid_coeff.transpose(1, 0, 2)) / 2
        grid_dc = np.logical_or(grid_dc, grid_dc.transpose(1, 0, 2))
    elif bspline_config_default.symmetry[trio] == 3:
        grid_coeff = (grid_coeff + 
                      grid_coeff.transpose(0, 2, 1) +
                      grid_coeff.transpose(1, 0, 2) +
                      grid_coeff.transpose(1, 2, 0) +
                      grid_coeff.transpose(2, 0, 1) +
                      grid_coeff.transpose(2, 1, 0)) / 6
        grid_dc = np.logical_or.reduce([grid_dc,
                                        grid_dc.transpose(0, 2, 1),
                                        grid_dc.transpose(1, 0, 2),
                                        grid_dc.transpose(1, 2, 0),
                                        grid_dc.transpose(2, 0, 1),
                                        grid_dc.transpose(2, 1, 0)
                                        ])
    else:
        raise ValueError("Invalid symmetry")
    model.coefficients[idx_lo:idx_hi] = bspline_config.compress_3B(grid_coeff, trio, fitting=False)
    model.data_coverage[idx_lo:idx_hi] = bspline_config.compress_3B(grid_dc, trio, fitting=False)

if level > 1:
    # Compress the model
    model_compressed = least_squares.WeightedLinearModel(bspline_config_default,
                                                         regularizer=regularizer,
                                                         data_coverage=None)
    component_sizes_compressed, component_offsets_compressed = bspline_config_default.get_interaction_partitions()
    n_params = np.sum(bspline_config_default.get_feature_partition_sizes())
    model_compressed.coefficients = np.zeros(n_params)
    model_compressed.data_coverage = np.zeros(n_params, dtype=bool)

    # Copy over 1- and 2-body parameters
    assert component_offsets[trio_list[0]] == component_offsets_compressed[trio_list[0]]  # should have same number of 1- and 2-body parameters
    trio_start_idx = component_offsets[trio_list[0]]
    model_compressed.coefficients[:trio_start_idx] = model.coefficients[:trio_start_idx]
    model_compressed.data_coverage[:trio_start_idx] = model.data_coverage[:trio_start_idx]

    # Copy over 3-body parameters
    for trio in trio_list:
        idx_lo = component_offsets[trio]
        idx_hi = idx_lo + component_sizes[trio]
        idx_lo_compressed = component_offsets_compressed[trio]
        idx_hi_compressed = idx_lo_compressed + component_sizes_compressed[trio]
        coefficients_slice = model.coefficients[idx_lo:idx_hi]
        data_coverage_slice = model.data_coverage[idx_lo:idx_hi]
        if bspline_config_default.symmetry[trio] == 1:
            model_compressed.coefficients[idx_lo_compressed:idx_hi_compressed] = coefficients_slice
            model_compressed.data_coverage[idx_lo_compressed:idx_hi_compressed] = data_coverage_slice
        else:
            grid_coeff = bspline_config.decompress_3B(coefficients_slice, trio)
            grid_dc = bspline_config.decompress_3B(data_coverage_slice, trio)
            model_compressed.coefficients[idx_lo_compressed:idx_hi_compressed] = bspline_config_default.compress_3B(grid_coeff, trio, fitting=False)
            model_compressed.data_coverage[idx_lo_compressed:idx_hi_compressed] = bspline_config_default.compress_3B(grid_dc, trio, fitting=False)
    model = model_compressed

model.to_json(out_model_filename)
