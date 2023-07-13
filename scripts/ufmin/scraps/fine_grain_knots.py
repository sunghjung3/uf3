from uf3.representation import bspline
from uf3.util import json_io

import numpy as np

import math, bisect


# user input
knot_strategy_base = "linear"
r_min_base = 0.1
r_max_base = 6.0
n_intervals_base = 16
knot_strategy_fine = "linear"
r_min_fine = 2.0
r_max_fine = 2.5
resolution_fine = 0.02
knots_path = "knots.json"
interaction = "Pt-Pt"


# generate base map (coarse)
knot_spacer_base = bspline.get_knot_spacer(knot_strategy_base)
base_map = knot_spacer_base(r_min_base, r_max_base, n_intervals_base)

# generate fine map in desired region
knot_spacer_fine = bspline.get_knot_spacer(knot_strategy_fine)
n_intervals_fine = math.ceil((r_max_fine - r_min_fine) / resolution_fine)
fine_map = knot_spacer_fine(r_min_fine, r_max_fine, n_intervals_fine, sequence=False)

# insert fine map into base map
insert_index = bisect.bisect_left(base_map, r_min_fine)
highest_remove_index = bisect.bisect_right(base_map, r_max_fine)

base_map = np.delete(base_map, range(insert_index, highest_remove_index))
base_map = np.insert(base_map, insert_index, fine_map)

# write map to file
base_map_dict = dict()
base_map_dict[interaction] = base_map
json_io.dump_interaction_map(base_map_dict, filename=knots_path, write=True)
