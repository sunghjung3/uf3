import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


results_file = "results.csv"
resolution_key = "resolution"
learning_weight_key = "lr"
curvature_2b_key = "curvature_2b"


df = pd.read_csv(results_file)
df = df[ df['forcecalls'] != -1 ]  # remove buggy rows
forcecalls = df['forcecalls']


resolutions = list(set(df[resolution_key]))
resolutions.sort()
plot_dim = math.ceil(math.sqrt(len(resolutions)))
fig, axs = plt.subplots(plot_dim, plot_dim)
for i, resolution in enumerate(resolutions):
    key = df[resolution_key] == resolution
    sub_df = df[key]
    ax_idx = [math.floor(i / plot_dim), i % plot_dim]
    try:
        ax = axs[ax_idx[0], ax_idx[1]]
    except IndexError as ie:
        ax = axs[i]
    except TypeError as te:
        ax = axs
    scatter = ax.scatter(sub_df[learning_weight_key], sub_df[curvature_2b_key], s=10/(1 - 0.9 * np.exp(-0.1 * (forcecalls[key] - forcecalls.min()))), c=forcecalls[key], cmap='viridis', vmin=forcecalls.min(), vmax=forcecalls.max())

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("# of forcecalls")
    scatter.set_clim(forcecalls.min(), forcecalls.max())
    ax.set_xlabel(learning_weight_key)
    ax.set_ylabel(curvature_2b_key)
    ax.set_yscale('log')
    ax.set_title(f"resolution {resolution}")
plt.show()
