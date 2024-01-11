from uf3.regression import least_squares

import numpy as np
import matplotlib.pyplot as plt


nModels = 20

coefficients = dict()
data_coverages = dict()
for i in range(0, nModels):
    model_file = "model_" + str(i) + ".json"
    model = least_squares.WeightedLinearModel.from_json(model_file)
    coefficients[i] = model.coefficients
    data_coverages[i] = model.data_coverage

coefficient_change = []
for i in range(1, nModels):
    change = np.abs((coefficients[i] - coefficients[i-1])/np.minimum(coefficients[i], coefficients[i-1]))
    change = change[data_coverages[i]]  # only coefficients with data coverage
    coefficient_change.append( (i, max(change)) )

max_value = max(change[np.where(~np.isinf(change))])
change = np.where(np.isinf(change), max_value, change)  # just for plotting

# plot
fig, ax = plt.subplots()
ax.plot(*zip(*coefficient_change))
ax.set_xlabel("Model Number")
ax.set_ylabel("Maximum Coefficient Change")
ax.set_yscale("log")
plt.show()
