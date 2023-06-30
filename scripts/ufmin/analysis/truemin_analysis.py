import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


with open("true_true_calc.pckl", 'rb') as f:
    E, F = pickle.load(f)

print("# of steps:", len(E))
print("energy:", E)

fmax = list()
for f in F:
    force = np.sum( np.square(f), axis=1 )
    fmax.append( np.sqrt( np.max(force) ) )
print("f_max:", fmax)


fig, axE = plt.subplots()
color = 'tab:blue'
axE.plot(E, color=color)
axE.set_xlabel("Step", color=color)
axE.set_ylabel("Energy (eV)", color=color)
axE.tick_params(axis ='y', labelcolor = color, which = 'minor')
#axE.set_yscale("log")
#axE.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

axF = axE.twinx()
color = 'tab:red'
axF.plot(fmax, color=color)
axF.set_ylabel("Max force (eV/A)", color=color)
axF.tick_params(axis ='y', labelcolor = color, which = 'minor')
axF.set_yscale("log")
#axF.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
axF.axhline(y=0.05, color='r', linestyle='--')


plt.title("True Optimization")
plt.show()
