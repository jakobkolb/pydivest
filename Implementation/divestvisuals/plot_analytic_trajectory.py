"""
Plot analytic trajectories.
"""
import matplotlib.pyplot as plt
import numpy as np

path = '/home/jakob/ownCloud/Documents/PhD/Project_Divestment/Implementation/' \
       'divestdata/analytic_results/analytic_trajectory.pkl'

data = np.load(path)
trj = data['e_trajectory']
indices = list(trj.columns)

fig = plt.figure()

ax1 = fig.add_subplot(221)
trj[indices[0:3]].plot(ax=ax1)

ax2 = fig.add_subplot(222)
trj[indices[3:7]].plot(ax=ax2)

print trj[[7]]
ax3 = fig.add_subplot(223)
trj[[7]].plot(ax=ax3)

ax4 = fig.add_subplot(224)
trj[[8]].plot(ax=ax4)

plt.show()
