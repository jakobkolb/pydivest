
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import sys


output_loc = sys.argv[1]

with open(output_loc, 'r') as myfile:
	labels = myfile.readline().strip().split('\t')
	header = myfile.readline()
	print labels

output_data = np.loadtxt(output_loc, delimiter = '\t', skiprows = 2)

tstart = 0
tend = -1

fig = mp.figure(figsize = (8.27,11.69), dpi=100)
fig.suptitle(header)
ax3 = fig.add_subplot(111)
ax3.set_title('accumulated sector capital')


ax3.plot(output_data[tstart:,0], output_data[tstart:,4], label = labels[4], zorder = 2)
ax3.plot(output_data[tstart:,0], output_data[tstart:,5], label = labels[5], zorder = 2)
ax3.plot(output_data[tstart:,0],(output_data[tstart:,4] + output_data[tstart:,5])/2., zorder = 2)
df = pd.DataFrame(index=output_data[tstart:,0])
df['A'] = output_data[tstart:,16]
mpbl = ax3.pcolorfast(ax3.get_xlim(), ax3.get_ylim(), df['A'].values[np.newaxis], cmap='RdYlGn', alpha = .3, vmin=-1, vmax=1, zorder = 1)
#mpbl =  ax3.get_children()[5]
cbar = mp.colorbar(mpbl, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['no \n consensus', 'dirty \n consensus', 'clean \n consensus'])

ax3.legend(loc=0)
mp.show()
