import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import networkx as nx
import sys


def plot_economy(output_loc):

	with open(output_loc+"_trajectory", 'r') as myfile:
		labels = myfile.readline().strip().split(',')
		print labels

	output_data = np.loadtxt(output_loc+"_trajectory", delimiter = ',', skiprows = 2)


	tstart = 0
	tend = -1

	fig = mp.figure(figsize = (11.69,8.27), dpi=100)
	#fig.suptitle(header)
	fig.suptitle('Solow Swan model with two sectors and heterogeneous households \n coupled to oppinion dynamics for savings decision')

	ax1 = fig.add_subplot(221)
	ax1.set_title('capital returns & wage')
	ln11 = ax1.plot(output_data[tstart:,0], output_data[tstart:,2], 'r',  label = labels[2])
	ln12 = ax1.plot(output_data[tstart:,0], output_data[tstart:,3], 'g', label = labels[3])


	ax2 = ax1.twinx()
	ln13 = ax2.plot(output_data[tstart:,0], output_data[tstart:,1], 'b', label = labels[1])

	lns = ln11 + ln12 + ln13
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc=0)

	ax21 = fig.add_subplot(222)
	ax21.set_title('Resource stock & extraction cost')
	ln21 = ax21.plot(output_data[tstart:,0], output_data[tstart:,8], 'r', label = labels[8])

	ax22 = ax21.twinx()
	ln22 = ax22.plot(output_data[tstart:,0], output_data[tstart:,15], 'b', label = labels[15])

	lns = ln21 + ln22
	labs = [l.get_label() for l in lns]
	ax21.legend(lns, labs, loc=0)


	ax3 = fig.add_subplot(224)
	ax3.set_title('accumulated sector capital')


	ax3.plot(output_data[tstart:,0], output_data[tstart:,4], label = labels[4], zorder = 2)
	ax3.plot(output_data[tstart:,0], output_data[tstart:,5], label = labels[5], zorder = 2)
	ax3.plot(output_data[tstart:,0],(output_data[tstart:,4] + output_data[tstart:,5])/2., zorder = 2)
	df = pd.DataFrame(index=output_data[tstart:,0])
	df['A'] = output_data[tstart:,16]
	mpbl = ax3.pcolorfast(ax3.get_xlim(), ax3.get_ylim(), df['A'].values[np.newaxis], cmap='RdYlGn', alpha = .3, vmin=-1, vmax=1,  zorder = 1)
	#mpbl =  ax3.get_children()[5]
	cbar = mp.colorbar(mpbl, ticks=[-1, 0, 1])
	cbar.ax.set_yticklabels(['no \n consensus', 'dirty \n consensus', 'clean \n consensus'])

	ax3.legend(loc=0)


	market_volume = output_data[tstart:,9] + output_data[tstart:,10]
	ax5 = fig.add_subplot(223)
	ax5.set_title('relative market share')
	ax5.plot(output_data[tstart:,0], output_data[tstart:,9]/market_volume, label = labels[9])
	ax5.plot(output_data[tstart:,0], output_data[tstart:,10]/market_volume, label = labels[10])
	#ax5.plot(output_data[tstart:,0], output_data[tstart:,11]/output_data[tstart:,9], label = labels[11])
	#ax5.plot(output_data[tstart:,0], output_data[tstart:,13]/output_data[tstart:,9], label = labels[13])

	ax5.legend(loc=0)

	#ax6 = fig.add_subplot(326)
	#ax6.set_title('factor productivities dirty')
	#ax6.plot(output_data[tstart:,0], output_data[tstart:,10], label = labels[10])
	#ax6.plot(output_data[tstart:,0], output_data[tstart:,12]/output_data[tstart:,10], label = labels[12])
	#ax6.plot(output_data[tstart:,0], output_data[tstart:,14]/output_data[tstart:,10], label = labels[14])
	#ax6.plot(output_data[tstart:,0], output_data[tstart:,15]/output_data[tstart:,10], label = labels[15])

	#ax6.legend(loc=0)	


	fig.savefig(output_loc+'.pdf', papertype='a4', orientation='portrait')

def plot_network(loc):

	adjacency = np.loadtxt(loc+'_network')
	labels 	  = np.loadtxt(loc+'_labels')

	fig2 = mp.figure()


	G = nx.DiGraph(adjacency)
	from networkx.drawing.nx_agraph import graphviz_layout
	layout = graphviz_layout(G) 
	nx.draw_networkx_nodes(G, pos=layout, node_color=labels, node_size=20)
	nx.draw_networkx_edges(G, pos=layout, width=.2)

	fig2.savefig(loc+'_network_plot')


if __name__ == '__main__':
	loc = sys.argv[1]
	plot_economy(loc)
	plot_network(loc)
