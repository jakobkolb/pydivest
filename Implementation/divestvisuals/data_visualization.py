import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import networkx as nx
import sys


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

def plot_trajectories(loc, params, indices):

    with open(loc) as target:
        dataframe = np.load(target)

    print dataframe.index.names

    mx = dataframe.index.values
    print mx
    print indices
    
    fig1 = mp.figure()
    subset = dataframe.loc[(0.05, 0.1,), '<mean_trajectory>'].unstack('observables')
    print subset
    subset.plot(fig=fig1)
    #fig2 = mp.figure()
    #mp.plot(range(10), np.random.randint(0,10,10))
    fig1.savefig('testfigure.pdf')



if __name__ == '__main__':
	loc = sys.argv[1]
        plot_trajectories(loc, 'phi')
