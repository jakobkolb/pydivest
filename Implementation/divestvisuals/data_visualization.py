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

def plot_trajectories(loc, name, params, indices):
    
    print 'plotting trajectories'

    with open(loc+name) as target:
        tmp = np.load(target).replace([np.inf, -np.inf], np.nan)
    dataframe = tmp.where(tmp<10**300, np.nan)

    #get the values of the first two index levels
    ivals = dataframe.index.levels[0]
    jvals = dataframe.index.levels[1]
    
    #get the values of the last index level 
    #which will be transformed into colums by unstack()
    columns = dataframe.index.levels[-1]

    for j, jval in enumerate(jvals):

        #create figure with enough space for ivals*columns plots
        fig = mp.figure(figsize = (4*len(ivals), 2*len(columns)))
        axes = []

        for i, ival in enumerate(ivals):
            
            #extract the mean and std data for each of the values of the first index
            subset = dataframe.loc[(ival, jval,), 
                    '<mean_trajectory>'].unstack('observables').dropna(axis=0, how='any')
            subset_errors = dataframe.loc[(ival, jval,), 
                    '<sem_trajectory>'].unstack('observables').dropna(axis=0, how='any')
            for c, column in enumerate(columns):

                #take absolute values of data but save intervals on which original 
                #data was negative (paint them afterwards)
                subset_j = subset.loc[1:, column]
                subset_j_errors = subset_errors.loc[1:, column]
                subset_abs = subset_j.abs()
                paint = subset_j.where(subset_j<0).dropna().index.values
                
                #add a subplot to the list of axes
                axes.append(mp.subplot2grid((len(columns), len(ivals)), (c,i)))
                #plot mean trajectory and insequrity interval
                #also mark the area in which the mean data was negative
                subset_abs.plot(ax = axes[-1])
                mp.fill_between(subset_abs.index, 
                        subset_abs - subset_j_errors,
                        subset_abs + subset_j_errors,
                        where = subset_abs - subset_j_errors > 0,
                        color='blue',
                        alpha=0.1)
                (subset_abs - subset_j_errors).plot(ax = axes[-1], color='blue', alpha=0.3)
                (subset_abs + subset_j_errors).plot(ax = axes[-1], color='blue', alpha=0.3)
                if len(paint)>0:
                    mp.axvspan(paint[0], paint[-1], alpha=0.2, color = 'red')

                #change y axis scalling to 'log' for plots with nonzero data
                if sum(np.sign(subset_abs))>0:
                    axes[-1].set_yscale('log', nonposy='mask')

                #adjust locator counts and add y axis label
                mp.locator_params(axis='x', tight=True, nbins=5)
                if i==0: mp.ylabel(column)
                if c!=len(columns)-1: 
                    axes[-1].xaxis.set_visible(False)
                

        #adjust the grid layout to avoid overlapping plots and save the figure
        fig.tight_layout()
        fig.savefig(loc+'testfigure_'+`jval`+'.pdf')



if __name__ == '__main__':
	loc = sys.argv[1]
        plot_trajectories(loc, 'phi')
