import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import matplotlib.colors as cl
import matplotlib.cm as cm
import networkx as nx
import itertools as it
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
        fig.savefig(loc+'testfigure_'+`round(jval,4)`+'.pdf')

def plot_obs_grid(SAVE_PATH, NAME_TRJ, NAME_CNS):

    """
    Loads the dataframe NAME of postprocessed data
    from the location given by SAVE_PATH
    and plots a heatmap for each of the layers of data
    in it.

    Parameters:
    -----------
    SAVE_PATH : string
        the location where the processed data is saved
    NAME_TRJ : string
        the name of the datafile containing the processed
        trajectory data
    NAME_CNS : string
        the name of the datafile containing the processed
        consensus data
    """

    print 'plotting observable grid'

    trj_data = np.load(SAVE_PATH + NAME_TRJ)
    cns_data = np.load(SAVE_PATH + NAME_CNS)
    
    parameter_levels = [list(p.values) for p in trj_data.index.levels[2:-2]]
    parameter_level_names = [name for name in trj_data.index.names[2:-2]]
    levels = tuple(i+2 for i in range(len(parameter_levels)))

    parameter_combinations = list(it.product(*parameter_levels))

    
    for p in parameter_combinations:
        print '(p, levels)', p, levels
        trj_d_slice = trj_data.xs(key=p, level=levels).dropna()
        cns_d_slice = cns_data.xs(key=p, level=levels)
        save_name = zip(parameter_level_names, p)

        plot_observables(trj_d_slice, cns_d_slice, SAVE_PATH, save_name)

def plot_observables(t_data_in, c_data_in, loc, save_name):
    """
    function to create a grid of plots of the values of
    the observable for the values of the parameters given in
    indices.


    Parameters:
    -----------
    loc : string
        location of the dataframe containing the simulation
        output
    t_name: string
        name of the experiments trajectory data file
    c_name: string
        name of the experiments consensus data file
    indices: dict
        dictionary of the indices that are used for the grid
        of plots of the values of observable
    """

    #some parameters:
    labelsize_1 = 12
    labelsize_2 = 30
    bgcolor_alpha = 0.3

    
    print 'plotting observable grids'

    #clean trajectory data
    t_data = t_data_in.where(t_data_in<10**300, np.nan)

    #split consensus data
    c_values = c_data_in['<mean_consensus_state>'] 
    c_times  = c_data_in['<mean_consensus_time>']
    c_times_min = c_data_in['<min_consensus_time>']
    c_times_max = c_data_in['<max_consensus_time>']
    c_times_nanmax = c_data_in['<nanmax_consensus_time>']
    c_times_sem = c_data_in['<sem_consensus_time>']

    norm = cl.Normalize(vmin = 0, vmax = 1)
    cmap = cm.get_cmap('RdBu')
    cmap.set_under('yellow')

    #get the values of the first two index levels
    ivals = t_data.index.levels[0]
    jvals = t_data.index.levels[1]

    ind_names = t_data.index.names
    
    #get the values of the last index level 
    #which will be transformed into colums by unstack()
    observables = t_data.index.levels[-1]
    
    for o, observable in enumerate(observables):
        print 'plotting ' + observable + ' to grid'

        #create figure with enough space for ivals*columns plots
        #plus colormap at the side
        fig = mp.figure(figsize = (4*len(ivals), 4*len(jvals)))
        axes = []

        for i, ival in enumerate(ivals):
            
            #extract the mean and std data for each of the values of the first index
            #and the given observable. Drop missing data and use second index as columns
            subset = t_data.xs(  
                    key=(ival, observable),
                    axis=0,
                    level=(0,-1))['<mean_trajectory>']\
                            .unstack(0)\
                            .dropna(axis=0, how='any')
            subset_errors = t_data.xs(
                    key=(ival, observable),
                    axis=0,
                    level=(0,-1))['<sem_trajectory>']\
                            .unstack(0)\
                            .dropna(axis=0, how='any')
            for j, jval in enumerate(jvals):

                #take absolute values of data but save intervals on which original 
                #data was negative (paint them afterwards)
                subset_j = subset.loc[1:, jval]
                subset_j_errors = subset_errors.loc[1:, jval]
                subset_abs = subset_j
                paint = subset_j.where(subset_j<0).dropna().index.values
                
                #add a subplot to the list of axes
                axes.append(mp.subplot2grid((len(ivals), len(jvals)+1), (len(ivals)-i-1,j)))

                #plot mean trajectory
                subset_abs.plot(ax = axes[-1])

                #plot standard error
                mp.fill_between(subset_abs.index, 
                        subset_abs - subset_j_errors,
                        subset_abs + subset_j_errors,
                        where = subset_abs - subset_j_errors > 0,
                        color='blue',
                        alpha=0.1)
                (subset_abs - subset_j_errors).plot(ax = axes[-1], color='blue', alpha=0.3)
                (subset_abs + subset_j_errors).plot(ax = axes[-1], color='blue', alpha=0.3)

                #plot consensus time with sem and min and max values
                
                c_time = c_times.loc[(ival,jval)]
                c_time_min = c_times_min.loc[(ival,jval)]
                c_time_max = c_times_max.loc[(ival,jval)]
                c_time_nanmax = c_times_nanmax.loc[(ival,jval)]
                c_time_sem = c_times_sem.loc[(ival,jval)]

                mp.axvline(c_time, color = 'grey')
                mp.axvline(c_time_min, ls = 'dashed', color = 'grey')
                mp.axvline(c_time_max, ls = 'dashed', color = 'grey')
                mp.axvline(c_time_nanmax, ls = 'dotted', color = 'grey')
                mp.axvspan(c_time - c_time_sem, c_time + c_time_sem, \
                        alpha = 0.2, color = 'grey')

                #mark areas where data was negative
                if len(paint)>0:
                    mp.axvspan(paint[0], paint[-1], alpha=0.2, color = 'red')

                #change y axis scale to 'log' for plots with nonzero data
                if sum(np.sign(subset_abs))>0:
                    axes[-1].set_yscale('log', nonposy='mask')

                #add tau and phi values to rows and columns
                if i==len(ivals)-1: mp.title(ind_names[1] + ' = ' + `round(jval,4)`, 
                        fontsize=labelsize_2)
                if j==0: mp.ylabel(ind_names[0] + ' = ' + `round(ival,4)`,
                        fontsize=labelsize_2)

                #remove x axis for all but the lowest row
                if i!=0: 
                    axes[-1].xaxis.set_visible(False)

                #remove axis grids for all rows
                axes[-1].grid(b=False)

                #set subplot background according to consensus oppinion state
                axes[-1].patch.set_facecolor(cmap(norm(c_values.loc[(ival,jval)])))
                axes[-1].patch.set_alpha(bgcolor_alpha)
                
        #plot colorbar
        cbar_ax = mp.subplot2grid(
                (len(ivals), len(jvals)+1),
                (0,len(jvals)),
                rowspan=len(ivals))
        sm = cm.ScalarMappable(cmap=cmap,norm=norm)
        sm._A = []
        cbar = fig.colorbar(cax=cbar_ax, mappable=sm, alpha=bgcolor_alpha)
        cbar.ax.tick_params(labelsize = labelsize_2)

        #adjust the grid layout to avoid overlapping plots and save the figure
        fig.tight_layout()
        fig.savefig(loc+'testfigure_'+observable+`save_name[0]`.strip('()').replace(', ', '=')+'.png', facecolor=fig.get_facecolor())
        fig.clf()
        mp.close()

if __name__ == '__main__':
	loc = sys.argv[1]
        plot_trajectories(loc, 'phi')
