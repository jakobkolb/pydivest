import sys
import numpy as np
import networkx as nx
import itertools as it
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm


def plot_tau_phi(SAVE_PATH, NAME, xlog=False,
                 ylog=False, fextension='.png'):
    """
    Loads the dataframe NAME of postprocessed data
    from the location given by SAVE_PATH
    and plots a heatmap for each of the layers of data
    in it.

    Parameters:
    -----------
    SAVE_PATH : string
        the location where the processed data is saved
    NAME : string
        the name of the datafile containing the processed
        data
    xlog : bool
        Wheter x axis is log scaled
    ylog : bool
        Wheter y axis is log scaled
    """

    data = np.load(SAVE_PATH + NAME)

    parameter_levels = [list(p.values) for p in data.index.levels[2:]]
    parameter_level_names = [name for name in data.index.names[2:]]
    levels = tuple(i+2 for i in range(len(parameter_levels)))

    parameter_combinations = list(it.product(*parameter_levels))

    for p in parameter_combinations:
        d_slice = data.xs(key=p, level=levels)
        print parameter_level_names
        save_name = zip(parameter_level_names, [str(x) for x in p])

        for level in list(d_slice.unstack().columns.levels[0]):
            TwoDFrame = d_slice[level].unstack().dropna()
            if TwoDFrame.empty \
                    or len(TwoDFrame.columns.values) <= 1 \
                    or len(TwoDFrame.index.values) <= 1:
                continue
            vmax = np.nanmax(TwoDFrame.values)
            vmin = np.nanmin(TwoDFrame.values)
            TwoDFrame.replace(np.nan, np.inf)
            cmap = cm.get_cmap('RdBu')
            cmap.set_under('yellow')
            cmap.set_over('black')
            fig = explore_Parameterspace(
                    TwoDFrame,
                    title=level,
                    cmap=cmap,
                    norm=cl.Normalize(vmin=0, vmax=vmax),
                    vmin=vmin,
                    vmax=vmax,
                    xlog=xlog,
                    ylog=ylog)
            target = SAVE_PATH + '/' + level.strip('<>') + `save_name`
            print target
            fig.savefig(target + fextension)
            fig.clf()
            plt.close()


def explore_Parameterspace(TwoDFrame, title="",
                           cmap='RdBu', norm=None,
                           vmin=None, vmax=None,
                           xlog=False, ylog=False):
    """
    Explore variables in a 2-dim Parameterspace

    Parameters
    ----------
    TwoDFrame : 2D pandas.DataFrame with index and column names
        The data to plot
    title : string
        Title of the plot (Default: "")
    cmap : string
        Colormap to use (Default: "RdBu")
    vmin : float
        Minimum value of the colormap (Default: None)
    vmax : float
        Maximum vlaue of the colormap (Defualt: None)
    xlog : bool
        Wheter x axis is log scaled
    ylog : bool
        Wheter y axis is log scaled

    Examples
    --------
    >>> import init_data
    >>> data = init_data.get_Data("phi")
    >>> explore_Parameterspace(data.unstack(level="deltaT")["<safe>",0.5].
    >>>                        unstack(level="phi"))
    """

    xparams = TwoDFrame.columns.values
    yparams = TwoDFrame.index.values
    values = TwoDFrame.values

    X, Y = np.meshgrid(xparams, yparams)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    c = plt.pcolormesh(X, Y, values,
                       cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    plt.colorbar(c, orientation="vertical")
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    plt.xlabel(TwoDFrame.columns.name)
    plt.ylabel(TwoDFrame.index.name)
    plt.title(title)
    plt.tight_layout()

    return fig


def plot_network(loc):

    adjacency = np.loadtxt(loc + '_network')
    labels = np.loadtxt(loc + '_labels')

    fig2 = plt.figure()

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
    dataframe = tmp.where(tmp < 10**300, np.nan)

    # get the values of the first two index levels
    ivals = dataframe.index.levels[0]
    jvals = dataframe.index.levels[1]

    # get the values of the last index level
    # which will be transformed into colums by unstack()
    columns = dataframe.index.levels[-1]

    for j, jval in enumerate(jvals):

        # create figure with enough space for ivals*columns plots
        fig = plt.figure(figsize=(4*len(ivals), 2*len(columns)))
        axes = []

        for i, ival in enumerate(ivals):

            # extract the mean and std data for each of the values of the first index
            subset = dataframe.loc[(ival, jval,),
                    '<mean_trajectory>'].unstack('observables').dropna(axis=0, how='any')
            subset_errors = dataframe.loc[(ival, jval,),
                    '<sem_trajectory>'].unstack('observables').dropna(axis=0, how='any')
            for c, column in enumerate(columns):

                # take absolute values of data but save intervals on which original 
                # data was negative (paint them afterwards)
                subset_j = subset.loc[1:, column]
                subset_j_errors = subset_errors.loc[1:, column]
                subset_abs = subset_j.abs()
                paint = subset_j.where(subset_j<0).dropna().index.values

                # add a subplot to the list of axes
                axes.append(plt.subplot2grid((len(columns), len(ivals)), (c, i)))
                # plot mean trajectory and insequrity interval
                # also mark the area in which the mean data was negative
                subset_abs.plot(ax = axes[-1])
                plt.fill_between(subset_abs.index, 
                        subset_abs - subset_j_errors,
                        subset_abs + subset_j_errors,
                        where = subset_abs - subset_j_errors > 0,
                        color='blue',
                        alpha=0.1)
                (subset_abs - subset_j_errors).plot(ax = axes[-1], color='blue', alpha=0.3)
                (subset_abs + subset_j_errors).plot(ax = axes[-1], color='blue', alpha=0.3)
                if len(paint)>0:
                    plt.axvspan(paint[0], paint[-1], alpha=0.2, color = 'red')

                # change y axis scalling to 'log' for plots with nonzero data
                if sum(np.sign(subset_abs))>0:
                    axes[-1].set_yscale('log', nonposy='mask')

                # adjust locator counts and add y axis label
                plt.locator_params(axis='x', tight=True, nbins=5)
                if i==0: plt.ylabel(column)
                if c!=len(columns)-1: 
                    axes[-1].xaxis.set_visible(False)

        # adjust the grid layout to avoid overlapping plots and save the figure
        fig.tight_layout()
        fig.savefig(loc+'testfigure_'+`round(jval,4)`+'.pdf')
        fig.clf()
        plt.close()


def plot_obs_grid(SAVE_PATH, NAME_TRJ, NAME_CNS, pos = None, file_extension='.png'):

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
    file_extension : string
        file type for images.
    """

    print 'plotting observable grid'
    print SAVE_PATH

    trj_data = np.load(SAVE_PATH + NAME_TRJ)
    cns_data = np.load(SAVE_PATH + NAME_CNS)

    parameter_levels = [list(p.values) for p in trj_data.index.levels[2:-2]]
    parameter_level_names = [name for name in trj_data.index.names[2:-2]]
    levels = tuple(i+2 for i in range(len(parameter_levels)))

    parameter_combinations = list(it.product(*parameter_levels))

    for p in parameter_combinations:
        trj_d_slice = trj_data.xs(key=p, level=levels).dropna()
        cns_d_slice = cns_data.xs(key=p, level=levels)
        save_name = zip(parameter_level_names, p)

        plot_observables(trj_d_slice, cns_d_slice,
                         SAVE_PATH, save_name, pos, file_extension)


def plot_observables(t_data_in, c_data_in, loc, save_name, pos, file_extension='.png'):
    """
    function to create a grid of plots of the values of
    the observable for the values of the parameters given in
    indices.


    Parameters:
    -----------
    t_data_in : DataFrame
        Pandas data frame of time series data for
        different parameter combinations
    c_data_in : DataFrame
        pandas data frame of convergence data for
        different parameter combinations
    loc : string
        path to save the output data
    save_name : string
        name string to save the output data
    pos : list
        possible cue orders to use for column titles
        in case columns are equivalent to different
        initial combinations of cue orders.
    file_extension : string
        file extension to save image files
    """

    # TO DO:
    # -Color management (use same colors for same observables)
    # -Error plots
    # -Figure title management. (title for each entry of plot_list)

    # list of observables that should be plotted together
    plot_list = [['r_c', 'r_d'], ['r_c_dot', 'r_d_dot'],
                 ['K_c', 'K_d'], ['K_c_cost', 'K_d_cost'],
                 ['P_c', 'P_d'], ['Y_c', 'Y_d'],
                 [str(x) for x in pos],
                 ['R'], ['decision_state'],
                 ['G']]
    title_list = {0: 'capital rates',
                  1: 'trends',
                  2: 'capital stocks',
                  3: 'capital cost',
                  4: 'labor shares',
                  5: 'market shares',
                  6: 'cue order frequencies',
                  7: 'R',
                  8: 'decisions',
                  9: 'resource stock'}
    lin_cmap = cm.get_cmap('rainbow')
    print plot_list

    # some parameters:
    labelsize_2 = 30
    bgcolor_alpha = 0.3

    print 'plotting observable grids'

    # clean trajectory data
    t_data = t_data_in.where(t_data_in < 10**300, np.nan)

    # split convergence data
    c_values = c_data_in['<mean_convergence_state>']
    c_times = c_data_in['<mean_convergence_time>']
    c_times_min = c_data_in['<min_convergence_time>']
    c_times_max = c_data_in['<max_convergence_time>']
    c_times_nanmax = c_data_in['<nanmax_convergence_time>']
    c_times_sem = c_data_in['<sem_convergence_time>']

    norm = cl.Normalize(vmin=min(c_values), vmax=max(c_values))
    cmap = cm.get_cmap('RdBu')
    cmap.set_under('yellow')

    # get the values of the first two index levels
    ivals = t_data.index.levels[0]
    jvals = t_data.index.levels[1]

    ind_names = t_data.index.names

    # get the values of the last index level
    # which will be transformed into columns by unstack()

    for p, pl in enumerate(plot_list):
        print 'plotting ' + `p` + ' to grid'

        # create figure with enough space for ivals*columns plots
        # plus color map at the side
        fig = plt.figure(figsize=(4*len(jvals), 4*len(ivals)))
        axes = []

        for i, ival in enumerate(ivals):

            # extract the mean and std data for each
            # of the values of the first index
            # and the given observable. Drop missing
            # data and use second index as columns
            for j, jval in enumerate(jvals):
                subset = t_data.xs(
                        key=(ival, jval),
                        axis=0,
                        level=(0, 1))['<mean_trajectory>']
                subset_j = subset\
                    .unstack('observables')[pl].dropna(axis=0, how='any')

                subset_errors = t_data.xs(
                        key=(ival, jval),
                        axis=0,
                        level=(0, 1))['<sem_trajectory>']
                subset_j_errors = subset_errors\
                    .unstack('observables')[pl].dropna(axis=0, how='any')

                # add a subplot to the list of axes
                axes.append(plt.subplot2grid((len(ivals), len(jvals) + 1),
                                             (len(ivals) - i - 1, j)))

                # plot mean trajectory
                subset_j.plot(ax=axes[-1], legend=(j == 0 and i == 0),
                              colormap = lin_cmap)

                # plot standard error
                # plt.fill_between(subset_j.index,
                #                 subset_j - subset_j_errors,
                #                 subset_j + subset_j_errors,
                #                 where=subset_j - subset_j_errors > 0,
                #                 color='blue',
                #                 alpha=0.1)
                (subset_j - subset_j_errors).plot(ax=axes[-1],
                                                  colormap=lin_cmap, alpha=0.3,
                                                  legend=False)
                (subset_j + subset_j_errors).plot(ax=axes[-1],
                                                  colormap=lin_cmap, alpha=0.3,
                                                  legend=False)

                # plot consensus time with sem and min and max values

                c_time = c_times.loc[(ival, jval)]
                c_time_min = c_times_min.loc[(ival, jval)]
                c_time_max = c_times_max.loc[(ival, jval)]
                c_time_nanmax = c_times_nanmax.loc[(ival, jval)]
                c_time_sem = c_times_sem.loc[(ival, jval)]

                if True:
                    c_time = c_time
                    c_time_min = c_time_min
                    c_time_max = c_time_max
                    c_time_nanmax = c_time_nanmax
                    c_time_sem = c_time_sem

#                plt.axvline(c_time, color='grey')
#                plt.axvline(c_time_min, ls='dashed', color='grey')
#                plt.axvline(c_time_max, ls='dashed', color='grey')
#                plt.axvline(c_time_nanmax, ls='dotted', color='grey')
#                plt.axvspan(c_time - c_time_sem, c_time + c_time_sem,
#                            alpha=0.2, color='grey')


#                # change y axis scale to 'log' for plots with nonzero data
#                if sum(np.sign(subset_j)) > 0 and observable in log_list:
#                    axes[-1].set_yscale('log', nonposy='mask')

                # add tau and phi values to rows and columns
                if pos is None:
                    if i == len(ivals) - 1:
                        plt.title(ind_names[1] + ' = '
                                               + `jval`,
                                  fontsize=labelsize_2)
                # if initial types are given,
                # create a string representation of the initial
                # type distribution to use as title
                elif pos is not None:
                    title = ''
                    if i == len(ivals) - 1:
                        op_count = [int(x) for x in
                                    jval.strip('[]').split(',')]
                        ct, ixs = np.unique(op_count, return_index=True)
                        for c, ix in zip(ct, ixs):
                            if c > 0:
                                title += '{}*{}\n'.format(c, pos[ix])
                        plt.title(title, fontsize=labelsize_2)
                    pass
                if j == 0:
                    plt.ylabel(ind_names[0] + ' = ' + `round(ival, 3)`,
                               fontsize=labelsize_2)

                # remove x axis for all but the lowest row
                if i != 0:
                    axes[-1].xaxis.set_visible(False)

                # remove axis grids for all rows
                axes[-1].grid(b=False)

                # set subplot background according to consensus oppinion state
                axes[-1].patch.set_facecolor(
                        cmap(norm(c_values.loc[(ival, jval)])))
                axes[-1].patch.set_alpha(bgcolor_alpha)

        # plot colorbar
        cbar_ax = plt.subplot2grid(
                (len(ivals), len(jvals)+1),
                (0, len(jvals)),
                rowspan=len(ivals))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(cax=cbar_ax, mappable=sm, alpha=bgcolor_alpha)
        cbar.ax.tick_params(labelsize = labelsize_2)

        # adjust the grid layout to avoid overlapping plots and save the figure
        fig.tight_layout()
        fig.savefig(loc+title_list[p]+`save_name[0]`.strip('()').replace(', ',
            '=').replace('.','o')+file_extension, facecolor=fig.get_facecolor())
        fig.clf()
        plt.close()

if __name__ == '__main__':
	loc = sys.argv[1]
        plot_trajectories(loc, 'phi')
