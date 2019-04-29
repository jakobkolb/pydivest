
# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3


from __future__ import print_function

import itertools as it
import sys

import matplotlib.cm as cm
import matplotlib.colors as cl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def load(*args, **kwargs):
    return np.load(*args, allow_pickle=True, **kwargs)


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
    if not SAVE_PATH.endswith('/'):
        SAVE_PATH += '/'
    data = load(SAVE_PATH + NAME)

    parameter_levels = [list(p.values) for p in data.index.levels[2:]]
    parameter_level_names = [name for name in data.index.names[2:]]
    levels = tuple(i + 2 for i in range(len(parameter_levels)))

    parameter_combinations = list(it.product(*parameter_levels))

    for p in parameter_combinations:
        d_slice = data.xs(key=p, level=levels)
        save_name = list(zip(parameter_level_names, [str(x) for x in p]))

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
            target = SAVE_PATH + '/' + level.strip('<>') + repr(save_name)
            fig.savefig(target + fextension)
            fig.clf()
            plt.close()


def tau_phi_linear(SAVE_PATH, NAME, xlog=False,
                   ylog=False, fextension='.png'):
    if not SAVE_PATH.endswith('/'):
        SAVE_PATH += '/'

    data = load(SAVE_PATH + NAME)

    parameter_levels = [list(p.values) for p in data.index.levels[2:]]
    parameter_level_names = [name for name in data.index.names[2:]]
    levels = tuple(i + 2 for i in range(len(parameter_levels)))

    parameter_combinations = list(it.product(*parameter_levels))

    for p in parameter_combinations:
        d_slice = data.xs(key=p, level=levels)
        save_name = list(zip(parameter_level_names, [str(x) for x in p]))

        print(save_name)
        d_slice_mean = d_slice['<mean_convergence_state>'].unstack('c_count')
        d_slice_err = d_slice['<sem_convergence_state>'].unstack('c_count')

        d_slice_mean.plot(style='-o', yerr=d_slice_err)
        plt.show()


def tau_phi_final(SAVE_PATH, NAME, xlog=False,
                  ylog=False, fextension='.png'):
    if not SAVE_PATH.endswith('/'):
        SAVE_PATH += '/'

    data = load(SAVE_PATH + NAME)['<mean_trajectory>'] \
        .xs(key=('R', 0.05, 300),
            level=('observables', 'alpha', 'timesteps'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[data > 239].plot.hist(legend=False, ax=ax, bins=20)
    ax.set_yscale('log')
    # ax.set_xlim([50, 300])
    # ax.set_ylim([10**2, 10**2.5])
    plt.show()


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
    adjacency = loadtxt(loc + '_network')
    labels = loadtxt(loc + '_labels')

    fig2 = plt.figure()

    G = nx.DiGraph(adjacency)
    from networkx.drawing.nx_agraph import graphviz_layout
    layout = graphviz_layout(G)
    nx.draw_networkx_nodes(G, pos=layout, node_color=labels, node_size=20)
    nx.draw_networkx_edges(G, pos=layout, width=.2)

    fig2.savefig(loc + '_network_plot')


def plot_amsterdam(path, name, cues=['[0]', '[1]']):
    if not path.endswith('/'):
        path += '/'

    data = load(path + name)
    colors = [x for x in "brwcmygk"][:len(cues)]
    if cues[-1] == '[5]':
        colors.append('#2E8B57')

    # create list of variable parameter combinations
    indices = it.product(*[j.tolist() for j in data.index.levels[:2]])
    # indices = [(1.6, 0.8),]
    # create list of observable names
    cols = [letter + cue for letter, cue in it.product(['', 'c', 'd'], cues)]
    cols.append('decision state')
    # split list of observable names into parts
    frequencies = cols[:len(cues)]
    cds = cols[len(cues):2 * len(cues)]
    dds = cols[2 * len(cues):3 * len(cues)]

    for i in indices:
        cd_sem = data['sem_trajectory'].unstack(3)[cols].xs(i)
        cd_mean = data['mean_trajectory'].unstack(3)[cols].xs(i)

        d_sem = data['sem_trajectory'].unstack(3).xs(i)
        d_mean = data['mean_trajectory'].unstack(3).xs(i)

        fig = plt.figure()
        # plot opinion frequencies
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        cd_mean[frequencies].plot.area(ax=ax1, alpha=0.2, color=colors)
        ax1.set_ylim(top=sum(cd_mean[frequencies].xs(
            cd_mean[frequencies].index.values[1]).values))
        ax1.set_ylabel('capital return')

        # plot decision according to opinions
        ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2)
        cd_mean[cds + dds].plot.area(ax=ax2, alpha=0.2, color=colors,
                                     legend=False)
        cd_mean['decision state'].plot(lw=3, color='red', legend=False)
        ax2.set_ylim(top=1.)
        ax2.set_ylabel('decisions according to cue order')

        # combine relevant parts of legends
        patches = ax1.get_legend().get_patches()
        handles, labels = ax1.get_legend_handles_labels()

        line = ax2.get_lines()[-1]
        labels.append(line.get_label())
        patches.append(line)

        ax1.clear()
        ax1.locator_params(axis='both', nticks=4)
        d_mean[['r_c', 'r_d']].plot(ax=ax1, color=['g', 'k'])
        ax1.set_ylabel('capital return')

        # draw legend
        leg = ax2.legend(patches, labels, bbox_to_anchor=(1.05, 1), loc=2,
                         borderaxespad=0.)

        # save figures
        plt.savefig(path + 'amsterdam_{}_{}.png'.format(*i),
                    bbox_extra_artists=(leg,), bbox_inches='tight')


def plot_parameter_dict(SAVE_PATH, NAME):
    if not SAVE_PATH.endswith('/'):
        SAVE_PATH += '/'

    data = load(SAVE_PATH + NAME)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    df = data['states_mean'].unstack('phi')
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    ax.set_xlim([0, len(df.columns)])
    ax.set_ylim([0, len(df.index)])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(SAVE_PATH + 'dict_plot.png')


def plot_trajectories(loc, name, params, indices):
    if not loc.endswith('/'):
        loc += '/'

    print('plotting trajectories')

    tmp = load(loc + name)
    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    dataframe = tmp.where(tmp < 10 ** 300, np.nan)

    # get the values of the first two index levels
    ivals = dataframe.index.levels[0]
    jvals = dataframe.index.levels[1]

    # get the values of the last index level
    # which will be transformed into colums by unstack()
    columns = dataframe.index.levels[-1]

    for j, jval in enumerate(jvals):

        # create figure with enough space for ivals*columns plots
        fig = plt.figure(figsize=(4 * len(ivals), 2 * len(columns)))
        axes = []

        for i, ival in enumerate(ivals):

            # extract the mean and std data for each of the values of the first index
            subset = dataframe.loc[(ival, jval,),
                                   'mean_trajectory'].unstack(
                'observables').dropna(axis=0, how='any')
            subset_errors = dataframe.loc[(ival, jval,),
                                          'sem_trajectory'].unstack(
                'observables').dropna(axis=0, how='any')
            for c, column in enumerate(columns):

                subset_j = subset.loc[:, column]
                subset_j_errors = subset_errors.loc[:, column]

                # add a subplot to the list of axes
                axes.append(
                    plt.subplot2grid((len(columns), len(ivals)), (c, i)))
                # plot mean e_trajectory and insequrity interval
                # also mark the area in which the mean data was negative
                subset_j.plot(ax=axes[-1])
                plt.fill_between(subset_j.index,
                                 subset_j - subset_j_errors,
                                 subset_j + subset_j_errors,
                                 color='blue',
                                 alpha=0.1)
                (subset_j - subset_j_errors).plot(ax=axes[-1], color='blue',
                                                  alpha=0.3)
                (subset_j + subset_j_errors).plot(ax=axes[-1], color='blue',
                                                  alpha=0.3)

                # change y axis scalling to 'log' for plots with nonzero data
                # if sum(np.sign(subset_j))>0:
                #     axes[-1].set_yscale('log', nonposy='mask')

                # adjust locator counts and add y axis label
                plt.locator_params(axis='x', tight=True, nbins=5)
                if i == 0: plt.ylabel(column)
                if c != len(columns) - 1:
                    axes[-1].xaxis.set_visible(False)

        # adjust the grid layout to avoid overlapping plots and save the figure
        print('saving figure {:5.2f}'.format(jval))
        fig.tight_layout()
        fig.savefig(loc + 'testfigure_{:5.2f}.pdf'.format(jval))
        fig.clf()
        plt.close()


def plot_obs_grid(SAVE_PATH, NAME_TRJ, NAME_CNS, pos=None, t_max=None,
                  file_extension='.png', test=False):
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
        e_trajectory data
    NAME_CNS : string
        the name of the datafile containing the processed
        consensus data
    file_extension : string
        file type for images.
    """

    if not SAVE_PATH.endswith('/'):
        SAVE_PATH += '/'

    print('plotting observable grid')

    trj_data = load(SAVE_PATH + NAME_TRJ)
    cns_data = load(SAVE_PATH + NAME_CNS)

    parameter_levels = [list(p.values) for p in trj_data.index.levels[2:-2]]
    parameter_level_names = [name for name in trj_data.index.names[2:-2]]
    levels = tuple(i + 2 for i in range(len(parameter_levels)))

    parameter_combinations = list(it.product(*parameter_levels))

    for p in parameter_combinations:
        trj_d_slice = trj_data.xs(key=p, level=levels)
        cns_d_slice = cns_data.xs(key=p, level=levels)
        p = (round(x, 4) for x in p)
        save_name = list(zip(parameter_level_names, p))

        plot_observables(trj_d_slice, cns_d_slice,
                         SAVE_PATH, save_name, pos, t_max, file_extension,
                         test=test)


def plot_observables(t_data_in, c_data_in, loc,
                     save_name, pos=None, t_max=None, file_extension='.png',
                     test=False):
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
                 ['K_c', 'K_d', 'C'], ['K_c_cost', 'K_d_cost'],
                 ['P_c', 'P_d'], ['Y_c', 'Y_d'],
                 [str(x) for x in pos],
                 ['R'], ['decision state'],
                 ['G'],
                 ['c_R', 'P_d_cost', 'K_d_cost'],
                 ['P_d_cost', 'P_c_cost']]
    title_list = {0: 'capital rates',
                  1: 'trends',
                  2: 'capital stocks',
                  3: 'capital cost',
                  4: 'labor shares',
                  5: 'market shares',
                  6: 'cue order frequencies',
                  7: 'R',
                  8: 'decisions',
                  9: 'resource stock',
                  10: 'dirty costs',
                  11: 'labor cost'}
    log_list = [2, 4, 5, 7, 9, 10]
    cops = ['c' + str(x) for x in pos]
    dops = ['d' + str(x) for x in pos]
    colors = [x for x in "brwcmygk"]
    colors.append('#2E8B57')
    colors = colors[:len(pos)]
    colorlist = colors + colors

    # some parameters:
    labelsize_2 = 30
    bgcolor_alpha = 0.3

    print('plotting observable grids')

    # clean e_trajectory data
    t_data = t_data_in.where(t_data_in < 10 ** 300, np.nan)

    # split convergence data
    c_values = c_data_in['<mean_convergence_state>']
    c_times = c_data_in['<mean_convergence_time>']
    c_times_min = c_data_in['<min_convergence_time>']
    c_times_max = c_data_in['<max_convergence_time>']
    c_times_nanmax = c_data_in['<nanmax_convergence_time>']
    c_times_sem = c_data_in['<sem_convergence_time>']

    norm = cl.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('RdBu')
    cmap.set_under('yellow')

    # get the values of the first two index levels
    ivals = t_data.index.levels[0]
    jvals = t_data.index.levels[1]

    ind_names = t_data.index.names

    # get the values of the last index level
    # which will be transformed into columns by unstack()

    for p in range(len(plot_list)):
        # for L in [8]:
        pl = plot_list[p]
        if len(pl) == 1:
            colors = colorlist[-1]
        else:
            colors = colorlist
        print('plotting {} to grid'.format(pl))

        # create figure with enough space for ivals*columns plots
        # plus color map at the side
        x_opinionspace = 3 if ind_names[0] == 'opinion' else 0
        y_opinionspace = 3 if ind_names[1] == 'opinion' else 0
        x_legendspace = 0 if title_list[p] == 'decisions' else 0
        x_legendgrid = 0 if title_list[p] == 'decisions' else 0
        x_colorbarspace = 2
        fig = plt.figure(figsize=
                         (4 * (len(jvals)
                               + x_opinionspace
                               + x_legendspace
                               + x_colorbarspace),
                          4 * (len(ivals)
                               + y_opinionspace)))

        axes = []

        leg = None

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
                subset_j = subset \
                    .unstack('observables')[pl].dropna(axis=0, how='any')
                subset_errors = t_data.xs(
                    key=(ival, jval),
                    axis=0,
                    level=(0, 1))['<sem_trajectory>']
                subset_j_errors = subset_errors \
                    .unstack('observables')[pl].dropna(axis=0, how='any')

                # add a subplot to the list of axes
                axes.append(plt.subplot2grid((len(ivals),
                                              len(jvals) + 1 + x_legendgrid),
                                             (len(ivals) - i - 1,
                                              j + x_legendgrid)))

                # plot mean e_trajectory (and fix the color problem..)
                pt = subset_j.plot(ax=axes[-1], legend=(j == 0 and i == 0),
                                   color=colors)

                # plot stacked decision for mean decision state:
                if title_list[p] == 'decisions':
                    subset.unstack('observables')[cops + dops] \
                        .plot.area(color=colorlist, ax=axes[-1], alpha=0.3,
                                   legend=(j == 0 and i == 0))
                    pt = subset_j.plot(ax=axes[-1], legend=False,
                                       color='red', lw=5)
                    axes[-1].set_ylim(0., 1.)

                # plot standard error
                for var in subset_j.columns:
                    x = subset_j.index.values
                    y1 = (subset_j[[var]] -
                          subset_j_errors[[var]]).values.T[0]
                    y2 = (subset_j[[var]] +
                          subset_j_errors[[var]]).values.T[0]
                    w = (subset_j[[var]] - subset_j_errors[[var]]
                         > 0).values.T[0]
                    plt.fill_between(x, y1, y2,
                                     where=w,
                                     color=colors,
                                     alpha=0.1)
                (subset_j - subset_j_errors).plot(ax=axes[-1],
                                                  color=colors, alpha=0.3,
                                                  legend=False)
                (subset_j + subset_j_errors).plot(ax=axes[-1],
                                                  color=colors, alpha=0.3,
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

                plt.axvline(c_time, color='grey')
                plt.axvline(c_time_min, ls='dashed', color='grey')
                plt.axvline(c_time_max, ls='dashed', color='grey')
                plt.axvline(c_time_nanmax, ls='dotted', color='grey')
                plt.axvspan(c_time - c_time_sem, c_time + c_time_sem,
                            alpha=0.2, color='grey')

                #                # change y axis scale to 'log' for plots with nonzero data
                if p in log_list:
                    axes[-1].set_yscale('log', nonposy='mask')

                # add tau and phi values to rows and columns
                if ind_names[1] != 'opinion':
                    if i == len(ivals) - 1:
                        plt.title('{0} = {1:4.2f}'.format(ind_names[0], jval),
                                  fontsize=labelsize_2)
                # if initial types are given,
                # create a string representation of the initial
                # type distribution to use as title
                elif ind_names[1] == 'opinion':
                    title = ''
                    if i == len(ivals) - 1:
                        op_count = [int(a) for a in
                                    jval.strip('[]').split(',')]
                        ixs = np.nonzero(op_count)[0]
                        ct = [op_count[ix] for ix in ixs]
                        for c, ix in zip(ct, ixs):
                            if c > 0:
                                title += '{}*{}\n'.format(c, pos[ix])
                        plt.title(title, fontsize=labelsize_2)
                    pass
                if ind_names[0] == 'opinion':
                    title = ''
                    if j == 0:
                        op_count = [int(a) for a in
                                    ival.strip('[]').split(',')]
                        ixs = np.nonzero(op_count)[0]
                        ct = [op_count[ix] for ix in ixs]
                        for c, ix in zip(ct, ixs):
                            if c > 0:
                                title += '{}*{}\n'.format(c, pos[ix])
                        plt.ylabel(title, fontsize=labelsize_2)
                else:
                    if j == 0:
                        plt.ylabel('{0} = {1:5.2f}'.format(ind_names[0], ival),
                                   fontsize=labelsize_2)

                # remove x axis for all but the lowest row
                # if i != 0:
                #     axes[-1].xaxis.set_visible(False)
                if t_max is not None:
                    axes[-1].set_xlim([0, t_max])

                # remove axis grids for all rows
                axes[-1].grid(b=False)

                # set subplot background according to consensus oppinion state
                if title_list[p] not in ['decisions', 'cue order frequencies']:
                    axes[-1].patch.set_facecolor(
                        cmap(norm(c_values.loc[(ival, jval)])))
                    axes[-1].patch.set_alpha(bgcolor_alpha)

                # set legend outside of plotting area for decisions
                if j == 0 and i == 0:
                    leg = pt.get_legend()
                    if title_list[p] in ['decisions', 'cue order frequencies']:
                        leg.set_bbox_to_anchor((1.2 * len(jvals)
                                                - 0 * x_opinionspace
                                                + .8, 3))
                        for t in leg.get_texts():
                            t.set_fontsize(labelsize_2)

        # plot colorbar
        if title_list[p] not in ['decisions', 'cue order frequencies']:
            cbar_ax = plt.subplot2grid(
                (len(ivals), len(jvals) + 1),
                (0, len(jvals)),
                rowspan=len(ivals))
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cbar = fig.colorbar(cax=cbar_ax, mappable=sm, alpha=bgcolor_alpha)
            cbar.ax.tick_params(labelsize=labelsize_2)

        # adjust the grid layout to avoid overlapping plots and save the figure
        tit = title_list[p].replace(' ', '_')
        snm = repr(save_name[0]).strip('()').replace(', ', '=').replace('.',
                                                                        'o')
        sloc = loc \
               + tit \
               + snm \
               + file_extension

        if not test:
            fig.tight_layout()
            fig.savefig(sloc, bbox_extra_artists=(leg,), bbox_inches='tight')
        else:
            print(sloc)
        fig.clf()
        plt.close()


def plot_phase_transition(loc, name, ):
    if not loc.endswith('/'):
        loc += '/'

    with open(loc + name) as target:
        tmp = load(target)
    tmp = tmp.xs(0.5, level='alpha')[['<mean_remaining_resource_fraction>']]

    tmp.unstack('N').plot()
    plt.show()


if __name__ == '__main__':
    loc = sys.argv[1]
    plot_trajectories(loc, 'phi')
