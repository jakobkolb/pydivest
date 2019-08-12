import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cond(df, c, ax, t_max, thresh=0.5, c_df=None):
    tmp = df

    if c_df is None:
        c_tmp = tmp
    else:
        c_tmp = c_df
    tmp1 = tmp[[c]].loc[c_tmp > thresh]
    tmp2 = tmp[[c]].loc[c_tmp < thresh]
    try:
        tmp1.unstack('sample').loc[:t_max].plot(ax=ax,
                                                alpha=.1,
                                                legend=False,
                                                color=['#878685'])
    except TypeError:
        print('no data')
    try:
        tmp2.unstack('sample').loc[:t_max].plot(ax=ax,
                                                legend=False,
                                                alpha=.1,
                                                color=['orange'])
    except TypeError:
        print('no data')


def plot_hist(ax, trj, cond, t_max, thresh, indicator, cond_indicator):
    tmp = trj.xs(level='tstep', key=t_max)[[indicator]]
    tmp_cond = cond.xs(level='tstep', key=t_max)
    tmp['$t_t => 2019$'] = float('nan')
    tmp['$t_t < 2019$'] = float('nan')
    tmp.loc[tmp_cond[cond_indicator] > thresh,
            '$t_t => 2019$'] = tmp.loc[tmp_cond[cond_indicator] > thresh, indicator]
    tmp.loc[tmp_cond[cond_indicator] < thresh,
            '$t_t < 2019$'] = tmp.loc[tmp_cond[cond_indicator] < thresh, indicator]

    y_max = trj[indicator].max()
    y_min = trj[indicator].xs(level='tstep', key=t_max).min()

    tmp[['$t_t < 2019$', '$t_t => 2019$']].plot(kind='hist',
                               ax=ax,
                               stacked=True,
                               colors=['orange', '#878685'],
                               orientation='horizontal',
                               bins=np.linspace(y_min, y_max, 40),
                               legend=False)
    ax.set_ylim([y_min, y_max])
    ax.axis('off')

    return y_min, y_max


def plot_scatter(ax,
                 trj,
                 cond,
                 t_scatt,
                 thresh,
                 indicator,
                 cond_indicator='inv_distance',
                 t_cond=None):
    import matplotlib.colors as cls

    tmp = trj.xs(level='tstep', key=t_scatt[0])[[indicator[0]]]
    tmp[indicator[1]] = trj.xs(level='tstep', key=t_scatt[1])[indicator[1]]
    tmp_cond = cond.xs(level='tstep', key=t_scatt[1])

    tmp['cond'] = float('nan')
    tmp.loc[tmp_cond[cond_indicator] < thresh, 'cond'] = 0
    tmp.loc[tmp_cond[cond_indicator] > thresh, 'cond'] = 1

    y_max = trj[indicator].max()
    y_min = trj[indicator].min()

    tmp.plot.scatter(
        x=indicator[1],
        y=indicator[0],
        c='cond',
        ax=ax,
        colormap=cls.ListedColormap(['orange', '#878685']),
        # colors=['orange', '#878685'],
        legend=False,
        colorbar=False)
    # ax.set_ylim([y_min, y_max])
    # ax.axis('off')
    plt.tick_params(
        axis='both',  # changes apply to the x and y axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    return y_min, y_max


def plot(fig, trj, trj2,
         thresh_time=10.,
         thresh_value=.5,
         t_max_1=20.,
         t_max_2=40.,
         plot_no='',
         hlines=[]):

    # setting up axes
    axes = []

    b_pad = .13
    t_pad = .2
    l_pad = .05

    whist = .12
    gap = .1

    wplots = 1 - whist - 2 * gap - l_pad

    w1 = .33 * wplots
    w2 = .33 * wplots
    w3 = .33 * wplots

    l1 = l_pad
    l2 = l1 + w1 + gap * .5
    l3 = l2 + w2 + gap

    axes.append(plt.axes((l1, b_pad, w1, 1 - t_pad)))
    axes.append(plt.axes((l2, b_pad, w2, 1 - t_pad), frameon=False))
    axes.append(plt.axes((l3, b_pad, w3, 1 - t_pad)))
    axes.append(plt.axes((l3 + w3, b_pad, whist, 1 - t_pad)))

    # setting parameters
    thresh_indicator = 'i_c'
    indicators = ['i_c', 'G']
    t_max = t_max_2
    t_scatt = [t_max_1, t_max]

    cond = pd.DataFrame(index=trj.index)
    cond2 = pd.DataFrame(index=trj2.index)
    for s in trj.index.levels[0]:
        cond.loc[(s),
                 thresh_indicator] = trj.loc[(s, thresh_time),
                                             thresh_indicator]
    for s in trj2.index.levels[0]:
        cond2.loc[(s),
                  thresh_indicator] = trj2.loc[(s, thresh_time),
                                               thresh_indicator]
    # cropping data
    trj = trj.loc[(trj.reset_index()['tstep'] <= t_max).values]

    # plotting
    plot_cond(trj,
              indicators[0],
              axes[0],
              t_scatt[0],
              thresh=thresh_value,
              c_df=cond[thresh_indicator])
    plot_cond(trj2,
              indicators[1],
              axes[2],
              t_max,
              thresh=thresh_value,
              c_df=cond2[thresh_indicator])
    y_min, y_max = plot_hist(axes[3],
                             trj2,
                             cond2,
                             t_max,
                             thresh_value,
                             indicators[1],
                             cond_indicator=thresh_indicator)
    plot_scatter(axes[1],
                 trj2,
                 cond2,
                 t_scatt,
                 thresh_value,
                 indicators,
                 cond_indicator=thresh_indicator)

    l_items = []
    for l in hlines:
        l_items.append(axes[3].axhline(l['value'], ls=':', label=l['label'],
                                       color=l['color']))
    if plot_no == str(1):
        labels = [l.get_label() for l in l_items]
        axes[3].legend(l_items, labels, frameon=False)

    # fixing stuff with axes labels and ticks

    axes[2].set_ylim([y_min, y_max])
    axes[1].invert_xaxis()

    axes[0].set_xlim([1, 20])
    axes[0].set_ylim([0, 1])
    axes[0].set_xlabel('')
    axes[0].set_ylabel(r'fraction of clean investment')
    axes[0].axvline(t_scatt[0], color='#878685')
    axes[0].axvline(thresh_time, color='k', alpha=.7)
    x_labels = axes[0].get_xticks().tolist()
    axes[0].set_xticklabels([int(x+2010) for x in x_labels])
    axes[0].text(0.9, 0.9, f'{plot_no}a', ha='center', va='center',
                 transform=axes[0].transAxes)

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel(f'  fossil resource remaining in {int(2010+t_max)}')
    axes[1].set_ylabel(f'  fraction of clean investment in {int(2010+t_scatt[0])}')
    axes[1].text(0.9, 0.9, f'{plot_no}b', ha='center', va='center',
                 transform=axes[1].transAxes)

    axes[2].set_ylabel('remaining fossil resource [Mtoe]')
    axes[2].set_xlabel('')
    axes[2].set_xlim([0, t_max])
    x_labels = axes[2].get_xticks().tolist()
    axes[2].set_xticklabels([int(x+2010) for x in x_labels])
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[2].locator_params(axis='y', nbins=6)
    axes[2].text(0.9, 0.9, f'{plot_no}c', ha='center', va='center',
                 transform=axes[2].transAxes)

    fig.tight_layout()
