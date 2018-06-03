import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class plot_routines(object):
    """
    this contains the routines to make different plots to compare the two
    types of model description
    """

    def __init__(self, micro_path, macro_path, micro_selector, macro_selector):
        with pd.HDFStore(micro_path + 'mean.h5') as store:
            self.micro_mean = store.select('dat', micro_selector)
        with pd.HDFStore(micro_path + 'std.h5') as store:
            self.micro_sem = store.select('dat', micro_selector)
        with pd.HDFStore(macro_path + 'mean.h5') as store:
            self.macro_mean = store.select('dat', macro_selector)
        with pd.HDFStore(macro_path + 'std.h5') as store:
            self.macro_sem = store.select('dat', macro_selector)

        self.data_sets = [self.micro_mean, self.micro_sem,
                          self.macro_mean, self.macro_sem]

        for d in self.data_sets:
            d.index = d.index.droplevel(['model', 'test', 'sample'])

        self.variable_combos = [['x', 'y', 'z'], ['c', 'g'], ['mu_c^c', 'mu_d^d'],
                                ['mu_c^d', 'mu_d^c']]
        self.latex_labels = [['$x$', '$y$', '$z$'], ['$c$', '$g$'],
                             ['$\mu^{(c)}_c$', '$\mu^{(d)}_d$'],
                             ['$\mu^{(c)}_d$', '$\mu^{(d)}_c$']]

        self.colors = 'rgbk'

    def mk_plots(self, bd):

        # select data for given value of bd
        local_datasets = []
        for d in self.data_sets:
            local_datasets.append(d.xs(bd, level=0))

        phi_vals = local_datasets[0].index.levels[0].values

        l_phi = len(phi_vals)
        l_vars = len(self.variable_combos)

        fig = plt.figure()
        axes = [fig.add_subplot(l_vars, l_phi, i + 1 + j * l_phi)
                for i in range(l_phi)
                for j in range(l_vars)]

        for i, phi in enumerate(phi_vals):
            for j, variables in enumerate(self.variable_combos):
                ax_id = self.grid_index(i, j, l_phi, l_vars) - 1
                # local data set for specifiv value of phi
                ldp = []
                for d in local_datasets:
                    ldp.append(d.xs(phi, level=0))
                print(self.grid_index(i, j, l_phi, l_vars))
                ldp[0][variables] \
                    .plot(
                    ax=axes[ax_id],
                    legend=False,
                    color=self.colors)
                for k, variable in enumerate(variables):
                    upper_limit = np.transpose(ldp[0][[variable]].values \
                                               + ldp[1][[variable]].values)[0]
                    lower_limit = np.transpose(ldp[0][[variable]].values \
                                               - ldp[1][[variable]].values)[0]
                    axes[ax_id].fill_between(ldp[0].index.values,
                                             upper_limit, lower_limit,
                                             color=self.colors[k],
                                             alpha=0.2)
                    ldp[2][variables] \
                        .plot(
                        ax=axes[ax_id],
                        legend=False,
                        color=self.colors,
                        style='-.')

        return fig

    def mk_opinon_plots(self, selector: dict,
                        y_limits: list,
                        legend_location: int,
                        ax: plt.axis,
                        legend: bool = True,
                        x_limits: list = [0, 900],
                        x_label: str = 't',
                        y_ticks: bool = True):

        micro_alpha = 0.8

        local_datasets = []
        for d in self.data_sets:
            local_datasets.append(d.xs(list(selector.values()), level=list(selector.keys())))

        variables = self.variable_combos[0]
        local_datasets[0][variables].plot(ax=ax,
                                          color=self.colors,
                                          alpha=micro_alpha,
                                          legend=legend)
        for k, variable in enumerate(variables):
            upper_limit = np.transpose(local_datasets[0][[variable]].values \
                                       + local_datasets[1][[variable]].values)[0]
            lower_limit = np.transpose(local_datasets[0][[variable]].values \
                                       - local_datasets[1][[variable]].values)[0]
            ax.fill_between(local_datasets[0].index.values,
                            upper_limit, lower_limit,
                            color='k',
                            alpha=0.05)
            ax.plot(local_datasets[0].index.values,
                    upper_limit,
                    color=self.colors[k],
                    alpha=0.2)
            ax.plot(local_datasets[0].index.values,
                    lower_limit,
                    color=self.colors[k],
                    alpha=0.2)

            local_datasets[2][[variable]] \
                .plot(ax=ax,
                      color=self.colors[k],
                      legend=False,
                      style='--',
                      linewidth=2
                      )
        ax.set_ylim(y_limits)
        ax.set_xlim(x_limits)
        ax.set_xlabel(x_label)
        k = len(variables)
        patches, labels = ax.get_legend_handles_labels()
        labels = self.latex_labels[0]
        if not y_ticks:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        if legend:
            lg = ax.legend(patches[:k], labels[:k],
                           loc=legend_location,
                           title='',
                           fontsize=12)
            lg.get_frame().set_alpha(0.8)

    def mk_4plots(self, selector: dict,
                  upper_limits: list = [1., 80, 11, 10],
                  lower_limits: list = [-.8, 0., 0., 0.],
                  legend_locations: list = [4, 7, 1, 7],
                  tmax: int = 1000):
        # set opacity for plots of micro data:
        micro_alpha = 0.8
        # select data for given value of bd and phi
        local_datasets = []
        for d in self.data_sets:
            local_datasets.append(d.xs(list(selector.values()), level=list(selector.keys())))

        l_vars = len(self.variable_combos)
        fig = plt.figure(figsize=(8, 6))
        axes = [fig.add_subplot(2, 2, i + 1) for i in range(l_vars)]

        for j, variables in enumerate(self.variable_combos):
            ax_id = j
            axes[ax_id].set_xlim([0, tmax])
            # local data set for specify value of phi
            ldp = local_datasets
            if j == 1:
                c_ax = axes[ax_id].twinx()
                ldp[0][variables[0]] \
                    .plot(
                    ax=axes[ax_id],
                    color=self.colors[0],
                    alpha=micro_alpha)
                ldp[0][variables[1]] \
                    .plot(
                    ax=c_ax,
                    color=self.colors[1],
                    alpha=micro_alpha)

                c_ax.set_ylim([0, 5])
            else:
                ldp[0][variables] \
                    .plot(
                    ax=axes[ax_id],
                    color=self.colors,
                    alpha=micro_alpha)
            for k, variable in enumerate(variables):
                ax = axes[ax_id]
                if variable == 'g':
                    ax = c_ax
                    c_ax.set_ylim([lower_limits[-1], upper_limits[-1]])

                upper_limit = np.transpose(ldp[0][[variable]].values \
                                           + ldp[1][[variable]].values)[0]
                lower_limit = np.transpose(ldp[0][[variable]].values \
                                           - ldp[1][[variable]].values)[0]
                ax.fill_between(ldp[0].index.values,
                                upper_limit, lower_limit,
                                color='k',
                                alpha=0.05)
                ax.plot(ldp[0].index.values,
                        upper_limit,
                        color=self.colors[k],
                        alpha=0.2)
                ax.plot(ldp[0].index.values,
                        lower_limit,
                        color=self.colors[k],
                        alpha=0.2)

                ldp[2][[variable]] \
                    .plot(ax=ax,
                          color=self.colors[k],
                          legend=False,
                          style='--',
                          linewidth=2
                          )

            ax = axes[ax_id]
            ax.set_ylim([lower_limits[j], upper_limits[j]])
            k = len(variables)
            patches, labels = ax.get_legend_handles_labels()
            labels = self.latex_labels[j]
            if j == 1:
                gArtist = plt.Line2D((0, 1), (0, 0), color=self.colors[1])
                cArtist = plt.Line2D((0, 1), (0, 0), color=self.colors[0])
                patches = [cArtist, gArtist]
            if j is 0:
                ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                ax.set_xlabel('t')
            lg = ax.legend(patches[:k], labels[:k],
                           loc=legend_locations[ax_id],
                           title='',
                           fontsize=12)
            lg.get_frame().set_alpha(0.8)

            # ax.set_title('')
            # ax.set_loc

        return fig

    def grid_index(self, row, col, n_rows, n_cols):
        """
        calculate index number in subplot grids
        grid_index(0,0) = 1 ~ upper left corner
        to index the axis list, one might use grid_index - 1
        """
        return col + 1 + row * n_cols
