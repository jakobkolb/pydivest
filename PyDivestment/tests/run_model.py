"""
Generic setup, run and plot routine to test the functionality of different model approximations
"""
import networkx as nx
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt


def run(name, model):
    """Perform test run and plot some output to check functionality.

    Parameters:
        name: string
            Name of the model that is tested
        model: object
            Model object, containing set_parameters() an run() routines as well
            as some getters for trajectories.
    Returns:
        return: int
            returns 1 if successful.
    """

    # investment_decisions:

    nopinions = [50, 50]
    possible_cue_orders = [[0], [1]]

    # Parameters:

    input_parameters = {'i_tau': 1, 'eps': 0.05, 'b_d': 1.2,
                        'b_c': 1., 'i_phi': 0.8, 'e': 100,
                        'G_0': 500, 'b_r0': 0.3 ** 2 * 100,
                        'possible_cue_orders': possible_cue_orders,
                        'C': 100, 'xi': 1. / 8., 'd_c': 0.06,
                        'campaign': False, 'learning': True,
                        'crs': True, 'test': True,
                        'R_depletion': False}

    # investment_decisions
    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full((n), i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:.copy()
    N = sum(nopinions)
    p = .2

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # investment
    clean_investment = np.ones(N)
    dirty_investment = np.ones(N)

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)

    m = model(*init_conditions, **input_parameters)

    # m._setup_model()

    m.run(t_max=100)

    m.R_depletion = True
    m.set_parameters()
    m.run(t_max=350)

    # Plot the results

    trj = m.get_unified_trajectory()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    trj[['n_c']].plot(ax=ax1)

    ax2 = fig.add_subplot(222)
    trj[['k_c', 'k_d']].plot(ax=ax2)
    ax2b = ax2.twinx()
    trj[['c']].plot(ax=ax2b, color='g')

    ax3 = fig.add_subplot(223)
    trj[['r_c', 'r_d']].plot(ax=ax3)

    ax4 = fig.add_subplot(224)
    trj[['g']].plot(ax=ax4)

    fig.tight_layout()
    fig.savefig('{}_test.png'.format(name))

    return 1
