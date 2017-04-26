"""Test the micro model."""

import datetime
from random import shuffle
import networkx as nx
import numpy as np

from ..micro_model import divestment_core as dc

output_location = \
    'test_output/' \
    + datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") \
    + '_output'

# Initial conditions:
for FFH in [False, True]:

    if FFH:
        nopinions = [10, 10, 10, 10, 10, 10, 10, 10]
        possible_opinions = [[2, 3],  # short term investor
                             [3, 2],  # long term investor
                             [4, 2],  # short term herder
                             [4, 3],  # trending herder
                             [4, 1],  # green conformer
                             [4, 0],  # dirty conformer
                             [1],  # gutmensch
                             [0]]  # redneck
        input_parameters = {'tau': 1, 'eps': 0.05, 'b_d': 1.2,
                            'b_c': 1., 'phi': 0.8, 'e': 100,
                            'G_0': 1500, 'b_r0': 0.1 ** 2 * 100,
                            'possible_opinions': possible_opinions,
                            'C': 1, 'xi': 1. / 8., 'beta': 0.06,
                            'campaign': False, 'learning': True}

    else:
        # investment_decisions:
        nopinions = [10, 10]
        possible_opinions = [[0], [1]]

        # Parameters:

        input_parameters = {'tau': 1, 'eps': 0.05, 'b_d': 1.2,
                            'b_c': 1., 'phi': 0.8, 'e': 100,
                            'G_0': 1500, 'b_r0': 0.1 ** 2 * 100,
                            'possible_opinions': possible_opinions,
                            'C': 1, 'xi': 1. / 8., 'beta': 0.06,
                            'campaign': False, 'learning': True}

    cops = ['c' + str(x) for x in possible_opinions]
    dops = ['d' + str(x) for x in possible_opinions]

    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full(n, i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:
    N = sum(nopinions)
    p = 10. / N

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    (mucc, mucd, mudc, mudd) = (1, 1, 1, 1)

    op = np.array(opinions)

    clean_investment = mucc * op + mudc * (1 - op)
    dirty_investment = mucd * op + mudd * (1 - op)

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)

    # Initialize Model

    model = dc.Divestment_Core(*init_conditions,
                               **input_parameters)
