
# coding: utf-8
import numpy as np
import networkx as nx
from random import shuffle

from pydivest.macro_model.integrate_equations_mean import \
    IntegrateEquationsMean as mean_model
from pydivest.macro_model.integrate_equations_aggregate import \
    IntegrateEquationsAggregate as aggregate_model

# investment_decisions:

nopinions = [50, 50]

# Parameters:

phi, b_d = 0.4, 1.25

# investment_decisions
opinions = []
for i, n in enumerate(nopinions):
    opinions.append(np.full((n), i, dtype='I'))
opinions = [item for sublist in opinions for item in sublist]
shuffle(opinions)

# network:
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

for interaction in [1, 2]:
    input_parameters = {'b_c': 1., 'phi': phi, 'tau': 1.,
                        'eps': 0.05, 'b_d': b_d, 'e': 100.,
                        'b_r0': 0.1 ** 2 * 100.,
                        'possible_opinions': [[0], [1]],
                        'xi': 1. / 8., 'beta': 0.06,
                        'L': 100., 'C': 100., 'G_0': 800.,
                        'campaign': False, 'learning': True,
                        'R_depletion': False, 'test': False,
                        'interaction': interaction,
                        'crs': True}

    mm = mean_model(*init_conditions, **input_parameters)
    mm.run(t_max=200)
    mm.R_depletion = True
    mm.run(t_max=600)
    trj_m = mm.get_unified_trajectory()

    ma = aggregate_model(*init_conditions, **input_parameters)
    ma.run(t_max=200)
    ma.R_depletion = True
    ma.run(t_max=600)
    trj_a = ma.get_unified_trajectory()

    for c in trj_m.columns:
        dif = trj_m[[c]] - trj_a[[c]]
        dif_cum = dif.cumsum()
        max_dif = dif_cum.max().values[0]
        print(max_dif)
        assert max_dif < 1e-3, 'failed at {} with interaction={}'\
            .format(c, interaction)
