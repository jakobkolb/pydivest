"""testing the mean approximation module by comparing it to the original reference implementation"""


from random import shuffle

import networkx as nx
import numpy as np
from pydivest.macro_model.integrate_equations_mean \
    import IntegrateEquationsMean as NewModel
from pydivest.macro_model.integrate_equations_test_reference \
    import Integrate_Equations as OldModel

# investment_decisions:

nopinions = [50, 50]

# Parameters:

phi, b_d = 0.4, 1.25

input_parameters = {'b_c': 1., 'phi': phi, 'tau': 1.,
                    'eps': 0.05, 'b_d': b_d, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100.,
                    'possible_que_orders': [[0], [1]],
                    'xi': 1. / 8., 'beta': 0.06,
                    'L': 100., 'C': 100., 'G_0': 800.,
                    'campaign': False, 'learning': True,
                    'R_depletion': False, 'test': False,
                    'interaction': 0}

# investment_decisions
opinions = []
for i, n in enumerate(nopinions):
    opinions.append(np.full(n, i, dtype='I'))
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

new = NewModel(*init_conditions, **input_parameters)

m_old = OldModel(*init_conditions, **input_parameters)
m_old.run(t_max=200)
m_old.R_depletion = True
m_old.run(t_max=600)
trj_old = m_old.get_m_trajectory()

m_new = NewModel(*init_conditions, **input_parameters)
m_new.run(t_max=200)
m_new.R_depletion = True
m_new.run(t_max=600)
trj_new = m_new.get_mean_trajectory()

trj_diff = trj_new[['x', 'y', 'z']] - trj_old[['x', 'y', 'z']]
trj_abs = trj_diff.abs().sum(axis=1)
cs = trj_abs.cumsum()
max_diff = cs.max()

assert max_diff < 1e-5
