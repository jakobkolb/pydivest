import cPickle as cp
import getpass
import itertools as it
import numpy as np
import sys
import time

import networkx as nx
import pandas as pd

from macro_model import integrate_equations as macro_model
from micro_model import divestment_core as micro_model
from pymofa.experiment_handling import experiment_handling, \
    even_time_series_spacing


def RUN_FUNC(b_c, phi, approximate, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the 
    initial values, parameters and convergence state and time 
    for each run.

    Parameters:
    -----------
    b_c : float > 0
        the solow residual in the clean sector
    phi : float \in [0,1]
        the rewiring probability for the network update
    approximate: bool
        if True: run macroscopic approximation
        if False: run micro-model
    test: int \in [0,1]
        wheter this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Parameters:

    input_params = {'b_c': b_c, 'phi': phi, 'tau': 1,
                    'eps': 0.05, 'b_d': 1.2, 'e': 100,
                    'G_0': 30000, 'b_r0': 0.1 ** 2 * 100,
                    'possible_opinions': [[0], [1]],
                    'C': 100, 'xi': 1. / 8., 'beta': 0.06,
                    'campaign': False, 'learning': True}

    # investment_decisions:
    nopinions = [100, 100]

    # network:
    N = sum(nopinions)
    k = 10

    # building initial conditions
    p = float(k) / N
    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=N)

    clean_investment = np.ones(N)
    dirty_investment = np.ones(N)

    init_conditions = (adjacency_matrix, investment_decisions,
                       clean_investment, dirty_investment)

    # initializing the model
    print 'approximate', approximate
    if approximate:
        m = macro_model.integrate_equations(*init_conditions, **input_params)
    elif not approximate:
        m = micro_model.divestment_core(*init_conditions, **input_params)

    # storing initial conditions and parameters

    res = {
        "initials": pd.DataFrame({"Investment decisions": investment_decisions,
                                  "Investment clean": m.investment_clean,
                                  "Investment dirty": m.investment_dirty}),
        "parameters": pd.Series({"tau": m.tau,
                                 "phi": m.phi,
                                 "N": m.n,
                                 "P": m.P,
                                 "savings rate": m.s,
                                 "clean capital depreciation rate": m.d_c,
                                 "dirty capital depreciation rate": m.d_d,
                                 "resource extraction efficiency": m.b_r0,
                                 "Solov residual clean": m.b_c,
                                 "Solov residual dirty": m.b_d,
                                 "pi": m.pi,
                                 "kappa_c": m.kappa_c,
                                 "kappa_d": m.kappa_d,
                                 "xi": m.xi,
                                 "resource efficiency": m.e,
                                 "epsilon": m.eps,
                                 "initial resource stock": m.G_0})}

    # run the model
    t_start = time.clock()
    t_max = 200
    m.R_depletion = False
    m.run(t_max=t_max)
    t_max += 200
    m.R_depletion = True
    m.run(t_max=t_max)

    # store exit status
    res["runtime"] = time.clock() - t_start

    # store data in case of successful run

    if exit_status in [0, 1]:
        # interpolate m_trajectory to get evenly spaced time series.
        df = m.m_trajectory
        dfo = even_time_series_spacing(df, 101, 0., t_max)
        res["economic_trajectory"] = dfo

    end = time.clock()
    res["runtime"] = end - start

    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
    except IOError:
        print "writing results failed for " + filename

    return exit_status


# get sub experiment and mode from command line
if len(sys.argv) > 1:
    input_int = int(sys.argv[1])
else:
    input_int = -1
if len(sys.argv) > 2:
    mode = int(sys.argv[2])
else:
    mode = None
if len(sys.argv) > 3:
    test = [bool(sys.argv[3])]
else:
    test = [False]

experiments = ['micro', 'macro', 'short']
sub_experiment = experiments[input_int]
folder = 'X6' + sub_experiment + 'trajectory'

# check if cluster or local
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/P/tmp/kolb/Divest_Experiments/divestdata/" \
                    + folder + "/raw_data" + '_' + sub_experiment + '/'
    SAVE_PATH_RES = "/home/kolb/Divest_Experiments/divestdata/" \
                    + folder + "/results" + '_' + sub_experiment + '/'
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/" \
        + folder + "/raw_data" + '_' + sub_experiment + '/'
    SAVE_PATH_RES = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/" \
        + folder + "/results" + '_' + sub_experiment + '/'

phis = [round(x, 5) for x in list(np.linspace(0.0, 1.0, 11))]

b_cs = [round(x, 5) for x in list(np.linspace(0.4, 2.0, 9))]

parameters = {'b_c': 0, 'phi': 1, 'macro_model': 2, 'test': 3}
b_c, phi, approximate, exact, test = \
    [1], [.8], [True], [False], [False]

NAME = 'b_c_scan_' + sub_experiment + '_trajectory'
INDEX = {0: "b_c", 1: "phi"}

if sub_experiment == 'micro':
    PARAM_COMBS = list(it.product(b_cs, phis, exact, test))

elif sub_experiment == 'macro':
    PARAM_COMBS = list(it.product(b_cs, phis, approximate, test))

elif sub_experiment == 'short':
    PARAM_COMBS = list(it.product(b_c, phi, approximate, test))

else:
    print sub_experiment, ' is not in the list of possible experiments'
    sys.exit()

# names and function dictionaries for post processing:
NAME1 = NAME + '_trajectory'
EVA1 = {"<mean_trajectory>":
            lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                      for f in fnames]).groupby(
                level=0).mean(),
        "<sem_trajectory>":
            lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                      for f in fnames]).groupby(level=0).sem()}
# full run
if mode == 0:
    print 'mode 0'
    SAMPLE_SIZE = 100 if sub_experiment == 'micro' else 3
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)

# test run
if mode == 1:
    print 'mode 1'
    SAMPLE_SIZE = 2 if sub_experiment == 'micro' else 3
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)

# debug and mess around mode:
if mode is None:
    print 'mode 2'
    SAMPLE_SIZE = 2 if sub_experiment == 'micro' else 3
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
