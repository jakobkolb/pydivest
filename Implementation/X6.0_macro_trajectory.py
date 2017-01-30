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
from divestvisuals.data_visualization import \
    plot_m_trajectories, plot_trajectories


def RUN_FUNC(b_d, phi, approximate, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the 
    initial values, parameters and convergence state and time 
    for each run.

    Parameters:
    -----------
    b_d : float > 0
        the solow residual in the dirty sector
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

    input_params = {'b_c': 1., 'phi': phi, 'tau': 1.,
                    'eps': 0.05, 'b_d': b_d, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100.,
                    'possible_opinions': [[0], [1]],
                    'xi': 1. / 8., 'beta': 0.06,
                    'P': 100., 'C': 100., 'G_0': 800.,
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

    clean_investment = np.ones(N) * 50. / float(N)
    dirty_investment = np.ones(N) * 50. / float(N)

    init_conditions = (adjacency_matrix, investment_decisions,
                       clean_investment, dirty_investment)

    # for testing reasons, I saved one set of initial conditions
    # with open('init.pkl', 'wb') as initfile:
    #    cp.dump(init_conditions, initfile)
    if test is True:
        print 'loading initial conditions'
        init_conditions = np.load('init.pkl')

    # initializing the model
    print 'approximate', approximate
    if approximate:
        m = macro_model.Integrate_Equations(*init_conditions, **input_params)
    elif not approximate:
        m = micro_model.Divestment_Core(*init_conditions, **input_params)

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

    t_max = 200 if not test else 1
    print t_max
    m.R_depletion = False
    exit_status = m.run(t_max=t_max)

    t_max += 400 if not test else 1
    print t_max
    m.R_depletion = True
    exit_status = m.run(t_max=t_max)

    res["runtime"] = time.clock() - t_start

    # store data in case of successful run
    if exit_status in [0, 1]:
        # interpolate m_trajectory to get evenly spaced time series.
        res["macro_trajectory"] = \
            even_time_series_spacing(m._get_m_trj(), 201, 0., t_max)
        res["switchlist"] = m._get_switch_list()

    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
    except IOError:
        print "writing results failed for " + filename

    return exit_status


# get sub experiment and mode from command line

# experiment, mode, test

if len(sys.argv) > 1:
    input_int = int(sys.argv[1])
else:
    input_int = -1
if len(sys.argv) > 2:
    mode = int(sys.argv[2])
else:
    mode = 0
if len(sys.argv) > 3:
    test = [bool(sys.argv[3])]
else:
    test = [False]

experiments = ['micro', 'macro', 'macroshort', 'microshort']
sub_experiment = experiments[input_int]
folder = 'X6' + sub_experiment + 'trajectory'

if test[0] is True:
    print sub_experiment, mode, test

# check if cluster or local
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Divest_Experiments/divestdata/" \
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

phis = [round(x, 5) for x in list(np.linspace(0.0, 0.9, 10))]

b_ds = [round(x, 5) for x in list(np.linspace(1., 1.5, 3))]

parameters = {'b_c': 0, 'phi': 1, 'macro_model': 2, 'test': 3}
b_d, phi, approximate, exact = \
    [1.2], [.8], [True], [False]

NAME = 'b_c_scan_' + sub_experiment + '_trajectory'
INDEX = {0: "b_c", 1: "phi"}

if sub_experiment == 'micro':
    PARAM_COMBS = list(it.product(b_ds, phis, exact, test))

elif sub_experiment == 'macro':
    PARAM_COMBS = list(it.product(b_ds, phis, approximate, test))

elif sub_experiment == 'macroshort' and test[0] is False:
    PARAM_COMBS = list(it.product(b_d, phis, approximate, test))

elif sub_experiment == 'microshort' and test[0] is False:
    PARAM_COMBS = list(it.product(b_d, phis, exact, test))

elif sub_experiment == 'macroshort' and test[0] is True:
    print '### testing mode ### macro'
    PARAM_COMBS = list(it.product(b_d, phi, approximate, test))

elif sub_experiment == 'microshort' and test[0] is True:
    print '### testing mode ### micro'
    PARAM_COMBS = list(it.product(b_d, [0.2, 0.7, 0.8], exact, test))

else:
    print sub_experiment, ' is not in the list of possible experiments'
    sys.exit()

# names and function dictionaries for post processing:
NAME1 = NAME + '_trajectory'
EVA1 = {"mean_trajectory":
            lambda fnames: pd.concat([np.load(f)["macro_trajectory"]
                                      for f in fnames]).groupby(
                level=0).mean(),
        "sem_trajectory":
            lambda fnames: pd.concat([np.load(f)["macro_trajectory"]
                                      for f in fnames]).groupby(level=0).std()}
NAME2 = NAME + '_switchlist'
CF2 = lambda fnames: pd.concat([np.load(f)["switchlist"] \
                                for f in fnames]).sortlevel(level=0)

# full run
if mode == 0:
    print 'mode 0'
    sys.stdout.flush()
    SAMPLE_SIZE = 100 if sub_experiment == 'micro' else 2
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.collect(CF2, NAME2)

# test run
if mode == 1:
    print 'mode 1'
    sys.stdout.flush()
    SAMPLE_SIZE = 2 if sub_experiment == 'micro' else 2
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)

# debug and mess around mode with plotting:
if mode is 3:
    print 'mode 2 - plotting'
    print 'short is ', test
    sys.stdout.flush()
    SAMPLE_SIZE = 2 if sub_experiment == 'microshort' else 2
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    print SAMPLE_SIZE
    if test[0] is True:
        handle.compute(RUN_FUNC)
        handle.resave(EVA1, NAME1)
        handle.collect(CF2, NAME2)
        # plot_m_trajectories(SAVE_PATH_RES, NAME1)
        #plot_trajectories(SAVE_PATH_RES, NAME1, None, None)
