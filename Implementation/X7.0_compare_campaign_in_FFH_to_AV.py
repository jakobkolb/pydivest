"""
Compare campaign dynamics in the full Fast and Frugal Heuristics mode to
the strapped down "adaptive voter" case.
Both cases have equal parameters and comparable initial conditions (equilibrium
with abundant fossil resource) as well as the same campaign starting size and
dynamics.
"""

import cPickle as cp
import getpass
import glob
import itertools as it
import numpy as np
import sys
import time

import networkx as nx
import pandas as pd

from micro_model import divestment_core as micro_model
from pymofa.experiment_handling import experiment_handling, \
    even_time_series_spacing
from divestvisuals.data_visualization import plot_trajectories, plot_amsterdam


def RUN_FUNC(b_d, phi, ffh, test, transition, filename):
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
    ffh: bool
        if True: run with fast and frugal heuristics
        if False: run with imitation only
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # fraction of households that will start the campaign
    ccount = .1

    # Make different types of decision makers. Cues are

    cue_names = {
        0: 'always dirty',
        1: 'always clean',
        2: 'capital rent',
        3: 'capital rent trend',
        4: 'peer pressure',
        5: 'campaignee'}
    if ffh:
        possible_opinions = [[2, 3],  # short term investor
                             [3, 2],  # long term investor
                             [4, 2],  # short term herder
                             [4, 3],  # trending herder
                             [4, 1],  # green conformer
                             [4, 0],  # dirty conformer
                             [1],  # gutmensch
                             [0]]  # redneck
    else:
        possible_opinions = [[1], [0]]

    # Parameters:

    input_params = {'b_c': 1., 'phi': phi, 'tau': 1.,
                    'eps': 0.05, 'b_d': b_d, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100., # alpha^2 * e
                    'possible_opinions': possible_opinions,
                    'xi': 1. / 8., 'beta': 0.06,
                    'P': 100., 'C': 100., 'G_0': 800.,
                    'campaign': False, 'learning': True,
                    'test': test, 'R_depletion': False}

    # building initial conditions

    if not transition:
        # network:
        N = 100
        k = 10
        if test:
            N = 30
            k = 3

        p = float(k) / N
        while True:
            net = nx.erdos_renyi_graph(N, p)
            if len(list(net)) > 1:
                break
        adjacency_matrix = nx.adj_matrix(net).toarray()

        # opinions and investment

        opinions = [np.random.randint(0, len(possible_opinions))
                    for x in range(N)]
        clean_investment = np.ones(N) * 50. / float(N)
        dirty_investment = np.ones(N) * 50. / float(N)

        init_conditions = (adjacency_matrix, opinions,
                           clean_investment, dirty_investment)

        t_1 = 500
        t_2 = 0

        # initializing the model
        m = micro_model.Divestment_Core(*init_conditions, **input_params)

    elif transition:
        # build list of initial conditions
        # phi, alpha and t_d are relevant,
        # t_a is not. Parse filename to get
        # wildcard for all relevant files.
        # replace characters before first
        # underscore with *
        [path, fname] = filename.rsplit('/', 1)
        path += '/*' + fname.rsplit('True', 1)[0] + 'False_*.pkl'
        init_files = glob.glob(path)
        input_params = np.load(
            init_files[np.random.randint(0, len(init_files))])

        # update input parameters where necessary
        input_params['campaign'] = True
        input_params['possible_opinions'].append([5])
        campaigner = len(input_params['possible_opinions']) - 1

        # make fraction of ccount households campaigners
        opinions = input_params['opinions']
        nccount = int(ccount * len(opinions))
        j = 0
        while j < nccount:
            n = np.random.randint(0, len(opinions))
            if opinions[n] != campaigner:
                opinions[n] = campaigner
                j += 1
        input_params['opinions'] = opinions

        t_1 = 200
        t_2 = 600

        # initializing the model
        m = micro_model.Divestment_Core(**input_params)

    # storing initial conditions and parameters
    res = {
        "initials": pd.DataFrame({"opinions": opinions,
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

    # start timer
    t_start = time.clock()

    # run model with abundant resource
    t_max = t_1 #if not test else 1
    m.R_depletion = False
    m.run(t_max=t_max)

    # run model with resource depletion
    t_max += t_2 #if not test else 1
    m.R_depletion = True
    exit_status = m.run(t_max=t_max)

    # for equilibration runs, save final state of the model:
    if not transition:
        final_state = m.final_state
        with open(filename, 'wb') as dumpfile:
            cp.dump(final_state, dumpfile)

    else:
        res["runtime"] = time.clock() - t_start

        # store data in case of successful run
        if exit_status in [0, 1]:
            res["micro_trajectory"] = \
                even_time_series_spacing(m._get_e_trj(), 401, 0., t_max)
            res["convergence_state"] = m.convergence_state
            res["convergence_time"] = m.convergence_time

        # save data
        with open(filename, 'wb') as dumpfile:
            cp.dump(res, dumpfile)
        try:
            tmp = np.load(filename)
        except IOError:
            print "writing results failed for " + filename

    return exit_status


# get sub experiment and mode from command line

# switch decision making
if len(sys.argv) > 1:
    ffh = bool(int(sys.argv[1]))
else:
    ffh = True
# switch transition
if len(sys.argv) > 2:
    transition = bool(int(sys.argv[2]))
else:
    transition = False
# switch testing mode
if len(sys.argv) > 3:
    test = bool(int(sys.argv[3]))
else:
    test = False
#switch experiment mode
if len(sys.argv) > 4:
    mode = int(sys.argv[4])
else:
    mode = 0

experiment = ['imitation', 'ffh'][int(ffh)] + ['_full', '_test'][int(test)]

folder = 'X7_' + experiment + '_trajectory'
NAME = 'b_c_scan_' + experiment + '_trajectory'

# check if cluster or local
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Divest_Experiments/divestdata/" \
                    + folder + "/raw_data" + '_' + experiment + '/'
    SAVE_PATH_RES = "/home/kolb/Divest_Experiments/divestdata/" \
                    + folder + "/results" + '_' + experiment + '/'
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/" \
        + folder + "/raw_data" + '_' + experiment + '/'
    SAVE_PATH_RES = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/" \
        + folder + "/results" + '_' + experiment + '/'

# create parameter combinations
phis = [round(x, 5) for x in list(np.linspace(0.0, 0.9, 10))]
b_ds = [round(x, 5) for x in list(np.linspace(1., 1.5, 3))]
b_d, phi = [1.2], [.8]

if test:
    PARAM_COMBS = list(it.product(b_d, phi, [ffh], [test], [transition]))
else:
    PARAM_COMBS = list(it.product(b_ds, phis, [ffh], [test], [transition]))

INDEX = {0: "b_c", 1: "phi"}

# names and function dictionaries for post processing:
NAME1 = NAME + '_trajectory'
EVA1 = {"mean_trajectory":
            lambda fnames: pd.concat([np.load(f)["micro_trajectory"]
                                      for f in fnames]).groupby(
                level=0).mean(),
        "sem_trajectory":
            lambda fnames: pd.concat([np.load(f)["micro_trajectory"]
                                      for f in fnames]).groupby(level=0).std()}
NAME2 = NAME + '_convergence_times'
CF2 = {'times':
           lambda fnames: pd.DataFrame(data=[np.load(f)["convergence_state"] \
                                for f in fnames]).sortlevel(level=0),
       'states':
           lambda fnames: pd.DataFrame(data=[np.load(f)["convergence_state"] \
                                for f in fnames]).sortlevel(level=0)
       }

# full run
if mode == 0:
    print 'mode 0'
    sys.stdout.flush()
    SAMPLE_SIZE = 100
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    if transition:
        handle.resave(EVA1, NAME1)
        handle.collect(CF2, NAME2)
        plot_trajectories(SAVE_PATH_RES, NAME1, None, None)

# test run
if mode == 1:
    print 'mode 1'
    sys.stdout.flush()
    SAMPLE_SIZE = 10
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    if transition:
        handle.resave(EVA1, NAME1)
        handle.collect(CF2, NAME2)
        plot_amsterdam(SAVE_PATH_RES, NAME1)
        plot_trajectories(SAVE_PATH_RES, NAME1, None, None)


# debug and mess around mode with plotting:
if mode is 3:
    print 'mode 2 - plotting'
    print 'short is ', test
    sys.stdout.flush()
    SAMPLE_SIZE = 2
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                 SAVE_PATH_RAW, SAVE_PATH_RES)
    print SAMPLE_SIZE
    if test is True:
        handle.compute(RUN_FUNC)
        handle.resave(EVA1, NAME1)
        handle.collect(CF2, NAME2)
        # plot_m_trajectories(SAVE_PATH_RES, NAME1)
        #plot_trajectories(SAVE_PATH_RES, NAME1, None, None)
