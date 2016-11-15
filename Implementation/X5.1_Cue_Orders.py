"""
This experiment focuses on the test of different cue orders for
the Take The Best Heuristic. All other parameters are set to
standard values or determined by the following timescales:

1) capital accumulation in the dirty sector,
    t_d = 1/(d_c*(1-kappa_c))
2) depletion of the fossil resource and
    t_G = G_0*e*d_c/(P*s*b_d**2)
3) opinion spreading in the adaptive voter model
   given one opinion dominates the other.
    t_a = tau*(1-phi)

tau and phi are of no interest so far, since the adaptive
voter dynamics is disabled so far.
t_d is fixed by standard values for d_c=0.06
and kappa_c=0.5.

Focus in terms of timescales is set to the last two remaining
degrees of freedom:

t_G sets the value for G_0 as all other parameters
are assumed to be fixed.

the ratio alpha = b_R/e<0 determines the share of the initial
resource that can be economically harvested.
"""

from pymofa.experiment_handling import (experiment_handle,
                                        even_time_series_spacing)
from divestcore import divestment_core as model
from divestvisuals.data_visualization import plot_obs_grid
from random import shuffle
import numpy as np
import scipy.stats as st
import networkx as nx
import pandas as pd
import cPickle as cp
import itertools as it
import sys
import getpass
import time
import types


def RUN_FUNC(t_G, nopinions, alpha,
             possible_opinions, eps, avm, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    t_G : float
        timescale of fossil resource depletion
        in a full fledged dirty economy
        input is given in relation to t_c
        such that the actual depletion time is
        t_G*t_c
    nopinions : list of integers
        integer value indicating the number of
        households that hold a specific opinion.
        N = sum(opinions)
    alpha: float
        the ratio alpha = (b_R0/e)**(1/2)
        that sets the share of the initial
        resource G_0 that can be harvested
        economically.
    possible_opinions : list of list of integers
        the set of cue orders that are allowed in the
        model. opinions determine the individual cue
        order, that a household uses.
    eps : float
        fraction of rewiring events that are random.
    avm: bool
        switch for adaptive voter dynamics in the model
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """
    assert isinstance(test, types.IntType),\
        'test must be int, is {!r}'.format(test)
    assert alpha < 1,\
        'alpha must be 0<alpha<1. is alpha = {}'.format(alpha)

    (N, p, tau, phi, P, b_d, b_R0, e, d_c, s) =\
        (sum(nopinions), 0.125, .1, .8, 500, 1.2, 1., 100, 0.06, 0.23)

    # capital accumulation of dirty capital
    # (t_d = 1/(d_c*(1-kappa_c)) with kappa_c = 0.5 :
    t_d = 1/(2.*d_c)

    # Rescale input times to capital accumulation time:
    t_G = t_G*t_d

    # set G_0 according to resource depletion time:
    # t_G = G_0*e*d_c/(P*s*b_d**2)
    G_0 = t_G*P*s*b_d**2/(e*d_c)

    # set b_R0 according to alpha and e:
    # alpha = (b_R0/e)**(1/2)
    b_R0 = alpha**2 * e

    # input parameters

    input_params = {
            'possible_opinions': possible_opinions,
            'tau': tau, 'phi': phi, 'eps': eps,
            'P': P, 'b_d': b_d, 'b_R0': b_R0, 'G_0': G_0,
            'e': e, 'd_c': d_c, 'test': bool(test)}

    # building initial conditions

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full(n, i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    init_conditions = (adjacency_matrix, opinions)

    # initializing the model

    m = model.divestment_core(*init_conditions, **input_params)
    if not avm:
        m.mode = 1

    # storing initial conditions and parameters

    res = {}
    res["initials"] = {
            "adjacency matrix": adjacency_matrix,
            "opinions": opinions,
            "possible opinions": possible_opinions}

    res["parameters"] = \
        pd.Series({"tau": m.tau,
                   "phi": m.phi,
                   "N": m.N,
                   "p": p,
                   "P": m.P,
                   "birth rate": m.r_b,
                   "savings rate": m.s,
                   "clean capital depreciation rate": m.d_c,
                   "dirty capital depreciation rate": m.d_d,
                   "resource extraction efficiency": m.b_R0,
                   "Solov residual clean": m.b_c,
                   "Solov residual dirty": m.b_d,
                   "pi": m.pi,
                   "kappa_c": m.kappa_c,
                   "kappa_d": m.kappa_d,
                   "rho": m.rho,
                   "resource efficiency": m.e,
                   "epsilon": m.eps,
                   "initial resource stock": m.G_0})

    # run the model
    if test:
        print input_params

    t_max = 300 if test == 0 else 50
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    # store exit status
    res["convergence"] = exit_status
    if test:
        print 'test output of variables'
        print (m.tau, m.phi, exit_status,
               m.convergence_state, m.convergence_time)
    # store data in case of successful run

    if exit_status in [0, 1]:
        res["convergence_data"] = \
                pd.DataFrame({"Investment decisions": m.investment_decisions,
                              "Investment clean": m.investment_clean,
                              "Investment dirty": m.investment_dirty})
        res["convergence_state"] = m.convergence_state
        res["convergence_time"] = m.convergence_time

        # interpolate trajectory to get evenly spaced time series.
        trajectory = m.trajectory
        headers = trajectory.pop(0)

        df = pd.DataFrame(trajectory, columns=headers)
        df = df.set_index('time')
        dfo = even_time_series_spacing(df, 101, 0., t_max)
        res["economic_trajectory"] = dfo

    end = time.clock()
    res["runtime"] = end-start

    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)

    return exit_status

# get sub experiment and mode from command line
if len(sys.argv) > 1:
    mode = int(sys.argv[1])     # sets mode (1:production, 2:test, 3:messy)
else:
    mode = 3
if len(sys.argv) > 2:
    noise = bool(int(sys.argv[2]))
else:
    noise = False

folder = 'X5.1_Cue_Orders'

# check if cluster or local
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW =\
        "/p/tmp/kolb/Divest_Experiments/divestdata/"\
        + folder + "/raw_data"
    SAVE_PATH_RES =\
        "/home/kolb/Divest_Experiments/divestdata/"\
        + folder + "/results"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"\
        + folder + "/raw_data"
    SAVE_PATH_RES = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"\
        + folder + "/results"

"""
Make different types of decision makers. Cues are
"""
cue_names = {
        0: 'always dirty',
        1: 'always clean',
        2: 'capital rent',
        3: 'capital rent trend',
        4: 'peer pressure'}

opinion_presets = [[2, 3],  # short term investor
                   [3, 2],  # long term investor
                   [4, 2],  # short term herder
                   [4, 3],  # trending herder
                   [4, 1],  # green conformer
                   [4, 0],  # dirty conformer
                   [1],     # gutmensch
                   [0]]     # redneck
"""
set different times for resource depletion
in units of capital accumulation time t_d = 1/(d_c*(1-kappa_d))
"""
t_Gs = [round(x, 5) for x in list(10**np.linspace(1.0, 2.0, 3))]

"""
Define different mixtures of decision makers to test
"""
opinions = [
        [100, 0, 0, 0, 0, 0, 0, 0],
        [0, 100, 0, 0, 0, 0, 0, 0],
        [0, 0, 100, 0, 0, 0, 0, 0],
        [0, 0, 0, 100, 0, 0, 0, 0],
        [0, 0, 0, 0, 100, 0, 0, 0],
        [0, 0, 0, 0, 0, 100, 0, 0],
        [0, 0, 0, 0, 0, 0, 100, 0],
        [0, 0, 0, 0, 0, 0, 0, 100]]
"""
Define set of alphas that will be tested against the sets of resource depletion
times and cue order mixtures
"""
alphas = [round(x, 5) for x in list(10**np.linspace(-3.0, -1.0, 3))]

"""
dictionary of the variable parameters in this experiment together with their
position in the index of the dictionary of results
"""
parameters = {
        't_G': 0,
        'cue_order': 1,
        'alpha': 2,
        'test': 3}
"""
Default values of variable parameter in this experiment
"""
t_G, cue_order, alpha, test = [5.], [2, 3], [0.001], [0]

NAME = 'Cue_order_testing'
INDEX = {
        0: "t_G",
        parameters['cue_order']: "cue_order",
        parameters['alpha']: "alpha"}
"""
set eps according to nose settings
"""
if noise:
    eps, avm = [0.05], [True]
    SAVE_PATH_RAW += '_N/'
    SAVE_PATH_RES += '_N/'
else:
    eps, avm = [0.0], [False]
    SAVE_PATH_RAW += '_NN/'
    SAVE_PATH_RES += '_NN/'

"""
create list of parameter combinations for
different experiment modes.
Make sure, opinion_presets are not expanded
"""
if mode == 1:
    PARAM_COMBS = list(it.product(
            t_Gs, opinions, alphas, [opinion_presets], eps, avm, test))

elif mode == 2:
    PARAM_COMBS = list(it.product(
            t_Gs, opinions, alphas, [opinion_presets], eps, avm, test))
elif mode == 3:
    PARAM_COMBS = list(it.product(
            t_Gs, opinions, alpha, [opinion_presets], eps, avm, test))
else:
    print mode, ' is not a valid experiment mode.\
    valid modes are 1: production, 2: test, 3: messy'
    sys.exit()

# names and function dictionaries for post processing:


NAME1 = NAME+'_trajectory'
EVA1 = {"<mean_trajectory>":
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                  for f in fnames]).groupby(level=0).mean(),
        "<sem_trajectory>":
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                  for f in fnames]).groupby(level=0).sem()
        }

NAME2 = NAME+'_convergence'
EVA2 = {"<mean_convergence_state>":
        lambda fnames: np.nanmean([np.load(f)["convergence_state"]
                                   for f in fnames]),
        "<mean_convergence_time>":
        lambda fnames: np.nanmean([np.load(f)["convergence_time"]
                                   for f in fnames]),
        "<min_convergence_time>":
        lambda fnames: np.nanmin([np.load(f)["convergence_time"]
                                  for f in fnames]),
        "<max_convergence_time>":
        lambda fnames: np.max([np.load(f)["convergence_time"]
                               for f in fnames]),
        "<nanmax_convergence_time>":
        lambda fnames: np.nanmax([np.load(f)["convergence_time"]
                                  for f in fnames]),
        "<sem_convergence_time>":
        lambda fnames: st.sem([np.load(f)["convergence_time"]
                               for f in fnames]),
        "<runtime>":
        lambda fnames: st.sem([np.load(f)["runtime"]
                               for f in fnames]),
        }

# full run
if mode == 1:
    SAMPLE_SIZE = 100
    handle = experiment_handle(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets)

# test run
if mode == 2:
    SAMPLE_SIZE = 2
    handle = experiment_handle(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets)

# debug and mess around mode:
if mode == 3:
    SAMPLE_SIZE = 2
    handle = experiment_handle(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets)
