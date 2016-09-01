"""
This experiment is dedicated to finding the dirty equilibrium of
the system. Thus, the fossil resource is assumed to be infinite 
and the system is run with noise and adaptive voter dynamics.

Variable parameters are:

1) alpha, as it indicates the proximity of the
   initial state to the depreciation of the fossil
   resource,

2) phi, as it governs the clustering amongst similar
   opinions,

3) d_c as it sets the timescale for capital accumulation
   and is therefore thought to change the qualitative
   nature of the transition.
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


def RUN_FUNC(t_a, phi, alpha,
             possible_opinions, eps, transition, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    t_a : float
        Timescale of opinion spreading given
        one opinion dominates. Timescale is
        given in units of the timescale of
        capital accumulation t_c
        such that the actual depletion time is
        t_a*t_c
    phi : list of integers
        rewiring probability of the adaptive voter
        dynamics. Governs the clustering in the
        network of households.
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
    transition: bool
        switch for resource depletion
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

    (N, p, P, b_d, b_R0, e, d_c, s) =\
        (100, 0.125, 500, 1.2, 1., 100, 0.06, 0.23)

    # capital accumulation of dirty capital
    # (t_d = 1/(d_c*(1-kappa_c)) with kappa_c = 0.5 :
    t_d = 1/(2.*d_c)

    # Rescale input times to capital accumulation time:
    t_a = t_a*t_d

    # set tau according to t_a and phi
    tau = t_a/(1.-phi)

    # set t_G to some value approx. half of run time
    t_G = 100*t_d

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
            'e': e, 'd_c': d_c, 'test': bool(test),
            'R_depletion': transition}

    # building initial conditions

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    opinions = [1 for x in range(N)]
    investment_clean = np.ones(N)
    investment_dirty = np.ones(N)

    init_conditions = (adjacency_matrix,
                       opinions,
                       investment_clean,
                       investment_dirty)

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

    t_max = 300 if transition else 3000
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    # save final state of the model
    final_state = m.final_state
    with open(filename + '_final', 'wb') as dumpfile:
        cp.dump(final_state, dumpfile)

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
    transition = [bool(int(sys.argv[2]))]
else:
    transition = [False]

folder = 'X5.2_Dirty_Equilibrium'

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
t_as = [round(x, 5) for x in list(10**np.linspace(0.1, 100.0, 4))]

"""
set array of phis to generate equilibrium conditions for
"""
phis = [round(x, 2) for x in list(np.linspace(0.0, 1.0, 11))[:-1]]

"""
Define set of alphas that will be tested against the sets of resource depletion
times and cue order mixtures
"""
alphas = [round(x, 5) for x in list(10**np.linspace(-3.0, -1.0, 2))]

"""
dictionary of the variable parameters in this experiment together with their
position in the index of the dictionary of results
"""
parameters = {
        't_a': 0,
        'phi': 1,
        'alpha': 2,
        'test': 3}
"""
Default values of variable parameter in this experiment
"""
t_a, phi, alpha, test = [0.1], [0.8], [0.001], [0]

NAME = 'Cue_order_testing'
INDEX = {
        0: "t_a",
        parameters['phi']: "phi",
        parameters['alpha']: "alpha"}
"""
set eps according to nose settings
"""
eps, avm = [0.05], [True]
SAVE_PATH_RAW += '_N/'
SAVE_PATH_RES += '_N/'

"""
create list of parameter combinations for
different experiment modes.
Make sure, opinion_presets are not expanded
"""
if mode == 1:
    PARAM_COMBS = list(it.product(
            t_as, phis, alphas, [opinion_presets], eps, transition, test))

elif mode == 2:
    PARAM_COMBS = list(it.product(
            t_a, phis, alphas, [opinion_presets], eps, transition, test))
elif mode == 3:
    PARAM_COMBS = list(it.product(
            t_a, phis, alpha, [opinion_presets], eps, transition, test))
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
    SAMPLE_SIZE = 100
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
