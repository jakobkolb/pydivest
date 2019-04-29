"""
This experiment is designed to test different timescales in the
model againts each other. Nameley the timescales for
1) capital accumulation in the dirty sector,
    t_d = 1/(d_c*(1-kappa_c))
2) depletion of the fossil resource and
    t_G = G_0*e*d_c/(L*s*b_d**2)
3) opinion spreading in the adaptive voter model
   given one opinion dominates the other.
    t_a = tau*(1-phi)
for this purpose, t_d is fixed by standard values for 
d_c=0.06 and kappa_c=0.5, and the 'consensus time' t_a
is varied in units of the capital accumulation time t_d.
This sets tau as phi is independently varied between 0 and 1.
Therefore, the tau-phi plot from former experiments becomes
a t_a-phi plot in this experiment.

t_G is varied as the third parameter, setting the value for G_0 as
all other parameters are assumed to be fixed.

A fourth quantity of interest is the ratio alpha = b_R/e<0
that the share of the initial resource that can be economically
harvested. This ratio is set to different values in different 
experiments to examine its qualitative role in the resource 
depletion process.

"""

# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3



import pickle as cp
import getpass
import glob
import itertools as it
import sys
import time
import types

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st

from pymofa.experiment_handling \
    import experiment_handling, even_time_series_spacing

from pydivest.divestvisuals.data_visualization \
    import plot_obs_grid, plot_tau_phi, tau_phi_final
from pydivest.micro_model import divestmentcore as model


def load(*args, **kwargs):
    return np.load(*args, allow_pickle=True, **kwargs)


def RUN_FUNC(t_a, phi, eps, t_G, alpha, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    t_a : float
        timescale for opinion spreading given
        that one opinion dominates the other
        t_a = tau*(1-phi)
        input is given in relation to t_c
        such that the actual opinion spreading
        time is t_a*t_c
    phi : float
        rewiring probability of the adaptive
        voter dynamics
    eps : float
        fraction of rewiring and imitation
        events that are noise (random)
    t_G : float
        timescale of fossil resource depletion
        in a full fledged dirty economy
        input is given in relation to t_c
        such that the actual depletion time is
        t_G*t_c
    alpha: float
        the ratio alpha = (b_R0/e)**(1/2)
        that sets the share of the initial
        resource G_0 that can be harvested
        economically.
    test: int \in [0,1]
        wheter this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """
    assert isinstance(test, int), 'test must be int, is {!r}'.format(test)
    assert alpha<1, 'alpha must be 0<alpha<1. is alpha = {}'.format(alpha)

    tau, N, p, P, b_d, b_R, e, d_c, s = .8, 100, 0.125, 500, 1.2, 1., 100, 0.06, 0.23

    # capital accumulation of dirty capital (t_d = 1/(d_c*(1-kappa_c)) with kappa_c = 0.5 :
    t_d = 1/(2.*d_c)

    # Rescale input times to capital accumulation time:
    t_G = t_G*t_d
    t_a = t_a*t_d

    # set tau according to consensus time in adaptive voter model if one opinion dominates:
    # t_a = tau*(1-phi)
    tau = t_a/(1-phi)

    # set G_0 according to resource depletion time:
    # t_G = G_0*e*d_c/(L*s*b_d**2)
    G_0 = t_G*P*s*b_d**2/(e*d_c)

    # set b_R0 according to alpha and e:
    # alpha = (b_R0/e)**(1/2)
    b_R0 = alpha**2 * e

    # input parameters

    input_params = {'tau':tau, 'phi':phi, 'eps':eps, \
            'L':P, 'b_d':b_d, 'b_R0':b_R0, 'G_0':G_0, \
            'e':e, 'd_c':d_c, 'test':bool(1)}

    # building initial conditions

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=N)

    init_conditions = (adjacency_matrix, investment_decisions)

    # initializing the model

    m = model.DivestmentCore(*init_conditions, **input_params)

    # storing initial conditions and parameters

    res = dict()

    # run the model
    print(test)
    if test:
        print(input_params)

    t_max = 300 if test == 0 else 5
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    # store exit status
    res["convergence"] = exit_status
    if test:
        print('test output of variables')
        print(m.tau, m.phi, exit_status,
              m.convergence_state, m.convergence_time)
    # store data in case of successful run

    if exit_status in [0, 1]:
        res["convergence_data"] = \
                pd.DataFrame({"Investment decisions": m.investment_decisions,
                              "Opinions": m.opinions,
                              "Investment clean": m.investment_clean,
                              "Investment dirty": m.investment_dirty})
        res["convergence_state"] = m.convergence_state
        res["convergence_time"] = m.convergence_time

        # interpolate trajectory to get evenly spaced time series.
        trajectory = m.get_economic_trajectory()
        dfo = even_time_series_spacing(trajectory, 101, 0., t_max)
        res["economic_trajectory"] = dfo

    end = time.clock()
    res["runtime"] = end-start

    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        tmp = load(filename)
    except IOError:
        print("writing results failed for " + filename)

    return exit_status

# get sub experiment and mode from command line
if len(sys.argv) > 1:
    input_int = int(sys.argv[1])  # determines the values of alpha
else:
    input_int = None
if len(sys.argv) > 2:
    noise = bool(int(sys.argv[2]))  # sets noise
else:
    noise = True
if len(sys.argv) > 3:
    mode = int(sys.argv[3])  # sets mode (production, test, messy)
else:
    mode = None

print(sys.argv)
test = mode

folder = 'X3Noise' if noise else 'X3NoNoise'

# check if cluster or local
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = '/p/tmp/kolb/Divest_Experiments/divestdata/{}/raw_data_{}/'.format(folder, input_int)
    SAVE_PATH_RES = '/home/kolb/Divest_Experiments/divestdata/{}/raw_data_{}/'.format(folder, input_int)
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = '/home/jakob/Project_Divestment/output_data/test/{}/raw_data_{}/'.format(folder, input_int)
    SAVE_PATH_RES = 'home/jakob/Project_Divestment/output_data/test/{}/raw_data_{}/'.format(folder, input_int)

if mode == 0:
    t_as = [round(x, 5) for x in list(10**np.linspace(-2.0, 2.0, 11))]
    phis = [round(x, 5) for x in list(np.linspace(0.0, 1.0, 11))[1:-1]]

    t_Gs = [round(x, 5) for x in list(10**np.linspace(-1.0, 1.0, 9))]
    alphas = [round(x, 5) for x in list(10**np.linspace(-4.0, 0, 5))]
else:
    t_as = [1.]
    phis = [.5]

    t_Gs = [1.]
    alphas = [10**-1]


parameters = {'t_a': 0, 'phi': 1, 'eps': 2, 't_G': 3, 'b_R0': 4, 'test': 5}
t_a, phi, eps, t_G, alpha, test = [1.], [.8], [0.0], [3.], [10.**(-2.)], [mode]
if noise:
    eps = [0.05]


NAME = 't_a_vs_phi_r_R0={}_timescales'.format(input_int)
INDEX = {0: 't_a', 1: 'phi', 2: 'eps', 3: 't_G', 4: 'b_R0', 5: 'test'}

if input_int < len(alphas) and input_int is not None:
    PARAM_COMBS = list(it.product(t_as, phis, eps, t_Gs,
                                  [alphas[input_int]], test))

elif input_int is None:
    PARAM_COMBS = list(it.product(t_as, phis, eps, t_Gs,
                                  alpha, test))

else:
    print(input_int, ' is not in the list of possible experiments')
    sys.exit()

# names and function dictionaries for post processing:

NAME1 = NAME+'_trajectory'
EVA1 = {"<mean_trajectory>":
        lambda fnames: pd.concat([load(f)["economic_trajectory"]
                                  for f in fnames]).groupby(level=0).mean(),
        "<sem_trajectory>":
        lambda fnames: pd.concat([load(f)["economic_trajectory"]
                                  for f in fnames]).groupby(level=0).sem()}

NAME2 = NAME+'_convergence'
EVA2 = {"<mean_convergence_state>":
        lambda fnames: np.nanmean([load(f)["convergence_state"]
                                   for f in fnames]),
        "<mean_convergence_time>":
        lambda fnames: np.nanmean([load(f)["convergence_time"]
                                   for f in fnames]),
        "<min_convergence_time>":
        lambda fnames: np.nanmin([load(f)["convergence_time"]
                                  for f in fnames]),
        "<max_convergence_time>":
        lambda fnames: np.max([load(f)["convergence_time"]
                               for f in fnames]),
        "<nanmax_convergence_time>":
        lambda fnames: np.nanmax([load(f)["convergence_time"]
                                  for f in fnames]),
        "<sem_convergence_time>":
        lambda fnames: st.sem([load(f)["convergence_time"]
                               for f in fnames]),
        "<runtime>":
        lambda fnames: st.sem([load(f)["runtime"]
                              for f in fnames])}

# full run
if mode == 0:
    SAMPLE_SIZE = 100
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS,
                                 INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2)

# test run
if mode == 1:
    SAMPLE_SIZE = 2
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2)

# debug and mess around mode:
if mode is None:
    SAMPLE_SIZE = 2
    handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS,
                                 INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2)
