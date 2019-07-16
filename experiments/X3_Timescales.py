"""
This experiment is designed to test different timescales in the
model against each other. Namely the timescales for
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

import getpass
import glob
import itertools as it
import os
import pickle as cp
import sys
import time
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st

from pydivest.default_params import ExperimentDefaults, ExperimentRoutines
from pydivest.divestvisuals.data_visualization import (plot_obs_grid,
                                                       plot_tau_phi,
                                                       tau_phi_final)

from pydivest.micro_model import divestmentcore as model

from pymofa.experiment_handling import (even_time_series_spacing,
                                        experiment_handling)


def load(*args, **kwargs):
    return np.load(*args, allow_pickle=True, **kwargs)


def RUN_FUNC(t_a, phi, eps, t_G, alpha, test):
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

    # setting model parameters

    defaults = ExperimentDefaults(phi=phi, eps=eps, test=test)

    b_r0 = alpha**2 * defaults.input_params['e']
    defaults.input_params['b_r0'] = b_r0

    defaults.calculate_timing(t_g=t_G, t_a=t_a)

    input_params = defaults.input_params

    # building initial conditions for social network

    N = 100  # use 100 households
    p = .125  # link probability for erdos renyi

    while True:
        net = nx.erdos_renyi_graph(N, p)

        if len(list(net)) > 1:
            break

    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=100)

    init_conditions = (adjacency_matrix, investment_decisions)

    # initializing the model

    m = model.DivestmentCore(*init_conditions, **input_params)

    # run the model
    print(test)

    if test:
        print(input_params)

    t_max = 300 if test == 0 else 5
    start = time.clock()
    exit_status = m.run(t_max)

    end = time.clock()

    if test:
        print('test output of variables')
        print(m.tau, m.phi, exit_status, m.convergence_state,
              m.convergence_time)

    # store data

    res = {
        "convergence": [exit_status],
        "runtime": [end - start],
        "convergence_state": [m.convergence_state],
        "convergence_time": [m.convergence_time]
    }

    df0 = pd.DataFrame.from_dict(res)
    df0.index.name = 'run'

    df1 = pd.DataFrame({
        "Investment decisions": m.investment_decisions,
        "Opinions": m.opinions,
        "Investment clean": m.investment_clean,
        "Investment dirty": m.investment_dirty
    })
    df1.index.name = 'agent'
    # interpolate trajectory to get evenly spaced time series.
    df2 = even_time_series_spacing(m.get_economic_trajectory(), 301, 0., t_max)
    df2.index.name = 'tstep'

    return exit_status, [df0, df1, df2]


def run_experiment(argv):
    # get sub experiment and mode from command line

    # set test

    if len(argv) > 1:
        test = bool(int(argv[1]))
    else:
        test = True
    # set mode (production, pp only, messy)

    if len(argv) > 3:
        mode = int(sys.argv[3])  # sets mode (production, test, messy)
    else:
        mode = 0

    # setup parameter values

    t_Gs = [100.]
    if test:
        t_as, phis, alphas, eps = [1.], [.8], [10.**(-2.)], [0.03]
    else:
        t_as = [round(x, 5) for x in list(10**np.linspace(-2.0, 2.0, 11))]
        phis = [round(x, 5) for x in list(np.linspace(0.0, 1.0, 11))]
        alphas = [0.01, 0.1, 0.5, 0.9]
        eps = [0.0, 0.01, 0.02, 0.03]

    param_combs = list(it.product(t_as, phis, eps, t_Gs, alphas, [test]))

    """
    set input/output paths
    """
    helper = ExperimentRoutines(run_func=RUN_FUNC,
                                param_combs=param_combs,
                                test=test,
                                subfolder='X3')
    save_path_raw, save_path_res = helper.get_paths()

    print(save_path_raw)

    # Create dummy runfunc output to pass its shape to experiment handle

    run_func_output = helper.run_func_output

    # set up computation handles

    sample_size = 100 if not test else 5

    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=sample_size,
                                         parameter_combinations=param_combs,
                                         path_raw=save_path_raw)

    pp_handles = []

    for operator in ['mean', 'std', 'collect']:
        rf = helper.get_pp_function(table_id=[0, 1, 2], operator=operator)
        handle = experiment_handling(run_func=rf,
                                     runfunc_output=run_func_output,
                                     sample_size=1,
                                     parameter_combinations=param_combs,
                                     path_raw=(save_path_res +
                                               f'/{operator}_trj.h5'),
                                     index=compute_handle.index)
        pp_handles.append(handle)

    # full run

    if mode == 0:
        compute_handle.compute()

        for handle in pp_handles:
            handle.compute()

    # post production only

    if mode == 1:
        for handle in pp_handles:
            handle.compute()


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
