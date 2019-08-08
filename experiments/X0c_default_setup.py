"""
This ist the default setup of the model with economic parameters estimated from
historical data and initial distribution of opinions fitted to keep the ratio
of clean to dirty capital constant at the start of the simulation (results from
experiment X0a)

As experiment X0b suggested, above N=200 households, additional households
don't give a big reduction in the variance of results, I use N=200 households
for this simulation.

The results can be used to verify, that the estimaten of the initial
distribution of opinions actually helped to get the desired model behavior e.g.
constant capital shares in the beginning of the simulation.
"""

# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3

import copy
import getpass
import itertools as it
import os
import pickle as cp
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from pydivest.default_params import ExperimentDefaults, ExperimentRoutines
from pydivest.divestvisuals.data_visualization import (plot_amsterdam,
                                                       plot_trajectories)
from pydivest.micro_model import divestmentcore as model
from pymofa.experiment_handling import (even_time_series_spacing,
                                        experiment_handling)


def load(*args, **kwargs):
    return np.load(*args, allow_pickle=True, **kwargs)


def RUN_FUNC(eps, phi, test):
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
    xi : float in [0,0.5]
        exponent for knowledge stock in the clean production function
    ffh: bool
        if True: run with fast and frugal heuristics
        if False: run with imitation only
    test: int in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Make different types of decision makers. Cues are

    possible_cue_orders = [
        [2, 3],  # short term investor
        [3, 2],  # long term investor
        [4, 2],  # short term herder
        [4, 3],  # trending herder
        [4, 1],  # green conformer
        [4, 0],  # dirty conformer
        [1],  # gutmensch
        [0]  # redneck
    ]

    # Parameters:
    defaults = ExperimentDefaults(params='fitted',
                                  phi=phi,
                                  eps=eps,
                                  possible_cue_orders=possible_cue_orders)

    input_params = defaults.input_params

    # building initial conditions

    # network:
    n = 200
    k = 10

    p = float(k) / n

    while True:
        net = nx.erdos_renyi_graph(n, p)

        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # initial opinions:
    fitted_opinions_distribution = [14, 7, 15, 13, 12, 16, 7, 16]
    x = [2 * x for x in fitted_opinions_distribution]
    opinions = []

    for i, xi in enumerate(x):
        opinions += int(np.round(xi)) * [i]
    np.random.shuffle(opinions)

    # initial investment.
    # give equally sized amounts of capital to households.
    # asign only clean or dirty capital to a household.
    # distribute independent of initial opinion (as I did, when I fitted the
    # initial distribution of opinions)
    n_clean = int(n * input_params['K_c0'] /
                  (input_params['K_c0'] + input_params['K_d0']))
    n_dirty = n - n_clean

    clean_investment = []
    dirty_investment = []

    for i in range(n):
        if i < n_clean:
            clean_investment += [input_params['K_c0'] * 1. / float(n_clean)]
            dirty_investment += [0]
        else:
            clean_investment += [0]
            dirty_investment += [input_params['K_d0'] * 1. / float(n_dirty)]

    cnt = Counter(opinions)
    dfi = pd.DataFrame(
        data=np.array([[cnt[i] for i in range(len(possible_cue_orders))]]),
        columns=[f'O{i+1}' for i in range(len(possible_cue_orders))],
        index=['opinions'])

    init_conditions = (adjacency_matrix, np.array(opinions),
                       np.array(clean_investment), np.array(dirty_investment))

    t_1 = 100 if not test else 5

    # initializing the model
    m = model.DivestmentCore(*init_conditions, **input_params)

    # start timer
    t_start = time.clock()

    # run model with abundant resource
    m.run(t_max=t_1)

    res = {}
    res["runtime"] = [time.clock() - t_start]

    # store data in case of successful run

    df1 = even_time_series_spacing(m.get_economic_trajectory(), 401, 0, 20)
    df3 = even_time_series_spacing(m.get_economic_trajectory(), 101, 0, t_1)

    df1 = df1[['time', 'wage', 'r_c', 'r_d', 'r_c_dot', 'r_d_dot', 'K_c',
                'K_d', 'P_c', 'P_d', 'G', 'R', 'C', 'Y_c', 'Y_d',
                'c_R',
                'consensus', 'decision state', 'G_alpha', 'i_c']]
    df3 = df3[['time', 'wage', 'r_c', 'r_d', 'r_c_dot', 'r_d_dot', 'K_c',
                'K_d', 'P_c', 'P_d', 'G', 'R', 'C', 'Y_c', 'Y_d',
                'c_R',
                'consensus', 'decision state', 'G_alpha', 'i_c']]

    df1.index.name = 'tstep'
    res["convergence_state"] = [m.convergence_state]
    res["convergence_time"] = [m.convergence_time]

    df2 = pd.DataFrame.from_dict(res)
    df2.index.name = 'i'

    # save data

    for df in [dfi, df1, df2, df3]:
        df['sample_id'] = None

    return 1, [df1, df2]


def run_experiment(argv):
    """
    Take arv input variables and run sub_experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test, mode, ffh/av, equi/trans],
    2)  set output folders according to switches,
    3)  generate parameter combinations,
    4)  define names and dictionaries of callables to apply to sub_experiment
        data for post processing,
    5)  run computation and/or post processing and/or plotting
        depending on execution on cluster or locally or depending on
        experimentation mode.

    Parameters
    ----------
    argv: list[N]
        List of parameters from terminal input

    Returns
    -------
    rt: int
        some return value to show whether sub_experiment succeeded
        return 1 if sucessfull.
    """
    """
    Get switches from input line in order of
    [test, mode, ffh on/of, equi/transition]
    """

    # switch testing mode

    if len(argv) > 1:
        test = bool(int(argv[1]))
    else:
        test = True
    """
    create parameter combinations and index
    """

    # mute these until further notice
    #epss = [round(x, 5) for x in list(np.linspace(0.0, 0.05, 6))]
    #phis = [round(x, 5) for x in list(np.linspace(0., 1., 11))]

    epss, phis = [0.02], [.5, .7, .8, .9]
    eps, phi = [0.02], [.5]

    if test:
        param_combs = list(it.product(eps, phi, [test]))
    else:
        param_combs = list(it.product(epss, phis, [test]))
    """
    set input/output paths
    """

    helper = ExperimentRoutines(run_func=RUN_FUNC,
                                param_combs=param_combs,
                                test=test,
                                subfolder=f'X0c')

    save_path_raw, save_path_res = helper.get_paths()
    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle
    run_func_output = helper.run_func_output

    # define computation handle

    sample_size = 500 if not test else 110

    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=sample_size,
                                         parameter_combinations=param_combs,
                                         path_raw=save_path_raw)

    # define post processing functions

    pp_handles = []

    for operator in ['mean', 'std', 'collect']:
        rf = helper.get_pp_function(table_id=[0, 1], operator=operator)
        handle = experiment_handling(run_func=rf,
                                     runfunc_output=run_func_output,
                                     sample_size=1,
                                     parameter_combinations=param_combs,
                                     path_raw=(save_path_res +
                                               f'/{operator}_trj.h5'),
                                     index=compute_handle.index)
        pp_handles.append(handle)

    # cluster mode: computation and post processing

    sys.stdout.flush()

    compute_handle.compute()

    for handle in pp_handles:
        handle.compute()

    return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
