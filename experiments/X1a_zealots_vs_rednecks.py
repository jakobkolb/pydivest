"""
test the sensitivity of the simulation on deviations in the initial conditions
in terms of cue order distributions.

Additionally, add campaigners as household types. Campaigners always invest in
the clean sector and can not imitate any other behavior.

Vary the initial fraction of rednecks and campaigners between 0% and 25% and
renormalize the rest of the initial distribution accordingly.
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


def RUN_FUNC(n_rn, n_cp, phi, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    n_rn: float in [0, 25]
        frequency of rednecks in percent
    n_cp: float in [0, 25]
        frequency of campaigners in percent
    phi: float in [0, 1]
        rewiring probability in adaptive voter update
    test: int in [0, 1]
        whether this is a test run, e.g.
        can be executed with lower runtime
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
        [0],  # redneck
        [5]   # campaigner
    ]

    # Parameters:
    defaults = ExperimentDefaults(params='fitted',
                                  phi=phi,
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
    # fitted distribution has length of N=100.
    fitted_opinions_distribution = [14, 7, 15, 13, 12, 16, 7, 16]

    # combined fraction of rednecks and campaigners
    n_add = n_rn + n_cp

    # combined fraction of the rest of the population
    n_rest = 100 - n_add

    # current fraction of the rest of the population
    n_rest_curr = sum(fitted_opinions_distribution[:7])

    # rescale rest distribution

    for i in range(7):
        fitted_opinions_distribution[i] *= n_rest/n_rest_curr

    # set number of rednecks according to input variable
    fitted_opinions_distribution[7] = n_rn

    # append number of campaigners according to input variable
    fitted_opinions_distribution.append(n_cp)
    # and set campaign option to True
    input_params['campaign'] = True

    # for N=200 households, I just use double the frequencies of the initial
    # fitted distribution.
    x = [n/sum(fitted_opinions_distribution) * x
         for x in fitted_opinions_distribution]
    opinions = []

    for i, xi in enumerate(x):
        opinions += int(np.round(xi)) * [i]
    np.random.shuffle(opinions)
    opinions = opinions[:n]

    # in case, opinions are too short due to rounding errors, add values
    if len(opinions) < n:
        for i in range(n - len(opinions)):
            opinions.append(opinions[i])
    # in case, opinions are too long, remove some.
    elif len(opinions) > n:
        opinions = opinions[:n]

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
    print(f'run took {res["runtime"]}', flush=True)
    # store data in case of successful run

    df1 = even_time_series_spacing(m.get_economic_trajectory(), 101, 0, t_1)
    df3 = even_time_series_spacing(m.get_economic_trajectory(), 401, 0, 20)

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
    return 1, [dfi, df1, df2]


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
        if int(argv[1]) < 2:
            test = bool(int(argv[1]))
            ic = False
        else:
            test = False
            ic = True
    else:
        test = True
        ic = None
    if len(argv) > 2:
        exp_id = argv[2]
    else:
        exp_id = 1
    """
    create parameter combinations and index
    """

    phis, n_rds, n_cps = [.5, .7, .8], range(16, 16, 1), range(0, 26, 1)
    phi, n_rd, n_cp = [.5], [0, 15, 25], [0, 15, 25]

    if test:
        param_combs = list(it.product(n_rd, n_cp, phi, [test]))
    else:
        param_combs = list(it.product(n_rds, n_cps, phis, [test]))
    """
    set input/output paths
    """

    if test:
        print('initializing helper', flush=True)

    helper = ExperimentRoutines(run_func=RUN_FUNC,
                                param_combs=param_combs,
                                test=test,
                                subfolder=f'X1a_{exp_id}')

    save_path_raw, save_path_res = helper.get_paths()
    """
    run computation and/or post processing and/or plotting
    """
    # Create dummy runfunc output to pass its shape to experiment handle
    run_func_output = helper.run_func_output

    if ic is True:
        print('set up initial conditions', flush=True)
        return 1
    # define computation handle

    sample_size = 200 if not test else 5

    if test:
        print('initializing compute handles', flush=True)
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

    if test:
        print('starting computation', flush=True)
    compute_handle.compute()

    # for handle in pp_handles:
    #     handle.compute()

    return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
