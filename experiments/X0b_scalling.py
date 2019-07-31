"""
This experiment is the test case for the set of fitted parameters.
I want to see, which behavior the model shows with these parameters and
different settings for the remainting social parameters phi and epsilon.

Therefore, I vary phi and esilon and explore the resulting data.
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


def RUN_FUNC(N, ffh, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    N: int
        Number of households
    ffh: bool
        if True: run with fast and frugal heuristics
        if False: run with imitation only
    test: int in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    """
    print(f'starting run with N={N}')
    # Make different types of decision makers. Cues are

    if ffh:
        possible_cue_orders = [
            [2, 3],  # short term investor
            [3, 2],  # long term investor
            [4, 2],  # short term herder
            [4, 3],  # trending herder
            [4, 1],  # green conformer
            [4, 0],  # dirty conformer
            [1],  # gutmensch
            [0]
        ]  # redneck
    else:
        possible_cue_orders = [[0], [1]]

    # Parameters:
    defaults = ExperimentDefaults(params='fitted',
                                  possible_cue_orders=possible_cue_orders)

    input_params = defaults.input_params

    # building initial conditions

    # network:
    n = N
    k = 10

    p = float(k) / n

    net = nx.erdos_renyi_graph(n, p)

    adjacency_matrix = nx.adj_matrix(net).toarray()

    # opinions and investment

    x = n * np.random.dirichlet(np.ones(len(possible_cue_orders)))

    opinions = []

    for i, xi in enumerate(x):
        opinions += int(np.round(xi)) * [i]
    np.random.shuffle(opinions)

    if len(opinions) > n:
        opinions = opinions[:n]
    elif len(opinions) < n:
        for i in range(n - len(opinions)):
            opinions += [opinions[np.random.randint(0, len(opinions))]]

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

    init_conditions = (adjacency_matrix, np.array(opinions),
                       np.array(clean_investment), np.array(dirty_investment))

    t_1 = 400 if not test else 200

    # initializing the model
    m = model.DivestmentCore(*init_conditions, **input_params)

    # start timer
    t_start = time.clock()

    # run model with abundant resource
    exit_status = m.run(t_max=t_1)

    res = {}
    res["runtime"] = [time.clock() - t_start]
    print(res['runtime'], N)

    # store data in case of successful run

    if test:
        exit_status = 1
    df1 = even_time_series_spacing(m.get_economic_trajectory(), 401, 5., t_1)
    df1.index.name = 'tstep'
    res["convergence_state"] = [m.convergence_state]
    res["convergence_time"] = [m.convergence_time]

    df2 = pd.DataFrame.from_dict(res)
    df2.index.name = 'i'

    # save data

    return exit_status, [df1, df2]


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

    # switch decision making

    if len(argv) > 2:
        ffh = bool(int(argv[2]))
    else:
        ffh = True
    """
    create parameter combinations and index
    """

    Ns = list(np.arange(0, 1000, 50))
    N = [10, 20, 30, 40, 50]

    if test:
        param_combs = list(it.product(N, [ffh], [test]))
    else:
        param_combs = list(it.product(Ns, [ffh], [test]))
    """
    set input/output paths
    """

    sd = ['imitation', 'ffh'][int(ffh)]

    helper = ExperimentRoutines(run_func=RUN_FUNC,
                                param_combs=param_combs,
                                test=test,
                                subfolder=f'X0b_{sd}')

    save_path_raw, save_path_res = helper.get_paths()
    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle
    run_func_output = helper.run_func_output

    # define computation handle

    sample_size = 10 if not test else 3

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