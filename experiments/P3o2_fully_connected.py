"""
This experiment compares the model dynamics for the case of a fully connected
network with only imitation updates with a partly connected network with
imitation and adaptation updates.
"""

# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3

import getpass
import itertools as it
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pydivest.default_params import ExperimentDefaults
from pydivest.micro_model.divestmentcore import DivestmentCore
from pymofa.experiment_handling import (even_time_series_spacing,
                                        experiment_handling)


def RUN_FUNC(fully_connected, phi, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    fully_connected: bool
        change wheter the model is running with default variables or with a
        fully connected network and phi and tau changed such that the rate of
        immitation events stays the same
    test: int in [0,1]
        wheter this is a test run, e.g.
        can be executed with lower runtime
    """

    # Parameters:

    ed = ExperimentDefaults()
    input_params = ed.input_params

    input_params['test'] = False
    input_params['phi'] = phi

    if fully_connected:
        input_params['phi'] = 0.
        input_params['tau'] = input_params['tau'] * phi
        input_params['fully_connected'] = True

    # investment_decisions:
    nopinions = [10, 190] if not test else [3, 30]

    # network:
    N = sum(nopinions)


    if fully_connected:
        adjacency_matrix = np.ones((N, N))
        for i in range(N):
            adjacency_matrix[i,i] = 0
    elif not fully_connected:
        k = 10

        # building initial conditions
        p = float(k) / N

        while True:
            net = nx.erdos_renyi_graph(N, p)

            if len(list(net)) > 1:
                break
        adjacency_matrix = nx.adj_matrix(net).toarray()
    print(sum(sum(adjacency_matrix)))

    investment_decisions = np.zeros(N, dtype='int')
    investment_decisions[:nopinions[0]] = 1

    clean_investment = np.ones(N) * 50. / float(N)
    dirty_investment = np.ones(N) * 50. / float(N)

    init_conditions = (adjacency_matrix, investment_decisions,
                       clean_investment, dirty_investment)

    # initializing the model

    model = DivestmentCore(*init_conditions, **input_params)

    t_max = 500 if not test else 30
    t_eq = 300 if not test else 30

    model.R_depletion = False
    model.run(t_max=t_eq)

    model.R_depletion = True
    model.set_parameters()
    exit_status = model.run(t_max=t_max + t_eq)

    # store data in case of successful run

    if exit_status in [0, 1]:
        df1 = even_time_series_spacing(model.get_aggregate_trajectory(),
                                       t_max + t_eq + 1, 0, t_max + t_eq)
        df2 = even_time_series_spacing(model.get_unified_trajectory(),
                                       t_max + t_eq + 1, 0, t_max + t_eq)

        for column in df1.columns:
            if column in df2.columns:
                df2.drop(column, axis=1, inplace=True)

        df_out = pd.concat([df1, df2], axis=1)


        df_out.index.name = 'tstep'
    else:
        df_out = None

    print(df_out.columns)

    return exit_status, df_out


# get sub experiment and mode from command line

# experiment, mode, test


def run_experiment(argv):
    """
    Take arv input variables and run experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test, mode, ffh/av, equi/trans],
    2)  set output folders according to switches,
    3)  generate parameter combinations,
    4)  define names and dictionaries of callables to apply to experiment
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
        some return value to show whether experiment succeeded
        return 1 if sucessfull.
    """
    """
    Get switches from input line in order of
    [test, mode, micro/macro]
    """

    # switch testing mode

    if len(argv) > 1:
        test = bool(int(argv[1]))
    else:
        test = False
    # switch sub_experiment mode
    """
    set input/output paths
    """

    respath = os.path.dirname(os.path.realpath(__file__)) + "/../output_data"

    if getpass.getuser() == "jakob":
        tmppath = respath
    elif getpass.getuser() == "kolb":
        tmppath = "/p/tmp/kolb/Divest_Experiments"
    else:
        tmppath = "./"

    folder = 'P3o2fc'

    # make sure, testing output goes to its own folder:

    test_folder = ['', 'test_output/'][int(test)]

    SAVE_PATH_RAW = \
        "{}/{}{}/" \
        .format(tmppath, test_folder, folder)
    SAVE_PATH_RES = \
        "{}/{}{}/" \
        .format(respath, test_folder, folder)
    """
    create parameter combinations and index
    """

    param_combs = list(it.product([True, False], [.5, .8, .9], [test]))
    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle

    try:
        if not Path(SAVE_PATH_RAW).exists():
            Path(SAVE_PATH_RAW).mkdir(parents=True, exist_ok=True)
        run_func_output = pd.read_pickle(SAVE_PATH_RAW + 'rfof.pkl')
    except:
        params = list(param_combs[0])
        run_func_output = RUN_FUNC(*params)[1]
        with open(SAVE_PATH_RAW + 'rfof.pkl', 'wb') as dmp:
            pd.to_pickle(run_func_output, dmp)

    sample_size = 100 if not test else 10

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=sample_size,
                                         parameter_combinations=param_combs,
                                         path_raw=SAVE_PATH_RAW)

    compute_handle.compute()

    return 1


if __name__ == "__main__":
    run_experiment(sys.argv)
