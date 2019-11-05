"""
Testing Model Hysteresis with micro simulations.

Bifurcation parameter xi for different total factor productivity in the dirty
sector.
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
from pydivest.micro_model.divestmentcore import DivestmentCore as MicroModel
from pydivest.macro_model.integrate_equations_aggregate import IntegrateEquationsAggregate as MacroModel
from pymofa.experiment_handling import (even_time_series_spacing,
                                        experiment_handling)


def load(*args, **kwargs):
    return np.load(*args, allow_pickle=True, **kwargs)


def RUN_FUNC(b_d, phi, approx, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    b_d: float in [1, 4]
        total factor productivity in the dirty sector
    phi: float in [0, 1]
        rewiring probability in adaptive voter update
    test: int in [0, 1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    """

    # Parameters:
    defaults = ExperimentDefaults(params='default',
                                  phi=phi,
                                  b_d=b_d,
                                  L=100.,
                                  test=False,
                                  R_depletion=False)

    input_params = defaults.input_params

    # building initial conditions

    # network:
    n = 100
    k = 10

    p = float(k) / n

    while True:
        net = nx.erdos_renyi_graph(n, p)

        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # initial opinions:
    # fitted distribution has length of N=100.
    x = int(n/2)
    opinions = [0]*(n-x) + [1]*x

    clean_investment = [1]*n
    dirty_investment = [1]*n

    init_conditions = (adjacency_matrix, np.array(opinions),
                       np.array(clean_investment), np.array(dirty_investment))

    # initializing the model
    if approx == 0:
        m = MicroModel(*init_conditions, **input_params)
    elif approx == 1:
        m = MacroModel(*init_conditions, **input_params)

    # run model with abundant resource

    t_max = 0
    t_n = 3 if test else 100
    xis = []
    data_points = 3 if test else 51
    xi_min = .1
    xi_max = .2
    t_0 = t_max
    for xi in np.linspace(xi_min, xi_max, data_points):
        if approx == 1:
            m.p_xi = xi
        else:
            m.xi = xi
        m.set_parameters()
        t_max += t_n
        xis += [xi]*t_n
        if test:
            print(t_max)
        m.run(t_max=t_max)
    # store data in case of successful run
    df1 = even_time_series_spacing(m.get_aggregate_trajectory(), len(xis), t_0,
                                  t_max)
    df1['xi'] = xis
    m.ag_trajectory = []
    m.init_aggregate_trajectory()

    t_0 = t_max+1
    for xi in np.linspace(xi_max, xi_min, data_points):
        if approx == 1:
            m.p_xi = xi
        else:
            m.xi = xi
        m.set_parameters()
        t_max += t_n
        xis += [xi]*t_n
        if test:
            print(t_max)
        m.run(t_max=t_max)
    # store data in case of successful
    df2 = even_time_series_spacing(m.get_aggregate_trajectory(), len(xis), t_0,
                                  t_max)
    df2['xi'] = xis
    df1 = df1[['xi', 'x', 'z', 'C']]
    df2 = df2[['xi', 'x', 'z', 'C']]

    # save data

    for df in [df1, df2]:
        df.index.name='tstep'
        df['sample_id'] = None
    print(df1.head())
    print(df2.head())
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
        approx = int(argv[2])
    else:
        approx = 0
    """
    create parameter combinations and index
    """

    phis, b_ds = np.linspace(0, 1, 21), np.linspace(3, 4, 3)
    phi, b_d, approx = [.5], [4], [0, 1]

    if test:
        param_combs = list(it.product(b_d, phi, approx, [test]))
    else:
        param_combs = list(it.product(b_d, phi, approx, [test]))
    """
    set input/output paths
    """

    if test:
        print('initializing helper', flush=True)

    helper = ExperimentRoutines(run_func=RUN_FUNC,
                                param_combs=param_combs,
                                test=test,
                                subfolder=f'X7')

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

    sample_size = 63 if not test else 1

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

    for handle in pp_handles:
         handle.compute()

    return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
