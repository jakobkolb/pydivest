"""
This experiment is meant to create trajectories of macroscopic variables from
1) the numeric micro model and
2) the analytic macro model
From these trajectories, I will calculate the distance
The variable Parameters are tau and phi.
"""

# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3

import argparse
import getpass
import itertools as it
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pydivest.default_params import ExperimentDefaults, ExperimentRoutines
from pydivest.macro_model.integrate_equations_aggregate import \
    IntegrateEquationsAggregate
from pydivest.macro_model.integrate_equations_rep import \
    Integrate_Equations as IntegrateEquationsRep
from pydivest.micro_model.divestmentcore import DivestmentCore
from pymofa.experiment_handling import (even_time_series_spacing,
                                        experiment_handling)


def RUN_FUNC(tau, phi, xi, kappa_c, approximate, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    tau : float > 0
        the frequency of social interactions
    phi : float in [0,1]
        the rewiring probability for the network update
    xi : float
        elasticity of knowledge in the clean sector
    kappa_c: float
        elasticity of capital in the clean sector
    approximate: bool
        if True: run macroscopic approximation
        if False: run micro-model
    test: int in [0,1]
        wheter this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    possible_cue_orders = [
        [1],  # gutmensch
        [0]  # redneck
    ]

    # Parameters:
    defaults = ExperimentDefaults(params='fitted',
                                  phi=phi,
                                  possible_cue_orders=possible_cue_orders)

    input_params = defaults.input_params

    # building initial conditions

    # network:
    nodes = 200
    degree = 10

    p = float(degree) / nodes

    while True:
        net = nx.erdos_renyi_graph(nodes, p)

        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # initial opinions:
    fitted_opinions_distribution = [15, 85]
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
    n_clean = int(nodes * input_params['K_c0'] /
                  (input_params['K_c0'] + input_params['K_d0']))
    n_dirty = nodes - n_clean

    clean_investment = []
    dirty_investment = []

    for i in range(nodes):
        if i < n_clean:
            clean_investment += [input_params['K_c0'] * 1. / float(n_clean)]
            dirty_investment += [0]
        else:
            clean_investment += [0]
            dirty_investment += [input_params['K_d0'] * 1. / float(n_dirty)]

    init_conditions = (adjacency_matrix, np.array(opinions),
                       np.array(clean_investment), np.array(dirty_investment))

    t_1 = 100 if not test else 5

    # initializing the model

    if approximate == 1:
        mdl = DivestmentCore(*init_conditions, **input_params)
    elif approximate == 2:
        mdl = IntegrateEquationsAggregate(*init_conditions, **input_params)
    elif approximate == 3:
        mdl = IntegrateEquationsRep(*init_conditions, **input_params)
    else:
        raise ValueError(
            'approximate must be in [1, 2, 3, 4] but is {}'.format(
                approximate))

    mdl.set_parameters()
    exit_status = mdl.run(t_max=t_1)

    # store data in case of successful run

    if exit_status in [0, 1]:
        df2 = even_time_series_spacing(mdl.get_aggregate_trajectory(), 201, 0,
                                       t_1)
        df1 = even_time_series_spacing(mdl.get_unified_trajectory(), 201, 0,
                                       t_1)

        if test:
            df3 = df2

        for col in df1.columns:
            if col in df2.columns:
                df2.drop(col, axis=1, inplace=True)

        if not test:
            df_out = pd.concat([df1, df2], axis=1)
        else:
            df_tmp = pd.concat([df1, df2], axis=1)

            for c in df_tmp.columns:
                if c in df3.columns:
                    df3.drop(c, axis=1, inplace=True)
            df_out = pd.concat([df_tmp, df3], axis=1)

        df_out.index.name = 'tstep'
    else:
        df_out = None

    # remove output that is not needed for production plot to write less on database
    rm_columns = [
        'mu_c^c', 'mu_c^d', 'mu_d^c', 'mu_d^d', 'l_c', 'l_d', 'r', 'r_c',
        'r_d', 'w', 'W_c', 'W_d', 'n_c', 'i_c', 'wage', 'r_c_dot', 'r_d_dot',
        'K_c', 'K_d', 'P_c', 'P_d', 'L', 'R', 'P_c_cost', 'P_d_cost',
        'K_c_cost', 'K_d_cost', 'c_R', 'consensus', 'decision state',
        'G_alpha', '[0]', '[1]', 'c[0]', 'c[1]', 'd[0]', 'd[1]'
    ]

    if not test:
        for column in df_out.columns:
            if column in rm_columns:
                df_out.drop(column, axis=1, inplace=True)

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

    parser = ExperimentRoutines.get_argparser()
    args = parser.parse_args()
    """
    create parameter combinations and index
    """

    tau, phi = [1.], [.5]

    if args.test:
        PARAM_COMBS = list(
            it.product(tau, phi, [0.1], [0.5], [args.approximate],
                       [args.test]))
    else:
        PARAM_COMBS = list(
            it.product(tau, phi, [0.1], [0.5], [args.approximate],
                       [args.test]))
    """
    set input/output paths
    """

    sub_experiment = ['micro', 'aggregate',
                      'representative'][args.approximate - 1]
    helper = ExperimentRoutines(run_func=RUN_FUNC,
                                param_combs=PARAM_COMBS,
                                test=args.test,
                                subfolder=f'P3o2b_{sub_experiment}')
    SAVE_PATH_RAW, SAVE_PATH_RES = helper.get_paths()
    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle

    run_func_output = helper.run_func_output

    SAMPLE_SIZE = 50 if not (args.test or args.approximate in [2, 3]) else 10

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=SAMPLE_SIZE,
                                         parameter_combinations=PARAM_COMBS,
                                         path_raw=SAVE_PATH_RAW)

    # define eva functions

    def mean(tau, phi, xi, kappa_c, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = f'tau={tau} & phi={phi} & xi={xi} & kappa_c={kappa_c} & approximate={approximate} & test={test}'

        with SafeHDFStore(compute_handle.path_raw) as store:
            try:
                trj = store.select("dat_0", where=query)
            except KeyError:
                trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(tau, phi, xi, kappa_c, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = f'tau={tau} & phi={phi} & xi={xi} & kappa_c={kappa_c} & approximate={approximate} & test={test}'

        with SafeHDFStore(compute_handle.path_raw) as store:
            try:
                trj = store.select("dat_0", where=query)
            except KeyError:
                trj = store.select("dat", where=query)

        df_out = trj.groupby(level='tstep').std()

        return 1, df_out

    eva_1_handle = experiment_handling(run_func=mean,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=PARAM_COMBS,
                                       path_raw=SAVE_PATH_RES + '/mean.h5')
    eva_2_handle = experiment_handling(run_func=std,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=PARAM_COMBS,
                                       path_raw=SAVE_PATH_RES + '/std.h5')

    if args.mode == 0:
        compute_handle.compute()

        return 1
    elif args.mode == 1:

        eva_1_handle.compute()
        eva_2_handle.compute()

        return 1
    else:
        # in case nothing happened:

        return 0


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
