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

import getpass
import itertools as it
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pydivest.default_params import ExperimentDefaults
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

    # Parameters:

    ed = ExperimentDefaults()
    input_params = ed.input_params

    input_params['phi'] = phi
    input_params['tau'] = tau
    input_params['xi'] = xi
    input_params['kappa_c'] = kappa_c
    input_params['test'] = False  # test

    # investment_decisions:
    nopinions = [100, 100]

    # network:
    N = sum(nopinions)
    k = 10

    # building initial conditions
    p = float(k) / N

    while True:
        net = nx.erdos_renyi_graph(N, p)

        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=N)

    clean_investment = np.ones(N) * 50. / float(N)
    dirty_investment = np.ones(N) * 50. / float(N)

    init_conditions = (adjacency_matrix, investment_decisions,
                       clean_investment, dirty_investment)

    # initializing the model

    if approximate == 1:
        m = DivestmentCore(*init_conditions, **input_params)
    elif approximate == 2:
        m = IntegrateEquationsAggregate(*init_conditions, **input_params)
    elif approximate == 3:
        m = IntegrateEquationsRep(*init_conditions, **input_params)
    else:
        raise ValueError(
            'approximate must be in [1, 2, 3] but is {}'.format(
                approximate))

    t_max = 300 if not test else 10
    m.set_parameters()
    m.n_trajectory_output=True
    exit_status = m.run(t_max=t_max)

    # store data in case of successful run

    if exit_status in [0, 1]:
        if approximate in [0, 1, 4]:
            df1 = even_time_series_spacing(m.get_aggregate_trajectory(), 201,
                                           0, t_max)
            df2 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0,
                                           t_max)

            df3 = even_time_series_spacing(m.get_network_trajectory(),
                                           201, 0, t_max)
        else:
            df2 = even_time_series_spacing(m.get_aggregate_trajectory(), 201,
                                           0, t_max)
            df1 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0,
                                           t_max)

            df3 = even_time_series_spacing(pd.DataFrame(data=[[0,0]],
                               columns=['local clustering coefficient',
                                        'mean shortest path']
                              ), 201, 0, t_max)

        for c in df1.columns:
            if c in df2.columns:
                df2.drop(c, axis=1, inplace=True)

        df_tmp = pd.concat([df1, df2], axis=1)

        for c in df_tmp.columns:
            if c in df3.columns:
                df3.drop(c, axis=1, inplace=True)
        df_out = pd.concat([df_tmp, df3], axis=1)

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

    if len(argv) > 2:
        mode = int(argv[2])
    else:
        mode = 0
    # switch micro macro model

    if len(argv) > 3:
        approximate = int(argv[3])
    else:
        approximate = 1
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

    sub_experiment = ['micro', 'aggregate', 'representative'][approximate - 1]
    folder = 'P3o2n'

    # make sure, testing output goes to its own folder:

    test_folder = ['', 'test_output/'][int(test)]

    SAVE_PATH_RAW = \
        "{}/{}{}/{}/" \
        .format(tmppath, test_folder, folder, sub_experiment)
    SAVE_PATH_RES = \
        "{}/{}{}/{}/" \
        .format(respath, test_folder, folder, sub_experiment)
    """
    create parameter combinations and index
    """

    tau, phi, xi = [1.], [0., .5, .85, .9, ,.95], [0.1]

    param_combs = list(
            it.product(tau, phi, xi, [0.5], [approximate], [test]))
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

    sample_size = 100 if not (test or approximate in [2, 3]) else 10

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=sample_size,
                                         parameter_combinations=param_combs,
                                         path_raw=SAVE_PATH_RAW)

    # define eva functions

    def mean(tau, phi, xi, kappa_c, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = (
            f'tau={tau} & phi={phi} & xi={xi} & kappa_c={kappa_c}'
            f'& approximate={approximate} & test={test}'
        )

        with SafeHDFStore(compute_handle.path_raw) as store:
            try:
                trj = store.select("dat_0", where=query)
            except KeyError:
                trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(tau, phi, xi, kappa_c, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = (
            f'tau={tau} & phi={phi} & xi={xi} & kappa_c={kappa_c}'
            f'& approximate={approximate} & test={test}'
        )

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
                                       parameter_combinations=param_combs,
                                       path_raw=SAVE_PATH_RES + '/mean.h5')
    eva_2_handle = experiment_handling(run_func=std,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=param_combs,
                                       path_raw=SAVE_PATH_RES + '/std.h5')

    if mode == 0:
        compute_handle.compute()

        return 1
    elif mode == 1:

        compute_handle.compute()

        eva_1_handle.compute()
        eva_2_handle.compute()

        return 1

    return 0


if __name__ == "__main__":
    run_experiment(sys.argv)
