"""
This experiment is meant to create trajectories of macroscopic variables from
1) the numeric micro model and
2) the analytic macro model
that can be compared to evaluate the validity and quality of the analytic
approximation.
The variable Parameters are b_d and phi.
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

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from pymofa.experiment_handling \
    import experiment_handling, even_time_series_spacing

from pydivest.macro_model.integrate_equations_aggregate \
    import IntegrateEquationsAggregate
from pydivest.macro_model.integrate_equations_rep \
    import Integrate_Equations as IntegrateEquationsRep
from pydivest.micro_model.divestmentcore import DivestmentCore
from parameters import ExperimentDefaults


def RUN_FUNC(b_d, phi, eps, approximate, test):
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
    phi : float \in [0,1]
        the rewiring probability for the network update
    eps : float
        the fraction of opinion formation events that is random
    approximate: bool
        if True: run macroscopic approximation
        if False: run micro-model
    test: int \in [0,1]
        wheter this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Parameters:

    ed = ExperimentDefaults()
    input_params = ed.input_params

    input_params['phi'] = phi
    input_params['b_d'] = b_d
    input_params['eps'] = eps
    input_params['test'] = test

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
    if test:
        print('initializing')
    if approximate == 1:
        m = DivestmentCore(*init_conditions, **input_params)
    elif approximate == 2:
        m = IntegrateEquationsAggregate(*init_conditions, **input_params)
    elif approximate == 3:
        m = IntegrateEquationsRep(*init_conditions, **input_params)
    else:
        raise ValueError('approximate must be in [1, 2, 3] but is {}'.format(approximate))

    if test:
        print('running')
    t_max = 900 if not test else 10
    m.set_parameters()
    exit_status = m.run(t_max=t_max)

    # store data in case of successful run
    if test:
        print('saving')
    if exit_status in [0, 1]:
        if approximate == 1:
            df1 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, 0., t_max)
            df2 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0., t_max)
        elif approximate == 2:
            df1 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, 0., t_max)
            df2 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0., t_max)
        elif approximate == 3:
            df1 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0., t_max)
            df2 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, 0., t_max)
        else:
            raise ValueError('approximate must be in [1, 2, 3, 4] but is {}'.format(approximate))

        for c in df1.columns:
            if c in df2.columns:
                df2.drop(c, axis=1, inplace=True)

        df_out = pd.concat([df1, df2], axis=1)

        df_out.index.name = 'tstep'
    else:
        df_out = None

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
    folder = 'P2'

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

    phis = [round(x, 5) for x in list(np.linspace(0.0, 0.9, 10))]
    b_ds = [round(x, 5) for x in list(np.linspace(1., 1.5, 3))]
    epss = [0.1, 0.05, 0.01]
    b_d, phi, eps = [1.25], [.4], [0.05]

    if test:
        PARAM_COMBS = list(it.product(b_d, phi, eps, [approximate], [True]))
    else:
        PARAM_COMBS = list(it.product(b_ds, phis, epss, [approximate], [test]))

    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle

    try:
        run_func_output = pd.read_pickle(SAVE_PATH_RAW + 'rfof.pkl')
    except:
        if not Path(SAVE_PATH_RAW).exists():
            Path(SAVE_PATH_RAW).mkdir(parents=True, exist_ok=True)
        params = list(PARAM_COMBS[0])
        params[-1] = True
        print('running run func with parameters', params)
        run_func_output = RUN_FUNC(*params)[1]
        with open(SAVE_PATH_RAW+'rfof.pkl', 'wb') as dmp:
            pd.to_pickle(run_func_output, dmp)

    SAMPLE_SIZE = 100 if not (test or approximate in [2, 3]) else 12

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=SAMPLE_SIZE,
                                         parameter_combinations=PARAM_COMBS,
                                         path_raw=SAVE_PATH_RAW
                                         )

    # define eva functions

    def mean(b_d, phi, eps, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'b_d={} & phi={} & eps={} & approximate={} & test={}'.format(b_d, phi, eps, approximate, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(b_d, phi, eps, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'b_d={} & phi={} & eps={} & approximate={} & test={}'.format(b_d, phi, eps, approximate, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').std()

    eva_1_handle = experiment_handling(run_func=mean,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=PARAM_COMBS,
                                       path_raw=SAVE_PATH_RES + '/mean.h5'
                                       )
    eva_2_handle = experiment_handling(run_func=std,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=PARAM_COMBS,
                                       path_raw=SAVE_PATH_RES + '/std.h5'
                                       )

    if mode == 0:
        compute_handle.compute()
        return 1
    elif mode == 1:
        eva_1_handle.compute()
        eva_2_handle.compute()
        return 1
    else:
        # in case nothing happened:
        return 0


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)

