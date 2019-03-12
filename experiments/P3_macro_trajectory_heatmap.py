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

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from pymofa.experiment_handling import experiment_handling, even_time_series_spacing

from pydivest.macro_model.integrate_equations_aggregate import IntegrateEquationsAggregate
from pydivest.macro_model.integrate_equations_rep import Integrate_Equations as IntegrateEquationsRep
from pydivest.micro_model.divestmentcore import DivestmentCore

from pydivest.default_params import ExperimentDefaults


def RUN_FUNC(tau, phi, eps, approximate, test):
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
    input_params['tau'] = tau
    input_params['eps'] = eps
    input_params['test'] = test

    # investment_decisions:
    nopinions = [50, 50]

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
        raise ValueError('approximate must be in [1, 2, 3] but is {}'.format(approximate))

    # THIS TURNED OUT TO BE TROUBLE! SINCE THE EQUILIBRIUM STATE APPARENTLY DEPENDS ON TAU
    # (even though i'm sure it is not supposed to.)

    # equilibration phase with small tau
    # to reach equilibrium distribution of
    # investment decisions
    t_eq = 1000 #if not test else 10
    t_max = t_eq
    if hasattr(m, 'trj_output_window'):
        m.trj_output_window = [t_eq, np.float('inf')]
    m.R_depletion = False
    m.set_parameters()
    m.run(t_max=t_max)

    # ONLY KEEPING THIS.
    # intermediate phase with original tau but still no resource depletion
    # to verify that the equilibrium distribution of investment decisions
    # is in fact independent from tau

    t_max += 300 #if not test else 2
    m.tau = tau
    m.set_parameters()
    m.run(t_max=t_max)

    # transition phase with resource depletion

    t_max += 600 #if not test else 6
    m.R_depletion = True
    m.set_parameters()
    exit_status = m.run(t_max=t_max)

    # store data in case of successful run
    if exit_status in [0, 1]:
        if approximate in [0, 1, 4]:
            df1 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, t_eq, t_max)
            df2 = even_time_series_spacing(m.get_unified_trajectory(), 201, t_eq, t_max)
        else:
            df2 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, t_eq, t_max)
            df1 = even_time_series_spacing(m.get_unified_trajectory(), 201, t_eq, t_max)


        for c in df2.columns:
            if c in df1.columns:
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
    folder = 'P3'

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

    phis = [round(x, 5) for x in list(np.linspace(0.0, 1., 21))]
    taus = [round(x, 5) for x in list(np.linspace(.5, 10., 20))]
    eps = [0.05, 0.01]
    tau, phi = [1.], [.8]

    if test:
        PARAM_COMBS = list(it.product(tau, phi, [0.05], [approximate], [test]))
    else:
        PARAM_COMBS = list(it.product(taus, phis, eps, [approximate], [test]))

    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle

    try:
        if not Path(SAVE_PATH_RAW).exists():
            Path(SAVE_PATH_RAW).mkdir(parents=True, exist_ok=True)
        run_func_output = pd.read_pickle(SAVE_PATH_RAW + 'rfof.pkl')
    except:
        params = list(PARAM_COMBS[0])
        params[-1] = True
        run_func_output = RUN_FUNC(*params)[1]
        with open(SAVE_PATH_RAW+'rfof.pkl', 'wb') as dmp:
            pd.to_pickle(run_func_output, dmp)

    SAMPLE_SIZE = 50 if not (test or approximate in [2, 3]) else 3

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=SAMPLE_SIZE,
                                         parameter_combinations=PARAM_COMBS,
                                         path_raw=SAVE_PATH_RAW
                                         )

    # define eva functions

    def mean(tau, phi, eps, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'tau={} & phi={} & eps={} & approximate={} & test={}'.format(tau, phi, eps, approximate, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(tau, phi, eps, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'tau={} & phi={} & eps={} & approximate={} & test={}'.format(tau, phi, eps, approximate, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        df_out = trj.groupby(level='tstep').std()

        return 1, df_out

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
