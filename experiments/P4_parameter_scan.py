"""
from parameters import ExperimentDefaults
Scan the economic variables of the system especially
b_R: the resource cost,
b_d: the total factor productivity in the dirty sector,
xi: the elasticity of knowledge in the clean sector.
to see, which of the two sectors dominates.
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
from pymofa.experiment_handling import experiment_handling, even_time_series_spacing

from pydivest.macro_model.integrate_equations_aggregate import IntegrateEquationsAggregate
from pydivest.macro_model.integrate_equations_rep import Integrate_Equations as IntegrateEquationsRep
from pydivest.micro_model.divestmentcore import DivestmentCore

from pydivest.default_params import ExperimentDefaults


def RUN_FUNC(b_d, b_R, xi, approximate, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the 
    initial values, parameters and convergence state and time 
    for each run.

    Parameters:
    -----------
    b_d: float
        total factor productivity in the dirty sector
    b_R: float
        resource extraction cost in units of the final product
    xi: float
        elasticity of knowledge in the production in the clean sector
    approximate: bool
        if True: run macroscopic approximation
        if False: run micro-model
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Parameters:

    input_params = ExperimentDefaults.input_params

    input_params['b_d'] = b_d
    input_params['b_r0'] = b_R
    input_params['xi'] = xi
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

    t_max = 300
    exit_status = m.run(t_max=t_max)

    # transition phase with resource depletion

    # store data in case of successful run
    if exit_status in [0, 1]:
        if approximate in [0, 1, 4]:
            df1 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, 0, t_max)
            df2 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0, t_max)
        else:
            df2 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, 0, t_max)
            df1 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0, t_max)

        for c in df2.columns:
            if c in df1.columns:
                df2.drop(c, axis=1, inplace=True)

        df_out = pd.concat([df1, df2], axis=1)
        df_out.index.name = 'tstep'
    else:
        df_out = None

    # remove output that is not needed for production plot to write less on database
    rm_columns = ['mu_c^c', 'mu_c^d', 'mu_d^c', 'mu_d^d', 'l_c', 'l_d', 'r', 'r_c', 'r_d',
                  'w', 'W_c', 'W_d', 'n_c', 'i_c', 'wage', 'r_c_dot', 'r_d_dot', 'K_c', 'K_d', 'P_c',
                  'P_d', 'L', 'R', 'P_c_cost', 'P_d_cost', 'K_c_cost',
                  'K_d_cost', 'c_R', 'consensus', 'decision state', 'G_alpha', '[0]',
                  '[1]', 'c[0]', 'c[1]', 'd[0]', 'd[1]']

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

    # switch testing mode
    if len(argv) > 1:
        test = bool(int(argv[1]))
    else:
        test = False
    # switch micro macro model
    if len(argv) > 2:
        approximate = int(argv[2])
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
    folder = 'P4'

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

    b_ds = [round(x, 5) for x in list(np.linspace(1., 4., 4))]
    b_Rs = [round(x, 5) for x in list(np.linspace(.1, .3, 3))]
    xis = [round(x, 5) for x in list(np.linspace(.0, .08, 81))]

    if test:
        PARAM_COMBS = list(it.product(list(np.linspace(3., 4., 4)),
                                      list(np.linspace(.1, .3, 3)),
                                      list(np.linspace(.0, .08, 9)), [approximate], [test]))
    else:
        PARAM_COMBS = list(it.product(b_ds, b_Rs, xis, [approximate], [test]))

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
        params[-1] = False
        run_func_output = RUN_FUNC(*params)[1]
        with open(SAVE_PATH_RAW+'rfof.pkl', 'wb') as dmp:
            pd.to_pickle(run_func_output, dmp)

    SAMPLE_SIZE = 10 if (test or approximate in [2, 3]) else 5

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=SAMPLE_SIZE,
                                         parameter_combinations=PARAM_COMBS,
                                         path_raw=SAVE_PATH_RAW
                                         )

    # define eva functions

    def mean(b_d, b_R, xi, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'b_d={} & b_R={} & xi={} & approximate={} & test={}'.format(b_d, b_R, xi, approximate, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(b_d, b_R, xi, approximate, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'b_d={} & b_R={} & xi={} & approximate={} & test={}'.format(b_d, b_R, xi, approximate, test)

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

    compute_handle.compute()
    eva_1_handle.compute()
    eva_2_handle.compute()
    return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
