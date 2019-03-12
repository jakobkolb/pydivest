"""
Compare Trajectory from micro simulation for new and old interaction between households.
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
from pymofa.experiment_handling import experiment_handling, \
even_time_series_spacing

from pydivest.micro_model.divestmentcore import DivestmentCore as micro
from pydivest.macro_model.integrate_equations_aggregate \
    import IntegrateEquationsAggregate as aggregate

from pydivest.default_params import ExperimentDefaults


def RUN_FUNC(interaction, phi, b_d, model, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the 
    initial values, parameters and convergence state and time 
    for each run.

    Parameters:
    -----------
    interaction: int
        determines the functional form of the imitation probability between households.

    b_d : float > 0
        the solow residual in the dirty sector
    phi : float \in [0,1]
        the rewiring probability for the network update
    model: bool
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
    input_params['interaction'] = interaction
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

    if model == 1:
        m = micro(*init_conditions, **input_params)
    elif model == 2:
        m = aggregate(*init_conditions, **input_params)
    else:
        raise ValueError(f'model needs to be in [1, 2] but is {model}')

    t_max = 300 if not test else 300
    m.R_depletion = False
    m.run(t_max=t_max)

    t_max += 600 if not test else 600
    m.R_depletion = True
    m.set_parameters()
    exit_status = m.run(t_max=t_max)

    # store data in case of successful run
    if exit_status in [0, 1]:
        # interpolate m_trajectory to get evenly spaced time series.
        df1 = even_time_series_spacing(m.get_aggregate_trajectory(), 201, 0., t_max)
        df2 = even_time_series_spacing(m.get_unified_trajectory(), 201, 0., t_max)

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
        test = True
    # switch sub_experiment mode
    if len(argv) > 2:
        mode = int(argv[2])
    else:
        mode = 0

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

    folder = 'P1'

    # make sure, testing output goes to its own folder:

    test_folder = ['', 'test_output/'][int(test)]

    save_path_raw = \
        "{}/{}{}/" \
            .format(tmppath, test_folder, folder)
    save_path_res = \
        "{}/{}{}/" \
            .format(respath, test_folder, folder)

    """
    create parameter combinations and index
    """

    phis = [round(x, 2) for x in list(np.linspace(0.0, 1., 21))]
    b_ds = [round(x, 2) for x in list(np.linspace(1., 1.5, 5))]
    interactions = [0, 1, 2]
    model = [1, 2]
    b_d, phi, interaction = [1.25], [.5], [0, 1]

    if test:
        param_combs = list(it.product(interaction, phi, b_d, model, [test]))
    else:
        param_combs = list(it.product(interactions, phis, b_ds, model, [test]))

    """
    run computation and/or post processing and/or plotting
    """
    # Create dummy runfunc output to pass its shape to experiment handle
    try:
        if not Path(save_path_raw).exists():
            Path(save_path_raw).mkdir()
        run_func_output = pd.read_pickle(save_path_raw + 'rfof.pkl')
    except:
        params = list(param_combs[0])
        params[-1] = True
        run_func_output = RUN_FUNC(*params)[1]
        with open(save_path_raw+'rfof.pkl', 'wb') as dmp:
            pd.to_pickle(run_func_output, dmp)

    sample_size = 100 if not test else 30

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=sample_size,
                                         parameter_combinations=param_combs,
                                         path_raw=save_path_raw
                                         )

    # define eva functions

    def mean(interaction, phi, b_d, model, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'b_d={} & phi={} & interaction={} & model={} & test={}'.format(b_d, phi, interaction, model, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(interaction, phi, b_d, model, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'b_d={} & phi={} & interaction={} & model={} & test={}'.format(b_d, phi, interaction, model, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').std()

    eva_1_handle = experiment_handling(run_func=mean,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=param_combs,
                                       path_raw=save_path_res + '/mean.h5'
                                       )
    eva_2_handle = experiment_handling(run_func=std,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=param_combs,
                                       path_raw=save_path_res + '/std.h5'
                                       )

    if mode == 0:
        # computation, parameters split between threads
        compute_handle.compute()
        return 1
    elif mode == 1:
        # post processing (all parameter combinations on one thread)
        print('post processing')
        eva_1_handle.compute()
        eva_2_handle.compute()
        return 1
    else:
        # in case nothing happened:
        return 0


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)