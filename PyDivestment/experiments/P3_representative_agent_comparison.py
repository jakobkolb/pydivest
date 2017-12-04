"""
This experiment is meant to create trajectories of macroscopic variables from
1) the numeric micro model and
2) the analytic macro model
3) the representative agent model
that can be compared to evaluate the validity and quality of the analytic
approximation.
The variable Parameters are b_d and phi.

since the representative agent model ist the most reduced version, 
the trajectories of the more complex models will have to be reduced to 
its simplicity.

Also, since the representative agent puts restraints on possible initial conditions, 
we will use it to set the initial conditions for the other models as well.
"""

# TODO: Find a measure that allows to compare trajectories in one real number,
# such that I can produce heat map plots for the parameter dependency of the
# quality of the approximation.


import getpass
import itertools as it
import os
import pickle as cp
import sys
import time
from random import shuffle

import networkx as nx
import numpy as np
import pandas as pd
from pymofa.experiment_handling import experiment_handling, even_time_series_spacing

from pydivest.macro_model.integrate_equations_rep import Integrate_Equations as rep
from pydivest.macro_model.integrate_equations_aggregate import Integrate_Equations as agg
from pydivest.macro_model.integrate_equations_mean import Integrate_Equations as mean
from pydivest.micro_model.divestmentcore import DivestmentCore as micro


def RUN_FUNC(b_d, phi, tau, approximate, test, filename):
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
    model: int
        if 0: run abm
        if 1: run mean approximation
        if 2: run aggregate approximation
        if 3: run representative agent approximation
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # SET PARAMETERS:

    # investment_decisions:

    possible_cue_orders = [[0], [1]]

    # Parameters:

    input_parameters = {'i_tau': tau, 'eps': 0.001, 'b_d': b_d,
                        'b_c': 1., 'i_phi': phi, 'e': 10,
                        'G_0': 10000, 'b_r0': 0.1 ** 2 * 100,
                        'possible_cue_orders': possible_cue_orders,
                        'C': 100, 'xi': 1. / 8., 'd_c': 0.06, 's': 0.23,
                        'pi': 1./2., 'L': 100,
                        'campaign': False, 'learning': True,
                        'crs': True}

    # investment_decisions
    nopinions = [90, 10]
    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full(n, i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:
    N = sum(nopinions)
    p = .2

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # use equilibrium value for only dirty investment here.
    Keq = (input_parameters['s'] * (1 - input_parameters['b_r0']
                                    / input_parameters['e']) / input_parameters['d_c']
           * input_parameters['b_d'] * input_parameters['L'] ** input_parameters['pi']) ** (1
                                                                                            / (input_parameters['pi']))

    # investment
    clean_investment = 1. / N * np.ones(N)
    dirty_investment = Keq / N * np.ones(N)

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)

    models = {}

    m_rep = rep(*init_conditions, **input_parameters)
    C, n = m_rep.find_initial_conditions()

    input_parameters['C'] = C

    # investment_decisions
    nopinions = [int(round((1. - n) * 100.)), int(round(n * 100.))]
    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full((n), i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:
    N = sum(nopinions)
    p = .2

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)

    # initializing the model
    if approximate == 0:
        m = m_rep
    elif approximate == 1:
        m = micro(*init_conditions, **input_parameters)
    elif approximate == 2:
        m = mean(*init_conditions, **input_parameters)
    elif approximate == 3:
        m = agg(*init_conditions, **input_parameters)
    else:
        raise ValueError('approximate must be in [1, 2, 3] but is {}'.format(approximate))

    # storing initial conditions and parameters

    res = {
        "initials": pd.DataFrame({"Investment clean": m.investment_clean,
                                  "Investment dirty": m.investment_dirty}),
        "parameters": pd.Series({"i_tau": m.tau,
                                 "i_phi": m.phi,
                                 "N": m.n,
                                 "L": m.L,
                                 "savings rate": m.s,
                                 "clean capital depreciation rate": m.d_c,
                                 "dirty capital depreciation rate": m.d_d,
                                 "resource extraction efficiency": m.b_r0,
                                 "Solov residual clean": m.b_c,
                                 "Solov residual dirty": m.b_d,
                                 "pi": m.pi,
                                 "kappa_c": m.kappa_c,
                                 "kappa_d": m.kappa_d,
                                 "xi": m.xi,
                                 "resource efficiency": m.e,
                                 "epsilon": m.eps,
                                 "initial resource stock": m.G_0,
                                 "interaction": 2})}

    # run the model
    t_start = time.clock()

    t_max = 500 if not test else 4
    exit_status = m.run(t_max=t_max)

    res["runtime"] = time.clock() - t_start

    # store data in case of successful run
    if exit_status in [0, 1]:
        if approximate == 0:
            res['unified_trajectory'] = m.get_unified_trajectory()
        elif approximate == 1:
            res['mean_macro_trajectory'] = even_time_series_spacing(m.get_mean_trajectory(), 201, 0., t_max)
            res['aggregate_macro_trajectory'] = even_time_series_spacing(m.get_aggregate_trajectory(), 201, 0., t_max)
            res['unified_trajectory'] = even_time_series_spacing(m.get_unified_trajectory(), 201, 0, t_max)
        elif approximate == 2:
            res['mean_macro_trajectory'] = m.get_mean_trajectory()
            res['unified_trajectory'] = m.get_unified_trajectory()
        elif approximate == 3:
            res['aggregate_macro_trajectory'] = m.get_aggregate_trajectory()
            res['unified_trajectory'] = m.get_unified_trajectory()

    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        np.load(filename)
    except IOError:
        print("writing results failed for " + filename)

    return exit_status

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

    respath = os.path.dirname(os.path.realpath(__file__)) + "/../divestdata"
    if getpass.getuser() == "jakob":
        tmppath = respath
    elif getpass.getuser() == "kolb":
        tmppath = "/p/tmp/kolb/Divest_Experiments"
    else:
        tmppath = "./"

    sub_experiment = ['res', 'micro', 'mean', 'aggregate'][approximate]
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
    taus = [10**x for x in list(np.linspace(-1, 4., 10))]
    phis = [round(x, 5) for x in list(np.linspace(0.0, 1., 10))]
    b_ds = [round(x, 5) for x in list(np.linspace(1., 1.5, 3))]
    b_d, phi = [1.2], [.8]

    if test:
        PARAM_COMBS = list(it.product(b_d, phi, taus, [approximate], [test]))
    else:
        PARAM_COMBS = list(it.product(b_ds, phis, taus, [approximate], [test]))

    INDEX = {0: "b_d", 1: "phi", 2: "tau"}

    """
    create names and dicts of callables for post processing
    """

    NAME0 = 'unified_trajectory'
    EVA0 = {"mean_trajectory":
                lambda fnames: pd.concat([np.load(f)["unified_trajectory"]
                                          for f in fnames]).groupby(
                    level=0).mean(),
            "sem_trajectory":
                lambda fnames: pd.concat([np.load(f)["unified_trajectory"]
                                          for f in fnames]).groupby(level=0).std(),
            }
    NAME1 = 'mean_trajectory'
    EVA1 = {"mean_trajectory":
            lambda fnames: pd.concat([np.load(f)["mean_macro_trajectory"]
                                      for f in fnames]).groupby(
                    level=0).mean(),
            "sem_trajectory":
            lambda fnames: pd.concat([np.load(f)["mean_macro_trajectory"]
                                      for f in fnames]).groupby(level=0).std(),
            }
    NAME2 = 'aggregate_trajectory'
    EVA2 = {"mean_trajectory":
            lambda fnames: pd.concat([np.load(f)["aggregate_macro_trajectory"]
                                      for f in fnames]).groupby(
                    level=0).mean(),
            "sem_trajectory":
            lambda fnames: pd.concat([np.load(f)["aggregate_macro_trajectory"]
                                      for f in fnames]).groupby(level=0).std(),
            }

    """
    run computation and/or post processing and/or plotting
    """

    # cluster mode: computation and post processing
    if mode == 0:
        print('calculating {}: {}'.format(approximate, sub_experiment))
        sys.stdout.flush()
        SAMPLE_SIZE = 100 if not (test or approximate in [2, 3]) else 2
        handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                     SAVE_PATH_RAW, SAVE_PATH_RES)
        handle.compute(RUN_FUNC)

        return 1

    # Post processing
    if mode == 1:
        sys.stdout.flush()
        SAMPLE_SIZE = 100 if not (test or approximate in [0, 2, 3]) else 5

        handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                     SAVE_PATH_RAW, SAVE_PATH_RES)

        if approximate == 0:
            print('post processing representative model')
            handle.resave(EVA0, NAME0)
        elif approximate == 1:
            print('post processing micro model')
            handle.resave(EVA0, NAME0)
            handle.resave(EVA1, NAME1)
            handle.resave(EVA2, NAME2)
        elif approximate == 2:
            print('post processing mean macro approximation')
            handle.resave(EVA0, NAME0)
            handle.resave(EVA1, NAME1)
        elif approximate == 3:
            print('post processing aggregate macro approximation')
            handle.resave(EVA0, NAME0)
            handle.resave(EVA2, NAME2)

        return 1

    # in case nothing happened:
    return 0


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
