"""
This experiment is meant to create trajectories of macroscopic variables from
1) the numeric micro model and
2) the analytic macro model
that can be compared to evaluate the validity and quality of the analytic
approximation.
The variable Parameters are b_d and phi.
"""

import getpass
import itertools as it
import os
import pickle as cp
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

from pydivest.divestvisuals.data_visualization import plot_trajectories
from pydivest.macro_model import integrate_equations as macro_model
from pydivest.micro_model import divestment_core as micro_model
from pymofa.experiment_handling import experiment_handling, \
    even_time_series_spacing


def RUN_FUNC(b_d, phi, approximate, test, filename):
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

    input_params = {'b_c': 1., 'phi': phi, 'tau': 1.,
                    'eps': 0.05, 'b_d': b_d, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100.,
                    'possible_opinions': [[0], [1]],
                    'xi': 1. / 8., 'beta': 0.06,
                    'P': 100., 'C': 100., 'G_0': 800.,
                    'campaign': False, 'learning': True}

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
    print('approximate', approximate)
    if approximate:
        m = macro_model.Integrate_Equations(*init_conditions, **input_params)
    else:
        m = micro_model.Divestment_Core(*init_conditions, **input_params)
        m.init_switchlist()

    # storing initial conditions and parameters

    res = {
        "initials": pd.DataFrame({"Investment decisions": investment_decisions,
                                  "Investment clean": m.investment_clean,
                                  "Investment dirty": m.investment_dirty}),
        "parameters": pd.Series({"tau": m.tau,
                                 "phi": m.phi,
                                 "N": m.n,
                                 "P": m.P,
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
                                 "initial resource stock": m.G_0})}

    # run the model
    t_start = time.clock()

    t_max = 200 if not test else 1
    m.R_depletion = False
    m.run(t_max=t_max)

    t_max += 400 if not test else 1
    m.R_depletion = True
    exit_status = m.run(t_max=t_max)

    res["runtime"] = time.clock() - t_start

    # store data in case of successful run
    if exit_status in [0, 1]:
        # interpolate m_trajectory to get evenly spaced time series.
        res["macro_trajectory"] = \
            even_time_series_spacing(m.get_m_trajectory(), 201, 0., t_max)
        if not approximate:
            res["switchlist"] = m.get_switch_list()

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
        approximate = 0

    """
    set input/output paths
    """

    respath = os.path.dirname(os.path.realpath(__file__)) + "/divestdata"
    if getpass.getuser() == "jakob":
        tmppath = respath
    elif getpass.getuser() == "kolb":
        tmppath = "/p/tmp/kolb/Divest_Experiments"
    else:
        tmppath = "./"

    sub_experiment = ['micro', 'macro'][approximate]
    folder = 'X6'

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
    b_d, phi, approximate, exact = [1.2], [.8], [True], [False]

    if test:
        PARAM_COMBS = list(it.product(b_d, phi, [bool(approximate)], [test]))
    else:
        PARAM_COMBS = list(it.product(b_ds, phis, [bool(approximate)], [test]))

    INDEX = {0: "b_d", 1: "phi"}

    """
    create names and dicts of callables for post processing
    """

    NAME = 'b_c_scan_' + sub_experiment + '_trajectory'

    NAME1 = NAME + '_trajectory'
    EVA1 = {"mean_trajectory":
            lambda fnames: pd.concat([np.load(f)["macro_trajectory"]
                                      for f in fnames]).groupby(
                    level=0).mean(),
            "sem_trajectory":
            lambda fnames: pd.concat([np.load(f)["macro_trajectory"]
                                      for f in fnames]).groupby(level=0).std()
            }
    NAME2 = NAME + '_switchlist'
    CF2 = {"switching_capital":
           lambda fnames: pd.concat([np.load(f)["switchlist"]
                                     for f in fnames]).sortlevel(level=0)
           }

    """
    run computation and/or post processing and/or plotting
    """

    # cluster mode: computation and post processing
    if mode == 0:
        print('cluster mode')
        sys.stdout.flush()
        SAMPLE_SIZE = 100 if not (test or approximate == 1) else 2
        handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                     SAVE_PATH_RAW, SAVE_PATH_RES)
        handle.compute(RUN_FUNC)
        handle.resave(EVA1, NAME1)
        if approximate == 0:
            handle.collect(CF2, NAME2)

        return 1

    # local mode: plotting only
    if mode == 1:
        print('plot mode')
        sys.stdout.flush()
        plot_trajectories(SAVE_PATH_RES, NAME1, None, None)

        return 1

    # in case nothing happened:
    return 0


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
