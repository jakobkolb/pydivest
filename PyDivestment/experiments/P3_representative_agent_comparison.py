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

# TODO: Find a measure that allows to copmare trajectories in one real number,
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

from pydivest.macro_model.integrate_equations_aggregate import Integrate_Equations as aggregate_macro_model
from pydivest.macro_model.integrate_equations_mean import Integrate_Equations as mean_macro_model
from pydivest.macro_model.integrate_equations_rep import Integrate_Equations as rep_agent_macro_model
from pydivest.micro_model import divestmentcore as micro_model


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

    input_params = {'b_c': 1., 'i_phi': phi, 'i_tau': tau,
                    'eps': 0.05, 'b_d': b_d, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100.,  # Todo: find out, why I did this
                    'possible_cue_orders': [[0], [1]],
                    'xi': 1. / 8., 'beta': 0.06,
                    'L': 100., 'C': 100., 'G_0': 800.,
                    'campaign': False, 'learning': True,
                    'interaction': 1}
    K_c = 50
    K_d = 50

    # SET INITIAL CONDITIONS

    # investment_decisions:
    nopinions = [100, 100]
    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full((n), i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:
    N = sum(nopinions)
    k = 10
    p = float(k) / N
    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    clean_investment = np.ones(N) * K_c / float(N)
    dirty_investment = np.ones(N) * K_d / float(N)

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)

    # initialize the representative agent model
    ra_model = rep_agent_macro_model(*init_conditions, **input_params)

    # get sane initial conditions for C and n
    C_val, n_val = ra_model.find_initial_conditions()

    # update input parameters
    input_params['C'] = C_val

    # this only works to decrease n. But since I start from a 50/50 distribution and
    # abundant fossil resrouces, this is the only case tha matters.
    # maybe crosscheck, that this is true, and n_val < 0.5

    assert n_val < 0.5
    while True:
        n_is = sum(investment_decisions)
        if n_is > n_val:
            # TODO: pick a random household and set it to invest in dirty.
            pass
        else:
            break


    # initializing the model
    if approximate == 1:
        m = micro_model.DivestmentCore(*init_conditions, **input_params)
    elif approximate == 2:
        m = mean_macro_model.Integrate_Equations(*init_conditions, **input_params)
    elif approximate == 3:
        m = aggregate_macro_model.Integrate_Equations(*init_conditions, **input_params)
    else:
        raise ValueError('approximate must be in [1, 2, 3] but is {}'.format(approximate))

    # storing initial conditions and parameters

    res = {
        "initials": pd.DataFrame({"Investment decisions": investment_decisions,
                                  "Investment clean": m.investment_clean,
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

    t_max = 200 if not test else 2
    m.R_depletion = False
    m.run(t_max=t_max)

    t_max += 400 if not test else 4
    m.R_depletion = True
    exit_status = m.run(t_max=t_max)

    res["runtime"] = time.clock() - t_start

    # store data in case of successful run
    if exit_status in [0, 1]:
        if approximate == 1:
            res['mean_macro_trajectory'] = even_time_series_spacing(m.get_mean_trajectory(), 201, 0., t_max)
            res['aggregate_macro_trajectory'] = even_time_series_spacing(m.get_mean_trajectory(), 201, 0., t_max)
        elif approximate == 2:
            res['mean_macro_trajectory'] = m.get_mean_trajectory()
        elif approximate == 3:
            res['aggregate_macro_trajectory'] = m.get_aggregate_trajectory()

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

    sub_experiment = ['micro', 'mean', 'aggregate'][approximate - 1]
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
    b_d, phi = [1.2], [.8]

    if test:
        PARAM_COMBS = list(it.product(b_d, phi, [approximate], [test]))
    else:
        PARAM_COMBS = list(it.product(b_ds, phis, [approximate], [test]))

    INDEX = {0: "b_d", 1: "phi"}

    """
    create names and dicts of callables for post processing
    """

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
        SAMPLE_SIZE = 100 if not (test or approximate in [2, 3]) else 2

        handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                     SAVE_PATH_RAW, SAVE_PATH_RES)

        if approximate == 1:
            print('post processing micro model')
            handle.resave(EVA1, NAME1)
            handle.resave(EVA2, NAME2)
        elif approximate == 2:
            print('post processing mean macro approximation')
            handle.resave(EVA1, NAME1)
        elif approximate == 3:
            print('post processing aggregate macro approximation')
            handle.resave(EVA2, NAME2)

        return 1

    # in case nothing happened:
    return 0


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
