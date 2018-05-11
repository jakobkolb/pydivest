"""
I want to know, whether the imitation process leads to equal return rates in both sectors.
Parameters that this could depend on are

1) the rate of exploration (random changes in opinion and rewiring),
2) also, the rate of rewiring could have an effect.

This should only work in the equilibrium condition where the environment stays constant.

"""

try:
    import cPickle as cp
except ImportError:
    import pickle as cp
import getpass
import itertools as it
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from pymofa.experiment_handling \
    import experiment_handling, even_time_series_spacing

from pydivest.divestvisuals.data_visualization \
    import plot_amsterdam, plot_trajectories
from pydivest.micro_model import divestmentcore as model


def RUN_FUNC(eps, phi, ffh, test, filename):
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
    xi : float \in [0,0.5]
        exponent for knowledge stock in the clean production function
    ffh: bool
        if True: run with fast and frugal heuristics
        if False: run with imitation only
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Make different types of decision makers. Cues are

    if ffh:
        possible_cue_orders = [[2, 3],  # short term investor
                             [3, 2],  # long term investor
                             [4, 2],  # short term herder
                             [4, 3],  # trending herder
                             [4, 1],  # green conformer
                             [4, 0],  # dirty conformer
                             [1],  # gutmensch
                             [0]]  # redneck
    else:
        possible_cue_orders = [[1], [0]]

    # Parameters:

    input_params = {'b_c': 1., 'i_phi': phi, 'i_tau': 1.,
                    'eps': eps, 'b_d': 1.5, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100.,  # alpha^2 * e
                    'possible_cue_orders': possible_cue_orders,
                    'xi': 1. / 8., 'beta': 0.06,
                    'L': 100., 'C': 100., 'G_0': 1600.,
                    'campaign': False, 'learning': True,
                    'test': test, 'R_depletion': False}

    # building initial conditions

    # network:
    n = 100
    k = 10
    if test:
        n = 30
        k = 3

    p = float(k) / n
    while True:
        net = nx.erdos_renyi_graph(n, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # opinions and investment

    opinions = [np.random.randint(0, len(possible_cue_orders))
                for x in range(n)]
    clean_investment = np.ones(n) * 50. / float(n)
    dirty_investment = np.ones(n) * 50. / float(n)

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)

    t_1 = 400

    # initializing the model
    m = model.DivestmentCore(*init_conditions, **input_params)

    # run model with abundant resource
    t_max = t_1 if not test else 1
    m.R_depletion = False
    exit_status = m.run(t_max=t_max)

    # store data in case of successful run
    if exit_status in [0, 1] or test:
        df_out = even_time_series_spacing(m.get_economic_trajectory(), 401, 0., t_max)
        df_out.index.name = 'tstep'
    else:
        df_out = None

    return exit_status, df_out


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
        test = bool(int(argv[1]))
    else:
        test = False
    # switch sub_experiment mode
    if len(argv) > 2:
        mode = int(argv[2])
    else:
        mode = 0
    # switch decision making
    if len(argv) > 3:
        ffh = bool(int(argv[3]))
    else:
        ffh = True

    """
    set input/output paths
    """
    respath = os.path.dirname(os.path.realpath(__file__)) + "/output_data"
    if getpass.getuser() == "jakob":
        tmppath = respath
    elif getpass.getuser() == "kolb":
        tmppath = "/p/tmp/kolb/Divest_Experiments"
    else:
        tmppath = "./"

    sub_experiment = ['imitation', 'ffh'][int(ffh)]
    folder = 'X1'

    # make sure, testing output goes to its own folder:

    test_folder = ['', 'test_output/'][int(test)]

    print(test_folder)

    # check if cluster or local and set paths accordingly
    save_path_raw = \
        "{}/{}{}/{}/" \
        .format(tmppath, test_folder, folder, sub_experiment)
    save_path_res = \
        "{}/{}{}/{}/" \
        .format(respath, test_folder, folder, sub_experiment)

    print(save_path_raw)

    """
    create parameter combinations and index
    """

    epss = [round(x, 5) for x in list(np.linspace(0.0, 0.1, 11))]
    phis = [round(x, 5) for x in list(np.linspace(0., 1., 11))]
    eps, phi = [0., 0.5], [.7, .9]

    if test:
        param_combs = list(it.product(eps, phi, [ffh], [test]))
    else:
        param_combs = list(it.product(epss, phis, [ffh], [test]))

    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle
    params = list(param_combs[0])
    params[-1] = True
    run_func_output = RUN_FUNC(*params)[1]

    sample_size = 100 if not test else 3

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=sample_size,
                                         parameter_combinations=param_combs,
                                         path_raw=save_path_raw
                                         )

    # define eva functions

    def mean(eps, phi, ffh, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'eps={} & phi={} & ffh={} & test={}'.format(eps, phi, ffh, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(eps, phi, ffh, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'eps={} & phi={} & ffh={} & test={}'.format(eps, phi, ffh, test)

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
