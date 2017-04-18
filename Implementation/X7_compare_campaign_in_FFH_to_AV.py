"""
Compare campaign dynamics in the full Fast and Frugal Heuristics mode to
the strapped down "adaptive voter" case.
Both cases have equal parameters and comparable initial conditions (equilibrium
with abundant fossil resource) as well as the same campaign starting size and
dynamics.
"""

import pickle as cp
import getpass
import glob
import itertools as it
import numpy as np
import sys
import os
import time

import networkx as nx
import pandas as pd

from micro_model \
    import divestment_core as micro_model
from pymofa.experiment_handling \
    import experiment_handling, even_time_series_spacing
from divestvisuals.data_visualization \
    import plot_trajectories, plot_amsterdam


def RUN_FUNC(b_d, phi, ffh, test, transition, filename):
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
    ffh: bool
        if True: run with fast and frugal heuristics
        if False: run with imitation only
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # fraction of households that will start the campaign
    ccount = .05

    # Make different types of decision makers. Cues are

    cue_names = {
        0: 'always dirty',
        1: 'always clean',
        2: 'capital rent',
        3: 'capital rent trend',
        4: 'peer pressure',
        5: 'campaignee'}
    if ffh:
        possible_opinions = [[2, 3],  # short term investor
                             [3, 2],  # long term investor
                             [4, 2],  # short term herder
                             [4, 3],  # trending herder
                             [4, 1],  # green conformer
                             [4, 0],  # dirty conformer
                             [1],  # gutmensch
                             [0]]  # redneck
    else:
        possible_opinions = [[1], [0]]

    # Parameters:

    input_params = {'b_c': 1., 'phi': phi, 'tau': 1.,
                    'eps': 0.05, 'b_d': b_d, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100.,  # alpha^2 * e
                    'possible_opinions': possible_opinions,
                    'xi': 1. / 8., 'beta': 0.06,
                    'P': 100., 'C': 100., 'G_0': 1600.,
                    'campaign': False, 'learning': True,
                    'test': test, 'R_depletion': False}

    # building initial conditions

    if not transition:
        # network:
        N = 100
        k = 10
        if test:
            N = 30
            k = 3

        p = float(k) / N
        while True:
            net = nx.erdos_renyi_graph(N, p)
            if len(list(net)) > 1:
                break
        adjacency_matrix = nx.adj_matrix(net).toarray()

        # opinions and investment

        opinions = [np.random.randint(0, len(possible_opinions))
                    for x in range(N)]
        clean_investment = np.ones(N) * 50. / float(N)
        dirty_investment = np.ones(N) * 50. / float(N)

        init_conditions = (adjacency_matrix, opinions,
                           clean_investment, dirty_investment)

        t_1 = 400
        t_2 = 0

        # initializing the model
        m = micro_model.Divestment_Core(*init_conditions, **input_params)

    else:
        # build list of initial conditions
        # phi, alpha and t_d are relevant,
        # t_a is not. Parse filename to get
        # wildcard for all relevant files.
        # replace characters before first
        # underscore with *
        [path, fname] = filename.rsplit('/', 1)
        plen = len(path)
        path += '/*' + fname.rsplit('True', 1)[0] + 'False_*.pkl'
        if phi == 1.0:
            path = path[:plen + 6] + '0o9' + path[plen + 9:]
        path = path.replace('trans', 'equi')
        init_files = glob.glob(path)
        input_params = np.load(
            init_files[np.random.randint(0, len(init_files))])['final_state']

        # update input parameters where necessary
        input_params['campaign'] = True
        input_params['possible_opinions'].append([5])
        campaigner = len(input_params['possible_opinions']) - 1

        # make fraction of ccount households campaigners
        opinions = input_params['opinions']
        decisions = input_params['investment decisions']
        del input_params['investment decisions']
        nccount = int(ccount * len(opinions))
        j = 0
        i = 0
        while j < nccount:
            i += 1
            n = np.random.randint(0, len(opinions))
            # recruit campaigners only people in favor of the cause
            if opinions[n] != campaigner and decisions[n] == 1:
                opinions[n] = campaigner
                j += 1
            if i > 100 * len(opinions):
                break
        input_params['opinions'] = opinions

        t_1 = 400
        t_2 = 600

        # initializing the model
        m = micro_model.Divestment_Core(**input_params)

    # storing initial conditions and parameters
    res = {
        "initials": pd.DataFrame({"opinions": opinions,
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

    # start timer
    t_start = time.clock()

    # run model with abundant resource
    t_max = t_1 if not test else 1
    m.R_depletion = False
    m.run(t_max=t_max)

    # run model with resource depletion
    t_max += t_2 if not test else 1
    m.R_depletion = True
    exit_status = m.run(t_max=t_max)

    # for equilibration runs, save final state of the model:
    if not transition:
        res['final_state'] = m.final_state

    res["runtime"] = time.clock() - t_start

    # store data in case of successful run
    if exit_status in [0, 1]:
        res["micro_trajectory"] = \
            even_time_series_spacing(m.get_e_trajectory(), 401, 0., t_max)
        res["convergence_state"] = m.convergence_state
        res["convergence_time"] = m.convergence_time

    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        np.load(filename)
    except IOError:
        print("writing results failed for " + filename)

    return exit_status


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
    # switch transition
    if len(argv) > 4:
        transition = bool(int(argv[4]))
    else:
        transition = False


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

    sub_experiment = ['imitation', 'ffh'][int(ffh)] \
                 + ['_equi', '_trans'][int(transition)]
    folder = 'X7'

    # make sure, testing output goes to its own folder:

    test_folder = ['', 'test_output/'][int(test)]

    # check if cluster or local and set paths accordingly
    SAVE_PATH_RAW = \
        "{}/{}{}/{}/" \
        .format(tmppath, test_folder, folder, sub_experiment)
    SAVE_PATH_RES = \
        "{}/{}{}/{}/" \
        .format(respath, test_folder, folder, sub_experiment)

    """
    create parameter combinations and index
    """

    phis = [round(x, 5) for x in list(np.linspace(0.0, 1., 11))]
    b_ds = [round(x, 5) for x in list(np.linspace(1., 2., 11))]
    b_d, phi = [1.75, 2.0], [.7, .8, .9]

    if ffh:
        possible_opinions = [[2, 3],  # short term investor
                             [3, 2],  # long term investor
                             [4, 2],  # short term herder
                             [4, 3],  # trending herder
                             [4, 1],  # green conformer
                             [4, 0],  # dirty conformer
                             [1],  # gutmensch
                             [0]]  # redneck
    else:
        possible_opinions = [[1], [0]]

    cue_list = [str(o) for o in possible_opinions]
    if transition:
        cue_list.append('[5]')

    if test:
        PARAM_COMBS = list(it.product(b_d, phi, [ffh], [test], [transition]))
    else:
        PARAM_COMBS = list(it.product(b_ds, phis, [ffh], [test], [transition]))

    INDEX = {0: "b_c", 1: "phi"}


    """
    create names and dicts of callables for post processing
    """

    NAME = 'b_c_scan_' + sub_experiment + '_trajectory'

    NAME1 = NAME + '_trajectory'
    EVA1 = {"mean_trajectory":
                lambda fnames: pd.concat([np.load(f)["micro_trajectory"]
                                          for f in fnames]).groupby(
                    level=0).mean(),
            "sem_trajectory":
                lambda fnames: pd.concat([np.load(f)["micro_trajectory"]
                                          for f in fnames]).groupby(
                    level=0).std()
            }
    NAME2 = NAME + '_convergence'
    EVA2 = {'times_mean':
                lambda fnames: np.nanmean([np.load(f)["convergence_time"]
                                           for f in fnames]),
            'states_mean':
                lambda fnames: np.nanmean([np.load(f)["convergence_state"]
                                           for f in fnames]),
            'times_std':
                lambda fnames: np.std([np.load(f)["convergence_time"]
                                       for f in fnames]),
            'states_std':
                lambda fnames: np.std([np.load(f)["convergence_state"]
                                       for f in fnames])
            }
    NAME3 = NAME + '_convergence_times'
    CF3 = {'times':
               lambda fnames: pd.DataFrame(data=[np.load(f)["convergence_time"]
                                                 for f in fnames]).sortlevel(
                   level=0),
           'states':
               lambda fnames: pd.DataFrame(
                   data=[np.load(f)["convergence_state"]
                         for f in fnames])
                   .sortlevel(level=0)
           }

    """
    run computation and/or post processing and/or plotting
    """

    # cluster mode: computation and post processing
    if mode == 0:
        print('cluster mode')
        sys.stdout.flush()

        SAMPLE_SIZE = 100 if not test else 2

        handle = experiment_handling(SAMPLE_SIZE, PARAM_COMBS, INDEX,
                                     SAVE_PATH_RAW, SAVE_PATH_RES)
        handle.compute(RUN_FUNC)
        handle.resave(EVA1, NAME1)
        handle.resave(EVA2, NAME2)
        handle.collect(CF3, NAME3)

        return 1

    # local mode: plotting only
    if mode == 1:
        print('plot mode')
        sys.stdout.flush()

        plot_amsterdam(SAVE_PATH_RES, NAME1, cues=cue_list)
        plot_trajectories(SAVE_PATH_RES, NAME1, None, None)

        return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
