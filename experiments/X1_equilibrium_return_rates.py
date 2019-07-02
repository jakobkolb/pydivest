"""
I want to know, whether the imitation process leads to equal return rates in
both sectors.
Parameters that this could depend on are

1) the rate of exploration (random changes in opinion and rewiring),
2) also, the rate of rewiring could have an effect.

This should only work in the equilibrium condition where the environment stays
constant.

"""

# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3

import copy
import getpass
import itertools as it
import os
import pickle as cp
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pydivest.default_params import ExperimentDefaults
from pydivest.divestvisuals.data_visualization import (plot_amsterdam,
                                                       plot_trajectories)
from pydivest.micro_model import divestmentcore as model
from pymofa.experiment_handling import (even_time_series_spacing,
                                        experiment_handling)


def load(*args, **kwargs):
    return np.load(*args, allow_pickle=True, **kwargs)


def RUN_FUNC(eps, phi, ffh, test):
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
    xi : float in [0,0.5]
        exponent for knowledge stock in the clean production function
    ffh: bool
        if True: run with fast and frugal heuristics
        if False: run with imitation only
    test: int in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Make different types of decision makers. Cues are

    if ffh:
        possible_opinions = [
            [2, 3],  # short term investor
            [3, 2],  # long term investor
            [4, 2],  # short term herder
            [4, 3],  # trending herder
            [4, 1],  # green conformer
            [4, 0],  # dirty conformer
            [1],  # gutmensch
            [0]
        ]  # redneck
    else:
        possible_opinions = [[1], [0]]

    # Parameters:
    input_params = ExperimentDefaults.input_params
    input_params['possible_cue_orders'] = possible_opinions
    input_params['phi'] = phi
    input_params['eps'] = eps

    # building initial conditions

    # network:
    n = 100
    k = 10

    p = float(k) / n

    while True:
        net = nx.erdos_renyi_graph(n, p)

        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # opinions and investment

    opinions = [np.random.randint(0, len(possible_opinions)) for x in range(n)]
    clean_investment = np.ones(n) * 50. / float(n)
    dirty_investment = np.ones(n) * 50. / float(n)

    init_conditions = (adjacency_matrix, opinions, clean_investment,
                       dirty_investment)

    t_1 = 400 if not test else 40

    # initializing the model
    m = model.DivestmentCore(*init_conditions, **input_params)

    # start timer
    t_start = time.clock()

    # run model with abundant resource
    t_max = t_1 if not test else 1
    m.R_depletion = False
    exit_status = m.run(t_max=t_max)

    res = {}
    res["runtime"] = [time.clock() - t_start]

    # store data in case of successful run

    if test:
        exit_status = 1
    df1 = even_time_series_spacing(m.get_economic_trajectory(), 401, 0.,
                                   t_max)
    df1.index.name = 'tstep'
    res["convergence_state"] = [m.convergence_state]
    res["convergence_time"] = [m.convergence_time]

    df2 = pd.DataFrame.from_dict(res)
    df2.index.name = 'i'

    # save data


#   with open(filename, 'wb') as dumpfile:
#       cp.dump(res, dumpfile)
#   try:
#       load(filename)
#   except IOError:
#       print("writing results failed for " + filename)

    return exit_status, [df1, df2]


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
        test = True
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
    respath = os.path.dirname(os.path.realpath(__file__)) + "/../output_data"

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

    epss = [round(x, 5) for x in list(np.linspace(0.0, 0.05, 11))]
    phis = [round(x, 5) for x in list(np.linspace(0., 1., 11))]
    eps, phi = [0., 0.5], [.7, .9]

    if ffh:
        possible_opinions = [
            [2, 3],  # short term investor
            [3, 2],  # long term investor
            [4, 2],  # short term herder
            [4, 3],  # trending herder
            [4, 1],  # green conformer
            [4, 0],  # dirty conformer
            [1],  # gutmensch
            [0]
        ]  # redneck
    else:
        possible_opinions = [[1], [0]]

    cue_list = [str(o) for o in possible_opinions]

    if test:
        param_combs = list(it.product(eps, phi, [ffh], [test]))
    else:
        param_combs = list(it.product(epss, phis, [ffh], [test]))

    index = {0: "eps", 1: "phi"}
    """
    create names and dicts of callables for post processing
    """

    name = 'b_c_scan_' + sub_experiment + '_trajectory'

    name1 = name + '_trajectory'
    eva1 = {
        "mean_trajectory":
        lambda fnames: pd.concat([load(f)["micro_trajectory"] for f in fnames]
                                 ).groupby(level=0).mean(),
        "sem_trajectory":
        lambda fnames: pd.concat([load(f)["micro_trajectory"] for f in fnames])
        .groupby(level=0).std()
    }
    name2 = name + '_convergence'
    eva2 = {
        'times_mean':
        lambda fnames: np.nanmean(
            [load(f)["convergence_time"] for f in fnames]),
        'states_mean':
        lambda fnames: np.nanmean(
            [load(f)["convergence_state"] for f in fnames]),
        'times_std':
        lambda fnames: np.std([load(f)["convergence_time"] for f in fnames]),
        'states_std':
        lambda fnames: np.std([load(f)["convergence_state"] for f in fnames])
    }
    name3 = name + '_convergence_times'
    cf3 = {
        'times':
        lambda fnames: pd.DataFrame(data=[
            load(f)["convergence_time"] for f in fnames
        ]).sortlevel(level=0),
        'states':
        lambda fnames: pd.DataFrame(data=[
            load(f)["convergence_state"] for f in fnames
        ]).sortlevel(level=0)
    }
    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle

    if not Path(save_path_raw).exists():
        os.makedirs(save_path_raw, exist_ok=True)
    try:
        run_func_output = pd.read_pickle(save_path_raw + 'rfof.pkl')
    except:
        params = list(param_combs[0])
        params[-1] = True
        run_func_output = RUN_FUNC(*params)[1]
        with open(save_path_raw + 'rfof.pkl', 'wb') as dmp:
            pd.to_pickle(run_func_output, dmp)

    # define computation handle

    sample_size = 100 if not test else 5

    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=sample_size,
                                         parameter_combinations=param_combs,
                                         path_raw=save_path_raw)

    # define post processing functions

    def mean(eps, phi, ffh, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'eps={} & phi={} & ffh={} & test={}'.format(
            eps, phi, ffh, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat_0", where=query)

        return 1, trj.groupby(level='tstep').mean()

    def std(eps, phi, ffh, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'eps={} & phi={} & ffh={} & test={}'.format(
            eps, phi, ffh, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat_0", where=query)

        return 1, trj.groupby(level='tstep').std()

    def collect(eps, phi, ffh, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = 'eps={} & phi={} & ffh={} & test={}'.format(
            eps, phi, ffh, test)

        with SafeHDFStore(compute_handle.path_raw) as store:
            data = store.select("dat_1", where=query)
        
        # drop index levels that arent compatible with shape of
        # saved run func output
        data.index = data.index.droplevel(['eps', 'phi', 'ffh', 'test',
                                           'sample'])

        # add dummy run func output, since pymofa is unable to handle it
        # otherwise
        return 1, [run_func_output[0], data]

    eva_1_handle = experiment_handling(run_func=mean,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=param_combs,
                                       path_raw=save_path_res + '/mean_trj.h5')
    eva_2_handle = experiment_handling(run_func=std,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=param_combs,
                                       path_raw=save_path_res + '/std_trj.h5')
    eva_3_handle = experiment_handling(run_func=collect,
                                       runfunc_output=run_func_output,
                                       sample_size=1,
                                       parameter_combinations=param_combs,
                                       path_raw=save_path_res + '/collected.h5')

    # cluster mode: computation and post processing

    if mode == 0:
        print('cluster mode')
        sys.stdout.flush()

        print('computing')
        compute_handle.compute()

        print('post processing')
        eva_1_handle.compute()
        eva_2_handle.compute()
        eva_3_handle.compute()

        return 1

    # local mode: plotting only

    if mode == 1:
        print('plot mode')
        sys.stdout.flush()

        plot_amsterdam(save_path_res, name1, cues=cue_list)
        plot_trajectories(save_path_res, name1, None, None)

        return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
