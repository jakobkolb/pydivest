"""
This experiment is
A) dedicated to finding the dirty equilibrium of
the system. Thus, the fossil resource is assumed to be infinite
and the system is run with noise and adaptive voter dynamics.
B) dedicated to use this equilibrium state as initial condition for
transition runs in which the resource depletion is activated.

Switches are:

1] the mode (1: cluster 2: plotting 3: debugging and testing)

Variable parameters are:

1) alpha, as it indicates the proximity of the
   initial state to the depreciation of the fossil
   resource,

2) phi, as it governs the clustering amongst similar
   opinions,

3) d_c as it sets the timescale for capital accumulation
   and is therefore thought to change the qualitative
   nature of the transition.
"""

from pymofa.experiment_handling import (experiment_handling,
                                        even_time_series_spacing)
from micro_model import divestment_core as model
from divestvisuals.data_visualization import plot_obs_grid, plot_tau_phi
import numpy as np
import scipy.stats as st
import networkx as nx
import pandas as pd
import pickle as cp
import itertools as it
import sys
import os
import getpass
import time
import types
import glob

save_path_init = ""


def RUN_FUNC(t_a, phi, alpha,
             t_d, possible_opinions, eps, transition, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    t_a : float
        Timescale of opinion spreading given
        one opinion dominates. Timescale is
        given in units of the timescale of
        capital accumulation t_c
        such that the actual depletion time is
        t_a*t_c
    phi : list of integers
        rewiring probability of the adaptive voter
        dynamics. Governs the clustering in the
        network of households.
    alpha: float
        the ratio alpha = (b_r0/e)**(1/2)
        that sets the share of the initial
        resource G_0 that can be harvested
        economically.
    t_d : float
        the capital accumulation timescale
        t_d = 1/(d_c(1-kappa_d))
    possible_opinions : list of list of integers
        the set of cue orders that are allowed in the
        model. investment_decisions determine the individual cue
        order, that a household uses.
    eps : float
        fraction of rewiring events that are random.
    transition: bool
        switch for resource depletion
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """
    assert isinstance(test, int),\
        'test must be int, is {!r}'.format(test)
    assert alpha < 1,\
        'alpha must be 0<alpha<1. is alpha = {}'.format(alpha)

    (N, p, tau, P, b_d, b_R0, e, s) =\
        (100, 0.125, 0.8, 500, 1.2, 1., 100, 0.23)

    # ROUND ONE: FIND EQUILIBRIUM DISTRIBUTIONS:
    if not transition:
        if test:
            tau = 1.
        # capital accumulation of dirty capital
        # (t_d = 1/(d_c*(1-kappa_c)) with kappa_c = 0.5 :
        d_c = 2./t_d

        # set t_G to some value approx. half of run time
        t_G = 50*t_d

        # set G_0 according to resource depletion time:
        # t_G = G_0*e*d_c/(P*s*b_d**2)
        G_0 = t_G*P*s*b_d**2/(e*d_c)

        # set b_r0 according to alpha and e:
        # alpha = (b_r0/e)**(1/2)
        b_R0 = alpha**2 * e

        # calculate equilibrium dirty capital
        # for full on dirty economy
        K_d0 = (s/d_c*b_d*P**(1./2.)*(1-alpha**2))**2.

        # set t_max for run
        t_max = 300 if not test else 5

        # building initial conditions

        while True:
            net = nx.erdos_renyi_graph(N, p)
            if len(list(net)) > 1:
                break
        adjacency_matrix = nx.adj_matrix(net).toarray()

        opinions = [np.random.randint(0, len(possible_opinions))
                    for x in range(N)]
        investment_clean = np.full((N), 0.1)
        investment_dirty = np.full((N), K_d0/N)

        # input parameters

        input_params = {'adjacency': adjacency_matrix,
                        'opinions': opinions,
                        'investment_clean': investment_clean,
                        'investment_dirty': investment_dirty,
                        'possible_opinions': possible_opinions,
                        'tau': tau, 'phi': phi, 'eps': eps,
                        'P': P, 'b_d': b_d, 'b_r0': b_R0, 'G_0': G_0,
                        'e': e, 'd_c': d_c, 'test': bool(test),
                        'R_depletion': transition}

    # ROUND TWO: TRANSITION
    if transition:
        # build list of initial conditions
        # phi, alpha and t_d are relevant,
        # t_a is not. Parse filename to get
        # wildcard for all relevant files.
        # replace characters before first
        # underscore with *
        path = (save_path_init
                + "/"
                + filename.split('/')[-1].split('True')[0]
                + '*.pkl_final')
        init_files = glob.glob(path)
        init_file = init_files[np.random.randint(0, len(init_files))]
        input_params = np.load(init_file)['final_state']

        # adapt parameters where necessary

        # set tau according to t_a and phi
        input_params['tau'] = t_a/(1.-phi)
        input_params['R_depletion'] = True

        # set t_max for run
        t_max = 300 if not test else 5

    # initializing the model

    m = model.Divestment_Core(**input_params)

    # storing initial conditions and parameters

    res = {}

    res["parameters"] = \
        pd.Series({"tau": m.tau,
                   "phi": m.phi,
                   "n": m.n,
                   "P": p,
                   "P": m.P,
                   "birth rate": m.r_b,
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
                   "initial resource stock": m.G_0})

    # run the model
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    # for equilibration runs, save final state of the model:
    if not transition:
        res['final_state'] = m.final_state

    # store exit status
    res["convergence"] = exit_status

    # store data in case of successful run

    res["convergence_data"] = \
            pd.DataFrame({"Investment decisions": m.investment_decisions,
                          "Investment clean": m.investment_clean,
                          "Investment dirty": m.investment_dirty})
    res["convergence_state"] = m.convergence_state
    res["convergence_time"] = m.convergence_time

    # interpolate e_trajectory to get evenly spaced time series.

    res["e_trajectory"] = \
        even_time_series_spacing(m.get_e_trajectory(), 201, 0., t_max)
    res["m_trajectory"] = \
        even_time_series_spacing(m.get_m_trajectory(), 201, 0., t_max)

    end = time.clock()
    res["runtime"] = end-start

    # save data
    if transition:
        with open(filename, 'wb') as dumpfile:
            cp.dump(res, dumpfile)
    else:
        with open(filename + '_final', 'wb') as dumpfile:
            cp.dump(res, dumpfile)

    return 1

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
    # switch
    if len(argv) > 3:
        transition = bool(int(argv[3]))
    else:
        transition = False
    if len(argv) > 4:
        no_heuristics = bool(int(argv[4]))
    else:
        no_heuristics = False

    """
    Set different output folders for equilibrium and transition.
    Differentiate between runs with and without Heuristics.
    Make folder names global variables to be able to access initial
    conditions for transition in run function.
    """

    respath = os.path.dirname(os.path.realpath(__file__)) + "/divestdata"
    if getpass.getuser() == "jakob":
        tmppath = respath
    elif getpass.getuser() == "kolb":
        tmppath = "/p/tmp/kolb/Divest_Experiments"
    else:
        tmppath = "./"


    folder = 'X5o2'

    sub_experiments = ['Dirty_Equilibrium',
                      'Dirty_Clean_Transition']
    sub_experiment = sub_experiments[int(transition)]
    heuristics = ['TTB', 'No_TTB'][int(no_heuristics)]
    test_folder = 'test_output/' if test else ''

    SAVE_PATH_RAW = "{}/{}{}/{}_{}_{}".format(tmppath, test_folder, 'raw', folder,
                                         sub_experiment, heuristics)
    SAVE_PATH_RES = "{}/{}{}_{}_{}".format(respath, test_folder, folder,
                                         sub_experiment, heuristics)
    SAVE_PATH_INIT = "{}/{}{}/{}_{}_{}".format(tmppath, test_folder, 'raw', folder,
                                         sub_experiments[0], heuristics)

    # make init path global, so run function can access it.
    global save_path_init
    save_path_init = SAVE_PATH_INIT

    """
    Make different types of decision makers. Cues are
    """
    cue_names = {
            0: 'always dirty',
            1: 'always clean',
            2: 'capital rent',
            3: 'capital rent trend',
            4: 'peer pressure'}
    opinion_presets = [[2, 3],  # short term investor
                       [3, 2],  # long term investor
                       [4, 2],  # short term herder
                       [4, 3],  # trending herder
                       [4, 1],  # green conformer
                       [4, 0],  # dirty conformer
                       [1],     # gutmensch
                       [0]]     # redneck
    if no_heuristics:
        opinion_presets = [[1], [0]]

    """
    set different times for resource depletion
    in units of capital accumulation time t_d = 1/(d_c*(1-kappa_d))
    """
    t_as = [round(x, 5) for x in list(10**np.linspace(0, 1, 3))]

    """
    set array of phis to generate equilibrium conditions for
    """
    phis = [round(x, 2) for x in list(np.linspace(0.0, 1.0, 11))[:-1]]

    """
    Define set of alphas that will be tested against the sets of resource depletion
    times and cue order mixtures
    """
    alphas = [0.1, 0.08, 0.05]

    """
    dictionary of the variable parameters in this experiment together with their
    position in the index of the dictionary of results
    """
    parameters = {
            't_a': 0,
            'phi': 1,
            'alpha': 2,
            'test': 3}
    """
    Default values of variable parameter in this experiment
    """
    t_a, phi, alpha, t_d, eps = [0.1], [0.8], [0.1], [30.], [0.05]

    NAME = 'Cue_order_testing'
    INDEX = {
            parameters["t_a"]: "t_a",
            parameters['phi']: "phi",
            parameters['alpha']: "alpha"}

    """
    create list of parameter combinations according to testing mode.
    Make sure, opinion_presets are not expanded
    """
    if not test:
        PARAM_COMBS = list(it.product(
            t_as, phis, alphas, t_d,
            [opinion_presets], eps,
            [transition], [test]))
        file_extension = '.pdf'
    else:
        PARAM_COMBS = list(it.product(
            t_as[:2], phis[:2], alphas, t_d,
            [opinion_presets], eps,
            [transition], [test]))
        file_extension = '.png'

    # names and function dictionaries for post processing:

    NAME1 = NAME+'_trajectory'
    EVA1 = {"<mean_trajectory>":
            lambda fnames: pd.concat([np.load(f)["e_trajectory"]
                                      for f in fnames]).groupby(level=0).mean(),
            "<sem_trajectory>":
            lambda fnames: pd.concat([np.load(f)["e_trajectory"]
                                      for f in fnames]).groupby(level=0).sem(),
            "<min_trajectory>":
            lambda fnames: pd.concat([np.load(f)["e_trajectory"]
                                      for f in
                                      fnames]).groupby(level=0).min(),
            "<max_trajectory>":
            lambda fnames: pd.concat([np.load(f)["e_trajectory"]
                                      for f in
                                      fnames]).groupby(level=0).max()
            }

    NAME2 = NAME+'_convergence'
    EVA2 = {"<mean_convergence_state>":
            lambda fnames: np.nanmean([np.load(f)["convergence_state"]
                                       for f in fnames]),
            "<mean_convergence_time>":
            lambda fnames: np.nanmean([np.load(f)["convergence_time"]
                                       for f in fnames]),
            "<min_convergence_time>":
            lambda fnames: np.nanmin([np.load(f)["convergence_time"]
                                      for f in fnames]),
            "<max_convergence_time>":
            lambda fnames: np.max([np.load(f)["convergence_time"]
                                   for f in fnames]),
            "<nanmax_convergence_time>":
            lambda fnames: np.nanmax([np.load(f)["convergence_time"]
                                      for f in fnames]),
            "<sem_convergence_time>":
            lambda fnames: st.sem([np.load(f)["convergence_time"]
                                   for f in fnames]),
            "<runtime>":
            lambda fnames: st.sem([np.load(f)["runtime"]
                                   for f in fnames]),
            }

    # Cluster - computation and plotting
    if mode == 0:
        SAMPLE_SIZE = 100 if not test else 2
        handle = experiment_handling(
                SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
        handle.compute(RUN_FUNC)
        if transition:
            handle.resave(EVA1, NAME1)  # economic trajectories
            handle.resave(EVA2, NAME2)  # final states
            plot_tau_phi(SAVE_PATH_RES, NAME2, ylog=True)
            plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets,
                          file_extension=file_extension, test=test)

    # Local - only plotting
    elif mode == 1:
        if transition:
            plot_tau_phi(SAVE_PATH_RES, NAME2, ylog=True)
            plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets,
                          file_extension=file_extension, test=test)
    # No valid mode - exit
    else:
        print(mode, ' is not a valid experiment mode. '
                    'valid modes are 1: production, 2: local')
        sys.exit()

    return 1

if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)