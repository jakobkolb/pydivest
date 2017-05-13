"""
Description of the Experiment:
------------------------------

This experiment focuses on the effects of learning in the clean
sector.

To emulate a clean -> dirty transition this experiment consists of
two steps:
1) Equilibrating the model with abundant fossil resources
2) Switch on fossil resource depletion and record transition
   to a clean economy.

Unlike previous experiments, this experiment incorporates learning
in the clean sector. This means, that the system has two stable states
(even with abundant fossil resource) where one is mostly dirty with
little clean investment and low clean technology knowledge and the other
is mostly clean with large clean investment and high clean technology
knowledge.
To make the system bistable, parameters have to be chosen such, that
with low knowledge stock r_d > r_c and with high knowledge stock r_d < r_c

Additionally, the experiment has two modi:

A) With heuristic decision making and several different types of
   households

B) Without heuristic decision making, only two types of households
   and only decision (instead of cue order) imitation in the
   social learning process.


Time scales in the experiment:
------------------------------

1) capital accumulation in the dirty sector,
    t_d = 1/(d_c*(1-kappa_c))
2) depletion of the fossil resource and
    t_G = G_0*e*d_c/(P*s*b_d**2)
3) opinion spreading in the adaptive voter model
   given one opinion dominates the other.
    t_a = tau*(1-phi)


Discussion of variable parameters (degrees of freedom):
-------------------------------------------------------

1) alpha, as it indicates the proximity of the
   initial state to the depreciation of the fossil
   resource,

2) phi, as it governs the clustering amongst similar
   opinions,

3) d_c as it sets the timescale for capital accumulation
   and is therefore thought to change the qualitative
   nature of the transition.
"""

import cPickle
import getpass
import glob
import itertools
import sys
import time
import types

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st

from PyDivestment.pydivest.divestvisuals.data_visualization import (plot_obs_grid, plot_tau_phi,
                                                                    tau_phi_final)
from PyDivestment.pydivest.micro_model import divestment_core as model
from ..pymofa.experiment_handling import (experiment_handling,
                                          even_time_series_spacing)


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
    assert isinstance(test, types.IntType),\
        'test must be int, is {!r}'.format(test)
    assert alpha < 1,\
        'alpha must be 0<alpha<1. is alpha = {}'.format(alpha)

    (n, p, tau, p, b_d, b_c, b_r0, e, s) =\
        (100, 0.125, 0.8, 500, 1.2, 0.32, 1., 100, 0.23)

    # ROUND ONE: FIND EQUILIBRIUM DISTRIBUTIONS:
    if not transition:
        tau = 1
        # capital accumulation of dirty capital
        # (t_d = 1/(d_c*(1-kappa_c)) with kappa_c = 0.5 :
        d_c = 2./t_d

        # set t_g to some value approx. half of run time
        t_g = 50*t_d

        # set G_0 according to resource depletion time:
        # t_g = G_0*e*d_c/(P*s*b_d**2)
        g_0 = t_g*p*s*b_d**2/(e*d_c)

        # set b_r0 according to alpha and e:
        # alpha = (b_r0/e)**(1/2)
        b_r0 = alpha**2 * e

        # calculate equilibrium dirty capital
        # for full on dirty economy
        k_d0 = (s / d_c * b_d * p ** (1. / 2.) * (1 - alpha ** 2)) ** 2.

        # set t_max for run
        t_max = 300

        # building initial conditions

        while True:
            net = nx.erdos_renyi_graph(n, p)
            if len(list(net)) > 1:
                break
        adjacency_matrix = nx.adj_matrix(net).toarray()

        opinions = [np.random.randint(0, len(possible_opinions))
                    for x in range(n)]
        if len(possible_opinions) == 2:
            opinions = [1 for x in range(n)]
        investment_clean = np.full((n, ), 0.1)
        investment_dirty = np.full((n, ), k_d0 / n)

        # input parameters

        input_params = {'adjacency': adjacency_matrix,
                        'investment_decisions': opinions,
                        'investment_clean': investment_clean,
                        'investment_dirty': investment_dirty,
                        'possible_opinions': possible_opinions,
                        'tau': tau, 'phi': phi, 'eps': eps,
                        'P': p, 'b_d': b_d, 'b_r0': b_r0, 'G_0': g_0,
                        'e': e, 'd_c': d_c, 'test': bool(test),
                        'b_c': b_c, 'learning': True,
                        'r_depletion': transition}
    # ROUND TWO: TRANSITION
    else:
        # build list of initial conditions
        # phi, alpha and t_d are relevant,
        # t_a is not. Parse filename to get
        # wildcard for all relevant files.
        # replace characters before first
        # underscore with *
        path = (SAVE_PATH_INIT
                + "/*_"
                + filename.split('/')[-1].split('_', 1)[-1].split('True')[0]
                + '*_final')
        init_files = glob.glob(path)
        input_params = np.load(
            init_files[np.random.randint(0, len(init_files))])

        # adapt parameters where necessary

        # set tau according to t_a and phi
        input_params['r_depletion'] = True
        input_params['learning'] = True

        # set t_max for run
        t_max = 300

    # initializing the model

    m = model.Divestment_Core(**input_params)

    if transition and t_a > 10.:
        m.mode = 1

    # storing initial conditions and parameters

    res = {"parameters": pd.Series({"tau": m.tau,
                                    "phi": m.phi,
                                    "n": m.n,
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
                                    "rho": m.rho,
                                    "resource efficiency": m.e,
                                    "epsilon": m.eps,
                                    "initial resource stock": m.g_0})}

    # run the model
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    # for equilibration runs, save final state of the model:
    if not transition:
        final_state = m.final_state
        with open(filename + '_final', 'wb') as dumpfile:
            cPickle.dump(final_state, dumpfile)

    # store exit status
    res["convergence"] = exit_status
    if test:
        print 'test output of variables'
        print (m.tau, m.phi, exit_status,
               m.convergence_state, m.convergence_time)
    # store data in case of successful run

    if exit_status in [0, 1]:
        res["convergence_data"] = \
                pd.DataFrame({"Investment decisions": m.investment_decisions,
                              "Investment clean": m.investment_clean,
                              "Investment dirty": m.investment_dirty})
        res["convergence_state"] = m.convergence_state
        res["convergence_time"] = m.convergence_time

        # interpolate e_trajectory to get evenly spaced time series.
        trajectory = m.e_trajectory
        headers = trajectory.pop(0)

        df = pd.DataFrame(trajectory, columns=headers)
        df = df.set_index('time')
        dfo = even_time_series_spacing(df, 201, 0., t_max)
        res["economic_trajectory"] = dfo

    end = time.clock()
    res["runtime"] = end-start

    # save data
    with open(filename, 'wb') as dumpfile:
        cPickle.dump(res, dumpfile)

    return exit_status

# get sub experiment and mode from command line
if len(sys.argv) > 1:
    mode = int(sys.argv[1])     # sets mode (1:production, 2:test, 3:messy)
else:
    mode = 3
if len(sys.argv) > 2:
    transition = [bool(int(sys.argv[2]))]
else:
    transition = [False]
if len(sys.argv) > 3:
    no_heuristics = bool(int(sys.argv[3]))
else:
    no_heuristics = False

"""
Set different output folders for equilibrium and transition.
Differentiate between runs with and without Heuristics.
Make folder names global variables to be able to access initial
conditions for transition in run function.
"""

if no_heuristics:
    FOLDER_EQUI = 'X5o4_Dirty_Equilibrium_No_TTB'
    FOLDER_TRANS = 'X5o4_Dirty_Clean_Transition_No_TTB'
else:
    FOLDER_EQUI = 'X5o4_Dirty_Equilibrium'
    FOLDER_TRANS = 'X5o4_Dirty_Clean_Transition'

if not any(transition):
    print 'EQUI'
    folder = FOLDER_EQUI
else:
    print 'TRANS'
    folder = FOLDER_TRANS

"""
set path variables according to local of cluster environment
"""
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = \
        "/P/tmp/kolb/Divest_Experiments/divestdata/" \
        + folder + "/raw_data"
    SAVE_PATH_RES =\
        "/home/kolb/Divest_Experiments/divestdata/"\
        + folder + "/results"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"\
        + folder + "/raw_data"
    SAVE_PATH_RES = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"\
        + folder + "/results"
else:
    SAVE_PATH_RAW = \
        "./" + folder + "/raw_data"
    SAVE_PATH_RES = \
        "./" + folder + "/results"

"""
set path variable for initial conditions for transition runs
"""
if getpass.getuser() == "kolb":
    SAVE_PATH_INIT = \
        "/P/tmp/kolb/Divest_Experiments/divestdata/" \
        + FOLDER_EQUI + "/raw_data"
elif getpass.getuser() == "jakob":
    SAVE_PATH_INIT = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"\
        + FOLDER_EQUI + "/raw_data"
else:
    SAVE_PATH_INIT = \
        "./" + FOLDER_EQUI + "/raw_data"

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
b_cs = [1, 3, 100]

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
    'b_c': 0,
        'phi': 1,
        'alpha': 2,
        'test': 3}
"""
Default values of variable parameter in this experiment
"""
t_a, phi, alpha, t_d, test = [0.1], [0.8], [0.1], [30.], [0]

NAME = 'Cue_order_testing'
INDEX = {
        0: "t_a",
        parameters['phi']: "phi",
        parameters['alpha']: "alpha"}
"""
set eps according to nose settings
"""
eps = [0.05]
SAVE_PATH_RAW += '_N/'
SAVE_PATH_RES += '_N/'
SAVE_PATH_INIT += '_N'

"""
create list of parameter combinations for
different experiment modes.
Make sure, opinion_presets are not expanded
"""
print mode
if mode == 1:  # Production
    PARAM_COMBS = list(itertools.product(
        b_cs, phis, alphas, t_d,
        [opinion_presets], eps,
        transition, test))

elif mode == 2:  # test
    PARAM_COMBS = list(itertools.product(
        b_cs, phis, alphas, t_d,
        [opinion_presets], eps,
        transition, test))

elif mode == 3:  # messy
    test = [True]
    phis = [round(x, 2) for x in list(np.linspace(0.0, 1.0, 5))[1:-1]]
    PARAM_COMBS = list(itertools.product(
        b_cs, phis, alpha, t_d,
        [opinion_presets], eps,
        transition, test))
elif mode == 4:
    print 'just experimental plotting'
else:
    print mode, ' is not a valid experiment mode.\
    valid modes are 1: production, 2: test, 3: messy'
    sys.exit()

# names and function dictionaries for post processing:

NAME1 = NAME+'_trajectory'
EVA1 = {"<mean_trajectory>":
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                  for f in fnames]).groupby(level=0).mean(),
        "<sem_trajectory>":
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                  for f in fnames]).groupby(level=0).sem(),
        "<min_trajectory>":
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                  for f in
                                  fnames]).groupby(level=0).min(),
        "<max_trajectory>":
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                  for f in
                                  fnames]).groupby(level=0).max()
        }


def foo(fnames):
    for f in fnames:
        print np.load(f)['convergence_state']
        print f

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

# full run
if mode == 1:
    SAMPLE_SIZE = 100
    handle = experiment_handling(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2, ylog=True)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets)

# test run
if mode == 2:
    SAMPLE_SIZE = 100
    handle = experiment_handling(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    # handle.compute(RUN_FUNC)
    # handle.resave(EVA1, NAME1)
    # handle.resave(EVA2, NAME2)
    # plot_tau_phi(SAVE_PATH_RES, NAME2, ylog=True)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets, t_max=400,
                  file_extension='.pdf')

# debug and mess around mode:
if mode == 3:
    SAMPLE_SIZE = 10
    handle = experiment_handling(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2, ylog=True)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets)

if mode == 4:
    #tau_phi_linear(SAVE_PATH_RES, NAME2)
    tau_phi_final(SAVE_PATH_RES, NAME1)
