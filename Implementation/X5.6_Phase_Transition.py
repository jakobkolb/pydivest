"""
This experiment investigates the phase transition in the adaptive voter
dynamics and finite size scaling for the number of households.
As order parameters for this phase transition, I chose the time t_m
at which there is a qualified majority for clean investment and the
amount of the fossil resource that is still remaining in the ground
at the time of consensus G_m.

To generate the initial conditions of the experiment, the system
converges to a stable state with abundant fossil resource.

To examine the transition, the final state of this equilibrated
ensemble is continued with resource depletion switched on.

Variable parameters are:

1) phi, the fraction of interaction events that lead to
   rewiring.

2) N, the number of households in the model

3) alpha: the smoothness of the depletion of the fossil
   resource.

Relevant timescales for this experiment are

t_G, the resource depletion time. This will be set to
t_G = 100
t_d, the dirty capital accumulation time. Due to fixed
values of kappa = 0.5 and delta_d = 0.06:
t_d = 33.3
t_c, the clean capital accumulation time is given by values
of d_c = d_d, kappa and xi=1/4 resulting in
t_c = 44.4

Just for the fun of it, I will check the pure adaptive
voter case against the case with heuristic decision making.
"""

from pymofa.experiment_handling import \
    experiment_handling, even_time_series_spacing
from micro_model import divestment_core as model
from divestvisuals.data_visualization import plot_phase_transition
import numpy as np
import scipy.stats as st
import networkx as nx
import pandas as pd
import cPickle as cp
import itertools as it
import sys
import getpass
import time
import types
import glob


def RUN_FUNC(phi, N, alpha,
             possible_opinions, eps, transition, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    phi : list of integers
        rewiring probability of the adaptive voter
        dynamics. Governs the clustering in the
        network of households.
    N : int
        Number of households in the model
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
    assert isinstance(test, types.IntType), \
        'test must be int, is {!r}'.format(test)
    assert alpha < 1, \
        'alpha must be 0<alpha<1. is alpha = {}'.format(alpha)

    # CHOCES OF PARAMETERS:
    # P = 0.125  # question: SHOULDN'T THE MEAN DEGREE STAY CONSTANT?? Yes!
    p = 10. / N
    tau = 1.
    P = float(N) * 10
    b_d = 1.2
    b_c = 1.
    d_c = 0.06
    e = 100.
    s = 0.23
    t_g = 100.

    # ROUND ONE: FIND EQUILIBRIUM DISTRIBUTIONS:
    if not transition:
        # set G_0 according to resource depletion time:
        # t_g = G_0*e*d_c/(P*s*b_d**2)
        G_0 = t_g * P * s * b_d ** 2 / (e * d_c)

        # set b_r0 according to alpha and e:
        # alpha = (b_r0/e)**(1/2)
        b_R0 = alpha ** 2 * e

        # calculate equilibrium dirty capital
        # for full on dirty economy
        k_d0 = (s / d_c * b_d * P ** (1. / 2.) * (1 - alpha ** 2)) ** 2.

        # set t_max for run in units of the social
        # equilibration time (since this is the
        # process that actually has to equilibrate
        t_max = 100

        # building initial conditions

        while True:
            net = nx.erdos_renyi_graph(N, p)
            if len(list(net)) > 1:
                break
        adjacency_matrix = nx.adj_matrix(net).toarray()

        opinions = [np.random.randint(0, len(possible_opinions))
                    for x in range(N)]
        investment_clean = np.full(N, 0.1)
        investment_dirty = np.full(N, k_d0 / N)

        # input parameters

        input_params = {'adjacency': adjacency_matrix,
                        'opinions': opinions,
                        'investment_clean': investment_clean,
                        'investment_dirty': investment_dirty,
                        'possible_opinions': possible_opinions,
                        'tau': tau, 'phi': phi, 'eps': eps,
                        'P': P, 'b_d': b_d, 'b_c': b_c,
                        'b_r0': b_R0, 'G_0': G_0,
                        'e': e, 'd_c': d_c, 'test': bool(test),
                        'R_depletion': transition, 'learning': True}

    # ROUND TWO: TRANSITION
    elif transition:
        print 'transition'
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
        init_file = init_files[np.random.randint(0, len(init_files))]
        input_params = np.load(init_file)['final_state']

        # adapt parameters where necessary

        # turn on resource depletion
        input_params['R_depletion'] = True

        # set t_max for run
        t_max = 200

    # initializing the model

    m = model.Divestment_Core(**input_params)


    # storing initial conditions and parameters

    res = {"parameters": pd.Series({"tau": m.tau,
                                    "phi": m.phi,
                                    "n": m.n,
                                    "P": P,
                                    "P": p,
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
                                    "initial resource stock": m.G_0})}

    # run the model
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    # for equilibration runs, save final state of the model:
    if not transition:
        res['final_state'] = m.final_state

    # for transition runs, store convergence time and
    # remaining fossil resource, as well as exit status.

    res["convergence"] = exit_status

    res["remaining_resource"] = m.G
    res["remaining_resource_fraction"] = m.convergence_state
    res["majority_time"] = m.convergence_time

    res["e_trajectory"] = \
        even_time_series_spacing(m._get_e_trj(), 201, 0., t_max)
    res["m_trajectory"] = \
        even_time_series_spacing(m._get_m_trj(), 201, 0., t_max)

    # store runtime:

    end = time.clock()
    res["runtime"] = end - start

    # save data
    if transition:
        with open(filename, 'wb') as dumpfile:
            cp.dump(res, dumpfile)
    else:
        with open(filename + '_final', 'wb') as dumpfile:
            cp.dump(res, dumpfile)

    return exit_status


# get sub experiment and mode from command line
if len(sys.argv) > 1:
    mode = int(sys.argv[1])  # sets mode (1:production, 2:test, 3:messy)
else:
    mode = 3
if len(sys.argv) > 2:
    transition = [bool(int(sys.argv[2]))]
else:
    transition = [False]
if len(sys.argv) > 3:
    no_heuristics = bool(int(sys.argv[3]))
else:
    no_heuristics = True

"""
Set different output folders for equilibrium and transition.
Differentiate between runs with and without Heuristics.
Make folder names global variables to be able to access initial
conditions for transition in run function.
"""

if no_heuristics:
    FOLDER_EQUI = 'X5o6_Phase_Transition_No_TTB'
    FOLDER_TRANS = 'X5o6_Phase_Transition_No_TTB'
else:
    FOLDER_EQUI = 'X5o6_Phase_Transition'
    FOLDER_TRANS = 'X5o6_Phase_Transition'

if not any(transition):
    print 'EQUI'
    folder = FOLDER_EQUI
elif any(transition):
    print 'TRANS'
    folder = FOLDER_TRANS

"""
set path variables according to local of cluster environment
"""
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = \
        "/p/tmp/kolb/Divest_Experiments/divestdata/" \
        + folder + "/raw_data"
    SAVE_PATH_RES = \
        "/home/kolb/Divest_Experiments/divestdata/" \
        + folder + "/results"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/" \
        + folder + "/raw_data"
    SAVE_PATH_RES = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/" \
        + folder + "/results"
"""
set path variable for initial conditions for transition runs
"""
if getpass.getuser() == "kolb":
    SAVE_PATH_INIT = \
        "/p/tmp/kolb/Divest_Experiments/divestdata/" \
        + FOLDER_EQUI + "/raw_data"
elif getpass.getuser() == "jakob":
    SAVE_PATH_INIT = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/" \
        + FOLDER_EQUI + "/raw_data"

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
                   [1],  # gutmensch
                   [0]]  # redneck
if no_heuristics:
    opinion_presets = [[1], [0]]

"""
set array of fractions of rewiring events phi
"""
phis = [round(x, 2) for x in list(np.linspace(0.0, 1.0, 21))[:-1]]

"""
set array of numbers of households N
"""
Ns = [10, 100, 300, 600]
"""
set different values for alpha, the fraction of the fossil resource
that must remain in the ground for economic reasons.
This fraction also qualitatively shapes the nature of the transition.
Small alpha results in abrupt transition, large alpha results in
smother transition.
"""
alphas = [0.01, 0.1, 0.5]

"""
dictionary of the variable parameters in this experiment together with their
position in the index of the dictionary of results
"""
parameters = {
    'phi': 0,
    'N': 1,
    'alpha': 2,
    'test': 3}
"""
Default values of variable parameter in this experiment
"""
t_a, phi, alpha, test, eps = [1.], [0.8], [0.1], [0], [0.05]

NAME = 'Cue_order_testing'
INDEX = {
    parameters['phi']: "phi",
    parameters['N']: "N",
    parameters['alpha']: "alpha"}

"""
create list of parameter combinations for
different experiment modes.
Make sure, opinion_presets are not expanded
"""
if mode == 1:  # Production
    PARAM_COMBS = list(it.product(
        phis, Ns, alphas,
        [opinion_presets], eps,
        transition, test))

elif mode == 2:  # test
    PARAM_COMBS = list(it.product(
        phis, Ns, alphas,
        [opinion_presets], eps,
        transition, test))

elif mode == 3:  # messy
    test = [True]
    phis = [round(x, 2) for x in [0.1, 0.9]]
    alphas = [0.01]
    Ns = [10]
    PARAM_COMBS = list(it.product(
        phis, Ns, alphas,
        [opinion_presets], eps,
        transition, test))
else:
    print mode, ' is not a valid experiment mode.\
    valid modes are 1: production, 2: test, 3: messy'
    sys.exit()


# names and function dictionaries for post processing:

def foo(fnames):
    print fnames
    for f in fnames:
        print np.load(f)['remaining_resource_fraction']
    return 1.


NAME1 = NAME + "_e_trajectory"
EVA1 = {"mean_trajectory":
            lambda fnames: pd.concat([np.load(f)["e_trajectory"]
                                      for f in fnames]).groupby(
                level=0).mean(),
        "sem_trajectory":
            lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                      for f in fnames]).groupby(level=0).sem(),
        "min_trajectory":
            lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                      for f in
                                      fnames]).groupby(level=0).min(),
        "max_trajectory":
            lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                      for f in
                                      fnames]).groupby(level=0).max()
        }

NAME2 = NAME + '_convergence'
EVA2 = {"mean_remaining_resource_fraction":
            lambda fnames: np.nanmean(
                [np.load(f)["remaining_resource_fraction"]
                 for f in fnames]),
        "sem_remaining_resource_fraction":
            lambda fnames: st.sem(
                [np.load(f)["remaining_resource_fraction"]
                 for f in fnames]),
        "mean_remaining_resource":
            lambda fnames: np.nanmean(
                [np.load(f)["remaining_resource"]
                 for f in fnames]),
        "mean_majority_time":
            lambda fnames: np.nanmean([np.load(f)["majority_time"]
                                       for f in fnames]),
        "min_majority_time":
            lambda fnames: np.nanmin([np.load(f)["majority_time"]
                                      for f in fnames]),
        "max_majority_time":
            lambda fnames: np.max([np.load(f)["majority_time"]
                                   for f in fnames]),
        "nanmax_majority_time":
            lambda fnames: np.nanmax([np.load(f)["majority_time"]
                                      for f in fnames]),
        "sem_majority_time":
            lambda fnames: st.sem([np.load(f)["majority_time"]
                                   for f in fnames]),
        "runtime":
            lambda fnames: st.sem([np.load(f)["runtime"]
                                   for f in fnames]),
        }
EVA_TEST = {"<some_test_case>": foo}

# full run
if mode == 1:
    SAMPLE_SIZE = 100
    if not transition:
        SAMPLE_SIZE = 50
    handle = experiment_handling(
        SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA2, NAME2)
    if transition[0]:
        plot_phase_transition(SAVE_PATH_RES, NAME2)

# test run
if mode == 2:
    SAMPLE_SIZE = 10
    if not transition:
        SAMPLE_SIZE = 5
    handle = experiment_handling(
        SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    if transition[0]:
        plot_phase_transition(SAVE_PATH_RES, NAME2)

# debug and mess around mode:
if mode == 3:
    SAMPLE_SIZE = 3
    handle = experiment_handling(
        SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    # handle.compute(RUN_FUNC)
    if transition[0]:
        # handle.resave(EVA2, NAME2)
        plot_phase_transition(SAVE_PATH_RES, NAME2)
