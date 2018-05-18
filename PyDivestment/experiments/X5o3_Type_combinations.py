"""
Description of the Experiment:
------------------------------

This experiment focuses on the test of combinations of
different cue orders for the Take The Best Heuristic
in the dirty clean transition.

Therefore, in a first step, resource depreciation is turned
off to reach a dirty equilibrium economy.
Then in a second step, the final state of the system in the
first step is taken as initial conditions and the resource
depreciation is turned on to start the transition.

Since it is not possible to reproduce the authentic network
structure of the adaptive voter dynamics without imitation
(which we don't want, since we preserve fixed opinion shares)
the next best we can do is using directed and random rewiring
only to get at least some clustering amongst opinions that
scales with phi.

Time scales in the experiment:
------------------------------

1) capital accumulation in the dirty sector,
    t_d = 1/(d_c*(1-kappa_c))
2) depletion of the fossil resource and
    t_G = G_0*e*d_c/(L*s*b_d**2)
3) opinion spreading in the adaptive voter model
   given one opinion dominates the other.
    t_a = tau*(1-phi)

Discussion of variable parameters (degrees of freedom):
-------------------------------------------------------

tau and phi are of no interest so far, since the adaptive
voter dynamics is disabled.
t_d is fixed by standard values for d_c=0.06
and kappa_c=0.5.

Focus in terms of timescales is set to the last two remaining
degrees of freedom:

t_G sets the value for G_0 as all other parameters
are assumed to be fixed.

the ratio alpha = b_R/e<0 determines the share of the initial
resource that can be economically harvested.
"""

try:
    import pickle as cp
except ImportError:
    import pickle as cp
import getpass
import glob
import itertools as it
import sys
import time
import types
from random import shuffle

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st

from pydivest.divestvisuals.data_visualization \
    import plot_obs_grid, plot_tau_phi, tau_phi_final
from pydivest.micro_model import divestmentcore as model
from pymofa.experiment_handling \
    import experiment_handling, even_time_series_spacing


def RUN_FUNC(nopinions, phi, alpha,
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

    (N, p, tau, p, b_d, b_r0, e, s) = \
        (sum(nopinions), 0.125, 1, 500, 1.2, 1., 100, 0.23)

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
        # t_G = G_0*e*d_c/(L*s*b_d**2)
        g_0 = t_G * p * s * b_d ** 2 / (e * d_c)

        # set b_r0 according to alpha and e:
        # alpha = (b_r0/e)**(1/2)
        b_r0 = alpha ** 2 * e

        # calculate equilibrium dirty capital
        # for full on dirty economy
        K_d0 = (s / d_c * b_d * p ** (1. / 2.) * (1 - alpha ** 2)) ** 2.

        # set t_max for run
        t_max = 50 if test else 300

        # building initial conditions

        while True:
            net = nx.erdos_renyi_graph(N, p)
            if len(list(net)) > 1:
                break
        adjacency_matrix = nx.adj_matrix(net).toarray()

        opinions = []
        for i, n in enumerate(nopinions):
            opinions.append(np.full(n, i, dtype='I'))
        opinions = [item for sublist in opinions for item in sublist]
        shuffle(opinions)

        investment_clean = np.full((N), 0.1)
        investment_dirty = np.full((N), K_d0/N)

        # input parameters

        input_params = {'adjacency': adjacency_matrix,
                        'investment_decisions': opinions,
                        'investment_clean': investment_clean,
                        'investment_dirty': investment_dirty,
                        'possible_opinions': possible_opinions,
                        'tau': tau, 'phi': phi, 'eps': eps,
                        'L': p, 'b_d': b_d, 'b_r0': b_r0, 'G_0': g_0,
                        'e': e, 'd_c': d_c, 'test': bool(test),
                        'r_depletion': transition}
    # ROUND TWO: TRANSITION
    if transition:
        # build list of initial conditions
        # phi, alpha and t_d are relevant,
        # t_a is not. Parse filename to get
        # wildcard for all relevant files.
        # replace characters before first
        # underscore with *
        path = (SAVE_PATH_INIT + '/'
                + filename.split('/')[-1].split('True')[0]
                + '*_final')
        init_files = glob.glob(path)
        input_params = np.load(
            init_files[np.random.randint(0, len(init_files))])

        opinions = input_params["investment_decisions"]
        adjacency = input_params["adjacency"]

        # make list of kinds and locations of investment_decisions
        op_kinds = list(np.unique(opinions))
        op_locs = []
        for o in op_kinds:
            op_o = []
            for i in range(len(opinions)):
                if opinions[i] == o:
                    op_o.append(i)
            op_locs.append(op_o)

        # count links in and in between groups:
        pairs = it.combinations(list(range(len(opinions))), 2)
        ingroup = 0
        intergroup = 0
        for i, j in pairs:
            if adjacency[i, j] == 1:
                if opinions[i] == opinions[j]:
                    ingroup += 1
                else:
                    intergroup += 1

        k = 0
        l = 0
        while k < phi*intergroup and l < len(opinions)**4:
            i, j = np.random.randint(len(opinions), size=2)
            if opinions[i] != opinions[j] and adjacency[i, j] == 1:
                np.random.shuffle(op_locs[op_kinds.index(opinions[i])])
                for n in op_locs[op_kinds.index(opinions[i])]:
                    if adjacency[i, n] == 0 and n != i:
                        adjacency[i, n] = 1
                        adjacency[n, i] = 1
                        adjacency[i, j] = 0
                        adjacency[j, i] = 0
                        k += 1
                        break
            l += 1

        # adapt parameters where necessary
        input_params['r_depletion'] = True

        # set t_max for run
        t_max = 300

    # initializing the model

    m = model.DivestmentCore(**input_params)

    # turn off avm since fragmentation of network is handles manually
    m.mode = 1

    # storing initial conditions and parameters

    res = {}

    res["parameters"] = \
        pd.Series({"tau": m.tau,
                   "phi": m.phi,
                   "n": m.n,
                   "L": p,
                   "L": m.P,
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
                   "initial resource stock": m.g_0})

    # run the model
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    # for equilibration runs, save final state of the model:
    if not transition:
        final_state = m.final_state
        with open(filename + '_final', 'wb') as dumpfile:
            cp.dump(final_state, dumpfile)

    # store exit status
    res["convergence"] = exit_status
    if test:
        print('test output of variables')
        print((m.tau, m.phi, exit_status,
               m.convergence_state, m.convergence_time))
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
        dfo = even_time_series_spacing(df, 101, 0., t_max)
        res["economic_trajectory"] = dfo

    end = time.clock()
    res["runtime"] = end-start

    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)

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

"""
Set different output folders for equilibrium and transition.
Make folder names global variables to be able to access initial
conditions for transition in run function.
"""
FOLDER_EQUI = 'X5o3_Types_Equilibrium'
FOLDER_TRANS = 'X5o3_Types_Transition'
if not any(transition):
    print('EQUI')
    folder = FOLDER_EQUI
elif any(transition):
    print('TRANS')
    folder = FOLDER_TRANS

"""
set path variables according to local of cluster environment
"""
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = \
        "/L/tmp/kolb/Divest_Experiments/divestdata/" \
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
"""
set path variable for initial conditions for transition runs
"""
if getpass.getuser() == "kolb":
    SAVE_PATH_INIT = \
        "/L/tmp/kolb/Divest_Experiments/divestdata/" \
        + FOLDER_EQUI + "/raw_data"
elif getpass.getuser() == "jakob":
    SAVE_PATH_INIT = \
        "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"\
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
                   [1],     # gutmensch
                   [0]]     # redneck

"""
Set different combinations of types of decision makers in
terms of cue order frequencies.
"""
opinions = [
        [10, 10, 10, 10, 10, 10, 10, 10],   # all equal
        [50, 0, 0, 0, 50, 0, 0, 0],         # shorty and green conf
        [40, 0, 0, 0, 50, 0, 10, 0],        # shorty, green conf & gutmensch
        [20, 0, 70, 0, 0, 0, 10, 0],        # shorty, short herder & gutmensch
        [40, 0, 0, 0, 40, 0, 10, 10],       # shorty, green conf, gutm & rednck
        [0, 0, 50, 0, 50, 0, 10, 10],       # short herder and green conf
        [0, 0, 40, 0, 40, 0, 10, 0]]        # short herder, green conf & gutm

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
        'opinion': 0,
        'phi': 1,
        'alpha': 2,
        'test': 3}
"""
Default values of variable parameter in this experiment
"""
opinion, phi, alpha, t_d, test =\
    [[10, 10, 10, 10, 10, 10, 10, 10]], [0.8], [0.1], [30.], [0]

NAME = 'Cue_order_testing'
INDEX = {
        0: "opinion",
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
if mode == 1:  # Production
    PARAM_COMBS = list(it.product(
        opinions, phis, alphas, t_d, [opinion_presets], eps, transition, test))

elif mode == 2:  # test
    PARAM_COMBS = list(it.product(
            opinions, phis, alphas,
            t_d, [opinion_presets], eps, transition, test))

elif mode == 3:  # messy
    test = [True]
    phis = [round(x, 2) for x in list(np.linspace(0.0, 1.0, 5))]
    PARAM_COMBS = list(it.product(
        opinions[:3], phis, alpha,
            t_d, [opinion_presets], eps, transition, test))
else:
    print(mode, ' is not a valid experiment mode.\
    valid modes are 1: production, 2: test, 3: messy')
    sys.exit()

# names and function dictionaries for post processing:


def foo(fnames):
    print(pd.concat([np.load(f)['economic_trajectory']
                     for f in fnames]).groupby(level=0).min())

NAME1 = NAME+'_trajectory'
EVA1 = {"<mean_trajectory>":
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"]
                                  for f in fnames]).groupby(level=0).mean(),
        #foo,
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
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets,
                  file_extension='.pdf')

# test run
if mode == 2:
    SAMPLE_SIZE = 100
    handle = experiment_handling(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    #handle.compute(RUN_FUNC)
    #handle.resave(EVA1, NAME1)
    #handle.resave(EVA2, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets,
                  file_extension='.pdf')

# debug and mess around mode:
if mode == 3:
    SAMPLE_SIZE = 10
    handle = experiment_handling(
            SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2, opinion_presets,
                  file_extension='.pdf')
