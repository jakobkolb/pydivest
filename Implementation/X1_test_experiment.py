
from pymofa import experiment_handling as eh
from divestcore import divestment_core as model
from divestvisuals.data_visualization import plot_economy, plot_network
from X1_visualization import plot_tau_phi as plt_tau_phi

import numpy as np
import networkx as nx
import pandas as pd
import cPickle as cp
import itertools as it
import sys
import getpass

#determine wheter the experiment is run locally or on cluster and
#define save path accordingly

if getpass.getuser() == "kolb":
    SAVE_PATH = "/home/kolb/Divest_Experiments/divestdata/X1/"
elif getpass.getuser() == "jakob":
    SAVE_PATH = "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/X1/"

SAVE_PATH_RAW = SAVE_PATH + "raw_data/"
SAVE_PATH_RES = SAVE_PATH + "results/"

def RUN_FUNC(tau, phi, link_density, N, L, delta_r, delta_c, C_d, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the 
    initial values, parameters and consensus state and time 
    for each run.

    Parameters:
    -----------
    tau : float > 0
        the social update timescale
    phi : float \in [0,1]
        the rewiring probability for the network update
    N : int
        the number of household agents
    L : int
        the initial number of members per household
    delta_r : float
        the resource extraction efficiency for the 
        extraction of the fossil resource in units of
        production_units/resource_uptake^2 to calculate the
        cost of resource extraction:
        cost = delta_r*resource_uptake^2
    delta_c : float \in [0,1)
        capital depreciation rate (is the same in both
        sectors so far)
    C_d : float
        Solov residual for the dirty sector
        should be signifficantly biger than
        for the clean sector (C_c = 1.)
        to ensure higher productivity of the
        fossil sector in the beginning

    """
    #building initial conditions

    while True:
        net = nx.erdos_renyi_graph(N, link_density)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=N)
    
    #initializing the model

    m = model.divestment_core(adjacency_matrix, investment_decisions, L, tau, phi)
    m.delta_r_present = delta_r
    m.delta_c = delta_c
    m.C_d = C_d

    #storing initial conditions and parameters

    res = {}
    res["initials"] = pd.DataFrame({"Investment decisions": investment_decisions,
                                    "Investment clean": m.investment_clean,
                                    "Investment dirty": m.investment_dirty})

    res["parameters"] = pd.Series({"tau": m.tau,
                                    "phi": m.phi,
                                    "N": m.N,
                                    "birth rate": m.net_birth_rate,
                                    "consumption level": m.consumption_level,
                                    "clean capital depreciation rate":m.delta_c,
                                    "dirty capital depreciation rate":m.delta_d,
                                    "resource extraction efficiency":m.delta_r_present,
                                    "Solov residual clean":m.C_c,
                                    "Solov residual dirty":m.C_d,
                                    "alpha clean":m.a_c,
                                    "alpha dirty":m.a_d,
                                    "beta clean":m.b_c,
                                    "beta dirty":m.b_d,
                                    "initial resource stock":m.R_start})

    #run the model

    exit_status = m.run()

    #store exit status

    res["consensus"] = exit_status

    #store data in case of sucessful run

    if exit_status == 1:
        res["concensus_data"] = pd.DataFrame({"Investment decisions": m.investment_decision,
                                            "Investment clean": m.investment_clean,
                                            "Investment dirty": m.investment_dirty})
        res["consensus_time"] = m.t
        res["consensus_state"] = m.consensus_state
        trajectory = m.trajectory
        headers = trajectory.pop(0)
        res["economic_trajectory"] = pd.DataFrame(trajectory, columns=headers)

    #save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
        print filename
    except IOError:
        print "writing results failed for " + filename
    
    return exit_status

def compute():
    """
    Not quite sure, what this function is god for. 
    copy and pasted it from wbarfuss example experiment.
    I thing this could also be accomplished by calling 
    the eh.compute() function directly during the experiment.
    """
    eh.compute(RUN_FUNC, PARAM_COMBS, SAMPLE_SIZE, SAVE_PATH_RAW)

def resave(sample_size=None):
    """
    dictionary of lambda functions to calculate 
    the average consensus state and consensus time from all
    runs given in the list of filenames (fnames) 
    that is handled internally by resave_data.

    Parameters:
    -----------
    sample_size : int
        the number of runs computed for one 
        combination of parameters e.g. the 
        size of the ensemble for statistical 
        analysis.
    """
    EVA={   "<mean_consensus_state>": 
            lambda fnames: np.mean([np.load(f)["consensus_state"] for f in fnames]),
            "<mean_consensus_time>": 
            lambda fnames: np.mean([np.load(f)["consensus_time"] for f in fnames])}

    eh.resave_data(SAVE_PATH_RAW, PARAM_COMBS, INDEX, EVA, NAME, sample_size)


#get subexperiment from comand line
if len(sys.argv)>1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0

# Default Experiment tau vs phi for different resource extraction costs
# Only raw data generation
if sub_experiment == 0:

    taus = np.arange(0., 1, 0.02)
    phis = np.arange(1., 1, 0.1)

    N, link_density, L, delta_r, delta_c, C_d = [100], [0.3], [10], [0.01], [0.01], [0.3]

    PARAM_COMBS = list(it.product(taus,\
        phis, link_density, N, L, delta_r, delta_c, C_d))

    NAME = "experiment_testing_tau_vs_phi"
    INDEX = {0: "tau", 1: "phi"}
    SAMPLE_SIZE = 10

    compute()
    resave(SAMPLE_SIZE)
    plt_tau_phi(SAVE_PATH_RES, NAME)

# Default Experiment tau vs phi for different resource extraction costs
# Only data post processing/resaving
if sub_experiment == 1:

    taus = np.arange(0., 1, 0.02)
    phis = np.arange(0., 1, 0.1)

    N, link_density, L, delta_r, delta_c, C_d = [100], [0.3], [10], [0.01], [0.01], [0.3]

    PARAM_COMBS = list(it.product(taus,\
        phis, link_density, N, L, delta_r, delta_c, C_d))

    NAME = "experiment_testing_tau_vs_phi"
    INDEX = {0: "tau", 1: "phi"}
    SAMPLE_SIZE = 10

#   compute()
    resave(SAMPLE_SIZE)
    plt_tau_phi(SAVE_PATH_RES, NAME)

