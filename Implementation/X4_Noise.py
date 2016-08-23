
from pymofa.experiment_handling import experiment_handle, even_time_series_spacing
from divestcore import divestment_core as model
from divestvisuals.data_visualization import plot_obs_grid, plot_tau_phi

from scipy import interpolate as ip

import numpy as np
import scipy.stats as st
import networkx as nx
import pandas as pd
import cPickle as cp
import itertools as it
import sys
import getpass
import time



def RUN_FUNC(tau, phi, eps, N, p, P, b_d, b_R, e, d_c, test, filename):
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
    eps : float \in [0,1]
        fraction of rewiring and imitation events that 
        are random.
    N : int
        the number of household agents
    p : float \in [0,1]
        connection probability for Erdos
        Renyi random graph
    P : int
        the initial population
    b_d : float
        Solov residual for the dirty sector
        should be signifficantly biger than
        for the clean sector (b_c = 1.)
        to ensure higher productivity of the
        fossil sector in the beginning
    b_R : float
        model parameter:
        pre factor of resource extraction cost
    e   : float
        resource efficiency (output/resource)
    d_c : float \in [0,1)
        capital depreciation rate (is the same in both
        sectors so far)
    test: int \in [0,1]
        wheter this is a test run, e.g.
        can be executed with lower runtime


    """
    t_max = 300 if test == 0 else 50

    #building initial conditions

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=N)
    
    #initializing the model

    m = model.divestment_core(adjacency_matrix, investment_decisions, P, tau, phi)
    m.e = e
    m.eps = eps
    m.b_R0 = b_R
    m.d_c = d_c
    m.b_d = b_d

    #storing initial conditions and parameters

    res = {}
    res["initials"] = \
            pd.DataFrame({"Investment decisions": investment_decisions,
                            "Investment clean": m.investment_clean,
                            "Investment dirty": m.investment_dirty})

    res["parameters"] = \
            pd.Series({ "tau": m.tau,
                        "phi": m.phi,
                        "N": m.N,
                        "p": p,
                        "P": m.P,
                        "birth rate": m.net_birth_rate,
                        "savings rate": m.s,
                        "clean capital depreciation rate":m.d_c,
                        "dirty capital depreciation rate":m.d_d,
                        "resource extraction efficiency":m.b_R0,
                        "Solov residual clean":m.b_c,
                        "Solov residual dirty":m.b_d,
                        "pi":m.pi,
                        "kappa":m.kappa_c,
                        "rho dirty":m.rho,
                        "e":m.e,
                        "epsilon":m.eps,
                        "initial resource stock":m.G_0})

    #run the model
    
    start = time.clock()
    exit_status = m.run(t_max=t_max)

    #store exit status
    print exit_status, m.convergence_state

    res["consensus"] = exit_status

    #store data in case of successful run

    if exit_status in [0,1]:
        res["consensus_data"] = \
                pd.DataFrame({"Investment decisions": m.investment_decision,
                            "Investment clean": m.investment_clean,
                            "Investment dirty": m.investment_dirty})
        res["consensus_state"] = m.consensus_state
        res["consensus_time"] = m.consensus_time

        # interpolate trajectory to get evenly spaced time series.
        trajectory = m.trajectory
        headers = trajectory.pop(0)

        df = pd.DataFrame(trajectory, columns=headers)
        df = df.set_index('time')
        dfo = even_time_series_spacing(df, 101, 0., t_max)
        res["economic_trajectory"] = dfo

    end = time.clock()
    res["runtime"] = end-start

    #save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
    except IOError:
        print "writing results failed for " + filename
    
    return exit_status

#get sub experiment and mode from command line
if len(sys.argv)>=1:
    input_int = int(sys.argv[1])
else:
    input_int = -1
if len(sys.argv)>=2:
    mode = int(sys.argv[2])
else:
    mode = None
if len(sys.argv)>=3:
    noise = bool(sys.argv[3])
else:
    noise = True

experiments = ['b_d', 'b_R', 'e', 'p', 'test']
sub_experiment = experiments[input_int]
folder = 'Noise' if noise else 'NoNoise'

#check if cluster or local
if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Divest_Experiments/divestdata/"+folder+"/raw_data" + '_' + sub_experiment + '/'
    SAVE_PATH_RES = "/home/kolb/Divest_Experiments/divestdata/"+folder+"/results" + '_' + sub_experiment + '/'
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"+folder+"/raw_data" + '_' + sub_experiment + '/'
    SAVE_PATH_RES = "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/"+folder+"/results" + '_' + sub_experiment + '/'

taus = [round(x,5) for x in list(np.linspace(0.0,1.0,11))[1:-1]]
phis = [round(x,5) for x in list(np.linspace(0.0,1.0,11))[1:-1]]

b_ds = [round(x,5) for x in list(1 + np.linspace( 0.0, 0.3, 9))]
b_Rs = [round(x,5) for x in list(10**np.linspace(-2.0, 2.0, 5))]
es   = [round(x,5) for x in list(4.**np.linspace( 0.0, 3.0, 4))]
ps  =  [round(x,5) for x in list(    np.linspace( 0.0, 0.3, 7))]
tests =[1]


parameters = {'tau':0, 'phi':1, 'eps':2, 'N':3, 'p':4, 'P':5, 'b_d':6, 'b_R':7, 'e':8, 'd_c':9, 'test':10}
tau, phi, eps, N, p, P, b_d, b_R, e, d_c , test = [.8], [.8], [0.0], [100], [0.125], [1000], [1.2], [1.], [100], [0.06], [0]
if noise:
    eps = [0.05]

NAME = 'tau_vs_phi_' + sub_experiment + '_sensitivity'
INDEX = {0: "tau", 1: "phi", parameters[sub_experiment]: sub_experiment}

if sub_experiment == 'b_d':
    PARAM_COMBS = list(it.product(taus,\
        phis, eps, N, p, P, b_ds, b_R, e, d_c, test))

elif sub_experiment == 'b_R':
    PARAM_COMBS = list(it.product(taus,\
        phis, eps, N, p, P, b_d, b_Rs, e, d_c, test))

elif sub_experiment == 'e':
    PARAM_COMBS = list(it.product(taus,\
        phis, eps, N, p, P, b_d, b_R, es, d_c, test))

elif sub_experiment =='p':
    PARAM_COMBS = list(it.product(taus,\
        phis, eps, N, ps, P, b_d, b_R, e, d_c, test))
    
elif sub_experiment == 'test':
    PARAM_COMBS = list(it.product(taus,\
        phis, eps, N, p, P, b_d, b_R, e, d_c, tests))


else:
    print sub_experiment, ' is not in the list of possible experiments'
    sys.exit()

#names and function dictionaries for post processing:

NAME1 = NAME+'_trajectory'
EVA1={   "<mean_trajectory>": 
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"] for f in fnames]).groupby(level=0).mean(),
        "<sem_trajectory>": 
        lambda fnames: pd.concat([np.load(f)["economic_trajectory"] for f in fnames]).groupby(level=0).sem()}

NAME2 = NAME+'_consensus'
EVA2={  "<mean_consensus_state>":
        lambda fnames: np.nanmean([np.load(f)["consensus_state"] for f in fnames]),
        "<mean_consensus_time>":
        lambda fnames: np.nanmean([np.load(f)["consensus_time"] for f in fnames]),
        "<min_consensus_time>":
        lambda fnames: np.nanmin([np.load(f)["consensus_time"] for f in fnames]),
        "<max_consensus_time>":
        lambda fnames: np.max([np.load(f)["consensus_time"] for f in fnames]),
        "<nanmax_consensus_time>":
        lambda fnames: np.nanmax([np.load(f)["consensus_time"] for f in fnames]),
        "<sem_consensus_time>":
        lambda fnames: st.sem([np.load(f)["consensus_time"] for f in fnames]),
        "<runtime>":
        lambda fnames: st.sem([np.load(f)["runtime"] for f in fnames]),
        }



# full run
if mode == 0:
    SAMPLE_SIZE = 100
    handle = experiment_handle(SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2)

# test run
if mode == 1:
    SAMPLE_SIZE = 2
    handle = experiment_handle(SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2)

# debug and mess around mode:
if mode == None:
    SAMPLE_SIZE = 2
    handle = experiment_handle(SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_RAW, SAVE_PATH_RES)
    handle.compute(RUN_FUNC)
    handle.resave(EVA1, NAME1)
    handle.resave(EVA2, NAME2)
    plot_tau_phi(SAVE_PATH_RES, NAME2)
    plot_obs_grid(SAVE_PATH_RES, NAME1, NAME2)
