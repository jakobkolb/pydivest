"""
Scan the economic variables of the system especially
b_R: the resource cost,
b_d: the total factor productivity in the dirty sector,
xi: the elasticity of knowledge in the clean sector.
to see, which of the two sectors dominates.
"""

import getpass
import itertools as it
import os
import sys
from pathlib import Path
import PyDSTool as pdt
import sympy as sp
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import pandas as pd
from pymofa.experiment_handling import experiment_handling, even_time_series_spacing

from pydivest.macro_model.integrate_equations_aggregate import IntegrateEquationsAggregate

from parameters import ExperimentDefaults

def RUN_FUNC(b_d, test):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the 
    initial values, parameters and convergence state and time 
    for each run.

    Parameters:
    -----------
    b_d: float
        total factor productivity in the dirty sector
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    """

    # Parameters:

    input_params = ExperimentDefaults.input_params

    input_params['b_d'] = b_d
    input_params['xi'] = 0.05
    input_params['test'] = test

    # investment_decisions:
    nopinions = [50, 50]

    # network:
    N = sum(nopinions)
    k = 10

    # building initial conditions
    p = float(k) / N
    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=N)

    clean_investment = np.ones(N) * 50. / float(N)
    dirty_investment = np.ones(N) * 50. / float(N)

    init_conditions = (adjacency_matrix, investment_decisions,
                       clean_investment, dirty_investment)

    m = IntegrateEquationsAggregate(*init_conditions, **input_params)

    DSargs = pdt.args(name='aggregated_approximation')
    v_e, v_pi, v_tau = sp.symbols('v_e v_pi v_tau')
    v_K_cc, v_K_cd, v_K_dc, v_K_dd = sp.symbols('K_cc K_cd K_dc K_dd')
    v_subs = {m.e: v_e, m.pi: v_pi, m.tau: v_tau,
              m.Kcc: v_K_cc, m.Kcd: v_K_cd,
              m.Kdc: v_K_dc, m.Kdd: v_K_dd}

    equations = {m.var_symbols[i]: str(m.rhs_raw[i].subs(v_subs)) for i in range(len(m.var_names))}

    equations_updated = {}
    for (symbol, value) in equations.items():
        if symbol in v_subs.keys():
            equations_updated[str(v_subs[symbol])] = value
        else:
            equations_updated[str(symbol)] = value

    params_updated = {}
    for (symbol, value) in m.list_parameters().items():
        if symbol in v_subs.keys():
            params_updated[str(v_subs[symbol])] = value
        else:
            params_updated[str(symbol)] = value

    initial_conditions = {}
    for (symbol, value) in m.list_initial_conditions().items():
        if symbol in v_subs.keys():
            initial_conditions[str(v_subs[symbol])] = value
        else:
            initial_conditions[str(symbol)] = value

    del equations_updated['G']
    del initial_conditions['G']

    params_updated['G'] = m.p_G_0

    initial_conditions['C'] = 1

    DSargs.pars = params_updated
    DSargs.varspecs = equations_updated
    DSargs.ics = initial_conditions
    DSargs.tdata = [0, 300]
    DSargs.algparams = {'init_step': 0.2}

    ode = pdt.Generator.Vode_ODEsystem(DSargs)
    traj = ode.compute('some name?')
    pts = traj.sample(dt=1.)
    ode.set(ics=pts[-1])
    PC = pdt.ContClass(ode)
    PCargs = pdt.args(name='EQ1', type='EP-C')
    PCargs.freepars = ['xi']
    PCargs.MaxNumPoints = 20000
    PCargs.MaxStepSize = 2
    PCargs.MinStepSize = 1e-10
    PCargs.StepSize = 2e-3
    PCargs.SaveEigen = True
    PCargs.LocBifPoints = 'LP'
    PCargs.StopAtPoints = 'LP'
    PC.newCurve(PCargs)
    PC['EQ1'].forward()

    res = PC['EQ1'].display(stability=True, figure='fig1', axes='somename')
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(f'Limit Point Manyfold for b_d={b_d}')
    fig.savefig(f'lp_manifold_xi_vs_C_with_bd={b_d}.png')

    exit_status = 1
    df_out = pd.DataFrame(PC['EQ1'].todict()).set_index('xi')

    return exit_status, df_out


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

    folder = 'P4'

    # make sure, testing output goes to its own folder:

    test_folder = ['', 'test_output/'][int(test)]

    SAVE_PATH_RAW = f'{tmppath}/{test_folder}{folder}/'
    SAVE_PATH_RES = f'{respath}/{test_folder}{folder}/'

    """
    create parameter combinations and index
    """

    b_ds = [round(x, 5) for x in list(np.linspace(1., 4., 41))]

    if test:
        PARAM_COMBS = list(it.product([3.], [test]))
    else:
        PARAM_COMBS = list(it.product(b_ds, [test]))

    """
    run computation and/or post processing and/or plotting
    """

    # Create dummy runfunc output to pass its shape to experiment handle

    try:
        if not Path(SAVE_PATH_RAW).exists():
            Path(SAVE_PATH_RAW).mkdir(parents=True, exist_ok=True)
        run_func_output = pd.read_pickle(SAVE_PATH_RAW + 'rfof.pkl')
    except:
        params = list(PARAM_COMBS[0])
        params[-1] = False
        run_func_output = RUN_FUNC(*params)[1]
        with open(SAVE_PATH_RAW+'rfof.pkl', 'wb') as dmp:
            pd.to_pickle(run_func_output, dmp)

    SAMPLE_SIZE = 1

    # initialize computation handle
    compute_handle = experiment_handling(run_func=RUN_FUNC,
                                         runfunc_output=run_func_output,
                                         sample_size=SAMPLE_SIZE,
                                         parameter_combinations=PARAM_COMBS,
                                         path_raw=SAVE_PATH_RAW
                                         )

    compute_handle.compute()
    return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)
