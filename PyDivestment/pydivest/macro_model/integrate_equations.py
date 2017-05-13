# coding: utf-8

from __future__ import print_function

from scipy.integrate import odeint
import numpy as np
import pandas as pd
import sympy as sp


class Integrate_Equations:
    def __init__(self, adjacency=None, investment_decisions=None,
                 investment_clean=None, investment_dirty=None,
                 possible_opinions=None,
                 tau=0.8, phi=.7, eps=0.05,
                 P=100., r_b=0, b_c=1., b_d=1.5, s=0.23, d_c=0.06,
                 b_r0=1., e=10, G_0=3000,
                 R_depletion=True, test=False,
                 C=1, beta=0.06, xi=1. / 8., learning=False,
                 campaign=False):

        # Social parameters

        # mean waiting time between social updates
        self.tau = float(tau)
        # rewiring probability for adaptive voter model
        self.phi = float(phi)
        # percentage of rewiring and imitation events that are noise
        self.eps = float(eps)
        # number of households (to interface with initial
        # conditions from micro model)
        self.n = float(adjacency.shape[0])
        # edges/nodes
        self.k = float(sum(sum(adjacency))) / self.n
        # investment_decisions as indices of possible_opinions
        self.investment_decisions = np.array(investment_decisions)

        # Sector parameters

        # Clean capital depreciation rate
        self.d_c = float(d_c)
        # Dirty capital depreciation rate
        self.d_d = float(self.d_c)
        # knowledge depreciation rate
        self.beta = float(self.d_c)
        # Resource harvest cost per unit (at full resource stock)
        self.b_r0 = float(b_r0)
        # percentage of income saved
        self.s = float(s)
        # solow residual for clean sector
        self.b_c = float(b_c)
        # solow residual for dirty sector
        self.b_d = float(b_d)
        # elasticity of knowledge
        self.xi = float(xi)
        # labor elasticity (equal in both sectors)
        self.pi = 1. / 2.
        # clean capital elasticity
        self.kappa_c = 1. - self.pi - self.xi
        # dirty capital elasticity
        self.kappa_d = 1. - self.pi
        # fossil->energy->output conversion efficiency (Leontief)
        self.e = float(e)
        # total labor
        self.P = float(P)
        # labor per household
        self.p = float(P) / self.n
        # total knowledge stock
        self.C = float(C)
        # unprofitable fraction of fossil reserve
        self.alpha = (b_r0 / e) ** 0.5

        # Ecosystem parameters

        # initial fossil resource stock
        self.G_0 = float(G_0)
        # total fossil resource stock
        self.G = float(G_0)
        # initial fossil resource stock per household
        self.g_0 = float(G_0) / self.n
        # toggle resource depletion
        self.R_depletion = R_depletion

        # system time
        self.t = 0

        # household investment in dirty capital
        if investment_dirty is None:
            self.investment_dirty = np.ones(self.n)
        else:
            self.investment_dirty = investment_dirty

        # household investment in clean capital
        if investment_clean is None:
            self.investment_clean = np.ones(self.n)
        else:
            self.investment_clean = investment_clean

        # system variables and their initial values
        def cl(adj, x, y):
            """
            calculate number of links between like links in x and y
            :param adj: adjacency matrix
            :param x: node vector
            :param y: node vector
            :return: number of like links
            """
            assert len(x) == len(y)

            n = len(x)
            ccc = 0

            for i in range(n):
                for j in range(n):
                    ccc += x[i] * adj[i, j] * y[j]

            return float(ccc)

        adj = adjacency
        c = self.investment_decisions
        d = - self.investment_decisions + 1

        cc = cl(adj, c, c) / 2
        cd = cl(adj, c, d)
        dd = cl(adj, d, d) / 2

        n = len(c)
        k = float(sum(sum(adj))) / 2

        nc = sum(c)
        nd = sum(d)

        self.x = float(nc - nd) / n
        self.y = float(cc - dd) / k
        self.z = float(cd) / k

        self.mucc = sum(investment_clean * c) / nc
        self.mucd = sum(investment_dirty * c) / nc
        self.mudc = sum(investment_clean * d) / nd
        self.mudd = sum(investment_dirty * d) / nd
        self.c = float(self.C) / n
        self.g = self.g_0

        self.k = float(k) / n

        # symbols for system variables
        # Define symbols for dynamic variables
        self.s_mucc, self.s_mucd, self.s_mudc, self.s_mudd = \
            sp.symbols('mu_c^c mu_c^d mu_d^c mu_d^d', positive=True, real=True)
        self.s_x, self.s_y, self.s_z = sp.symbols('x y z')
        self.s_c, self.s_g = sp.symbols('c, g')

        # Define symbols for parameters
        bc, bd, bR, e, delta, rs, xi, epsilon \
            = sp.symbols('b_c b_d b_R e delta s xi epsilon')
        phi, tau, pi, kappac, kappad, N, g0, k, p \
            = sp.symbols(' phi tau pi kappa_c kappa_d N g_0 k p')

        # Define lists of symbols and values for parameters to substitute
        # in rhs expression
        param_symbols = [bc, bd, bR, e, delta, rs, xi,
                         epsilon, phi, tau, pi, kappac, kappad, N, g0, k, p]
        param_values = [self.b_c, self.b_d, self.b_r0, self.e, self.d_c,
                        self.s, self.xi, self.eps, self.phi, self.tau, self.pi,
                        self.kappa_c, self.kappa_d,
                        self.n,
                        float(self.g_0), self.k,
                        float(self.p)]

        # Load right hand side of ode system
        rhs = np.load(__file__.rsplit('/', 1)[0] + '/res_raw.pkl')
        pcl = np.load(__file__.rsplit('/', 1)[0] + '/pcl.pkl')

        # substitute parameters into rhs and simplify once.
        self.rhs = rhs.subs({symbol: value for symbol, value
                             in zip(param_symbols, param_values)})
        self.pcl = pcl.subs({symbol: value for symbol, value
                             in zip(param_symbols, param_values)})

        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.columns = \
            ['x', 'y', 'z', 'mucc', 'mucd', 'mudc', 'mudd', 'c', 'g']
        self.m_trajectory = pd.DataFrame(columns=self.columns)

        # dictionary for final state
        self.final_state = {}

    def get_m_trajectory(self):

        return self.m_trajectory

    def dot_rhs(self, values, t):
        var_symbols = [self.s_x, self.s_y, self.s_z, self.s_mucc,
                       self.s_mucd, self.s_mudc, self.s_mudd,
                       self.s_c, self.s_g]
        if values[-1] < self.alpha * self.g_0:
            values[-1] = self.alpha * self.g_0
        # add to g such that 1 - alpha**2 * (g/G_0)**2 remains positive
        subs1 = {var: val for (var, val) in zip(var_symbols, values)}
        rval = list(self.rhs.subs(subs1).evalf())
        # print [rc, rd, Kc, Kd, Pc, Pd, R] for debugging
        # print list(self.pcl.subs(subs1).evalf())
        if not self.R_depletion:
            rval[-1] = 0
        return rval

    def run(self, t_max):
        if t_max > self.t:
            t = np.linspace(self.t, t_max, 50)
            initial_conditions = [self.x, self.y, self.z, self.mucc,
                                  self.mucd, self.mudc, self.mudd,
                                  self.c, self.g]
            trajectory = odeint(self.dot_rhs, initial_conditions, t)
            df = pd.DataFrame(trajectory, index=t, columns=self.columns)
            self.m_trajectory = pd.concat([self.m_trajectory, df])

            # update aggregated variables:
            self.x = trajectory[-1, 0]
            self.y = trajectory[-1, 1]
            self.z = trajectory[-1, 2]
            self.mucc = trajectory[-1, 3]
            self.mucd = trajectory[-1, 4]
            self.mudc = trajectory[-1, 5]
            self.mudd = trajectory[-1, 6]
            self.c = trajectory[-1, 7]
            self.g_0 = trajectory[-1, 8]
            self.C = self.c * self.n
            self.G = self.g * self.n
            self.t = t_max
        elif t_max <= self.t:
            print('upper time limit is smaller than system time', self.t)

        return 1


if __name__ == '__main__':
    """
    Perform test run and plot some output to check
    functionality
    """
    import datetime
    import pandas as pd
    import numpy as np
    import networkx as nx
    from random import shuffle
    import matplotlib.pyplot as plt

    output_location = 'test_output/' \
                      + datetime.datetime.now().strftime(
        "%d_%m_%H-%M-%Ss") + '_output'

    # investment_decisions:

    nopinions = [50, 50]
    possible_opinions = [[0], [1]]

    # Parameters:

    input_parameters = {'tau': 1, 'eps': 0.05, 'b_d': 1.2,
                        'b_c': 0.4, 'phi': 0.8, 'e': 100,
                        'G_0': 30000, 'b_r0': 0.1 ** 2 * 100,
                        'possible_opinions': possible_opinions,
                        'c': 100, 'xi': 1. / 8., 'beta': 0.06,
                        'campaign': False, 'learning': True}

    # investment_decisions
    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full((n), i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:
    N = sum(nopinions)
    p = .2

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # investment
    clean_investment = np.ones(N)
    dirty_investment = np.ones(N)

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)

    model = Integrate_Equations(*init_conditions, **input_parameters)

    model.run(t_max=2)

    trj = model.m_trajectory

    print(trj)

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    trj[model.columns[0:3]].plot(ax=ax1)

    ax2 = fig.add_subplot(222)
    trj[model.columns[3:7]].plot(ax=ax2)

    ax3 = fig.add_subplot(223)
    trj[model.columns[7]].plot(ax=ax3)

    ax4 = fig.add_subplot(224)
    trj[model.columns[8]].plot(ax=ax4)

    plt.show()
