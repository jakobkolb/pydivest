# coding: utf-8

from __future__ import print_function

from scipy.integrate import odeint
import pickle as pkl
import numpy as np
import pandas as pd
import sympy as sp
from sympy.abc import epsilon, tau, phi


class Integrate_Equations:
    def __init__(self, adjacency=None, investment_decisions=None,
                 investment_clean=None, investment_dirty=None,
                 i_tau=0.8, i_phi=.7, eps=0.05,
                 b_c=1., b_d=1.5, s=0.23, d_c=0.06,
                 b_r0=1., e=10,
                 pi=0.5, kappa_c=0.4, kappa_d=0.5, xi=1. / 8.,
                 L=100., G_0=3000, C=1,
                 R_depletion=True,
                 interaction=2, crs=True, **kwargs):

        if len(kwargs.keys()) > 0:
            print('got superfluous keyword arguments')
            print(kwargs.keys())

        # Social parameters

        # interaction either with 1) tanh(Wi-Wj) or 2) (Wi-Wj)/(Wi+Wj)
        self.interaction = interaction
        # mean waiting time between social updates
        self.tau = float(i_tau)
        # rewiring probability for adaptive voter model
        self.phi = float(i_phi)
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
        self.pi = pi
        # clean capital elasticity
        self.kappa_c = 1. - self.pi - self.xi
        # dirty capital elasticity
        self.kappa_d = 1. - self.pi
        print('pi = {}, xi = {}, kappa_c = {}, kappa_d = {}'.format(self.pi, self.xi, self.kappa_c, self.kappa_d))
        # fossil->energy->output conversion efficiency (Leontief)
        self.e = float(e)
        # total labor
        self.L = float(L)
        # labor per household
        self.l = float(L) / self.n
        # total knowledge stock
        self.C = float(C)
        # knowledge per household
        self.c = float(self.C) / self.n
        # unprofitable fraction of fossil reserve
        self.alpha = (b_r0 / e) ** 0.5

        # Ecosystem parameters

        # initial fossil resource stock
        self.G_0 = float(G_0)
        # initial resource stock per household
        self.g_0 = float(self.G_0) / self.n
        # total fossil resource stock
        self.G = float(G_0)
        # total fossil resource stock per household
        self.g = float(self.G_0) / self.n
        # toggle resource depletion
        self.R_depletion = R_depletion

        # system time
        self.t = 0

        # household investment in dirty capital
        if investment_dirty is None:
            self.investment_dirty = np.ones(int(self.n))
        else:
            self.investment_dirty = investment_dirty

        # household investment in clean capital
        if investment_clean is None:
            self.investment_clean = np.ones(int(self.n))
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
        k_n = float(sum(sum(adj))) / 2

        nc = sum(c)
        nd = sum(d)

        self.x = float(nc - nd) / n
        self.y = float(cc - dd) / k_n
        self.z = float(cd) / k_n

        self.mucc = sum(investment_clean * c) / nc
        self.mucd = sum(investment_clean * d) / nd
        self.mudc = sum(investment_dirty * c) / nc
        self.mudd = sum(investment_dirty * d) / nd
        self.c = float(self.C) / n
        self.g = self.g_0

        self.k = float(k_n) / n

        # Define variables and parameters for the adaptive voter model

        # number of nodes
        N = sp.Symbol('N', integer=True)
        # number of dirty nodes
        Nd = sp.Symbol('N_d', integer=True)
        # number of clean nodes
        Nc = sp.Symbol('N_c', integer=True)
        # number of edges
        K = sp.Symbol('K', integer=True)
        # number of clean edges
        cc = sp.Symbol('[cc]', integer=True)
        # number of dirty edges
        dd = sp.Symbol('[dd]', integer=True)
        # number of mixed edges
        cd = sp.Symbol('[cd]', integer=True)
        # average number of neighbors of clean nodes
        kc = sp.Symbol('k_c', integer=True)
        # average number of neighbors of dirty nodes
        kd = sp.Symbol('k_d', integer=True)
        # Nc - Nd
        X = sp.Symbol('X', real=True)
        # cc - dd
        Y = sp.Symbol('Y', real=True)
        # cd
        Z = sp.Symbol('Z', real=True, positive=True)
        # wealth of dirty node
        Wd = sp.Symbol('W_d')
        # wealth of clean node
        Wc = sp.Symbol('W_c')
        # imitation probabilities
        Pcd, Pdc = sp.symbols('Pcd Pdc')

        # Define variables and parameters for the economic subsystem:

        # Total labor and labor shares in sectors
        L, Lc, Ld = sp.symbols('L L_c L_d', positive=True, real=True)
        # Total capital in sectors
        Kc, Kd = sp.symbols('K_c K_d', positive=True, real=True)
        # Equilibrium wage and capital return rates in sectors
        w, rc, rd = sp.symbols('w r_c r_d', positive=True, real=True)
        # Resource usage rage, resource stock, knowledge Stock
        R, G, C = sp.symbols('R, G, C', positive=True, real=True)
        # average capital endowments of clean and dirty households
        # lower (first) index: capital type, upper (second) index: household type
        mucc, mucd, mudc, mudd = sp.symbols('mu_c^c mu_c^d mu_d^c mu_d^d', positive=True, real=True)
        # savings rate, capital depreciaten rate, and elasticities of labor, capital and knowledge
        rs, delta, pi, kappac, kappad, xi = sp.symbols('s delta pi kappa_c, kappa_d xi',
                                                       positive=True, real=True)
        # solow residuals of clean and dirty sector,
        # pre factor for resource cost, energy efficiency, initial resource stock
        bc, bd, bR, e, G0 = sp.symbols('b_c b_d b_R e G_0', positive=True, real=True)
        # substitutions for resolution on constraints from market clearing.
        Xc, Xd, XR = sp.symbols('X_c X_d X_R', positive=True, real=True)

        # Defination of relations between variables and calculation of
        # substitution of *primitive variables* by *state variables* of the system

        eqs = [
            # total number of households is fixed,
            Nd + Nc - N,
            # total number of edges is fixed,
            cc + dd + cd - K,
            # definition of state space variables
            X - Nc + Nd,
            Y - cc + dd,
            Z - cd,
            # mean degrees of clean and dirty nodes
            kc - (2 * cc + cd) / Nc,
            kd - (2 * dd + cd) / Nd
        ]
        vars1 = (Nc, Nd, cc, dd, cd, kc, kd)

        subs1 = sp.solve(eqs, vars1, dict=True)[0]

        # define expected wealth as expected income.
        subs1[Wc] = rc * mucc + rd * mudc
        subs1[Wd] = rc * mucd + rd * mudd

        if interaction == 0:
            subs1[Pcd] = (Wd - Wc).subs(subs1)
            subs1[Pdc] = (Wc - Wd).subs(subs1)
        elif interaction == 1:
            subs1[Pcd] = (1. / (1 + sp.exp(8. * (Wd - Wc) / (Wc + Wd)))).subs(subs1)
            subs1[Pdc] = (1. / (1 + sp.exp(8. * (Wc - Wd) / (Wc + Wd)))).subs(subs1)
        elif interaction == 2:
            subs1[Pcd] = ((1. / 2) * ((Wd - Wc) / (Wd + Wc) + 1)).subs(subs1)
            subs1[Pdc] = ((1. / 2) * ((Wc - Wd) / (Wd + Wc) + 1)).subs(subs1)

        # Jumps in state space i.e. Effect of events on state vector S = (X, Y, Z) - denoted r = X-X' in van Kampen

        # regular adaptive voter events
        s1 = sp.Matrix([0, 1, -1])  # clean investor rewires
        s2 = sp.Matrix([0, -1, -1])  # dirty investor rewires
        s3 = sp.Matrix([-2, -kc, -1 + (1 - 1. / kc) * ((2 * cc - cd) / Nc)])  # clean investor imitates c -> d
        s4 = sp.Matrix([2, kd, -1 + (1 - 1. / kd) * ((2 * dd - cd) / Nd)])  # dirty investor imitates d -> c

        # noise events

        s5 = sp.Matrix([-2, -(2 * cc + cd) / Nc, (2 * cc - cd) / Nc])  # c -> d
        s6 = sp.Matrix([2, (2 * dd + cd) / Nd, (2 * dd - cd) / Nd])  # d -> c
        s7 = sp.Matrix([0, -1, 1])  # c-c -> c-d
        s8 = sp.Matrix([0, 1, -1])  # c-d -> c-c
        s9 = sp.Matrix([0, 1, 1])  # d-d -> d-c
        s10 = sp.Matrix([0, -1, -1])  # d-c -> d-d

        # Probabilities per unit time for events to occur (denoted by W in van Kampen)

        p1 = 1. / tau * (1 - epsilon) * (Nc / N) * cd / (Nc * kc) * phi  # clean investor rewires
        p2 = 1. / tau * (1 - epsilon) * (Nd / N) * cd / (Nd * kd) * phi  # dirty investor rewires
        p3 = 1. / tau * (1 - epsilon) * (Nc / N) * cd / (Nc * kc) * (1 - phi) * Pcd  # clean investor imitates c -> d
        p4 = 1. / tau * (1 - epsilon) * (Nd / N) * cd / (Nd * kd) * (1 - phi) * Pdc  # dirty investor imitates d -> c
        p5 = 1. / tau * epsilon * (1. / 2) * Nc / N  # c -> d
        p6 = 1. / tau * epsilon * (1. / 2) * Nd / N  # d -> c
        p7 = 1. / tau * epsilon * Nc / N * (2 * cc) / (2 * cc + cd) * Nd / N  # c-c -> c-d
        p8 = 1. / tau * epsilon * Nc / N * cd / (2 * cc + cd) * Nc / N  # c-d -> c-c
        p9 = 1. / tau * epsilon * Nd / N * (2 * dd) / (2 * dd + cd) * Nc / N  # d-d -> d-c
        p10 = 1. / tau * epsilon * Nd / N * cd / (2 * dd + cd) * Nd / N  # d-c -> d-d

        # Create S and r matrices to write down rhs markov jump process for pair based proxy:

        r = sp.Matrix(s1)
        for i, si in enumerate([s2, s3, s4, s5, s6, s7, s8, s9, s10]):
            r = r.col_insert(i + 1, si)

        W = sp.Matrix([p1])
        for j, pj in enumerate([sp.Matrix([p]) for p in [p2, p3, p4, p5, p6, p7, p8, p9, p10]]):
            W = W.col_insert(j + 1, pj)

        # rhs of the pair based proxy is given by the first jump moment. This is formally given by
        #
        # $\int r W(S,r) dr$
        #
        # which in our case is equal to
        #
        # $\sum_i r_i W_{i, j}(S)$
        #
        # To calculate this, we first write the jumps and transition matrix in terms of
        # X, Y, Z and then substitute with rescalled variables and eliminate N.

        r = r.subs(subs1)
        W = W.subs(subs1)

        x, y, z = sp.symbols('x y z')
        c, g, l, g0, k = sp.symbols('c, g, l, g_0, k', positive=True, real=True)
        subs4 = {Kc: (N / 2. * (1 + x) * mucc + N / 2. * (1 - x) * mudc),
                 Kd: (N / 2. * (1 + x) * mucd + N / 2. * (1 - x) * mudd),
                 C: N * c,
                 L: N * l,
                 G: N * g,
                 G0: N * g0,
                 X: N * x,
                 Y: N * k * y,
                 Z: N * k * z,
                 K: N * k}

        variables = [x, y, z, c, g, l, g0, k]

        r = r.subs(subs4)
        W = W.subs(subs4)
        for i in range(len(W)):
            W[i] = W[i].collect(N)
        for i in range(len(r)):
            r[i] = r[i].collect(N)
            # flaming hack to circumvent sympy's inability to collect with core.add.Add.
            # eyeballing the expressions it is obvious that this is justified.
            if isinstance(r[i], sp.add.Add):
                r[i] = r[i].subs({N: 1})

        # **Next, we treat the equations describing economic production and capital accumulation**
        #
        # Substitutute solutions to algebraic constraints of economic system
        # (market clearing for labor and expressions for capital rent and resource flow)

        subs2 = {w: pi * L ** (pi - 1.) * (Xc + Xd * XR) ** (1. - pi),
                 rc: kappac / Kc * Xc * L ** pi * (Xc + Xd * XR) ** (-pi),
                 rd: kappad / Kd * Xd * XR * L ** pi * (Xc + Xd * XR) ** (-pi),
                 R: bd / e * Kd ** kappad * L ** pi * (Xd * XR / (Xc + Xd * XR)) ** pi,
                 Lc: L * Xc / (Xc + Xd * XR),
                 Ld: L * Xd * XR / (Xc + Xd * XR),
                 Wc: rc * mucc + rd * mudc,
                 Wd: rc * mucd + rd * mudd}

        subs3 = {Xc: (bc * Kc ** kappac * C ** xi) ** (1. / (1. - pi)),
                 Xd: (bd * Kd ** kappad) ** (1. / (1. - pi)),
                 XR: (1. - bR / e * (G0 / G) ** 2) ** (1. / (1. - pi))}

        # Substitutions to ensure constant returns to scale:

        subs5 = {kappac: 1. - pi - xi,
                 kappad: 1. - pi}

        # Write down dynamic equations for the economic subsystem in
        # terms of means of clean and dirty capital stocks for clean and dirty households

        rhsECO = sp.Matrix([(rs * rc - delta) * mucc + rs * rd * mudc + rs * w * L / N,
                            -delta * mudc,
                            -delta * mucd,
                            rs * rc * mucd + (rs * rd - delta) * mudd + rs * w * L / N,
                            bc * Lc ** pi * (Nc * mucc + Nd * mucd) ** kappac * C ** xi - delta * C,
                            -R])

        # Write down changes in means of capital stocks through agents'
        # switching of opinions and add them to the capital accumulation terms

        dtNcd = 1. / tau * Nc * (
        Nc / N * cd / (2 * cc + cd) * (1 - phi) * (1 - epsilon) * Pcd + epsilon * 1. / 2 * Nc / N)
        dtNdc = 1. / tau * Nd * (
        Nd / N * cd / (2 * dd + cd) * (1 - phi) * (1 - epsilon) * Pdc + epsilon * 1. / 2 * Nd / N)

        rhsECO_switch = sp.Matrix([(mucd - mucc) * dtNdc / Nc,
                                   (mudd - mudc) * dtNdc / Nc,
                                   (mucc - mucd) * dtNcd / Nd,
                                   (mudc - mudd) * dtNcd / Nd,
                                   0,
                                   0])

        try:
            rhs = np.load('mean_rhs.pkl')
            print('loading rhs successful')
        except:
            # After eliminating N, we can write down the first jump moment:

            print('simplify PBP rhs')
            rhsPBP = sp.Matrix(r * sp.Transpose(W))
            rhsPBP = sp.Matrix(sp.simplify(rhsPBP))

            print('simplifying eco switch terms')
            rhsECO_switch = sp.simplify(rhsECO_switch.subs(subs1))

            rhsECO = rhsECO + rhsECO_switch

            # Next, we have to write the economic system in terms of X, Y, Z and then
            # in terms of rescaled variables and check the dependency on the system size N:
            # - 1) substitute primitive variables for dependent variables (subs1)
            # - 2) substitute dependent variables for system variables (subs4)

            rhsECO = rhsECO.subs(subs1).subs(subs2).subs(subs3).subs(subs4).subs(subs5)

            # In the PBP rhs substitute economic variables for their proper expressions
            # ($r_c$, $r_d$ ect.) and then again substitute lingering 'primitive' variables with rescaled ones

            rhsPBP = rhsPBP.subs(subs2).subs(subs3).subs(subs5)
            rhsPBP = rhsPBP.subs(subs1).subs(subs4).subs({N: 1})

            rhsECO = rhsECO.subs({N: 1})
            rhs = sp.Matrix([rhsPBP, rhsECO]).subs(subs1)
            with open('mean_rhs.pkl', 'wb') as outfile:
                pkl.dump(rhs, outfile)
                print('saving rhs successful')

        # Define lists of symbols and values for parameters to substitute
        # in rhs expression
        param_symbols = [bc, bd, bR, e, delta, rs, xi,
                         epsilon, phi, tau, pi, kappac, kappad, N, g0, k, l]
        param_values = [self.b_c, self.b_d, self.b_r0, self.e, self.d_c,
                        self.s, self.xi, self.eps, self.phi, self.tau, self.pi,
                        self.kappa_c, self.kappa_d, self.n,
                        float(self.g_0), self.k, float(self.l)]

        self.subs_params = {symbol: value for symbol, value
                            in zip(param_symbols, param_values)}
        self.rhs = rhs.subs(self.subs_params)

        self.independent_vars = {'mu_c^c': mucc, 'mu_c^d': mudc,
                                 'mu_d^c': mudc, 'mu_d^d': mudd,
                                 'x': x, 'y': y, 'z': z,
                                 'R': R, 'c': c, 'g': g}
        self.dependent_vars = {'w': w, 'rc': rc, 'rd': rd, 'R': R, 'Lc': Lc, 'Ld': Ld, 'L': L, 'rs': rs, 'l': l}

        for key in self.dependent_vars.keys():
            self.dependent_vars[key] = self.dependent_vars[key].subs(subs2).subs(subs3)\
                .subs(subs1).subs(subs4).subs({N: 1}).subs(self.subs_params)

        self.var_symbols = [x, y, z, mucc, mudc, mucd, mudd, c, g]
        self.var_names = ['x', 'y', 'z', 'mu_c^c', 'mu_d^c', 'mu_c^d', 'mu_d^d', 'c', 'g']

        self.m_trajectory = pd.DataFrame(columns=self.var_names)

        # dictionary for final state
        self.final_state = {}

    def dot_rhs(self, values, t):
        print(t)
        # add to g such that 1 - alpha**2 * (g/G_0)**2 remains positive
        if values[-1] < self.alpha * self.g_0:
            values[-1] = self.alpha * self.g_0

        # evaluate expression by substituting symbol values
        subs1 = {var: val for (var, val) in zip(self.var_symbols, values)}
        rval = list(self.rhs.subs(subs1).evalf())

        if not self.R_depletion:
            rval[-1] = 0
        return rval

    def run(self, t_max):
        if t_max > self.t:
            print('integrating equations from t={} to t={}'.format(self.t, t_max))
            t = np.linspace(self.t, t_max, 50)
            initial_conditions = [self.x, self.y, self.z, self.mucc,
                                  self.mudc, self.mucd, self.mudd,
                                  self.c, self.g]

            trajectory = odeint(self.dot_rhs, initial_conditions, t)
            df = pd.DataFrame(trajectory, index=t, columns=self.var_names)
            self.m_trajectory = pd.concat([self.m_trajectory, df])

            # update aggregated variables:
            (self.x, self.y, self.z,
             self.mucc, self.mudc, self.mucd, self.mudd,
             self.c, self.g) = trajectory[-1]

            self.C = self.c * self.n
            self.G = self.g * self.n
            self.t = t_max
        elif t_max <= self.t:
            print('upper time limit is smaller than system time', self.t)

        return 1

    def get_mean_trajectory(self):

        return self.m_trajectory

    def get_unified_trajectory(self):
        """
        Calculates unified trajectory in per capita variables
        
        Returns
        -------
        Dataframe of unified per capita variables
        """

        L = self.dependent_vars['L']
        l = self.dependent_vars['l']

        columns = ['k_c', 'k_d', 'l_c', 'l_d', 'g', 'c', 'r', 'n_c', 'i_c', 'r_c', 'r_d', 'w']
        var_expressions = [(self.independent_vars['mu_c^c'] + self.independent_vars['mu_d^c']) / l,
                           (self.independent_vars['mu_c^d'] + self.independent_vars['mu_d^d']) / l,
                           self.dependent_vars['Lc'] / L,
                           self.dependent_vars['Ld'] / L,
                           self.independent_vars['g'] / l,
                           self.independent_vars['c'] / l,
                           self.dependent_vars['R'] / L,
                           (self.independent_vars['x'] + 1.) / 2.,
                           (self.dependent_vars['rc'] * self.independent_vars['mu_c^c'] + self.dependent_vars['rd'] * self.independent_vars['mu_d^c'])
                           / (self.dependent_vars['rc'] * (self.independent_vars['mu_c^c'] + self.independent_vars['mu_c^d'])
                              + self.dependent_vars['rd'] * (self.independent_vars['mu_d^c'] + self.independent_vars['mu_d^d'])),
                           self.dependent_vars['rc'],
                           self.dependent_vars['rd'],
                           self.dependent_vars['w']]
        t_values = self.m_trajectory.index.values
        data = np.zeros((len(t_values), len(columns)))
        for i, t in enumerate(t_values):
            Yi = self.m_trajectory.loc[t]
            sbs = {var_symbol: Yi[var_name] for var_symbol, var_name in zip(self.var_symbols, self.var_names)}
            data[i, :] = [var.subs(sbs) for var in var_expressions]

        return pd.DataFrame(index=t_values, columns=columns, data=data)

if __name__ == '__main__':
    """
    Perform test run and plot some output to check
    functionality
    """
    import datetime
    import networkx as nx
    from random import shuffle
    import matplotlib.pyplot as plt

    output_location = 'test_output/' \
                      + datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") + '_output'

    # investment_decisions:

    nopinions = [50, 50]
    possible_opinions = [[0], [1]]

    # Parameters:

    input_parameters = {'i_tau': 1, 'eps': 0.05, 'b_d': 1.2,
                        'b_c': 0.4, 'i_phi': 0.8, 'e': 100,
                        'G_0': 30000, 'b_r0': 0.1 ** 2 * 100,
                        'possible_opinions': possible_opinions,
                        'c': 100, 'xi': 1./8., 'beta': 0.06,
                        'campaign': False, 'learning': True,
                        'imitation': 2}

    # investment_decisions
    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full(n, i, dtype='I'))
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

    model.run(t_max=1)

    trj = model.get_unified_trajectory()

    print(trj)

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    trj[model.var_names[0:3]].plot(ax=ax1)

    ax2 = fig.add_subplot(222)
    trj[model.var_names[3:7]].plot(ax=ax2)

    ax3 = fig.add_subplot(223)
    trj[model.var_names[7]].plot(ax=ax3)

    ax4 = fig.add_subplot(224)
    trj[model.var_names[8]].plot(ax=ax4)

    plt.show()
