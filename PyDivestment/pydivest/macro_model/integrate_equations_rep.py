
# coding: utf-8

# Equations for Representative Household approximation of
# Network-based micro-model for divestment of bounded rational households.

# Imports and setup

import sys

import numpy as np
import pandas as pd
import sympy as sp
try:
    from assimulo.problem import Implicit_Problem
    from assimulo.solvers import IDA
except ImportError:
    print('assimulo not available. Running model impossible.')
from scipy.optimize import root


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

        self.t_max = 0

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
        # investment_decisions as indices of possible_cue_orders
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
        # Solow residual for clean sector
        self.b_c = float(b_c)
        # Solow residual for dirty sector
        self.b_d = float(b_d)
        # elasticity of knowledge
        self.xi = float(xi)
        # labor elasticity (equal in both sectors)
        self.pi = float(pi)
        # clean and dirty capital elasticity
        if crs:
            self.kappa_c = 1. - self.pi - self.xi
            self.kappa_d = 1. - self.pi
        else:
            self.kappa_c = float(kappa_c)
            self.kappa_d = float(kappa_d)
        print('pi = {}, xi = {}, kappa_c = {}, kappa_d = {}'
              .format(self.pi, self.xi, self.kappa_c, self.kappa_d), flush=True)
        # fossil->energy->output conversion efficiency (Leontief)
        self.e = float(e)
        # total labor
        self.L = float(L)
        # total knowledge stock
        self.C = float(C)
        # unprofitable fraction of fossil reserve
        self.alpha = (b_r0 / e) ** 0.5

        # Ecosystem parameters

        # initial fossil resource stock
        self.G_0 = float(G_0)
        # total fossil resource stock
        self.G = float(G_0)
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

        # Total capital in the clean and dirty sector
        self.Kd_0 = sum(self.investment_dirty)
        self.Kc_0 = sum(self.investment_clean)

        # Define variables and parameters for the economic subsystem:

        # Total labor and labor shares in sectors
        L, Lc, Ld = sp.symbols('L L_c L_d', positive=True, real=True)
        # Total capital in sectors
        Kc, Kd = sp.symbols('K_c K_d', positive=True, real=True)
        # Equilibrium wage and capital return rates in sectors
        w, rc, rd = sp.symbols('w r_c r_d', positive=True, real=True)
        # Resource usage rage, resource stock, knowledge Stock
        R, G, C = sp.symbols('R, G, C', positive=True, real=True)
        # savings rate, capital depreciaten rate, and elasticities of labor, capital and knowledge
        (rs, delta, pi, kappac, kappad, xi, mu) = \
            sp.symbols('s delta pi kappa_c, kappa_d xi, mu', positive=True, real=True, integer=True)
        # solow residuals of clean and dirty sector, pre factor for
        # resource cost, energy efficiency, initial resource stock
        bc, bd, bR, e, G0 = sp.symbols('b_c b_d b_R e G_0', positive=True, real=True)
        # substitutions for resolution on constraints from market clearing.
        Xc, Xd, XR = sp.symbols('X_c X_d X_R', positive=True, real=True)
        # and their time derivatives.
        dXc, dXd, dXR = sp.symbols('\dot{X}_c \dot{X}_d \dot{X}_R', real=True)
        # fraction of savings going into clean sector
        n = sp.symbols('n', positive=True, real=True)
        # time derivatives
        dKc, dKd, dC, dG = sp.symbols('\dot{K}_c \dot{K}_d \dot{C} \dot{G}')

        # Values for parameters
        self.subs_params = {bc: self.b_c,
                            bd: self.b_d,
                            bR: self.b_r0,
                            e: self.e,
                            rs: self.s,
                            delta: self.d_c,
                            pi: self.pi,
                            kappac: self.kappa_c,
                            kappad: self.kappa_d,
                            xi: self.xi,
                            mu: 2.,
                            G0: self.G_0,
                            L: self.L}

        # **Treatment the equations describing economic production and capital accumulation**
        #
        # Substitute solutions to algebraic constraints of economic system
        # (market clearing for labor and expressions for capital rent and resource flow)

        # These are double checked
        subs2 = {w: pi * L**(pi-1.) * (Xc + Xd*XR)**(1.-pi),
                 rc: kappac/Kc*Xc*L**pi*(Xc + Xd*XR)**(-pi),
                 rd: kappad/Kd*Xd*XR*L**pi*(Xc + Xd*XR)**(-pi),
                 R:  bd/e*Kd**kappad*L**pi*(Xd*XR/(Xc + Xd*XR))**pi,
                 Lc: L*Xc/(Xc + Xd*XR),
                 Ld: L*Xd*XR/(Xc + Xd*XR)}

        # These are double checked
        subs3 = {Xc: (bc*Kc**kappac * C**xi)**(1./(1.-pi)),
                 Xd: (bd*Kd**kappad)**(1./(1.-pi)),
                 XR: (1.-bR/e*(G0/G)**mu)**(1./(1.-pi))}
        # Those too
        subs4 = {dXc: (1./(1.-pi))*(bc*Kc**kappac * C**xi)**(pi/(1.-pi))*bc*(kappac*Kc**(kappac-1)*dKc*C**xi
                                                                             + Kc**kappac*xi*C**(xi-1)*dC),
                 dXd: (1./(1.-pi))*(bd*Kd**kappad)**(pi/(1.-pi))*bd*kappad*Kd**(kappad-1)*dKd,
                 dXR: (1./(1.-pi))*(1.-bR/e*(G0/G)**mu)**(pi/(1.-pi))*(mu*bR/e*(G0**mu/G**(mu+1))*dG)}

        # Dynamic equations for the economic variables depending on n,
        # the fraction of savings going into the clean sector

        subs5 = {dKc: n*rs*(rc*Kc + rd*Kd + w*L) - delta*Kc,
                 dKd: - delta*Kd + (1-n)*rs*(rc*Kc + rd*Kd + w*L),
                 dC: bc*Lc**pi*Kc**kappac * C**xi - delta*C,
                 dG: -R}

        self.independent_vars = {'Kc': Kc, 'Kd': Kd, 'G': G, 'C': C, 'n': n}
        self.dependent_vars = {'w': w, 'rc': rc, 'rd': rd, 'R': R, 'Lc': Lc, 'Ld': Ld, 'L': L, 'rs': rs}
        for key in self.dependent_vars.keys():
            self.dependent_vars[key] = self.dependent_vars[key].subs(subs2).subs(subs3).subs(self.subs_params)

        # We want returns to capital to be equal in the clean and the dirty sector. This means, that for the initial
        # conditions the returns and their derivatives with respect to time have to be equal. Then, for the integration,
        # it is sufficient to keep the condition for the time derivatives of the returns.
        # This defines implicit condition for n, the fraction of savings invested in the clean sector.

        raw_rdiff = (rc - rd).subs(subs2)
        rdiff = raw_rdiff.subs(subs3)
        raw_drdiff = L**pi*(-pi)*(Xc + Xd*XR)**(-pi-1)*(dXc + dXd*XR + Xd*dXR)*(kappac/Kc*Xc - kappad/Kd*Xd*XR) + \
            L**pi*(Xc + Xd*XR)**(-pi)*(kappac*(dXc*Kc - Xc*dKc)/(Kc**2.)
                                       - kappad*((dXd*XR + Xd*dXR)*Kd - Xd*XR*dKd)/(Kd**2.))
        drdiff_g_const = raw_drdiff.subs(subs4).subs(subs5).subs(subs2).subs(subs3)
        drdiff = raw_drdiff.subs(subs4).subs(subs5).subs(subs2).subs(subs3)

        # List of dynamic variables and the right hand side of their dynamic equations as well as a list of
        # indicators of whether these equations are explicit of implicit

        self.var_symbols = [Kc, Kd, G, C, n]
        self.var_names = ['Kc', 'Kd', 'G', 'C', 'n']
        self.m_trajectory = pd.DataFrame(columns=self.var_names)
        self.Y0 = np.zeros(len(self.var_symbols))
        self.Yd0 = np.zeros(len(self.var_symbols))
        self.t0 = 0
        self.sw0 = [True, False, False]

        rhs_1 = sp.Matrix([dKc, dKd, dG, dC, drdiff]).subs(subs5).subs(subs2).subs(subs3)
        rhs_2 = sp.Matrix([dKc, dKd, dG, dC, n - 1]).subs(subs5).subs(subs2).subs(subs3)
        rhs_3 = sp.Matrix([dKc, dKd, 0., dC, drdiff_g_const]).subs(subs5).subs(subs2).subs(subs3)
        rhs_4 = sp.Matrix([dKc, dKd, 0., dC, n - 1]).subs(subs5).subs(subs2).subs(subs3)

        self.eq_implicit = [False, False, False, False, True]

        self.rhs_1 = rhs_1.subs(self.subs_params)
        self.rhs_2 = rhs_2.subs(self.subs_params)
        self.rhs_3 = rhs_3.subs(self.subs_params)
        self.rhs_4 = rhs_4.subs(self.subs_params)
        self.rdiff = rdiff.subs(self.subs_params)
        self.drdiff = drdiff.subs(self.subs_params)
        self.drdiff_g_const = drdiff_g_const.subs(self.subs_params)

    @staticmethod
    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    def find_initial_conditions(self):
        """
        # Defines sane initial conditions, that solve the residue function (as Frank called it).
        # This means in my case, that rc-rd as well as d/dt (rc-rd) have to be zero.
        
        Returns
        -------
        C: float
            the total renewable knowledge stock
        n: float
            the fraction of investment going into the clean sector
        """
        print('sanitizing initial conditions to')

        # define initial values for Kc, Kd, C and G
        Y0 = [self.Kc_0, self.Kd_0, self.G_0]
        sym = self.var_symbols[:3]
        subs_ini = {symbol: value for symbol, value in zip(sym, Y0)}

        C = self.var_symbols[3]
        F_ini_1 = sp.simplify(self.rdiff.subs(subs_ini))
        fun1 = lambda x: F_ini_1.subs({C: x}).evalf()
        r1 = root(fun1, 1.)
        subs_ini[C] = r1.x[0]

        n = self.var_symbols[4]
        if self.R_depletion:
            F_ini_2 = sp.simplify(self.drdiff.subs(subs_ini))
        else:
            F_ini_2 = sp.simplify(self.drdiff_g_const.subs(subs_ini))
        fun2 = lambda x: F_ini_2.subs({n: x}).evalf()
        r2 = root(fun2, .5)
        subs_ini[n] = r2.x[0]

        self.Y0 = np.array([subs_ini[var] for var in self.var_symbols])
        self.Yd0 = np.array(list(self.rhs_1.subs(subs_ini)))
        self.t0 = 0
        self.sw0 = [True, False, False]
        print(subs_ini)
        return r1.x[0], r2.x[0]

    def run(self, t_max, **kwargs):

        self.t_max = t_max
        # Define the problem for assimulo and run the simulation

        def prep_rhs(t, Y, Yd, sw):

            sbs = {var: val for (var, val) in zip(self.var_symbols, Y)}
            if sw[0] or sw[2]:
                if self.R_depletion:
                    rval = self.rhs_1.subs(sbs)
                else:
                    rval = self.rhs_3.subs(sbs)
            else:
                if self.R_depletion:
                    rval = self.rhs_2.subs(sbs)
                else:
                    rval = self.rhs_4.subs(sbs)

            for i in [0, 1, 2, 3]:
                rval[i] = Yd[i] - sp.simplify(rval[i])
            rval = np.array([float(x) for x in rval.evalf()])

            self.progress(t, self.t_max, 'representative agent running')

            return rval

        def state_events(t, Y, Yd, sw):

            event_1 = Y[-1] - 1
            event_2 = Y[0]  # This is just a place holder. Originally, this was to check if rc < rd again.

            # print('events', event_1, event_2, sw)
            return np.array([event_1, event_2, 0])

        def handle_event(solver, event_info):
            if event_info[0] != 0:
                solver.sw[0] = False
                solver.sw[1] = True
                subs_ini = {symbol: value for symbol, value in zip(self.var_symbols, solver.y)}
                solver.yd = np.array([float(x) for x in list(self.rhs_2.subs(subs_ini).evalf())])
                # print('first event, n reaching 1')
                # print(solver.y)
                # print(solver.yd)
                solver.re_init(solver.t, solver.y, solver.yd, sw0=solver.sw)
            elif event_info[1] != 0:
                solver.sw[1] = False
                solver.sw[2] = True
            pass

        mod = Implicit_Problem(prep_rhs,
                               self.Y0,
                               self.Yd0,
                               self.t0,
                               sw0=self.sw0)
        mod.algvar = self.eq_implicit
        mod.state_events = state_events
        mod.handle_event = handle_event
        sim = IDA(mod)
        sim.rtol = 1.e-8        # Sets the relative tolerance
        sim.atol = 1.e-6        # Sets the absolute tolerance
        t, Y, Yd = sim.simulate(t_max)

        df = pd.DataFrame(Y, index=t, columns=self.var_names)
        self.m_trajectory = pd.concat([self.m_trajectory, df]).groupby(level=0).mean()

        return

    def get_aggregate_trajectory(self):

        return self.m_trajectory

    def get_unified_trajectory(self):
        """calcualtes and returns a unified output trajectory in terms of per capita variables.
        
        The question is: do I devide by total labor, or by labor employed in the respective sector.
        For capital, it certainly makes sense, to devide by labor employed in the sector. Same for resource
        use and knowledge. For labor employed in the respective sector, it certainly makes sense to put it in
        terms of total labor. For remaining resource stock, I am not sure.
        
        Talking to jobst, I figured, it does not really matter, as long as the quantities can be computed from
        all of the different models. So I should start from the 'means' description, since it has the most
        restricting form.
        
        Starting there, It is probably easiest to just calculate per capita quantities."""

        L = self.dependent_vars['L']
        columns = ['k_c', 'k_d', 'l_c', 'l_d', 'g', 'c', 'r', 'n_c', 'i_c', 'r_c', 'r_d', 'w']
        var_expressions = [self.independent_vars['Kc'] / L,
                           self.independent_vars['Kd'] / L,
                           self.dependent_vars['Lc'] / L,
                           self.dependent_vars['Ld'] / L,
                           self.independent_vars['G'] / L,
                           self.independent_vars['C'] / L,
                           self.dependent_vars['R'] / L,
                           self.independent_vars['n'],
                           self.independent_vars['n']
                           * self.dependent_vars['rs']
                           * (self.independent_vars['Kc']*self.dependent_vars['rc']
                              + self.independent_vars['Kd']*self.dependent_vars['rd']) / L,
                           self.dependent_vars['rc'],
                           self.dependent_vars['rd'],
                           self.dependent_vars['w']]
        t_values = self.m_trajectory.index.values
        data = np.zeros((len(t_values), len(columns)))
        for i, t in enumerate(t_values):
            self.progress(i, len(t_values), 'calculating dependant variables')
            Yi = self.m_trajectory.loc[t]
            sbs = {var_symbol: Yi[var_name] for var_symbol, var_name in zip(self.var_symbols, self.var_names)}
            data[i, :] = [var.subs(sbs) for var in var_expressions]

        return pd.DataFrame(index=t_values, columns=columns, data=data)

if __name__ == "__main__":

    """
    Perform test run and plot some output to check
    functionality
    """
    import networkx as nx
    from random import shuffle
    import matplotlib.pyplot as plt

    # investment_decisions:

    nopinions = [50, 50]
    possible_cue_orders = [[0], [1]]

    # Parameters:

    input_parameters = {'i_tau': 1, 'eps': 0.05, 'b_d': 1.2,
                        'b_c': 0.4, 'i_phi': 0.8, 'e': 100,
                        'G_0': 3, 'b_r0': 0.1 ** 2 * 100,
                        'possible_cue_orders': possible_cue_orders,
                        'C': 100, 'xi': 1. / 8., 'd_c': 0.06,
                        'campaign': False, 'learning': True,
                        'crs': True}

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

    m = Integrate_Equations(*init_conditions, **input_parameters)
    m.find_initial_conditions()
    m.R_depletion = True
    t, Y = m.run(t_max=3)
    # Plot the results

    trj = m.get_unified_trajectory()

    fig = plt.figure()
    ax1 = fig.add_subplot(141)
    trj[['n_c']].plot(ax=ax1)

    ax2 = fig.add_subplot(142)
    trj[['k_c', 'k_d']].plot(ax=ax2)
    ax2b = ax2.twinx()
    trj[['c']].plot(ax=ax2b)

    ax3 = fig.add_subplot(143)
    trj[['r_c', 'r_d']].plot(ax=ax3)

    ax4 = fig.add_subplot(144)
    trj[['g']].plot(ax=ax4)

    plt.show()
