# coding: utf-8

from __future__ import print_function

import sympy as sp
from scipy.integrate import odeint

import pandas as pd
import numpy as np

from .Integrate_Equations import IntegrateEquations


class IntegrateEquationsMean(IntegrateEquations):
    def __init__(self, adjacency=None, investment_decisions=None,
                 investment_clean=None, investment_dirty=None,
                 tau=0.8, phi=.7, eps=0.05,
                 b_c=1., b_d=1.5, s=0.23, d_c=0.06,
                 b_r0=1., e=10,
                 pi=0.5, kappa_c=0.4, kappa_d=0.5, xi=1. / 8.,
                 L=100., G_0=3000, C=1,
                 R_depletion=True,
                 interaction=0, crs=True, test=False,
                 **kwargs):

        super().__init__(adjacency=adjacency, investment_decisions=investment_decisions,
                         investment_clean=investment_clean, investment_dirty=investment_dirty,
                         tau=tau, phi=phi, eps=eps,
                         pi=pi, kappa_c=kappa_c, kappa_d=kappa_d, xi=xi,
                         L=L, b_c=b_c, b_d=b_d, s=s, d_c=d_c,
                         b_r0=b_r0, e=e, G_0=G_0, C=C,
                         R_depletion=R_depletion, test=test, crs=crs, interaction=interaction)

        if len(kwargs.items()) > 0:
            print('got superfluous keyword arguments')
            print(kwargs.keys())
        # ensure constant returns to scale as required for mean approximation.
        self.p_kappa_d = 1 - pi
        self.p_kappa_c = 1 - pi - xi

        # Parse Initial conditions for mean capital endowments

        c = self.investment_decisions
        d = - self.investment_decisions + 1

        nc = sum(c)
        nd = sum(d)

        self.v_mucc = sum(investment_clean * c) / nc
        self.v_mucd = sum(investment_clean * d) / nd
        self.v_mudc = sum(investment_dirty * c) / nc
        self.v_mudd = sum(investment_dirty * d) / nd

        # define symbols for mean capital endowments
        mucc, mucd, mudc, mudd = sp.symbols('mu_c^c mu_c^d mu_d^c mu_d^d', positive=True, real=True)

        # Add new variables to list of independent Variables.
        for key, val in [('mu_c^c', mucc), ('mu_d^c', mudc), ('mu_c^d', mucd), ('mu_d^d', mudd)]:
            self.independent_vars[key] = val

        # create list of symbols and names of all independent variables
        self.var_symbols = [self.x, self.y, self.z, mucc, mucd, mudc, mudd, self.c, self.g]
        self.var_names = ['x', 'y', 'z', 'mu_c^c', 'mu_d^c', 'mu_c^d', 'mu_d^d', 'c', 'g']

        # define expected wealth as expected income.
        self.subs1[self.Wc] = self.rc * mucc + self.rd * mudc
        self.subs1[self.Wd] = self.rc * mucd + self.rd * mudd

        # Define clean and dirty capital as weighted sums over average endowments
        self.subs4[self.Kc] = (self.N / 2. * (1 + self.x) * mucc + self.N / 2. * (1 - self.x) * mucd)
        self.subs4[self.Kd] = (self.N / 2. * (1 + self.x) * mudc + self.N / 2. * (1 - self.x) * mudd)

        # ensure constant returns to scale by eliminating kappa
        self.subs5 = {self.kappac: 1. - self.pi - self.xi,
                      self.kappad: 1. - self.pi}

        # Write down dynamic equations for the economic subsystem in terms of means of clean and dirty capital stocks
        # for clean and dirty households

        self.rhsECO_1 = sp.Matrix(
            [(self.rs * self.rc - self.delta) * mucc + self.rs * self.rd * mudc + self.rs * self.w * self.P / self.N,
             -self.delta * mucd,
             -self.delta * mudc,
             self.rs * self.rc * mucd + (self.rs * self.rd - self.delta) * mudd + self.rs * self.w * self.P / self.N,
             self.bc * self.Pc ** self.pi * (
                     self.Nc * mucc + self.Nd * mucd) ** self.kappac * self.C ** self.xi - self.delta * self.C,
             -self.R])

        self.rhsECO_switch_1 = sp.Matrix([(mucd - mucc) * self.dtNdc / self.Nc,
                                          (mucc - mucd) * self.dtNcd / self.Nd,
                                          (mudd - mudc) * self.dtNdc / self.Nc,
                                          (mudc - mudd) * self.dtNcd / self.Nd,
                                          0,
                                          0])

        self.rhsECO_switch_2 = self.rhsECO_switch_1.subs(self.subs1)

        self.rhsECO_2 = self.rhsECO_1 + self.rhsECO_switch_2

        # In the economic system, substitute:
        # 1)primitive variables for dependent variables (subs2)
        # 2)dependent variables for system variables (subs3)

        self.rhsECO_3 = self.rhsECO_2.subs(self.subs1).subs(self.subs2).subs(self.subs3).subs(self.subs4).subs(self.subs5)

        # In the PBP rhs substitute:
        # dependent variables for system variables

        self.rhsPBP = self.rhsPBP.subs(self.subs1)

        self.rhsPBP_2 = self.rhsPBP.subs(self.subs1).subs(self.subs2).subs(self.subs3).subs(self.subs4).subs(self.subs5)

        # Combine dynamic equations of economic and social subsystem:

        self.rhs_raw = sp.Matrix([self.rhsPBP_2, self.rhsECO_3]).subs(self.subs1)

        # Set parameter values in rhs and dependent variables:
        self.set_parameters()

        self.m_trajectory = pd.DataFrame(columns=self.var_names)

        # dictionary for final state
        self.final_state = {}

    def get_m_trajectory(self):

        return self.m_trajectory

    def run(self, t_max=100, t_steps=500):
        """
        run the model for a given time t_max and produce results in resolution t_steps
        Parameters
        ----------
        t_max: float
            upper limit of simulation time
        t_steps: int
            number of timesteps of result

        Returns
        -------
        rval: int
            positive, if the simulation succeeded.
        """
        self.p_t_max = t_max

        if t_max > self.v_t:
            if self.p_test:
                print('integrating equations from t={} to t={}'.format(self.v_t, t_max))

            t = np.linspace(self.v_t, t_max, t_steps)

            initial_conditions = [self.v_x, self.v_y, self.v_z,
                                  self.v_mucc, self.v_mucd,
                                  self.v_mudc, self.v_mudd,
                                  self.v_c, self.v_g]

            print(initial_conditions)

            trajectory = odeint(self.dot_rhs, initial_conditions, t)

            df = pd.DataFrame(trajectory, index=t, columns=self.var_names)
            self.m_trajectory = pd.concat([self.m_trajectory, df])

            # update aggregated variables:
            (self.v_x, self.v_y, self.v_z,
             self.v_mucc, self.v_mucd,
             self.v_mudc, self.v_mudd,
             self.v_c, self.v_g) = trajectory[-1]

            print(trajectory[-1])

            self.v_C = self.c * self.p_n
            self.v_G = self.g * self.p_n
            self.v_t = t_max

        elif t_max <= self.v_t:
            if self.p_test:
                print('upper time limit is smaller than system time', self.v_t)

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
                        'C': 100, 'xi': 1. / 8., 'beta': 0.06,
                        'campaign': False, 'learning': True}

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

    model = IntegrateEquationsMean(*init_conditions, **input_parameters)

    model.run(t_max=2)

    trj = model.get_m_trajectory()

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
