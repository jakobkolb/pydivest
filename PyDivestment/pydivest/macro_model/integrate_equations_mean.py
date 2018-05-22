# coding: utf-8

from __future__ import print_function

import sympy as sp
from scipy.integrate import odeint

import pandas as pd
import numpy as np

from .integrate_equations import IntegrateEquations


class IntegrateEquationsMean(IntegrateEquations):
    def __init__(self, adjacency=None, investment_decisions=None,
                 investment_clean=None, investment_dirty=None,
                 tau=0.8, phi=.7, eps=0.05,
                 b_c=1., b_d=1.5, s=0.23, d_c=0.06,
                 b_r0=1., e=10,
                 pi=0.5, xi=1. / 8.,
                 L=100., G_0=3000, C=1,
                 R_depletion=True,
                 interaction=1, crs=True, test=False,
                 **kwargs):
        """
        Implements the same interface as the aggregate approximation but requires constant
        returns to scale i.e. the crs, kappa_c and kappa_d variables have no effect.

        Parameters
        ----------
        adjacency: ndarray
            Acquaintance matrix between the households. Has to be symmetric unweighted and without self loops.
        investment_decisions: list
            Initial investment decisions of households. Will be updated
            from their actual heuristic decision making during initialization
        investment_clean: list
            Initial household endowments in the clean sector
        investment_dirty: list
            Initial household endowments in the dirty sector
        i_tau: float
            Mean waiting time between household opinion updates
        i_phi: float
            Rewiring probability in the network adaptation process
        eps: float
            fraction of exploration events (noise) in the opinion formation process
        b_c: float
            Solow residual of the production function of the clean sector
        b_d: float
            Solow residual of the production function of the dirty sector
        s: float
            Savings rate of the households
        d_c: float
            Capital depreciation rate
        b_r0: float
            Resource cost factor
        e: float
            Resource efficiency in the dirty sector
        pi: float
            labor elasticity for both sectors
        xi:
            elasticity of the knowledge stock in the clean sector
        L: float
            Total labor (fixed)
        G_0: float
            Total initial resource stock
        C: float
            Total initial knowledge stock
        resource_depletion: bool
            Switch to turn resource depreciation on or off
        interaction: int
            Switch for different imitation probabilities.
            if 0: tanh(Wi-Wj) interaction,
            if 1: interaction as in Traulsen, 2010 but with relative differences
            if 2: (Wi-Wj)/(Wi+Wj) interaction.
        """

        super().__init__(adjacency=adjacency, investment_decisions=investment_decisions,
                         investment_clean=investment_clean, investment_dirty=investment_dirty,
                         tau=tau, phi=phi, eps=eps,
                         pi=pi, xi=xi,
                         L=L, b_c=b_c, b_d=b_d, s=s, d_c=d_c,
                         b_r0=b_r0, e=e, G_0=G_0, C=C,
                         R_depletion=R_depletion, test=test, crs=crs, interaction=interaction)

        if len(kwargs.items()) > 0:
            print('got superfluous keyword arguments')
            print(kwargs.keys())
        if 'kappa_c' in kwargs.keys():
            print('value for kappa_c provided will have no effect, since the mean approximation'
                  'requires constant returns to scale')
        if 'kappa_d' in kwargs.keys():
            print('value for kappa_d provided will have no effect, since the mean approximation'
                  'requires constant returns to scale')
        # ensure constant returns to scale as required for mean approximation.
        self.p_kappa_d = 1 - pi
        self.p_kappa_c = 1 - pi - xi

        # Parse Initial conditions for mean capital endowments

        c = self.investment_decisions
        d = - self.investment_decisions + 1

        nc = sum(c)
        nd = sum(d)

        self.v_mucc = sum(self.investment_clean * c) / nc
        self.v_mucd = sum(self.investment_clean * d) / nd
        self.v_mudc = sum(self.investment_dirty * c) / nc
        self.v_mudd = sum(self.investment_dirty * d) / nd

        # define symbols for mean capital endowments
        self.mucc, self.mucd, self.mudc, self.mudd = sp.symbols('mu_c^c mu_c^d mu_d^c mu_d^d', positive=True, real=True)

        # create list of symbols and names of all independent variables
        self.var_symbols = [self.x, self.y, self.z, self.mucc, self.mucd, self.mudc, self.mudd, self.c, self.g]
        self.var_names = ['x', 'y', 'z', 'mu_c^c', 'mu_c^d', 'mu_d^c', 'mu_d^d', 'c', 'g']

        # define expected wealth as expected income.
        self.subs1[self.Wc] = self.rc * self.mucc + self.rd * self.mudc
        self.subs1[self.Wd] = self.rc * self.mucd + self.rd * self.mudd

        # Define clean and dirty capital as weighted sums over average endowments
        self.subs4[self.Kc] = (self.N / 2. * (1 + self.x) * self.mucc + self.N / 2. * (1 - self.x) * self.mucd)
        self.subs4[self.Kd] = (self.N / 2. * (1 + self.x) * self.mudc + self.N / 2. * (1 - self.x) * self.mudd)

        self.subs4[self.C] = self.N * self.c
        self.subs4[self.G] = self.N * self.g
        self.subs4[self.P] = self.N * self.p
        self.subs4[self.G0] = self.N * self.g0

        # ensure constant returns to scale by eliminating kappa
        self.subs5 = {self.kappac: 1. - self.pi - self.xi,
                      self.kappad: 1. - self.pi}

        # Write down dynamic equations for the economic subsystem in terms of means of clean and dirty capital stocks
        # for clean and dirty households

        self.rhsECO_1 = sp.Matrix(
            [(self.rs * self.rc - self.delta) * self.mucc + self.rs * self.rd * self.mudc
             + self.rs * self.w * self.P / self.N,
             -self.delta * self.mucd,
             -self.delta * self.mudc,
             self.rs * self.rc * self.mucd + (self.rs * self.rd - self.delta) * self.mudd
             + self.rs * self.w * self.P / self.N,
             self.bc * self.Pc ** self.pi * self.Kc ** self.kappac * self.C ** self.xi - self.delta * self.C,
             -self.R])

        self.rhsECO_switch_1 = sp.Matrix([
            # change of clean capital owned by clean investors
            (self.mucd - self.mucc) * self.dtNdc / self.Nc,
            # change of clean capital owned by dirty investors
            (self.mucc - self.mucd) * self.dtNcd / self.Nd,
            # change in dirty capital owned by clean investors
            (self.mudd - self.mudc) * self.dtNdc / self.Nc,
            # change in dirty capital owned by dirty investors
            (self.mudc - self.mudd) * self.dtNcd / self.Nd,
            0,
            0])

        self.rhsECO_switch_2 = self.rhsECO_switch_1.subs(self.subs1)

        self.rhsECO_2 = self.rhsECO_1 + self.rhsECO_switch_2

        # In the economic system, substitute:
        # 1)primitive variables for dependent variables (subs2)
        # 2)dependent variables for system variables (subs3)

        self.rhsECO_3 = self.rhsECO_2.subs(self.subs1).subs(self.subs2).subs(self.subs3)\
            .subs(self.subs4).subs(self.subs5)

        # In the PBP rhs substitute:
        # dependent variables for system variables
        # NOTE TO SELF: DO NOT WRITE TO PARENT CLASS VARIABLES. THIS WILL BACKFIRE, IF OTHER
        # CLASSES INHERIT FROM THE SAME PARENT!

        self.rhsPBP_1 = self.rhsPBP.subs(self.subs1)

        self.rhsPBP_2 = self.rhsPBP_1.subs(self.subs1).subs(self.subs2).subs(self.subs3)\
            .subs(self.subs4).subs(self.subs5)

        # Combine dynamic equations of economic and social subsystem:

        self.rhs_raw = sp.Matrix([self.rhsPBP_2, self.rhsECO_3]).subs(self.subs1)

        # update dependent vars with specific approximation variables
        self.update_dependent_vars()
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

            trajectory = odeint(self.dot_rhs, initial_conditions, t)

            df = pd.DataFrame(trajectory, index=t, columns=self.var_names)
            self.m_trajectory = pd.concat([self.m_trajectory, df])

            # update aggregated variables:
            (self.v_x, self.v_y, self.v_z,
             self.v_mucc, self.v_mucd,
             self.v_mudc, self.v_mudd,
             self.v_c, self.v_g) = trajectory[-1]

            self.v_C = self.c * self.p_n
            self.v_G = self.g * self.p_n
            self.v_t = t_max

        elif t_max <= self.v_t:
            if self.p_test:
                print('upper time limit is smaller than system time', self.v_t)

        return 1

    def get_unified_trajectory(self):
        """
        Calculates unified trajectory in per capita variables

        Returns
        -------
        Dataframe of unified per capita variables if calculation succeeds,
        else return -1 for TypeError and -2 for ValueError
        """

        L = self.dependent_vars['L']
        l = self.dependent_vars['l']

        columns = ['k_c', 'k_d', 'l_c', 'l_d', 'g', 'c', 'r',
                   'n_c', 'i_c', 'r_c', 'r_d', 'w',
                   'W_c', 'W_d', 'Pcd', 'Pdc']
        var_expressions = [(self.independent_vars['mu_c^c'] * self.dependent_vars['Nc']
                            + self.independent_vars['mu_c^d'] * self.dependent_vars['Nd']) / L,
                           (self.independent_vars['mu_d^c'] * self.dependent_vars['Nc']
                            + self.independent_vars['mu_d^d'] * self.dependent_vars['Nd']) / L,
                           self.dependent_vars['Lc'] / L,
                           self.dependent_vars['Ld'] / L,
                           self.independent_vars['g'] / l,
                           self.independent_vars['c'] / l,
                           self.dependent_vars['R'] / L,
                           (self.independent_vars['x'] + 1.) / 2.,
                           (self.dependent_vars['rc'] * self.independent_vars['mu_c^c']
                            + self.dependent_vars['rd'] * self.independent_vars['mu_d^c'])
                           / (self.dependent_vars['rc'] * (self.independent_vars['mu_c^c']
                                                           + self.independent_vars['mu_c^d'])
                              + self.dependent_vars['rd'] * (self.independent_vars['mu_d^c']
                                                             + self.independent_vars['mu_d^d'])),
                           self.dependent_vars['rc'],
                           self.dependent_vars['rd'],
                           self.dependent_vars['w'],
                           self.dependent_vars['W_c'],
                           self.dependent_vars['W_d'],
                           self.dependent_vars['Pcd'],
                           self.dependent_vars['Pdc']
                           ]

        return self.calculate_unified_trajectory(columns=columns,
                                                 var_expressions=var_expressions)
