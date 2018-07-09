# coding: utf-8

# Equations for Representative Household approximation of
# Network-based micro-model for divestment of bounded rational households.

# Imports and setup

import sys

import numpy as np
import pandas as pd
import sympy as sp
from sympy import lambdify


from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
# except ImportError:
#     print('assimulo not available. Running model impossible.')
#     Implicit_Problem = None
#     IDA = None
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
                 interaction=2, crs=True, test=False, **kwargs):

        # set debug flag
        self.test = test

        # use lambdify to speed up evaluation of rhs of the system
        self.lambdify = True

        # report unnecessary keyword arguments
        if len(kwargs.keys()) > 0 and self.test:
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
        # depending on requirement of constant returns to scale
        if crs:
            self.kappa_c = 1. - self.pi - self.xi
            self.kappa_d = 1. - self.pi
        else:
            self.kappa_c = float(kappa_c)
            self.kappa_d = float(kappa_d)
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

        # Dummy initial knowledge
        self.C_0 = 1.

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

        self.param_symbols = [bc, bd, bR, e, rs, delta, pi, kappac, kappad, xi, mu, G0, L]

        # **Treatment the equations describing economic production and capital accumulation**

        # Substitute solutions to algebraic constraints of economic system
        # (market clearing for labor and expressions for capital rent and resource flow)

        # Solution to algebraic constraints from capital and laber market clearing as well
        # as efficient resource use in the dirty sector.
        subs2 = {w: pi * L ** (pi - 1.) * (Xc + Xd * XR) ** (1. - pi),
                 rc: kappac / Kc * Xc * L ** pi * (Xc + Xd * XR) ** (-pi),
                 rd: kappad / Kd * Xd * XR * L ** pi * (Xc + Xd * XR) ** (-pi),
                 R: bd / e * Kd ** kappad * L ** pi * (Xd * XR / (Xc + Xd * XR)) ** pi,
                 Lc: L * Xc / (Xc + Xd * XR),
                 Ld: L * Xd * XR / (Xc + Xd * XR)}

        subs2_empty_G = {w: bc * pi * L ** (pi - 1.) * Kc ** kappac * C ** xi,
                         rc: bc * L ** pi * kappac * Kc ** (kappac - 1.) * C ** xi,
                         rd: 0.,
                         R: 0.,
                         Lc: L,
                         Ld: 0.}

        # Substitutions to the above expressions
        subs3 = {Xc: (bc * Kc ** kappac * C ** xi) ** (1. / (1. - pi)),
                 Xd: (bd * Kd ** kappad) ** (1. / (1. - pi)),
                 XR: (1. - bR / e * (G0 / G) ** mu) ** (1. / (1. - pi))}

        # derivatives of the above substitutions
        subs4 = {dXc: (1. / (1. - pi)) * (bc * Kc ** kappac * C ** xi) ** (pi / (1. - pi)) * bc * (
                kappac * Kc ** (kappac - 1) * dKc * C ** xi
                + Kc ** kappac * xi * C ** (xi - 1) * dC),
                 dXd: (1. / (1. - pi)) * (bd * Kd ** kappad) ** (pi / (1. - pi)) * bd * kappad * Kd ** (
                         kappad - 1) * dKd,
                 dXR: (1. / (1. - pi)) * (1. - bR / e * (G0 / G) ** mu) ** (pi / (1. - pi)) * (
                         mu * bR / e * (G0 ** mu / G ** (mu + 1)) * dG)}

        # Dynamic equations for the economic variables depending on n,
        # the fraction of savings going into the clean sector

        # ToDo: find better fix.
        # Hacky lower bound for Capital and Knowledge to prevent breakdown of solver in long runs.
        lb = 1e-9

        subs5 = {dKc: n * rs * (rc * Kc + rd * Kd + w * L) - delta * (Kc - lb),
                 dKd: - delta * (Kd - lb) + (1 - n) * rs * (rc * Kc + rd * Kd + w * L),
                 dC: bc * Lc ** pi * Kc ** kappac * C ** xi - delta * (C - lb),
                 dG: -R}

        # For the case of infinite fossil resources, we copy the above dynamic equations and
        # set the resource depletion term to zero.
        subs5_g_const = subs5.copy()
        subs5_g_const[dG] = 0

        # To simplify the output functions, we make a list of dependent and independent variables
        self.independent_vars = {'Kc': Kc, 'Kd': Kd, 'G': G, 'C': C, 'n': n}
        self.dependent_vars_raw = {'w': w, 'rc': rc, 'rd': rd, 'R': R, 'Lc': Lc, 'Ld': Ld, 'L': L, 'rs': rs}
        for key in self.dependent_vars_raw.keys():
            self.dependent_vars_raw[key] = self.dependent_vars_raw[key].subs(subs2).subs(subs3)

        self.dependent_vars = {}

        # We want returns to capital to be equal in the clean and the dirty sector. This means, that for the initial
        # conditions the returns and their derivatives with respect to time have to be equal. Then, for the integration,
        # it is sufficient to keep the condition for the time derivatives of the returns.
        # This defines implicit condition for n, the fraction of savings invested in the clean sector.

        # difference of capital return rates (is required to be zero).
        # Unfortunately, this expression turns out to be independent of n,
        # such that it is useless as an algebraic constraint.
        # Consequently, we use it to calculate the initial value of C
        # and use its time differential to fix the value of n.
        raw_rdiff = (rc - rd).subs(subs2)
        self.rdiff_raw = raw_rdiff.subs(subs3)

        # difference of capital return rates differentiated with respect to t which depends on n.
        raw_drdiff = L ** pi * (-pi) * (Xc + Xd * XR) ** (-pi - 1) * (dXc + dXd * XR + Xd * dXR) * (
                kappac / Kc * Xc - kappad / Kd * Xd * XR) + \
                     L ** pi * (Xc + Xd * XR) ** (-pi) * (kappac * (dXc * Kc - Xc * dKc) / (Kc ** 2.)
                                                          - kappad * ((dXd * XR + Xd * dXR) * Kd - Xd * XR * dKd)
                                                          / (Kd ** 2.))

        # we substitute with different expressions depending on whether the fossil resource is finite or not.
        self.drdiff_g_const_raw = raw_drdiff.subs(subs4).subs(subs5_g_const).subs(subs2).subs(subs3)
        self.drdiff_raw = raw_drdiff.subs(subs4).subs(subs5).subs(subs2).subs(subs3)

        self.drdiff_g_const, self.drdiff, self.rdiff = None, None, None

        # List of dynamic variables and the right hand side of their dynamic equations as well as a list of
        # indicators of whether these equations are explicit of implicit

        self.var_symbols = [Kc, Kd, G, C, n]
        self.var_names = ['Kc', 'Kd', 'G', 'C', 'n']

        # we have to define different versions of the rhs for the
        # boundaries (where n=1 or n=0) and below as well as for finite and infinite fossil resources:

        # n = 0, rc < rd
        self.rhs_1_raw = sp.Matrix([dKc, dKd, dG, dC, n]).subs(subs5).subs(subs2).subs(subs3)
        self.rhs_2_raw = sp.Matrix([dKc, dKd, 0., dC, n]).subs(subs5_g_const).subs(subs2).subs(subs3)

        # n is variable, rc == rd
        self.rhs_3_raw = sp.Matrix([dKc, dKd, dG, dC, self.drdiff_raw]).subs(subs5).subs(subs2).subs(subs3)
        self.rhs_4_raw = sp.Matrix([dKc, dKd, 0., dC, self.drdiff_g_const_raw]).subs(subs5_g_const).subs(subs2) \
            .subs(subs3)

        # n is 1, rc > rd
        self.rhs_5_raw = sp.Matrix([dKc, dKd, dG, dC, n - 1]).subs(subs5).subs(subs2).subs(subs3)
        self.rhs_6_raw = sp.Matrix([dKc, dKd, 0., dC, n - 1]).subs(subs5_g_const).subs(subs2).subs(subs3)

        # G exhausted => n is 1, rc == 0, dG ==0
        self.rhs_7_raw = sp.Matrix([dKc, dKd, 0., dC, n - 1]).subs(subs5_g_const).subs(subs2_empty_G).subs(subs3)

        # put righthandsides into dictionary of the form {number of rhs: expression}
        # to deal with substitutions and selection more efficiently later
        self.righthandsides_raw = {repr(i + 1): rhs for i, rhs in
                                   enumerate([self.rhs_1_raw, self.rhs_2_raw, self.rhs_3_raw, self.rhs_4_raw,
                                              self.rhs_5_raw, self.rhs_6_raw, self.rhs_7_raw])}
        self.righthandsides = {}
        self.righthandsides_lambda = {}

        # ToDo: remove this if not neccessary anymore
        # set dummy rhs for version with parameters replaced generated by set_parameters()
        self.rhs_1, self.rhs_2, self.rhs_3, self.rhs_4, self.rhs_5, self.rhs_6, self.rhs_7 = \
            None, None, None, None, None, None, None

        self.eq_implicit = [False, False, False, False, True]

        # initialize output trajectory
        self.m_trajectory = pd.DataFrame(columns=self.var_names)

        # set dummy initial conditions
        self.Y0 = np.zeros(len(self.var_symbols))
        self.Yd0 = np.zeros(len(self.var_symbols))
        self.sw0 = []

        self.sim = None
        self.set_parameters()

    def set_parameters(self):
        """(re)set parameters in rhs of system and dependen variable expressions"""

        if self.test:
            print('resetting parameter values...')

        # list all parameter values,
        param_values = [self.b_c, self.b_d, self.b_r0, self.e, self.s, self.d_c, self.pi, self.kappa_c,
                        self.kappa_d, self.xi, 2., self.G_0, self.L]

        # link parameter symbols to values,
        subs_params = {symbol: value for symbol, value
                       in zip(self.param_symbols, param_values)}

        self.rdiff = self.rdiff_raw.subs(subs_params)
        self.drdiff = self.drdiff_raw.subs(subs_params)
        self.drdiff_g_const = self.drdiff_g_const_raw.subs(subs_params)

        if self.lambdify:
            # replace parameters in righthandsides
            for key, rhs in self.righthandsides_raw.items():
                self.righthandsides[key] = rhs.subs(subs_params)

            # lambdify rhs expressions
            if self.test:
                print('lambdify rhs expressions')
            for key, rhs in self.righthandsides.items():
                self.righthandsides_lambda[key] = [lambdify(tuple(self.var_symbols), r_i) for r_i in rhs]
        else:
            # ToDo: cleanup
            self.rhs_1 = self.rhs_1_raw.subs(subs_params)
            self.rhs_2 = self.rhs_2_raw.subs(subs_params)
            self.rhs_3 = self.rhs_3_raw.subs(subs_params)
            self.rhs_4 = self.rhs_4_raw.subs(subs_params)
            self.rhs_5 = self.rhs_5_raw.subs(subs_params)
            self.rhs_6 = self.rhs_6_raw.subs(subs_params)
            self.rhs_7 = self.rhs_7_raw.subs(subs_params)

        # replace parameter symbols in raw independent variables.
        for key in self.dependent_vars_raw.keys():
            self.dependent_vars[key] = self.dependent_vars_raw[key].subs(subs_params)

        self._setup_model()
        if self.test:
            print('successful')

    @staticmethod
    def _progress_report(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    def _find_n(self, Y):
        subs_ini = {symbol: value for symbol, value in zip(self.var_symbols[:4], Y[:4])}
        n = self.var_symbols[4]
        print(subs_ini)
        if self.R_depletion:
            fun2 = lambdify(n, self.drdiff.subs(subs_ini))
        else:
            fun2 = lambdify(n, self.drdiff_g_const.subs(subs_ini))
        return root(fun2, .5).x[0]

    def _setup_model(self, Y=None):
        """
        # Defines sane initial conditions, that solve the residue function (as Frank called it).
        # This means in my case, that rc-rd as well as d/dt (rc-rd) have to be zero.
        
        Returns
        -------
        sw: the switch vector that indicates the regime of the system for the given initial conditions.
            (rc>rd, rc=rd, rc>rd)
        n: float
            the fraction of investment going into the clean sector
        """

        # define initial values for Kc, Kd, C and G
        if Y is None:
            Y0 = [self.Kc_0, self.Kd_0, self.G_0, self.C_0]
        else:
            Y0 = Y[:4]
        sym = self.var_symbols[:4]
        subs_ini = {symbol: value for symbol, value in zip(sym, Y0)}

        rc = float(self.dependent_vars['rc'].subs(subs_ini))
        rd = float(self.dependent_vars['rd'].subs(subs_ini))

        n = self.var_symbols[4]

        # set n and switches according to capital returns
        if rc > rd:
            self.sw0 = [False, False, True, False]
            n_val = 1.
        elif rc < rd:
            self.sw0 = [True, False, False, False]
            n_val = 0.
        else:
            self.sw0 = [False, True, False, False]
            if self.R_depletion:
                fun2 = lambdify(n, self.drdiff.subs(subs_ini))
            else:
                fun2 = lambdify(n, self.drdiff_g_const.subs(subs_ini))
            n_val = root(fun2, .5).x[0]

        subs_ini[n] = n_val

        # ToDo: Cleanup
        if self.lambdify:
            Y = [subs_ini[symbol] for name, symbol in self.independent_vars.items()]
            # calculate Yd0 from rhs according to switches
            if self.sw0[0]:
                if self.R_depletion:
                    rhs_select = 1
                else:
                    rhs_select = 2
            elif self.sw0[1]:
                if self.R_depletion:
                    rhs_select = 3
                else:
                    rhs_select = 4
            elif self.sw0[2]:
                if self.R_depletion:
                    rhs_select = 5
                else:
                    rhs_select = 6
            else:
                raise ValueError("one of the entries of sw0 has to be true.")
            Yd0 = np.asarray([rhs_i(*Y) for rhs_i in self.righthandsides_lambda[repr(rhs_select)]])
        else:
            # calculate Yd0 from rhs according to switches
            if self.sw0[0]:
                # calculate dy from the appropriate rhs (1 or 2).
                if self.R_depletion:
                    Yd0 = np.array([float(x) for x in list(self.rhs_1.subs(subs_ini).evalf())])
                else:
                    Yd0 = np.array([float(x) for x in list(self.rhs_2.subs(subs_ini).evalf())])
            elif self.sw0[1]:
                # calculate dy from the appropriate rhs (3 or 4).
                if self.R_depletion:
                    Yd0 = np.array([float(x) for x in list(self.rhs_3.subs(subs_ini).evalf())])
                else:
                    Yd0 = np.array([float(x) for x in list(self.rhs_4.subs(subs_ini).evalf())])
            elif self.sw0[2]:
                print('selecting rhs 5 or 6')
                # calculate dy from the appropriate rhs (5 or 6).
                if self.R_depletion:
                    Yd0 = np.array([float(x) for x in list(self.rhs_5.subs(subs_ini).evalf())])
                else:
                    Yd0 = np.array([float(x) for x in list(self.rhs_6.subs(subs_ini).evalf())])
            else:
                raise ValueError("one of the entries of sw0 has to be true.")

        # calculate Y0
        Y0 = [subs_ini[symbol] for symbol in self.var_symbols]

        # Define the problem for Assimulo with Y0 and Yd0
        def prep_rhs(t, Y, Yd, sw):
            if self.test:
                print('t = {}, Y = {}'.format(t, Y))

            print('t = {}, Y = {}'.format(t, Y))

            # ToDo: Cleanup
            if self.lambdify:
                # select rhs
                if sw[0]:
                    if self.R_depletion:
                        rhs_select = 1
                    else:
                        rhs_select = 2
                elif sw[1]:
                    if self.R_depletion:
                        rhs_select = 3
                    else:
                        rhs_select = 4
                elif sw[2]:
                    if self.R_depletion:
                        rhs_select = 5
                    else:
                        rhs_select = 6
                elif sw[3]:
                    rhs_select = 7
                else:
                    raise ValueError('system state undetermined')
                # evaluate rhs
                rval = np.asarray([rhs_i(*Y) for rhs_i in self.righthandsides_lambda[repr(rhs_select)]])

            else:
                sbs = {var: val for (var, val) in zip(self.var_symbols, Y)}
                if sw[0]:
                    if self.R_depletion:
                        rval = self.rhs_1.subs(sbs)
                    else:
                        rval = self.rhs_2.subs(sbs)
                elif sw[1]:
                    if self.R_depletion:
                        rval = self.rhs_3.subs(sbs)
                    else:
                        rval = self.rhs_4.subs(sbs)
                elif sw[2]:
                    if self.R_depletion:
                        rval = self.rhs_5.subs(sbs)
                    else:
                        rval = self.rhs_6.subs(sbs)
                elif sw[3]:
                    rval = self.rhs_7.subs(sbs)

            for i in [0, 1, 2, 3]:
                rval[i] = Yd[i] - sp.simplify(rval[i])

            if not self.lambdify:
                rval = np.array([float(x) for x in rval.evalf()])

            if self.test:
                self._progress_report(t, self.t_max, 'representative agent running')
                pass

            return rval

        def state_events(t, Y, Yd, sw):
            """check for events. such as transgression of n across 0 and 1 and crossings or r_c and r_d

            events are expressions that change sign from negative to positive in case of the specific event."""

            sbs = {var_symbol: Y[var_index] for var_index, var_symbol in enumerate(self.var_symbols)}

            rc = float(self.dependent_vars['rc'].subs(sbs))
            rd = float(self.dependent_vars['rd'].subs(sbs))

            # since the difference between rc and rd is only zero in the bounds of the solvers accuracy,
            # we need to subtract some epsilon to make sure that we only detect the relevant zero crossings.
            eps = 0.0

            event_1 = 1. if sw[2] else Y[-1] - 1  # n crossing 1 from below, exp. positive if n>1
            event_2 = 1. if sw[0] else - Y[-1]  # n crossing 0 from above exp. positive if n<0

            # check exit from state 2 where rc>rd, hence n = 1 (condition inactive in state 0 and 1).
            event_3 = 1. if sw[1] or sw[0] else rd - eps - rc  # check if rc < rd again, exp. positive, if rc < rd + eps

            # check exit from state 0 where rd>rc, hence n = 0 (condition inactive in state 1 and 2).
            event_4 = 1. if sw[1] or sw[2] else rc - eps - rd  # check if rc > rd again, exp. positive, if rd < rc + eps

            # check if resource is exhausted
            event_5 = self.G_0 - Y[2] * (self.e / self.b_r0) ** (1. / 2.)

            # print(t, event_1, event_2, event_3, event_4, event_5)

            return np.array([event_1, event_2, event_3, event_4, event_5])

        def handle_event(solver, event_info):

            ev = event_info[0]
            # event_1, n reaches 1 from below.
            print('event info is', event_info)
            subs_ini = {symbol: value for symbol, value in zip(self.var_symbols, solver.y)}

            if ev[2] == 1 or ev[3] == 1:
                # the system is back in its optimizing (variable n mode).
                # set the switches,
                # find the appropriate n,
                # reinitialize the solver with the right yd.
                if ev[2] == 1:
                    print('rc<rd again, n crossed 0 from below')
                else:
                    print('rd>rc again, n crossed 1 from above')
                # find n that solves the residual function
                solver.y[4] = self._find_n(solver.y)
                if solver.y[4] < 0:
                    # n is below New Event0 already. Skip and jump to case 'n crossed 0 from above'
                    print('back to optimizing range, but n={} already out of bounds'.format(solver.y[4]))
                    ev[1] = 1  # continue with n < 0
                elif solver.y[4] > 1:
                    # n is above 1 already. Skip and jump to case 'n crossed 1 from below'
                    print('back to optimizing range, but n={} already out of bounds'.format(solver.y[4]))
                    ev[0] = 1  # continue with n > 1
                else:
                    print('back in optimizing range, n={} in bounds'.format(solver.y[4]))
                    solver.sw = [False, True, False, False]
                    # put it into the substitution list.
                    subs_ini = {symbol: value for symbol, value in zip(self.var_symbols, solver.y)}
                    print(subs_ini)
                    # calculate dy from the appropriate rhs (3 or 4).
                # ToDo: Cleanup
                    if self.lambdify:
                        if self.R_depletion:
                            solver.yd = np.asarray([rhs_i(*solver.y)
                                                    for rhs_i in self.righthandsides_lambda['3']])
                        else:
                            solver.yd = np.asarray([rhs_i(*solver.y)
                                                    for rhs_i in self.righthandsides_lambda['4']])
                    else:
                        if self.R_depletion:
                            solver.yd = np.array([float(x) for x in list(self.rhs_3.subs(subs_ini).evalf())])
                        else:
                            solver.yd = np.array([float(x) for x in list(self.rhs_4.subs(subs_ini).evalf())])

            if ev[0] == 1:
                # n crossed 1 from below.
                print('n crossed 1 from below')
                solver.y[4] = 1
                solver.sw = [False, False, True, False]
                # calculate dy from the appropriate rhs (5 or 6).
                # ToDo: Cleanup
                if self.lambdify:
                    if self.R_depletion:
                        solver.yd = np.asarray([rhs_i(*solver.y)
                                                for rhs_i in self.righthandsides_lambda['5']])
                    else:
                        solver.yd = np.asarray([rhs_i(*solver.y)
                                                for rhs_i in self.righthandsides_lambda['6']])
                else:
                    if self.R_depletion:
                        solver.yd = np.array([float(x) for x in list(self.rhs_5.subs(subs_ini).evalf())])
                    else:
                        solver.yd = np.array([float(x) for x in list(self.rhs_6.subs(subs_ini).evalf())])
            elif ev[1] == 1:
                # n crossed 0 from above
                solver.y[4] = 0
                print('n crossed 0 from above')
                solver.sw = [True, False, False, False]
                # calculate dy from the appropriate rhs (1 or 2).
                # ToDo: Cleanup
                if self.lambdify:
                    if self.R_depletion:
                        solver.yd = np.asarray([rhs_i(*solver.y)
                                                for rhs_i in self.righthandsides_lambda['1']])
                    else:
                        solver.yd = np.asarray([rhs_i(*solver.y)
                                                for rhs_i in self.righthandsides_lambda['2']])
                else:
                    if self.R_depletion:
                        solver.yd = np.array([float(x) for x in list(self.rhs_1.subs(subs_ini).evalf())])
                    else:
                        solver.yd = np.array([float(x) for x in list(self.rhs_2.subs(subs_ini).evalf())])
            elif ev[4] == 1:
                # resource is exhausted
                print('resource exhausted')
                solver.sw = [False, False, False, True]
                # ToDo: Cleanup
                if self.lambdify:
                    solver.yd = np.asarray([rhs_i(*solver.y)
                                            for rhs_i in self.righthandsides_lambda['7']])
                else:
                    solver.yd = np.array([float(x) for x in list(self.rhs_7.subs(subs_ini).evalf())])

            solver.re_init(solver.t, solver.y, solver.yd, sw0=solver.sw)
            print('switched state to {}'.format(solver.sw))

        mod = Implicit_Problem(prep_rhs,
                               Y0,
                               Yd0,
                               self.t,
                               sw0=self.sw0)
        mod.algvar = self.eq_implicit
        mod.state_events = state_events
        mod.handle_event = handle_event
        self.sim = IDA(mod)
        self.sim.rtol = 1.e-12  # Sets the relative tolerance
        self.sim.atol = 1.e-10  # Sets the absolute tolerance

        if self.test:
            print(subs_ini)
        return n_val

    def run(self, t_max, **kwargs):

        self.t_max = t_max

        t, Y, Yd = self.sim.simulate(t_max)

        self.t = t[-1]
        self.Y0 = Y[-1]
        self.Yd0 = Yd[-1]
        [self.Kc_0, self.Kd_0, self.G_0, self.C_0] = Y[-1][:4]

        df = pd.DataFrame(Y, index=t, columns=self.var_names)
        self.m_trajectory = pd.concat([self.m_trajectory, df]).groupby(level=0).mean()

        return 1

    def get_unified_trajectory(self):
        """calculates and returns a unified output trajectory in terms of per capita variables.
        
        The question is: do I divide by total labor, or by labor employed in the respective sector.
        For capital, it certainly makes sense, to divide by labor employed in the sector. Same for resource
        use and knowledge. For labor employed in the respective sector, it certainly makes sense to put it in
        terms of total labor. For remaining resource stock, I am not sure.
        
        Talking to Jobst, I figured, it does not really matter, as long as the quantities can be computed from
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
                           * (self.independent_vars['Kc'] * self.dependent_vars['rc']
                              + self.independent_vars['Kd'] * self.dependent_vars['rd']) / L,
                           self.dependent_vars['rc'],
                           self.dependent_vars['rd'],
                           self.dependent_vars['w']]
        t_values = self.m_trajectory.index.values
        data = np.zeros((len(t_values), len(columns)))
        if self.lambdify:
            var_expressions_lambdified = [lambdify(tuple(self.var_symbols), expr) for expr in var_expressions]
            for i, t in enumerate(t_values):
                if self.test:
                    self._progress_report(i, len(t_values), 'calculating dependant variables')
                Yi = self.m_trajectory.loc[t]
                data[i, :] = [expr(*Yi) for expr in var_expressions_lambdified]

        else:
            for i, t in enumerate(t_values):
                if self.test:
                    self._progress_report(i, len(t_values), 'calculating dependant variables')
                Yi = self.m_trajectory.loc[t]
                sbs = {var_symbol: Yi[var_name] for var_symbol, var_name in zip(self.var_symbols, self.var_names)}
                data[i, :] = [var.subs(sbs) for var in var_expressions]

        return pd.DataFrame(index=t_values, columns=columns, data=data)

    def get_aggregate_trajectory(self):
        """return a mock aggregate trajectory with correct shape but containing zeros"""

        columns = ['x', 'y', 'z', 'K_c^c', 'K_d^c', 'K_c^d', 'K_d^d', 'C', 'G']
        index = self.m_trajectory.index

        return pd.DataFrame(0, index=index, columns=columns)

    def get_mean_trajectory(self):
        """return a mock mean trajectory with correct shape but containing zeros"""

        columns = ['x', 'y', 'z', 'mu_c^c', 'mu_d^c', 'mu_c^d', 'mu_d^d', 'c', 'g']
        index = self.m_trajectory.index

        return pd.DataFrame(0, index=index, columns=columns)


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
                        'b_c': 1., 'i_phi': 0.8, 'e': 100,
                        'G_0': 50, 'b_r0': 0.3 ** 2 * 100,
                        'possible_cue_orders': possible_cue_orders,
                        'C': 100, 'xi': 1. / 8., 'd_c': 0.06,
                        'campaign': False, 'learning': True,
                        'crs': True, 'test': True,
                        'R_depletion': False}

    # investment_decisions
    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full((n), i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:.copy()
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

    # m._setup_model()

    m.run(t_max=50)

    m.R_depletion = True
    m.set_parameters()
    m.run(t_max=100)

    # Plot the results

    trj = m.get_unified_trajectory()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    trj[['n_c']].plot(ax=ax1)

    ax2 = fig.add_subplot(222)
    trj[['k_c', 'k_d']].plot(ax=ax2)
    ax2b = ax2.twinx()
    trj[['c']].plot(ax=ax2b, color='g')

    ax3 = fig.add_subplot(223)
    trj[['r_c', 'r_d']].plot(ax=ax3)

    ax4 = fig.add_subplot(224)
    trj[['g']].plot(ax=ax4)

    fig.tight_layout()
    fig.savefig('representative_agent_test.png')
